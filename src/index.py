#!/usr/bin/env python3
"""
Hybrid search index backed by Qdrant vector database.

Each HybridIndex connects to a single Qdrant collection containing:
  - "dense" named vector  (sentence-transformer embeddings, cosine)
  - "sparse" named vector (BM25-weighted token hashes)
  - Payload fields: chunk_id, source_file, chunk_index, text, total_chunks

RoutedIndex wraps a primary + optional manual collection and routes
queries by scope ("primary", "manual", "both").
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from src.embed import (
        build_query_sparse_vector,
        token_to_index,
        tokenize_for_bm25,
    )
except ImportError:
    from embed import (
        build_query_sparse_vector,
        token_to_index,
        tokenize_for_bm25,
    )


# ── Qdrant payload proxy ─────────────────────────────────────────────────

class _QdrantPayloadProxy:
    """List-like facade over a Qdrant collection's payloads.

    Supports ``len()``, integer indexing (by point ID), and iteration
    (scroll in ID order) so that existing code using ``self.metadata``
    keeps working without loading everything into memory.
    """

    def __init__(self, client, collection_name: str):
        self._client = client
        self._collection = collection_name
        self._count: Optional[int] = None

    def __len__(self) -> int:
        if self._count is None:
            info = self._client.get_collection(self._collection)
            self._count = info.points_count or 0
        return self._count

    def __getitem__(self, idx: int) -> Dict:
        if not isinstance(idx, int):
            raise TypeError(f"index must be int, got {type(idx)}")
        pts = self._client.retrieve(
            collection_name=self._collection,
            ids=[idx],
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            raise IndexError(f"Point {idx} not found in '{self._collection}'")
        return pts[0].payload

    def __iter__(self):
        offset = None
        while True:
            pts, offset = self._client.scroll(
                collection_name=self._collection,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in pts:
                yield p.payload
            if offset is None:
                break


# ── Helpers ───────────────────────────────────────────────────────────────

def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return v / norms


_FILENAME_STOPWORDS = frozenset({
    'the', 'is', 'at', 'which', 'on', 'for', 'and', 'or', 'to', 'in',
    'of', 'with', 'what', 'how', 'can', 'do', 'does', 'are', 'was',
    'be', 'it', 'its', 'an', 'as', 'by', 'from', 'that', 'this',
    'my', 'me', 'we', 'you', 'your', 'their', 'our', 'into', 'about',
    'please', 'compare', 'suggest', 'recommend', 'show', 'tell',
    'give', 'list', 'between', 'vs', 'versus', 'than', 'should',
    'would', 'could', 'will', 'need', 'want', 'like', 'have', 'has',
    'pdf', 'datasheet', 'spec', 'specs', 'specification', 'specifications',
    'supermicro', 'server', 'servers', 'system', 'systems', 'series',
    'rackmount', 'product', 'products', 'page', 'web', 'txt',
})


# ── HybridIndex ──────────────────────────────────────────────────────────

class HybridIndex:
    """Hybrid search index combining dense (semantic) and sparse (keyword)
    vectors stored in a single Qdrant collection.

    Uses Reciprocal Rank Fusion (RRF) with adaptive weighting to combine
    semantic, keyword, and filename-matching channels.
    """

    def __init__(
        self,
        collection_name: str,
        qdrant_client,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        shared_model: Optional[SentenceTransformer] = None,
    ):
        self.client = qdrant_client
        self.collection = collection_name
        self.model_name = model_name

        info = self.client.get_collection(collection_name)
        self._point_count = info.points_count or 0
        print(f"Connected to Qdrant collection '{collection_name}': "
              f"{self._point_count:,} points")

        if shared_model is not None:
            self.model = shared_model
            print("Reusing shared embedding model")
        else:
            print(f"Loading embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)

        self.metadata = _QdrantPayloadProxy(self.client, self.collection)

        self._build_faq_question_bank()

    # ── Source-filter helper ──────────────────────────────────────────

    def _make_source_filter(self, source_filter: str):
        """Convert a source_filter substring into a Qdrant Filter.

        Uses the full-text index on ``source_file_text`` to approximate
        the legacy ``source_filter in source_file`` behaviour.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchText

        search_text = (
            source_filter
            .replace(":", " ")
            .replace("-", " ")
            .replace("_", " ")
            .strip()
        )
        if not search_text:
            return None
        return Filter(
            must=[FieldCondition(
                key="source_file_text",
                match=MatchText(text=search_text),
            )]
        )

    def _check_filter_has_results(self, qf) -> bool:
        if qf is None:
            return True
        count = self.client.count(
            collection_name=self.collection,
            count_filter=qf,
            exact=False,
        )
        return count.count > 0

    # ── FAQ question bank ─────────────────────────────────────────────

    def _build_faq_question_bank(self):
        """Build a lightweight in-memory vector bank of FAQ question titles."""
        from qdrant_client.models import FieldCondition, Filter, MatchText

        faq_filter = Filter(must=[
            FieldCondition(
                key="source_file_text",
                match=MatchText(text="FAQ"),
            )
        ])

        faq_points = []
        offset = None
        while True:
            pts, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=faq_filter,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            faq_points.extend(pts)
            if offset is None:
                break

        faq_questions: Dict[str, List[int]] = {}
        for point in faq_points:
            meta = point.payload
            source = meta.get("source_file", "")
            if "FAQ:" not in source:
                continue

            text = meta.get("text", "")
            match = re.search(r'Q:\s*(.+?)(?:\n|$)', text)
            if match:
                question = match.group(1).strip()
            else:
                q = source.replace("web_page_", "").replace(".txt", "")
                q = re.sub(r'^FAQ:_?', '', q).replace("_", " ").strip()
                if not q:
                    continue
                question = q

            if question not in faq_questions:
                faq_questions[question] = []
            faq_questions[question].append(point.id)

        if not faq_questions:
            self._faq_questions: List[str] = []
            self._faq_embeddings: Optional[np.ndarray] = None
            self._faq_chunk_map: Dict[int, List[int]] = {}
            print("[FAQ Bank] No FAQ entries found")
            return

        self._faq_questions = list(faq_questions.keys())
        self._faq_chunk_map = {
            i: faq_questions[q] for i, q in enumerate(self._faq_questions)
        }

        embeddings = self.model.encode(self._faq_questions)
        self._faq_embeddings = _normalize_rows(embeddings.astype("float32"))

        total_chunks = sum(len(v) for v in self._faq_chunk_map.values())
        print(f"[FAQ Bank] Built question bank: {len(self._faq_questions)} "
              f"questions, {total_chunks} chunks")

    def search_faq_questions(
        self, query: str, top_k: int = 5,
    ) -> List[Tuple[int, str, float, List[int]]]:
        """Match a user query against FAQ question titles via cosine similarity."""
        if self._faq_embeddings is None or not self._faq_questions:
            return []

        query_emb = _normalize_rows(
            self.model.encode([query]).astype("float32")
        )
        scores = np.dot(self._faq_embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for qi in top_indices:
            results.append((
                int(qi),
                self._faq_questions[qi],
                float(scores[qi]),
                self._faq_chunk_map[int(qi)],
            ))
        return results

    # ── Chunk lookup ──────────────────────────────────────────────────

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve a single chunk payload by its ``chunk_id`` field."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        pts, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[
                FieldCondition(
                    key="chunk_id",
                    match=MatchValue(value=chunk_id),
                )
            ]),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return pts[0].payload if pts else None

    def _batch_get_payloads(self, point_ids: List[int]) -> Dict[int, Dict]:
        """Fetch payloads for a batch of point IDs."""
        if not point_ids:
            return {}
        pts = self.client.retrieve(
            collection_name=self.collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        return {p.id: p.payload for p in pts}

    # ── Individual search channels ────────────────────────────────────

    def search_semantic(
        self, query: str, top_k: int = 20, qdrant_filter=None,
    ) -> List[Tuple[int, float]]:
        """Dense vector (semantic) search via Qdrant."""
        query_emb = self.model.encode([query]).astype("float32")

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_emb[0].tolist(),
            using="dense",
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=False,
        )
        return [(p.id, p.score) for p in results.points]

    def search_keyword(
        self, query: str, top_k: int = 20, qdrant_filter=None,
    ) -> List[Tuple[int, float]]:
        """Sparse vector (BM25-like keyword) search via Qdrant."""
        from qdrant_client.models import SparseVector

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        s_idx, s_val = build_query_sparse_vector(tokens)
        if not s_idx:
            return []

        results = self.client.query_points(
            collection_name=self.collection,
            query=SparseVector(indices=s_idx, values=s_val),
            using="sparse",
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=False,
        )
        return [(p.id, p.score) for p in results.points]

    def search_by_filename(
        self, query: str, top_k: int = 20, qdrant_filter=None,
    ) -> List[Tuple[int, float]]:
        """Retrieve chunks whose source-file tokens match query terms."""
        from qdrant_client.models import FieldCondition, Filter, MatchText

        raw_tokens = re.findall(
            r'\b\w{2,}\b',
            query.lower().replace('-', ' ').replace('_', ' '),
        )
        terms = list(dict.fromkeys(
            t for t in raw_tokens if len(t) >= 2 and t not in _FILENAME_STOPWORDS
        ))
        if not terms:
            return []

        should_clauses = [
            FieldCondition(key="source_file_text", match=MatchText(text=t))
            for t in terms
        ]
        fn_filter = Filter(should=should_clauses)

        if qdrant_filter is not None:
            fn_filter = Filter(
                must=[qdrant_filter, fn_filter],
            )

        pts, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=fn_filter,
            limit=min(top_k * 10, 500),
            with_payload=["source_file"],
            with_vectors=False,
        )

        chunk_scores: Dict[int, int] = {}
        for p in pts:
            source = p.payload.get("source_file", "").lower()
            src_tokens = set(re.findall(
                r'\b\w{2,}\b',
                source.replace('_', ' ').replace('-', ' '),
            ))
            match_count = sum(1 for t in terms if t in src_tokens)
            if match_count > 0:
                chunk_scores[p.id] = match_count

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k]]

    # ── Query type detection ──────────────────────────────────────────

    def _is_product_code_query(self, query: str) -> bool:
        words = query.strip().split()
        if len(words) <= 3:
            for word in words:
                has_letter = any(c.isalpha() for c in word)
                has_digit = any(c.isdigit() for c in word)
                if has_letter and has_digit:
                    return True
                if word.upper().startswith(
                    ('SYS-', 'AS-', 'SSG-', 'SBI-', 'AOC-', 'X1', 'H1')
                ):
                    return True
        return False

    def _is_keyword_heavy_query(self, query: str) -> bool:
        return len(query.strip().split()) <= 2

    # ── Source-type boosting ──────────────────────────────────────────

    _RE_MANUAL = re.compile(r'^MNL-', re.IGNORECASE)
    _RE_GUIDE = re.compile(
        r'[Uu]ser[_\s]?[Gg]uide|^QRG-|^BMC_IPMI|^IPMI', re.IGNORECASE,
    )
    _RE_CHASSIS = re.compile(r'^(?:SC\d|CSE-)', re.IGNORECASE)

    def _source_type_boost(self, source_file: str) -> float:
        if source_file.startswith('web_page_FAQ'):
            return 1.4
        if source_file.startswith('accessory_'):
            return 1.3
        if source_file.startswith('web_page_') or source_file.startswith('web_product_'):
            return 1.2
        if self._RE_MANUAL.search(source_file):
            return 0.6
        if self._RE_GUIDE.search(source_file):
            return 0.7
        if self._RE_CHASSIS.search(source_file):
            return 1.0
        if source_file.endswith('.pdf'):
            return 1.15
        return 1.0

    # ── BM25 query expansion ─────────────────────────────────────────

    def _expand_query_for_bm25(self, query: str) -> str:
        words = query.lower().split()
        expansions: set = set()

        for word in words:
            if word.endswith('ies') and len(word) > 4:
                expansions.add(word[:-3] + 'y')
            elif word.endswith('ses') and len(word) > 4:
                expansions.add(word[:-2])
            elif word.endswith('es') and len(word) > 3:
                expansions.add(word[:-2])
                expansions.add(word[:-1])
            elif word.endswith('s') and not word.endswith('ss') and len(word) > 3:
                expansions.add(word[:-1])
            if word.endswith('en') and len(word) > 4:
                expansions.add(word[:-2])
            if word.endswith('ing') and len(word) > 5:
                expansions.add(word[:-3])
                expansions.add(word[:-3] + 'e')
            if word.endswith('ed') and len(word) > 4:
                expansions.add(word[:-2])
                expansions.add(word[:-1])

        new_terms = expansions - set(words)
        new_terms = {t for t in new_terms if len(t) > 2}
        if new_terms:
            return query + ' ' + ' '.join(new_terms)
        return query

    # ── Hybrid search (RRF fusion) ───────────────────────────────────

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = None,
        keyword_weight: float = None,
        rrf_k: int = 60,
        max_per_source: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        """Hybrid search combining semantic + keyword + filename channels
        via Reciprocal Rank Fusion with adaptive weighting.

        Returns list of ``(payload_dict, rrf_score)`` tuples.
        """
        # Build Qdrant filter from source_filter string
        qdrant_filter = None
        using_filter = False
        if source_filter:
            qdrant_filter = self._make_source_filter(source_filter)
            if qdrant_filter and not self._check_filter_has_results(qdrant_filter):
                print(f"[DEBUG] source_filter='{source_filter}' matched 0 "
                      "chunks, falling back to unfiltered")
                qdrant_filter = None
            else:
                using_filter = True

        # Auto-detect weights
        if semantic_weight is None or keyword_weight is None:
            if self._is_product_code_query(query):
                semantic_weight = 0.2
                keyword_weight = 0.8
            elif self._is_keyword_heavy_query(query):
                semantic_weight = 0.25
                keyword_weight = 0.75
            else:
                semantic_weight = 0.5
                keyword_weight = 0.5

        fetch_k = max(top_k * 20, 200) if using_filter else max(top_k * 5, 30)

        # Channel 1: semantic (dense)
        semantic_results = self.search_semantic(query, fetch_k, qdrant_filter)

        # Channel 2: keyword (sparse) with query expansion
        bm25_query = self._expand_query_for_bm25(query)
        keyword_results = self.search_keyword(bm25_query, fetch_k, qdrant_filter)

        # Channel 3: filename matching
        filename_results = self.search_by_filename(query, fetch_k, qdrant_filter)

        # ── RRF score calculation ─────────────────────────────────────
        rrf_scores: Dict[int, float] = {}

        for rank, (pid, _) in enumerate(semantic_results):
            rrf_scores[pid] = rrf_scores.get(pid, 0) + (
                semantic_weight * (1.0 / (rrf_k + rank + 1))
            )

        for rank, (pid, _) in enumerate(keyword_results):
            rrf_scores[pid] = rrf_scores.get(pid, 0) + (
                keyword_weight * (1.0 / (rrf_k + rank + 1))
            )

        if filename_results:
            best_match_score = filename_results[0][1]
            filename_weight = 1.0 if best_match_score >= 2 else 0.3
            for rank, (pid, _score) in enumerate(filename_results):
                rrf_scores[pid] = rrf_scores.get(pid, 0) + (
                    filename_weight * (1.0 / (rrf_k + rank + 1))
                )

        if not rrf_scores:
            return []

        # ── Fetch payloads for candidates ─────────────────────────────
        sorted_pids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        candidate_count = max(top_k * 5, 50)
        payloads = self._batch_get_payloads(sorted_pids[:candidate_count])

        # ── Source-type boost + boilerplate penalty ───────────────────
        for pid in list(rrf_scores):
            meta = payloads.get(pid)
            if meta is None:
                continue
            source_file = meta.get("source_file", "")
            text = meta.get("text", "")

            text_flat = re.sub(r'\s+', ' ', text.lower())
            if ('global leader in high performance' in text_flat
                    and 'broad range of skus' in text_flat):
                rrf_scores[pid] *= 0.3
                continue

            rrf_scores[pid] *= self._source_type_boost(source_file)

        # Re-sort after boosts
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # ── Source-diversity path (broad queries) ─────────────────────
        if max_per_source is not None:
            source_counts: Dict[str, int] = defaultdict(int)
            diversified: List[Tuple[Dict, float]] = []
            for pid, score in sorted_indices:
                meta = payloads.get(pid)
                if meta is None:
                    continue
                src = meta.get("source_file", "")
                norm_src = re.sub(r'__[0-9a-f]{8,}\.', '.', src)
                if source_counts[norm_src] < max_per_source:
                    source_counts[norm_src] += 1
                    diversified.append((meta, score))
                    if len(diversified) >= top_k:
                        break
            return diversified

        # ── Concentrated path (context expansion) ─────────────────────
        initial_results: List[Tuple[int, Dict, float]] = []
        for pid, score in sorted_indices[:top_k]:
            meta = payloads.get(pid)
            if meta is not None:
                initial_results.append((pid, meta, score))

        if initial_results:
            top_source = initial_results[0][1].get("source_file", "")
            top_total = initial_results[0][1].get("total_chunks", 1)
            top_score = initial_results[0][2]

            should_expand = (
                top_total > 5
                and (len(initial_results) < 2
                     or top_score > initial_results[1][2] * 1.03)
            )

            if should_expand:
                existing_pids = {r[0] for r in initial_results}
                sibling_chunks = self._fetch_sibling_chunks(
                    top_source, existing_pids, top_score,
                )
                max_siblings = top_k // 2
                siblings_to_add = sibling_chunks[:max_siblings]

                final = [initial_results[0]]
                final.extend(siblings_to_add)
                sib_pids = {s[0] for s in siblings_to_add}
                remaining = [r for r in initial_results[1:]
                             if r[0] not in sib_pids]
                final.extend(remaining)
                initial_results = final[:top_k]

        return [(meta, score) for _, meta, score in initial_results]

    def _fetch_sibling_chunks(
        self,
        source_file: str,
        exclude_pids: set,
        base_score: float,
    ) -> List[Tuple[int, Dict, float]]:
        """Fetch sibling chunks from the same source document."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        pts, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_file),
                )
            ]),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        siblings = []
        for p in pts:
            if p.id not in exclude_pids:
                siblings.append((p.id, p.payload, base_score * 0.95))

        siblings.sort(key=lambda x: x[1].get("chunk_index", 0))
        return siblings

    # ── Default search ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        return self.search_hybrid(query, top_k)


# Backward-compat alias
VectorIndex = HybridIndex


# ── RoutedIndex ───────────────────────────────────────────────────────────

class RoutedIndex:
    """Multi-collection wrapper that routes queries to primary and/or
    manual Qdrant collections based on a ``scope`` parameter.
    """

    def __init__(
        self,
        qdrant_client,
        primary_collection: str,
        manual_collection: Optional[str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.primary = HybridIndex(
            primary_collection, qdrant_client, model_name,
        )
        self._model_name = model_name
        self.manual: Optional[HybridIndex] = None

        print(f"[RoutedIndex] Primary collection: '{primary_collection}' "
              f"({self.primary._point_count:,} points)")

        if manual_collection:
            try:
                self.manual = HybridIndex(
                    manual_collection, qdrant_client, model_name,
                    shared_model=self.primary.model,
                )
                print(f"[RoutedIndex] Manual collection: '{manual_collection}' "
                      f"({self.manual._point_count:,} points)")
            except Exception as e:
                print(f"[RoutedIndex] Manual collection '{manual_collection}' "
                      f"unavailable: {e}")

    # ── Delegated search ──────────────────────────────────────────────

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        scope: str = "primary",
        manual_top_k: int = 5,
        **kwargs,
    ) -> List[Tuple[Dict, float]]:
        if scope == "manual":
            if self.manual:
                return self.manual.search_hybrid(query, top_k, **kwargs)
            return self.primary.search_hybrid(query, top_k, **kwargs)

        results = self.primary.search_hybrid(query, top_k, **kwargs)

        if scope == "both" and self.manual:
            manual_results = self.manual.search_hybrid(
                query, manual_top_k, **kwargs,
            )
            seen = {r[0].get("chunk_id") for r in results}
            for chunk, score in manual_results:
                cid = chunk.get("chunk_id")
                if cid not in seen:
                    seen.add(cid)
                    results.append((chunk, score))

        return results

    def search_faq_questions(self, query: str, top_k: int = 5):
        return self.primary.search_faq_questions(query, top_k)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Look up a chunk by ID across primary and manual collections."""
        result = self.primary.get_chunk_by_id(chunk_id)
        if result is None and self.manual:
            result = self.manual.get_chunk_by_id(chunk_id)
        return result

    @property
    def metadata(self):
        return self.primary.metadata

    @property
    def model(self):
        return self.primary.model


# ── CLI test harness ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Test hybrid search against Qdrant",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION_PRIMARY", "supermicro_primary"),
    )
    parser.add_argument("--query", help="Test query to search")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--compare", action="store_true",
                        help="Compare semantic vs keyword vs hybrid")

    args = parser.parse_args()

    try:
        from src.embed import get_qdrant_client
    except ImportError:
        from embed import get_qdrant_client

    client = get_qdrant_client(args.qdrant_url)
    index = HybridIndex(args.collection, client)

    if args.query:
        print(f"\n{'=' * 60}")
        print(f"Query: '{args.query}'")
        print(f"{'=' * 60}")

        if args.compare:
            print(f"\nProduct code query: {index._is_product_code_query(args.query)}")
            print(f"Keyword-heavy query: {index._is_keyword_heavy_query(args.query)}")

            print("\n--- SEMANTIC (dense) ---")
            for i, (pid, score) in enumerate(
                index.search_semantic(args.query, args.top_k), 1,
            ):
                meta = index.metadata[pid]
                print(f"{i}. [{score:.4f}] {meta['source_file']}")

            print("\n--- KEYWORD (sparse) ---")
            for i, (pid, score) in enumerate(
                index.search_keyword(args.query, args.top_k), 1,
            ):
                meta = index.metadata[pid]
                print(f"{i}. [{score:.4f}] {meta['source_file']}")

            print("\n--- HYBRID (RRF) ---")
            for i, (chunk, score) in enumerate(
                index.search_hybrid(args.query, args.top_k), 1,
            ):
                print(f"{i}. [{score:.6f}] {chunk['source_file']}")
        else:
            results = index.search(args.query, args.top_k)
            print(f"\nTop {len(results)} hybrid results:")
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\n{i}. Score: {score:.6f}")
                print(f"   Source: {chunk['source_file']}")
                print(f"   Text: {chunk['text'][:200]}...")
    else:
        print("\nIndex loaded. Use --query to test.")
