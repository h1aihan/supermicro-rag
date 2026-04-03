#!/usr/bin/env python3
"""
Benchmark Qdrant-backed RAG vs the FAISS baseline (super_results.txt).

Measures:
  - Per-query latency breakdown: planning, retrieval, reranking, LLM, total
  - Source overlap with the FAISS baseline run
  - Aggregate latency stats (p50, p95, mean)
  - Qdrant collection health snapshot

Usage:
  python scripts/benchmark_qdrant.py                      # run all queries
  python scripts/benchmark_qdrant.py --category list      # one category
  python scripts/benchmark_qdrant.py --retrieval-only      # skip LLM (measure retrieval latency only)
  python scripts/benchmark_qdrant.py --baseline super_results.txt  # compare against FAISS baseline
  python scripts/benchmark_qdrant.py -o benchmark_out.jsonl        # write JSONL results
"""

import argparse
import ast
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)


# ── Baseline parser ──────────────────────────────────────────────────────

def parse_baseline(path: str) -> Dict[str, Dict]:
    """Parse super_results.txt into {query_id: {query, sources, answer_len, num_sources}}."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"={60,}", text)

    baseline: Dict[str, Dict] = {}
    i = 0
    while i < len(blocks):
        block = blocks[i].strip()
        header = re.search(r"\[(\d+)/\d+\]\s+\[(\w+)\]\s+(\S+)", block)
        if header:
            _num, category, qid = header.group(1), header.group(2), header.group(3)
            query_m = re.search(r"Query:\s*(.+)", block)
            query = query_m.group(1).strip() if query_m else ""

            body = blocks[i + 1] if i + 1 < len(blocks) else ""
            src_m = re.search(r"Sources:\s*(\[.+?\])", body, re.DOTALL)
            sources = []
            if src_m:
                try:
                    sources = ast.literal_eval(src_m.group(1))
                except Exception:
                    pass

            summary_m = re.search(r">>>\s+\[" + re.escape(qid) + r"\]\s+len=(\d+)\s+src=(\d+)", body)
            answer_len = int(summary_m.group(1)) if summary_m else 0

            baseline[qid] = {
                "query": query,
                "category": category,
                "sources": sources,
                "answer_len": answer_len,
                "num_sources": len(sources),
            }
        i += 1

    return baseline


# ── Qdrant collection snapshot ───────────────────────────────────────────

def qdrant_snapshot(qdrant_url: str, collections: List[str]) -> Dict:
    """Fetch basic stats from each Qdrant collection."""
    import httpx
    info = {}
    for coll in collections:
        try:
            resp = httpx.get(f"{qdrant_url}/collections/{coll}", timeout=5)
            data = resp.json().get("result", {})
            info[coll] = {
                "points_count": data.get("points_count", 0),
                "segments_count": data.get("segments_count", 0),
                "status": data.get("status", "unknown"),
                "optimizer_status": str(data.get("optimizer_status", "unknown")),
                "vectors_count": data.get("vectors_count", 0),
            }
        except Exception as e:
            info[coll] = {"error": str(e)}
    return info


# ── Source comparison ────────────────────────────────────────────────────

def _norm_source(s: str) -> str:
    """Normalize source filename for comparison (strip hashes, lowercase)."""
    s = re.sub(r'__[0-9a-f]{8,}(?=\.)', '', s)
    return s.lower().strip()


def compare_sources(
    qdrant_sources: List[str],
    baseline_sources: List[str],
    top_n: int = 5,
) -> Dict:
    """Compare source lists between Qdrant and FAISS baseline."""
    q_norm = [_norm_source(s) for s in qdrant_sources[:top_n]]
    b_norm = [_norm_source(s) for s in baseline_sources[:top_n]]
    q_set = set(q_norm)
    b_set = set(b_norm)
    overlap = q_set & b_set
    return {
        "qdrant_top": qdrant_sources[:top_n],
        "baseline_top": baseline_sources[:top_n],
        "overlap_count": len(overlap),
        "overlap_pct": len(overlap) / max(len(b_set), 1) * 100,
        "only_qdrant": list(q_set - b_set),
        "only_baseline": list(b_set - q_set),
    }


# ── Timed retrieval ─────────────────────────────────────────────────────

def timed_answer(bot, query: str, conversation_context: str = "") -> Dict:
    """Run bot.answer() with per-stage timing via monkey-patched wrappers."""
    timings: Dict[str, float] = {}

    from src.query_planner import plan_query as _orig_plan
    from src.query import rerank_chunks as _orig_rerank

    orig_retrieve = bot.query_processor.retrieve
    orig_retrieve_faq = bot.query_processor.retrieve_faq

    def timed_retrieve(*a, **kw):
        t0 = time.perf_counter()
        r = orig_retrieve(*a, **kw)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        return r

    def timed_retrieve_faq(*a, **kw):
        t0 = time.perf_counter()
        r = orig_retrieve_faq(*a, **kw)
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        return r

    bot.query_processor.retrieve = timed_retrieve
    bot.query_processor.retrieve_faq = timed_retrieve_faq

    t_total = time.perf_counter()
    result = bot.answer(query, conversation_context=conversation_context)
    timings["total_ms"] = (time.perf_counter() - t_total) * 1000

    bot.query_processor.retrieve = orig_retrieve
    bot.query_processor.retrieve_faq = orig_retrieve_faq

    if "retrieval_ms" in timings:
        timings["llm_ms"] = timings["total_ms"] - timings["retrieval_ms"]

    result["timings"] = timings
    return result


def timed_retrieve_only(bot, query: str, conversation_context: str = "") -> Dict:
    """Run only the retrieval stage (no LLM) with timing."""
    from src.query_planner import plan_query

    t_plan = time.perf_counter()
    effective_conversation = conversation_context or None
    plan = plan_query(query, conversation_context=effective_conversation)
    plan_ms = (time.perf_counter() - t_plan) * 1000

    search_queries = plan.search_queries if plan.search_queries else [query]
    rag_query = search_queries[0]

    t_ret = time.perf_counter()
    if plan.intent == "faq":
        chunks = bot.query_processor.retrieve_faq(query, 5)
    else:
        chunks = bot.query_processor.retrieve(
            rag_query,
            top_k=max(bot.top_k, 10),
            scope=plan.search_scope,
        )
    retrieval_ms = (time.perf_counter() - t_ret) * 1000

    sources = list(dict.fromkeys(c.get("source_file", "") for c in chunks))

    return {
        "answer": "(retrieval-only mode)",
        "sources": sources,
        "chunks": chunks,
        "plan": plan,
        "timings": {
            "plan_ms": plan_ms,
            "retrieval_ms": retrieval_ms,
            "total_ms": plan_ms + retrieval_ms,
        },
    }


# ── Latency stats ───────────────────────────────────────────────────────

def latency_summary(all_timings: List[Dict]) -> Dict:
    """Compute p50/p95/mean/max for each timing key."""
    keys = set()
    for t in all_timings:
        keys.update(t.keys())

    summary = {}
    for k in sorted(keys):
        vals = [t[k] for t in all_timings if k in t]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        summary[k] = {
            "mean": round(statistics.mean(vals), 1),
            "p50": round(vals_sorted[n // 2], 1),
            "p95": round(vals_sorted[int(n * 0.95)], 1),
            "max": round(max(vals), 1),
            "min": round(min(vals), 1),
            "count": n,
        }
    return summary


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Qdrant RAG vs FAISS baseline")
    parser.add_argument("--baseline", default=str(REPO_ROOT / "test_results.txt"),
                        help="Path to FAISS baseline results file (test_results.txt or super_results.txt)")
    parser.add_argument("--category", help="Run only this category (list, detail, etc.)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip LLM calls, measure retrieval latency only")
    parser.add_argument("--limit", type=int, help="Max number of queries to run")
    parser.add_argument("-o", "--output", help="Write JSONL results to this file")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    args = parser.parse_args()

    # Load test queries
    from tests.test_product_queries import PRODUCT_TEST_QUERIES, FOLLOWUP_TEST_QUERIES
    all_queries = PRODUCT_TEST_QUERIES + FOLLOWUP_TEST_QUERIES

    if args.category:
        all_queries = [q for q in all_queries if q["category"] == args.category]
    if args.limit:
        all_queries = all_queries[:args.limit]

    print(f"{'=' * 70}")
    print(f"  Qdrant RAG Benchmark — {len(all_queries)} queries")
    print(f"  Mode: {'retrieval-only' if args.retrieval_only else 'full (retrieval + LLM)'}")
    print(f"{'=' * 70}\n")

    # Qdrant health snapshot
    primary_coll = os.getenv("QDRANT_COLLECTION_PRIMARY", "supermicro_primary")
    manual_coll = os.getenv("QDRANT_COLLECTION_MANUAL", "supermicro_manual")
    snapshot = qdrant_snapshot(args.qdrant_url, [primary_coll, manual_coll])
    print("Qdrant Collections:")
    for coll, info in snapshot.items():
        if "error" in info:
            print(f"  {coll}: ERROR - {info['error']}")
        else:
            print(f"  {coll}: {info['points_count']:,} points, "
                  f"{info['segments_count']} segments, "
                  f"status={info['status']}, "
                  f"optimizer={info['optimizer_status']}")
    print()

    # Parse baseline
    baseline = {}
    if Path(args.baseline).exists():
        baseline = parse_baseline(args.baseline)
        print(f"FAISS baseline loaded: {len(baseline)} queries from {args.baseline}\n")
    else:
        print(f"No baseline file at {args.baseline}, skipping source comparison\n")

    # Init chatbot
    from src.chatbot import SupermicroChatbot
    from src.embed import get_qdrant_client

    client = get_qdrant_client(args.qdrant_url, os.getenv("QDRANT_API_KEY"))
    bot = SupermicroChatbot(
        qdrant_client=client,
        primary_collection=primary_coll,
        manual_collection=manual_coll,
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.5")),
        top_p=float(os.getenv("LLM_TOP_P", "1.0")),
        top_k=int(os.getenv("TOP_K", "10")),
    )

    # Run queries
    results = []
    all_timings = []
    source_overlaps = []
    output_fh = open(args.output, "w") if args.output else None

    try:
        for i, item in enumerate(all_queries, 1):
            qid = item["id"]
            query = item["query"]
            category = item["category"]
            conversation = item.get("conversation", "")

            print(f"[{i}/{len(all_queries)}] [{category}] {qid}")
            print(f"  Query: {query}")

            if args.retrieval_only:
                result = timed_retrieve_only(bot, query, conversation)
            else:
                result = timed_answer(bot, query, conversation)

            timings = result.get("timings", {})
            sources = result.get("sources", [])
            answer_len = len(result.get("answer", ""))
            all_timings.append(timings)

            timing_str = ", ".join(f"{k}={v:.0f}" for k, v in timings.items())
            print(f"  Sources ({len(sources)}): {sources[:5]}")
            print(f"  Timings: {timing_str}")

            # Source comparison
            src_cmp = None
            if qid in baseline:
                src_cmp = compare_sources(sources, baseline[qid]["sources"])
                source_overlaps.append(src_cmp["overlap_pct"])
                overlap_str = f"{src_cmp['overlap_count']}/{min(5, len(baseline[qid]['sources']))}"
                print(f"  vs FAISS: {overlap_str} sources overlap ({src_cmp['overlap_pct']:.0f}%)")

            print()

            row = {
                "id": qid,
                "category": category,
                "query": query,
                "sources": sources,
                "answer_len": answer_len,
                "num_sources": len(sources),
                "timings": timings,
                "source_comparison": src_cmp,
            }
            results.append(row)

            if output_fh:
                output_fh.write(json.dumps(row, default=str) + "\n")
                output_fh.flush()

    except KeyboardInterrupt:
        print("\n\nInterrupted. Printing partial results...\n")
    finally:
        if output_fh:
            output_fh.close()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK SUMMARY — {len(results)}/{len(all_queries)} queries completed")
    print(f"{'=' * 70}\n")

    if all_timings:
        print("Latency (ms):")
        summary = latency_summary(all_timings)
        for key, stats in summary.items():
            print(f"  {key:20s}  mean={stats['mean']:7.1f}  "
                  f"p50={stats['p50']:7.1f}  p95={stats['p95']:7.1f}  "
                  f"max={stats['max']:7.1f}  (n={stats['count']})")
        print()

    if source_overlaps:
        avg_overlap = statistics.mean(source_overlaps)
        print(f"Source Overlap vs FAISS baseline (top-5):")
        print(f"  Average: {avg_overlap:.1f}%")
        print(f"  Queries with 100% overlap: "
              f"{sum(1 for o in source_overlaps if o >= 100)}/{len(source_overlaps)}")
        print(f"  Queries with 0% overlap: "
              f"{sum(1 for o in source_overlaps if o == 0)}/{len(source_overlaps)}")

        changed = [(r["id"], r["source_comparison"])
                    for r in results
                    if r.get("source_comparison") and r["source_comparison"]["overlap_pct"] < 100]
        if changed:
            print(f"\n  Queries with changed sources ({len(changed)}):")
            for qid, cmp in changed[:15]:
                print(f"    {qid}: {cmp['overlap_count']} overlap, "
                      f"+{len(cmp['only_qdrant'])} new, "
                      f"-{len(cmp['only_baseline'])} dropped")
        print()

    # Qdrant collection summary
    print("Qdrant Collection Health:")
    for coll, info in snapshot.items():
        if "error" not in info:
            print(f"  {coll}: {info['points_count']:,} points, "
                  f"status={info['status']}")
    print()

    if args.output:
        print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
