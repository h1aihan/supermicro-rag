#!/usr/bin/env python3
"""
Build an entity-relationship graph from the FAISS index metadata.

Uses a hybrid approach:
  1. Regex extraction for high-confidence structured patterns
     (product codes, chassis models, motherboard models, part numbers)
  2. LLM extraction (Haiku) for fuzzy/natural-language relationships
     (compatibility, product families, GPU support, etc.)

Output: entity_graph.json — a lightweight adjacency-list graph loaded
at query time for multi-hop retrieval (e.g., system → chassis → parts).
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

_repo_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_repo_root / ".env", override=False)

# ── Entity-type regex patterns ──────────────────────────────────────────────

ENTITY_PATTERNS: Dict[str, re.Pattern] = {
    "system": re.compile(r"\b((?:SYS|AS|SSG|SBI)-[\w-]{4,})\b", re.IGNORECASE),
    "chassis": re.compile(r"\b(CSE-[\w-]{4,})\b", re.IGNORECASE),
    "motherboard": re.compile(
        r"\b((?:X1[0-9]|H1[0-9])[A-Z0-9]{1,6}(?:-[A-Z0-9]+)*)\b"
    ),
    "accessory_mcp": re.compile(r"\b(MCP-[\w-]{6,})\b", re.IGNORECASE),
    "psu": re.compile(r"\b(PWS-[\w-]{4,})\b", re.IGNORECASE),
    "aoc": re.compile(r"\b(AOC-[\w-]{4,})\b", re.IGNORECASE),
    "fan": re.compile(r"\b(FAN-[\w-]{4,})\b", re.IGNORECASE),
    "cable": re.compile(r"\b(CBL-[\w-]{4,})\b", re.IGNORECASE),
    "heatsink": re.compile(r"\b(SNK-[\w-]{4,})\b", re.IGNORECASE),
    "riser": re.compile(r"\b(RSC-[\w-]{4,})\b", re.IGNORECASE),
    "backplane": re.compile(r"\b(BPN-[\w-]{4,})\b", re.IGNORECASE),
}

ACCESSORY_TYPES = {"accessory_mcp", "psu", "aoc", "fan", "cable", "heatsink", "riser", "backplane"}

# ── Boilerplate detection ────────────────────────────────────────────────────
# Supermicro product web pages embed a sidebar/navigation menu listing every
# product family.  These phrases appear verbatim in 100+ chunks and cause the
# LLM to create false entity relationships (e.g. attributing GPU/HPC support
# to a mainstream 1U server).

_NAV_BOILERPLATE_PHRASES = [
    "broadest portfolio of single processor servers",
    "extremely large in-memory computing",
    "modular building block design, future proof",
    "highest density gpu platforms for deployments",
    "purpose-built liquid-cooled, hpc-at-scale",
    "highest performing 2u twin architecture",
    "multi-node architecture optimized for single processor",
    "density maximized storage systems optimized",
    "petascale all-flash array with nvidia grace",
    "server-grade performance and ai inferencing at the edge",
]

_FOOTER_BOILERPLATE_PHRASES = [
    "global leader in high performance",
    "broad range of skus",
]

_HEADER_ONLY_PATTERN = re.compile(
    r"^(Product SKUs|SuperServer|Motherboard|Processor|System Memory|"
    r"On-Board Devices|Input / Output|System BIOS|Management|Security|"
    r"PC Health Monitoring|Chassis|Dimensions and Weight|Front Panel|"
    r"Expansion Slots|Drive Bays|System Cooling|Power Supply|"
    r"Operating Environment|Publish Status)(\s*\n\s*)*",
    re.MULTILINE,
)


def _is_boilerplate_chunk(text: str) -> bool:
    """Return True if a chunk is mostly navigation boilerplate or empty headers."""
    text_lower = text.lower()

    nav_hits = sum(1 for phrase in _NAV_BOILERPLATE_PHRASES if phrase in text_lower)
    if nav_hits >= 3:
        return True

    footer_hits = sum(1 for phrase in _FOOTER_BOILERPLATE_PHRASES if phrase in text_lower)
    if footer_hits >= 2:
        return True

    stripped = _HEADER_ONLY_PATTERN.sub("", text).strip()
    if len(stripped) < 30:
        return True

    return False

# ── Regex-based relationship extraction ─────────────────────────────────────

_RE_CHASSIS_LINE = re.compile(
    r"(?:Chassis|chassis)\s+(?:Model\s+)?:?\s*(CSE-[\w-]+)", re.IGNORECASE
)
_RE_MOBO_LINE = re.compile(
    r"(?:Motherboard|motherboard)\s+(?:Model\s+)?:?\s*(?:Super\s+)?((?:X1[0-9]|H1[0-9])[\w-]+)",
    re.IGNORECASE,
)
_RE_PARTS_TABLE = re.compile(
    r"((?:MCP|PWS|AOC|FAN|CBL|SNK|RSC|BPN)-[\w-]+)\s*\|\s*(\d+\w*|-)\s*\|\s*(.+)",
    re.IGNORECASE,
)

# Matches "Compatible Chassis: SC512F, 515, 813, 813M, 814, ..."
_RE_COMPATIBLE_CHASSIS = re.compile(
    r"Compatible\s+Chassis\s*:\s*(.+)", re.IGNORECASE
)

# Matches "Compatible With: CSE-512, 515, 813-816"
_RE_COMPATIBLE_WITH = re.compile(
    r"Compatible\s+With\s*:\s*(.+)", re.IGNORECASE
)

_RE_FAMILY_TOKEN = re.compile(
    r"^(?:SC|CSE-?)?(\d{3}[A-Z]?|[A-Z]{1,2}\d{2,3}[A-Z]?)$", re.IGNORECASE
)


def _normalize(entity: str) -> str:
    """Upper-case product codes for consistent graph keys."""
    return entity.upper().strip()


def _chassis_to_family(chassis_model: str) -> Optional[str]:
    """Extract the chassis family prefix from a full chassis model number.

    Old style (digits first):
      CSE-813MF2TS-R0RCNBP  →  SC813M
      CSE-815TQ-R706WB      →  SC815
      CSE-512F-600LB        →  SC512F
      CSE-113MTQ-600CB      →  SC113M

    New style (letters + digits):
      CSE-LA26TS-R1K23AWP1  →  SCLA26T
      CSE-LA15TQC-R504W     →  SCLA15T
      CSE-LB26AC12-R1K23AW  →  SCLB26A
    """
    raw = chassis_model.upper().strip()
    raw = re.sub(r"^CSE-?", "", raw)
    # Old style: starts with 3 digits + optional letter
    m = re.match(r"(\d{3}[A-Z]?)", raw)
    if m:
        return f"SC{m.group(1)}"
    # New style: 1-2 letters + 2-3 digits + optional letter (e.g., LA26T, LB26A)
    m = re.match(r"([A-Z]{1,2}\d{2,3}[A-Z]?)", raw)
    if m:
        return f"SC{m.group(1)}"
    return None


# ── Graph data structures ───────────────────────────────────────────────────

class EntityGraph:
    def __init__(self):
        self.entities: Dict[str, dict] = {}
        self.edges: List[dict] = []

    def add_entity(self, name: str, etype: str, chunk_id: str):
        name = _normalize(name)
        if name not in self.entities:
            self.entities[name] = {"type": etype, "chunk_ids": set()}
        self.entities[name]["chunk_ids"].add(chunk_id)

    def add_edge(self, source: str, relation: str, target: str, chunk_id: str = ""):
        source, target = _normalize(source), _normalize(target)
        self.edges.append({
            "source": source,
            "relation": relation,
            "target": target,
            "chunk_id": chunk_id,
        })

    def to_adjacency(self) -> dict:
        """Convert to adjacency-list format for JSON serialization."""
        adj: Dict[str, dict] = {}
        for name, info in self.entities.items():
            adj[name] = {
                "type": info["type"],
                "chunk_ids": sorted(info["chunk_ids"]),
                "edges": [],
            }
        for edge in self.edges:
            src, tgt = edge["source"], edge["target"]
            if src not in adj:
                adj[src] = {"type": "unknown", "chunk_ids": [], "edges": []}
            adj[src]["edges"].append({
                "relation": edge["relation"],
                "target": tgt,
            })
            # Reverse edge
            if tgt not in adj:
                adj[tgt] = {"type": "unknown", "chunk_ids": [], "edges": []}
            reverse_rel = _reverse_relation(edge["relation"])
            adj[tgt]["edges"].append({
                "relation": reverse_rel,
                "target": src,
            })
        # Deduplicate edges per entity
        for name in adj:
            seen = set()
            unique = []
            for e in adj[name]["edges"]:
                key = (e["relation"], e["target"])
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            adj[name]["edges"] = unique
        return adj


def _reverse_relation(rel: str) -> str:
    REVERSE_MAP = {
        "uses_chassis": "used_by_system",
        "uses_motherboard": "used_by_system",
        "has_part": "part_of",
        "has_standard_part": "standard_part_of",
        "supports_gpu": "supported_by",
        "supports_cpu": "supported_by",
        "compatible_with": "compatible_with",
        "belongs_to_family": "has_member",
        "belongs_to_generation": "has_member",
        "alternative_to": "alternative_to",
        "requires": "required_by",
        "included_with": "includes",
        "validated_for": "validated_by",
    }
    return REVERSE_MAP.get(rel, f"reverse_{rel}")


# ── Phase 1: Regex extraction ──────────────────────────────────────────────

def extract_entities_regex(chunk_id: str, text: str, graph: EntityGraph):
    """Extract entities and structured relationships from a single chunk."""
    for etype, pattern in ENTITY_PATTERNS.items():
        for match in pattern.finditer(text):
            canonical_type = etype if etype not in ACCESSORY_TYPES else "accessory"
            graph.add_entity(match.group(1), canonical_type, chunk_id)

    source_file = chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else ""
    system_model = _infer_system_from_filename(source_file)

    # System → Chassis relationship
    for m in _RE_CHASSIS_LINE.finditer(text):
        chassis = m.group(1)
        if system_model:
            graph.add_edge(system_model, "uses_chassis", chassis, chunk_id)

    # System → Motherboard relationship
    for m in _RE_MOBO_LINE.finditer(text):
        mobo = m.group(1)
        if system_model:
            graph.add_edge(system_model, "uses_motherboard", mobo, chunk_id)

    # Parts-table rows (chassis/system web pages)
    for m in _RE_PARTS_TABLE.finditer(text):
        part_number = m.group(1)
        description = m.group(3).strip()
        parent = system_model or _infer_chassis_from_filename(source_file)
        if parent:
            graph.add_edge(parent, "has_part", part_number, chunk_id)
            graph.add_entity(part_number, "accessory", chunk_id)

    # Accessory → chassis family compatibility (from eStore data)
    # "Compatible Chassis: SC512F, 515, 813, 813M, 814, 815, ..."
    accessory_pn = _infer_accessory_from_text(text)
    if accessory_pn:
        for regex in (_RE_COMPATIBLE_CHASSIS, _RE_COMPATIBLE_WITH):
            for m in regex.finditer(text):
                families = _parse_family_tokens(m.group(1))
                for fam in families:
                    graph.add_entity(fam, "chassis_family", chunk_id)
                    graph.add_edge(accessory_pn, "compatible_with", fam, chunk_id)


def _infer_system_from_filename(filename: str) -> Optional[str]:
    """Try to extract a system model from the source filename."""
    m = re.match(r"((?:sys|as|ssg|sbi)-[\w-]+)\.pdf", filename, re.IGNORECASE)
    if m:
        return _normalize(m.group(1))
    m = re.search(r"web_page_((?:SYS|AS|SSG|SBI)-[\w-]+)", filename, re.IGNORECASE)
    if m:
        return _normalize(m.group(1))
    m = re.search(r"web_product_((?:SYS|AS|SSG|SBI)-[\w-]+)", filename, re.IGNORECASE)
    if m:
        return _normalize(m.group(1))
    return None


def _infer_chassis_from_filename(filename: str) -> Optional[str]:
    m = re.search(r"(CSE-[\w-]+)", filename, re.IGNORECASE)
    if m:
        return _normalize(m.group(1))
    return None


def _infer_accessory_from_text(text: str) -> Optional[str]:
    """Extract the primary accessory part number from chunk text."""
    m = re.search(
        r"Part\s+Number\s*:\s*((?:MCP|PWS|AOC|FAN|CBL|SNK|RSC|BPN)-[\w-]+)",
        text, re.IGNORECASE,
    )
    if m:
        return _normalize(m.group(1))
    return None


def _parse_family_tokens(raw: str) -> List[str]:
    """Parse a comma-separated list of chassis family tokens into normalized SC* keys.

    "SC512F, 515, 813, 813M, 814"  →  ["SC512F", "SC515", "SC813", "SC813M", "SC814"]
    "CSE-512, 515, 813-816"        →  ["SC512", "SC515", "SC813", "SC814", "SC815", "SC816"]
    """
    families = []
    tokens = [t.strip() for t in raw.split(",")]
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        # Handle ranges like "813-816"
        range_m = re.match(r"(?:SC|CSE-?)?(\d{3})([A-Z]?)\s*-\s*(?:SC|CSE-?)?(\d{3})([A-Z]?)", tok, re.IGNORECASE)
        if range_m:
            start, end = int(range_m.group(1)), int(range_m.group(3))
            for num in range(start, end + 1):
                families.append(f"SC{num}")
            continue
        # Single token
        fam_m = _RE_FAMILY_TOKEN.match(tok)
        if fam_m:
            families.append(f"SC{fam_m.group(1).upper()}")
    return families


# ── Chassis Family Registry ──────────────────────────────────────────────────

def _build_family_registry(graph: EntityGraph) -> Dict[str, str]:
    """Build a mapping from every known CSE-* chassis SKU to its SC* family.

    Also covers system-to-family by following system → uses_chassis → chassis → family.
    Returns dict like {"CSE-813MF2TS-R0RCNBP": "SC813M", "CSE-813MFTQC-505CB": "SC813M"}.
    """
    registry: Dict[str, str] = {}
    for name, info in graph.entities.items():
        if info["type"] == "chassis":
            fam = _chassis_to_family(name)
            if fam:
                registry[name] = fam
    return registry


# ── Phase 0: Structured Import from estore_accessories.jsonl ────────────────

_RE_SYSTEM_CODE = re.compile(
    r"\b((?:SYS|AS|SSG|SBI)-[\w-]{4,})\b", re.IGNORECASE
)
_RE_CHASSIS_CODE = re.compile(
    r"\b(CSE-[\w-]{4,})\b", re.IGNORECASE
)


def _import_structured_accessories(
    graph: EntityGraph,
    accessories_path: str,
    family_registry: Dict[str, str],
) -> int:
    """Import edges from estore_accessories.jsonl structured fields.

    Creates high-confidence edges from:
      - compatible_chassis → accessory compatible_with chassis_family
      - validated_systems  → accessory validated_for system
      - validated_chassis  → accessory compatible_with chassis (+ family)
    """
    if not os.path.exists(accessories_path):
        print(f"[EntityGraph] Phase 0: {accessories_path} not found, skipping")
        return 0

    edges_added = 0
    with open(accessories_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            pn = rec.get("part_number", "").strip()
            if not pn:
                continue
            pn = _normalize(pn)
            graph.add_entity(pn, "accessory", "")

            # compatible_chassis: family tokens like "SC813, SC514" or raw text
            cc = rec.get("compatible_chassis", "").strip()
            if cc:
                families = _parse_family_tokens(cc)
                for fam in families:
                    graph.add_entity(fam, "chassis_family", "")
                    graph.add_edge(pn, "compatible_with", fam)
                    edges_added += 1

                # Also parse any CSE-* codes in the raw text
                for m in _RE_CHASSIS_CODE.finditer(cc):
                    chassis = _normalize(m.group(1))
                    graph.add_entity(chassis, "chassis", "")
                    graph.add_edge(pn, "compatible_with", chassis)
                    edges_added += 1
                    fam = family_registry.get(chassis) or _chassis_to_family(chassis)
                    if fam:
                        graph.add_entity(fam, "chassis_family", "")
                        graph.add_edge(pn, "compatible_with", fam)
                        edges_added += 1

            # validated_systems: comma-separated system codes
            vs = rec.get("validated_systems", "").strip()
            if vs:
                for m in _RE_SYSTEM_CODE.finditer(vs):
                    sys_code = _normalize(m.group(1))
                    graph.add_entity(sys_code, "system", "")
                    graph.add_edge(pn, "validated_for", sys_code)
                    edges_added += 1

            # validated_chassis: comma-separated chassis SKUs
            vc = rec.get("validated_chassis", "").strip()
            if vc:
                for m in _RE_CHASSIS_CODE.finditer(vc):
                    chassis = _normalize(m.group(1))
                    graph.add_entity(chassis, "chassis", "")
                    graph.add_edge(pn, "compatible_with", chassis)
                    edges_added += 1
                    fam = family_registry.get(chassis) or _chassis_to_family(chassis)
                    if fam:
                        graph.add_entity(fam, "chassis_family", "")
                        graph.add_edge(pn, "compatible_with", fam)
                        edges_added += 1

    return edges_added


# ── Phase 1c: BOM Enhancement with Family-Level Edges ───────────────────────

def _enhance_bom_with_family_edges(
    graph: EntityGraph,
    family_registry: Dict[str, str],
) -> int:
    """Add has_standard_part edges from chassis families for every has_part edge.

    When a BOM page lists: CSE-813MFTQC-505CB has_part MCP-290-00102-0N
    This adds:          SC813M has_standard_part MCP-290-00102-0N

    This ensures any system using ANY SC813M chassis variant can find the part.
    """
    edges_added = 0
    family_parts: Dict[str, Set[str]] = defaultdict(set)

    for edge in list(graph.edges):
        if edge["relation"] != "has_part":
            continue
        parent = edge["source"]
        part = edge["target"]
        fam = family_registry.get(parent)
        if fam and part not in family_parts[fam]:
            family_parts[fam].add(part)
            graph.add_entity(fam, "chassis_family", "")
            graph.add_edge(fam, "has_standard_part", part, edge.get("chunk_id", ""))
            edges_added += 1

    return edges_added


# ── Phase 3 helpers: Document classifier for LLM ────────────────────────────

_MULTI_PRODUCT_PATTERNS = [
    re.compile(r"CPU_Information", re.IGNORECASE),
    re.compile(r"Mainstream_Servers", re.IGNORECASE),
    re.compile(r"Product_SKUs", re.IGNORECASE),
    re.compile(r"Compare_Products", re.IGNORECASE),
    re.compile(r"_Servers_?_?\|", re.IGNORECASE),
    re.compile(r"All_Systems", re.IGNORECASE),
    re.compile(r"BIOS.*Firmware.*Download", re.IGNORECASE),
]


def _should_skip_llm(source_file: str, text: str) -> bool:
    """Return True if this chunk should NOT be sent to the LLM for extraction.

    Skips:
    - Multi-product listing/comparison pages (noisy, many false edges)
    - Chunks dominated by BOM/parts table rows (already handled by regex)
    """
    for pat in _MULTI_PRODUCT_PATTERNS:
        if pat.search(source_file):
            return True

    bom_rows = len(_RE_PARTS_TABLE.findall(text))
    non_bom_lines = len([
        l for l in text.split("\n")
        if l.strip() and not _RE_PARTS_TABLE.match(l.strip())
    ])
    if bom_rows >= 3 and non_bom_lines < bom_rows:
        return True

    return False


# ── Phase 2: LLM-assisted extraction ───────────────────────────────────────

EXTRACTION_PROMPT = """Extract entity relationships from this Supermicro product documentation chunk.

Entity types: system (SYS-*, AS-*, SSG-*), chassis (CSE-*), motherboard (X1*/H1*),
accessory (MCP-*, AOC-*, PWS-*, FAN-*, CBL-*, SNK-*, RSC-*, BPN-*),
gpu, cpu_family, chassis_family (SC* prefix), product_family

Relationship types: uses_chassis, uses_motherboard, has_part, compatible_with,
supports_gpu, supports_cpu, belongs_to_family, belongs_to_generation,
requires, included_with, alternative_to

Rules:
- Only extract relationships EXPLICITLY stated in the text. Do not infer.
- Skip generic boilerplate (copyright notices, navigation text).
- For compatibility mentions like "designed for Twin series", use compatible_with.
- For GPU support like "supports up to 8 NVIDIA H100", use supports_gpu with just the GPU name.
- For CPU mentions like "AMD EPYC 9005 Series", use supports_cpu with "EPYC 9005".
- For chassis family: when text mentions a chassis model (CSE-*), extract a belongs_to_family
  relationship with the family prefix. Family prefix format: "SC" + the model's leading
  identifier (e.g., CSE-813MF2TS → SC813M, CSE-LA26TS → SCLA26T, CSE-512F → SC512F).
- For accessory compatibility: when text lists compatible chassis families or models,
  extract compatible_with edges using the SC* family prefix.
- Output an empty array [] if no relationships are found.

Output ONLY a JSON array of triples (no other text):
[{"source": "SYS-511R-M", "relation": "uses_chassis", "target": "CSE-813MF2TS"}]

TEXT:
"""


def _call_llm_for_source(
    client,
    model: str,
    source_file: str,
    source_chunks: List[dict],
    max_retries: int = 3,
) -> List[dict]:
    """Make a single LLM call for one source file with retry on rate limit."""
    combined = "\n---\n".join(c["text"][:600] for c in source_chunks[:8])
    if len(combined) < 50:
        return []

    prompt = EXTRACTION_PROMPT + combined[:3000]

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=500,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            triples = json.loads(raw)
            if not isinstance(triples, list):
                return []

            valid = []
            for t in triples:
                src = t.get("source", "").strip()
                rel = t.get("relation", "").strip()
                tgt = t.get("target", "").strip()
                if src and rel and tgt and len(src) > 1 and len(tgt) > 1:
                    valid.append({"source": src, "relation": rel, "target": tgt,
                                  "chunk_id": source_chunks[0]["chunk_id"]})
            return valid

        except json.JSONDecodeError:
            return []
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str or "overloaded" in err_str:
                wait = 2 ** attempt + 1
                time.sleep(wait)
                continue
            print(f"[EntityGraph] LLM error on {source_file}: {e}")
            return []

    return []


def extract_relationships_llm(
    chunks: List[dict],
    graph: EntityGraph,
    batch_size: int = 5,
    provider: str = "anthropic",
    max_workers: int = 8,
) -> int:
    """Send chunks to a fast LLM for relationship extraction (parallel).

    Uses ThreadPoolExecutor for concurrency. Each worker retries on 429/rate-limit
    with exponential backoff. Keep max_workers <= 10 to stay within typical
    Anthropic API rate limits.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_added = 0

    if provider == "anthropic":
        try:
            from anthropic import Anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("[EntityGraph] No ANTHROPIC_API_KEY, skipping LLM extraction")
                return 0
            client = Anthropic(api_key=api_key)
            model = os.getenv("PLANNER_MODEL", "claude-haiku-4-5")
        except ImportError:
            print("[EntityGraph] anthropic package not installed, skipping LLM extraction")
            return 0
    else:
        print(f"[EntityGraph] LLM extraction not implemented for provider={provider}")
        return 0

    by_source: Dict[str, List[dict]] = defaultdict(list)
    for c in chunks:
        by_source[c["source_file"]].append(c)

    sources = list(by_source.keys())
    print(f"[EntityGraph] LLM extraction: {len(sources)} source files, model={model}, workers={max_workers}")

    done_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_call_llm_for_source, client, model, sf, by_source[sf]): sf
            for sf in sources
        }
        for future in as_completed(futures):
            done_count += 1
            triples = future.result()
            for t in triples:
                graph.add_edge(t["source"], t["relation"], t["target"], t["chunk_id"])
                total_added += 1
            if done_count % 100 == 0:
                print(f"[EntityGraph]   ... {done_count}/{len(sources)} sources ({total_added} triples so far)")

    return total_added


# ── Main build pipeline ─────────────────────────────────────────────────────

def build_graph(
    metadata_path: str,
    output_path: str,
    use_llm: bool = True,
    llm_provider: str = "anthropic",
    accessories_path: str = None,
) -> dict:
    """Build the entity graph from metadata.jsonl and structured data sources.

    Pipeline:
      Phase 1:  Regex extraction from chunk text
      Phase 1b: Chassis → family linkage
      Phase 1c: Build family registry + BOM family-level edges
      Phase 0:  Structured import from estore_accessories.jsonl
      Phase 2:  LLM extraction (scoped, skips BOM/multi-product)

    Returns the adjacency-list dict (also saved to output_path).
    """
    if accessories_path is None:
        accessories_path = str(_repo_root / "data" / "accessories" / "estore_accessories.jsonl")

    print(f"[EntityGraph] Loading metadata from {metadata_path}")
    chunks = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"[EntityGraph] Loaded {len(chunks)} chunks")

    graph = EntityGraph()

    # Phase 1: regex extraction (skip boilerplate chunks)
    print("[EntityGraph] Phase 1: regex extraction ...")
    t0 = time.time()
    boilerplate_skipped = 0
    for c in chunks:
        text = c.get("text", "")
        if _is_boilerplate_chunk(text):
            boilerplate_skipped += 1
            continue
        extract_entities_regex(c["chunk_id"], text, graph)
    print(f"[EntityGraph]   {len(graph.entities)} entities, {len(graph.edges)} edges "
          f"({boilerplate_skipped} boilerplate chunks skipped) ({time.time()-t0:.1f}s)")

    # Phase 1b: chassis → family edges
    print("[EntityGraph] Phase 1b: chassis family linkage ...")
    family_count = 0
    for name, info in list(graph.entities.items()):
        if info["type"] == "chassis":
            fam = _chassis_to_family(name)
            if fam:
                graph.add_entity(fam, "chassis_family", "")
                graph.add_edge(name, "belongs_to_family", fam)
                family_count += 1
    print(f"[EntityGraph]   {family_count} chassis→family edges added")

    # Phase 1c: build family registry and enhance BOM edges
    print("[EntityGraph] Phase 1c: family registry + BOM enhancement ...")
    family_registry = _build_family_registry(graph)
    print(f"[EntityGraph]   Family registry: {len(family_registry)} chassis→family mappings")
    bom_edges = _enhance_bom_with_family_edges(graph, family_registry)
    print(f"[EntityGraph]   +{bom_edges} has_standard_part edges (family-level BOM)")

    # Phase 0: structured import from estore_accessories.jsonl
    print(f"[EntityGraph] Phase 0: structured import from {accessories_path} ...")
    t_acc = time.time()
    acc_edges = _import_structured_accessories(graph, accessories_path, family_registry)
    print(f"[EntityGraph]   +{acc_edges} edges from structured accessories ({time.time()-t_acc:.1f}s)")

    # Phase 2: LLM extraction (scoped — skip BOM chunks and multi-product docs)
    if use_llm:
        print("[EntityGraph] Phase 2: LLM-assisted extraction (scoped) ...")
        t1 = time.time()
        meaningful = [
            c for c in chunks
            if len(c.get("text", "")) > 80
            and not c.get("text", "").strip().startswith("--- Page")
            and not _is_boilerplate_chunk(c.get("text", ""))
            and not _should_skip_llm(c.get("source_file", ""), c.get("text", ""))
        ]
        skipped_by_classifier = sum(
            1 for c in chunks
            if len(c.get("text", "")) > 80
            and not c.get("text", "").strip().startswith("--- Page")
            and not _is_boilerplate_chunk(c.get("text", ""))
            and _should_skip_llm(c.get("source_file", ""), c.get("text", ""))
        )
        print(f"[EntityGraph]   {len(meaningful)} chunks for LLM ({skipped_by_classifier} skipped by classifier)")
        n_added = extract_relationships_llm(meaningful, graph, provider=llm_provider)
        print(f"[EntityGraph]   +{n_added} LLM triples ({time.time()-t1:.1f}s)")

    # Serialize
    adj = graph.to_adjacency()
    print(f"[EntityGraph] Final graph: {len(adj)} nodes, "
          f"{sum(len(n['edges']) for n in adj.values())} total edge entries")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(adj, f, indent=2)
    print(f"[EntityGraph] Saved to {output_path}")

    return adj


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build entity-relationship graph for Supermicro RAG")
    parser.add_argument(
        "--metadata", default="embeddings/faiss_index/metadata.jsonl",
        help="Path to metadata.jsonl (default: embeddings/faiss_index/metadata.jsonl)",
    )
    parser.add_argument(
        "--output", default="embeddings/faiss_index/entity_graph.json",
        help="Output path for the graph JSON",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM extraction (regex only — faster, cheaper, less coverage)",
    )
    parser.add_argument(
        "--provider", default=None,
        help="LLM provider for extraction (default: from .env LLM_PROVIDER or anthropic)",
    )
    parser.add_argument(
        "--accessories", default=None,
        help="Path to estore_accessories.jsonl (default: data/accessories/estore_accessories.jsonl)",
    )
    args = parser.parse_args()

    provider = args.provider or os.getenv("LLM_PROVIDER", "anthropic")
    build_graph(
        args.metadata, args.output,
        use_llm=not args.no_llm,
        llm_provider=provider,
        accessories_path=args.accessories,
    )


if __name__ == "__main__":
    main()
