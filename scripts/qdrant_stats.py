#!/usr/bin/env python3
"""
Qdrant operational health dashboard.

Shows collection stats, disk usage, index status, and compares
against FAISS baseline sizes when available.

Usage:
  python scripts/qdrant_stats.py
  python scripts/qdrant_stats.py --qdrant-url http://remote:6333
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def get_collection_details(url: str, name: str) -> dict:
    import httpx
    try:
        resp = httpx.get(f"{url}/collections/{name}", timeout=5)
        resp.raise_for_status()
        return resp.json().get("result", {})
    except Exception as e:
        return {"_error": str(e)}


def get_cluster_info(url: str) -> dict:
    import httpx
    try:
        resp = httpx.get(f"{url}/cluster", timeout=5)
        return resp.json().get("result", {})
    except Exception:
        return {}


def get_telemetry(url: str) -> dict:
    import httpx
    try:
        resp = httpx.get(f"{url}/telemetry", timeout=5)
        return resp.json().get("result", {})
    except Exception:
        return {}


def print_collection(name: str, details: dict):
    if "_error" in details:
        print(f"\n  {name}: ERROR - {details['_error']}")
        return

    pts = details.get("points_count", 0)
    segs = details.get("segments_count", 0)
    status = details.get("status", "?")
    opt = details.get("optimizer_status", "?")
    vecs = details.get("vectors_count", 0)

    print(f"\n  Collection: {name}")
    print(f"  {'Points':20s} {pts:>12,}")
    print(f"  {'Vectors':20s} {vecs:>12,}")
    print(f"  {'Segments':20s} {segs:>12}")
    print(f"  {'Status':20s} {status!s:>12}")
    print(f"  {'Optimizer':20s} {str(opt):>12}")

    # Vector config
    cfg = details.get("config", {})
    params = cfg.get("params", {})
    vectors_cfg = params.get("vectors", {})
    if isinstance(vectors_cfg, dict):
        for vname, vcfg in vectors_cfg.items():
            if isinstance(vcfg, dict):
                dim = vcfg.get("size", "?")
                dist = vcfg.get("distance", "?")
                print(f"  {'Vector ' + vname:20s} dim={dim}, distance={dist}")

    # Payload indexes
    payload_schema = details.get("payload_schema", {})
    if payload_schema:
        idx_names = list(payload_schema.keys())
        print(f"  {'Payload indexes':20s} {', '.join(idx_names)}")


def faiss_baseline_sizes(embed_dir: Path) -> dict:
    """Count chunks in FAISS metadata files to compare against Qdrant point counts."""
    sizes = {}
    for subdir in ("primary_index", "manual_index"):
        meta = embed_dir / subdir / "metadata.jsonl"
        if meta.exists():
            count = sum(1 for _ in meta.open())
            faiss_path = embed_dir / subdir / "faiss.index"
            faiss_size = faiss_path.stat().st_size if faiss_path.exists() else 0
            sizes[subdir] = {"chunks": count, "faiss_bytes": faiss_size}
    return sizes


def docker_qdrant_stats() -> dict:
    """Get Qdrant Docker container memory/CPU from docker stats."""
    try:
        out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format",
             "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"],
            text=True, timeout=5,
        )
        for line in out.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) >= 4 and "qdrant" in parts[0].lower():
                return {
                    "container": parts[0],
                    "cpu": parts[1],
                    "memory": parts[2],
                    "mem_pct": parts[3],
                }
    except Exception:
        pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Qdrant operational health check")
    parser.add_argument("--qdrant-url",
                        default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    primary = os.getenv("QDRANT_COLLECTION_PRIMARY", "supermicro_primary")
    manual = os.getenv("QDRANT_COLLECTION_MANUAL", "supermicro_manual")

    print(f"{'=' * 60}")
    print(f"  Qdrant Health Check — {args.qdrant_url}")
    print(f"{'=' * 60}")

    # Collection details
    collections = {}
    for name in (primary, manual):
        details = get_collection_details(args.qdrant_url, name)
        collections[name] = details
        print_collection(name, details)

    # Cluster info
    cluster = get_cluster_info(args.qdrant_url)
    if cluster:
        print(f"\n  Cluster status: {cluster.get('status', '?')}")
        peer_id = cluster.get("peer_id")
        if peer_id:
            print(f"  Peer ID: {peer_id}")

    # Docker stats
    print(f"\n{'=' * 60}")
    print(f"  Docker Container")
    print(f"{'=' * 60}")
    dstats = docker_qdrant_stats()
    if dstats:
        print(f"  Container: {dstats['container']}")
        print(f"  CPU:       {dstats['cpu']}")
        print(f"  Memory:    {dstats['memory']} ({dstats['mem_pct']})")
    else:
        print("  (Qdrant Docker container not found or docker not available)")

    # Qdrant storage disk usage
    print(f"\n{'=' * 60}")
    print(f"  Disk Usage")
    print(f"{'=' * 60}")
    qdrant_data = Path.home() / "qdrant_data"
    if qdrant_data.exists():
        try:
            out = subprocess.check_output(
                ["du", "-sh", str(qdrant_data)], text=True, timeout=5,
            )
            print(f"  Qdrant data:  {out.strip().split()[0]:>10s}  ({qdrant_data})")
        except Exception:
            print(f"  Qdrant data:  (could not measure, path: {qdrant_data})")
    else:
        print(f"  Qdrant data:  (no local storage at {qdrant_data})")

    # FAISS baseline comparison
    embed_dir = REPO_ROOT / "embeddings"
    faiss = faiss_baseline_sizes(embed_dir)
    if faiss:
        print(f"\n{'=' * 60}")
        print(f"  FAISS Baseline Comparison")
        print(f"{'=' * 60}")
        coll_map = {"primary_index": primary, "manual_index": manual}
        for subdir, info in faiss.items():
            coll_name = coll_map.get(subdir, subdir)
            qdrant_pts = collections.get(coll_name, {}).get("points_count", "?")
            match_str = ""
            if isinstance(qdrant_pts, int):
                if qdrant_pts == info["chunks"]:
                    match_str = " (MATCH)"
                else:
                    diff = qdrant_pts - info["chunks"]
                    match_str = f" (diff: {diff:+d})"
            print(f"  {subdir:20s}  FAISS: {info['chunks']:>8,} chunks, "
                  f"{_fmt_bytes(info['faiss_bytes']):>10s}  |  "
                  f"Qdrant: {qdrant_pts!s:>8s} points{match_str}")

    print()

    if args.json:
        output = {
            "qdrant_url": args.qdrant_url,
            "collections": {},
            "docker": dstats,
            "faiss_baseline": faiss,
        }
        for name, details in collections.items():
            if "_error" not in details:
                output["collections"][name] = {
                    "points_count": details.get("points_count"),
                    "segments_count": details.get("segments_count"),
                    "status": str(details.get("status")),
                    "optimizer_status": str(details.get("optimizer_status")),
                }
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
