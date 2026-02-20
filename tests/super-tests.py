"""
Extended test suite for the Supermicro RAG system — sourced from SuperGPT evaluation
spreadsheet (recommendation, SKU details, comparison, general questions).

Run to see results:
  python tests/super-tests.py                        # run all, print answers
  python tests/super-tests.py --summary              # + one-line quality hint per query
  python tests/super-tests.py --dry-run              # print queries only
  python tests/super-tests.py --category recommendation  # one category only
  python tests/super-tests.py --output FILE          # write output to file

Requires: .env with OPENAI_API_KEY (and optionally ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_dotenv = REPO_ROOT / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_dotenv, override=False)


# =============================================================================
# Recommendation questions
# =============================================================================

RECOMMENDATION_QUERIES = [
    {
        "id": "rec_file_server",
        "category": "recommendation",
        "query": "please suggest one system for small business as file server",
        "expect": "A storage-capable system recommendation suitable for SMB file serving",
    },
    {
        "id": "rec_azure_certified",
        "category": "recommendation",
        "query": "recommend a azure local certified system",
        "expect": "Azure Stack HCI or Azure-certified Supermicro system",
    },
    {
        "id": "rec_high_perf_low_cost",
        "category": "recommendation",
        "query": "suggest a high-performance server with the lowest cost",
        "expect": "Cost-effective high-performance system recommendation",
    },
    {
        "id": "rec_high_dense_compute",
        "category": "recommendation",
        "query": "please recommend High Dense compute system",
        "expect": "High-density multi-node system (MicroCloud, Twin, FatTwin, etc.)",
    },
    {
        "id": "rec_eight_h200_gpu",
        "category": "recommendation",
        "query": "recommend me a supermicro system with eight H200 GPU",
        "expect": "8-GPU H200 system (likely SYS-821GE or HGX-based)",
    },
    {
        "id": "rec_x13_up_cloud",
        "category": "recommendation",
        "query": "Recommend me an X13 UP Intel Xeon Scalable server that is suitable for cloud computing",
        "expect": "X13-generation single-processor CloudDC or mainstream server",
    },
    {
        "id": "rec_choose_111e_vs_112b",
        "category": "recommendation",
        "query": "I'm having a tough time choosing between SYS-111E-WR and SYS-112B-WR. What do you recommend?",
        "expect": "Comparison and recommendation between SYS-111E-WR and SYS-112B-WR",
    },
    {
        "id": "rec_x14_wio_up",
        "category": "recommendation",
        "query": "Recommend me an X14 WIO Intel UP Server",
        "expect": "X14-generation WIO single-processor server recommendation",
    },
    {
        "id": "rec_x14_hyper_up",
        "category": "recommendation",
        "query": "Recommend me an X14 UP Hyper system",
        "expect": "X14 single-processor Hyper series system",
    },
    {
        "id": "rec_x14_clouddc_up",
        "category": "recommendation",
        "query": "Recommend me an X14 CloudDC Intel UP Server",
        "expect": "X14 single-processor CloudDC server recommendation",
    },
    {
        "id": "rec_x14_up_intel",
        "category": "recommendation",
        "query": "Recommend me an X14 UP Intel System",
        "expect": "X14-generation Intel single-processor system",
    },
    {
        "id": "rec_nvidia_6000_pro",
        "category": "recommendation",
        "query": "Recommend me GPU systems that support NVIDIA 6000 PRO",
        "expect": "Systems supporting NVIDIA RTX 6000 PRO or similar GPU",
    },
    {
        "id": "rec_xeon_6",
        "category": "recommendation",
        "query": "Recommend intel xeon 6 system to me",
        "expect": "System with Intel Xeon 6 (Granite Rapids or Sierra Forest) processor support",
    },
]


# =============================================================================
# SKU detail questions
# =============================================================================

SKU_DETAIL_QUERIES = [
    {
        "id": "sku_ssg640_psu_type",
        "category": "sku_detail",
        "query": "Is the DC PSU or AC PSU standard for the SSG-640SP-E1CR60?",
        "expect": "PSU type info for SSG-640SP-E1CR60 storage server",
    },
    {
        "id": "sku_as2126ft_spec",
        "category": "sku_detail",
        "query": "Do you know the system spec the AS-2126FT-HE-LCC?",
        "expect": "Specs for AS-2126FT-HE-LCC FlexTwin system",
    },
    {
        "id": "sku_similar_to_4125gs",
        "category": "sku_detail",
        "query": "what are the most similar products to as-4125gs-tnrt2",
        "expect": "Products similar to AS-4125GS-TNRT2 (GPU server variants)",
    },
    {
        "id": "sku_compare_4125gs_variants",
        "category": "sku_detail",
        "query": "give me a table comparison of as-4125gs-tnrt and as-4125gs-tnrt1 and as-4125gs-tnrt2",
        "expect": "Table comparing three AS-4125GS variants with specs for each",
    },
    {
        "id": "sku_621c_fhfl_gpu",
        "category": "sku_detail",
        "query": "Does SYS-621C-TN12R support 4x of FHFL GPUs",
        "expect": "GPU support info (full-height full-length) for SYS-621C-TN12R",
    },
    {
        "id": "sku_611c_dc_psu",
        "category": "sku_detail",
        "query": "Does SYS-611C-TN4R have a DC PSU option?",
        "expect": "DC power supply availability for SYS-611C-TN4R",
    },
    {
        "id": "sku_120u_bios_bmc",
        "category": "sku_detail",
        "query": "What is the current BIOS and BMC version for SYS-120U-TNR?",
        "expect": "Firmware version info for SYS-120U-TNR",
    },
    {
        "id": "sku_222bt_rhel",
        "category": "sku_detail",
        "query": "Which version of OS RHEL can be supported for SYS-222BT-HER?",
        "expect": "RHEL (Red Hat Enterprise Linux) support info for SYS-222BT-HER",
    },
    {
        "id": "sku_621c_rear_nvme",
        "category": "sku_detail",
        "query": "Can SYS-621C-TN12R support NVMe in rear drive bays?",
        "expect": "Rear drive bay NVMe support info for SYS-621C-TN12R",
    },
    {
        "id": "sku_421ge_cpu",
        "category": "sku_detail",
        "query": "tell me supported CPU on SYS-421GE-NBRT-LCC",
        "expect": "Supported processor list for SYS-421GE-NBRT-LCC",
    },
    {
        "id": "sku_521c_up_or_dp",
        "category": "sku_detail",
        "query": "Is SYS-521C-NR a UP or DP server?",
        "expect": "Whether SYS-521C-NR is uniprocessor (UP) or dual-processor (DP)",
    },
    {
        "id": "sku_521c_x13",
        "category": "sku_detail",
        "query": "Is SYS-521C-NR an X13 server?",
        "expect": "Platform generation info for SYS-521C-NR",
    },
    {
        "id": "sku_521c_uni_or_dual",
        "category": "sku_detail",
        "query": "Is SYS-521C-NR a uniprocessor or dual processor server?",
        "expect": "Processor count for SYS-521C-NR",
    },
    {
        "id": "sku_521c_nvme_aiom",
        "category": "sku_detail",
        "query": "If I use 2x NVMe Configuration for SYS-521C-NR, will it be able to support a x16 AIOM?",
        "expect": "AIOM slot availability with 2x NVMe config for SYS-521C-NR",
    },
    {
        "id": "sku_521c_memory_speed",
        "category": "sku_detail",
        "query": "What is the maximum speed of memory for 2 Dimm per channel for SYS-521C-NR?",
        "expect": "DDR memory speed at 2 DIMMs per channel for SYS-521C-NR",
    },
    {
        "id": "sku_521c_toshiba_drives",
        "category": "sku_detail",
        "query": "What specific Toshiba drives does SYS-521C-NR support?",
        "expect": "Toshiba drive compatibility for SYS-521C-NR",
    },
    {
        "id": "sku_521c_pcie",
        "category": "sku_detail",
        "query": "Can you tell me about the PCIe options for SYS-521C-NR?",
        "expect": "PCIe slot configuration and options for SYS-521C-NR",
    },
    {
        "id": "sku_111c_double_gpu",
        "category": "sku_detail",
        "query": "Can SYS-111C-NR support double-width GPUs?",
        "expect": "Double-width GPU support info for SYS-111C-NR",
    },
    {
        "id": "sku_111c_option_b",
        "category": "sku_detail",
        "query": "What is the option B configuration of SYS-111C-NR?",
        "expect": "Option B drive/storage configuration for SYS-111C-NR",
    },
    {
        "id": "sku_111c_alternatives",
        "category": "sku_detail",
        "query": "What are some alternative options for SYS-111C-NR?",
        "expect": "Alternative or similar systems to SYS-111C-NR",
    },
    {
        "id": "sku_111c_addon_cards",
        "category": "sku_detail",
        "query": "Please list some of the Add-on card options for SYS-111C-NR",
        "expect": "Supported add-on cards (AOC) for SYS-111C-NR",
    },
    {
        "id": "sku_111c_raid",
        "category": "sku_detail",
        "query": "Does SYS-111C-NR support RAID?",
        "expect": "RAID support info for SYS-111C-NR",
    },
    {
        "id": "sku_111c_aoc_ag_i2m",
        "category": "sku_detail",
        "query": "DOes SYS-111C-NR support AOC-AG-i2M?",
        "expect": "Whether AOC-AG-i2M add-on card is compatible with SYS-111C-NR",
    },
]


# =============================================================================
# Comparison questions
# =============================================================================

COMPARISON_QUERIES = [
    {
        "id": "cmp_h14_h13_hyper_table",
        "category": "comparison",
        "query": "Please compare H14 Hyper 1U, 2U and H13 Hyper 1U,2U server major spec. into table",
        "expect": "Table comparing H14 vs H13 Hyper in 1U and 2U form factors",
    },
    {
        "id": "cmp_flextwin_vs_bigtwin",
        "category": "comparison",
        "query": "what are the differences between FlexTwin and BigTwin",
        "expect": "Architecture/feature differences between FlexTwin and BigTwin multi-node systems",
    },
    {
        "id": "cmp_h14_h13_hyper_2u",
        "category": "comparison",
        "query": "What are the advantages of H14 Hyper 2U compared to H13 Hyper 2U?",
        "expect": "H14 vs H13 generation improvements for 2U Hyper servers",
    },
    {
        "id": "cmp_h13_vs_h14_hyper",
        "category": "comparison",
        "query": "what is the difference between H13 Hyper and H14 Hyper?",
        "expect": "Generation comparison between H13 and H14 Hyper product lines",
    },
    {
        "id": "cmp_x14_x13_clouddc",
        "category": "comparison",
        "query": "tell me X14 CloudDC and X13 CloudDC major differences",
        "expect": "X14 vs X13 CloudDC generation differences",
    },
    {
        "id": "cmp_flextwin_bigtwin_2",
        "category": "comparison",
        "query": "what's the difference between FlexTwin and BigTwin",
        "expect": "FlexTwin vs BigTwin architecture comparison",
    },
    {
        "id": "cmp_421ge_vs_422ga",
        "category": "comparison",
        "query": "what is difference between SYS-421GE-NBRT-LCC and SYS-422GA-NBRT-LCC",
        "expect": "Comparison of these two GPU server variants",
    },
    {
        "id": "cmp_a21ge_vs_a22ga",
        "category": "comparison",
        "query": "what is major difference between SYS-A21GE-NBRT and SYS-A22GA-NBRT",
        "expect": "Comparison of these two AMD GPU server variants",
    },
    {
        "id": "cmp_511e_111e_521e",
        "category": "comparison",
        "query": "SYS-511E-WR vs SYS-111E-WR vs SYS-521E-WR",
        "expect": "Three-way comparison of WIO E-series servers",
    },
    {
        "id": "cmp_111r_vs_511r",
        "category": "comparison",
        "query": "What are some differences between SYS-111R-M and SYS-511R-W?",
        "expect": "Comparison of these two 1U server models",
    },
    {
        "id": "cmp_512b_vs_112b",
        "category": "comparison",
        "query": "SYS-512B-WR vs SYS-112B-WR",
        "expect": "Comparison between 5th-gen and 1st-gen WIO B-series models",
    },
    {
        "id": "cmp_choose_111e_511e",
        "category": "comparison",
        "query": "Help me choose between SYS-111E-WR and SYS-511E-WR",
        "expect": "Comparison and recommendation between these two WIO servers",
    },
    {
        "id": "cmp_x13_x14_grandtwin",
        "category": "comparison",
        "query": "what are differences of X13 and X14 GrandTwin?",
        "expect": "Generation comparison for GrandTwin multi-node platform",
    },
]


# =============================================================================
# General questions
# =============================================================================

GENERAL_QUERIES = [
    {
        "id": "gen_gb200_nvl72",
        "category": "general",
        "query": "what is supermicro GB200 NVL72?",
        "expect": "Info about Supermicro GB200 NVL72 GPU system",
    },
    {
        "id": "gen_toploading_products",
        "category": "general",
        "query": "what are the product name of Supermicro toploading?",
        "expect": "Top-loading chassis or system product names",
    },
    {
        "id": "gen_lot9_clouddc",
        "category": "general",
        "query": "Do you have Lot 9 certification for CloudDC",
        "expect": "Lot 9 energy efficiency certification info for CloudDC products",
    },
    {
        "id": "gen_openbmc_support",
        "category": "general",
        "query": "Which systems from Supermicro support OpenBMC?",
        "expect": "Systems with OpenBMC management support",
    },
    {
        "id": "gen_fattwin_intel_amd",
        "category": "general",
        "query": "list FatTwin system with Intel and AMD platform",
        "expect": "FatTwin models for both Intel and AMD platforms",
    },
    {
        "id": "gen_fattwin_amd",
        "category": "general",
        "query": "List FatTwin server with AMD",
        "expect": "AMD-based FatTwin server models",
    },
    {
        "id": "gen_grandtwin_advantage",
        "category": "general",
        "query": "what is advantage of GrandTwin",
        "expect": "Benefits/advantages of GrandTwin architecture",
    },
    {
        "id": "gen_grandtwin_amd",
        "category": "general",
        "query": "list of GrandTwin for AMD server",
        "expect": "AMD-based GrandTwin server models",
    },
    {
        "id": "gen_gold_series_gpu",
        "category": "general",
        "query": "which are Gold Series version in GPU?",
        "expect": "Gold Series pre-configured GPU server SKUs",
    },
    {
        "id": "gen_h100_air_cooled",
        "category": "general",
        "query": "tell me about Supermicro H100 air cooled GPU system",
        "expect": "Air-cooled systems supporting NVIDIA H100 GPUs",
    },
    {
        "id": "gen_b200_air_cooled_specs",
        "category": "general",
        "query": "tell me specs of supermicro B200 Air cooled system?",
        "expect": "Specs for air-cooled Supermicro B200 GPU system",
    },
    {
        "id": "gen_intel_hgx_b200",
        "category": "general",
        "query": "tell me about supermicro intel CPU based HGX B200 system",
        "expect": "Intel-CPU-based HGX B200 system info",
    },
    {
        "id": "gen_hgx_b200_form_factor",
        "category": "general",
        "query": "what is the form factor of supermicro HGX B200 system",
        "expect": "Form factor (rack units) for HGX B200 system",
    },
    {
        "id": "gen_hgx_b200_intel_cpu_type",
        "category": "general",
        "query": "Which Intel CPU type is supported on Supermicro HGX B200",
        "expect": "Intel CPU families supported on HGX B200 platform",
    },
    {
        "id": "gen_hgx_b200_intel_cpu",
        "category": "general",
        "query": "Which Intel CPU is supported on Supermicro HGX B200",
        "expect": "Specific Intel CPU models supported on HGX B200",
    },
    {
        "id": "gen_hgx_b200_psu",
        "category": "general",
        "query": "what is power supply specs of Supermicro HGX B200",
        "expect": "PSU wattage and specs for HGX B200 chassis",
    },
    {
        "id": "gen_b200_air_specs_2",
        "category": "general",
        "query": "tell me specs of NVIDIA B200 Air cooled system?",
        "expect": "Specs for NVIDIA B200 air-cooled GPU server",
    },
    {
        "id": "gen_x13_wio_up_intel",
        "category": "general",
        "query": "Can you list for me all X13 WIO UP Intel servers?",
        "expect": "List of X13-generation WIO single-processor Intel servers",
    },
    {
        "id": "gen_x13_wio_up_xeon_scalable",
        "category": "general",
        "query": "Can you list for me all X13 WIO UP Intel Xeon Scalable servers?",
        "expect": "X13 WIO UP servers with Intel Xeon Scalable processors",
    },
    {
        "id": "gen_x13_1u_wio_xeon_e",
        "category": "general",
        "query": "Please list all X13 1U WIO/Mainstream with Xeon-E UP servers",
        "expect": "X13 1U servers with Xeon-E processors",
    },
    {
        "id": "gen_x13_1u_wio_xeon_e_cpu",
        "category": "general",
        "query": "What processors do the X13 1U WIO/Mainstream with Xeon-E servers support?",
        "expect": "Supported Xeon-E processor models for X13 1U WIO/Mainstream",
    },
    {
        "id": "gen_x13_mainstream_aiom",
        "category": "general",
        "query": "Do any of the X13 UP Intel Mainstream servers support AIOMs?",
        "expect": "AIOM support info for X13 UP Intel Mainstream servers",
    },
    {
        "id": "gen_x13_up_mainstream_list",
        "category": "general",
        "query": "List for me the X13 UP Intel Mainstream servers",
        "expect": "List of X13 single-processor Intel Mainstream server models",
    },
    {
        "id": "gen_x14_up_wio",
        "category": "general",
        "query": "What are the X14 UP WIO Systems?",
        "expect": "X14-generation single-processor WIO server models",
    },
]


# =============================================================================
# All queries combined
# =============================================================================

ALL_QUERIES = RECOMMENDATION_QUERIES + SKU_DETAIL_QUERIES + COMPARISON_QUERIES + GENERAL_QUERIES

CATEGORIES = sorted(set(q["category"] for q in ALL_QUERIES))


# =============================================================================
# Runner infrastructure (shared with test_product_queries.py pattern)
# =============================================================================

def get_chatbot(model_override=None):
    """Lazy init chatbot. model_override replaces the main LLM model (e.g. 'claude-sonnet-4-5' for cheaper testing)."""
    from src.chatbot import SupermicroChatbot
    index_dir = os.getenv("INDEX_DIR", "embeddings/faiss_index/")
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = model_override or os.getenv("LLM_MODEL", "gpt-5.2")
    if model_override and provider == "anthropic":
        os.environ["ANTHROPIC_MODEL"] = model_override
    return SupermicroChatbot(
        index_dir=index_dir,
        llm_model=model,
        llm_provider=provider,
        top_k=int(os.getenv("TOP_K", "10")),
    )


def run_query(chatbot, item: dict) -> dict:
    """Run one test query and return answer + sources + plan debug info."""
    conversation = item.get("conversation", "")
    result = chatbot.answer(item["query"], conversation_context=conversation)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "answer_length": len(result.get("answer", "")),
        "num_sources": len(result.get("sources", [])),
        "plan": result.get("plan"),
        "search_queries": result.get("search_queries", []),
        "rag_top_k": result.get("rag_top_k"),
        "max_per_source": result.get("max_per_source"),
    }


def _format_plan(out: dict) -> str:
    """Format query plan debug info for display."""
    plan = out.get("plan")
    if not plan:
        return "[QueryPlanner] (no plan)"
    parts = [f"intent={plan.intent}"]
    if plan.product_codes:
        parts.append(f"codes={plan.product_codes}")
    sq = out.get("search_queries", [])
    if sq:
        parts.append(f"queries={sq}")
    if plan.form_factor:
        parts.append(f"ff={plan.form_factor}")
    if plan.tags:
        parts.append(f"tags={plan.tags}")
    if plan.keywords:
        parts.append(f"kw={plan.keywords}")
    parts.append(f"catalog={'Y' if plan.use_catalog else 'N'}")
    parts.append(f"rag={'Y' if plan.use_rag else 'N'}")
    top_k = out.get("rag_top_k", "?")
    mps = out.get("max_per_source")
    parts.append(f"top_k={top_k}")
    parts.append(f"max_per_src={mps}")
    return f"[QueryPlanner] {', '.join(parts)}"


def _quality_hint(out: dict, item: dict) -> str:
    """One-line quality hint for terminal summary."""
    ans = out.get("answer", "")
    n_src = out.get("num_sources", 0)
    n_char = out.get("answer_length", 0)
    hints = []
    if not ans:
        hints.append("EMPTY")
    elif "OPENAI_API_KEY" in ans or "Error calling" in ans:
        hints.append("API_ERROR")
    elif "No relevant information" in ans and n_src == 0:
        hints.append("NO_CONTEXT")
    elif n_src == 0:
        hints.append("NO_SOURCES")
    elif n_char < 100 and "don't have" in ans.lower():
        hints.append("MISSING_DATA")
    return " | ".join(hints) if hints else "ok"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run SuperGPT evaluation queries")
    parser.add_argument("--dry-run", action="store_true", help="Only print queries, do not call chatbot")
    parser.add_argument("--category", choices=CATEGORIES, help="Run only this category")
    parser.add_argument("--id", dest="query_id", help="Run only the test with this id")
    parser.add_argument("--summary", action="store_true", help="Print one-line quality hint after each answer")
    parser.add_argument("--output", "-o", dest="output_file", metavar="FILE", help="Write output to FILE")
    parser.add_argument("--model", dest="model", metavar="MODEL",
                        help="Override main LLM model (e.g. 'claude-sonnet-4-5' for cheaper testing)")
    args = parser.parse_args()

    queries = ALL_QUERIES
    if args.category:
        queries = [q for q in queries if q["category"] == args.category]
    if args.query_id:
        queries = [q for q in queries if q["id"] == args.query_id]
        if not queries:
            print(f"No test with id '{args.query_id}'")
            sys.exit(1)

    if args.dry_run:
        for q in queries:
            print(f"[{q['category']}] {q['id']}: {q['query']}")
        print(f"\nTotal: {len(queries)} queries")
        return

    chatbot = get_chatbot(model_override=args.model)
    summary_lines = []
    out_file = open(args.output_file, "w", encoding="utf-8") if getattr(args, "output_file", None) else None

    def log(msg: str = ""):
        print(msg)
        if out_file:
            out_file.write(msg + "\n" if msg else "\n")
            out_file.flush()

    for i, item in enumerate(queries, 1):
        log()
        log("=" * 80)
        log(f"  [{i}/{len(queries)}] [{item['category']}] {item['id']}")
        log(f"  Query: {item['query']}")
        log(f"  Expect: {item['expect']}")
        log("=" * 80)
        try:
            out = run_query(chatbot, item)
            log(_format_plan(out))
            log(out["answer"])
            if out["sources"]:
                log()
                log("Sources: " + str(out["sources"]))
            else:
                log()
                log("Sources: (none)")
            hint = _quality_hint(out, item)
            if args.summary:
                line = f"  [{item['id']}] len={out.get('answer_length', 0)} src={out.get('num_sources', 0)}  {hint}"
                summary_lines.append((item["id"], line, hint))
                log()
                log("  >>> " + line)
        except Exception as e:
            log(f"Error: {e}")
            if args.summary:
                summary_lines.append((item["id"], f"  [{item['id']}] EXCEPTION: {e}", "EXCEPTION"))
            import traceback
            traceback.print_exc()
            if out_file:
                traceback.print_exc(file=out_file)

    if args.summary and summary_lines:
        log()
        log("=" * 80)
        log("QUALITY SUMMARY")
        log("=" * 80)
        for _id, line, hint in summary_lines:
            log(line)
        problems = [h for (_, _, h) in summary_lines if h not in ("ok",)]
        if problems:
            log()
            log(f"  Potential issues: {len(problems)} query(ies) with hints other than 'ok'")

    # --- Token usage summary ---
    try:
        from src.chatbot import get_llm_usage
        from src.query_planner import get_planner_usage
        llm = get_llm_usage()
        planner = get_planner_usage()
        log()
        log("=" * 80)
        log("TOKEN USAGE SUMMARY")
        log("=" * 80)
        log(f"  Main LLM   — calls: {llm.get('calls', 0)}, input: {llm.get('input_tokens', 0):,}, "
            f"output: {llm.get('output_tokens', 0):,}, "
            f"cache_read: {llm.get('cache_read', 0):,}, cache_create: {llm.get('cache_creation', 0):,}")
        log(f"  Planner    — calls: {planner.get('calls', 0)}, input: {planner.get('input_tokens', 0):,}, "
            f"output: {planner.get('output_tokens', 0):,}, "
            f"cache_read: {planner.get('cache_read', 0):,}, cache_create: {planner.get('cache_creation', 0):,}")
        total_in = llm.get("input_tokens", 0) + planner.get("input_tokens", 0)
        total_out = llm.get("output_tokens", 0) + planner.get("output_tokens", 0)
        log(f"  TOTAL      — input: {total_in:,}, output: {total_out:,}")
        provider = os.getenv("LLM_PROVIDER", "openai")
        if provider == "anthropic":
            main_model = args.model or os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")
            pricing = {
                "claude-opus-4-5": (15.0, 75.0),
                "claude-sonnet-4-5": (3.0, 15.0),
                "claude-haiku-4-5": (0.80, 4.0),
            }
            p_in, p_out = pricing.get(main_model, (15.0, 75.0))
            est = (llm.get("input_tokens", 0) * p_in + llm.get("output_tokens", 0) * p_out) / 1_000_000
            est += (planner.get("input_tokens", 0) * 0.80 + planner.get("output_tokens", 0) * 4.0) / 1_000_000
            log(f"  Est. cost  — ~${est:.2f} (main={main_model}, planner=haiku)")
    except Exception:
        pass

    log()
    log("Done.")
    if out_file:
        out_file.close()
        print(f"Output written to {args.output_file}")


# =============================================================================
# Pytest support
# =============================================================================

def pytest_generate_tests(metafunc):
    if "super_query" not in metafunc.fixturenames:
        return
    metafunc.parametrize("super_query", ALL_QUERIES, ids=[q["id"] for q in ALL_QUERIES])


def test_super_query_returns_answer(super_query):
    """Each query returns a non-empty answer."""
    chatbot = get_chatbot()
    result = chatbot.answer(super_query["query"])
    answer = result.get("answer", "")
    assert answer, f"Empty answer for: {super_query['query']}"
    assert "OPENAI_API_KEY" not in answer and "Error calling" not in answer, (
        f"Looks like an API/config error: {answer[:200]}"
    )


def test_super_query_has_sources(super_query):
    """Queries should return at least one source."""
    chatbot = get_chatbot()
    result = chatbot.answer(super_query["query"])
    sources = result.get("sources", [])
    assert isinstance(sources, list), "sources should be a list"


if __name__ == "__main__":
    main()
