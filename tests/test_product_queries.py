"""
Unified test suite for the Supermicro RAG system.

Covers all query types: listing, detail, SKU-specific detail, recommendations,
comparisons, general knowledge, conversational, misspell/partial, multi-product,
and follow-up handling.

Run to see results:
  python tests/test_product_queries.py              # run all, print answers
  python tests/test_product_queries.py --summary     # + one-line quality hint per query
  python tests/test_product_queries.py --dry-run     # print queries only
  python tests/test_product_queries.py --category followup  # one category only
  pytest tests/test_product_queries.py -v            # run as pytest (assertions)

Use --output FILE to write the same output to a file for later review.

Requires: .env with OPENAI_API_KEY (and optionally ANTHROPIC_API_KEY).
For listing tests to return catalog data, ensure data/pages/products.jsonl exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root on path so "src" and "tests" both work
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env from repo root
_dotenv = REPO_ROOT / ".env"
if _dotenv.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_dotenv, override=False)


# =============================================================================
# Product test queries (list / detail / compare / general / recommendation)
# =============================================================================

PRODUCT_TEST_QUERIES = [
    # --- Listing: catalog + RAG ---
    {
        "id": "list_1u",
        "category": "list",
        "query": "List all 1U servers",
        "expect": "Multiple 1U products; catalog or docs",
    },
    {
        "id": "list_gpu",
        "category": "list",
        "query": "What GPU servers do you have?",
        "expect": "GPU-optimized or GPU-capable systems",
    },
    {
        "id": "list_gold_series",
        "category": "list",
        "query": "Show me Gold Series products",
        "expect": "Pre-configured Gold Series SKUs",
    },
    {
        "id": "list_2u_clouddc",
        "category": "list",
        "query": "What 2U CloudDC servers are available?",
        "expect": "2U form factor + CloudDC tag",
    },
    {
        "id": "list_storage",
        "category": "list",
        "query": "List storage solutions",
        "expect": "SuperStorage or storage-focused products",
    },
    {
        "id": "list_fattwin_dual_platform",
        "category": "list",
        "query": "Show me FatTwin servers available in both Intel and AMD variants",
        "expect": "FatTwin products covering Intel and AMD CPU platforms",
    },
    # --- Conversational: global/golden SKUs, GPU counts, motherboards, switches ---
    {
        "id": "conv_global_skus",
        "category": "conversational",
        "query": "What is the Global SKU program?",
        "expect": "Program explanation from docs (not a product list)",
    },
    {
        "id": "conv_list_global_skus",
        "category": "conversational",
        "query": "Give me a list of global skus",
        "expect": "RAG answer about global SKU program or catalog if applicable",
    },
    {
        "id": "conv_golden_skus",
        "category": "conversational",
        "query": "What are golden skus?",
        "expect": "Gold Series pre-configured SKUs (catalog or docs)",
    },
    {
        "id": "conv_how_many_gpus_server",
        "category": "conversational",
        "query": "How many GPUs does the SYS-521GE-TNRT support?",
        "expect": "GPU count or expansion slots for this model",
    },
    {
        "id": "conv_how_many_gpu_servers",
        "category": "conversational",
        "query": "How many GPU servers do you have?",
        "expect": "Enumeration of GPU server offerings or count",
    },
    {
        "id": "conv_motherboard_x13",
        "category": "conversational",
        "query": "Tell me about the X13DEI motherboard",
        "expect": "Form factor, chipset, CPU support, features",
    },
    {
        "id": "conv_motherboard_intel",
        "category": "conversational",
        "query": "Which motherboards support 4th Gen Intel Xeon?",
        "expect": "Board models or series that support Sapphire Rapids",
    },
    {
        "id": "conv_motherboard_atx",
        "category": "conversational",
        "query": "What Supermicro motherboards are ATX?",
        "expect": "ATX form factor boards",
    },
    {
        "id": "conv_switches",
        "category": "conversational",
        "query": "Do you have network switches?",
        "expect": "Switch products or clarification that focus is servers/storage",
    },
    {
        "id": "conv_switches_nvme",
        "category": "conversational",
        "query": "What NVMe or PCIe switches do Supermicro systems support?",
        "expect": "Switch/expander options in servers or docs",
    },
    # --- Detail / specs: specific product or model ---
    {
        "id": "detail_sys_model",
        "category": "detail",
        "query": "What are the specs of SYS-521GE-TNRT?",
        "expect": "Form factor, CPU, GPU, memory, storage for this model",
    },
    {
        "id": "detail_partial_model",
        "category": "detail",
        "query": "Tell me about the 521GE",
        "expect": "Info about 521GE family or specific SKU",
    },
    {
        "id": "detail_8u_gpu",
        "category": "detail",
        "query": "What is the power consumption of the 8U GPU server?",
        "expect": "Power specs for 8U GPU system(s)",
    },
    {
        "id": "spec_memory",
        "category": "detail",
        "query": "Which servers support more than 2TB of memory?",
        "expect": "Models with high memory capacity",
    },
    {
        "id": "spec_nvidia",
        "category": "detail",
        "query": "What GPU servers support NVIDIA HGX H100?",
        "expect": "Systems supporting HGX H100",
    },
    {
        "id": "detail_ssg640_psu",
        "category": "detail",
        "query": "What power supply type ships standard with the SSG-640SP-E1CR60?",
        "expect": "AC vs DC PSU info for this storage server model",
    },
    {
        "id": "detail_621c_gpu_slots",
        "category": "detail",
        "query": "How many full-height full-length GPUs can the SYS-621C-TN12R accommodate?",
        "expect": "FHFL GPU slot count or expansion capability for SYS-621C-TN12R",
    },
    {
        "id": "detail_421ge_processors",
        "category": "detail",
        "query": "What processors are compatible with the SYS-421GE-NBRT-LCC?",
        "expect": "CPU compatibility list for this GPU server model",
    },
    {
        "id": "detail_521c_socket_count",
        "category": "detail",
        "query": "Does the SYS-521C-NR have one or two CPU sockets?",
        "expect": "Single-socket (UP) vs dual-socket (DP) clarification",
    },
    {
        "id": "detail_111c_gpu_width",
        "category": "detail",
        "query": "Will double-width GPU cards fit in the SYS-111C-NR?",
        "expect": "Physical GPU card clearance or slot width info",
    },
    {
        "id": "detail_111c_raid_support",
        "category": "detail",
        "query": "What RAID capabilities does the SYS-111C-NR offer?",
        "expect": "Hardware or software RAID options for this 1U model",
    },
    {
        "id": "detail_521c_expansion",
        "category": "detail",
        "query": "Describe the PCIe expansion slots available on SYS-521C-NR",
        "expect": "PCIe lane layout, generations, and slot config",
    },
    {
        "id": "detail_4125gs_variant_table",
        "category": "detail",
        "query": "Compare the AS-4125GS-TNRT, TNRT1, and TNRT2 variants side by side",
        "expect": "Tabular or structured comparison of all three SKU variants",
    },
    # --- Compare ---
    {
        "id": "compare_1u_2u",
        "category": "compare",
        "query": "Compare 1U and 2U servers",
        "expect": "Differences in density, expandability, use cases",
    },
    {
        "id": "compare_gold_standard",
        "category": "compare",
        "query": "Difference between Gold Series and standard SKUs",
        "expect": "Pre-configured vs build-to-order or similar",
    },
    {
        "id": "compare_hyper_h14_vs_h13",
        "category": "compare",
        "query": "How does the H14 Hyper series improve over H13 Hyper?",
        "expect": "Generational upgrades in CPU, memory, I/O between H13 and H14 Hyper",
    },
    {
        "id": "compare_flex_vs_big_twin",
        "category": "compare",
        "query": "FlexTwin vs BigTwin — what sets them apart?",
        "expect": "Architectural and feature differences between these multi-node platforms",
    },
    {
        "id": "compare_clouddc_generations",
        "category": "compare",
        "query": "What changed between the X13 and X14 CloudDC platforms?",
        "expect": "Platform evolution from X13 to X14 in the CloudDC line",
    },
    {
        "id": "compare_421ge_422ga",
        "category": "compare",
        "query": "Compare SYS-421GE-NBRT-LCC against SYS-422GA-NBRT-LCC",
        "expect": "Spec or architecture differences between these two GPU server models",
    },
    {
        "id": "compare_grandtwin_gens",
        "category": "compare",
        "query": "X13 GrandTwin vs X14 GrandTwin — key differences?",
        "expect": "Generational comparison for the GrandTwin multi-node platform",
    },
    {
        "id": "compare_wio_e_series",
        "category": "compare",
        "query": "Break down the differences between SYS-511E-WR, SYS-111E-WR, and SYS-521E-WR",
        "expect": "Three-way comparison covering form factor, expansion, and use cases",
    },
    # --- Misspelled / partial / fuzzy product names ---
    {
        "id": "misspell_521ge_space",
        "category": "misspell",
        "query": "What is the 521 GE?",
        "expect": "Should resolve to SYS-521GE-TNRT despite the space",
    },
    {
        "id": "misspell_521ge_typo",
        "category": "misspell",
        "query": "Tell me about SYS-521GE-TNR",
        "expect": "Should find SYS-521GE-TNRT even with missing T at end",
    },
    {
        "id": "misspell_microcloud_typo",
        "category": "misspell",
        "query": "What micro cloud servers do you have?",
        "expect": "Should find MicroCloud products despite space in name",
    },
    {
        "id": "misspell_bigtwin",
        "category": "misspell",
        "query": "Tell me about big twin servers",
        "expect": "Should find BigTwin multi-node systems",
    },
    {
        "id": "partial_just_number",
        "category": "misspell",
        "query": "530MT",
        "expect": "Should find SYS-530MT-H12TRF MicroCloud or similar",
    },
    {
        "id": "partial_chassis",
        "category": "misspell",
        "query": "What is the X13DEG motherboard?",
        "expect": "Should find X13DEG-OA or X13DEG-OAD motherboard info",
    },
    {
        "id": "partial_lowercase",
        "category": "misspell",
        "query": "sys-821ge",
        "expect": "Should find SYS-821GE despite all lowercase",
    },
    # --- Multiple products in one query ---
    {
        "id": "multi_compare_two",
        "category": "multi",
        "query": "Compare SYS-521GE-TNRT and SYS-421GE-TNRT",
        "expect": "Comparison of both models with specs; should have sources for both",
    },
    {
        "id": "multi_compare_form_factors",
        "category": "multi",
        "query": "What's the difference between the 4U and 5U GPU servers?",
        "expect": "Compare 4U vs 5U GPU systems (421GE vs 521GE or similar)",
    },
    {
        "id": "multi_list_two_families",
        "category": "multi",
        "query": "List both MicroCloud and BigTwin servers",
        "expect": "Products from both MicroCloud and Twin/BigTwin families",
    },
    {
        "id": "multi_three_products",
        "category": "multi",
        "query": "Give me specs on SYS-521GE-TNRT, SYS-821GE-TNHR, and AS-3015MR-H10TNR",
        "expect": "Specs for all three products; sources should cover all three",
    },
    {
        "id": "multi_gpu_storage",
        "category": "multi",
        "query": "Do you have servers that support both GPUs and high storage capacity?",
        "expect": "Systems with GPU support AND significant storage (e.g., GPU servers with many drive bays)",
    },
    # --- General knowledge ---
    {
        "id": "general_ipmi",
        "category": "general",
        "query": "How do I configure IPMI?",
        "expect": "IPMI configuration from docs",
    },
    {
        "id": "general_dcscm",
        "category": "general",
        "query": "What is DCSCM?",
        "expect": "Definition/explanation from docs",
    },
    {
        "id": "general_gb200_platform",
        "category": "general",
        "query": "Tell me about the GB200 NVL72 platform",
        "expect": "Supermicro GB200 NVL72 GPU rack-scale architecture info",
    },
    {
        "id": "general_grandtwin_benefits",
        "category": "general",
        "query": "Why would I choose a GrandTwin over other multi-node systems?",
        "expect": "Advantages of GrandTwin architecture vs alternatives",
    },
    {
        "id": "general_h100_air_cooling",
        "category": "general",
        "query": "What air-cooled options exist for H100 GPUs?",
        "expect": "Supermicro systems with air-cooled NVIDIA H100 support",
    },
    {
        "id": "general_b200_specs",
        "category": "general",
        "query": "What are the specifications of the B200 air-cooled GPU server?",
        "expect": "Hardware specs for air-cooled B200 GPU system",
    },
    {
        "id": "general_openbmc_support",
        "category": "general",
        "query": "What Supermicro products offer OpenBMC support?",
        "expect": "Server models or platforms with OpenBMC management",
    },
    {
        "id": "general_x14_wio_lineup",
        "category": "general",
        "query": "List the X14 single-processor WIO server lineup",
        "expect": "X14-generation UP WIO models or product family overview",
    },
    {
        "id": "general_gold_gpu_skus",
        "category": "general",
        "query": "Are there Gold Series SKUs for GPU servers?",
        "expect": "Gold Series pre-configured GPU server offerings",
    },
    # --- Recommendation: user asks for a system suggestion ---
    {
        "id": "rec_smb_file_server",
        "category": "recommendation",
        "query": "What server would you recommend for a small business file sharing setup?",
        "expect": "Storage-friendly system suitable for SMB NAS or file server use",
    },
    {
        "id": "rec_8gpu_h200",
        "category": "recommendation",
        "query": "I need a system that can fit 8 H200 GPUs, what are my options?",
        "expect": "8-GPU HGX or large-form-factor server with H200 support",
    },
    {
        "id": "rec_dense_multinode",
        "category": "recommendation",
        "query": "Suggest a high-density multi-node compute platform",
        "expect": "Dense multi-node product (Twin, FatTwin, MicroCloud, etc.)",
    },
    {
        "id": "rec_111e_vs_112b",
        "category": "recommendation",
        "query": "Help me decide between SYS-111E-WR and SYS-112B-WR for my workload",
        "expect": "Side-by-side analysis with a recommendation based on trade-offs",
    },
    {
        "id": "rec_xeon6_system",
        "category": "recommendation",
        "query": "What systems support the latest Intel Xeon 6 processors?",
        "expect": "Granite Rapids or Sierra Forest server options",
    },
    {
        "id": "rec_x14_hyper_single",
        "category": "recommendation",
        "query": "Suggest an X14 Hyper single-socket server",
        "expect": "X14-gen UP Hyper model recommendation",
    },
    {
        "id": "rec_rtx6000_pro",
        "category": "recommendation",
        "query": "Which servers are compatible with NVIDIA RTX 6000 Pro GPUs?",
        "expect": "Workstation or GPU server models supporting RTX 6000 Pro",
    },
    # --- Accessory / part number lookups (cross-document graph traversal) ---
    {
        "id": "accessory_railkit_511r",
        "category": "accessory",
        "query": "Find me the rail kit part number for SYS-511R-M",
        "expect": "MCP-290-00056-0N or similar rail kit SKU compatible with 813M chassis",
    },
    {
        "id": "accessory_psu_options_1u",
        "category": "accessory",
        "query": "What power supply options are available for 1U servers?",
        "expect": "PWS-series 1U power supply models with wattage info",
    },
    {
        "id": "accessory_cable_mgmt_1u",
        "category": "accessory",
        "query": "Do you have a cable management arm for 1U chassis?",
        "expect": "MCP-290 cable management arm part numbers for 1U",
    },
    {
        "id": "accessory_railkit_2u",
        "category": "accessory",
        "query": "What rail kit do I need for a 2U rackmount server?",
        "expect": "2U rail kit part numbers (MCP-290 series)",
    },
    {
        "id": "accessory_compatible_addons",
        "category": "accessory",
        "query": "What add-on cards are compatible with SYS-521GE-TNRT?",
        "expect": "AOC or add-on card part numbers compatible with this GPU server",
    },
    # --- Multi-constraint product discovery ---
    {
        "id": "discovery_dual_2u_epyc_12bay",
        "category": "discovery",
        "query": "Find me a dual processor 2U AMD EPYC 9005 system that supports 12 3.5 drive bays",
        "expect": "AS-2025HS-TNR or similar 2U dual EPYC with 12x 3.5 inch bays",
    },
    {
        "id": "discovery_1u_single_nvme",
        "category": "discovery",
        "query": "I need a 1U single processor server with at least 10 NVMe drive bays",
        "expect": "1U single-socket systems with 10+ NVMe bays (e.g., SYS-111C-NR, AS-1115CS-TNR)",
    },
    {
        "id": "discovery_2u_dual_xeon_24dimm",
        "category": "discovery",
        "query": "Show me 2U dual Intel Xeon servers with 24 or more DIMM slots",
        "expect": "2U dual-socket Xeon systems with high memory capacity",
    },
    {
        "id": "discovery_1u_dual_gpu_capable",
        "category": "discovery",
        "query": "What 1U dual processor servers support GPUs?",
        "expect": "1U dual-socket systems with GPU support listed",
    },
    {
        "id": "discovery_2u_storage_sas",
        "category": "discovery",
        "query": "Find a 2U server with SAS drive support and hot-swap bays",
        "expect": "2U systems with SAS-capable hot-swap storage configurations",
    },
    # --- FAQ / eStore operational questions ---
    {
        "id": "faq_return_policy",
        "category": "faq",
        "query": "What is the return policy for Supermicro eStore?",
        "expect": "30-day return window; software/lifestyle non-returnable; restocking fee up to 15%",
    },
    {
        "id": "faq_shipping_international",
        "category": "faq",
        "query": "Does Supermicro ship internationally?",
        "expect": "US and Canada only; does not ship to US territories",
    },
    {
        "id": "faq_payment_methods",
        "category": "faq",
        "query": "What payment methods does Supermicro eStore accept?",
        "expect": "Visa, MasterCard, American Express, Discover; no check/money order/PO",
    },
    {
        "id": "faq_cancel_order",
        "category": "faq",
        "query": "How do I cancel my order on the eStore?",
        "expect": "Cancel via My Account > My Orders; cancellation is permanent",
    },
    {
        "id": "faq_warranty_servers",
        "category": "faq",
        "query": "What is the warranty on Supermicro eStore servers?",
        "expect": "3-year labor and parts; 1-year cross-shipping; extended warranty must be added at purchase",
    },
    {
        "id": "faq_software_license_key",
        "category": "faq",
        "query": "Where do I find and generate my software license key on the eStore?",
        "expect": "My Account > My Software for Software Orders, Product Keys, Generate Key; see Generation Software Key guide",
    },
    {
        "id": "faq_tax_exemption",
        "category": "faq",
        "query": "How do I apply for tax exemption on the eStore?",
        "expect": "Apply through eStore account > My Account > tax exemption link; reviewed within 1 business day",
    },
    {
        "id": "faq_free_shipping",
        "category": "faq",
        "query": "Does Supermicro offer free shipping?",
        "expect": "Free shipping within Continental US on purchases over $200 before tax",
    },
    # --- FAQ wording variation tests ---
    {
        "id": "faq_var_refund_wording",
        "category": "faq",
        "query": "Can I get a refund if I changed my mind about a purchase?",
        "expect": "30-day return window; restocking fee up to 15%; must obtain RMA",
    },
    {
        "id": "faq_var_guest_checkout",
        "category": "faq",
        "query": "Can I order from the eStore without creating an account?",
        "expect": "Account is required to place an order; no guest checkout available",
    },
    {
        "id": "faq_var_backorder_partial_ship",
        "category": "faq",
        "query": "If part of my order is backordered, will you send what's available now?",
        "expect": "Entire order ships together; recommend placing separate orders",
    },
    {
        "id": "faq_var_password_requirements",
        "category": "faq",
        "query": "My new password keeps getting rejected, what are the rules?",
        "expect": "At least 12 characters; requires lowercase, uppercase, digit, special character",
    },
    {
        "id": "faq_var_combine_accounts",
        "category": "faq",
        "query": "I have two eStore accounts, can you merge them?",
        "expect": "No option to combine accounts",
    },
    {
        "id": "faq_var_credit_card_types",
        "category": "faq",
        "query": "Can I pay with PayPal or a wire transfer?",
        "expect": "PayPal accepted; wire transfer available (contact support to enable); one payment method per order",
    },
    {
        "id": "faq_var_track_my_order",
        "category": "faq",
        "query": "How can I check where my package is right now?",
        "expect": "Tracking info available via My Account > My Orders or shipment confirmation email",
    },
    {
        "id": "faq_var_tax_canada_reseller",
        "category": "faq",
        "query": "I'm reselling from Canada, do I still get charged sales tax?",
        "expect": "No tax on orders shipped to Canada",
    },
]


# =============================================================================
# Follow-up tests: multi-turn conversations
# Each has a "conversation" (prior turns) and "query" (the current question).
# Tests both TRUE follow-ups and NEW product questions mid-session.
# =============================================================================

FOLLOWUP_TEST_QUERIES = [
    # --- TRUE follow-ups: should use conversation context ---
    {
        "id": "followup_its_storage",
        "category": "followup",
        "conversation": "User: What is SYS-521GE-TNRT?\nAssistant: The SYS-521GE-TNRT is a 5U GPU SuperServer with dual Intel Xeon Scalable processors, up to 10 GPUs, and 32 DIMM slots.",
        "query": "What about its storage options?",
        "expect": "Storage info for SYS-521GE-TNRT (the product from conversation); should NOT return unrelated products",
        "should_followup": True,
        "bad_if_contains": [],
    },
    {
        "id": "followup_how_many_gpus",
        "category": "followup",
        "conversation": "User: Tell me about SYS-421GE-TNRT\nAssistant: The SYS-421GE-TNRT is a 4U GPU server supporting multiple GPUs with PCIe 5.0.",
        "query": "How many GPUs does it support?",
        "expect": "GPU count for SYS-421GE-TNRT (from conversation); referential 'it' detected",
        "should_followup": True,
        "bad_if_contains": [],
    },
    {
        "id": "followup_tell_me_more",
        "category": "followup",
        "conversation": "User: What is the X13DEI motherboard?\nAssistant: The X13DEI is a Supermicro motherboard supporting 4th Gen Intel Xeon Scalable processors in an ATX form factor.",
        "query": "Tell me more about that",
        "expect": "More detail on X13DEI (from conversation); continuation phrase detected",
        "should_followup": True,
        "bad_if_contains": [],
    },
    {
        "id": "followup_affirmative_yes",
        "category": "followup",
        "conversation": "User: What MicroCloud systems are available?\nAssistant: The H13 MicroCloud family includes AS-3015MR-H10TNR (10 nodes), AS-3015MR-H8TNR (8 nodes), and AS-3015MR-H5TNR (5 nodes). Would you like details on any of these?",
        "query": "yes",
        "expect": "More detail on MicroCloud (uses last assistant message for retrieval)",
        "should_followup": True,
        "bad_if_contains": [],
    },
    {
        "id": "followup_more_details",
        "category": "followup",
        "conversation": "User: What is SYS-521GE-TNRT?\nAssistant: The SYS-521GE-TNRT is a 5U GPU SuperServer with dual Intel Xeon processors and up to 10 GPUs.",
        "query": "More details please",
        "expect": "Expanded info about SYS-521GE-TNRT; continuation phrase detected",
        "should_followup": True,
        "bad_if_contains": [],
    },

    # --- NEW product in same session: must NOT follow up ---
    {
        "id": "new_product_721ge_after_521ge",
        "category": "followup",
        "conversation": "User: What is 521GE?\nAssistant: The SYS-521GE-TNRT is a 5U GPU SuperServer with dual Intel Xeon Scalable processors, up to 10 GPUs, 32 DIMM slots, and 4x 2700W PSUs.",
        "query": "721ge?",
        "expect": "Info about 721GE (NOT 521GE); should treat as new query, not follow-up. If no datasheet, say so — do NOT hallucinate specs.",
        "should_followup": False,
        "bad_if_contains": ["521GE-TNRT"],
    },
    {
        "id": "new_product_821ge_after_521ge",
        "category": "followup",
        "conversation": "User: What is SYS-521GE-TNRT?\nAssistant: The SYS-521GE-TNRT is a 5U GPU SuperServer.",
        "query": "821ge?",
        "expect": "Info about 821GE only; must NOT mix in 521GE specs or hallucinate from naming conventions",
        "should_followup": False,
        "bad_if_contains": ["521GE-TNRT"],
    },
    {
        "id": "new_product_x13_after_521ge",
        "category": "followup",
        "conversation": "User: Tell me about SYS-521GE-TNRT\nAssistant: The SYS-521GE-TNRT is a 5U GPU server.",
        "query": "x13",
        "expect": "X13 motherboard family info; must NOT be about 521GE",
        "should_followup": False,
        "bad_if_contains": ["521GE"],
    },
    {
        "id": "new_product_full_code_after_session",
        "category": "followup",
        "conversation": "User: What is SYS-521GE-TNRT?\nAssistant: The SYS-521GE-TNRT is a 5U GPU SuperServer.",
        "query": "SYS-530MT-H12TRF",
        "expect": "Specs for SYS-530MT-H12TRF MicroCloud; must NOT reference 521GE",
        "should_followup": False,
        "bad_if_contains": ["521GE"],
    },
    {
        "id": "new_product_microcloud_after_gpu",
        "category": "followup",
        "conversation": "User: What GPU servers do you have?\nAssistant: Supermicro offers GPU-optimized servers including SYS-521GE-TNRT (5U, 10 GPUs), SYS-421GE-TNRT (4U), and SYS-420GP-TNR (4U).",
        "query": "list microcloud systems",
        "expect": "MicroCloud product listing; must NOT return GPU server info from conversation",
        "should_followup": False,
        "bad_if_contains": ["521GE", "421GE", "420GP"],
    },
    {
        "id": "new_general_after_product",
        "category": "followup",
        "conversation": "User: What is SYS-521GE-TNRT?\nAssistant: The SYS-521GE-TNRT is a 5U GPU SuperServer with dual Intel Xeon processors.",
        "query": "How does IPMI work?",
        "expect": "General IPMI explanation; must NOT reference 521GE or GPU server",
        "should_followup": False,
        "bad_if_contains": ["521GE"],
    },
]


def get_chatbot(model_override=None):
    """Lazy init chatbot. model_override replaces the main LLM model (e.g. 'claude-sonnet-4-5' for cheaper testing)."""
    from src.chatbot import SupermicroChatbot
    from src.embed import get_qdrant_client

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    primary_collection = os.getenv("QDRANT_COLLECTION_PRIMARY", "supermicro_primary")
    manual_collection = os.getenv("QDRANT_COLLECTION_MANUAL", "supermicro_manual")

    provider = os.getenv("LLM_PROVIDER", "openai")
    model = model_override or os.getenv("LLM_MODEL", "gpt-5.2")
    if model_override and provider == "anthropic":
        os.environ["ANTHROPIC_MODEL"] = model_override

    client = get_qdrant_client(qdrant_url, qdrant_api_key)
    return SupermicroChatbot(
        qdrant_client=client,
        primary_collection=primary_collection,
        manual_collection=manual_collection,
        llm_model=model,
        llm_provider=provider,
        top_k=int(os.getenv("TOP_K", "10")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.5")),
        top_p=float(os.getenv("LLM_TOP_P", "1.0")),
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


# =============================================================================
# Script mode: run all queries and print results
# =============================================================================

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
    elif item["category"] in ("list", "detail", "conversational", "recommendation") and n_src == 0:
        hints.append("NO_SOURCES")
    elif n_char < 100 and "don't have" in ans.lower():
        hints.append("MISSING_DATA")
    # Follow-up specific checks
    if item.get("bad_if_contains"):
        for bad_term in item["bad_if_contains"]:
            if bad_term.lower() in ans.lower():
                hints.append(f"CONTAMINATED({bad_term})")
    return " | ".join(hints) if hints else "ok"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run product queries and print results")
    parser.add_argument("--dry-run", action="store_true", help="Only print queries, do not call chatbot")
    parser.add_argument("--category", choices=["list", "detail", "compare", "general", "conversational", "recommendation", "followup", "misspell", "multi", "accessory", "discovery", "faq"], help="Run only this category")
    parser.add_argument("--id", dest="query_id", help="Run only the test with this id (e.g. list_1u)")
    parser.add_argument("--summary", action="store_true", help="Print a one-line quality hint after each answer and a summary at the end")
    parser.add_argument("--output", "-o", dest="output_file", metavar="FILE", help="Write test output to FILE (same as terminal)")
    parser.add_argument("--model", dest="model", metavar="MODEL",
                        help="Override main LLM model (e.g. 'claude-sonnet-4-5' for cheaper testing)")
    args = parser.parse_args()

    all_queries = PRODUCT_TEST_QUERIES + FOLLOWUP_TEST_QUERIES
    queries = all_queries
    if args.category:
        queries = [q for q in queries if q["category"] == args.category]
    if args.query_id:
        queries = [q for q in queries if q["id"] == args.query_id]
        if not queries:
            print(f"No test with id '{args.query_id}'")
            sys.exit(1)

    if args.dry_run:
        for q in queries:
            extra = ""
            if q.get("conversation"):
                extra = f"  [conv: {q['conversation'][:60]}...]"
            print(f"[{q['category']}] {q['id']}: {q['query']}{extra}")
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
        if item.get("conversation"):
            log(f"  Conversation: {item['conversation'][:120]}...")
        log(f"  Query: {item['query']}")
        log(f"  Expect: {item['expect']}")
        if "should_followup" in item:
            log(f"  Should follow-up: {item['should_followup']}")
        if item.get("bad_if_contains"):
            log(f"  Bad if answer contains: {item['bad_if_contains']}")
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
# Pytest: parametrized tests (assert answer and sources)
# =============================================================================

ALL_QUERIES = PRODUCT_TEST_QUERIES + FOLLOWUP_TEST_QUERIES


def pytest_generate_tests(metafunc):
    if "product_query" not in metafunc.fixturenames:
        return
    metafunc.parametrize("product_query", ALL_QUERIES, ids=[q["id"] for q in ALL_QUERIES])


def test_product_query_returns_answer(product_query):
    """Each product query returns a non-empty answer."""
    chatbot = get_chatbot()
    conversation = product_query.get("conversation", "")
    result = chatbot.answer(product_query["query"], conversation_context=conversation)
    answer = result.get("answer", "")
    assert answer, f"Empty answer for: {product_query['query']}"
    assert "OPENAI_API_KEY" not in answer and "Error calling" not in answer, (
        f"Looks like an API/config error: {answer[:200]}"
    )


def test_product_query_list_has_sources(product_query):
    """Listing, detail, recommendation, and FAQ queries should typically have at least one source."""
    if product_query["category"] not in ("list", "detail", "conversational", "recommendation", "faq"):
        return  # skip for compare/general/followup
    chatbot = get_chatbot()
    result = chatbot.answer(product_query["query"])
    sources = result.get("sources", [])
    assert isinstance(sources, list), "sources should be a list"


def test_followup_no_contamination(product_query):
    """Follow-up tests with should_followup=False must NOT contain bad terms from prior conversation."""
    if product_query.get("should_followup") is not False:
        return  # only check new-product-in-session tests
    bad_terms = product_query.get("bad_if_contains", [])
    if not bad_terms:
        return
    chatbot = get_chatbot()
    conversation = product_query.get("conversation", "")
    result = chatbot.answer(product_query["query"], conversation_context=conversation)
    answer = result.get("answer", "")
    for term in bad_terms:
        assert term.lower() not in answer.lower(), (
            f"Answer for '{product_query['query']}' is contaminated with '{term}' from prior conversation. "
            f"This should be treated as a NEW query, not a follow-up.\nAnswer excerpt: {answer[:300]}"
        )


if __name__ == "__main__":
    main()
