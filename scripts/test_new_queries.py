"""Test the 10 new accessory + discovery queries through the retrieval pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from src.chatbot import SupermicroChatbot

bot = SupermicroChatbot(
    llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
    llm_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.5")),
    top_p=float(os.getenv("LLM_TOP_P", "1.0")),
)

QUERIES = [
    ("accessory_railkit_511r", "Find me the rail kit part number for SYS-511R-M"),
    ("accessory_psu_1u", "What power supply options are available for 1U servers?"),
    ("accessory_cable_mgmt", "Do you have a cable management arm for 1U chassis?"),
    ("accessory_railkit_2u", "What rail kit do I need for a 2U rackmount server?"),
    ("accessory_addons_521ge", "What add-on cards are compatible with SYS-521GE-TNRT?"),
    ("discovery_epyc_2u_12bay", "Find me a dual processor 2U AMD EPYC 9005 system that supports 12 3.5 drive bays"),
    ("discovery_1u_nvme", "I need a 1U single processor server with at least 10 NVMe drive bays"),
    ("discovery_2u_xeon_24dimm", "Show me 2U dual Intel Xeon servers with 24 or more DIMM slots"),
    ("discovery_1u_dual_gpu", "What 1U dual processor servers support GPUs?"),
    ("discovery_2u_sas", "Find a 2U server with SAS drive support and hot-swap bays"),
]

for qid, query in QUERIES:
    print(f"\n{'='*70}")
    print(f"[{qid}] {query}")
    print('='*70)
    result = bot.answer(query)
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    # Print first 500 chars of answer
    print(f"ANSWER: {answer[:500]}{'...' if len(answer) > 500 else ''}")
    print(f"SOURCES ({len(sources)}): {sources[:8]}")
    print()
