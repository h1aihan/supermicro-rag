#!/usr/bin/env python3
"""
Main chatbot interface for Supermicro RAG system.
"""

import os
import argparse
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Support running as:
# - python -m src.chatbot   (package mode)
# - python src/chatbot.py   (script mode)
try:
    from src.query import RAGQueryProcessor
except ImportError:
    from query import RAGQueryProcessor


# Load environment variables
_repo_root_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_repo_root_env, override=False)


# =============================================================================
# SYSTEM PROMPT - Domain knowledge for Supermicro products
# =============================================================================
SYSTEM_MESSAGE = """You are a technical assistant specializing in Supermicro server and storage products.

## SUPERMICRO PRODUCT NAMING CONVENTIONS
- Server systems: SYS-{series}{form factor}{features}-{suffix} (e.g., SYS-521GE-TNRT, SYS-421GE-TNRT)
- AMD systems: AS-{series} (e.g., AS-4125GS-TNRT)
- Motherboards: X{generation}{chipset}-{features} (e.g., X13DEI-T, X12SPi-TF)
- Chassis: SC{series} or CSE-{series} (e.g., SC847, CSE-826)
- Power supplies: PWS-{wattage}{features} (e.g., PWS-1K28P-SQ)
- Blades: SBI-{series} (e.g., SBI-7428R-T3)
- Add-on cards: AOC-{type}-{features} (e.g., AOC-S3908L-H8IR)

## WHEN ANSWERING PRODUCT QUESTIONS
1. If asked about a partial model number (e.g., "521GE"), look for full model numbers containing that string
2. For product questions, provide key specifications when available:
   - Form factor (1U, 2U, 4U, etc.)
   - CPU support (Intel Xeon, AMD EPYC, etc.)
   - GPU support (if applicable)
   - Memory capacity and type
   - Storage options
   - Network connectivity
   - Target use cases
3. If multiple sources cover the same product, synthesize the information

## RESPONSE GUIDELINES
- Aim for 200-350 words - detailed enough to be helpful, but not rambling
- Focus on what you CAN answer, not what you can't
- When using information from the provided context, cite the source briefly
- You may supplement with your general knowledge when context is incomplete
- For comparisons, use tables

## CRITICAL: AVOID THESE BAD HABITS
- Do NOT list things you "need" or "would need" to answer better
- Do NOT say "the context doesn't include X" for multiple items - one brief mention is enough
- Do NOT write long explanations of what information is missing
- Do NOT over-hedge with phrases like "I can't confirm without...", "treat as TBD", etc.
- Do NOT reference unrelated products from conversation history

## WHEN DATA IS INCOMPLETE
If the exact product datasheet isn't available:
1. State briefly: "I don't have [product] datasheet in my database."
2. Provide what you DO know (general product family info, naming conventions)
3. Ask ONE clarifying question if needed
4. STOP - do not ramble about what's missing"""


def get_llm_response(prompt: str, model: str = "gpt-5.2", provider: str = "openai") -> str:
    """
    Get response from LLM.
    
    Args:
        prompt: Full prompt including system message, context, and question
        model: Model name
        provider: LLM provider (openai, ollama)
        
    Returns:
        LLM response text
    """
    if provider == "openai":
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return (
                    "Missing OPENAI_API_KEY. Create a `.env` file in the repo root and set:\n"
                    "  OPENAI_API_KEY=sk-...\n"
                    "Then re-run the chatbot (or set LLM_PROVIDER=ollama to avoid OpenAI)."
                )

            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5  # Balanced temperature for factual but flexible answers
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error calling OpenAI API: {e}"
    
    elif provider == "anthropic":
        try:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return (
                    "Missing ANTHROPIC_API_KEY. Create a `.env` file in the repo root and set:\n"
                    "  ANTHROPIC_API_KEY=sk-ant-...\n"
                    "Then re-run the chatbot."
                )
            
            client = Anthropic(api_key=api_key)
            anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")
            
            response = client.messages.create(
                model=anthropic_model,
                max_tokens=1024,
                system=SYSTEM_MESSAGE,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        except Exception as e:
            return f"Error calling Anthropic API: {e}"
    
    elif provider == "ollama":
        try:
            import requests
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
            
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.5}  # Balanced temperature
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        
        except Exception as e:
            return f"Error calling Ollama API: {e}"
    
    else:
        return f"Unknown LLM provider: {provider}"


class SupermicroChatbot:
    """Main chatbot class."""
    
    def __init__(
        self,
        index_dir: str = "embeddings/faiss_index/",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        llm_provider: str = "openai",
        top_k: int = 5
    ):
        """
        Initialize the chatbot.
        
        Args:
            index_dir: Directory containing FAISS index
            embedding_model: Sentence transformer model name
            llm_model: LLM model name
            llm_provider: LLM provider (openai, ollama)
            top_k: Number of chunks to retrieve
        """
        self.query_processor = RAGQueryProcessor(index_dir, embedding_model)
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.top_k = top_k
    
    def answer(self, question: str, conversation_context: str = "") -> dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            conversation_context: Previous conversation turns for context
            
        Returns:
            Dictionary with answer, sources, and retrieved chunks
        """
        # For follow-up questions, extract key terms from conversation
        retrieval_query = question
        if conversation_context:
            import re
            
            # Check if current query already contains a product code
            current_product_codes = re.findall(r'\b(?:SYS|AS|SSG|SBI|AOC)-[\w-]+\b', question, re.IGNORECASE)
            
            if current_product_codes:
                # Query has explicit product code - NOT a follow-up, use query as-is
                print(f"[DEBUG] Query has explicit product code: {current_product_codes}, using as-is")
            else:
                # Check if this looks like a follow-up (pronouns, short vague questions)
                has_pronouns = any(
                    word in question.lower() for word in ['it', 'this', 'that', 'its', 'their', 'one', 'the same']
                )
                is_short_vague = len(question.split()) <= 5
                
                is_followup = has_pronouns or is_short_vague
                
                if is_followup:
                    # Extract product codes - prioritize user messages, then assistant
                    # Parse conversation to separate user vs assistant content
                    user_content = ""
                    for line in conversation_context.split('\n'):
                        if line.startswith('User:') or line.startswith('user:'):
                            user_content += line + " "
                    
                    # First try user messages (the original query topic)
                    user_codes = re.findall(r'\b(?:SYS|AS|SSG|SBI|AOC)-[\w-]+\b', user_content, re.IGNORECASE)
                    # Then try full context (includes assistant mentions)
                    all_codes = re.findall(r'\b(?:SYS|AS|SSG|SBI|AOC)-[\w-]+\b', conversation_context, re.IGNORECASE)
                    
                    # Deduplicate while preserving order
                    seen = set()
                    user_codes_unique = []
                    for code in user_codes:
                        upper = code.upper()
                        if upper not in seen:
                            seen.add(upper)
                            user_codes_unique.append(code)
                    
                    all_codes_unique = []
                    seen_all = set()
                    for code in all_codes:
                        upper = code.upper()
                        if upper not in seen_all:
                            seen_all.add(upper)
                            all_codes_unique.append(code)
                    
                    print(f"[DEBUG] Follow-up detected. User codes: {user_codes_unique}, All codes: {all_codes_unique}")
                    
                    # Use FIRST product code from user messages, or first from all if none in user
                    product_code = None
                    if user_codes_unique:
                        product_code = user_codes_unique[0]  # First code user mentioned
                    elif all_codes_unique:
                        product_code = all_codes_unique[0]  # First code mentioned anywhere
                    
                    if product_code:
                        retrieval_query = f"{product_code} {question}"
                        print(f"[DEBUG] Expanded query: {retrieval_query}")
        
        # Retrieve relevant chunks
        chunks = self.query_processor.retrieve(retrieval_query, self.top_k)
        
        if not chunks:
            return {
                "answer": "No relevant information found in the documentation.",
                "sources": [],
                "chunks": []
            }
        
        # Format context
        context = self.query_processor.format_context(chunks)
        
        # Build prompt with structured format (include conversation context)
        prompt = self._build_user_prompt(question, context, chunks, conversation_context)
        
        # Get LLM response
        answer = get_llm_response(prompt, self.llm_model, self.llm_provider)
        
        # Extract unique sources (preserve ranking order - first occurrence wins)
        seen = set()
        sources = []
        for chunk in chunks:
            src = chunk["source_file"]
            if src not in seen:
                seen.add(src)
                sources.append(src)
        
        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks
        }
    
    def _build_user_prompt(self, question: str, context: str, chunks: list, conversation_context: str = "") -> str:
        """
        Build a structured user prompt for the LLM.
        
        Args:
            question: User's question
            context: Formatted context from retrieved chunks
            chunks: List of retrieved chunks (for metadata)
            conversation_context: Previous conversation turns
            
        Returns:
            Formatted prompt string
        """
        # Identify unique source types for context
        source_files = list(set(chunk["source_file"] for chunk in chunks))
        source_summary = ", ".join(source_files[:5])
        if len(source_files) > 5:
            source_summary += f" (+{len(source_files) - 5} more)"
        
        # Include conversation history if available
        conversation_section = ""
        if conversation_context:
            conversation_section = f"""## CONVERSATION HISTORY
{conversation_context}
---

"""
        
        prompt = f"""{conversation_section}## RETRIEVED CONTEXT
The following excerpts were retrieved from Supermicro documentation.
Sources: {source_summary}

---
{context}
---

## USER QUESTION
{question}

## INSTRUCTIONS
1. Use the retrieved context as your primary source of information
2. You may supplement with your general knowledge when the context is incomplete
3. When citing information from the context, mention the source document
4. For product questions, provide key specs: form factor, CPU, GPU, memory, storage, networking
5. If this is a follow-up question, refer to the conversation history for context
6. Be helpful and informative"""
        
        return prompt
    
    def interactive_mode(self):
        """Run interactive chat mode."""
        print("=" * 80)
        print("Supermicro RAG Chatbot")
        print("=" * 80)
        print("Ask questions about Supermicro products and documentation.")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nSearching documentation...")
                result = self.answer(question)
                
                print("\n" + "=" * 80)
                print("Answer:")
                print("=" * 80)
                print(result["answer"])
                
                if result["sources"]:
                    print("\n" + "=" * 80)
                    print("Sources:")
                    print("=" * 80)
                    for source in result["sources"]:
                        print(f"  - {source}")
                
                print()
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Supermicro RAG Chatbot"
    )
    parser.add_argument(
        "--query",
        help="Single question to answer"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--index-dir",
        default="embeddings/faiss_index/",
        help="Directory containing FAISS index (default: embeddings/faiss_index/)"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (default: from .env or gpt-5.2)"
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="LLM provider: openai or ollama (default: from .env or openai)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Get LLM settings from environment or args
    llm_model = args.llm_model or os.getenv("LLM_MODEL", "gpt-5.2")
    llm_provider = args.llm_provider or os.getenv("LLM_PROVIDER", "openai")
    
    # Initialize chatbot
    chatbot = SupermicroChatbot(
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        llm_model=llm_model,
        llm_provider=llm_provider,
        top_k=args.top_k
    )
    
    if args.interactive:
        chatbot.interactive_mode()
    elif args.query:
        result = chatbot.answer(args.query)
        print("\n" + "=" * 80)
        print("Answer:")
        print("=" * 80)
        print(result["answer"])
        if result["sources"]:
            print("\n" + "=" * 80)
            print("Sources:")
            print("=" * 80)
            for source in result["sources"]:
                print(f"  - {source}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
