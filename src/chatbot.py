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
                    {"role": "system", "content": "You are a technical assistant specializing in Supermicro products and solutions. Answer questions based on the provided context from Supermicro documentation. If the context doesn't contain the answer, say so. Always cite your sources."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error calling OpenAI API: {e}"
    
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
                        {"role": "system", "content": "You are a technical assistant specializing in Supermicro products and solutions. Answer questions based on the provided context from Supermicro documentation. If the context doesn't contain the answer, say so. Always cite your sources."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
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
    
    def answer(self, question: str) -> dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer, sources, and retrieved chunks
        """
        # Retrieve relevant chunks
        chunks = self.query_processor.retrieve(question, self.top_k)
        
        if not chunks:
            return {
                "answer": "No relevant information found in the documentation.",
                "sources": [],
                "chunks": []
            }
        
        # Format context
        context = self.query_processor.format_context(chunks)
        
        # Build prompt
        prompt = f"""Context from Supermicro documentation:

{context}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so. Always cite the source documents."""
        
        # Get LLM response
        answer = get_llm_response(prompt, self.llm_model, self.llm_provider)
        
        # Extract unique sources
        sources = list(set(chunk["source_file"] for chunk in chunks))
        
        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks
        }
    
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
