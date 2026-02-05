"""
Minimal HTTP server wrapper for the existing CLI chatbot.

Routes:
- GET  /health     : health check
- GET  /           : lightweight chat UI
- POST /api/chat   : JSON chat endpoint
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.chatbot import SupermicroChatbot

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = REPO_ROOT / "static"

_chatbot: Optional["SupermicroChatbot"] = None


def _csv_env(name: str) -> List[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def get_chatbot() -> SupermicroChatbot:
    global _chatbot
    if _chatbot is not None:
        return _chatbot

    # Import lazily so the web server can start quickly (and health checks pass)
    # even if FAISS / sentence-transformers are slow to import or misconfigured.
    from src.chatbot import SupermicroChatbot

    index_dir = os.getenv("INDEX_DIR", "embeddings/faiss_index/")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model = os.getenv("LLM_MODEL", "gpt-5.2")
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    top_k = int(os.getenv("TOP_K", "5"))

    _chatbot = SupermicroChatbot(
        index_dir=index_dir,
        embedding_model=embedding_model,
        llm_model=llm_model,
        llm_provider=llm_provider,
        top_k=top_k,
    )
    return _chatbot


app = FastAPI(title="Supermicro RAG", version="0.1.0")

# Emit a startup log line ASAP (helps debug App Runner "no logs" situations).
@app.on_event("startup")
def _startup_log() -> None:
    # Keep this very lightweight: no FAISS/model loading here.
    print(
        "[startup] server booted",
        {
            "python": sys.version.split()[0],
            "cwd": os.getcwd(),
            "static_index_exists": (STATIC_DIR / "index.html").exists(),
        },
        flush=True,
    )

# Optional CORS (only needed if hosting UI separately)
cors_origins = _csv_env("CORS_ORIGINS")
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None  # Previous conversation turns


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/chat")
def chat(req: ChatRequest):
    bot = get_chatbot()
    
    # Format conversation history if provided
    conversation_context = ""
    print(f"[DEBUG] Received history: {len(req.history) if req.history else 0} messages")
    if req.history:
        print(f"[DEBUG] History content: {[h.role + ': ' + h.content[:50] + '...' for h in req.history]}")
        history_parts = []
        for msg in req.history[-6:]:  # Keep last 6 messages (3 turns) for context
            role = "User" if msg.role == "user" else "Assistant"
            history_parts.append(f"{role}: {msg.content}")
        if history_parts:
            conversation_context = "\n".join(history_parts) + "\n\n"
    
    # Combine history with current message for context-aware retrieval
    full_query = req.message
    if conversation_context:
        # Add conversation context to help with follow-up questions
        full_query = f"{conversation_context}Current question: {req.message}"
    
    result = bot.answer(req.message, conversation_context=conversation_context)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }

