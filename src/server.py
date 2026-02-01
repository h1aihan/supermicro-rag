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


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/chat")
def chat(req: ChatRequest):
    bot = get_chatbot()
    result = bot.answer(req.message)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }

