# backend/main.py
from __future__ import annotations

import os as _os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from embedding_service import get_embedding_service

from routes.health import router as health_router
from routes.auth import router as auth_router
from routes.notes import router as notes_router
from routes.embeddings import router as embeddings_router
from routes.graph import router as graph_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables
    init_db()

    # Preload model + cache a tiny warmup embedding to avoid first-call latency
    svc = get_embedding_service()
    try:
        svc.encode(["warmup"])
    except Exception as e:
        print("[startup] warmup failed:", e)
    yield
    # (optional) shutdown hooks


app = FastAPI(title="Semantic Embedding Graph Engine", version="1.2.1", lifespan=lifespan)

# Diagnostic logging
print(f"[STARTUP] FRONTEND_ORIGIN: {settings.FRONTEND_ORIGIN}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Register Routers ----------

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(notes_router)
app.include_router(embeddings_router)
app.include_router(graph_router)

# ---------- Local Dev Entry ----------

if __name__ == "__main__":
    import uvicorn
    port = int(_os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
