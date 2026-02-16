from __future__ import annotations

import importlib.util as _importlib
import os as _os
from typing import Any, Dict

from fastapi import APIRouter, Depends

from auth import get_current_user
from embedding_service import get_embedding_service
from models import User

router = APIRouter(tags=["health"])


def _has_module(name: str) -> bool:
    return _importlib.find_spec(name) is not None


@router.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/api/stats")
def stats(current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    svc = get_embedding_service()
    info = svc.info()
    info.update({
        "has_faiss": _has_module("faiss") or _has_module("faiss_cpu") or _has_module("faiss_gpu"),
        "has_umap": _has_module("umap"),
        "pid": _os.getpid(),
    })
    return info
