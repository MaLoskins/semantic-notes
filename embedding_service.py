from __future__ import annotations

# --- Ensure loky silence at import time for any downstream libs (failsafe) ---
import os as _os
import warnings as _warnings
if "LOKY_MAX_CPU_COUNT" not in _os.environ:
    try:
        _os.environ["LOKY_MAX_CPU_COUNT"] = str(_os.cpu_count() or 1)
    except Exception:
        _os.environ["LOKY_MAX_CPU_COUNT"] = "1"
_warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)
# ------------------------------------------------------------------------------

import os
import hashlib
from typing import List, Optional, Dict, Any
import numpy as np
from cachetools import LRUCache
from threading import Lock

# We import lazily to keep import time fast if a different backend is used
_SentenceTransformer = None  # type: ignore

def _lazy_import_sentence_transformers():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _SentenceTransformer = SentenceTransformer
    return _SentenceTransformer

# Optional OpenAI backend (kept lazy and optional)
def _lazy_import_openai_client():
    from openai import OpenAI  # type: ignore
    return OpenAI

def _norm_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

class EmbeddingService:
    _instance: "EmbeddingService" = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = EmbeddingService()
            return cls._instance

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: str = "sentence-transformers",
        device: Optional[str] = None,
        cache_size: int = 10000,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self.backend = backend
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(batch_size)))
        self.normalize = bool(os.getenv("EMBEDDING_NORMALIZE", str(normalize)).lower() in ("1", "true", "yes"))
        self._cache = LRUCache(maxsize=int(os.getenv("EMBEDDING_CACHE_SIZE", str(cache_size))))
        self._model = None

    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "cache_len": len(self._cache),
            "cache_maxsize": self._cache.maxsize,
            "dimension": (self._get_dim() or 0),
        }

    def _get_model(self):
        if self._model is not None:
            return self._model

        if self.backend == "sentence-transformers":
            ST = _lazy_import_sentence_transformers()
            self._model = ST(self.model_name, device=self.device)  # downloads on first use if needed
            return self._model

        elif self.backend == "openai":
            OpenAIClient = _lazy_import_openai_client()
            self._model = OpenAIClient()
            return self._model

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _get_dim(self) -> Optional[int]:
        if self.backend == "sentence-transformers":
            model = self._get_model()
            try:
                return int(model.get_sentence_embedding_dimension())
            except Exception:
                return None
        elif self.backend == "openai":
            # For OpenAI we infer from a tiny run if unknown
            return None
        return None

    def _cache_key(self, texts: List[str]) -> str:
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode("utf-8", errors="ignore"))
            h.update(b"\x00")
        h.update(self.model_name.encode())
        h.update(self.backend.encode())
        return h.hexdigest()

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._get_dim() or 0), dtype=np.float32)

        key = self._cache_key(texts)
        if key in self._cache:
            arr = self._cache[key]
            return arr

        if self.backend == "sentence-transformers":
            model = self._get_model()
            vectors = model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=False)
            if self.normalize:
                vectors = _norm_rows(vectors.astype(np.float32, copy=False))
            self._cache[key] = vectors
            return vectors

        elif self.backend == "openai":
            client = self._get_model()
            model_name = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            resp = client.embeddings.create(model=model_name, input=texts)  # type: ignore
            vectors = [np.array(item.embedding, dtype=np.float32) for item in resp.data]  # type: ignore
            arr = np.vstack(vectors)
            if self.normalize:
                arr = _norm_rows(arr)
            self._cache[key] = arr
            return arr

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")


# FastAPI-friendly helper
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService.get_instance()