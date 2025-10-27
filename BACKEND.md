# CODEBASE EXTRACTION

**Source Directory:** `C:\Users\XH673HG\OneDrive - EY\Desktop\9-APPLICATIONS\semantic-notes`
**Generated:** 2025-10-27 13:11:16
**Total Files:** 12

---

## Directory Structure

```
semantic-notes/
├── .env
├── .env.example
├── .gitignore
├── auth.py
├── database.py
├── db_service.py
├── docker-compose.yml
├── embedding_service.py
├── graph_service.py
├── init-db.sql
├── main.py
├── models.py
```

## Code Files


### 📄 .env

```
# Database Configuration
POSTGRES_USER=semantic_user
POSTGRES_PASSWORD=semantic_dev_password_2024
POSTGRES_DB=semantic_notes
DATABASE_URL=postgresql://semantic_user:semantic_dev_password_2024@localhost:5432/semantic_notes

# JWT Configuration
JWT_SECRET_KEY=your_jwt_secret_key_minimum_32_characters_for_security
JWT_ALGORITHM=HS256
JWT_EXPIRE_DAYS=7

# Embedding Service (existing)
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=64
FRONTEND_ORIGIN=http://localhost:3000
```


### 📄 .env.example

```
# Database Configuration
POSTGRES_USER=your_db_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=your_db_name
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# JWT Configuration
JWT_SECRET_KEY=your_secret_key_here_min_32_chars
JWT_ALGORITHM=HS256
JWT_EXPIRE_DAYS=7

# Embedding Service
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=64
FRONTEND_ORIGIN=
```


### 📄 .gitignore

```
node_modules
__pycache__
.env
postgres-data
.venv
```


### 📄 auth.py

```python
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import os

from database import get_db
from models import User

security = HTTPBearer()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise RuntimeError("Critical configuration error: JWT_SECRET_KEY environment variable is not set.")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))


def hash_password(password: str) -> str:
    """Hash password securely using bcrypt (max 72 bytes)."""
    password_bytes = password.encode("utf-8")[:72]
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password using bcrypt check."""
    password_bytes = plain_password.encode("utf-8")[:72]
    return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))


def create_access_token(user_id: int, username: str) -> str:
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)
    payload = {"sub": str(user_id), "username": username, "exp": expire}
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return {"user_id": int(payload.get("sub")), "username": payload.get("username")}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("user_id")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user
```


### 📄 database.py

```python
import os
import time
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in the environment (.env)")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=15,
    max_overflow=30,
    future=True,
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

log = logging.getLogger("database")
logging.basicConfig(level=logging.INFO)

def _wait_for_db(max_attempts: int = 30, delay_seconds: float = 1.0) -> None:
    """Poll the DB until it accepts TCP connections."""
    attempt = 0
    last_err: Exception | None = None
    while attempt < max_attempts:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("Database connection OK.")
            return
        except OperationalError as e:
            last_err = e
            attempt += 1
            time.sleep(delay_seconds)
    # Exhausted retries
    raise OperationalError(f"Could not connect to database after {max_attempts} attempts", params=None, orig=last_err)

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

# Initialize database tables
def init_db():
    try:
        _wait_for_db()
        Base.metadata.create_all(bind=engine)
        log.info("Database initialized and tables created successfully.")
    except Exception as e:
        log.error(f"Database initialization failed: {e}")
        raise

```


### 📄 db_service.py

```python
from sqlalchemy.orm import Session
from datetime import datetime
from models import User, Note, Embedding
from auth import hash_password, verify_password


# -------------------- USER OPERATIONS -------------------- #

def create_user(db: Session, username: str, password: str, email: str = None) -> User:
    try:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            raise ValueError("User already exists")
        user = User(
            username=username,
            password_hash=hash_password(password),
            email=email,
            created_at=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise e


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.password_hash):
        return None
    update_last_login(db, user.id)
    return user


def update_last_login(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.last_login = datetime.utcnow()
        db.commit()


# -------------------- NOTE OPERATIONS -------------------- #

def get_user_notes(db: Session, user_id: int, include_deleted: bool = False) -> list[Note]:
    query = db.query(Note).filter(Note.user_id == user_id)
    if not include_deleted:
        query = query.filter(Note.is_deleted == False)
    return query.order_by(Note.created_at.desc()).all()


def get_note_by_id(db: Session, note_id: int, user_id: int) -> Note | None:
    return db.query(Note).filter(Note.id == note_id, Note.user_id == user_id).first()


def create_note(db: Session, user_id: int, title: str, content: str, tags: str = "") -> Note:
    try:
        note = Note(
            user_id=user_id,
            title=title,
            content=content,
            tags=tags,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(note)
        db.commit()
        db.refresh(note)
        return note
    except Exception as e:
        db.rollback()
        raise e


def update_note(db: Session, note_id: int, user_id: int, title: str, content: str, tags: str) -> Note | None:
    try:
        note = get_note_by_id(db, note_id, user_id)
        if not note or note.is_deleted:
            return None
        note.title = title
        note.content = content
        note.tags = tags
        note.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(note)
        return note
    except Exception as e:
        db.rollback()
        raise e


def soft_delete_note(db: Session, note_id: int, user_id: int) -> bool:
    note = get_note_by_id(db, note_id, user_id)
    if not note or note.is_deleted:
        return False
    note.is_deleted = True
    note.deleted_at = datetime.utcnow()
    db.commit()
    return True


def restore_note(db: Session, note_id: int, user_id: int) -> bool:
    try:
        note = get_note_by_id(db, note_id, user_id)
        if not note or not note.is_deleted:
            return False
        note.is_deleted = False
        note.deleted_at = None
        note.updated_at = datetime.utcnow()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e


def permanent_delete_note(db: Session, note_id: int, user_id: int) -> bool:
    note = get_note_by_id(db, note_id, user_id)
    if not note:
        return False
    db.delete(note)
    db.commit()
    return True


def get_user_trash(db: Session, user_id: int) -> list[Note]:
    return db.query(Note).filter(Note.user_id == user_id, Note.is_deleted == True).all()


def empty_trash(db: Session, user_id: int) -> int:
    deleted_notes = db.query(Note).filter(Note.user_id == user_id, Note.is_deleted == True).all()
    count = len(deleted_notes)
    for note in deleted_notes:
        db.delete(note)
    db.commit()
    return count


# -------------------- EMBEDDING OPERATIONS -------------------- #

def get_note_embedding(db: Session, note_id: int) -> Embedding | None:
    return db.query(Embedding).filter(Embedding.note_id == note_id).first()


def save_note_embedding(db: Session, note_id: int, content_hash: str, embedding_vector: list[float], model_name: str) -> Embedding:
    existing = get_note_embedding(db, note_id)
    if existing:
        existing.content_hash = content_hash
        existing.embedding_vector = embedding_vector
        existing.model_name = model_name
        existing.created_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing
    embedding = Embedding(
        note_id=note_id,
        content_hash=content_hash,
        embedding_vector=embedding_vector,
        model_name=model_name,
        created_at=datetime.utcnow()
    )
    db.add(embedding)
    db.commit()
    db.refresh(embedding)
    return embedding


def batch_get_embeddings(db: Session, note_ids: list[int]) -> dict[int, Embedding]:
    embeddings = db.query(Embedding).filter(Embedding.note_id.in_(note_ids)).all()
    return {e.note_id: e for e in embeddings}


# -------------------- BULK IMPORT -------------------- #

def bulk_create_notes(db: Session, user_id: int, notes_list: list[dict]) -> dict[str, int]:
    """
    Bulk create notes for a user.
    Returns mapping of old_id -> new_id for embedding migration.

    Args:
        db: Database session
        user_id: User ID to associate notes with
        notes_list: List of note dicts with title, content, tags, created_at, updated_at

    Returns:
        Dictionary mapping old note IDs to new database IDs
    """
    from datetime import datetime
    from models import Note

    id_mapping = {}

    try:
        for note_data in notes_list:
            old_id = note_data.get('id')
            created_at = note_data.get('createdAt') or note_data.get('created_at')
            updated_at = note_data.get('updatedAt') or note_data.get('updated_at')

            from datetime import timezone

            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            if created_at and created_at.tzinfo is not None:
                created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)

            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            if updated_at and updated_at.tzinfo is not None:
                updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)

            new_note = Note(
                user_id=user_id,
                title=note_data.get('title', 'Untitled'),
                content=note_data.get('content', ''),
                tags=note_data.get('tags', ''),
                created_at=created_at or datetime.utcnow(),
                updated_at=updated_at or datetime.utcnow(),
                is_deleted=note_data.get('is_deleted', False),
                deleted_at=note_data.get('deleted_at')
            )

            db.add(new_note)
            db.flush()
            if old_id:
                id_mapping[str(old_id)] = new_note.id

        db.commit()
        return id_mapping

    except Exception as e:
        db.rollback()
        raise e
```


### 📄 docker-compose.yml

```yaml
services:
  postgres:
    image: postgres:15-alpine
    container_name: semantic-notes-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
      TZ: Australia/Melbourne
    command: >
      postgres -c listen_addresses='*' -c port=5432 -c max_connections=200
    ports:
      - "5432:5432"
    volumes:
      - ./postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -h 127.0.0.1 -p 5432 -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 10
    networks:
      - semantic-notes-network

networks:
  semantic-notes-network:
    driver: bridge

```


### 📄 embedding_service.py

```python
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
```


### 📄 graph_service.py

```python
# backend/graph_service.py
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
import numpy as np

# Try FAISS (optional). Fallback is pure NumPy.
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering

# UMAP is optional; fall back to PCA if unavailable
try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


def _pairwise_cosine_from_normalized(X: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix for already L2-normalized embeddings."""
    # Cosine equals dot product when rows are unit vectors
    return X @ X.T


def _knn_indices_cosine(X: np.ndarray, k: int) -> np.ndarray:
    """Return for each row the indices of top-k nearest neighbors by cosine similarity (excluding self)."""
    n = X.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.int32)

    # Ensure k is valid
    k = max(1, min(k, max(1, n - 1)))

    if _HAS_FAISS:
        d = X.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(X.astype(np.float32, copy=False))
        sims, idx = index.search(X.astype(np.float32, copy=False), k + 1)  # include self
        return idx[:, 1:]  # drop self
    else:
        # Pure NumPy fallback
        S = _pairwise_cosine_from_normalized(X)
        np.fill_diagonal(S, -np.inf)  # exclude self
        idx = np.argpartition(-S, kth=np.minimum(k, n - 1) - 1, axis=1)[:, :k]
        # Sort the top-k by similarity descending for nicer ordering
        row_indices = np.arange(n)[:, None]
        row_sims = S[row_indices, idx]
        order = np.argsort(-row_sims, axis=1)
        idx_sorted = idx[row_indices, order]
        return idx_sorted


def reduce_dimensions(
    X: np.ndarray,
    method: Literal["none", "pca", "umap", "tsne"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    if X.size == 0 or n_components <= 0 or method == "none":
        return None

    if method == "pca" or (method == "umap" and not _HAS_UMAP):
        # PCA is fast and deterministic
        n_components = min(n_components, X.shape[1])
        coords = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
        return coords.astype(np.float32, copy=False)

    if method == "umap" and _HAS_UMAP:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        coords = reducer.fit_transform(X)
        return coords.astype(np.float32, copy=False)

    if method == "tsne":
        # t-SNE is slower; use a safe perplexity
        perplexity = max(5, min(30, X.shape[0] // 3))
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        coords = tsne.fit_transform(X)
        return coords.astype(np.float32, copy=False)

    # Fallback
    return None


def _safe_cluster_count(requested: Optional[int], n_samples: int) -> int:
    """Clamp cluster count to valid bounds for sklearn (≥2 and ≤ n_samples)."""
    if n_samples < 2:
        return 0  # nothing to cluster
    k = 0
    if requested and requested >= 2:
        k = requested
    else:
        # Heuristic default (~sqrt(n/2), bounded [2, 20])
        k = max(2, min(20, int(np.sqrt(max(2, n_samples / 2)))))

    # Absolute safety clamps
    k = min(k, n_samples)
    k = max(2, k) if n_samples >= 2 else 0
    return k


def cluster_embeddings(
    X: np.ndarray,
    method: Literal["none", "kmeans", "agglomerative"] = "none",
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    n = X.shape[0]
    if method == "none" or X.size == 0 or n < 2:
        return None

    k = _safe_cluster_count(n_clusters, n)
    if k < 2:
        return None  # not enough samples to form ≥2 clusters

    try:
        if method == "kmeans":
            model = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = model.fit_predict(X)
            return labels.astype(np.int32, copy=False)

        if method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(X)
            return labels.astype(np.int32, copy=False)

        return None
    except ValueError:
        # As a last resort (e.g., degenerate data), fall back to a single cluster or None
        return None


def build_similarity_graph(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    include_embeddings: bool = False,
    dr_method: Literal["none", "pca", "umap", "tsne"] = "pca",
    n_components: int = 2,
    cluster_method: Literal["none", "kmeans", "agglomerative"] = "none",
    n_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON-friendly graph dict from embeddings and options."""
    X = embeddings.astype(np.float32, copy=False)
    n = X.shape[0]
    if n == 0:
        return {"nodes": [], "edges": []}

    # Dimensionality reduction (for x/y layout)
    coords = reduce_dimensions(X, method=dr_method, n_components=n_components)

    # Clustering (robust)
    clusters = cluster_embeddings(X, method=cluster_method, n_clusters=n_clusters)

    # Nodes
    nodes: List[Dict[str, Any]] = []
    for i in range(n):
        node: Dict[str, Any] = {
            "id": str(i),
            "label": labels[i] if labels and i < len(labels) else f"Doc {i}",
        }
        if include_embeddings:
            node["embedding"] = X[i].tolist()
        if coords is not None and coords.shape[1] >= 2:
            node["x"], node["y"] = float(coords[i, 0]), float(coords[i, 1])
        if clusters is not None:
            node["cluster"] = int(clusters[i])
        nodes.append(node)

    # Edges
    edges: List[Dict[str, Any]] = []
    if threshold is not None:
        S = _pairwise_cosine_from_normalized(X)
        for i in range(n):
            for j in range(i + 1, n):
                w = float(S[i, j])
                if w >= float(threshold):
                    edges.append({"source": str(i), "target": str(j), "weight": round(w, 6)})
    else:
        k = int(top_k) if top_k is not None else 5
        k = max(1, min(k, max(1, n - 1)))
        idx = _knn_indices_cosine(X, k=k)
        S = _pairwise_cosine_from_normalized(X)
        seen = set()
        for i in range(n):
            for j in idx[i]:
                a, b = (i, int(j)) if i < j else (int(j), i)
                if a == b:
                    continue
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                w = float(S[a, b])
                edges.append({"source": str(a), "target": str(b), "weight": round(w, 6)})

    return {"nodes": nodes, "edges": edges}

```


### 📄 init-db.sql

```sql
-- Ensure the correct database exists
CREATE DATABASE semantic_notes;

\c semantic_notes
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- Notes table
CREATE TABLE IF NOT EXISTS notes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_notes_user_id ON notes(user_id);
CREATE INDEX IF NOT EXISTS idx_notes_user_deleted ON notes(user_id, is_deleted);

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    note_id INTEGER UNIQUE NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    content_hash VARCHAR(64) NOT NULL,
    embedding_vector FLOAT4[] NOT NULL,
    model_name VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embeddings_note_id ON embeddings(note_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(content_hash);
```


### 📄 main.py

```python
# backend/main.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
import importlib.util as _importlib
import os as _os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import init_db, get_db
from models import User
import db_service
from embedding_service import get_embedding_service
from auth import create_access_token, get_current_user

# ---------- Pydantic Schemas ----------

class EmbedRequest(BaseModel):
    documents: List[str] = Field(..., description="List of texts to embed")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


class GraphRequest(BaseModel):
    documents: List[str] = Field(..., description="List of texts to graph")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum cosine similarity for an edge")
    top_k: Optional[int] = Field(5, ge=1, description="kNN edges per node if threshold not provided")
    include_embeddings: bool = Field(False, description="Include embedding vectors in node payload")
    dr_method: Literal["none", "pca", "umap", "tsne"] = Field("pca", description="Dimensionality reduction for x/y")
    n_components: int = Field(2, ge=2, le=10, description="Output dimensions for DR")
    cluster: Literal["none", "kmeans", "agglomerative"] = Field("none", description="Clustering algorithm")
    n_clusters: Optional[int] = Field(None, ge=2, description="Number of clusters (optional)")
    labels: Optional[List[str]] = Field(None, description="Optional labels per document")


class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

# ---------- App Setup ----------

def _has_module(name: str) -> bool:
    return _importlib.find_spec(name) is not None


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

from dotenv import load_dotenv
load_dotenv()

# Diagnostic logging
frontend_origin = _os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
print(f"[STARTUP] FRONTEND_ORIGIN: {frontend_origin}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Routes ----------

@app.post("/api/notes/import")
async def import_notes(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Bulk import notes from localStorage.
    Request body: { "notes": [...], "trash": [...] }
    Returns: { "imported": count, "id_mapping": {old_id: new_id} }
    """
    try:
        from pydantic import BaseModel, ValidationError, conlist, ConfigDict
        from datetime import datetime

        class ImportNoteSchema(BaseModel):
            # Accept plain notes from local export (id is created server-side)
            title: str
            content: str
            tags: Optional[str] = ""
            is_deleted: Optional[bool] = False
            # Accept optional timestamps commonly present in exports
            createdAt: Optional[datetime] = None
            updatedAt: Optional[datetime] = None
            deletedAt: Optional[datetime] = None
            deleted_at: Optional[datetime] = None

            model_config = ConfigDict(extra="ignore")  # ignore other client-only fields

        class ImportRequest(BaseModel):
            notes: conlist(ImportNoteSchema, min_length=0) = []
            trash: conlist(ImportNoteSchema, min_length=0) = []

        try:
            raw_data = await request.json()
            data = ImportRequest(**raw_data)
        except ValidationError as e:
            # validation errors bubble as 400
            raise HTTPException(status_code=400, detail=e.errors())  # <- current behavior :contentReference[oaicite:5]{index=5}

        notes_to_import: List[Dict[str, Any]] = []
        for note in data.notes:
            d = note.model_dump(by_alias=True)
            # map timestamps to server-friendly field names (snake_case on server)
            created = d.get("createdAt")
            updated = d.get("updatedAt")
            item = {
                "title": d.get("title", ""),
                "content": d.get("content", ""),
                "tags": d.get("tags", "") or "",
                "is_deleted": False,
                "deleted_at": None,
                "created_at": created,
                "updated_at": updated or created,
            }
            notes_to_import.append(item)

        trash_to_import: List[Dict[str, Any]] = []
        for trash_note in data.trash:
            d = trash_note.model_dump(by_alias=True)
            deleted_at = d.get("deletedAt") or d.get("deleted_at")
            created = d.get("createdAt")
            updated = d.get("updatedAt") or created
            item = {
                "title": d.get("title", ""),
                "content": d.get("content", ""),
                "tags": d.get("tags", "") or "",
                "is_deleted": True,
                "deleted_at": deleted_at,
                "created_at": created,
                "updated_at": updated,
            }
            trash_to_import.append(item)

        id_mapping: Dict[int, int] = {}
        if notes_to_import:
            id_mapping = db_service.bulk_create_notes(db, current_user.id, notes_to_import)

        if trash_to_import:
            trash_mapping = db_service.bulk_create_notes(db, current_user.id, trash_to_import)
            id_mapping.update(trash_mapping)

        total_imported = len(notes_to_import) + len(trash_to_import)
        return {"imported": total_imported, "id_mapping": id_mapping}

    except HTTPException:
        raise
    except Exception as e:
        # Previously this path produced: "Import failed: 'ImportNoteSchema' object has no attribute 'get'"
        # because we treated Pydantic models like dicts. Fixed by using model_dump().
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")  # prior bug site :contentReference[oaicite:6]{index=6}


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/stats")
def stats() -> Dict[str, Any]:
    svc = get_embedding_service()
    info = svc.info()
    info.update({
        "has_faiss": _has_module("faiss") or _has_module("faiss_cpu") or _has_module("faiss_gpu"),
        "has_umap": _has_module("umap"),
        "pid": _os.getpid(),
    })
    return info


@app.post("/api/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    service = get_embedding_service()
    vecs = service.encode(req.documents)
    return EmbedResponse(embeddings=vecs.tolist())


@app.post("/api/graph", response_model=GraphResponse)
def graph(req: GraphRequest) -> GraphResponse:
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    threshold = req.threshold
    top_k = None if threshold is not None else req.top_k

    service = get_embedding_service()
    X = service.encode(req.documents)

    from graph_service import build_similarity_graph

    graph_dict = build_similarity_graph(
        embeddings=X,
        labels=req.labels,
        threshold=threshold,
        top_k=top_k,
        include_embeddings=req.include_embeddings,
        dr_method=req.dr_method,
        n_components=req.n_components,
        cluster_method=req.cluster,
        n_clusters=req.n_clusters,
    )
    return GraphResponse(**graph_dict)


# ---------- New API Models ----------

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    user_id: int


class NoteRequest(BaseModel):
    title: str
    content: str
    tags: str = ""


from datetime import datetime
from pydantic import ConfigDict

class NoteResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={datetime: lambda v: v.isoformat()}
    )

    id: int
    title: str
    content: str
    tags: str
    created_at: datetime
    updated_at: datetime
    is_deleted: bool


# ---------- Auth Endpoints ----------

@app.post("/api/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    existing = db_service.get_user_by_username(db, req.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    user = db_service.create_user(db, req.username, req.password, req.email)
    token = create_access_token(user.id, user.username)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=user.username,
        user_id=user.id,
    )


@app.post("/api/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db_service.authenticate_user(db, req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.id, user.username)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=user.username,
        user_id=user.id,
    )


@app.get("/api/auth/me", response_model=dict)
def get_me(current_user: User = Depends(get_current_user)):
    return {"id": current_user.id, "username": current_user.username, "email": current_user.email}


# ---------- Note Endpoints ----------

@app.get("/api/notes", response_model=list[NoteResponse])
def list_notes(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    notes = db_service.get_user_notes(db, current_user.id)
    return notes


@app.post("/api/notes", response_model=NoteResponse)
def create_new_note(req: NoteRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    note = db_service.create_note(db, current_user.id, req.title, req.content, req.tags)
    return note


@app.put("/api/notes/{note_id}", response_model=NoteResponse)
def update_existing_note(note_id: int, req: NoteRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    updated_note = db_service.update_note(db, note_id, current_user.id, req.title, req.content, req.tags)
    if not updated_note:
        raise HTTPException(status_code=404, detail="Note not found")
    return updated_note


@app.post("/api/notes/{note_id}/trash")
def trash_note(note_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    success = db_service.soft_delete_note(db, note_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"status": "moved to trash"}


@app.post("/api/notes/{note_id}/restore")
def restore_note_endpoint(note_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    success = db_service.restore_note(db, note_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Note not found or not deleted")
    return {"status": "restored"}


@app.delete("/api/notes/{note_id}")
def permanently_delete_note(note_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    success = db_service.permanent_delete_note(db, note_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"status": "permanently deleted"}


@app.get("/api/trash", response_model=list[NoteResponse])
def get_trash(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db_service.get_user_trash(db, current_user.id)


@app.post("/api/trash/empty")
def empty_trash_endpoint(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    deleted_count = db_service.empty_trash(db, current_user.id)
    return {"deleted_count": deleted_count}


# ---------- Embedding Endpoints ----------

from fastapi import Request

@app.post("/api/embeddings/batch")
async def save_embeddings_batch(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Save multiple embeddings to database.
    Request body: { "embeddings": [{ "note_id": int, "content_hash": str, "embedding": [float...], "model_name": str }] }
    """
    try:
        data = await request.json()
        embeddings_data = data.get("embeddings", [])
        
        if not embeddings_data:
            raise HTTPException(status_code=400, detail="No embeddings provided")
        
        note_ids = [e["note_id"] for e in embeddings_data]
        for note_id in note_ids:
            note = db_service.get_note_by_id(db, note_id, current_user.id)
            if not note:
                raise HTTPException(status_code=403, detail=f"Note {note_id} not found or access denied")
        
        saved_count = 0
        for emb_data in embeddings_data:
            try:
                db_service.save_note_embedding(
                    db,
                    note_id=emb_data["note_id"],
                    content_hash=emb_data["content_hash"],
                    embedding_vector=emb_data["embedding"],
                    model_name=emb_data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
                )
                saved_count += 1
            except Exception as e:
                print(f"Failed to save embedding for note {emb_data['note_id']}: {e}")
        
        return {"saved": saved_count, "total": len(embeddings_data)}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/embeddings")
async def get_embeddings(
    note_ids: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Fetch embeddings for specific note IDs.
    Query param: note_ids=1,2,3
    Returns: { "embeddings": { "note_id": { "content_hash": str, "embedding": [float...], "model_name": str } } }
    """
    try:
        if not note_ids:
            return {"embeddings": {}}
        
        note_id_list = [int(nid.strip()) for nid in note_ids.split(",")]
        
        for note_id in note_id_list:
            note = db_service.get_note_by_id(db, note_id, current_user.id)
            if not note:
                raise HTTPException(status_code=403, detail=f"Note {note_id} not found or access denied")
        
        embeddings_dict = db_service.batch_get_embeddings(db, note_id_list)
        
        result = {}
        for note_id, embedding in embeddings_dict.items():
            result[str(note_id)] = {
                "content_hash": embedding.content_hash,
                "embedding": embedding.embedding_vector,
                "model_name": embedding.model_name
            }
        
        return {"embeddings": result}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Local Dev Entry ----------

if __name__ == "__main__":
    import uvicorn
    port = int(_os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

```


### 📄 models.py

```python
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, ARRAY, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime)

    notes = relationship("Note", back_populates="user", cascade="all, delete-orphan")


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="notes")
    embedding = relationship("Embedding", back_populates="note", uselist=False, cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    note_id = Column(Integer, ForeignKey("notes.id", ondelete="CASCADE"), unique=True, nullable=False)
    content_hash = Column(String(64), nullable=False)
    embedding_vector = Column(ARRAY(Float), nullable=False)
    model_name = Column(String(100), default="sentence-transformers/all-MiniLM-L6-v2")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    note = relationship("Note", back_populates="embedding")
```
