# CODEBASE EXTRACTION

**Source Directory:** `C:\Users\matth\Desktop\8-SANDBOX\notes\semantic-notes`
**Generated:** 2025-10-22 14:10:59
**Total Files:** 42

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
├── frontend/
│   ├── .env
│   ├── .env.example
│   ├── index.html
│   ├── src/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── AuthGuard.jsx
│   │   │   ├── ConfirmDialog.jsx
│   │   │   ├── ExportGraphModal.jsx
│   │   │   ├── GraphControlsPanel.jsx
│   │   │   ├── GraphVisualization.jsx
│   │   │   ├── ImportConfirmModal.jsx
│   │   │   ├── ImportLocalNotesModal.jsx
│   │   │   ├── LoginForm.css
│   │   │   ├── LoginForm.jsx
│   │   │   ├── MarkdownCheatsheet.jsx
│   │   │   ├── MarkdownPreview.jsx
│   │   │   ├── NoteEditor.jsx
│   │   │   ├── NotesList.jsx
│   │   │   ├── SimilarNotesModal.jsx
│   │   │   ├── ToastNotification.jsx
│   │   │   ├── TrashView.jsx
│   │   │   └── UnsavedChangesDialog.jsx
│   │   ├── contexts/
│   │   │   └── AuthContext.jsx
│   │   ├── hooks/
│   │   │   └── useNotes.js
│   │   ├── main.jsx
│   │   ├── services/
│   │   │   ├── api.js
│   │   │   └── dbApi.js
│   │   └── utils/
│   │       └── graphExport.js
│   └── vite.config.js
├── graph_service.py
├── init-db.sql
├── main.py
├── models.py
└── test_client.py
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
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=15, max_overflow=30)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

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
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized and tables created successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
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
version: '3.9'

services:
  postgres:
    image: postgres:15-alpine
    container_name: semantic-notes-db
    environment:
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=${POSTGRES_DB}
    ports:
      - "5432:5432"
    volumes:
      - ./postgres-data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 5s
      retries: 5
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


### 📄 frontend\.env

```
VITE_API_BASE_URL=http://localhost:8000
```


### 📄 frontend\.env.example

```
# Example environment variables for the frontend
# Replace the base URL below with your backend API endpoint
VITE_API_BASE_URL=
```


### 📄 frontend\index.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Semantic Notes</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```


### 📄 frontend\src\App.css

```css
/* Professional Dark Theme for Semantic Notes */

:root {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --bg-hover: #475569;
  
  --text-primary: #f1f5f9;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  
  --accent-primary: #3b82f6;
  --accent-primary-hover: #2563eb;
  --accent-success: #10b981;
  --accent-danger: #ef4444;
  --accent-danger-hover: #dc2626;
  
  --border: #334155;
  --border-focus: #3b82f6;
  
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.4);
  
  --radius-sm: 0.25rem;
  --radius-md: 0.375rem;
  --radius-lg: 0.5rem;
  
  --transition: all 0.2s ease;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-primary);
  color: var(--text-secondary);
  height: 100vh;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
}

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* Header */
.app-header {
  background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
  padding: 1rem 2rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: var(--shadow-md);
  z-index: 100;
}

.app-header h1 {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.025em;
}

.header-actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

.search-bar {
  position: relative;
}

.search-input {
  padding: 0.5rem 1rem;
  padding-left: 2.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  width: 280px;
  font-size: 0.875rem;
  transition: var(--transition);
}

.search-input:focus {
  outline: none;
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.search-input::placeholder {
  color: var(--text-dim);
}

.search-icon {
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-dim);
  pointer-events: none;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
  padding: 0.375rem 0.75rem;
  border-radius: var(--radius-md);
  background: rgba(0, 0, 0, 0.2);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent-success);
}

.status-indicator.disconnected {
  background: var(--accent-danger);
}

/* Main Layout */
.main-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 380px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Buttons */
.btn {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition);
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  white-space: nowrap;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--accent-primary);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: var(--accent-primary-hover);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--bg-hover);
}

.btn-danger {
  background: var(--accent-danger);
  color: white;
}

.btn-danger:hover:not(:disabled) {
  background: var(--accent-danger-hover);
}

.btn-icon {
  padding: 0.5rem;
  min-width: 2rem;
}

/* Note Editor */
.note-editor {
  padding: 1.5rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  overflow-y: auto;
}

.editor-header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}

.editor-header h2 {
  color: var(--text-primary);
  font-size: 1.125rem;
  font-weight: 600;
}

.unsaved-indicator {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  color: var(--text-muted);
  font-size: 0.8125rem;
}

.unsaved-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent-danger);
  box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
}

.editor-form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  flex: 1;
}

.form-group {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-muted);
}

.form-input {
  width: 100%;
  padding: 0.75rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: 0.875rem;
  transition: var(--transition);
}

.form-input:focus {
  outline: none;
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-input::placeholder {
  color: var(--text-dim);
}

.form-textarea {
  min-height: 300px;
  resize: vertical;
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  line-height: 1.6;
}

.char-count {
  font-size: 0.75rem;
  color: var(--text-dim);
  text-align: right;
}

.editor-actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
}

.keyboard-hint {
  margin-left: auto;
  font-size: 0.75rem;
  color: var(--text-dim);
}

/* Notes List */
.notes-list {
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}

.list-header h3 {
  color: var(--text-primary);
  font-size: 1rem;
  font-weight: 600;
}

.list-controls {
  display: flex;
  gap: 0.5rem;
}

.form-select {
  padding: 0.375rem 0.75rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 0.75rem;
  cursor: pointer;
  transition: var(--transition);
}

.form-select:hover {
  background: var(--bg-tertiary);
}

.form-select:focus {
  outline: none;
  border-color: var(--border-focus);
}

.empty-message {
  text-align: center;
  padding: 3rem 1rem;
  color: var(--text-dim);
}

.empty-message-title {
  font-size: 1rem;
  font-weight: 500;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}

.empty-message-hint {
  font-size: 0.875rem;
  color: var(--text-dim);
}

.notes-items {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.note-item {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1rem;
  cursor: pointer;
  transition: var(--transition);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}

.note-item:hover {
  background: var(--bg-tertiary);
  transform: translateX(4px);
  box-shadow: var(--shadow-md);
}

.note-item.selected {
  border-color: var(--accent-primary);
  background: var(--bg-tertiary);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
}

.note-item-content {
  flex: 1;
  min-width: 0;
}

.note-item-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.5rem;
  gap: 0.75rem;
}

.note-item-title {
  color: var(--text-primary);
  font-size: 0.9375rem;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.note-date {
  font-size: 0.75rem;
  color: var(--text-dim);
  white-space: nowrap;
}

.note-preview {
  color: var(--text-muted);
  font-size: 0.8125rem;
  line-height: 1.5;
  margin-bottom: 0.5rem;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.note-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.375rem;
  margin-top: 0.5rem;
}

.tag {
  display: inline-block;
  padding: 0.125rem 0.5rem;
  background: rgba(59, 130, 246, 0.15);
  color: var(--accent-primary);
  border-radius: var(--radius-sm);
  font-size: 0.6875rem;
  font-weight: 500;
  border: 1px solid rgba(59, 130, 246, 0.2);
}

.note-item-actions {
  display: flex;
  gap: 0.25rem;
  opacity: 0;
  transition: opacity 0.2s;
}

.note-item:hover .note-item-actions {
  opacity: 1;
}

.action-btn {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  transition: var(--transition);
  color: var(--text-muted);
  font-weight: 500;
}

.action-btn:hover {
  background: var(--bg-hover);
  color: var(--text-secondary);
}

.action-btn.delete {
  color: var(--accent-danger);
}

.action-btn.delete:hover {
  background: rgba(239, 68, 68, 0.1);
  border-color: var(--accent-danger);
}

/* Graph Container */
.graph-container {
  flex: 1;
  background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary));
  position: relative;
  overflow: hidden;
}

.graph-visualization {
  width: 100%;
  height: 100%;
  position: relative;
}

.graph-controls {
  position: absolute;
  top: 1rem;
  right: 1rem;
  display: flex;
  gap: 0.5rem;
  z-index: 10;
}

.control-btn {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-secondary);
  transition: var(--transition);
  box-shadow: var(--shadow-md);
}

.control-btn:hover {
  background: var(--bg-tertiary);
  transform: scale(1.05);
}

.node-tooltip {
  position: absolute;
  top: 1rem;
  left: 1rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  color: var(--text-secondary);
  box-shadow: var(--shadow-lg);
  pointer-events: none;
  z-index: 100;
  max-width: 300px;
}

.tooltip-label {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.tooltip-meta {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.selected-note-preview {
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  padding: 1rem;
  border-radius: var(--radius-lg);
  max-width: 350px;
  max-height: 250px;
  overflow-y: auto;
  box-shadow: var(--shadow-lg);
  z-index: 10;
}

.selected-note-preview h3 {
  color: var(--text-primary);
  margin-bottom: 0.5rem;
  font-size: 0.9375rem;
  font-weight: 600;
}

.selected-note-preview p {
  color: var(--text-muted);
  font-size: 0.8125rem;
  line-height: 1.6;
}

/* Loading and Error States */
.loading,
.empty-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  color: var(--text-dim);
  padding: 2rem;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.error-banner {
  background: rgba(239, 68, 68, 0.1);
  border-bottom: 1px solid var(--accent-danger);
  color: var(--accent-danger);
  padding: 0.75rem 2rem;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.stats-bar {
  padding: 1rem 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.75rem;
  color: var(--text-dim);
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

/* Scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
  background: var(--bg-tertiary);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--bg-hover);
}

/* Animations */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.fade-in {
  animation: fadeIn 0.3s ease-out;
}

/* Responsive */
@media (max-width: 768px) {
  .sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--border);
  }
  
  .main-content {
    flex-direction: column;
  }
  
  .graph-container {
    min-height: 400px;
  }
  
  .selected-note-preview {
    max-width: calc(100% - 2rem);
  }
  
  .header-actions {
    gap: 0.5rem;
  }
  
  .search-input {
    width: 200px;
  }
}
/* Import Modal and Success Banner */
.success-banner {
  background: rgba(16, 185, 129, 0.12);
  border-bottom: 1px solid var(--accent-success);
  color: var(--accent-success);
  padding: 0.75rem 2rem;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* Modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  -webkit-backdrop-filter: blur(2px);
  backdrop-filter: blur(2px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
  animation: fadeIn 0.2s ease-out;
}

.modal {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  width: min(520px, 92vw);
  box-shadow: var(--shadow-lg);
  overflow: hidden;
}

.modal-header {
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--border);
}

.modal-header h3 {
  color: var(--text-primary);
  font-size: 1.0625rem;
  font-weight: 600;
}

.modal-body {
  padding: 1rem 1.25rem;
  color: var(--text-secondary);
  line-height: 1.6;
}

.modal-list {
  margin: 0.5rem 0 0 1rem;
  color: var(--text-muted);
}

.modal-actions {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
  padding: 0.75rem 1.25rem;
  border-top: 1px solid var(--border);
}
/* Toast Notification */
.toast-container {
  position: fixed;
  top: 1rem;
  right: 1rem;
  z-index: 300;
  pointer-events: none;
}

.toast-card {
  min-width: 280px;
  max-width: 420px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent-primary);
  color: var(--text-secondary);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  padding: 0.75rem 0.75rem 0.75rem 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  pointer-events: auto;
}

.toast-message {
  color: var(--text-primary);
  font-size: 0.875rem;
  font-weight: 500;
}

.toast-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.toast-close {
  background: transparent;
  border: none;
  color: var(--text-dim);
  cursor: pointer;
  font-size: 1rem;
  padding: 0.25rem 0.5rem;
  border-radius: var(--radius-sm);
  transition: var(--transition);
}

.toast-close:hover {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

/* Small Button */
.btn-sm {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
}

/* Sidebar Tabs */
.sidebar-tabs {
  display: flex;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--border);
  background: var(--bg-secondary);
  position: sticky;
  top: 0;
  z-index: 5;
}

.tab-btn {
  padding: 0.375rem 0.75rem;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-muted);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: var(--transition);
  font-size: 0.8125rem;
  font-weight: 500;
}

.tab-btn:hover {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

.tab-btn.active {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
}

/* Trash View */
.trash-view {
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.trash-items {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.trash-item {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  transition: var(--transition);
}

.trash-item:hover {
  background: var(--bg-tertiary);
  transform: translateX(4px);
  box-shadow: var(--shadow-md);
}

.trash-item-content {
  flex: 1;
  min-width: 0;
}

.trash-item-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.5rem;
  gap: 0.75rem;
}

/* Ensure note-item-actions visible in trash list */
.trash-item .note-item-actions {
  opacity: 1;
}
/* Similar Notes Modal */
.similar-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-height: 50vh;
  overflow: auto;
}

.similar-item {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.75rem;
  display: flex;
  justify-content: space-between;
  align-items: stretch;
  gap: 0.75rem;
  transition: var(--transition);
}

.similar-item:hover {
  background: var(--bg-tertiary);
  box-shadow: var(--shadow-md);
}

.similar-main {
  flex: 1;
  min-width: 0;
  cursor: pointer;
}

.similar-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
}

.similar-title {
  color: var(--text-primary);
  font-weight: 600;
  font-size: 0.9375rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.similar-score {
  font-size: 0.75rem;
  color: var(--text-dim);
  white-space: nowrap;
}

.similar-preview {
  color: var(--text-muted);
  font-size: 0.8125rem;
  line-height: 1.5;
  margin: 0.25rem 0 0.5rem;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.similar-meter {
  height: 6px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
}

.similar-meter-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-success), var(--accent-primary));
}

.similar-actions {
  display: flex;
  gap: 0.375rem;
  align-items: center;
}
/* --- Semantic Search Controls --- */
.search-controls {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

.toggle-switch {
  display: inline-flex;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: var(--bg-primary);
  box-shadow: var(--shadow-sm);
}

.toggle-btn {
  padding: 0.375rem 0.75rem;
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.8125rem;
  font-weight: 500;
  transition: var(--transition);
}

.toggle-btn:hover {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

.toggle-btn.active {
  background: var(--accent-primary);
  color: white;
  box-shadow: var(--shadow-sm);
}

.threshold-control {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.threshold-label {
  font-size: 0.75rem;
  color: var(--text-dim);
}

.threshold-slider {
  width: 140px;
  accent-color: var(--accent-primary);
}

/* Inline spinner inside search bar (semantic loading) */
.search-inline-spinner {
  position: absolute;
  right: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  width: 16px;
  height: 16px;
  border: 2px solid var(--border);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

/* Small inline search error chip */
.search-error {
  font-size: 0.75rem;
  color: var(--accent-danger);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid var(--accent-danger);
  padding: 0.25rem 0.5rem;
  border-radius: var(--radius-sm);
}

/* Result similarity indicator in list */
.similarity-badge {
  font-size: 0.75rem;
  color: var(--accent-success);
  background: rgba(16, 185, 129, 0.15);
  border: 1px solid rgba(16, 185, 129, 0.35);
  padding: 0.125rem 0.5rem;
  border-radius: 999px;
  margin-right: 0.5rem;
}

/* Why this matched (closest sentence) */
.why-match {
  color: var(--text-muted);
  font-size: 0.8125rem;
  line-height: 1.5;
  margin-bottom: 0.5rem;
}

/* Responsive tweaks */
@media (max-width: 768px) {
  .threshold-slider {
    width: 110px;
  }
}
/* --- Markdown Editor & Preview --- */
.content-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.content-toolbar-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* Read-only preview styled like an input area */
.markdown-preview,
.markdown-pane,
.markdown-preview-only {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem;
  color: var(--text-secondary);
  font-size: 0.9375rem;
  line-height: 1.7;
}

/* Typography */
.markdown-preview h1,
.markdown-pane h1 { font-size: 1.625rem; color: var(--text-primary); margin: 0.5rem 0 0.75rem; }
.markdown-preview h2,
.markdown-pane h2 { font-size: 1.375rem; color: var(--text-primary); margin: 0.75rem 0 0.5rem; }
.markdown-preview h3,
.markdown-pane h3 { font-size: 1.125rem; color: var(--text-primary); margin: 0.75rem 0 0.5rem; }
.markdown-preview h4,
.markdown-pane h4 { font-size: 1rem;    color: var(--text-primary); margin: 0.75rem 0 0.5rem; }
.markdown-preview h5,
.markdown-pane h5 { font-size: 0.9375rem; color: var(--text-primary); margin: 0.5rem 0; }
.markdown-preview h6,
.markdown-pane h6 { font-size: 0.875rem; color: var(--text-muted); letter-spacing: 0.02em; margin: 0.5rem 0; }

.markdown-preview p,
.markdown-pane p {
  margin: 0.5rem 0 0.75rem;
}

.markdown-preview strong,
.markdown-pane strong {
  color: var(--text-primary);
}

.markdown-preview em,
.markdown-pane em {
  color: var(--text-secondary);
}

/* Lists */
.markdown-preview ul,
.markdown-pane ul,
.markdown-preview ol,
.markdown-pane ol {
  margin: 0.5rem 0 0.75rem 1.25rem;
}

.markdown-preview li,
.markdown-pane li {
  margin: 0.25rem 0;
}

/* Blockquotes */
.markdown-preview blockquote,
.markdown-pane blockquote {
  margin: 0.75rem 0;
  padding: 0.5rem 0.75rem;
  border-left: 3px solid var(--accent-primary);
  background: rgba(59, 130, 246, 0.08);
  color: var(--text-muted);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}

/* Inline code */
.inline-code,
.markdown-preview :not(pre) > code,
.markdown-pane :not(pre) > code {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.85em;
}

/* Code blocks */
.markdown-preview pre,
.markdown-pane pre {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.75rem 1rem;
  overflow: auto;
  box-shadow: var(--shadow-sm);
}

.markdown-preview pre code,
.markdown-pane pre code {
  background: transparent;
  border: none;
  padding: 0;
  color: var(--text-secondary);
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.85rem;
  line-height: 1.6;
}

/* Links */
.markdown-preview a,
.markdown-pane a {
  color: var(--accent-primary);
  text-decoration: none;
}

.markdown-preview a:hover,
.markdown-pane a:hover {
  color: var(--accent-primary-hover);
  text-decoration: underline;
}

/* Tables (GFM) */
.markdown-preview table,
.markdown-pane table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.5rem 0 1rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.markdown-preview th,
.markdown-pane th,
.markdown-preview td,
.markdown-pane td {
  border: 1px solid var(--border);
  padding: 0.5rem 0.75rem;
  text-align: left;
}

.markdown-preview th,
.markdown-pane th {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.markdown-preview tr:nth-child(even) td,
.markdown-pane tr:nth-child(even) td {
  background: rgba(255, 255, 255, 0.02);
}

/* Images */
.markdown-preview img,
.markdown-pane img {
  max-width: 100%;
  display: block;
  margin: 0.5rem 0;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}

/* Horizontal rule */
.markdown-preview hr,
.markdown-pane hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1rem 0;
}

/* Preview only box (single-pane preview) */
.markdown-preview-only {
  min-height: 300px;
}

/* Split view */
.split-container {
  display: flex;
  gap: 0.75rem;
  height: 40vh;
  min-height: 300px;
}

.split-pane {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.split-pane-editor .form-textarea {
  height: 100%;
  min-height: 0;
  resize: none;
}

.split-pane-preview .markdown-pane {
  height: 100%;
  overflow: auto;
}

/* Responsive behavior for split view */
@media (max-width: 900px) {
  .split-container {
    flex-direction: column;
    height: auto;
  }
  .split-pane-editor .form-textarea {
    resize: vertical;
  }
}

/* Ensure preview text contrast for hljs themes */
.markdown-preview .hljs,
.markdown-pane .hljs {
  color: var(--text-secondary);
  background: transparent;
}
/* --- Graph Controls Panel --- */
.graph-controls-panel {
  position: absolute;
  left: 1rem;
  bottom: 1rem;
  z-index: 20;
  width: 320px;
  max-width: calc(100% - 2rem);
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  color: var(--text-secondary);
  pointer-events: auto; /* allow interacting with controls */
  overflow: hidden;
}

.graph-controls-panel.top-left {
  top: 1rem;
  bottom: auto;
}

.graph-controls-panel.bottom-left {
  bottom: 1rem;
  top: auto;
}

.gcp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.5rem 0.5rem 0.5rem 0.375rem;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
}

.gcp-collapse-btn {
  flex: 0 0 auto;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg-primary);
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 0.875rem;
  line-height: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}

.gcp-collapse-btn:hover {
  background: var(--bg-tertiary);
  transform: scale(1.04);
}

.gcp-title {
  flex: 1 1 auto;
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.9375rem;
}

.gcp-meta {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.gcp-stats {
  font-size: 0.75rem;
  color: var(--text-dim);
  white-space: nowrap;
}

.gcp-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid var(--border);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.gcp-body {
  /* Smooth collapse/expand using grid trick */
  display: grid;
  grid-template-rows: 1fr;
  transition: grid-template-rows 0.25s ease, opacity 0.25s ease;
  opacity: 1;
}

.graph-controls-panel.collapsed .gcp-body {
  grid-template-rows: 0fr;
  opacity: 0;
}

.gcp-body > * {
  min-height: 0; /* required for grid collapse trick */
}

.gcp-body-inner {
  padding: 0.75rem 0.75rem 0.75rem 0.75rem;
}

.gcp-row {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
  margin-bottom: 0.75rem;
}

.gcp-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--text-muted);
}

.gcp-select {
  padding: 0.375rem 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 0.8125rem;
  transition: var(--transition);
  outline: none;
  cursor: pointer;
}

.gcp-select:hover {
  background: var(--bg-tertiary);
}

.gcp-select:focus {
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.gcp-range {
  width: 100%;
  accent-color: var(--accent-primary);
}

.gcp-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 0.25rem;
}

/* Responsive sizing */
@media (max-width: 900px) {
  .graph-controls-panel {
    width: 300px;
  }
}

@media (max-width: 600px) {
  .graph-controls-panel {
    width: calc(100% - 2rem);
  }
}
```


### 📄 frontend\src\App.jsx

```
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from './contexts/AuthContext';
import AuthGuard from './components/AuthGuard';
import NoteEditor from './components/NoteEditor';
import NotesList from './components/NotesList';
import GraphVisualization from './components/GraphVisualization';
import { useNotes } from './hooks/useNotes';
import apiService from './services/api';
import './App.css';
import ImportConfirmModal from './components/ImportConfirmModal';
import ToastNotification from './components/ToastNotification';
import TrashView from './components/TrashView';
import UnsavedChangesDialog from './components/UnsavedChangesDialog';
import SimilarNotesModal from './components/SimilarNotesModal';
import ImportLocalNotesModal from './components/ImportLocalNotesModal';
const GRAPH_UPDATE_DEBOUNCE = 500;
const SEMANTIC_QUERY_DEBOUNCE = 500;
const MIN_SEM_QUERY_LEN = 3;

const GC_LS_KEY = 'graph-controls-prefs-v1';
const DEFAULT_GRAPH_PARAMS = {
  connection: 'knn',
  k_neighbors: 5,
  similarity_threshold: 0.7,
  dim_reduction: 'pca',
  clustering: null,
  n_clusters: 5,
};
const clamp = (n, min, max) => Math.min(max, Math.max(min, n));

export default function App() {
  const {
    notes,
    trashedNotes,
    loading: notesLoading,
    error: notesError,
    addNote,
    updateNote,
    deleteNote,
    moveToTrash,
    restoreFromTrash,
    permanentDelete,
    emptyTrash,
    getStats,
    exportNotes,
    importNotes
  } = useNotes();

  const [selectedNote, setSelectedNote] = useState(null);
  const [editingNote, setEditingNote] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  // Search controls
  const [searchMode, setSearchMode] = useState('text'); // 'text' | 'semantic'
  const [minSimilarity, setMinSimilarity] = useState(60); // 0-100%
  const [semanticResults, setSemanticResults] = useState([]); // [{index, score, percent}]
  const [semanticLoading, setSemanticLoading] = useState(false);
  const [semanticError, setSemanticError] = useState('');
  const [graphData, setGraphData] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [showImportModal, setShowImportModal] = useState(false);
  const [pendingImportedNotes, setPendingImportedNotes] = useState([]);
  const [successMessage, setSuccessMessage] = useState('');
 
  const [activeTab, setActiveTab] = useState('notes');
  const [toastOpen, setToastOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [lastTrashedId, setLastTrashedId] = useState(null);

  // Similar notes modal state
  const [similarOpen, setSimilarOpen] = useState(false);
  const [similarBaseDoc, setSimilarBaseDoc] = useState('');
  const [similarBaseTitle, setSimilarBaseTitle] = useState('');
  const [similarExcludeIndex, setSimilarExcludeIndex] = useState(null);
  
  const graphRef = useRef(null);
  const updateTimerRef = useRef(null);
  const fileInputRef = useRef(null);
  const editorRef = useRef(null);
  const [isEditorDirty, setIsEditorDirty] = useState(false);
  const [unsavedOpen, setUnsavedOpen] = useState(false);
  const [pendingAction, setPendingAction] = useState(null);
  const semanticTimerRef = useRef(null);

  // Graph controls (persisted)
  const [graphParams, setGraphParams] = useState(() => {
    try {
      const raw = localStorage.getItem(GC_LS_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      return { ...DEFAULT_GRAPH_PARAMS, ...(parsed || {}) };
    } catch {
      return DEFAULT_GRAPH_PARAMS;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(GC_LS_KEY, JSON.stringify(graphParams));
    } catch { /* ignore quota */ }
  }, [graphParams]);

  const handleControlsChange = useCallback((partial) => {
    setGraphParams((prev) => ({ ...prev, ...partial }));
  }, []);

  const handleControlsReset = useCallback(() => {
    setGraphParams(DEFAULT_GRAPH_PARAMS);
  }, []);
 
  // Check backend connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await apiService.checkHealth();
        setConnected(true);
        setError(null);
      } catch (err) {
        setConnected(false);
        setError('Backend unavailable. Ensure server is running on http://localhost:8000');
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);
 
  // Auto-clear success messages
  useEffect(() => {
    if (!successMessage) return;
    const t = setTimeout(() => setSuccessMessage(''), 4000);
    return () => clearTimeout(t);
  }, [successMessage]);
 
  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (graphRef.current) {
        setDimensions({
          width: graphRef.current.clientWidth,
          height: graphRef.current.clientHeight
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Generate graph with debouncing
  useEffect(() => {
    if (updateTimerRef.current) {
      clearTimeout(updateTimerRef.current);
    }

    if (!connected || notes.length < 2) {
      setGraphData(null);
      return;
    }

    updateTimerRef.current = setTimeout(async () => {
      setGraphLoading(true);
      try {
        const documents = notes.map(note =>
          `${note.title}. ${note.content} ${note.tags || ''}`
        );
        
        const labels = notes.map(note =>
          note.title.length > 30 ? `${note.title.substring(0, 30)}...` : note.title
        );

        const connection = graphParams.connection === 'threshold' ? 'threshold' : 'knn';
        const kNeighborsRaw = clamp(parseInt(graphParams.k_neighbors ?? 5, 10), 1, 10);
        const kNeighbors = Math.min(kNeighborsRaw, Math.max(1, notes.length - 1));
        const similarity_threshold = Math.max(0, Math.min(1, Number(graphParams.similarity_threshold ?? 0.7)));
        const dim_reduction = graphParams.dim_reduction === 'none' ? null : (graphParams.dim_reduction ?? 'pca');
        const clustering = graphParams.clustering ?? null;
        const n_clusters = clustering ? clamp(parseInt(graphParams.n_clusters ?? 5, 10), 2, 20) : undefined;

        const graph = await apiService.buildGraph({
          documents,
          labels,
          connection,
          k_neighbors: connection === 'knn' ? kNeighbors : undefined,
          similarity_threshold: connection === 'threshold' ? similarity_threshold : undefined,
          dim_reduction,
          clustering,
          n_clusters,
        });

        setGraphData(graph);
        setError(null);
      } catch (err) {
        console.error('Graph generation failed:', err);
        setError(`Graph generation failed: ${err.message}`);
      } finally {
        setGraphLoading(false);
      }
    }, GRAPH_UPDATE_DEBOUNCE);

    return () => {
      if (updateTimerRef.current) {
        clearTimeout(updateTimerRef.current);
      }
    };
  }, [notes, connected, graphParams]);

  // Semantic Search (debounced)
  useEffect(() => {
    if (searchMode !== 'semantic') {
      setSemanticLoading(false);
      setSemanticError('');
      setSemanticResults([]);
      return;
    }
    const q = String(searchTerm || '').trim();
    if (!q) {
      setSemanticResults([]);
      setSemanticLoading(false);
      setSemanticError('');
      return;
    }
    if (q.length < MIN_SEM_QUERY_LEN) {
      setSemanticResults([]);
      setSemanticLoading(false);
      setSemanticError(`Type at least ${MIN_SEM_QUERY_LEN} characters for semantic search`);
      return;
    }
    if (!connected) {
      setSemanticResults([]);
      setSemanticLoading(false);
      setSemanticError('Semantic search requires backend connection');
      return;
    }

    if (semanticTimerRef.current) {
      clearTimeout(semanticTimerRef.current);
    }
    let cancelled = false;
    semanticTimerRef.current = setTimeout(async () => {
      setSemanticLoading(true);
      setSemanticError('');
      try {
        const [noteEmbs, queryEmb] = await Promise.all([
          apiService.getEmbeddingsForNotes(notes),
          apiService.embedText(q),
        ]);
        const scored = [];
        for (let i = 0; i < notes.length; i++) {
          const v = noteEmbs[i];
          if (!Array.isArray(v)) continue;
          const s = apiService.cosineSimilarity(queryEmb, v);
          scored.push({ index: i, score: s, percent: Math.round(s * 100) });
        }
        scored.sort((a, b) => b.score - a.score);
        if (!cancelled) setSemanticResults(scored);
      } catch (e) {
        console.error('Semantic search failed:', e);
        if (!cancelled) setSemanticError(e?.message || 'Semantic search failed');
      } finally {
        if (!cancelled) setSemanticLoading(false);
      }
    }, SEMANTIC_QUERY_DEBOUNCE);

    return () => {
      cancelled = true;
      if (semanticTimerRef.current) clearTimeout(semanticTimerRef.current);
    };
  }, [searchMode, searchTerm, notes, connected]);

  const handleSaveNote = useCallback((noteData) => {
    if (editingNote) {
      updateNote(editingNote.originalIndex, noteData);
      setEditingNote(null);
    } else {
      addNote(noteData);
      setIsCreating(false);
    }
  }, [editingNote, updateNote, addNote]);

  const handleEditNote = useCallback((index) => {
    // If there are unsaved changes in the current editor, prompt first
    setSelectedNote(prev => prev); // no-op to keep dependencies minimal
    const action = { type: 'edit', index };
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction(action);
      setUnsavedOpen(true);
    } else {
      setEditingNote({ ...notes[index], originalIndex: index });
      setIsCreating(false);
      setSelectedNote(null);
    }
  }, [notes, isCreating, editingNote, isEditorDirty]);

  const handleDeleteNote = useCallback((index) => {
    const moved = moveToTrash(index);
    if (moved) {
      setLastTrashedId(moved.id);
      setToastMessage('Note moved to trash. Undo?');
      setToastOpen(true);
    }
    if (selectedNote === index) setSelectedNote(null);
  }, [moveToTrash, selectedNote]);

  const handleUndo = useCallback(() => {
    if (lastTrashedId != null) {
      restoreFromTrash(lastTrashedId);
      setLastTrashedId(null);
    }
  }, [lastTrashedId, restoreFromTrash]);

  const handleNodeClick = useCallback((nodeId) => {
    const action = { type: 'nodeSelect', index: nodeId };
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction(action);
      setUnsavedOpen(true);
    } else {
      setSelectedNote(nodeId);
      setEditingNote(null);
      setIsCreating(false);
    }
  }, [isCreating, editingNote, isEditorDirty]);

  const handleNewNote = useCallback(() => {
    const action = { type: 'new' };
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction(action);
      setUnsavedOpen(true);
    } else {
      setIsCreating(true);
      setEditingNote(null);
      setSelectedNote(null);
    }
  }, [isCreating, editingNote, isEditorDirty]);

  const handleCancel = useCallback(() => {
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction({ type: 'cancel' });
      setUnsavedOpen(true);
    } else {
      setIsCreating(false);
      setEditingNote(null);
    }
  }, [isCreating, editingNote, isEditorDirty]);
 
  const handleImportClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);
 
  const handleFileSelected = useCallback((e) => {
    try {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      if (!file.name.toLowerCase().endsWith('.json')) {
        setError('Please select a JSON file');
        e.target.value = '';
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const text = String(reader.result || '');
          const parsed = JSON.parse(text);
          const incoming = Array.isArray(parsed) ? parsed : (parsed && Array.isArray(parsed.notes) ? parsed.notes : null);
          if (!incoming) {
            throw new Error('Invalid file format. Expected { "notes": [...] }');
          }
          if (!Array.isArray(incoming)) {
            throw new Error('Invalid notes format in file');
          }
          if (incoming.length === 0) {
            setError('No notes found in file');
            e.target.value = '';
            return;
          }
          setPendingImportedNotes(incoming);
          setShowImportModal(true);
          setError(null);
        } catch (err) {
          console.error('Import parse error:', err);
          setError(`Import failed: ${err.message}`);
        } finally {
          e.target.value = '';
        }
      };
      reader.onerror = () => {
        setError('Failed to read file');
        e.target.value = '';
      };
      reader.readAsText(file);
    } catch (err) {
      setError(`Import failed: ${err.message}`);
      if (e?.target) e.target.value = '';
    }
  }, []);
 
  const confirmReplace = useCallback(() => {
    try {
      const res = importNotes(pendingImportedNotes, 'replace');
      setSuccessMessage(`Imported ${res.imported} notes successfully`);
    } catch (err) {
      setError(`Import failed: ${err.message}`);
    } finally {
      setShowImportModal(false);
      setPendingImportedNotes([]);
      setSelectedNote(null);
      setEditingNote(null);
      setIsCreating(false);
    }
  }, [importNotes, pendingImportedNotes]);
 
  const confirmMerge = useCallback(() => {
    try {
      const res = importNotes(pendingImportedNotes, 'merge');
      setSuccessMessage(`Imported ${res.imported} notes successfully`);
    } catch (err) {
      setError(`Import failed: ${err.message}`);
    } finally {
      setShowImportModal(false);
      setPendingImportedNotes([]);
    }
  }, [importNotes, pendingImportedNotes]);
 
  // Navigation helpers that respect unsaved changes
  const executeAction = useCallback((action) => {
    if (!action) return;
    switch (action.type) {
      case 'cancel':
        setIsCreating(false);
        setEditingNote(null);
        break;
      case 'new':
        setIsCreating(true);
        setEditingNote(null);
        setSelectedNote(null);
        break;
      case 'edit': {
        const idx = action.index;
        if (idx != null && notes[idx]) {
          setEditingNote({ ...notes[idx], originalIndex: idx });
          setIsCreating(false);
          setSelectedNote(null);
        }
        break;
      }
      case 'selectNote':
      case 'nodeSelect': {
        const idx = action.index;
        if (idx != null) {
          setSelectedNote(idx);
          setEditingNote(null);
          setIsCreating(false);
        }
        break;
      }
      case 'tab':
        setActiveTab(action.tab);
        break;
      default:
        break;
    }
  }, [notes]);
  
  const requestNavigate = useCallback((action) => {
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction(action);
      setUnsavedOpen(true);
    } else {
      executeAction(action);
    }
  }, [isCreating, editingNote, isEditorDirty, executeAction]);
  
  const handleTabChange = useCallback((tab) => {
    requestNavigate({ type: 'tab', tab });
  }, [requestNavigate]);

  // ---------- Find Similar Notes helpers ----------
  const buildDocText = useCallback((n) => {
    const t = String(n?.title || '').trim();
    const c = String(n?.content || '').trim();
    const g = String(n?.tags || '').trim();
    return `${t}. ${c} ${g}`.trim();
  }, []);

  const openSimilar = useCallback((baseDoc, baseTitle, excludeIndex = null) => {
    setSimilarBaseDoc(baseDoc);
    setSimilarBaseTitle(baseTitle);
    setSimilarExcludeIndex(excludeIndex);
    setSimilarOpen(true);
  }, []);

  const handleFindSimilarFromEditor = useCallback(() => {
    let data = null;
    if (editorRef.current && typeof editorRef.current.getCurrentData === 'function') {
      data = editorRef.current.getCurrentData();
    } else if (editingNote) {
      data = editingNote;
    }
    if (!data) {
      setError('No note data to analyze');
      return;
    }
    const doc = buildDocText(data);
    const title = String(data.title || 'This note');
    const exclude = editingNote?.originalIndex ?? null;
    openSimilar(doc, title, exclude);
  }, [editorRef, editingNote, buildDocText, openSimilar]);

  const handleFindSimilarFromList = useCallback((index) => {
    const n = notes[index];
    if (!n) return;
    const doc = buildDocText(n);
    openSimilar(doc, n.title || 'This note', index);
  }, [notes, buildDocText, openSimilar]);

  const LINKS_KEY = 'semantic-links-v1';
  const addLink = useCallback((aId, bId) => {
    try {
      if (aId == null || bId == null) {
        setError('Unable to link: missing note id(s). Save the note first.');
        return;
      }
      const raw = localStorage.getItem(LINKS_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      const pair = aId < bId ? [aId, bId] : [bId, aId];
      const exists = Array.isArray(arr) && arr.some((p) => Array.isArray(p) && p[0] === pair[0] && p[1] === pair[1]);
      const next = exists ? arr : [...(Array.isArray(arr) ? arr : []), pair];
      localStorage.setItem(LINKS_KEY, JSON.stringify(next));
      setSuccessMessage('Notes linked');
    } catch (e) {
      setError('Failed to save link');
    }
  }, []);
 
  const saveAndContinue = useCallback(() => {
    if (!editorRef.current) return;
    const data = editorRef.current.getCurrentData();
    const titleOk = String(data.title || '').trim().length > 0;
    const contentOk = String(data.content || '').trim().length > 0;
    if (!titleOk || !contentOk) {
      alert('Title and Content are required');
      return;
    }
    handleSaveNote(data);
    setUnsavedOpen(false);
    if (pendingAction) {
      executeAction(pendingAction);
      setPendingAction(null);
    }
  }, [handleSaveNote, pendingAction, executeAction]);
  
  const discardChanges = useCallback(() => {
    setUnsavedOpen(false);
    // Close the editor and proceed
    setIsCreating(false);
    setEditingNote(null);
    if (pendingAction) {
      executeAction(pendingAction);
      setPendingAction(null);
    }
  }, [pendingAction, executeAction]);
  
  const cancelDialog = useCallback(() => {
    setUnsavedOpen(false);
  }, []);
  
  const stats = getStats();

  const { user, logout, isAuthenticated } = useAuth();

   // LocalStorage import modal logic
   const [showImportLocalModal, setShowImportLocalModal] = useState(false);
   useEffect(() => {
     if (isAuthenticated && user) {
       const hasCheckedImport = localStorage.getItem('import-checked');
       if (!hasCheckedImport) {
         const notesData = localStorage.getItem('semantic-notes-data');
         const trashData = localStorage.getItem('semantic-notes-trash');
         if (notesData || trashData) {
           setShowImportLocalModal(true);
         }
         localStorage.setItem('import-checked', 'true');
       }
     }
   }, [isAuthenticated, user]);

   const handleImportComplete = (importedCount) => {
     setShowImportLocalModal(false);
     window.location.reload();
   };

   const handleImportSkip = () => {
     setShowImportLocalModal(false);
   };

  return (
    <AuthGuard>
       {showImportLocalModal && (
         <ImportLocalNotesModal
           onClose={handleImportSkip}
           onImportComplete={handleImportComplete}
         />
       )}
      <div className="app">
        <header className="app-header">
        <h1>Semantic Notes</h1>
        
        <div className="header-actions">
          <div className="search-bar">
            <span className="search-icon">⌕</span>
            <input
              type="text"
              placeholder={searchMode === 'semantic' ? 'Semantic search…' : 'Search notes...'}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
            {searchMode === 'semantic' && semanticLoading && (
              <span className="search-inline-spinner" aria-label="Searching" />
            )}
          </div>

          <div className="search-controls">
            <div className="toggle-switch" role="group" aria-label="Search mode">
              <button
                className={`toggle-btn ${searchMode === 'text' ? 'active' : ''}`}
                onClick={() => setSearchMode('text')}
                title="Text Search"
              >
                Text
              </button>
              <button
                className={`toggle-btn ${searchMode === 'semantic' ? 'active' : ''}`}
                onClick={() => setSearchMode('semantic')}
                title="Semantic Search"
              >
                Semantic
              </button>
            </div>

            {searchMode === 'semantic' && (
              <div className="threshold-control" title="Minimum similarity threshold">
                <label className="threshold-label">
                  Min similarity: {minSimilarity}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={minSimilarity}
                  onChange={(e) => setMinSimilarity(Number(e.target.value))}
                  className="threshold-slider"
                />
              </div>
            )}
          </div>

          {searchMode === 'semantic' && semanticError && (
            <div className="search-error" title={semanticError}>⚠ {semanticError}</div>
          )}

          <button onClick={handleNewNote} className="btn btn-primary">
            New Note
          </button>

          <button
            onClick={exportNotes}
            className="btn btn-secondary btn-icon"
            title="Export Notes"
          >
            ↓
          </button>
 
          <button
            onClick={handleImportClick}
            className="btn btn-secondary btn-icon"
            title="Import Notes"
          >
            ↑
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            style={{ display: 'none' }}
            onChange={handleFileSelected}
          />

          <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            <span className={`status-indicator ${connected ? '' : 'disconnected'}`} />
            {connected ? 'Connected' : 'Disconnected'}
          </div>
          {isAuthenticated && user && (
            <div className="user-section">
              <span className="username">👤 {user.username}</span>
              <button
                onClick={logout}
                className="logout-button"
                title="Logout"
              >
                Logout
              </button>
            </div>
          )}
        </div>
      </header>

      {(error || notesError) && (
        <div className="error-banner">
          ⚠ {error || notesError}
        </div>
      )}
 
      {successMessage && (
        <div className="success-banner">
          ✓ {successMessage}
        </div>
      )}
 
      <div className="main-content">
        <div className="sidebar">
          <div className="sidebar-tabs">
            <button
              className={`tab-btn ${activeTab === 'notes' ? 'active' : ''}`}
              onClick={() => handleTabChange('notes')}
              title="View Notes"
            >
              Notes
            </button>
            <button
              className={`tab-btn ${activeTab === 'trash' ? 'active' : ''}`}
              onClick={() => handleTabChange('trash')}
              title="View Trash"
            >
              Trash{trashedNotes.length ? ` (${trashedNotes.length})` : ''}
            </button>
          </div>

          {activeTab === 'notes' ? (
            isCreating || editingNote ? (
              <NoteEditor
                ref={editorRef}
                note={editingNote}
                onSave={handleSaveNote}
                onCancel={handleCancel}
                onDirtyChange={setIsEditorDirty}
                onFindSimilar={handleFindSimilarFromEditor}
              />
            ) : (
              <>
                <NotesList
                  notes={notes}
                  onSelect={(i) => requestNavigate({ type: 'selectNote', index: i })}
                  onEdit={handleEditNote}
                  onDelete={handleDeleteNote}
                  selectedNote={selectedNote}
                  searchTerm={searchTerm}
                  onFindSimilar={handleFindSimilarFromList}
                  searchMode={searchMode}
                  semanticResults={semanticResults}
                  minSimilarity={minSimilarity}
                  semanticLoading={semanticLoading}
                  semanticError={semanticError}
                />
                
                {(notes.length > 0 || trashedNotes.length > 0) && (
                  <div className="stats-bar">
                    <div>{stats.totalNotes} notes • {stats.totalWords} words</div>
                    <div>{stats.totalTags} unique tags • {stats.trashCount} in trash</div>
                  </div>
                )}
              </>
            )
          ) : (
            <TrashView
              trashedNotes={trashedNotes}
              onRestore={restoreFromTrash}
              onDeleteForever={permanentDelete}
              onEmptyTrash={emptyTrash}
            />
          )}
        </div>

        <div className="graph-container" ref={graphRef}>
          {notesLoading ? (
            <div className="loading">
              <div className="loading-spinner" />
              <div>Loading notes...</div>
            </div>
          ) : notes.length === 0 ? (
            <div className="empty-state">
              <h3>Welcome to Semantic Notes</h3>
              <p>Create your first note to get started.</p>
              <p style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
                Notes will be connected based on semantic similarity.
              </p>
            </div>
          ) : notes.length === 1 ? (
            <div className="empty-state">
              <p>Create at least 2 notes to visualize connections</p>
            </div>
          ) : graphLoading ? (
            <div className="loading">
              <div className="loading-spinner" />
              <div>Generating semantic graph...</div>
            </div>
          ) : graphData ? (
            <GraphVisualization
              graphData={graphData}
              onNodeClick={handleNodeClick}
              selectedNote={selectedNote}
              width={dimensions.width}
              height={dimensions.height}
              controlsParams={graphParams}
              onControlsChange={handleControlsChange}
              onControlsReset={handleControlsReset}
              stats={{
                nodes: graphData?.nodes?.length || 0,
                edges: graphData?.edges?.length || 0
              }}
              loading={graphLoading}
              panelPosition="bottom-left"
            />
          ) : (
            <div className="empty-state">
              <p>Unable to generate graph. Check connection.</p>
            </div>
          )}

          {selectedNote !== null && notes[selectedNote] && (
            <div className="selected-note-preview fade-in">
              <h3>{notes[selectedNote].title}</h3>
              <p>{notes[selectedNote].content}</p>
              {notes[selectedNote].tags && (
                <div className="note-tags" style={{ marginTop: '0.5rem' }}>
                  {notes[selectedNote].tags.split(',').map((tag, i) => (
                    <span key={i} className="tag">
                      {tag.trim()}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
 
      <SimilarNotesModal
        isOpen={similarOpen}
        baseDoc={similarBaseDoc}
        baseTitle={similarBaseTitle}
        notes={notes}
        excludeIndex={similarExcludeIndex}
        topK={8}
        onClose={() => setSimilarOpen(false)}
        onSelect={(idx) => {
          setSimilarOpen(false);
          requestNavigate({ type: 'selectNote', index: idx });
        }}
        onLink={similarExcludeIndex != null ? ((idx) => {
          const src = notes[similarExcludeIndex]?.id;
          const dst = notes[idx]?.id;
          addLink(src, dst);
        }) : undefined}
      />

      <ImportConfirmModal
        isOpen={showImportModal}
        count={pendingImportedNotes.length}
        onReplace={confirmReplace}
        onMerge={confirmMerge}
        onCancel={() => {
          setShowImportModal(false);
          setPendingImportedNotes([]);
        }}
      />

      <ToastNotification
        isOpen={toastOpen}
        message={toastMessage}
        actionLabel="Undo"
        onAction={handleUndo}
        onClose={() => setToastOpen(false)}
        duration={5000}
      />

      <UnsavedChangesDialog
        isOpen={unsavedOpen}
        onSaveAndContinue={saveAndContinue}
        onDiscard={discardChanges}
        onCancel={cancelDialog}
      />
    </div>
    </AuthGuard>
  );
}
```


### 📄 frontend\src\components\AuthGuard.jsx

```
import { useAuth } from '../contexts/AuthContext';
import LoginForm from './LoginForm';

export default function AuthGuard({ children }) {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh'
      }}>
        <div>Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginForm />;
  }

  return children;
}
```


### 📄 frontend\src\components\ConfirmDialog.jsx

```
import React from 'react';

export default function ConfirmDialog({
  isOpen = false,
  title = 'Confirm',
  message = '',
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
  danger = false
}) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
      <div className="modal">
        <div className="modal-header">
          <h3 id="confirm-title">{title}</h3>
        </div>
        <div className="modal-body">
          <p>{message}</p>
        </div>
        <div className="modal-actions">
          <button
            className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`}
            onClick={() => onConfirm && onConfirm()}
          >
            {confirmLabel}
          </button>
          <button className="btn btn-secondary" onClick={() => onCancel && onCancel()}>
            {cancelLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 frontend\src\components\ExportGraphModal.jsx

```
import React, { useEffect, useMemo, useState } from 'react';
import {
  defaultFilename,
  savePng,
  saveSvg,
  saveJson,
  copyPng,
  copySvg,
  copyJson,
  formatTimestamp,
} from '../utils/graphExport';

export default function ExportGraphModal({
  isOpen = false,
  onClose,
  svgRef,
  graphData,
  params = {},
  transform = { x: 0, y: 0, k: 1 },
  onNotify, // optional toast notifier: (msg: string) => void
}) {
  const [format, setFormat] = useState('png'); // 'png' | 'svg' | 'json'
  const [filename, setFilename] = useState(defaultFilename('png'));
  const [scale, setScale] = useState(2); // PNG resolution multiplier
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isOpen) return;
    setFormat('png');
    setScale(2);
    setError('');
    setBusy(false);
    setFilename(defaultFilename('png'));
  }, [isOpen]);

  const metadata = useMemo(() => ({
    exported_at: new Date().toISOString(),
    timestamp: formatTimestamp(),
    params,
    transform,
    stats: {
      nodes: graphData?.nodes?.length || 0,
      edges: graphData?.edges?.length || 0,
    },
  }), [params, transform, graphData]);

  const ensureExt = (name, ext) => {
    const lower = String(name || '').toLowerCase().trim();
    if (!lower.endsWith(`.${ext}`)) return `${name}.${ext}`;
    return name;
  };

  const handleDownload = async () => {
    try {
      setBusy(true);
      setError('');
      const svgEl = svgRef?.current;
      if (!svgEl) throw new Error('SVG not available');

      if (format === 'png') {
        const fname = ensureExt(filename || defaultFilename('png'), 'png');
        await savePng(svgEl, fname, scale);
        onNotify?.(`Exported PNG: ${fname}`);
      } else if (format === 'svg') {
        const fname = ensureExt(filename || defaultFilename('svg'), 'svg');
        saveSvg(svgEl, fname);
        onNotify?.(`Exported SVG: ${fname}`);
      } else if (format === 'json') {
        const fname = ensureExt(filename || defaultFilename('json'), 'json');
        saveJson(graphData, fname, metadata);
        onNotify?.(`Exported JSON: ${fname}`);
      }
      onClose?.();
    } catch (e) {
      setError(e?.message || 'Export failed');
    } finally {
      setBusy(false);
    }
  };

  const handleCopy = async () => {
    try {
      setBusy(true);
      setError('');
      const svgEl = svgRef?.current;
      if (format === 'png') {
        await copyPng(svgEl, scale);
        onNotify?.('PNG copied to clipboard');
      } else if (format === 'svg') {
        await copySvg(svgEl);
        onNotify?.('SVG copied to clipboard');
      } else if (format === 'json') {
        await copyJson(graphData, metadata);
        onNotify?.('JSON copied to clipboard');
      }
      onClose?.();
    } catch (e) {
      // Graceful fallback for clipboard limitations
      setError(e?.message || 'Copy failed. Your browser may not support clipboard for this format.');
    } finally {
      setBusy(false);
    }
  };

  if (!isOpen) return null;

  const stop = (e) => e.stopPropagation();

  return (
    <div className="modal-overlay" onClick={() => !busy && onClose?.()}>
      <div className="modal" role="dialog" aria-modal="true" onClick={stop}>
        <div className="modal-header">
          <h3>Export Graph</h3>
        </div>

        <div className="modal-body">
          <div className="gcp-row" style={{ marginBottom: '0.75rem' }}>
            <label className="gcp-label" htmlFor="export-format">Format</label>
            <div id="export-format" className="radio-group" role="radiogroup" aria-label="Export format">
              <label className="radio">
                <input
                  type="radio"
                  name="format"
                  value="png"
                  checked={format === 'png'}
                  onChange={() => {
                    setFormat('png');
                    setFilename(defaultFilename('png'));
                  }}
                  disabled={busy}
                />
                PNG (image)
              </label>
              <label className="radio" style={{ marginLeft: '1rem' }}>
                <input
                  type="radio"
                  name="format"
                  value="svg"
                  checked={format === 'svg'}
                  onChange={() => {
                    setFormat('svg');
                    setFilename(defaultFilename('svg'));
                  }}
                  disabled={busy}
                />
                SVG (vector)
              </label>
              <label className="radio" style={{ marginLeft: '1rem' }}>
                <input
                  type="radio"
                  name="format"
                  value="json"
                  checked={format === 'json'}
                  onChange={() => {
                    setFormat('json');
                    setFilename(defaultFilename('json'));
                  }}
                  disabled={busy}
                />
                JSON (data)
              </label>
            </div>
          </div>

          {format === 'png' && (
            <div className="gcp-row" style={{ marginBottom: '0.75rem' }}>
              <label className="gcp-label" htmlFor="export-scale">Resolution</label>
              <select
                id="export-scale"
                className="gcp-select"
                value={scale}
                onChange={(e) => setScale(parseInt(e.target.value, 10) || 2)}
                disabled={busy}
                title="Export scale factor"
              >
                <option value={1}>1x (standard)</option>
                <option value={2}>2x (high)</option>
                <option value={3}>3x (ultra)</option>
              </select>
            </div>
          )}

          <div className="gcp-row" style={{ marginBottom: '0.25rem' }}>
            <label className="gcp-label" htmlFor="export-filename">Filename</label>
            <input
              id="export-filename"
              className="gcp-select"
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              disabled={busy}
              placeholder={defaultFilename(format)}
              title="Choose a filename"
            />
          </div>

          <div className="small" style={{ opacity: 0.8, marginTop: '0.25rem' }}>
            Tip: Exported image reflects current zoom/pan and styling.
          </div>

          {error && (
            <div className="error-banner" style={{ marginTop: '0.5rem' }}>
              ⚠ {error}
            </div>
          )}
        </div>

        <div className="modal-actions">
          <button className="btn btn-primary" onClick={handleDownload} disabled={busy}>
            {busy ? 'Exporting…' : 'Download'}
          </button>
          <button className="btn btn-secondary" onClick={handleCopy} disabled={busy} title="Copy to clipboard">
            Copy
          </button>
          <button className="btn" onClick={() => !busy && onClose?.()} disabled={busy}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 frontend\src\components\GraphControlsPanel.jsx

```
import React, { useEffect, useMemo, useState } from 'react';

const COLLAPSE_LS_KEY = 'graph-controls-collapsed-v1';

function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

export default function GraphControlsPanel({
  params,
  onChange,
  onReset,
  stats = { nodes: 0, edges: 0 },
  loading = false,
  position = 'bottom-left', // 'top-left' | 'bottom-left'
}) {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      const raw = localStorage.getItem(COLLAPSE_LS_KEY);
      return raw ? JSON.parse(raw) === true : false;
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSE_LS_KEY, JSON.stringify(collapsed));
    } catch {
      /* ignore */
    }
  }, [collapsed]);

  const isKnn = params?.connection === 'knn';
  const showClusters = !!params?.clustering;

  const panelClass = useMemo(() => {
    const base = 'graph-controls-panel';
    const pos = position === 'top-left' ? 'top-left' : 'bottom-left';
    return `${base} ${pos} ${collapsed ? 'collapsed' : ''}`;
  }, [position, collapsed]);

  return (
    <div className={panelClass} aria-live="polite">
      <div className="gcp-header">
        <button
          className="gcp-collapse-btn"
          onClick={() => setCollapsed((c) => !c)}
          aria-expanded={!collapsed}
          aria-controls="gcp-body"
          title={collapsed ? 'Expand controls' : 'Collapse controls'}
        >
          {collapsed ? '▸' : '▾'}
        </button>

        <div className="gcp-title" title="Graph Controls">Graph Controls</div>

        <div className="gcp-meta">
          <span className="gcp-stats" title="Current graph statistics">
            {stats.nodes} nodes, {stats.edges} edges
          </span>
          {loading && <span className="gcp-spinner" aria-label="Regenerating graph" />}
        </div>
      </div>

      <div id="gcp-body" className="gcp-body">
        <div className="gcp-body-inner">
          <div className="gcp-row">
            <label
              className="gcp-label"
              htmlFor="gcp-connection"
              title="How nodes are connected: k nearest neighbors or similarity threshold"
            >
              Connection
            </label>
            <select
              id="gcp-connection"
              className="gcp-select"
              value={params.connection}
              onChange={(e) => onChange({ connection: e.target.value === 'threshold' ? 'threshold' : 'knn' })}
              title="kNN: connect each node to its k most similar neighbors; Threshold: connect nodes whose similarity is above a set value"
            >
              <option value="knn">kNN</option>
              <option value="threshold">Threshold</option>
            </select>
          </div>

          {isKnn ? (
            <div className="gcp-row">
              <label className="gcp-label" htmlFor="gcp-k" title="Number of nearest neighbors per node (1-10)">
                k neighbors: {clamp(params.k_neighbors ?? 5, 1, 10)}
              </label>
              <input
                id="gcp-k"
                className="gcp-range"
                type="range"
                min="1"
                max="10"
                step="1"
                value={clamp(params.k_neighbors ?? 5, 1, 10)}
                onChange={(e) => onChange({ k_neighbors: clamp(parseInt(e.target.value, 10), 1, 10) })}
                title="Connect each node to its k most similar neighbors"
              />
            </div>
          ) : (
            <div className="gcp-row">
              <label
                className="gcp-label"
                htmlFor="gcp-threshold"
                title="Minimum cosine similarity (0.00-1.00) required to draw an edge"
              >
                Threshold: {(params.similarity_threshold ?? 0.7).toFixed(2)}
              </label>
              <input
                id="gcp-threshold"
                className="gcp-range"
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={clamp(Number(params.similarity_threshold ?? 0.7), 0, 1)}
                onChange={(e) => onChange({ similarity_threshold: clamp(parseFloat(e.target.value), 0, 1) })}
                title="Edges connect nodes with similarity equal to or greater than this value"
              />
            </div>
          )}

          <div className="gcp-row">
            <label
              className="gcp-label"
              htmlFor="gcp-dr"
              title="Dimensionality reduction for layout: project embeddings to 2D"
            >
              Dimensionality reduction
            </label>
            <select
              id="gcp-dr"
              className="gcp-select"
              value={params.dim_reduction ?? 'none'}
              onChange={(e) => {
                const v = e.target.value;
                onChange({ dim_reduction: v === 'none' ? null : v });
              }}
              title="Choose PCA (fast, linear), UMAP/t-SNE (nonlinear, capture local structure), or None"
            >
              <option value="pca">PCA</option>
              <option value="umap">UMAP</option>
              <option value="tsne">t-SNE</option>
              <option value="none">None</option>
            </select>
          </div>

          <div className="gcp-row">
            <label className="gcp-label" htmlFor="gcp-cluster" title="Cluster nodes into groups">
              Clustering
            </label>
            <select
              id="gcp-cluster"
              className="gcp-select"
              value={params.clustering ?? 'none'}
              onChange={(e) => {
                const v = e.target.value;
                onChange({ clustering: v === 'none' ? null : v });
              }}
              title="None: no clustering; k-means: partition into k clusters; Agglomerative: hierarchical clustering"
            >
              <option value="none">None</option>
              <option value="kmeans">k-means</option>
              <option value="agglomerative">Agglomerative</option>
            </select>
          </div>

          {showClusters && (
            <div className="gcp-row">
              <label className="gcp-label" htmlFor="gcp-nc" title="Number of clusters (2-20)">
                Clusters: {clamp(params.n_clusters ?? 5, 2, 20)}
              </label>
              <input
                id="gcp-nc"
                className="gcp-range"
                type="range"
                min="2"
                max="20"
                step="1"
                value={clamp(params.n_clusters ?? 5, 2, 20)}
                onChange={(e) => onChange({ n_clusters: clamp(parseInt(e.target.value, 10), 2, 20) })}
                title="Number of clusters for the selected clustering method"
              />
            </div>
          )}

          <div className="gcp-actions">
            <button
              className="btn btn-secondary btn-sm"
              onClick={onReset}
              title="Reset all controls to their default values"
            >
              Reset to defaults
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
```


### 📄 frontend\src\components\GraphVisualization.jsx

```
import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import GraphControlsPanel from './GraphControlsPanel';
import ExportGraphModal from './ExportGraphModal';
import ToastNotification from './ToastNotification';

const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];

export default function GraphVisualization({
  graphData,
  onNodeClick,
  selectedNote,
  width = 800,
  height = 600,
  controlsParams = {},
  onControlsChange = () => {},
  onControlsReset = () => {},
  stats = { nodes: 0, edges: 0 },
  loading = false,
  panelPosition = 'bottom-left'
}) {
  const svgRef = useRef(null);
  const simulationRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [isRunning, setIsRunning] = useState(true);

  // Export modal and notifications
  const [exportOpen, setExportOpen] = useState(false);
  const [exportTransform, setExportTransform] = useState({ x: 0, y: 0, k: 1 });
  const [toastOpen, setToastOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState('');

  useEffect(() => {
    if (!graphData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { nodes, edges } = graphData;
    if (!nodes || nodes.length === 0) return;

    const g = svg.append('g');
    
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => g.attr('transform', event.transform));
    
    svg.call(zoom);

    // Prepare data

    const nodeData = nodes.map(n => ({
      ...n,                                // keep your original props (name/title/label/cluster/color)
      id: String(n.id),
      x: typeof n.x === 'number' ? (n.x * width / 2 + width / 2) : Math.random() * width,
      y: typeof n.y === 'number' ? (n.y * height / 2 + height / 2) : Math.random() * height,
    }));

    const linkData = edges.map(e => ({
      ...e,                                // keep link props (weight/type/etc.)
      source: String(e.source),
      target: String(e.target),
    }));


    const weightScale = d3.scaleLinear().domain([0, 1]).range([0.1, 1]);

    // Create simulation
    const simulation = d3.forceSimulation(nodeData)
      .force('link', d3.forceLink(linkData)
        .id(d => d.id)
        .distance(d => 120 * (1 - (d.weight || 0.5)))
        .strength(d => weightScale(d.weight || 0.5)))
      .force('charge', d3.forceManyBody().strength(-400).distanceMax(250))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(35).strength(0.7))
      .force('x', d3.forceX(width / 2).strength(0.05))
      .force('y', d3.forceY(height / 2).strength(0.05));

    simulationRef.current = simulation;

    // Gradients
    const defs = svg.append('defs');
    linkData.forEach((d, i) => {
      const gradient = defs.append('linearGradient')
        .attr('id', `grad-${i}`)
        .attr('gradientUnits', 'userSpaceOnUse');
      
      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', '#4a5568')
        .attr('stop-opacity', 0.2);
      gradient.append('stop')
        .attr('offset', '50%')
        .attr('stop-color', '#4a5568')
        .attr('stop-opacity', weightScale(d.weight || 0.5));
      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', '#4a5568')
        .attr('stop-opacity', 0.2);
    });

    // Links
    const link = g.append('g')
      .selectAll('line')
      .data(linkData)
      .enter().append('line')
      .attr('stroke', (d, i) => `url(#grad-${i})`)
      .attr('stroke-width', d => Math.max(1, (d.weight || 0.5) * 4))
      .style('pointer-events', 'none');

    // Link labels
    const linkLabels = g.append('g')
      .selectAll('text')
      .data(linkData.filter(d => d.weight > 0.5))
      .enter().append('text')
      .text(d => d.weight.toFixed(2))
      .style('font-size', '10px')
      .style('fill', '#6b7280')
      .style('text-anchor', 'middle')
      .style('pointer-events', 'none');

    // Nodes
    const node = g.append('g')
      .selectAll('g')
      .data(nodeData)
      .enter().append('g')
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    node.append('circle')
      .attr('r', d => d.id === String(selectedNote) ? 12 : 10)
      .attr('fill', d => {
        if (d.id === String(selectedNote)) return '#10b981';
        if (d.cluster !== undefined) return COLORS[d.cluster % COLORS.length];
        return COLORS[0];
      })
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 2)
      .style('filter', 'drop-shadow(0 4px 6px rgba(0, 0, 0, 0.3))');

    node.append('text')
      .text(d => d.label || `Note ${d.id}`)
      .attr('y', -15)
      .style('font-size', '8px')
      .style('font-weight', '500')
      .style('fill', '#e5e7eb')
      .style('text-anchor', 'middle')
      .style('pointer-events', 'none')
      .style('filter', 'drop-shadow(0 1px 2px rgba(0, 0, 0, 0.8))');

    node.on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick(parseInt(d.id));
      })
      .on('mouseenter', (event, d) => {
        setHoveredNode(d);
        d3.select(event.currentTarget).select('circle')
          .transition().duration(200)
          .attr('r', 14)
          .attr('stroke-width', 3);
      })
      .on('mouseleave', (event, d) => {
        setHoveredNode(null);
        d3.select(event.currentTarget).select('circle')
          .transition().duration(200)
          .attr('r', d.id === String(selectedNote) ? 12 : 10)
          .attr('stroke-width', 2);
      });

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      linkLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      if (!event.sourceEvent.shiftKey) {
        d.fx = null;
        d.fy = null;
      }
    }

    return () => simulation.stop();
  }, [graphData, selectedNote, width, height, onNodeClick]);

  const handleResetView = () => {
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom().scaleExtent([0.1, 10]);
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
  };

  const handleToggleSimulation = () => {
    if (simulationRef.current) {
      if (isRunning) {
        simulationRef.current.stop();
      } else {
        simulationRef.current.alpha(0.3).restart();
      }
      setIsRunning(!isRunning);
    }
  };

  const handleOpenExport = () => {
    try {
      const t = d3.zoomTransform(svgRef.current);
      setExportTransform({ x: t.x, y: t.y, k: t.k });
    } catch {
      setExportTransform({ x: 0, y: 0, k: 1 });
    }
    setExportOpen(true);
  };

  const handleNotify = (msg) => {
    setToastMessage(msg);
    setToastOpen(true);
  };

  return (
    <div className="graph-visualization">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{
          background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
          borderRadius: '8px'
        }}
      />

      <GraphControlsPanel
        params={controlsParams}
        onChange={onControlsChange}
        onReset={onControlsReset}
        stats={stats}
        loading={loading}
        position={panelPosition}
      />
      
      <div className="graph-controls">
        <button onClick={handleResetView} className="control-btn" title="Reset View">
          ⟲
        </button>
        <button onClick={handleToggleSimulation} className="control-btn" title="Toggle Physics">
          {isRunning ? '❚❚' : '▶'}
        </button>
        <button onClick={handleOpenExport} className="control-btn" title="Export Graph">
          ⤓
        </button>
      </div>

      {hoveredNode && (
        <div className="node-tooltip">
          <div className="tooltip-label">{hoveredNode.label}</div>
          <div className="tooltip-meta">
            ID: {hoveredNode.id}
            {hoveredNode.cluster !== undefined && ` • Cluster: ${hoveredNode.cluster}`}
          </div>
        </div>
      )}

      <ExportGraphModal
        isOpen={exportOpen}
        onClose={() => setExportOpen(false)}
        svgRef={svgRef}
        graphData={graphData}
        params={controlsParams}
        transform={exportTransform}
        onNotify={handleNotify}
      />

      <ToastNotification
        isOpen={toastOpen}
        message={toastMessage}
        onClose={() => setToastOpen(false)}
        duration={4000}
      />
    </div>
  );
}
```


### 📄 frontend\src\components\ImportConfirmModal.jsx

```
import React, { useEffect } from 'react';

export default function ImportConfirmModal({ isOpen, count = 0, onReplace, onMerge, onCancel }) {
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e) => {
      if (e.key === 'Escape') onCancel?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onCancel]);

  if (!isOpen) return null;

  const stop = (e) => e.stopPropagation();

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" role="dialog" aria-modal="true" onClick={stop}>
        <div className="modal-header">
          <h3>Import Notes</h3>
        </div>
        <div className="modal-body">
          <p>Detected {count} notes in the selected file.</p>
          <p>How would you like to import them?</p>
          <ul className="modal-list">
            <li><strong>Replace all notes</strong> — clears current notes and loads the imported ones.</li>
            <li><strong>Merge with existing</strong> — keeps current notes and adds imported notes. Conflicting IDs will be regenerated.</li>
          </ul>
        </div>
        <div className="modal-actions">
          <button className="btn btn-danger" onClick={onReplace} title="Replace current notes">
            Replace All
          </button>
          <button className="btn btn-primary" onClick={onMerge} title="Merge imported notes">
            Merge
          </button>
          <button className="btn btn-secondary" onClick={onCancel} title="Cancel">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 frontend\src\components\ImportLocalNotesModal.jsx

```
import { useState, useEffect } from 'react';
import apiService from '../services/api';

export default function ImportLocalNotesModal({ onClose, onImportComplete }) {
  const [localNotes, setLocalNotes] = useState([]);
  const [localTrash, setLocalTrash] = useState([]);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    // Check for localStorage notes
    const notesData = localStorage.getItem('semantic-notes-data');
    const trashData = localStorage.getItem('semantic-notes-trash');
    
    if (notesData) {
      try {
        const parsed = JSON.parse(notesData);
        setLocalNotes(parsed.notes || []);
      } catch (e) {
        console.error('Failed to parse notes:', e);
      }
    }
    
    if (trashData) {
      try {
        const parsed = JSON.parse(trashData);
        setLocalTrash(parsed.trash || []);
      } catch (e) {
        console.error('Failed to parse trash:', e);
      }
    }
  }, []);

  const handleImport = async () => {
    setImporting(true);
    setError('');
    
    try {
      const response = await apiService.request('/api/notes/import', {
        method: 'POST',
        body: JSON.stringify({
          notes: localNotes,
          trash: localTrash
        })
      });
      
      setSuccess(true);
      setTimeout(() => {
        onImportComplete(response.imported);
      }, 1500);
    } catch (err) {
      setError(err.message || 'Import failed. Please try again.');
      setImporting(false);
    }
  };

  const handleSkip = () => {
    onClose();
  };

  const totalToImport = localNotes.length + localTrash.length;

  if (totalToImport === 0) {
    return null; // Don't show modal if no notes to import
  }

  return (
    <div className="modal-overlay">
      <div className="modal-content import-modal">
        <h2>Import Your Local Notes</h2>
        
        {!success ? (
          <>
            <p>
              We found <strong>{localNotes.length} notes</strong>
              {localTrash.length > 0 && ` and ${localTrash.length} items in trash`} 
              {' '}stored locally on this device.
            </p>
            
            <p>
              Would you like to import them to your account? This will:
            </p>
            
            <ul className="import-benefits">
              <li>✓ Sync your notes across devices</li>
              <li>✓ Keep your data safe in the cloud</li>
              <li>✓ Preserve all timestamps and tags</li>
              <li>✓ Maintain your local cache for speed</li>
            </ul>
            
            {error && (
              <div className="error-message">{error}</div>
            )}
            
            <div className="modal-actions">
              <button
                onClick={handleImport}
                disabled={importing}
                className="btn-primary"
              >
                {importing ? 'Importing...' : `Import ${totalToImport} Notes`}
              </button>
              <button
                onClick={handleSkip}
                disabled={importing}
                className="btn-secondary"
              >
                Skip for Now
              </button>
            </div>
            
            <p className="import-note">
              Note: Your local notes will remain available even if you skip.
            </p>
          </>
        ) : (
          <div className="success-message">
            <div className="success-icon">✓</div>
            <p>Successfully imported {totalToImport} notes!</p>
            <p>Redirecting...</p>
          </div>
        )}
      </div>
    </div>
  );
}
```


### 📄 frontend\src\components\LoginForm.css

```css
/* Styling for the login form */
.login-form-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background: #f4f6f8;
  font-family: Arial, Helvetica, sans-serif;
}

.login-form {
  background: #ffffff;
  padding: 2rem 2.5rem;
  border-radius: 8px;
  box-shadow: 0 3px 12px rgba(0, 0, 0, 0.15);
  width: 100%;
  max-width: 400px;
}

.login-form h2 {
  margin-bottom: 1.5rem;
  text-align: center;
  color: #333;
}

.login-form label {
  display: block;
  margin-bottom: 0.5rem;
  color: #555;
  font-weight: 600;
}

.login-form input {
  width: 100%;
  padding: 0.75rem;
  border: 1px solid #ccc;
  border-radius: 5px;
  margin-bottom: 1.25rem;
  font-size: 1rem;
}

.login-form input:focus {
  border-color: #1976d2;
  outline: none;
  box-shadow: 0 0 0 2px rgba(25, 118, 210, 0.2);
}

.login-form button {
  width: 100%;
  padding: 0.8rem;
  background-color: #1976d2;
  color: white;
  font-size: 1rem;
  font-weight: bold;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.25s;
}

.login-form button:hover {
  background-color: #125ca1;
}

.login-form .error {
  color: #d32f2f;
  font-size: 0.9rem;
  margin-bottom: 1rem;
  text-align: center;
}
```


### 📄 frontend\src\components\LoginForm.jsx

```
import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import './LoginForm.css';

export default function LoginForm() {
  const [mode, setMode] = useState('login'); // 'login' or 'register'
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const { login, register } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (username.length < 3) {
      setError('Username must be at least 3 characters');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }
    
    setLoading(true);
    try {
      if (mode === 'login') {
        await login(username, password);
      } else {
        await register(username, password, email);
      }
    } catch (err) {
      let message = 'Authentication failed';
      if (err instanceof Error) {
        message = err.message;
      } else if (typeof err === 'string') {
        message = err;
      } else if (err && err.detail) {
        message = err.detail;
      }
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      backgroundColor: '#f8f8f8',
    }}>
      <form onSubmit={handleSubmit} style={{
        background: '#fff',
        padding: '2rem',
        borderRadius: '8px',
        boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
        maxWidth: '400px',
        width: '100%',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <h2 style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          {mode === 'login' ? 'Login' : 'Register'}
        </h2>

        <label>Username</label>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          style={{ marginBottom: '1rem', padding: '0.5rem' }}
        />

        {mode === 'register' && (
          <>
            <label>Email (optional)</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={{ marginBottom: '1rem', padding: '0.5rem' }}
            />
          </>
        )}

        <label>Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          style={{ marginBottom: '1rem', padding: '0.5rem' }}
        />

        {error && (
          <div style={{ color: 'red', marginBottom: '1rem', textAlign: 'center' }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            padding: '0.75rem',
            backgroundColor: '#007bff',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '1rem'
          }}
        >
          {loading ? 'Processing...' : mode === 'login' ? 'Login' : 'Register'}
        </button>

        <p style={{ textAlign: 'center', marginTop: '1rem' }}>
          {mode === 'login' ? (
            <>
              Don't have an account?{' '}
              <button
                type="button"
                onClick={() => setMode('register')}
                style={{ color: '#007bff', border: 'none', background: 'none', cursor: 'pointer' }}
              >
                Register
              </button>
            </>
          ) : (
            <>
              Already have an account?{' '}
              <button
                type="button"
                onClick={() => setMode('login')}
                style={{ color: '#007bff', border: 'none', background: 'none', cursor: 'pointer' }}
              >
                Login
              </button>
            </>
          )}
        </p>
      </form>
    </div>
  );
}
```


### 📄 frontend\src\components\MarkdownCheatsheet.jsx

```
import React from 'react';

function Row({ label, example }) {
  return (
    <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.5rem' }}>
      <div style={{ minWidth: 140, color: 'var(--text-muted)', fontWeight: 500 }}>{label}</div>
      <pre style={{
        margin: 0,
        padding: '0.5rem 0.75rem',
        background: 'var(--bg-primary)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        overflowX: 'auto',
        fontSize: '0.8125rem',
        lineHeight: 1.5
      }}>
        <code>{example}</code>
      </pre>
    </div>
  );
}

export default function MarkdownCheatsheet({ isOpen, onClose }) {
  if (!isOpen) return null;
  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="md-cheatsheet-title">
      <div className="modal">
        <div className="modal-header">
          <h3 id="md-cheatsheet-title">Markdown Cheatsheet</h3>
        </div>

        <div className="modal-body">
          <p style={{ marginBottom: '0.75rem' }}>
            Quick reference for common Markdown syntax supported in the editor and preview.
          </p>

          <Row label="Headers" example={
`# H1
## H2
### H3
#### H4
##### H5
###### H6`
          } />

          <Row label="Bold / Italic" example={
`**bold** or __bold__
*italic* or _italic_
~~strikethrough~~`
          } />

          <Row label="Lists" example={
`- item A
- item B
  - nested
1. First
2. Second`
          } />

          <Row label="Links / Images" example={
`[OpenAI](https://openai.com)
![Alt text](https://placehold.co/200x100)`
          } />

          <Row label="Inline code" example={
"`code` with backticks"
          } />

          <Row label="Code block" example={
"```js\nfunction hello(name) {\n  console.log('Hello ' + name);\n}\n```"
          } />

          <Row label="Blockquote" example={
`> A wise quote
> - Author`
          } />

          <Row label="Tables (GFM)" example={
`| Name | Role  |
|-----:|:-----:|
| Alice| Admin |
| Bob  | User  |`
          } />
        </div>

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 frontend\src\components\MarkdownPreview.jsx

```
import React, { forwardRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';

const MarkdownPreview = forwardRef(function MarkdownPreview({ content, className = '', style, ...divProps }, ref) {
  return (
    <div ref={ref} className={`markdown-preview ${className}`} style={style} {...divProps}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          a: ({ node, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
          code({ node, inline, className, children, ...props }) {
            const text = String(children).replace(/\n$/, '');
            if (inline) {
              return (
                <code className={`inline-code ${className || ''}`.trim()} {...props}>
                  {text}
                </code>
              );
            }
            return (
              <pre className={`code-block ${className || ''}`.trim()}>
                <code {...props}>{text}</code>
              </pre>
            );
          },
          img: ({ node, ...props }) => <img {...props} loading="lazy" />,
        }}
      >
        {content || ''}
      </ReactMarkdown>
    </div>
  );
});

export default MarkdownPreview;
```


### 📄 frontend\src\components\NoteEditor.jsx

```
import React, { useState, useEffect, useMemo, useRef, forwardRef, useImperativeHandle } from 'react';
import MarkdownPreview from './MarkdownPreview';
import MarkdownCheatsheet from './MarkdownCheatsheet';

export default forwardRef(function NoteEditor({ note, onSave, onCancel, onDirtyChange, onFindSimilar }, ref) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [tags, setTags] = useState('');
  const [original, setOriginal] = useState({ title: '', content: '', tags: '' });

  // Markdown editor modes and helpers
  const [viewMode, setViewMode] = useState('edit'); // 'edit' | 'preview' | 'split'
  const [cheatsheetOpen, setCheatsheetOpen] = useState(false);
  const textareaRef = useRef(null);
  const previewRef = useRef(null);
  const syncingRef = useRef(false);
 
  // Load note into form and establish original snapshot
  useEffect(() => {
    if (note) {
      const t = note.title || '';
      const c = note.content || '';
      const g = note.tags || '';
      setTitle(t);
      setContent(c);
      setTags(g);
      setOriginal({ title: t, content: c, tags: g });
    } else {
      setTitle('');
      setContent('');
      setTags('');
      setOriginal({ title: '', content: '', tags: '' });
    }
  }, [note]);

  // Dirty detection
  const isDirty = useMemo(() => {
    return title !== original.title || content !== original.content || tags !== original.tags;
  }, [title, content, tags, original]);

  // Notify parent when dirty state changes
  const lastDirty = useRef(isDirty);
  useEffect(() => {
    if (lastDirty.current !== isDirty) {
      lastDirty.current = isDirty;
      if (typeof onDirtyChange === 'function') onDirtyChange(isDirty);
    }
  }, [isDirty, onDirtyChange]);

  // Warn when trying to close/refresh tab with unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (!isDirty) return;
      e.preventDefault();
      e.returnValue = '';
      return '';
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty]);

  const buildNoteData = () => {
    const noteData = {
      ...note,
      title: title.trim(),
      content: content.trim(),
      tags: tags.trim(),
      updatedAt: new Date().toISOString()
    };
    if (!note?.id) {
      noteData.createdAt = new Date().toISOString();
    }
    return noteData;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!title.trim()) {
      alert('Title is required');
      return;
    }
    if (!content.trim()) {
      alert('Content is required');
      return;
    }

    const noteData = buildNoteData();
    onSave(noteData);
    // Mark clean after successful save
    setOriginal({ title: noteData.title, content: noteData.content, tags: noteData.tags });
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && (e.key === 'Enter' || e.key.toLowerCase() === 's')) {
      e.preventDefault();
      handleSubmit(e);
    } else if (e.key === 'Escape') {
      onCancel();
    }
  };

  // Expose imperative API to parent (App) for Save & Continue flow
  useImperativeHandle(ref, () => ({
    isDirty: () => isDirty,
    getCurrentData: () => buildNoteData(),
    submit: () => {
      const fakeEvt = { preventDefault: () => {} };
      handleSubmit(fakeEvt);
    }
  }), [isDirty, title, content, tags, note]);

  // Scroll sync between editor and preview
  const syncScroll = (from) => {
    if (syncingRef.current) return;
    const ta = textareaRef.current;
    const pv = previewRef.current;
    if (!ta || !pv) return;

    const src = from === 'textarea' ? ta : pv;
    const dst = from === 'textarea' ? pv : ta;

    const srcScrollable = Math.max(1, src.scrollHeight - src.clientHeight);
    const ratio = src.scrollTop / srcScrollable;
    const dstScrollable = Math.max(1, dst.scrollHeight - dst.clientHeight);
    syncingRef.current = true;
    try {
      dst.scrollTop = ratio * dstScrollable;
    } finally {
      // release lock on next tick to avoid feedback loop
      setTimeout(() => { syncingRef.current = false; }, 0);
    }
  };

  const handleTextareaScroll = () => syncScroll('textarea');
  const handlePreviewScroll = () => syncScroll('preview');

  const isEditing = note?.id !== undefined;
 
  return (
    <div className="note-editor">
      <div className="editor-header">
        <h2>{isEditing ? 'Edit Note' : 'New Note'}</h2>
        {isDirty && (
          <div className="unsaved-indicator">
            <span className="unsaved-dot" />
            <span>Unsaved changes</span>
          </div>
        )}
      </div>
      
      <form onSubmit={handleSubmit} className="editor-form">
        <div className="form-group">
          <label className="form-label">Title</label>
          <input
            type="text"
            placeholder="Note title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={handleKeyDown}
            className="form-input"
            autoFocus
          />
        </div>

        <div className="form-group" style={{ flex: 1 }}>
          <div className="content-toolbar">
            <label className="form-label">Content</label>
            <div className="content-toolbar-actions">
              <div className="toggle-switch" role="group" aria-label="Editor mode">
                <button
                  type="button"
                  className={`toggle-btn ${viewMode === 'edit' ? 'active' : ''}`}
                  aria-pressed={viewMode === 'edit'}
                  onClick={() => setViewMode('edit')}
                  title="Edit markdown"
                >
                  Edit
                </button>
                <button
                  type="button"
                  className={`toggle-btn ${viewMode === 'preview' ? 'active' : ''}`}
                  aria-pressed={viewMode === 'preview'}
                  onClick={() => setViewMode('preview')}
                  title="Preview formatted markdown"
                >
                  Preview
                </button>
                <button
                  type="button"
                  className={`toggle-btn ${viewMode === 'split' ? 'active' : ''}`}
                  aria-pressed={viewMode === 'split'}
                  onClick={() => setViewMode('split')}
                  title="Edit and preview side-by-side"
                >
                  Split
                </button>
              </div>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setCheatsheetOpen(true)}
                title="Markdown syntax help"
              >
                Cheatsheet
              </button>
            </div>
          </div>

          {viewMode === 'edit' && (
            <textarea
              ref={textareaRef}
              placeholder="Write your note here..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
              onKeyDown={handleKeyDown}
              onScroll={handleTextareaScroll}
              className="form-input form-textarea"
            />
          )}

          {viewMode === 'preview' && (
            <MarkdownPreview
              ref={previewRef}
              content={content}
              className="form-input markdown-preview-only"
              style={{ minHeight: 300, overflow: 'auto' }}
              onScroll={handlePreviewScroll}
            />
          )}

          {viewMode === 'split' && (
            <div className="split-container">
              <div className="split-pane split-pane-editor">
                <textarea
                  ref={textareaRef}
                  placeholder="Write your note here..."
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onScroll={handleTextareaScroll}
                  className="form-input form-textarea"
                />
              </div>
              <div className="split-pane split-pane-preview">
                <MarkdownPreview
                  ref={previewRef}
                  content={content}
                  className="markdown-pane"
                  style={{ height: '100%', overflow: 'auto' }}
                  onScroll={handlePreviewScroll}
                />
              </div>
            </div>
          )}

          <div className="char-count">
            {content.length} characters
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Tags</label>
          <input
            type="text"
            placeholder="comma, separated, tags"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            onKeyDown={handleKeyDown}
            className="form-input"
          />
        </div>

        <div className="editor-actions">
          <button type="submit" className="btn btn-primary">
            {isEditing ? 'Update' : 'Create'}
          </button>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onFindSimilar && onFindSimilar()}
            className="btn btn-secondary"
            title="Find notes similar to this one"
          >
            Find Similar
          </button>
          <span className="keyboard-hint">
            Ctrl+Enter or Ctrl+S to save • Esc to cancel
          </span>
        </div>
      </form>

      <MarkdownCheatsheet
        isOpen={cheatsheetOpen}
        onClose={() => setCheatsheetOpen(false)}
      />
    </div>
  );
});
```


### 📄 frontend\src\components\NotesList.jsx

```
import React, { useState, useMemo } from 'react';

const PREVIEW_LENGTH = 120;

function formatRelativeTime(dateString) {
  if (!dateString) return '';
  
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  
  return date.toLocaleDateString();
}

function extractTags(notes) {
  const tagSet = new Set();
  notes.forEach(note => {
    if (note.tags) {
      note.tags.split(',').forEach(tag => {
        const trimmed = tag.trim();
        if (trimmed) tagSet.add(trimmed);
      });
    }
  });
  return Array.from(tagSet).sort();
}

export default function NotesList({
  notes,
  onSelect,
  onEdit,
  onDelete,
  selectedNote,
  searchTerm = '',
  onFindSimilar,
  searchMode = 'text',
  semanticResults = [],
  minSimilarity = 60,
  semanticLoading = false,
  semanticError = ''
}) {
  const [sortBy, setSortBy] = useState('updated');
  const [filterTag, setFilterTag] = useState('');

  // Map from note index to semantic result for quick lookup
  const semanticMap = useMemo(() => {
    const m = new Map();
    (semanticResults || []).forEach(r => {
      if (r && typeof r.index === 'number') m.set(r.index, r);
    });
    return m;
  }, [semanticResults]);

  // Compute a simple "why matched" snippet by finding the sentence with most token overlap
  function bestWhySnippet(text, query) {
    const content = String(text || '');
    const q = String(query || '').toLowerCase();
    if (!q) return '';
    const sentences = content.split(/(?<=[.!?])\s+/);
    const qTokens = new Set(q.split(/\W+/).filter(Boolean));
    let best = '';
    let bestScore = -1;
    for (const s of sentences) {
      const tokens = s.toLowerCase().split(/\W+/).filter(Boolean);
      if (tokens.length === 0) continue;
      let overlap = 0;
      for (const t of tokens) if (qTokens.has(t)) overlap++;
      const score = overlap / Math.max(1, tokens.length);
      if (score > bestScore) {
        bestScore = score;
        best = s;
      }
    }
    return best || sentences[0] || content.substring(0, PREVIEW_LENGTH);
  }

  const allTags = useMemo(() => extractTags(notes), [notes]);

  const processedNotes = useMemo(() => {
    // Semantic mode: build from semanticResults to preserve relevance ordering
    if (searchMode === 'semantic') {
      let arr = (semanticResults || []).map(r => {
        const n = notes[r.index];
        if (!n) return null;
        return { ...n, originalIndex: r.index, _sem: r };
      }).filter(Boolean);

      // Threshold filter
      arr = arr.filter(item => (item._sem?.percent ?? 0) >= minSimilarity);

      // Tag filter (keep intersection with selected tag)
      if (filterTag) {
        arr = arr.filter(note => note.tags?.includes(filterTag));
      }

      // Keep relevance ordering (semanticResults already sorted)
      return arr;
    }

    // Text mode: legacy keyword search
    let filtered = notes.map((note, index) => ({ ...note, originalIndex: index }));

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(note =>
        note.title.toLowerCase().includes(term) ||
        note.content.toLowerCase().includes(term) ||
        (note.tags && note.tags.toLowerCase().includes(term))
      );
    }

    if (filterTag) {
      filtered = filtered.filter(note => note.tags?.includes(filterTag));
    }

    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'updated':
          return new Date(b.updatedAt || b.createdAt || 0) -
                 new Date(a.updatedAt || a.createdAt || 0);
        case 'created':
          return new Date(b.createdAt || 0) - new Date(a.createdAt || 0);
        case 'title':
          return a.title.localeCompare(b.title);
        default:
          return 0;
      }
    });

    return filtered;
  }, [notes, searchTerm, filterTag, sortBy, searchMode, semanticResults, minSimilarity]);

  const handleDelete = (index, e) => {
    e.stopPropagation();
    onDelete(index);
  };

  return (
    <div className="notes-list">
      <div className="list-header">
        <h3>Notes ({processedNotes.length})</h3>
        
        <div className="list-controls">
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="form-select"
          >
            <option value="updated">Recent</option>
            <option value="created">Created</option>
            <option value="title">Title</option>
          </select>

          {allTags.length > 0 && (
            <select
              value={filterTag}
              onChange={(e) => setFilterTag(e.target.value)}
              className="form-select"
            >
              <option value="">All Tags</option>
              {allTags.map(tag => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
          )}
        </div>
      </div>

      {processedNotes.length === 0 ? (
        <div className="empty-message">
          {searchMode === 'semantic' ? (
            <>
              <div className="empty-message-title">No semantic matches</div>
              <div className="empty-message-hint">
                {semanticError ? semanticError : 'Try lowering the similarity threshold or refining your query'}
              </div>
            </>
          ) : notes.length === 0 ? (
            <>
              <div className="empty-message-title">No notes yet</div>
              <div className="empty-message-hint">Create your first note to get started</div>
            </>
          ) : (
            <>
              <div className="empty-message-title">No matches</div>
              <div className="empty-message-hint">Try different search terms or filters</div>
            </>
          )}
        </div>
      ) : (
        <div className="notes-items">
          {processedNotes.map((note) => (
            <div
              key={note.originalIndex}
              className={`note-item ${selectedNote === note.originalIndex ? 'selected' : ''}`}
              onClick={() => onSelect(note.originalIndex)}
            >
              <div className="note-item-content">
                <div className="note-item-header">
                  <h4 className="note-item-title">{note.title}</h4>
                  {searchMode === 'semantic' && note._sem && (
                    <span className="similarity-badge">{note._sem.percent}% match</span>
                  )}
                  <span className="note-date">
                    {formatRelativeTime(note.updatedAt || note.createdAt)}
                  </span>
                </div>
                
                {searchMode === 'semantic' ? (
                  <div className="why-match">
                    {bestWhySnippet(note.content, searchTerm)}
                  </div>
                ) : (
                  <p className="note-preview">
                    {note.content.substring(0, PREVIEW_LENGTH)}
                    {note.content.length > PREVIEW_LENGTH && '...'}
                  </p>
                )}
                
                {note.tags && (
                  <div className="note-tags">
                    {note.tags.split(',').map((tag, i) => (
                      <span key={i} className="tag">
                        {tag.trim()}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              
              <div className="note-item-actions">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onEdit(note.originalIndex);
                  }}
                  className="action-btn"
                  title="Edit"
                >
                  Edit
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onFindSimilar && onFindSimilar(note.originalIndex);
                  }}
                  className="action-btn"
                  title="Find Similar"
                >
                  Similar
                </button>
                <button
                  onClick={(e) => handleDelete(note.originalIndex, e)}
                  className="action-btn delete"
                  title="Delete"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```


### 📄 frontend\src\components\SimilarNotesModal.jsx

```
import React, { useEffect, useState } from 'react';
import apiService from '../services/api';

export default function SimilarNotesModal({
  isOpen = false,
  baseDoc = '',
  baseTitle = 'This note',
  notes = [],
  excludeIndex = null,
  topK = 8,
  onClose,
  onSelect,
  onLink
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState([]);

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError('');
      try {
        if (!baseDoc?.trim()) {
          setResults([]);
          setError('Note content is empty.');
          return;
        }
        const [embs, baseEmb] = await Promise.all([
          apiService.getEmbeddingsForNotes(notes),
          apiService.embedText(baseDoc),
        ]);
        const scored = [];
        for (let i = 0; i < notes.length; i++) {
          if (excludeIndex != null && i === excludeIndex) continue;
          const v = embs[i];
          if (!Array.isArray(v)) continue;
          const score = apiService.cosineSimilarity(baseEmb, v);
          scored.push({
            index: i,
            score,
            percent: Math.round(score * 100),
            title: notes[i]?.title || '(Untitled)',
            preview: (notes[i]?.content || '').substring(0, 160)
          });
        }
        scored.sort((a, b) => b.score - a.score);
        const top = scored.slice(0, topK);
        if (!cancelled) setResults(top);
      } catch (e) {
        console.error('Find similar failed:', e);
        if (!cancelled) setError(e?.message || 'Failed to compute similarities');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [isOpen, baseDoc, notes, excludeIndex, topK]);

  if (!isOpen) return null;

  const stop = (e) => e.stopPropagation();

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" onClick={stop}>
        <div className="modal-header">
          <h3>Notes similar to “{baseTitle || 'Untitled'}”</h3>
        </div>
        <div className="modal-body">
          {loading ? (
            <div className="loading" style={{ position: 'static', transform: 'none', padding: 0 }}>
              <div className="loading-spinner" />
              <div>Computing similarities...</div>
            </div>
          ) : error ? (
            <div className="error-banner" style={{ position: 'static' }}>
              ⚠ {error}
            </div>
          ) : results.length === 0 ? (
            <div className="empty-message">
              <div className="empty-message-title">No similar notes found</div>
              <div className="empty-message-hint">Try adding more content or different keywords</div>
            </div>
          ) : (
            <div className="similar-list">
              {results.map((r) => (
                <div key={r.index} className="similar-item">
                  <div
                    className="similar-main"
                    onClick={() => onSelect && onSelect(r.index)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter') onSelect && onSelect(r.index); }}
                  >
                    <div className="similar-header">
                      <div className="similar-title">{r.title}</div>
                      <div className="similar-score">{r.percent}% similar</div>
                    </div>
                    <div className="similar-preview">
                      {r.preview}{(notes[r.index]?.content || '').length > r.preview.length ? '…' : ''}
                    </div>
                    <div className="similar-meter">
                      <div className="similar-meter-fill" style={{ width: `${Math.min(100, Math.max(0, r.percent))}%` }} />
                    </div>
                  </div>
                  <div className="similar-actions">
                    <button className="btn btn-secondary btn-sm" onClick={() => onSelect && onSelect(r.index)} title="Open note">Open</button>
                    <button className="btn btn-primary btn-sm" onClick={() => onLink && onLink(r.index)} title="Link notes">Link</button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 frontend\src\components\ToastNotification.jsx

```
import React, { useEffect, useRef } from 'react';

export default function ToastNotification({
  isOpen = false,
  message = '',
  actionLabel = 'Undo',
  onAction,
  onClose,
  duration = 5000
}) {
  const timerRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      if (onClose) onClose();
    }, duration);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isOpen, message, duration, onClose]);

  if (!isOpen) return null;

  return (
    <div className="toast-container" role="status" aria-live="polite">
      <div className="toast-card fade-in">
        <div className="toast-message">{message}</div>

        <div className="toast-actions">
          {onAction && (
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => {
                if (onAction) onAction();
                if (onClose) onClose();
              }}
            >
              {actionLabel}
            </button>
          )}
          <button
            className="toast-close"
            aria-label="Dismiss notification"
            title="Dismiss"
            onClick={() => onClose && onClose()}
          >
            ×
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 frontend\src\components\TrashView.jsx

```
import React, { useMemo, useState } from 'react';
import ConfirmDialog from './ConfirmDialog';

const PREVIEW_LENGTH = 120;

function formatRelativeTime(dateString) {
  if (!dateString) return '';
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString();
}

export default function TrashView({
  trashedNotes = [],
  onRestore,
  onDeleteForever,
  onEmptyTrash
}) {
  const [confirmId, setConfirmId] = useState(null);
  const [confirmEmpty, setConfirmEmpty] = useState(false);

  const items = useMemo(() => {
    // Sort by deletedAt desc
    return [...trashedNotes].sort((a, b) =>
      new Date(b.deletedAt || 0) - new Date(a.deletedAt || 0)
    );
  }, [trashedNotes]);

  return (
    <div className="trash-view">
      <div className="list-header">
        <h3>Trash ({items.length})</h3>
        <div className="list-controls">
          <button
            className="btn btn-danger"
            disabled={items.length === 0}
            onClick={() => setConfirmEmpty(true)}
            title="Permanently delete all trashed notes"
          >
            Empty Trash
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="empty-message">
          <div className="empty-message-title">Trash is empty</div>
          <div className="empty-message-hint">Deleted notes will appear here</div>
        </div>
      ) : (
        <div className="trash-items">
          {items.map(note => (
            <div key={note.id} className="trash-item">
              <div className="trash-item-content">
                <div className="trash-item-header">
                  <h4 className="note-item-title">{note.title}</h4>
                  <span className="note-date">Deleted {formatRelativeTime(note.deletedAt)}</span>
                </div>
                <p className="note-preview">
                  {note.content.substring(0, PREVIEW_LENGTH)}
                  {note.content.length > PREVIEW_LENGTH && '...'}
                </p>
                {note.tags && (
                  <div className="note-tags">
                    {note.tags.split(',').map((tag, i) => (
                      <span key={i} className="tag">{tag.trim()}</span>
                    ))}
                  </div>
                )}
              </div>

              <div className="note-item-actions">
                <button
                  className="action-btn"
                  title="Restore"
                  onClick={() => onRestore && onRestore(note.id)}
                >
                  Restore
                </button>
                <button
                  className="action-btn delete"
                  title="Delete Forever"
                  onClick={() => setConfirmId(note.id)}
                >
                  Delete Forever
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <ConfirmDialog
        isOpen={confirmId !== null}
        title="Delete Forever?"
        message="This will permanently delete the note from trash. This action cannot be undone."
        confirmLabel="Delete Forever"
        cancelLabel="Cancel"
        danger
        onConfirm={() => {
          if (confirmId != null && onDeleteForever) {
            onDeleteForever(confirmId);
          }
          setConfirmId(null);
        }}
        onCancel={() => setConfirmId(null)}
      />

      <ConfirmDialog
        isOpen={confirmEmpty}
        title="Empty Trash?"
        message="All notes in the trash will be permanently deleted. This action cannot be undone."
        confirmLabel="Empty Trash"
        cancelLabel="Cancel"
        danger
        onConfirm={() => {
          if (onEmptyTrash) onEmptyTrash();
          setConfirmEmpty(false);
        }}
        onCancel={() => setConfirmEmpty(false)}
      />
    </div>
  );
}
```


### 📄 frontend\src\components\UnsavedChangesDialog.jsx

```
import React, { useEffect } from 'react';

export default function UnsavedChangesDialog({ 
  isOpen = false,
  onSaveAndContinue,
  onDiscard,
  onCancel
}) {
  if (!isOpen) return null;

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel && onCancel();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        onSaveAndContinue && onSaveAndContinue();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        onSaveAndContinue && onSaveAndContinue();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onCancel, onSaveAndContinue]);

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="unsaved-title">
      <div className="modal">
        <div className="modal-header">
          <h3 id="unsaved-title">Unsaved changes</h3>
        </div>
        <div className="modal-body">
          <p>You have unsaved changes. If you continue without saving, your edits will be lost.</p>
        </div>
        <div className="modal-actions">
          <button
            className="btn btn-primary"
            onClick={() => onSaveAndContinue && onSaveAndContinue()}
            autoFocus
          >
            Save & Continue
          </button>
          <button
            className="btn btn-danger"
            onClick={() => onDiscard && onDiscard()}
          >
            Discard
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => onCancel && onCancel()}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 frontend\src\contexts\AuthContext.jsx

```
import { createContext, useContext, useState, useEffect } from 'react';
import apiService from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const validateAndRestoreSession = async () => {
      const storedToken = localStorage.getItem('auth_token');
      if (!storedToken) {
        setLoading(false);
        return;
      }

      apiService.setAuthToken(storedToken);

      try {
        const userResponse = await apiService.request('/api/auth/me');
        if (userResponse && userResponse.username) {
          setUser({ username: userResponse.username, userId: userResponse.user_id });
          setToken(storedToken);
          setIsAuthenticated(true);
        } else {
          throw new Error('Invalid token');
        }
      } catch (error) {
        console.error('Token validation failed:', error);
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');
        setIsAuthenticated(false);
      } finally {
        setLoading(false);
      }
    };

    validateAndRestoreSession();
  }, []);

  const login = async (username, password) => {
    const response = await apiService.login(username, password);
    const { access_token, username: user_name, user_id } = response;
    
    setToken(access_token);
    setUser({ username: user_name, userId: user_id });
    setIsAuthenticated(true);
    
    localStorage.setItem('auth_token', access_token);
    localStorage.setItem('auth_user', JSON.stringify({ username: user_name, userId: user_id }));
    apiService.setAuthToken(access_token);
    
    return response;
  };

  const register = async (username, password, email) => {
    const response = await apiService.register(username, password, email);
    const { access_token, username: user_name, user_id } = response;
    
    setToken(access_token);
    setUser({ username: user_name, userId: user_id });
    setIsAuthenticated(true);
    
    localStorage.setItem('auth_token', access_token);
    localStorage.setItem('auth_user', JSON.stringify({ username: user_name, userId: user_id }));
    apiService.setAuthToken(access_token);
    
    return response;
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    setIsAuthenticated(false);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    apiService.setAuthToken(null);
  };

  const value = {
    user,
    token,
    isAuthenticated,
    loading,
    login,
    register,
    logout
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
```


### 📄 frontend\src\hooks\useNotes.js

```javascript
import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import dbApi from '../services/dbApi';

const STORAGE_KEY = 'semantic-notes-data';
const TRASH_STORAGE_KEY = 'semantic-notes-trash';

function loadFromStorage() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return parsed.notes || [];
    }
  } catch (err) {
    console.error('Failed to load notes:', err);
  }
  return [];
}

function saveToStorage(notes) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      notes,
      lastUpdated: new Date().toISOString()
    }));
  } catch (err) {
    console.error('Failed to save notes:', err);
    throw new Error('Storage quota exceeded');
  }
}

function loadTrashFromStorage() {
  try {
    const stored = localStorage.getItem(TRASH_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return parsed.trash || [];
    }
  } catch (err) {
    console.error('Failed to load trash:', err);
  }
  return [];
}

function saveTrashToStorage(trash) {
  try {
    localStorage.setItem(TRASH_STORAGE_KEY, JSON.stringify({
      trash,
      lastUpdated: new Date().toISOString()
    }));
  } catch (err) {
    console.error('Failed to save trash:', err);
    throw new Error('Storage quota exceeded');
  }
}

function generateRandomId() {
  return Date.now() + Math.floor(Math.random() * 100000);
}

function normalizeNote(n) {
  if (typeof n !== 'object' || n === null) {
    throw new Error('Invalid note format');
  }
  const title = String(n.title ?? '').trim();
  const content = String(n.content ?? '').trim();
  if (!title || !content) {
    throw new Error('Each note must have title and content');
  }
  const tags = n.tags != null ? String(n.tags) : '';
  let id = n.id;
  if (id == null || (typeof id !== 'number' && typeof id !== 'string') || !Number.isFinite(Number(id))) {
    id = generateRandomId();
  } else {
    id = Number(id);
  }
  const createdAt = n.createdAt ? new Date(n.createdAt).toISOString() : new Date().toISOString();
  const updatedAt = n.updatedAt ? new Date(n.updatedAt).toISOString() : createdAt;
  return { id, title, content, tags, createdAt, updatedAt };
}

export function useNotes() {
  const { isAuthenticated } = useAuth();
  const [notes, setNotes] = useState([]);
  const [trashedNotes, setTrashedNotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initNotes = async () => {
      if (isAuthenticated) {
        await loadNotesFromDatabase();
      } else {
        loadFromLocal();
      }
    };

    initNotes();
  }, [isAuthenticated]);

  const loadNotesFromDatabase = async () => {
    setLoading(true);
    setError(null);
    try {
      const dbNotes = await dbApi.fetchNotes();
      const dbTrash = await dbApi.fetchTrash();
      setNotes(dbNotes);
      setTrashedNotes(dbTrash);
      saveAllToStorage(dbNotes, dbTrash);
    } catch (err) {
      console.error('Failed to load from database, falling back to localStorage:', err);
      setError('Failed to sync with database');
      loadFromLocal();
    } finally {
      setLoading(false);
    }
  };

  const loadFromLocal = () => {
    const loadedNotes = loadFromStorage();
    const loadedTrash = loadTrashFromStorage();
    setNotes(loadedNotes);
    setTrashedNotes(loadedTrash);
    setLoading(false);
  };

  const saveAllToStorage = (notesData, trashData) => {
    try {
      saveToStorage(notesData);
      saveTrashToStorage(trashData);
    } catch (err) {
      console.error('Failed to cache notes locally:', err);
    }
  };

  useEffect(() => {
    if (!loading) {
      try {
        saveToStorage(notes);
        setError(null);
      } catch (err) {
        setError(err.message);
      }
    }
  }, [notes, loading]);

  useEffect(() => {
    if (!loading) {
      try {
        saveTrashToStorage(trashedNotes);
        setError(null);
      } catch (err) {
        setError(err.message);
      }
    }
  }, [trashedNotes, loading]);

  const addNote = useCallback(async (noteData) => {
    if (!isAuthenticated) {
      const newNote = {
        id: Date.now(),
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        ...noteData
      };
      setNotes(prev => [...prev, newNote]);
      return newNote;
    }

    try {
      const dbNote = await dbApi.createNote({
        title: noteData.title,
        content: noteData.content,
        tags: noteData.tags || ''
      });
      setNotes(prev => [dbNote, ...prev]);
      saveAllToStorage([dbNote, ...notes], trashedNotes);
      return dbNote;
    } catch (err) {
      console.error('Failed to create note:', err);
      setError('Failed to create note');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const updateNote = useCallback(async (index, noteData) => {
    if (!isAuthenticated) {
      setNotes(prev => {
        const updated = [...prev];
        updated[index] = {
          ...updated[index],
          ...noteData,
          updatedAt: new Date().toISOString()
        };
        return updated;
      });
      return;
    }
    const note = notes[index];
    if (!note) return;
    try {
      const updatedNote = await dbApi.updateNote(note.id, {
        title: noteData.title,
        content: noteData.content,
        tags: noteData.tags || ''
      });
      const newList = notes.map((n, i) => (i === index ? updatedNote : n));
      setNotes(newList);
      saveAllToStorage(newList, trashedNotes);
    } catch (err) {
      console.error('Failed to update note:', err);
      setError('Failed to update note');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const deleteNote = useCallback((index) => {
    setNotes(prev => prev.filter((_, i) => i !== index));
  }, []);

  // Trash management
  const moveToTrash = useCallback(async (index) => {
    const note = notes[index];
    if (!note) return;
    if (!isAuthenticated) {
      let movedNote = null;
      setNotes(prev => {
        const n = prev[index];
        movedNote = n;
        setTrashedNotes(trash => [
          { ...n, deletedAt: new Date().toISOString() },
          ...trash
        ]);
        return prev.filter((_, i) => i !== index);
      });
      return movedNote;
    }

    try {
      await dbApi.moveToTrash(note.id);
      const trashedNote = { ...note, is_deleted: true, deleted_at: new Date().toISOString() };
      setNotes(prev => prev.filter((_, i) => i !== index));
      setTrashedNotes(prev => [trashedNote, ...prev]);
      saveAllToStorage(notes.filter((_, i) => i !== index), [trashedNote, ...trashedNotes]);
      return trashedNote;
    } catch (err) {
      console.error('Failed to move to trash:', err);
      setError('Failed to move note to trash');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const restoreFromTrash = useCallback(async (id) => {
    if (!isAuthenticated) {
      let restored = null;
      setTrashedNotes(prev => {
        const idx = prev.findIndex(n => n.id === id);
        if (idx === -1) return prev;
        restored = { ...prev[idx] };
        const updatedTrash = prev.filter((_, i) => i !== idx);
        setNotes(notesPrev => [
          ...notesPrev,
          {
            id: restored.id,
            title: restored.title,
            content: restored.content,
            tags: restored.tags,
            createdAt: restored.createdAt,
            updatedAt: new Date().toISOString()
          }
        ]);
        return updatedTrash;
      });
      return restored;
    }

    try {
      await dbApi.restoreNote(id);
      const note = trashedNotes.find(n => n.id === id);
      if (!note) return;
      const restoredNote = { ...note, is_deleted: false, deleted_at: null };
      const updatedTrash = trashedNotes.filter(n => n.id !== id);
      setTrashedNotes(updatedTrash);
      setNotes(prev => [restoredNote, ...prev]);
      saveAllToStorage([restoredNote, ...notes], updatedTrash);
    } catch (err) {
      console.error('Failed to restore note:', err);
      setError('Failed to restore note');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const permanentDelete = useCallback(async (id) => {
    if (!isAuthenticated) {
      let deleted = false;
      setTrashedNotes(prev => {
        const next = prev.filter(n => {
          if (!deleted && n.id === id) deleted = true;
          return n.id !== id;
        });
        return next;
      });
      return deleted;
    }

    try {
      await dbApi.permanentDelete(id);
      const updatedTrash = trashedNotes.filter(n => n.id !== id);
      setTrashedNotes(updatedTrash);
      saveAllToStorage(notes, updatedTrash);
    } catch (err) {
      console.error('Failed to permanently delete note:', err);
      setError('Failed to permanently delete note');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const emptyTrash = useCallback(async () => {
    if (!isAuthenticated) {
      setTrashedNotes([]);
      return;
    }

    try {
      await dbApi.emptyTrash();
      setTrashedNotes([]);
      saveAllToStorage(notes, []);
    } catch (err) {
      console.error('Failed to empty trash:', err);
      setError('Failed to empty trash');
      throw err;
    }
  }, [isAuthenticated, notes]);
  
  const importNotes = useCallback((incomingNotes, mode = 'merge') => {
    if (!Array.isArray(incomingNotes)) {
      throw new Error('Invalid import format: expected notes array');
    }
    const normalized = incomingNotes.map(n => normalizeNote(n));
    if (mode === 'replace') {
      setNotes(normalized);
      return { imported: normalized.length, mode };
    }
    if (mode === 'merge') {
      setNotes(prev => {
        const existingIds = new Set(prev.map(n => n.id));
        const merged = [...prev];
        for (const note of normalized) {
          let id = note.id;
          if (existingIds.has(id)) {
            do {
              id = generateRandomId();
            } while (existingIds.has(id));
          }
          existingIds.add(id);
          merged.push({ ...note, id });
        }
        return merged;
      });
      return { imported: normalized.length, mode };
    }
    throw new Error('Invalid import mode');
  }, []);
  
  const searchNotes = useCallback((term) => {
    if (!term) return notes;
    const lower = term.toLowerCase();
    return notes.filter(note => 
      note.title.toLowerCase().includes(lower) ||
      note.content.toLowerCase().includes(lower) ||
      (note.tags && note.tags.toLowerCase().includes(lower))
    );
  }, [notes]);

  const getAllTags = useCallback(() => {
    const tagSet = new Set();
    notes.forEach(note => {
      if (note.tags) {
        note.tags.split(',').forEach(tag => {
          const trimmed = tag.trim();
          if (trimmed) tagSet.add(trimmed);
        });
      }
    });
    return Array.from(tagSet).sort();
  }, [notes]);

  const exportNotes = useCallback(() => {
    const data = {
      notes,
      exportedAt: new Date().toISOString(),
      version: '1.0'
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `semantic-notes-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [notes]);

  const getStats = useCallback(() => {
    const totalNotes = notes.length;
    const totalWords = notes.reduce((sum, note) =>
      sum + note.content.split(/\s+/).filter(w => w).length, 0
    );
    const totalChars = notes.reduce((sum, note) => sum + note.content.length, 0);
    const tags = getAllTags();
    
    return {
      totalNotes,
      totalWords,
      totalChars,
      totalTags: tags.length,
      averageNoteLength: totalNotes > 0 ? Math.round(totalChars / totalNotes) : 0,
      trashCount: trashedNotes.length
    };
  }, [notes, getAllTags, trashedNotes]);

  return {
    notes,
    trashedNotes,
    loading,
    error,
    addNote,
    updateNote,
    deleteNote,
    moveToTrash,
    restoreFromTrash,
    permanentDelete,
    emptyTrash,
    searchNotes,
    getAllTags,
    exportNotes,
    importNotes,
    getStats
  };
}
```


### 📄 frontend\src\main.jsx

```
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

import { AuthProvider } from './contexts/AuthContext'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </React.StrictMode>,
)
```


### 📄 frontend\src\services\api.js

```javascript
const EMB_LS_KEY = 'semantic-emb-cache-v1';

class APIService {
  constructor(baseUrl = import.meta.env.VITE_API_BASE_URL) {
    this.baseUrl = (baseUrl || '').replace(/\/$/, '');
    // Embedding cache: { [noteId]: { h: number, v: number[] } }
    this._embCache = this._loadEmbCache();
  }

  // Add token storage
  static authToken = null;

  setAuthToken(token) {
    APIService.authToken = token;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (APIService.authToken) {
      headers['Authorization'] = `Bearer ${APIService.authToken}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  async checkHealth() {
    return this.request('/api/health');
  }

  async getStats() {
    return this.request('/api/stats');
  }

  async embedDocuments(documents) {
    if (!documents?.length) {
      throw new Error('No documents provided');
    }
    return this.request('/api/embed', {
      method: 'POST',
      body: JSON.stringify({ documents })
    });
  }

  async buildGraph(options) {
    const {
      documents,
      labels = null,

      // legacy params (kept for back-compat)
      mode = 'knn',
      top_k = 2,
      threshold = 0.3,
      dr_method = 'pca',
      n_components = 2,
      cluster = 'none',

      // new canonical params
      connection = undefined,           // 'knn' | 'threshold'
      k_neighbors = undefined,          // 1-10
      similarity_threshold = undefined, // 0-1
      dim_reduction = undefined,        // 'pca' | 'umap' | 'tsne' | null
      clustering = undefined,           // 'kmeans' | 'agglomerative' | null

      n_clusters = null,
      include_embeddings = false,
    } = options || {};

    if (!documents?.length) {
      throw new Error('No documents provided');
    }

    const conn = (connection ?? (mode === 'threshold' ? 'threshold' : 'knn')) === 'threshold' ? 'threshold' : 'knn';

    const k = k_neighbors ?? top_k ?? 2;
    const th = similarity_threshold ?? threshold ?? 0.3;

    // allow null to disable DR
    const dr = dim_reduction === undefined ? dr_method : dim_reduction;
    // normalize clustering
    const clust = clustering === undefined
      ? (cluster === 'none' ? null : cluster)
      : clustering;

    const payload = {
      documents,
      n_components: n_components ?? 2,
      include_embeddings
    };

    if (labels) payload.labels = labels;

    // Ensure explicit, backend-friendly values for DR and clustering
    payload.dr_method = dr === null ? 'none' : (dr ?? 'pca');
    payload.cluster = clust ? clust : 'none';
    if (clust && n_clusters != null) {
      payload.n_clusters = n_clusters;
    }

    if (conn === 'knn') {
      payload.top_k = k;
    } else {
      payload.threshold = th;
    }

    return this.request('/api/graph', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  // ---------- Embeddings: helpers, caching, similarity ----------

  getNoteText(note) {
    const title = String(note?.title || '').trim();
    const content = String(note?.content || '').trim();
    const tags = String(note?.tags || '').trim();
    return `${title}. ${content} ${tags}`.trim();
  }

  _hashString(str) {
    // FNV-1a 32-bit
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = (h >>> 0) * 0x01000193;
    }
    return h >>> 0;
  }

  hashNote(note) {
    return this._hashString(`${note?.id ?? 'new'}::${this.getNoteText(note)}`);
  }

  _loadEmbCache() {
    try {
      const raw = localStorage.getItem(EMB_LS_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') return parsed;
    } catch (e) {
      console.warn('Embedding cache load failed:', e);
    }
    return {};
  }

  _saveEmbCache() {
    try {
      localStorage.setItem(EMB_LS_KEY, JSON.stringify(this._embCache));
    } catch (e) {
      // If quota exceeded, drop cache silently
      console.warn('Embedding cache save failed:', e);
    }
  }

  async embedText(text) {
    const res = await this.embedDocuments([text]);
    const emb = res?.embeddings?.[0];
    if (!emb) throw new Error('Failed to compute embedding');
    return emb;
  }

  async saveEmbeddingsToDatabase(embeddings) {
    try {
      await this.request('/api/embeddings/batch', {
        method: 'POST',
        body: JSON.stringify({ embeddings }),
      });
    } catch (error) {
      console.error('Failed to save embeddings to database:', error);
    }
  }

  async fetchEmbeddingsFromDatabase(noteIds) {
    try {
      const response = await this.request(`/api/embeddings?note_ids=${noteIds.join(',')}`);
      return response.embeddings || {};
    } catch (error) {
      console.error('Failed to fetch embeddings from database:', error);
      return {};
    }
  }

  async getEmbeddingsForNotes(notes) {
    if (!notes || notes.length === 0) return {};

    const cache = this._loadEmbCache();
    const embeddings = {};
    const notesToCompute = [];
    const notesToFetchFromDb = [];
    const authToken = APIService.authToken;

    for (const note of notes) {
      const cached = cache[note.id];
      const noteHash = this.hashNote(note);
      if (cached && cached.h === noteHash) {
        embeddings[note.id] = cached.v;
      } else if (authToken) {
        notesToFetchFromDb.push(note);
      } else {
        notesToCompute.push(note);
      }
    }

    if (notesToFetchFromDb.length > 0 && authToken) {
      const noteIds = notesToFetchFromDb.map(n => n.id);
      const dbEmbeddings = await this.fetchEmbeddingsFromDatabase(noteIds);

      for (const note of notesToFetchFromDb) {
        const dbEmb = dbEmbeddings[note.id];
        const noteHash = this.hashNote(note);
        if (dbEmb && dbEmb.content_hash === noteHash) {
          embeddings[note.id] = dbEmb.embedding;
          cache[note.id] = { h: noteHash, v: dbEmb.embedding };
        } else {
          notesToCompute.push(note);
        }
      }
      this._saveEmbCache();
    }

    if (notesToCompute.length > 0) {
      const texts = notesToCompute.map(n => this.getNoteText(n));
      const res = await this.embedDocuments(texts);
      const vecs = res.embeddings || [];
      for (let i = 0; i < notesToCompute.length; i++) {
        const note = notesToCompute[i];
        embeddings[note.id] = vecs[i];
        cache[note.id] = { h: this.hashNote(note), v: vecs[i] };
      }
      this._saveEmbCache();

      if (authToken && notesToCompute.length > 0) {
        const embeddingsToSave = notesToCompute
          .filter(n => embeddings[n.id])
          .map(n => ({
            note_id: n.id,
            content_hash: this.hashNote(n),
            embedding: embeddings[n.id],
            model_name: 'sentence-transformers/all-MiniLM-L6-v2',
          }));

        if (embeddingsToSave.length > 0) {
          this.saveEmbeddingsToDatabase(embeddingsToSave).catch(err =>
            console.error('Background embedding sync failed:', err)
          );
        }
      }
    }

    return embeddings;
  }

  // Safe cosine similarity (normalizes if needed)
  cosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      const x = a[i];
      const y = b[i];
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }
  
  async register(username, password, email = null) {
    const response = await this.request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, password, email }),
    });
    return response;
  }

  async login(username, password) {
    const response = await this.request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    return response;
  }

  async getCurrentUser() {
    const response = await this.request('/api/auth/me');
    return response;
  }
}

const apiService = new APIService();

export default apiService;
export { APIService };
```


### 📄 frontend\src\services\dbApi.js

```javascript
import apiService from './api';

const dbApi = {
  async fetchNotes() {
    return await apiService.request('/api/notes');
  },

  async createNote(noteData) {
    return await apiService.request('/api/notes', {
      method: 'POST',
      body: JSON.stringify(noteData),
    });
  },

  async updateNote(noteId, noteData) {
    return await apiService.request(`/api/notes/${noteId}`, {
      method: 'PUT',
      body: JSON.stringify(noteData),
    });
  },

  async moveToTrash(noteId) {
    return await apiService.request(`/api/notes/${noteId}/trash`, {
      method: 'POST',
    });
  },

  async restoreNote(noteId) {
    return await apiService.request(`/api/notes/${noteId}/restore`, {
      method: 'POST',
    });
  },

  async permanentDelete(noteId) {
    return await apiService.request(`/api/notes/${noteId}`, {
      method: 'DELETE',
    });
  },

  async fetchTrash() {
    return await apiService.request('/api/trash');
  },

  async emptyTrash() {
    return await apiService.request('/api/trash/empty', {
      method: 'POST',
    });
  },

  async importNotes(notes, trash) {
    return await apiService.request('/api/notes/import', {
      method: 'POST',
      body: JSON.stringify({ notes, trash }),
    });
  },
};

export default dbApi;
```


### 📄 frontend\src\utils\graphExport.js

```javascript
/**
 * Graph export utilities: PNG, SVG, JSON and clipboard helpers
 */

function pad2(n) {
  return String(n).padStart(2, '0');
}

export function formatTimestamp(date = new Date()) {
  const y = date.getFullYear();
  const m = pad2(date.getMonth() + 1);
  const d = pad2(date.getDate());
  const hh = pad2(date.getHours());
  const mm = pad2(date.getMinutes());
  return `${y}-${m}-${d}-${hh}-${mm}`;
}

export function defaultFilename(format = 'png') {
  const ts = formatTimestamp();
  if (format === 'json') return `semantic-graph-data-${ts}.json`;
  return `semantic-graph-${ts}.${format}`;
}

/**
 * Serialize an SVG element to string, ensuring necessary namespaces.
 */
export function getSvgString(svgEl) {
  if (!svgEl) throw new Error('SVG element is required');
  const cloned = svgEl.cloneNode(true);

  // Ensure width/height attributes exist (for canvas rasterization)
  const width = Number(svgEl.getAttribute('width') || svgEl.clientWidth || 800);
  const height = Number(svgEl.getAttribute('height') || svgEl.clientHeight || 600);
  cloned.setAttribute('width', String(width));
  cloned.setAttribute('height', String(height));

  // Add xmlns if missing
  if (!cloned.getAttribute('xmlns')) {
    cloned.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  }
  if (!cloned.getAttribute('xmlns:xlink')) {
    cloned.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
  }

  // Serialize
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(cloned);

  // Fix for some browsers that omit namespaces
  if (!source.match(/^<svg[^>]+xmlns="/)) {
    source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  if (!source.match(/^<svg[^>]+"http:\/\/www\.w3\.org\/1999\/xlink"/)) {
    source = source.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
  }

  // Ensure proper XML header
  return `<?xml version="1.0" standalone="no"?>${source}`;
}

/**
 * Download a Blob with the given filename.
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }, 0);
}

/**
 * Convert an SVG element to a PNG Blob by drawing into a high-DPI canvas.
 * scale: rasterization scale factor (2 for 2x resolution)
 */
export async function svgToPng(svgEl, scale = 2) {
  const width = Number(svgEl.getAttribute('width') || svgEl.clientWidth || 800);
  const height = Number(svgEl.getAttribute('height') || svgEl.clientHeight || 600);
  const svgString = getSvgString(svgEl);

  const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  try {
    const img = await new Promise((resolve, reject) => {
      const image = new Image();
      // Important: set crossOrigin to avoid taint if external resources are referenced
      image.crossOrigin = 'anonymous';
      image.onload = () => resolve(image);
      image.onerror = (e) => reject(new Error('Failed to load SVG for rasterization'));
      image.src = url;
    });

    const canvas = document.createElement('canvas');
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);

    const ctx = canvas.getContext('2d');
    // High quality rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Scale context and draw the SVG image
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, width, height);

    const pngBlob = await new Promise((resolve) => {
      canvas.toBlob((b) => resolve(b), 'image/png');
    });

    if (!pngBlob) throw new Error('Canvas export produced empty image');
    return pngBlob;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function savePng(svgEl, filename, scale = 2) {
  const blob = await svgToPng(svgEl, scale);
  downloadBlob(blob, filename);
  return blob;
}

export function saveSvg(svgEl, filename) {
  const svgString = getSvgString(svgEl);
  const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  downloadBlob(svgBlob, filename);
  return svgBlob;
}

export function buildJsonExport(graphData, metadata = {}) {
  const payload = {
    type: 'semantic-graph',
    version: 1,
    created_at: new Date().toISOString(),
    metadata,
    nodes: Array.isArray(graphData?.nodes) ? graphData.nodes : [],
    edges: Array.isArray(graphData?.edges) ? graphData.edges : [],
  };
  return JSON.stringify(payload, null, 2);
}

export function saveJson(graphData, filename, metadata = {}) {
  const json = buildJsonExport(graphData, metadata);
  const blob = new Blob([json], { type: 'application/json;charset=utf-8' });
  downloadBlob(blob, filename);
  return blob;
}

/**
 * Clipboard helpers
 */
export async function copyBlobToClipboard(blob, mime) {
  if (!navigator?.clipboard || !window.ClipboardItem) {
    throw new Error('Clipboard API not supported');
  }
  const item = new ClipboardItem({ [mime]: blob });
  await navigator.clipboard.write([item]);
}

export async function copyPng(svgEl, scale = 2) {
  const blob = await svgToPng(svgEl, scale);
  await copyBlobToClipboard(blob, 'image/png');
}

export async function copySvg(svgEl) {
  // Prefer writing as image/svg+xml if supported, otherwise as text
  const svgString = getSvgString(svgEl);
  if (navigator?.clipboard && window.ClipboardItem) {
    const blob = new Blob([svgString], { type: 'image/svg+xml' });
    try {
      await copyBlobToClipboard(blob, 'image/svg+xml');
      return;
    } catch {
      // fallback to text
    }
  }
  await navigator.clipboard.writeText(svgString);
}

export async function copyJson(graphData, metadata = {}) {
  const json = buildJsonExport(graphData, metadata);
  await navigator.clipboard.writeText(json);
}
```


### 📄 frontend\vite.config.js

```javascript
// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,        // <— add this
    open: true,
    // Proxy API requests to the FastAPI backend
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Optimize chunks
    rollupOptions: {
      output: {
        manualChunks: {
          'd3': ['d3'],
          'react-vendor': ['react', 'react-dom'],
        }
      }
    }
  }
})
```


### 📄 graph_service.py

```python
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any, Literal
import numpy as np

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

    if _HAS_FAISS:
        # Use inner product (dot) for normalized vectors = cosine
        d = X.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(X.astype(np.float32, copy=False))
        # Query itself then drop self (first index)
        sims, idx = index.search(X.astype(np.float32, copy=False), k + 1)
        return idx[:, 1:]
    else:
        # Pure NumPy fallback
        S = _pairwise_cosine_from_normalized(X)
        # Exclude self by setting diagonal to -inf
        np.fill_diagonal(S, -np.inf)
        # Argpartition to get k largest per row
        idx = np.argpartition(-S, kth=np.minimum(k, n-1)-1, axis=1)[:, :k]
        # Optional: sort top-k by similarity descending for nicer ordering
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
        # t-SNE is slower; use perplexity rule-of-thumb
        perplexity = max(5, min(30, X.shape[0] // 3))
        tsne = TSNE(n_components=n_components, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state)
        coords = tsne.fit_transform(X)
        return coords.astype(np.float32, copy=False)

    # Fallback
    return None

def cluster_embeddings(
    X: np.ndarray,
    method: Literal["none", "kmeans", "agglomerative"] = "none",
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Optional[np.ndarray]:
    if method == "none" or X.size == 0:
        return None

    if n_clusters is None:
        # Heuristic: ~sqrt(n/2), min 2, max 20
        n = X.shape[0]
        n_clusters = max(2, min(20, int(np.sqrt(max(2, n/2)))))

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        labels = model.fit_predict(X)
        return labels.astype(np.int32, copy=False)

    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        return labels.astype(np.int32, copy=False)

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
    # Clustering
    clusters = cluster_embeddings(X, method=cluster_method, n_clusters=n_clusters)

    # Nodes
    nodes = []
    for i in range(n):
        node = {
            "id": str(i),
            "label": labels[i] if labels and i < len(labels) else None,
        }
        if node["label"] is None:
            # default: trimmed preview of the text; caller may pass labels for better display
            node["label"] = f"Doc {i}"
        if include_embeddings:
            node["embedding"] = X[i].tolist()
        if coords is not None and coords.shape[1] >= 2:
            node["x"], node["y"] = float(coords[i, 0]), float(coords[i, 1])
        if clusters is not None:
            node["cluster"] = int(clusters[i])
        nodes.append(node)

    # Edges
    edges = []
    added = set()  # to avoid duplicates for kNN case
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
        # Build undirected edges; add once per pair
        S = _pairwise_cosine_from_normalized(X)
        for i in range(n):
            for j in idx[i]:
                a, b = (i, int(j)) if i < j else (int(j), i)
                if a == b:
                    continue
                key = (a, b)
                if key in added:
                    continue
                added.add(key)
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
from __future__ import annotations

# --- Set loky env + silence its Windows warning BEFORE any 3rd-party imports ---
import os as _os
import warnings as _warnings

if "LOKY_MAX_CPU_COUNT" not in _os.environ:
    try:
        _os.environ["LOKY_MAX_CPU_COUNT"] = str(_os.cpu_count() or 1)
    except Exception:
        _os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Silence only the specific loky UserWarning about physical cores on Windows
_warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)

# -------------------------------------------------------------------------------

from typing import List, Optional, Literal, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import importlib.util
import numpy as np

from embedding_service import get_embedding_service
from graph_service import build_similarity_graph
from database import get_db, init_db
from models import User, Note, Embedding
from auth import hash_password, create_access_token, get_current_user
import db_service
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status


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
    return importlib.util.find_spec(name) is not None


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

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Diagnostic logging
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
print(f"[STARTUP] FRONTEND_ORIGIN: {frontend_origin}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Routes ----------

from fastapi import Request


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
        from pydantic import BaseModel, ValidationError, conlist

        class ImportNoteSchema(BaseModel):
            title: str
            content: str
            tags: str | None = ""
            is_deleted: bool | None = False

        class ImportRequest(BaseModel):
            notes: conlist(ImportNoteSchema, min_length=0) = []
            trash: conlist(ImportNoteSchema, min_length=0) = []

        try:
            raw_data = await request.json()
            data = ImportRequest(**raw_data)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors())

        notes_to_import = data.notes
        trash_to_import = data.trash

        if not notes_to_import and not trash_to_import:
            return {"imported": 0, "id_mapping": {}}

        id_mapping = {}
        if notes_to_import:
            id_mapping = db_service.bulk_create_notes(db, current_user.id, notes_to_import)

        trash_mapping = {}
        if trash_to_import:
            trash_notes = []
            for trash_note in trash_to_import:
                trash_note_copy = trash_note.copy()
                trash_note_copy['is_deleted'] = True
                trash_note_copy['deleted_at'] = trash_note.get('deletedAt') or trash_note.get('deleted_at')
                trash_notes.append(trash_note_copy)

            trash_mapping = db_service.bulk_create_notes(db, current_user.id, trash_notes)
            id_mapping.update(trash_mapping)

        total_imported = len(notes_to_import) + len(trash_to_import)

        return {
            "imported": total_imported,
            "id_mapping": id_mapping
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
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
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No documents provided")

    service = get_embedding_service()
    vecs = service.encode(req.documents)
    return EmbedResponse(embeddings=vecs.tolist())


@app.post("/api/graph", response_model=GraphResponse)
def graph(req: GraphRequest) -> GraphResponse:
    if not req.documents:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No documents provided")

    threshold = req.threshold
    top_k = None if threshold is not None else req.top_k

    service = get_embedding_service()
    X = service.encode(req.documents)

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


class NoteResponse(BaseModel):
    id: int
    title: str
    content: str
    tags: str
    created_at: str
    updated_at: str
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


### 📄 test_client.py

```python
#!/usr/bin/env python3
"""Integration test script for the Semantic Embedding Graph Engine backend.

Usage:
  python test_client.py --base-url http://localhost:8000
  python test_client.py --spawn --base-url http://127.0.0.1:8000
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import requests
except Exception:
    print("This script requires the 'requests' package. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)

SAMPLE_DOCS = [
    "AI in Finance: using machine learning for fraud detection, risk scoring, and portfolio optimization.",
    "Neural Network Research: transformers, attention mechanisms, and representation learning.",
    "Blockchain Ledger Overview: decentralized consensus, cryptographic hashes, and transaction validation.",
    "Jazz music theory and improvisation techniques from the bebop era.",
    "Big data processing pipelines with Spark and distributed systems design.",
    "Auditing digital asset custodians: controls, risk matrices, and proof-of-reserves verification."
]

def wait_for_health(base_url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/api/health", timeout=3)
            if r.ok and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def pretty(title: str):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def assert_close(a: float, b: float, tol: float = 1e-3, msg: str = ""):
    if abs(a-b) > tol:
        raise AssertionError(msg or f"Values not close: {a} vs {b} (tol={tol})")

def cosine_sim(a: List[float], b: List[float]) -> float:
    # inputs are expected unit vectors already
    return sum(x*y for x, y in zip(a, b))

def top_pairs(embeddings: List[List[float]], k: int = 5) -> List[Tuple[int,int,float]]:
    n = len(embeddings)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            w = cosine_sim(embeddings[i], embeddings[j])
            pairs.append((i, j, w))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:k]

def run_tests(base_url: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Stats (optional)
    try:
        pretty("0) /api/stats")
        s = requests.get(f"{base_url}/api/stats", timeout=5)
        if s.ok:
            print(json.dumps(s.json(), indent=2))
    except Exception:
        pass

    # 1) Health
    pretty("1) /api/health")
    r = requests.get(f"{base_url}/api/health", timeout=5)
    r.raise_for_status()
    print("Health:", r.json())

    # 2) Embed
    pretty("2) /api/embed")
    t0 = time.time()
    r = requests.post(f"{base_url}/api/embed", json={"documents": SAMPLE_DOCS}, timeout=60)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    embeds = data["embeddings"]
    print(f"Received {len(embeds)} embeddings in {elapsed:.3f}s")
    assert len(embeds) == len(SAMPLE_DOCS), "Number of embeddings != number of docs"
    dim = len(embeds[0])
    print("Embedding dim:", dim)

    # Check L2 normalization
    for i, e in enumerate(embeds):
        norm = math.sqrt(sum(v*v for v in e))
        assert_close(norm, 1.0, tol=2e-2, msg=f"Embedding {i} not unit-normalized (||e||={norm:.4f})")
    print("All embeddings are ~unit length ✔")

    # Print top similar pairs
    pairs = top_pairs(embeds, k=6)
    print("\nTop similar pairs (cosine):")
    for i, j, w in pairs:
        print(f"  ({i}, {j}) -> {w:.3f}")

    # 3) Graph (threshold mode)
    pretty("3) /api/graph (threshold=0.5, PCA+KMeans)")
    payload = {
        "documents": SAMPLE_DOCS,
        "threshold": 0.5,
        "include_embeddings": False,
        "dr_method": "pca",
        "n_components": 2,
        "cluster": "kmeans",
        "n_clusters": 3,
        "labels": [f"Doc {i}" for i in range(len(SAMPLE_DOCS))]
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/api/graph", json=payload, timeout=60)
    elapsed = time.time() - t0
    r.raise_for_status()
    g1 = r.json()
    print(f"Graph built in {elapsed:.3f}s with {len(g1['nodes'])} nodes and {len(g1['edges'])} edges")
    # Validate edges respect threshold
    for e in g1["edges"]:
        assert e["weight"] >= 0.5, "Edge weight below threshold"
        assert 0.0 <= e["weight"] <= 1.0, "Invalid cosine weight range"
    # Validate nodes have x/y and optional cluster
    for n in g1["nodes"]:
        assert "x" in n and "y" in n, "Missing x/y layout"
        assert "label" in n, "Missing label"
    (out_dir / "graph_threshold.json").write_text(json.dumps(g1, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "graph_threshold.json")

    # 3b) Threshold sweep to help pick a value
    pretty("3b) Threshold sweep")
    sweep = []
    for th in [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]:
        r = requests.post(f"{base_url}/api/graph", json={"documents": SAMPLE_DOCS, "threshold": th}, timeout=60)
        r.raise_for_status()
        g = r.json()
        sweep.append({"threshold": th, "edges": len(g["edges"])})
    print(json.dumps(sweep, indent=2))
    (out_dir / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    # 4) Graph (kNN mode)
    pretty("4) /api/graph (top_k=2, UMAP if available else PCA)")
    payload = {
        "documents": SAMPLE_DOCS,
        "top_k": 2,
        "include_embeddings": True,
        "dr_method": "pca",
        "n_components": 2,
        "cluster": "none"
    }
    r = requests.post(f"{base_url}/api/graph", json=payload, timeout=60)
    r.raise_for_status()
    g2 = r.json()
    print(f"kNN Graph: {len(g2['nodes'])} nodes, {len(g2['edges'])} edges")
    # Basic validations
    for e in g2["edges"]:
        assert 0.0 <= e["weight"] <= 1.0, "Invalid cosine weight"
    # Verify embeddings included
    assert "embedding" in g2["nodes"][0], "Embeddings not included as requested"
    (out_dir / "graph_knn.json").write_text(json.dumps(g2, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "graph_knn.json")

    print("\nAll tests passed ✔")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--spawn", action="store_true", help="Spawn uvicorn server locally for the test run")
    parser.add_argument("--wait", type=float, default=45.0, help="Seconds to wait for server health")
    parser.add_argument("--out", default="tests/output", help="Where to write test artifacts")
    args = parser.parse_args()

    proc = None
    try:
        if args.spawn:
            # Spawn uvicorn from local project
            env = os.environ.copy()
            cmd = [sys.executable, "-m", "uvicorn", "main:app", "--port", "8000", "--host", "127.0.0.1"]
            print("Spawning server:", " ".join(cmd))
            proc = subprocess.Popen(cmd, cwd=Path(__file__).parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Wait for health
            if not wait_for_health(args.base_url, timeout=args.wait):
                # Dump a few lines of logs to help debugging
                if proc and proc.stdout:
                    print("\n--- Server output (last 50 lines) ---")
                    lines = proc.stdout.readlines()[-50:]
                    for line in lines:
                        print(line.rstrip())
                raise SystemExit("Server did not become healthy in time.")
        else:
            if not wait_for_health(args.base_url, timeout=args.wait):
                raise SystemExit("Server not reachable. Start it with: uvicorn main:app --reload")

        out_dir = Path(args.out)
        run_tests(args.base_url, out_dir)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

if __name__ == "__main__":
    main()

```
