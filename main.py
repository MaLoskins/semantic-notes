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
