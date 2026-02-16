from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from repositories.note_repo import get_note_by_id
from repositories.embedding_repo import save_note_embedding, batch_get_embeddings
from embedding_service import get_embedding_service
from models import User
from schemas.embeddings import EmbedRequest, EmbedResponse

router = APIRouter(tags=["embeddings"])


@router.post("/api/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, current_user: User = Depends(get_current_user)) -> EmbedResponse:
    if not req.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    service = get_embedding_service()
    vecs = service.encode(req.documents)
    return EmbedResponse(embeddings=vecs.tolist())


@router.post("/api/embeddings/batch")
async def save_embeddings_batch(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
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
            note = get_note_by_id(db, note_id, current_user.id)
            if not note:
                raise HTTPException(status_code=403, detail=f"Note {note_id} not found or access denied")

        saved_count = 0
        for emb_data in embeddings_data:
            try:
                save_note_embedding(
                    db,
                    note_id=emb_data["note_id"],
                    content_hash=emb_data["content_hash"],
                    embedding_vector=emb_data["embedding"],
                    model_name=emb_data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                )
                saved_count += 1
            except Exception as e:
                print(f"Failed to save embedding for note {emb_data['note_id']}: {e}")

        return {"saved": saved_count, "total": len(embeddings_data)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/embeddings")
async def get_embeddings(
    note_ids: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
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
            note = get_note_by_id(db, note_id, current_user.id)
            if not note:
                raise HTTPException(status_code=403, detail=f"Note {note_id} not found or access denied")

        embeddings_dict = batch_get_embeddings(db, note_id_list)

        result = {}
        for note_id, embedding in embeddings_dict.items():
            result[str(note_id)] = {
                "content_hash": embedding.content_hash,
                "embedding": embedding.embedding_vector,
                "model_name": embedding.model_name,
            }

        return {"embeddings": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
