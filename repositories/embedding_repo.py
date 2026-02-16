from sqlalchemy.orm import Session
from datetime import datetime

from models import Embedding


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
        db.flush()
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
    db.flush()
    db.refresh(embedding)
    return embedding


def batch_get_embeddings(db: Session, note_ids: list[int]) -> dict[int, Embedding]:
    embeddings = db.query(Embedding).filter(Embedding.note_id.in_(note_ids)).all()
    return {e.note_id: e for e in embeddings}
