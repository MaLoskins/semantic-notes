from sqlalchemy.orm import Session
from datetime import datetime
from models import User, Note, Embedding
from auth import hash_password, verify_password


# -------------------- USER OPERATIONS -------------------- #

def create_user(db: Session, username: str, password: str, email: str = None) -> User:
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


def update_note(db: Session, note_id: int, user_id: int, title: str, content: str, tags: str) -> Note | None:
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


def soft_delete_note(db: Session, note_id: int, user_id: int) -> bool:
    note = get_note_by_id(db, note_id, user_id)
    if not note or note.is_deleted:
        return False
    note.is_deleted = True
    note.deleted_at = datetime.utcnow()
    db.commit()
    return True


def restore_note(db: Session, note_id: int, user_id: int) -> bool:
    note = get_note_by_id(db, note_id, user_id)
    if not note or not note.is_deleted:
        return False
    note.is_deleted = False
    note.deleted_at = None
    db.commit()
    return True


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

            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))

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