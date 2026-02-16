from sqlalchemy.orm import Session
from datetime import datetime, timezone

from models import Note


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
    db.flush()
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
    db.flush()
    db.refresh(note)
    return note


def soft_delete_note(db: Session, note_id: int, user_id: int) -> bool:
    note = get_note_by_id(db, note_id, user_id)
    if not note or note.is_deleted:
        return False
    note.is_deleted = True
    note.deleted_at = datetime.utcnow()
    return True


def restore_note(db: Session, note_id: int, user_id: int) -> bool:
    note = get_note_by_id(db, note_id, user_id)
    if not note or not note.is_deleted:
        return False
    note.is_deleted = False
    note.deleted_at = None
    note.updated_at = datetime.utcnow()
    return True


def permanent_delete_note(db: Session, note_id: int, user_id: int) -> bool:
    note = get_note_by_id(db, note_id, user_id)
    if not note:
        return False
    db.delete(note)
    return True


def get_user_trash(db: Session, user_id: int) -> list[Note]:
    return db.query(Note).filter(Note.user_id == user_id, Note.is_deleted == True).all()


def empty_trash(db: Session, user_id: int) -> int:
    count = db.query(Note).filter(Note.user_id == user_id, Note.is_deleted == True).delete()
    return count


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
    id_mapping = {}

    for note_data in notes_list:
        old_id = note_data.get('id')
        created_at = note_data.get('createdAt') or note_data.get('created_at')
        updated_at = note_data.get('updatedAt') or note_data.get('updated_at')

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

    return id_mapping
