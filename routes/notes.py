from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from repositories.note_repo import (
    bulk_create_notes,
    create_note,
    empty_trash,
    get_user_notes,
    get_user_trash,
    permanent_delete_note,
    restore_note,
    soft_delete_note,
    update_note,
)
from models import User
from schemas.notes import ImportRequest, NoteRequest, NoteResponse

router = APIRouter(tags=["notes"])


# ---------- Import (must be before /{note_id} routes) ----------


@router.post("/api/notes/import")
async def import_notes(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Bulk import notes from localStorage.
    Request body: { "notes": [...], "trash": [...] }
    Returns: { "imported": count, "id_mapping": {old_id: new_id} }
    """
    try:
        try:
            raw_data = await request.json()
            data = ImportRequest(**raw_data)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors())

        notes_to_import: List[Dict[str, Any]] = []
        for note in data.notes:
            d = note.model_dump(by_alias=True)
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
            id_mapping = bulk_create_notes(db, current_user.id, notes_to_import)

        if trash_to_import:
            trash_mapping = bulk_create_notes(db, current_user.id, trash_to_import)
            id_mapping.update(trash_mapping)

        total_imported = len(notes_to_import) + len(trash_to_import)
        return {"imported": total_imported, "id_mapping": id_mapping}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


# ---------- Note CRUD ----------


@router.get("/api/notes", response_model=list[NoteResponse])
def list_notes(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    notes = get_user_notes(db, current_user.id)
    return notes


@router.post("/api/notes", response_model=NoteResponse)
def create_new_note(
    req: NoteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    note = create_note(db, current_user.id, req.title, req.content, req.tags)
    return note


@router.put("/api/notes/{note_id}", response_model=NoteResponse)
def update_existing_note(
    note_id: int,
    req: NoteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    updated_note = update_note(db, note_id, current_user.id, req.title, req.content, req.tags)
    if not updated_note:
        raise HTTPException(status_code=404, detail="Note not found")
    return updated_note


@router.post("/api/notes/{note_id}/trash")
def trash_note(
    note_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    success = soft_delete_note(db, note_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"status": "moved to trash"}


@router.post("/api/notes/{note_id}/restore")
def restore_note_endpoint(
    note_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    success = restore_note(db, note_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Note not found or not deleted")
    return {"status": "restored"}


@router.delete("/api/notes/{note_id}")
def permanently_delete_note(
    note_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    success = permanent_delete_note(db, note_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"status": "permanently deleted"}


# ---------- Trash ----------


@router.get("/api/trash", response_model=list[NoteResponse])
def get_trash(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return get_user_trash(db, current_user.id)


@router.post("/api/trash/empty")
def empty_trash_endpoint(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    deleted_count = empty_trash(db, current_user.id)
    return {"deleted_count": deleted_count}
