from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, conlist


class NoteRequest(BaseModel):
    title: str
    content: str
    tags: str = ""


class NoteResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )

    id: int
    title: str
    content: str
    tags: str
    created_at: datetime
    updated_at: datetime
    is_deleted: bool


class ImportNoteSchema(BaseModel):
    """Accept plain notes from local export (id is created server-side)."""

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
