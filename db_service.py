"""
Backward-compatibility shim.

All functions have been moved to domain-specific repository modules:
  - repositories/user_repo.py      (user CRUD + auth)
  - repositories/note_repo.py      (note CRUD + trash + bulk import)
  - repositories/embedding_repo.py (embedding storage/retrieval)

This file re-exports everything so that any stale imports still resolve.
Prefer importing directly from the repositories package instead.
"""

from repositories.user_repo import (        # noqa: F401
    create_user,
    get_user_by_username,
    authenticate_user,
    update_last_login,
)
from repositories.note_repo import (        # noqa: F401
    get_user_notes,
    get_note_by_id,
    create_note,
    update_note,
    soft_delete_note,
    restore_note,
    permanent_delete_note,
    get_user_trash,
    empty_trash,
    bulk_create_notes,
)
from repositories.embedding_repo import (   # noqa: F401
    get_note_embedding,
    save_note_embedding,
    batch_get_embeddings,
)
