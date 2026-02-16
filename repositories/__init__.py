"""Domain-specific repository modules."""

from repositories.user_repo import (
    create_user,
    get_user_by_username,
    authenticate_user,
    update_last_login,
)
from repositories.note_repo import (
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
from repositories.embedding_repo import (
    get_note_embedding,
    save_note_embedding,
    batch_get_embeddings,
)
