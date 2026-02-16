import React, { useMemo, useState } from 'react';
import ConfirmDialog from './ConfirmDialog';
import MarkdownPreview from './MarkdownPreview';
import { formatRelativeTime } from '../utils/formatTime';

const PREVIEW_LENGTH = 120;

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
        {items.map(note => {
          const full = String(note.content || '');
          const snippet = full.slice(0, PREVIEW_LENGTH);
          return (
            <div key={note.id} className="trash-item">
              <div className="trash-item-content">
                <div className="trash-item-header">
                  <h4 className="note-item-title">{note.title}</h4>
                  <span className="note-date">Deleted {formatRelativeTime(note.deletedAt)}</span>
                </div>
                <div
                  className="note-preview markdown-snippet markdown-snippet--compact"
                  style={{ maxHeight: 140, overflow: 'hidden' }}
                >
                  <MarkdownPreview content={snippet} />
                  {full.length > PREVIEW_LENGTH && <span className="truncate-ellipsis">…</span>}
                </div>


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
          );
        })}
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