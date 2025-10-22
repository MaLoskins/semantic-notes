import React, { useMemo, useState } from 'react';
import ConfirmDialog from './ConfirmDialog';

const PREVIEW_LENGTH = 120;

function formatRelativeTime(dateString) {
  if (!dateString) return '';
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return date.toLocaleDateString();
}

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
          {items.map(note => (
            <div key={note.id} className="trash-item">
              <div className="trash-item-content">
                <div className="trash-item-header">
                  <h4 className="note-item-title">{note.title}</h4>
                  <span className="note-date">Deleted {formatRelativeTime(note.deletedAt)}</span>
                </div>
                <p className="note-preview">
                  {note.content.substring(0, PREVIEW_LENGTH)}
                  {note.content.length > PREVIEW_LENGTH && '...'}
                </p>
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
          ))}
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