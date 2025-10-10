import React from 'react';

export default function ConfirmDialog({
  isOpen = false,
  title = 'Confirm',
  message = '',
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
  danger = false
}) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
      <div className="modal">
        <div className="modal-header">
          <h3 id="confirm-title">{title}</h3>
        </div>
        <div className="modal-body">
          <p>{message}</p>
        </div>
        <div className="modal-actions">
          <button
            className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`}
            onClick={() => onConfirm && onConfirm()}
          >
            {confirmLabel}
          </button>
          <button className="btn btn-secondary" onClick={() => onCancel && onCancel()}>
            {cancelLabel}
          </button>
        </div>
      </div>
    </div>
  );
}