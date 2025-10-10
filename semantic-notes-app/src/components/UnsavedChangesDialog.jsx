import React, { useEffect } from 'react';

export default function UnsavedChangesDialog({ 
  isOpen = false,
  onSaveAndContinue,
  onDiscard,
  onCancel
}) {
  if (!isOpen) return null;

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel && onCancel();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        onSaveAndContinue && onSaveAndContinue();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        onSaveAndContinue && onSaveAndContinue();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onCancel, onSaveAndContinue]);

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="unsaved-title">
      <div className="modal">
        <div className="modal-header">
          <h3 id="unsaved-title">Unsaved changes</h3>
        </div>
        <div className="modal-body">
          <p>You have unsaved changes. If you continue without saving, your edits will be lost.</p>
        </div>
        <div className="modal-actions">
          <button
            className="btn btn-primary"
            onClick={() => onSaveAndContinue && onSaveAndContinue()}
            autoFocus
          >
            Save & Continue
          </button>
          <button
            className="btn btn-danger"
            onClick={() => onDiscard && onDiscard()}
          >
            Discard
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => onCancel && onCancel()}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}