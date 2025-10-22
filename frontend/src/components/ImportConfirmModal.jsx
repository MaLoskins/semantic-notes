import React, { useEffect } from 'react';

export default function ImportConfirmModal({ isOpen, count = 0, onReplace, onMerge, onCancel }) {
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e) => {
      if (e.key === 'Escape') onCancel?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onCancel]);

  if (!isOpen) return null;

  const stop = (e) => e.stopPropagation();

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" role="dialog" aria-modal="true" onClick={stop}>
        <div className="modal-header">
          <h3>Import Notes</h3>
        </div>
        <div className="modal-body">
          <p>Detected {count} notes in the selected file.</p>
          <p>How would you like to import them?</p>
          <ul className="modal-list">
            <li><strong>Replace all notes</strong> — clears current notes and loads the imported ones.</li>
            <li><strong>Merge with existing</strong> — keeps current notes and adds imported notes. Conflicting IDs will be regenerated.</li>
          </ul>
        </div>
        <div className="modal-actions">
          <button className="btn btn-danger" onClick={onReplace} title="Replace current notes">
            Replace All
          </button>
          <button className="btn btn-primary" onClick={onMerge} title="Merge imported notes">
            Merge
          </button>
          <button className="btn btn-secondary" onClick={onCancel} title="Cancel">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}