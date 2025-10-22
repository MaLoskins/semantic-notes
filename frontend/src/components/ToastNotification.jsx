import React, { useEffect, useRef } from 'react';

export default function ToastNotification({
  isOpen = false,
  message = '',
  actionLabel = 'Undo',
  onAction,
  onClose,
  duration = 5000
}) {
  const timerRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      if (onClose) onClose();
    }, duration);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isOpen, message, duration, onClose]);

  if (!isOpen) return null;

  return (
    <div className="toast-container" role="status" aria-live="polite">
      <div className="toast-card fade-in">
        <div className="toast-message">{message}</div>

        <div className="toast-actions">
          {onAction && (
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => {
                if (onAction) onAction();
                if (onClose) onClose();
              }}
            >
              {actionLabel}
            </button>
          )}
          <button
            className="toast-close"
            aria-label="Dismiss notification"
            title="Dismiss"
            onClick={() => onClose && onClose()}
          >
            ×
          </button>
        </div>
      </div>
    </div>
  );
}