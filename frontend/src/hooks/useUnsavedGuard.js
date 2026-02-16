import { useState, useCallback } from 'react';

export function useUnsavedGuard() {
  const [isEditorDirty, setIsEditorDirty] = useState(false);
  const [unsavedOpen, setUnsavedOpen] = useState(false);
  const [pendingAction, setPendingAction] = useState(null);

  /**
   * Attempt to navigate. If the editor is dirty and actively editing,
   * the action is stored as pending and the dialog is shown.
   * Returns true if navigation can proceed immediately, false if blocked.
   */
  const guardedNavigate = useCallback((action, editingActive) => {
    if (editingActive && isEditorDirty) {
      setPendingAction(action);
      setUnsavedOpen(true);
      return false;
    }
    return true;
  }, [isEditorDirty]);

  const cancelDialog = useCallback(() => {
    setUnsavedOpen(false);
  }, []);

  return {
    isEditorDirty,
    setIsEditorDirty,
    unsavedOpen,
    setUnsavedOpen,
    pendingAction,
    setPendingAction,
    guardedNavigate,
    cancelDialog,
  };
}
