import React, { useState, useEffect, useMemo, useRef, forwardRef, useImperativeHandle } from 'react';
import MarkdownPreview from './MarkdownPreview';
import MarkdownCheatsheet from './MarkdownCheatsheet';

export default forwardRef(function NoteEditor({ note, onSave, onCancel, onDirtyChange, onFindSimilar }, ref) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [tags, setTags] = useState('');
  const [original, setOriginal] = useState({ title: '', content: '', tags: '' });

  // Markdown editor modes and helpers
  const [viewMode, setViewMode] = useState('edit'); // 'edit' | 'preview' | 'split'
  const [cheatsheetOpen, setCheatsheetOpen] = useState(false);
  const textareaRef = useRef(null);
  const previewRef = useRef(null);
  const syncingRef = useRef(false);
 
  // Load note into form and establish original snapshot
  useEffect(() => {
    if (note) {
      const t = note.title || '';
      const c = note.content || '';
      const g = note.tags || '';
      setTitle(t);
      setContent(c);
      setTags(g);
      setOriginal({ title: t, content: c, tags: g });
    } else {
      setTitle('');
      setContent('');
      setTags('');
      setOriginal({ title: '', content: '', tags: '' });
    }
  }, [note]);

  // Dirty detection
  const isDirty = useMemo(() => {
    return title !== original.title || content !== original.content || tags !== original.tags;
  }, [title, content, tags, original]);

  // Notify parent when dirty state changes
  const lastDirty = useRef(isDirty);
  useEffect(() => {
    if (lastDirty.current !== isDirty) {
      lastDirty.current = isDirty;
      if (typeof onDirtyChange === 'function') onDirtyChange(isDirty);
    }
  }, [isDirty, onDirtyChange]);

  // Warn when trying to close/refresh tab with unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (!isDirty) return;
      e.preventDefault();
      e.returnValue = '';
      return '';
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty]);

  const buildNoteData = () => {
    const noteData = {
      ...note,
      title: title.trim(),
      content: content.trim(),
      tags: tags.trim(),
      updatedAt: new Date().toISOString()
    };
    if (!note?.id) {
      noteData.createdAt = new Date().toISOString();
    }
    return noteData;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!title.trim()) {
      alert('Title is required');
      return;
    }
    if (!content.trim()) {
      alert('Content is required');
      return;
    }

    const noteData = buildNoteData();
    onSave(noteData);
    // Mark clean after successful save
    setOriginal({ title: noteData.title, content: noteData.content, tags: noteData.tags });
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && (e.key === 'Enter' || e.key.toLowerCase() === 's')) {
      e.preventDefault();
      handleSubmit(e);
    } else if (e.key === 'Escape') {
      onCancel();
    }
  };

  // Expose imperative API to parent (App) for Save & Continue flow
  useImperativeHandle(ref, () => ({
    isDirty: () => isDirty,
    getCurrentData: () => buildNoteData(),
    submit: () => {
      const fakeEvt = { preventDefault: () => {} };
      handleSubmit(fakeEvt);
    }
  }), [isDirty, title, content, tags, note]);

  // Scroll sync between editor and preview
  const syncScroll = (from) => {
    if (syncingRef.current) return;
    const ta = textareaRef.current;
    const pv = previewRef.current;
    if (!ta || !pv) return;

    const src = from === 'textarea' ? ta : pv;
    const dst = from === 'textarea' ? pv : ta;

    const srcScrollable = Math.max(1, src.scrollHeight - src.clientHeight);
    const ratio = src.scrollTop / srcScrollable;
    const dstScrollable = Math.max(1, dst.scrollHeight - dst.clientHeight);
    syncingRef.current = true;
    try {
      dst.scrollTop = ratio * dstScrollable;
    } finally {
      // release lock on next tick to avoid feedback loop
      setTimeout(() => { syncingRef.current = false; }, 0);
    }
  };

  const handleTextareaScroll = () => syncScroll('textarea');
  const handlePreviewScroll = () => syncScroll('preview');

  const isEditing = note?.id !== undefined;
 
  return (
    <div className="note-editor">
      <div className="editor-header">
        <h2>{isEditing ? 'Edit Note' : 'New Note'}</h2>
        {isDirty && (
          <div className="unsaved-indicator">
            <span className="unsaved-dot" />
            <span>Unsaved changes</span>
          </div>
        )}
      </div>
      
      <form onSubmit={handleSubmit} className="editor-form">
        <div className="form-group">
          <label className="form-label">Title</label>
          <input
            type="text"
            placeholder="Note title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={handleKeyDown}
            className="form-input"
            autoFocus
          />
        </div>

        <div className="form-group" style={{ flex: 1 }}>
          <div className="content-toolbar">
            <label className="form-label">Content</label>
            <div className="content-toolbar-actions">
              <div className="toggle-switch" role="group" aria-label="Editor mode">
                <button
                  type="button"
                  className={`toggle-btn ${viewMode === 'edit' ? 'active' : ''}`}
                  aria-pressed={viewMode === 'edit'}
                  onClick={() => setViewMode('edit')}
                  title="Edit markdown"
                >
                  Edit
                </button>
                <button
                  type="button"
                  className={`toggle-btn ${viewMode === 'preview' ? 'active' : ''}`}
                  aria-pressed={viewMode === 'preview'}
                  onClick={() => setViewMode('preview')}
                  title="Preview formatted markdown"
                >
                  Preview
                </button>
                <button
                  type="button"
                  className={`toggle-btn ${viewMode === 'split' ? 'active' : ''}`}
                  aria-pressed={viewMode === 'split'}
                  onClick={() => setViewMode('split')}
                  title="Edit and preview side-by-side"
                >
                  Split
                </button>
              </div>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setCheatsheetOpen(true)}
                title="Markdown syntax help"
              >
                Cheatsheet
              </button>
            </div>
          </div>

          {viewMode === 'edit' && (
            <textarea
              ref={textareaRef}
              placeholder="Write your note here..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
              onKeyDown={handleKeyDown}
              onScroll={handleTextareaScroll}
              className="form-input form-textarea"
            />
          )}

          {viewMode === 'preview' && (
            <MarkdownPreview
              ref={previewRef}
              content={content}
              className="form-input markdown-preview-only"
              style={{ minHeight: 300, overflow: 'auto' }}
              onScroll={handlePreviewScroll}
            />
          )}

          {viewMode === 'split' && (
            <div className="split-container">
              <div className="split-pane split-pane-editor">
                <textarea
                  ref={textareaRef}
                  placeholder="Write your note here..."
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onScroll={handleTextareaScroll}
                  className="form-input form-textarea"
                />
              </div>
              <div className="split-pane split-pane-preview">
                <MarkdownPreview
                  ref={previewRef}
                  content={content}
                  className="markdown-pane"
                  style={{ height: '100%', overflow: 'auto' }}
                  onScroll={handlePreviewScroll}
                />
              </div>
            </div>
          )}

          <div className="char-count">
            {content.length} characters
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Tags</label>
          <input
            type="text"
            placeholder="comma, separated, tags"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            onKeyDown={handleKeyDown}
            className="form-input"
          />
        </div>

        <div className="editor-actions">
          <button type="submit" className="btn btn-primary">
            {isEditing ? 'Update' : 'Create'}
          </button>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onFindSimilar && onFindSimilar()}
            className="btn btn-secondary"
            title="Find notes similar to this one"
          >
            Find Similar
          </button>
          <span className="keyboard-hint">
            Ctrl+Enter or Ctrl+S to save • Esc to cancel
          </span>
        </div>
      </form>

      <MarkdownCheatsheet
        isOpen={cheatsheetOpen}
        onClose={() => setCheatsheetOpen(false)}
      />
    </div>
  );
});