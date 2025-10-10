// components/NoteEditor.jsx
import React, { useState, useEffect } from 'react';

export default function NoteEditor({ note, onSave, onCancel }) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [tags, setTags] = useState('');

  useEffect(() => {
    if (note) {
      setTitle(note.title || '');
      setContent(note.content || '');
      setTags(note.tags || '');
    } else {
      setTitle('');
      setContent('');
      setTags('');
    }
  }, [note]);

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!title.trim()) {
      alert('Please provide a title for your note');
      return;
    }
    
    if (!content.trim()) {
      alert('Please provide content for your note');
      return;
    }

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

    onSave(noteData);
  };

  const handleKeyDown = (e) => {
    // Submit on Ctrl/Cmd + Enter
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleSubmit(e);
    }
    // Cancel on Escape
    if (e.key === 'Escape') {
      onCancel();
    }
  };

  const isEditing = note?.id !== undefined;

  return (
    <div className="note-editor">
      <div className="editor-header">
        <h2>{isEditing ? 'Edit Note' : 'New Note'}</h2>
      </div>
      
      <form onSubmit={handleSubmit} className="editor-form">
        <div className="form-group">
          <input
            type="text"
            placeholder="Note title..."
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={handleKeyDown}
            className="note-title-input"
            autoFocus
          />
        </div>

        <div className="form-group">
          <textarea
            placeholder="Write your note here... (Markdown supported)"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            onKeyDown={handleKeyDown}
            className="note-content-input"
            rows={12}
          />
          <div className="char-count">
            {content.length} characters
          </div>
        </div>

        <div className="form-group">
          <input
            type="text"
            placeholder="Tags (comma-separated, optional)"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            onKeyDown={handleKeyDown}
            className="note-tags-input"
          />
        </div>

        <div className="editor-actions">
          <button type="submit" className="btn btn-primary">
            {isEditing ? 'Update Note' : 'Create Note'}
          </button>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
          <span className="keyboard-hint">
            Ctrl+Enter to save • Esc to cancel
          </span>
        </div>
      </form>
    </div>
  );
}