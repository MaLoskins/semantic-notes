// components/NoteEditor.jsx
import React, { useState, useEffect, useRef } from 'react';

export default function NoteEditor({ note, onSave, onCancel }) {
  const [formData, setFormData] = useState({
    title: '',
    content: '',
    tags: ''
  });
  
  const titleRef = useRef(null);

  useEffect(() => {
    setFormData({
      title: note?.title || '',
      content: note?.content || '',
      tags: note?.tags || ''
    });
    titleRef.current?.focus();
  }, [note]);

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!formData.title.trim() || !formData.content.trim()) {
      return;
    }

    onSave({
      ...note,
      ...formData,
      title: formData.title.trim(),
      content: formData.content.trim(),
      tags: formData.tags.trim()
    });
  };

  const handleChange = (field) => (e) => {
    setFormData(prev => ({ ...prev, [field]: e.target.value }));
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleSubmit(e);
    } else if (e.key === 'Escape') {
      onCancel();
    }
  };

  const isEditing = Boolean(note?.id);
  const isValid = formData.title.trim() && formData.content.trim();

  return (
    <div className="note-editor">
      <div className="editor-header">
        <h2>{isEditing ? 'Edit Note' : 'New Note'}</h2>
      </div>
      
      <form onSubmit={handleSubmit} className="editor-form">
        <div className="form-group">
          <label htmlFor="title" className="form-label">Title</label>
          <input
            id="title"
            ref={titleRef}
            type="text"
            placeholder="Enter note title"
            value={formData.title}
            onChange={handleChange('title')}
            onKeyDown={handleKeyDown}
            className="note-input"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="content" className="form-label">Content</label>
          <textarea
            id="content"
            placeholder="Write your note content here"
            value={formData.content}
            onChange={handleChange('content')}
            onKeyDown={handleKeyDown}
            className="note-input note-content-input"
            required
          />
          <div className="char-count">
            {formData.content.length} characters
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="tags" className="form-label">Tags</label>
          <input
            id="tags"
            type="text"
            placeholder="Enter tags separated by commas"
            value={formData.tags}
            onChange={handleChange('tags')}
            onKeyDown={handleKeyDown}
            className="note-input"
          />
        </div>

        <div className="editor-actions">
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={!isValid}
          >
            {isEditing ? 'Update' : 'Create'}
          </button>
          <button 
            type="button" 
            onClick={onCancel} 
            className="btn btn-secondary"
          >
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