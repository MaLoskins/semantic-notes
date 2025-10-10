import React, { useState, useMemo } from 'react';

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

function extractTags(notes) {
  const tagSet = new Set();
  notes.forEach(note => {
    if (note.tags) {
      note.tags.split(',').forEach(tag => {
        const trimmed = tag.trim();
        if (trimmed) tagSet.add(trimmed);
      });
    }
  });
  return Array.from(tagSet).sort();
}

export default function NotesList({ 
  notes, 
  onSelect, 
  onEdit, 
  onDelete, 
  selectedNote,
  searchTerm = '' 
}) {
  const [sortBy, setSortBy] = useState('updated');
  const [filterTag, setFilterTag] = useState('');

  const allTags = useMemo(() => extractTags(notes), [notes]);

  const processedNotes = useMemo(() => {
    let filtered = notes.map((note, index) => ({ ...note, originalIndex: index }));

    // Search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(note => 
        note.title.toLowerCase().includes(term) ||
        note.content.toLowerCase().includes(term) ||
        (note.tags && note.tags.toLowerCase().includes(term))
      );
    }

    // Tag filter
    if (filterTag) {
      filtered = filtered.filter(note => note.tags?.includes(filterTag));
    }

    // Sort
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'updated':
          return new Date(b.updatedAt || b.createdAt || 0) - 
                 new Date(a.updatedAt || a.createdAt || 0);
        case 'created':
          return new Date(b.createdAt || 0) - new Date(a.createdAt || 0);
        case 'title':
          return a.title.localeCompare(b.title);
        default:
          return 0;
      }
    });

    return filtered;
  }, [notes, searchTerm, filterTag, sortBy]);

  const handleDelete = (index, e) => {
    e.stopPropagation();
    if (window.confirm('Delete this note?')) {
      onDelete(index);
    }
  };

  return (
    <div className="notes-list">
      <div className="list-header">
        <h3>Notes ({processedNotes.length})</h3>
        
        <div className="list-controls">
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="form-select"
          >
            <option value="updated">Recent</option>
            <option value="created">Created</option>
            <option value="title">Title</option>
          </select>

          {allTags.length > 0 && (
            <select
              value={filterTag}
              onChange={(e) => setFilterTag(e.target.value)}
              className="form-select"
            >
              <option value="">All Tags</option>
              {allTags.map(tag => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
          )}
        </div>
      </div>

      {processedNotes.length === 0 ? (
        <div className="empty-message">
          {notes.length === 0 ? (
            <>
              <div className="empty-message-title">No notes yet</div>
              <div className="empty-message-hint">Create your first note to get started</div>
            </>
          ) : (
            <>
              <div className="empty-message-title">No matches</div>
              <div className="empty-message-hint">Try different search terms or filters</div>
            </>
          )}
        </div>
      ) : (
        <div className="notes-items">
          {processedNotes.map((note) => (
            <div
              key={note.originalIndex}
              className={`note-item ${selectedNote === note.originalIndex ? 'selected' : ''}`}
              onClick={() => onSelect(note.originalIndex)}
            >
              <div className="note-item-content">
                <div className="note-item-header">
                  <h4 className="note-item-title">{note.title}</h4>
                  <span className="note-date">
                    {formatRelativeTime(note.updatedAt || note.createdAt)}
                  </span>
                </div>
                
                <p className="note-preview">
                  {note.content.substring(0, PREVIEW_LENGTH)}
                  {note.content.length > PREVIEW_LENGTH && '...'}
                </p>
                
                {note.tags && (
                  <div className="note-tags">
                    {note.tags.split(',').map((tag, i) => (
                      <span key={i} className="tag">
                        {tag.trim()}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              
              <div className="note-item-actions">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onEdit(note.originalIndex);
                  }}
                  className="action-btn"
                  title="Edit"
                >
                  Edit
                </button>
                <button
                  onClick={(e) => handleDelete(note.originalIndex, e)}
                  className="action-btn delete"
                  title="Delete"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}