// components/NotesList.jsx
import React, { useState, useMemo } from 'react';

export default function NotesList({ 
  notes, 
  onSelect, 
  onEdit, 
  onDelete, 
  selectedNote,
  searchTerm = '' 
}) {
  const [sortBy, setSortBy] = useState('updated'); // 'updated', 'created', 'title'
  const [filterTag, setFilterTag] = useState('');

  // Extract all unique tags
  const allTags = useMemo(() => {
    const tags = new Set();
    notes.forEach(note => {
      if (note.tags) {
        note.tags.split(',').forEach(tag => {
          tags.add(tag.trim());
        });
      }
    });
    return Array.from(tags);
  }, [notes]);

  // Filter and sort notes
  const processedNotes = useMemo(() => {
    let filtered = notes.map((note, index) => ({ ...note, originalIndex: index }));

    // Apply search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(note => 
        note.title.toLowerCase().includes(term) ||
        note.content.toLowerCase().includes(term) ||
        (note.tags && note.tags.toLowerCase().includes(term))
      );
    }

    // Apply tag filter
    if (filterTag) {
      filtered = filtered.filter(note => 
        note.tags && note.tags.includes(filterTag)
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'updated':
          return new Date(b.updatedAt || b.createdAt || 0) - new Date(a.updatedAt || a.createdAt || 0);
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

  const formatDate = (dateString) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      const diffHours = Math.floor(diffTime / (1000 * 60 * 60));
      if (diffHours === 0) {
        const diffMins = Math.floor(diffTime / (1000 * 60));
        return diffMins === 0 ? 'Just now' : `${diffMins}m ago`;
      }
      return `${diffHours}h ago`;
    } else if (diffDays === 1) {
      return 'Yesterday';
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const handleDelete = (index, e) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this note?')) {
      onDelete(index);
    }
  };

  return (
    <div className="notes-list">
      <div className="list-header">
        <h3>📝 Notes ({processedNotes.length})</h3>
        
        <div className="list-controls">
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="sort-select"
          >
            <option value="updated">Last Updated</option>
            <option value="created">Created Date</option>
            <option value="title">Title (A-Z)</option>
          </select>

          {allTags.length > 0 && (
            <select
              value={filterTag}
              onChange={(e) => setFilterTag(e.target.value)}
              className="filter-select"
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
        <div className="no-notes">
          {notes.length === 0 ? (
            <>
              <p>📭 No notes yet</p>
              <p className="hint">Create your first note to get started!</p>
            </>
          ) : (
            <>
              <p>No notes match your filters</p>
              <p className="hint">Try adjusting your search or filters</p>
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
                  <h4>{note.title}</h4>
                  <span className="note-date">{formatDate(note.updatedAt || note.createdAt)}</span>
                </div>
                
                <p className="note-preview">
                  {note.content.substring(0, 120)}
                  {note.content.length > 120 && '...'}
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
                  className="action-btn edit-btn"
                  title="Edit"
                >
                  ✏️
                </button>
                <button
                  onClick={(e) => handleDelete(note.originalIndex, e)}
                  className="action-btn delete-btn"
                  title="Delete"
                >
                  🗑️
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}