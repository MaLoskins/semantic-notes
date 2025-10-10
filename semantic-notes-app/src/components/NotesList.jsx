// components/NotesList.jsx
import React, { useState, useMemo } from 'react';

const EditIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
    <path d="M10.586 1.586a2 2 0 112.828 2.828L4.828 13H2v-2.828l8.586-8.586z" 
          stroke="currentColor" strokeWidth="1" fill="none"/>
  </svg>
);

const DeleteIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
    <path d="M4 3h6m-.5 0v8a1 1 0 01-1 1h-3a1 1 0 01-1-1V3m1 0V2a1 1 0 011-1h1a1 1 0 011 1v1" 
          stroke="currentColor" strokeWidth="1" strokeLinecap="round" fill="none"/>
  </svg>
);

export default function NotesList({ 
  notes, 
  onSelect, 
  onEdit, 
  onDelete, 
  selectedNote,
  searchTerm,
  stats 
}) {
  const [sortBy, setSortBy] = useState('updated');
  const [filterTag, setFilterTag] = useState('');

  const allTags = useMemo(() => {
    const tags = new Set();
    notes.forEach(note => {
      if (note.tags) {
        note.tags.split(',').forEach(tag => tags.add(tag.trim()));
      }
    });
    return Array.from(tags).sort();
  }, [notes]);

  const filteredNotes = useMemo(() => {
    let filtered = notes.map((note, index) => ({ ...note, originalIndex: index }));

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(note => 
        note.title.toLowerCase().includes(term) ||
        note.content.toLowerCase().includes(term) ||
        note.tags?.toLowerCase().includes(term)
      );
    }

    if (filterTag) {
      filtered = filtered.filter(note => note.tags?.includes(filterTag));
    }

    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'updated':
          return new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt);
        case 'created':
          return new Date(b.createdAt) - new Date(a.createdAt);
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
    const diff = now - date;
    const days = Math.floor(diff / 86400000);
    
    if (days === 0) {
      const hours = Math.floor(diff / 3600000);
      if (hours === 0) {
        const mins = Math.floor(diff / 60000);
        return mins === 0 ? 'Just now' : `${mins}m ago`;
      }
      return `${hours}h ago`;
    }
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="notes-list">
      <div className="list-header">
        <h3 className="list-title">Notes ({filteredNotes.length})</h3>
        
        <div className="list-controls">
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="select-input"
            aria-label="Sort notes"
          >
            <option value="updated">Modified</option>
            <option value="created">Created</option>
            <option value="title">Title</option>
          </select>

          {allTags.length > 0 && (
            <select
              value={filterTag}
              onChange={(e) => setFilterTag(e.target.value)}
              className="select-input"
              aria-label="Filter by tag"
            >
              <option value="">All Tags</option>
              {allTags.map(tag => (
                <option key={tag} value={tag}>{tag}</option>
              ))}
            </select>
          )}
        </div>
      </div>

      <div className="notes-container">
        {filteredNotes.length === 0 ? (
          <div className="empty-state">
            {notes.length === 0 ? (
              <>
                <h3>No notes yet</h3>
                <p>Create your first note to get started</p>
              </>
            ) : (
              <>
                <h3>No matches found</h3>
                <p>Try adjusting your search or filters</p>
              </>
            )}
          </div>
        ) : (
          filteredNotes.map(note => (
            <article
              key={note.originalIndex}
              className={`note-item ${selectedNote === note.originalIndex ? 'selected' : ''}`}
              onClick={() => onSelect(note.originalIndex)}
            >
              <div className="note-header">
                <h4 className="note-title">{note.title}</h4>
                <time className="note-date">{formatDate(note.updatedAt || note.createdAt)}</time>
              </div>
              
              <p className="note-preview">
                {note.content}
              </p>
              
              {note.tags && (
                <div className="note-tags">
                  {note.tags.split(',').map((tag, i) => (
                    <span key={i} className="tag">{tag.trim()}</span>
                  ))}
                </div>
              )}
              
              <div className="note-actions">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onEdit(note.originalIndex);
                  }}
                  className="icon-btn"
                  title="Edit"
                  aria-label="Edit note"
                >
                  <EditIcon />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm('Delete this note?')) {
                      onDelete(note.originalIndex);
                    }
                  }}
                  className="icon-btn btn-danger"
                  title="Delete"
                  aria-label="Delete note"
                >
                  <DeleteIcon />
                </button>
              </div>
            </article>
          ))
        )}
      </div>

      {stats && notes.length > 0 && (
        <footer className="stats-footer">
          <span>{stats.totalNotes} notes</span>
          <span>{stats.totalWords} words</span>
          <span>{stats.totalTags} tags</span>
        </footer>
      )}
    </div>
  );
}