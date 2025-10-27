import React, { useState, useMemo } from 'react';
import MarkdownPreview from './MarkdownPreview';

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
  searchTerm = '',
  onFindSimilar,
  searchMode = 'text',
  semanticResults = [],
  minSimilarity = 60,
  semanticLoading = false,
  semanticError = ''
}) {
  const [sortBy, setSortBy] = useState('updated');
  const [filterTag, setFilterTag] = useState('');

  // Map from note index to semantic result for quick lookup
  const semanticMap = useMemo(() => {
    const m = new Map();
    (semanticResults || []).forEach(r => {
      if (r && typeof r.index === 'number') m.set(r.index, r);
    });
    return m;
  }, [semanticResults]);

  // Compute a simple "why matched" snippet by finding the sentence with most token overlap
  function bestWhySnippet(text, query) {
    const content = String(text || '');
    const q = String(query || '').toLowerCase();
    if (!q) return '';
    const sentences = content.split(/(?<=[.!?])\s+/);
    const qTokens = new Set(q.split(/\W+/).filter(Boolean));
    let best = '';
    let bestScore = -1;
    for (const s of sentences) {
      const tokens = s.toLowerCase().split(/\W+/).filter(Boolean);
      if (tokens.length === 0) continue;
      let overlap = 0;
      for (const t of tokens) if (qTokens.has(t)) overlap++;
      const score = overlap / Math.max(1, tokens.length);
      if (score > bestScore) {
        bestScore = score;
        best = s;
      }
    }
    return best || sentences[0] || content.substring(0, PREVIEW_LENGTH);
  }

  const allTags = useMemo(() => extractTags(notes), [notes]);

  const processedNotes = useMemo(() => {
    // Semantic mode: build from semanticResults to preserve relevance ordering
    if (searchMode === 'semantic') {
      let arr = (semanticResults || []).map(r => {
        const n = notes[r.index];
        if (!n) return null;
        return { ...n, originalIndex: r.index, _sem: r };
      }).filter(Boolean);

      // Threshold filter
      arr = arr.filter(item => (item._sem?.percent ?? 0) >= minSimilarity);

      // Tag filter (keep intersection with selected tag)
      if (filterTag) {
        arr = arr.filter(note => note.tags?.includes(filterTag));
      }

      // Keep relevance ordering (semanticResults already sorted)
      return arr;
    }

    // Text mode: legacy keyword search
    let filtered = notes.map((note, index) => ({ ...note, originalIndex: index }));

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(note =>
        note.title.toLowerCase().includes(term) ||
        note.content.toLowerCase().includes(term) ||
        (note.tags && note.tags.toLowerCase().includes(term))
      );
    }

    if (filterTag) {
      filtered = filtered.filter(note => note.tags?.includes(filterTag));
    }

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
  }, [notes, searchTerm, filterTag, sortBy, searchMode, semanticResults, minSimilarity]);

  const handleDelete = (index, e) => {
    e.stopPropagation();
    onDelete(index);
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
        {searchMode === 'semantic' ? (
          <>
            <div className="empty-message-title">No semantic matches</div>
            <div className="empty-message-hint">
              {semanticError ? semanticError : 'Try lowering the similarity threshold or refining your query'}
            </div>
          </>
        ) : notes.length === 0 ? (
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
        {processedNotes.map((note) => {
          const full = String(note.content || '');
          const snippet = full.slice(0, PREVIEW_LENGTH); // safe char-slice (fast)
          return (
            <div
              key={note.originalIndex}
              className={`note-item ${selectedNote === note.originalIndex ? 'selected' : ''}`}
              onClick={() => onSelect(note.originalIndex)}
            >
              <div className="note-item-content">
                <div className="note-item-header">
                  <h4 className="note-item-title">{note.title}</h4>
                  {searchMode === 'semantic' && note._sem && (
                    <span className="similarity-badge">{note._sem.percent}% match</span>
                  )}
                  <span className="note-date">
                    {formatRelativeTime(note.updatedAt || note.createdAt)}
                  </span>
                </div>
                {searchMode === 'semantic' ? (
                  <div
                    className="why-match markdown-snippet markdown-snippet--compact"
                    style={{ maxHeight: 140, overflow: 'hidden' }}
                  >
                    <MarkdownPreview content={bestWhySnippet(full, searchTerm)} />
                  </div>
                ) : (
                  <div
                    className="note-preview markdown-snippet markdown-snippet--compact"
                    style={{ maxHeight: 140, overflow: 'hidden' }}
                  >
                    <MarkdownPreview content={snippet} />
                    {full.length > PREVIEW_LENGTH && <span className="truncate-ellipsis">…</span>}
                  </div>
                )}


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
                  onClick={(e) => {
                    e.stopPropagation();
                    onFindSimilar && onFindSimilar(note.originalIndex);
                  }}
                  className="action-btn"
                  title="Find Similar"
                >
                  Similar
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
          );
        })}
      </div>
    )}
  </div>
);
}