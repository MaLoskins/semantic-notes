import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'semantic-notes-data';

function loadFromStorage() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return parsed.notes || [];
    }
  } catch (err) {
    console.error('Failed to load notes:', err);
  }
  return [];
}

function saveToStorage(notes) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      notes,
      lastUpdated: new Date().toISOString()
    }));
  } catch (err) {
    console.error('Failed to save notes:', err);
    throw new Error('Storage quota exceeded');
  }
}

function generateRandomId() {
  return Date.now() + Math.floor(Math.random() * 100000);
}

function normalizeNote(n) {
  if (typeof n !== 'object' || n === null) {
    throw new Error('Invalid note format');
  }
  const title = String(n.title ?? '').trim();
  const content = String(n.content ?? '').trim();
  if (!title || !content) {
    throw new Error('Each note must have title and content');
  }
  const tags = n.tags != null ? String(n.tags) : '';
  let id = n.id;
  if (id == null || (typeof id !== 'number' && typeof id !== 'string') || !Number.isFinite(Number(id))) {
    id = generateRandomId();
  } else {
    id = Number(id);
  }
  const createdAt = n.createdAt ? new Date(n.createdAt).toISOString() : new Date().toISOString();
  const updatedAt = n.updatedAt ? new Date(n.updatedAt).toISOString() : createdAt;
  return { id, title, content, tags, createdAt, updatedAt };
}

export function useNotes() {
  const [notes, setNotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setNotes(loadFromStorage());
    setLoading(false);
  }, []);

  useEffect(() => {
    if (!loading) {
      try {
        saveToStorage(notes);
        setError(null);
      } catch (err) {
        setError(err.message);
      }
    }
  }, [notes, loading]);

  const addNote = useCallback((noteData) => {
    const newNote = {
      id: Date.now(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      ...noteData
    };
    setNotes(prev => [...prev, newNote]);
    return newNote;
  }, []);

  const updateNote = useCallback((index, noteData) => {
    setNotes(prev => {
      const updated = [...prev];
      updated[index] = {
        ...updated[index],
        ...noteData,
        updatedAt: new Date().toISOString()
      };
      return updated;
    });
  }, []);

  const deleteNote = useCallback((index) => {
    setNotes(prev => prev.filter((_, i) => i !== index));
  }, []);
  
  const importNotes = useCallback((incomingNotes, mode = 'merge') => {
    if (!Array.isArray(incomingNotes)) {
      throw new Error('Invalid import format: expected notes array');
    }
    const normalized = incomingNotes.map(n => normalizeNote(n));
    if (mode === 'replace') {
      setNotes(normalized);
      return { imported: normalized.length, mode };
    }
    if (mode === 'merge') {
      setNotes(prev => {
        const existingIds = new Set(prev.map(n => n.id));
        const merged = [...prev];
        for (const note of normalized) {
          let id = note.id;
          if (existingIds.has(id)) {
            do {
              id = generateRandomId();
            } while (existingIds.has(id));
          }
          existingIds.add(id);
          merged.push({ ...note, id });
        }
        return merged;
      });
      return { imported: normalized.length, mode };
    }
    throw new Error('Invalid import mode');
  }, []);
  
  const searchNotes = useCallback((term) => {
    if (!term) return notes;
    const lower = term.toLowerCase();
    return notes.filter(note => 
      note.title.toLowerCase().includes(lower) ||
      note.content.toLowerCase().includes(lower) ||
      (note.tags && note.tags.toLowerCase().includes(lower))
    );
  }, [notes]);

  const getAllTags = useCallback(() => {
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
  }, [notes]);

  const exportNotes = useCallback(() => {
    const data = {
      notes,
      exportedAt: new Date().toISOString(),
      version: '1.0'
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `semantic-notes-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [notes]);

  const getStats = useCallback(() => {
    const totalNotes = notes.length;
    const totalWords = notes.reduce((sum, note) => 
      sum + note.content.split(/\s+/).filter(w => w).length, 0
    );
    const totalChars = notes.reduce((sum, note) => sum + note.content.length, 0);
    const tags = getAllTags();
    
    return {
      totalNotes,
      totalWords,
      totalChars,
      totalTags: tags.length,
      averageNoteLength: totalNotes > 0 ? Math.round(totalChars / totalNotes) : 0
    };
  }, [notes, getAllTags]);

  return {
    notes,
    loading,
    error,
    addNote,
    updateNote,
    deleteNote,
    searchNotes,
    getAllTags,
    exportNotes,
    importNotes,
    getStats
  };
}