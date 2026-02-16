import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import dbApi from '../services/dbApi';

const STORAGE_KEY = 'semantic-notes-data';
const TRASH_STORAGE_KEY = 'semantic-notes-trash';

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

function loadTrashFromStorage() {
  try {
    const stored = localStorage.getItem(TRASH_STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      return parsed.trash || [];
    }
  } catch (err) {
    console.error('Failed to load trash:', err);
  }
  return [];
}

function saveTrashToStorage(trash) {
  try {
    localStorage.setItem(TRASH_STORAGE_KEY, JSON.stringify({
      trash,
      lastUpdated: new Date().toISOString()
    }));
  } catch (err) {
    console.error('Failed to save trash:', err);
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
  const { isAuthenticated } = useAuth();
  const [notes, setNotes] = useState([]);
  const [trashedNotes, setTrashedNotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const initNotes = async () => {
      if (isAuthenticated) {
        await loadNotesFromDatabase();
      } else {
        loadFromLocal();
      }
    };

    initNotes();
  }, [isAuthenticated]);

  const loadNotesFromDatabase = async () => {
    setLoading(true);
    setError(null);
    try {
      const dbNotes = await dbApi.fetchNotes();
      const dbTrash = await dbApi.fetchTrash();
      setNotes(dbNotes);
      setTrashedNotes(dbTrash);
      saveAllToStorage(dbNotes, dbTrash);
    } catch (err) {
      console.error('Failed to load from database, falling back to localStorage:', err);
      setError('Failed to sync with database');
      loadFromLocal();
    } finally {
      setLoading(false);
    }
  };

  const loadFromLocal = () => {
    const loadedNotes = loadFromStorage();
    const loadedTrash = loadTrashFromStorage();
    setNotes(loadedNotes);
    setTrashedNotes(loadedTrash);
    setLoading(false);
  };

  const saveAllToStorage = (notesData, trashData) => {
    try {
      saveToStorage(notesData);
      saveTrashToStorage(trashData);
    } catch (err) {
      console.error('Failed to cache notes locally:', err);
    }
  };

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

  useEffect(() => {
    if (!loading) {
      try {
        saveTrashToStorage(trashedNotes);
        setError(null);
      } catch (err) {
        setError(err.message);
      }
    }
  }, [trashedNotes, loading]);

  const addNote = useCallback(async (noteData) => {
    if (!isAuthenticated) {
      const newNote = {
        id: Date.now(),
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        ...noteData
      };
      setNotes(prev => [...prev, newNote]);
      return newNote;
    }

    try {
      const dbNote = await dbApi.createNote({
        title: noteData.title,
        content: noteData.content,
        tags: noteData.tags || ''
      });
      setNotes(prev => [dbNote, ...prev]);
      return dbNote;
    } catch (err) {
      console.error('Failed to create note:', err);
      setError('Failed to create note');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const updateNote = useCallback(async (index, noteData) => {
    if (!isAuthenticated) {
      setNotes(prev => {
        const updated = [...prev];
        updated[index] = {
          ...updated[index],
          ...noteData,
          updatedAt: new Date().toISOString()
        };
        return updated;
      });
      return;
    }
    const note = notes[index];
    if (!note) return;
    try {
      const updatedNote = await dbApi.updateNote(note.id, {
        title: noteData.title,
        content: noteData.content,
        tags: noteData.tags || ''
      });
      const newList = notes.map((n, i) => (i === index ? updatedNote : n));
      setNotes(newList);
    } catch (err) {
      console.error('Failed to update note:', err);
      setError('Failed to update note');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  // Trash management
  const moveToTrash = useCallback(async (index) => {
    const note = notes[index];
    if (!note) return;
    if (!isAuthenticated) {
      let movedNote = null;
      setNotes(prev => {
        const n = prev[index];
        movedNote = n;
        setTrashedNotes(trash => [
          { ...n, deletedAt: new Date().toISOString() },
          ...trash
        ]);
        return prev.filter((_, i) => i !== index);
      });
      return movedNote;
    }

    try {
      await dbApi.moveToTrash(note.id);
      const trashedNote = { ...note, is_deleted: true, deleted_at: new Date().toISOString() };
      setNotes(prev => prev.filter((_, i) => i !== index));
      setTrashedNotes(prev => [trashedNote, ...prev]);
      return trashedNote;
    } catch (err) {
      console.error('Failed to move to trash:', err);
      setError('Failed to move note to trash');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const restoreFromTrash = useCallback(async (id) => {
    if (!isAuthenticated) {
      let restored = null;
      setTrashedNotes(prev => {
        const idx = prev.findIndex(n => n.id === id);
        if (idx === -1) return prev;
        restored = { ...prev[idx] };
        const updatedTrash = prev.filter((_, i) => i !== idx);
        setNotes(notesPrev => [
          ...notesPrev,
          {
            id: restored.id,
            title: restored.title,
            content: restored.content,
            tags: restored.tags,
            createdAt: restored.createdAt,
            updatedAt: new Date().toISOString()
          }
        ]);
        return updatedTrash;
      });
      return restored;
    }

    try {
      await dbApi.restoreNote(id);
      const note = trashedNotes.find(n => n.id === id);
      if (!note) return;
      const restoredNote = { ...note, is_deleted: false, deleted_at: null };
      const updatedTrash = trashedNotes.filter(n => n.id !== id);
      setTrashedNotes(updatedTrash);
      setNotes(prev => [restoredNote, ...prev]);
    } catch (err) {
      console.error('Failed to restore note:', err);
      setError('Failed to restore note');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const permanentDelete = useCallback(async (id) => {
    if (!isAuthenticated) {
      let deleted = false;
      setTrashedNotes(prev => {
        const next = prev.filter(n => {
          if (!deleted && n.id === id) deleted = true;
          return n.id !== id;
        });
        return next;
      });
      return deleted;
    }

    try {
      await dbApi.permanentDelete(id);
      const updatedTrash = trashedNotes.filter(n => n.id !== id);
      setTrashedNotes(updatedTrash);
    } catch (err) {
      console.error('Failed to permanently delete note:', err);
      setError('Failed to permanently delete note');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const emptyTrash = useCallback(async () => {
    if (!isAuthenticated) {
      setTrashedNotes([]);
      return;
    }

    try {
      await dbApi.emptyTrash();
      setTrashedNotes([]);
    } catch (err) {
      console.error('Failed to empty trash:', err);
      setError('Failed to empty trash');
      throw err;
    }
  }, [isAuthenticated, notes]);
  
  const importNotes = useCallback(async (incomingNotes) => {
    if (!Array.isArray(incomingNotes)) {
      throw new Error('Invalid import format: expected notes array');
    }
    try {
      const response = await dbApi.importNotes(incomingNotes);
      const { notes: importedNotes, id_mapping } = response;

      // Apply new IDs using mapping
      const updatedNotes = incomingNotes.map(note => {
        const newId = id_mapping[note.id];
        return newId ? { ...note, id: newId } : note;
      });

      // Update state and local storage
      setNotes(updatedNotes);

      return { success: true, imported: importedNotes.length };
    } catch (err) {
      console.error('Failed to import notes:', err);
      setError('Failed to import notes');
      throw err;
    }
  }, [trashedNotes]);
  
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
      averageNoteLength: totalNotes > 0 ? Math.round(totalChars / totalNotes) : 0,
      trashCount: trashedNotes.length
    };
  }, [notes, getAllTags, trashedNotes]);

  return {
    notes,
    trashedNotes,
    loading,
    error,
    addNote,
    updateNote,
    moveToTrash,
    restoreFromTrash,
    permanentDelete,
    emptyTrash,
    getAllTags,
    exportNotes,
    importNotes,
    getStats
  };
}