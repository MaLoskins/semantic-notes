// hooks/useNotes.js
import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'semantic-notes-data';

export function useNotes() {
  const [notes, setNotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load notes from localStorage on mount
  useEffect(() => {
    try {
      const storedData = localStorage.getItem(STORAGE_KEY);
      if (storedData) {
        const parsedData = JSON.parse(storedData);
        setNotes(parsedData.notes || []);
      }
    } catch (err) {
      console.error('Failed to load notes from storage:', err);
      setError('Failed to load saved notes');
    } finally {
      setLoading(false);
    }
  }, []);

  // Save notes to localStorage whenever they change
  useEffect(() => {
    if (!loading && notes.length > 0) {
      try {
        const dataToStore = {
          notes,
          lastUpdated: new Date().toISOString()
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(dataToStore));
      } catch (err) {
        console.error('Failed to save notes to storage:', err);
        setError('Failed to save notes');
      }
    }
  }, [notes, loading]);

  // Add a new note
  const addNote = useCallback((noteData) => {
    const newNote = {
      id: Date.now(), // Simple ID generation
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      ...noteData
    };
    
    setNotes(prevNotes => [...prevNotes, newNote]);
    return newNote;
  }, []);

  // Update an existing note
  const updateNote = useCallback((index, noteData) => {
    setNotes(prevNotes => {
      const newNotes = [...prevNotes];
      newNotes[index] = {
        ...newNotes[index],
        ...noteData,
        updatedAt: new Date().toISOString()
      };
      return newNotes;
    });
  }, []);

  // Delete a note
  const deleteNote = useCallback((index) => {
    setNotes(prevNotes => prevNotes.filter((_, i) => i !== index));
  }, []);

  // Get a single note by index
  const getNote = useCallback((index) => {
    return notes[index] || null;
  }, [notes]);

  // Search notes
  const searchNotes = useCallback((searchTerm) => {
    if (!searchTerm) return notes;
    
    const term = searchTerm.toLowerCase();
    return notes.filter(note => 
      note.title.toLowerCase().includes(term) ||
      note.content.toLowerCase().includes(term) ||
      (note.tags && note.tags.toLowerCase().includes(term))
    );
  }, [notes]);

  // Get notes by tag
  const getNotesByTag = useCallback((tag) => {
    if (!tag) return notes;
    
    return notes.filter(note => 
      note.tags && note.tags.includes(tag)
    );
  }, [notes]);

  // Get all unique tags
  const getAllTags = useCallback(() => {
    const tagsSet = new Set();
    notes.forEach(note => {
      if (note.tags) {
        note.tags.split(',').forEach(tag => {
          const trimmedTag = tag.trim();
          if (trimmedTag) {
            tagsSet.add(trimmedTag);
          }
        });
      }
    });
    return Array.from(tagsSet).sort();
  }, [notes]);

  // Export notes as JSON
  const exportNotes = useCallback(() => {
    const dataToExport = {
      notes,
      exportedAt: new Date().toISOString(),
      version: '1.0'
    };
    
    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
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

  // Import notes from JSON
  const importNotes = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const importedData = JSON.parse(e.target.result);
          if (importedData.notes && Array.isArray(importedData.notes)) {
            setNotes(prevNotes => [...prevNotes, ...importedData.notes]);
            resolve(importedData.notes.length);
          } else {
            reject(new Error('Invalid file format'));
          }
        } catch (err) {
          reject(err);
        }
      };
      
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }, []);

  // Clear all notes
  const clearAllNotes = useCallback(() => {
    if (window.confirm('Are you sure you want to delete all notes? This action cannot be undone.')) {
      setNotes([]);
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  // Get statistics
  const getStats = useCallback(() => {
    const totalNotes = notes.length;
    const totalWords = notes.reduce((sum, note) => 
      sum + note.content.split(/\s+/).length, 0
    );
    const totalChars = notes.reduce((sum, note) => 
      sum + note.content.length, 0
    );
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
    getNote,
    searchNotes,
    getNotesByTag,
    getAllTags,
    exportNotes,
    importNotes,
    clearAllNotes,
    getStats,
    setError
  };
}