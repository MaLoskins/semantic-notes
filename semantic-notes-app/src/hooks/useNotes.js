// hooks/useNotes.js
import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'semantic-notes-data';

export function useNotes() {
  const [notes, setNotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load from localStorage
  useEffect(() => {
    try {
      const storedData = localStorage.getItem(STORAGE_KEY);
      if (storedData) {
        const { notes = [] } = JSON.parse(storedData);
        setNotes(notes);
      }
    } catch (err) {
      console.error('Failed to load notes:', err);
      setError('Failed to load saved notes');
    } finally {
      setLoading(false);
    }
  }, []);

  // Save to localStorage
  useEffect(() => {
    if (!loading && notes.length > 0) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
          notes,
          lastUpdated: new Date().toISOString()
        }));
      } catch (err) {
        console.error('Failed to save notes:', err);
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
    const totalWords = notes.reduce((sum, note) => 
      sum + note.content.split(/\s+/).filter(Boolean).length, 0
    );
    
    const tagsSet = new Set();
    notes.forEach(note => {
      if (note.tags) {
        note.tags.split(',').forEach(tag => {
          const trimmed = tag.trim();
          if (trimmed) tagsSet.add(trimmed);
        });
      }
    });
    
    return {
      totalNotes: notes.length,
      totalWords,
      totalTags: tagsSet.size
    };
  }, [notes]);

  return {
    notes,
    loading,
    error,
    addNote,
    updateNote,
    deleteNote,
    getStats,
    exportNotes,
    setError
  };
}