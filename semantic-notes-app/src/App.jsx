// App.jsx
import React, { useState, useEffect, useCallback } from 'react';
import NoteEditor from './components/NoteEditor';
import NotesList from './components/NotesList';
import GraphVisualization from './components/GraphVisualization';
import { useNotes } from './hooks/useNotes';
import { useGraph } from './hooks/useGraph';
import apiService from './services/api';
import './App.css';

// Icons as simple SVG components
const PlusIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    <path d="M8 2v12M2 8h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

const ExportIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    <path d="M8 10V2m0 0L5 5m3-3l3 3M3 10v3a1 1 0 001 1h8a1 1 0 001-1v-3" 
          stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
  </svg>
);

const SearchIcon = () => (
  <svg className="search-icon" viewBox="0 0 16 16" fill="currentColor">
    <path d="M7 1a6 6 0 104.472 10.207l3.321 3.321a1 1 0 001.414-1.414l-3.321-3.321A6 6 0 007 1zm0 2a4 4 0 110 8 4 4 0 010-8z"/>
  </svg>
);

export default function App() {
  const {
    notes,
    loading: notesLoading,
    error: notesError,
    addNote,
    updateNote,
    deleteNote,
    getStats,
    exportNotes,
  } = useNotes();

  const [selectedNote, setSelectedNote] = useState(null);
  const [editingNote, setEditingNote] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [connected, setConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);

  const { graphData, loading: graphLoading, error: graphError } = useGraph(notes, connected);

  // Check backend connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await apiService.checkHealth();
        setConnected(true);
        setConnectionError(null);
      } catch (err) {
        setConnected(false);
        setConnectionError('Backend server not available');
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  // Handle note operations
  const handleSaveNote = useCallback((noteData) => {
    if (editingNote) {
      updateNote(editingNote.originalIndex, noteData);
      setEditingNote(null);
    } else {
      addNote(noteData);
      setIsCreating(false);
    }
  }, [editingNote, updateNote, addNote]);

  const handleEditNote = useCallback((index) => {
    setEditingNote({ ...notes[index], originalIndex: index });
    setIsCreating(false);
    setSelectedNote(null);
  }, [notes]);

  const handleDeleteNote = useCallback((index) => {
    deleteNote(index);
    if (selectedNote === index) setSelectedNote(null);
  }, [deleteNote, selectedNote]);

  const handleNewNote = () => {
    setIsCreating(true);
    setEditingNote(null);
    setSelectedNote(null);
  };

  const handleCancel = () => {
    setIsCreating(false);
    setEditingNote(null);
  };

  const error = notesError || connectionError || graphError;
  const stats = getStats();

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Semantic Notes</h1>
        
        <div className="header-actions">
          <div className="search-bar">
            <SearchIcon />
            <input
              type="text"
              placeholder="Search notes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
              aria-label="Search notes"
            />
          </div>

          <button onClick={handleNewNote} className="btn btn-primary">
            <PlusIcon />
            New Note
          </button>

          <button 
            onClick={exportNotes} 
            className="btn btn-ghost" 
            title="Export Notes"
            aria-label="Export notes"
          >
            <ExportIcon />
          </button>

          <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            <span className="status-indicator" />
            {connected ? 'Connected' : 'Offline'}
          </div>
        </div>
      </header>

      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}

      <div className="main-content">
        <aside className="sidebar">
          {isCreating || editingNote ? (
            <NoteEditor
              note={editingNote}
              onSave={handleSaveNote}
              onCancel={handleCancel}
            />
          ) : (
            <NotesList
              notes={notes}
              onSelect={setSelectedNote}
              onEdit={handleEditNote}
              onDelete={handleDeleteNote}
              selectedNote={selectedNote}
              searchTerm={searchTerm}
              stats={stats}
            />
          )}
        </aside>

        <main className="graph-container">
          <GraphVisualization
            graphData={graphData}
            loading={notesLoading || graphLoading}
            notes={notes}
            selectedNote={selectedNote}
            onNodeClick={setSelectedNote}
          />
        </main>
      </div>
    </div>
  );
}