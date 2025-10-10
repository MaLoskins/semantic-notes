import React, { useState, useEffect, useCallback, useRef } from 'react';
import NoteEditor from './components/NoteEditor';
import NotesList from './components/NotesList';
import GraphVisualization from './components/GraphVisualization';
import { useNotes } from './hooks/useNotes';
import apiService from './services/api';
import './App.css';

const GRAPH_UPDATE_DEBOUNCE = 500;

export default function App() {
  const {
    notes,
    loading: notesLoading,
    error: notesError,
    addNote,
    updateNote,
    deleteNote,
    getStats,
    exportNotes
  } = useNotes();

  const [selectedNote, setSelectedNote] = useState(null);
  const [editingNote, setEditingNote] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [graphData, setGraphData] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  
  const graphRef = useRef(null);
  const updateTimerRef = useRef(null);

  // Check backend connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await apiService.checkHealth();
        setConnected(true);
        setError(null);
      } catch (err) {
        setConnected(false);
        setError('Backend unavailable. Ensure server is running on http://localhost:8000');
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (graphRef.current) {
        setDimensions({
          width: graphRef.current.clientWidth,
          height: graphRef.current.clientHeight
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Generate graph with debouncing
  useEffect(() => {
    if (updateTimerRef.current) {
      clearTimeout(updateTimerRef.current);
    }

    if (!connected || notes.length < 2) {
      setGraphData(null);
      return;
    }

    updateTimerRef.current = setTimeout(async () => {
      setGraphLoading(true);
      try {
        const documents = notes.map(note => 
          `${note.title}. ${note.content} ${note.tags || ''}`
        );
        
        const labels = notes.map(note => 
          note.title.length > 30 ? `${note.title.substring(0, 30)}...` : note.title
        );

        const graph = await apiService.buildGraph({
          documents,
          labels,
          mode: 'knn',
          top_k: Math.min(2, notes.length - 1),
          dr_method: 'pca'
        });

        setGraphData(graph);
        setError(null);
      } catch (err) {
        console.error('Graph generation failed:', err);
        setError(`Graph generation failed: ${err.message}`);
      } finally {
        setGraphLoading(false);
      }
    }, GRAPH_UPDATE_DEBOUNCE);

    return () => {
      if (updateTimerRef.current) {
        clearTimeout(updateTimerRef.current);
      }
    };
  }, [notes, connected]);

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

  const handleNodeClick = useCallback((nodeId) => {
    setSelectedNote(nodeId);
    setEditingNote(null);
    setIsCreating(false);
  }, []);

  const handleNewNote = useCallback(() => {
    setIsCreating(true);
    setEditingNote(null);
    setSelectedNote(null);
  }, []);

  const handleCancel = useCallback(() => {
    setIsCreating(false);
    setEditingNote(null);
  }, []);

  const stats = getStats();

  return (
    <div className="app">
      <header className="app-header">
        <h1>Semantic Notes</h1>
        
        <div className="header-actions">
          <div className="search-bar">
            <span className="search-icon">⌕</span>
            <input
              type="text"
              placeholder="Search notes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          <button onClick={handleNewNote} className="btn btn-primary">
            New Note
          </button>

          <button 
            onClick={exportNotes} 
            className="btn btn-secondary btn-icon" 
            title="Export Notes"
          >
            ↓
          </button>

          <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            <span className={`status-indicator ${connected ? '' : 'disconnected'}`} />
            {connected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </header>

      {(error || notesError) && (
        <div className="error-banner">
          ⚠ {error || notesError}
        </div>
      )}

      <div className="main-content">
        <div className="sidebar">
          {isCreating || editingNote ? (
            <NoteEditor
              note={editingNote}
              onSave={handleSaveNote}
              onCancel={handleCancel}
            />
          ) : (
            <>
              <NotesList
                notes={notes}
                onSelect={setSelectedNote}
                onEdit={handleEditNote}
                onDelete={handleDeleteNote}
                selectedNote={selectedNote}
                searchTerm={searchTerm}
              />
              
              {notes.length > 0 && (
                <div className="stats-bar">
                  <div>{stats.totalNotes} notes • {stats.totalWords} words</div>
                  <div>{stats.totalTags} unique tags</div>
                </div>
              )}
            </>
          )}
        </div>

        <div className="graph-container" ref={graphRef}>
          {notesLoading ? (
            <div className="loading">
              <div className="loading-spinner" />
              <div>Loading notes...</div>
            </div>
          ) : notes.length === 0 ? (
            <div className="empty-state">
              <h3>Welcome to Semantic Notes</h3>
              <p>Create your first note to get started.</p>
              <p style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
                Notes will be connected based on semantic similarity.
              </p>
            </div>
          ) : notes.length === 1 ? (
            <div className="empty-state">
              <p>Create at least 2 notes to visualize connections</p>
            </div>
          ) : graphLoading ? (
            <div className="loading">
              <div className="loading-spinner" />
              <div>Generating semantic graph...</div>
            </div>
          ) : graphData ? (
            <GraphVisualization
              graphData={graphData}
              onNodeClick={handleNodeClick}
              selectedNote={selectedNote}
              width={dimensions.width}
              height={dimensions.height}
            />
          ) : (
            <div className="empty-state">
              <p>Unable to generate graph. Check connection.</p>
            </div>
          )}

          {selectedNote !== null && notes[selectedNote] && (
            <div className="selected-note-preview fade-in">
              <h3>{notes[selectedNote].title}</h3>
              <p>{notes[selectedNote].content}</p>
              {notes[selectedNote].tags && (
                <div className="note-tags" style={{ marginTop: '0.5rem' }}>
                  {notes[selectedNote].tags.split(',').map((tag, i) => (
                    <span key={i} className="tag">
                      {tag.trim()}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}