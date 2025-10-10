// App.jsx - Main Application Component
import React, { useState, useEffect, useCallback } from 'react';
import NoteEditor from './components/NoteEditor';
import NotesList from './components/NotesList';
import GraphVisualization from './components/GraphVisualization';
import { useNotes } from './hooks/useNotes';
import apiService from './services/api';
import './App.css';

export default function App() {
  // Notes management
  const {
    notes,
    loading: notesLoading,
    error: notesError,
    addNote,
    updateNote,
    deleteNote,
    getStats,
    exportNotes,
    clearAllNotes
  } = useNotes();

  // UI state
  const [selectedNote, setSelectedNote] = useState(null);
  const [editingNote, setEditingNote] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [graphData, setGraphData] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [graphDimensions, setGraphDimensions] = useState({ width: 800, height: 600 });

  // Check backend connection on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await apiService.checkHealth();
        setConnected(true);
        setError(null);
        
        // Get backend stats
        const stats = await apiService.getStats();
        console.log('Backend stats:', stats);
      } catch (err) {
        setConnected(false);
        setError('Cannot connect to backend. Please ensure the server is running on http://localhost:8000');
      }
    };

    checkConnection();
    // Re-check connection every 10 seconds
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  // Update graph dimensions on window resize
  useEffect(() => {
    const updateDimensions = () => {
      const graphContainer = document.querySelector('.graph-container');
      if (graphContainer) {
        setGraphDimensions({
          width: graphContainer.clientWidth,
          height: graphContainer.clientHeight
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Generate graph when notes change or connection is established
  useEffect(() => {
    const generateGraph = async () => {
      if (!connected || notes.length < 2) {
        setGraphData(null);
        return;
      }

      setGraphLoading(true);
      try {
        // Combine title and content for better semantic matching
        const documents = notes.map(note => 
          `${note.title}. ${note.content} ${note.tags || ''}`
        );
        
        const labels = notes.map(note => 
          note.title.length > 30 
            ? note.title.substring(0, 30) + '...' 
            : note.title
        );

        const graph = await apiService.buildGraph({
          documents,
          labels,
          mode: 'knn',
          top_k: Math.min(2, notes.length - 1), // KNN with k=2 as requested
          dr_method: 'pca' // PCA as requested
        });

        setGraphData(graph);
        setError(null);
      } catch (err) {
        console.error('Failed to generate graph:', err);
        setError(`Failed to generate semantic graph: ${err.message}`);
      } finally {
        setGraphLoading(false);
      }
    };

    generateGraph();
  }, [notes, connected]);

  // Handle note operations
  const handleSaveNote = useCallback((noteData) => {
    if (noteData.id !== undefined) {
      // Editing existing note
      updateNote(editingNote.originalIndex, noteData);
      setEditingNote(null);
    } else {
      // Creating new note
      addNote(noteData);
      setIsCreating(false);
    }
  }, [editingNote, updateNote, addNote]);

  const handleEditNote = useCallback((index) => {
    const note = notes[index];
    setEditingNote({ ...note, originalIndex: index });
    setIsCreating(false);
    setSelectedNote(null);
  }, [notes]);

  const handleDeleteNote = useCallback((index) => {
    deleteNote(index);
    if (selectedNote === index) {
      setSelectedNote(null);
    }
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
      {/* Header */}
      <header className="app-header">
        <h1>
          🧠 Semantic Notes
        </h1>
        
        <div className="header-actions">
          <div className="search-bar">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              placeholder="Search notes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          <button onClick={handleNewNote} className="btn btn-primary">
            ➕ New Note
          </button>

          <button onClick={exportNotes} className="btn btn-secondary" title="Export Notes">
            💾
          </button>

          <span className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
        </div>
      </header>

      {/* Error Banner */}
      {(error || notesError) && (
        <div className="error-banner">
          ⚠️ {error || notesError}
        </div>
      )}

      {/* Main Content */}
      <div className="main-content">
        {/* Sidebar */}
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
                <div style={{ 
                  padding: '1rem', 
                  borderTop: '1px solid var(--border-color)',
                  fontSize: '0.75rem',
                  color: 'var(--text-dimmed)'
                }}>
                  <div>📊 {stats.totalNotes} notes • {stats.totalWords} words</div>
                  <div>🏷️ {stats.totalTags} unique tags</div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Graph Container */}
        <div className="graph-container">
          {notesLoading ? (
            <div className="loading">
              <div>Loading notes...</div>
            </div>
          ) : notes.length === 0 ? (
            <div className="empty-state">
              <h3>Welcome to Semantic Notes! 🚀</h3>
              <p>Create your first note to get started.</p>
              <p style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
                Your notes will be connected based on their semantic similarity.
              </p>
            </div>
          ) : notes.length === 1 ? (
            <div className="empty-state">
              <p>Create at least 2 notes to visualize their semantic connections</p>
            </div>
          ) : graphLoading ? (
            <div className="loading">
              <div>🔄 Generating semantic graph...</div>
            </div>
          ) : graphData ? (
            <GraphVisualization
              graphData={graphData}
              onNodeClick={handleNodeClick}
              selectedNote={selectedNote}
              width={graphDimensions.width}
              height={graphDimensions.height}
            />
          ) : (
            <div className="empty-state">
              <p>Unable to generate graph. Check your connection.</p>
            </div>
          )}

          {/* Selected Note Preview */}
          {selectedNote !== null && notes[selectedNote] && (
            <div className="selected-note-preview fade-in">
              <h3>{notes[selectedNote].title}</h3>
              <p>{notes[selectedNote].content}</p>
              {notes[selectedNote].tags && (
                <div style={{ marginTop: '0.5rem' }}>
                  {notes[selectedNote].tags.split(',').map((tag, i) => (
                    <span key={i} className="tag" style={{ marginRight: '0.25rem' }}>
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