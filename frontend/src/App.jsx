import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from './contexts/AuthContext';
import AuthGuard from './components/AuthGuard';
import NoteEditor from './components/NoteEditor';
import NotesList from './components/NotesList';
import GraphVisualization from './components/GraphVisualization';
import { useNotes } from './hooks/useNotes';
import apiService from './services/api';
import './App.css';
import ImportConfirmModal from './components/ImportConfirmModal';
import ToastNotification from './components/ToastNotification';
import TrashView from './components/TrashView';
import UnsavedChangesDialog from './components/UnsavedChangesDialog';
import SimilarNotesModal from './components/SimilarNotesModal';
import ImportLocalNotesModal from './components/ImportLocalNotesModal';
const GRAPH_UPDATE_DEBOUNCE = 500;
const SEMANTIC_QUERY_DEBOUNCE = 500;
const MIN_SEM_QUERY_LEN = 3;

const GC_LS_KEY = 'graph-controls-prefs-v1';
const DEFAULT_GRAPH_PARAMS = {
  connection: 'knn',
  k_neighbors: 5,
  similarity_threshold: 0.7,
  dim_reduction: 'pca',
  clustering: null,
  n_clusters: 5,
};
const clamp = (n, min, max) => Math.min(max, Math.max(min, n));

export default function App() {
  const {
    notes,
    trashedNotes,
    loading: notesLoading,
    error: notesError,
    addNote,
    updateNote,
    deleteNote,
    moveToTrash,
    restoreFromTrash,
    permanentDelete,
    emptyTrash,
    getStats,
    exportNotes,
    importNotes
  } = useNotes();

  const [selectedNote, setSelectedNote] = useState(null);
  const [editingNote, setEditingNote] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  // Search controls
  const [searchMode, setSearchMode] = useState('text'); // 'text' | 'semantic'
  const [minSimilarity, setMinSimilarity] = useState(60); // 0-100%
  const [semanticResults, setSemanticResults] = useState([]); // [{index, score, percent}]
  const [semanticLoading, setSemanticLoading] = useState(false);
  const [semanticError, setSemanticError] = useState('');
  const [graphData, setGraphData] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [showImportModal, setShowImportModal] = useState(false);
  const [pendingImportedNotes, setPendingImportedNotes] = useState([]);
  const [successMessage, setSuccessMessage] = useState('');
 
  const [activeTab, setActiveTab] = useState('notes');
  const [toastOpen, setToastOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [lastTrashedId, setLastTrashedId] = useState(null);

  // Similar notes modal state
  const [similarOpen, setSimilarOpen] = useState(false);
  const [similarBaseDoc, setSimilarBaseDoc] = useState('');
  const [similarBaseTitle, setSimilarBaseTitle] = useState('');
  const [similarExcludeIndex, setSimilarExcludeIndex] = useState(null);
  
  const graphRef = useRef(null);
  const updateTimerRef = useRef(null);
  const fileInputRef = useRef(null);
  const editorRef = useRef(null);
  const [isEditorDirty, setIsEditorDirty] = useState(false);
  const [unsavedOpen, setUnsavedOpen] = useState(false);
  const [pendingAction, setPendingAction] = useState(null);
  const semanticTimerRef = useRef(null);

  // Graph controls (persisted)
  const [graphParams, setGraphParams] = useState(() => {
    try {
      const raw = localStorage.getItem(GC_LS_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      return { ...DEFAULT_GRAPH_PARAMS, ...(parsed || {}) };
    } catch {
      return DEFAULT_GRAPH_PARAMS;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(GC_LS_KEY, JSON.stringify(graphParams));
    } catch { /* ignore quota */ }
  }, [graphParams]);

  const handleControlsChange = useCallback((partial) => {
    setGraphParams((prev) => ({ ...prev, ...partial }));
  }, []);

  const handleControlsReset = useCallback(() => {
    setGraphParams(DEFAULT_GRAPH_PARAMS);
  }, []);
 
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
 
  // Auto-clear success messages
  useEffect(() => {
    if (!successMessage) return;
    const t = setTimeout(() => setSuccessMessage(''), 4000);
    return () => clearTimeout(t);
  }, [successMessage]);
 
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

        const connection = graphParams.connection === 'threshold' ? 'threshold' : 'knn';
        const kNeighborsRaw = clamp(parseInt(graphParams.k_neighbors ?? 5, 10), 1, 10);
        const kNeighbors = Math.min(kNeighborsRaw, Math.max(1, notes.length - 1));
        const similarity_threshold = Math.max(0, Math.min(1, Number(graphParams.similarity_threshold ?? 0.7)));
        const dim_reduction = graphParams.dim_reduction === 'none' ? null : (graphParams.dim_reduction ?? 'pca');
        const clustering = graphParams.clustering ?? null;
        const n_clusters = clustering ? clamp(parseInt(graphParams.n_clusters ?? 5, 10), 2, 20) : undefined;

        const graph = await apiService.buildGraph({
          documents,
          labels,
          connection,
          k_neighbors: connection === 'knn' ? kNeighbors : undefined,
          similarity_threshold: connection === 'threshold' ? similarity_threshold : undefined,
          dim_reduction,
          clustering,
          n_clusters,
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
  }, [notes, connected, graphParams]);

  // Semantic Search (debounced)
  useEffect(() => {
    if (searchMode !== 'semantic') {
      setSemanticLoading(false);
      setSemanticError('');
      setSemanticResults([]);
      return;
    }
    const q = String(searchTerm || '').trim();
    if (!q) {
      setSemanticResults([]);
      setSemanticLoading(false);
      setSemanticError('');
      return;
    }
    if (q.length < MIN_SEM_QUERY_LEN) {
      setSemanticResults([]);
      setSemanticLoading(false);
      setSemanticError(`Type at least ${MIN_SEM_QUERY_LEN} characters for semantic search`);
      return;
    }
    if (!connected) {
      setSemanticResults([]);
      setSemanticLoading(false);
      setSemanticError('Semantic search requires backend connection');
      return;
    }

    if (semanticTimerRef.current) {
      clearTimeout(semanticTimerRef.current);
    }
    let cancelled = false;
    semanticTimerRef.current = setTimeout(async () => {
      setSemanticLoading(true);
      setSemanticError('');
      try {
        const [noteEmbs, queryEmb] = await Promise.all([
          apiService.getEmbeddingsForNotes(notes),
          apiService.embedText(q),
        ]);
        const scored = [];
        for (let i = 0; i < notes.length; i++) {
          const v = noteEmbs[i];
          if (!Array.isArray(v)) continue;
          const s = apiService.cosineSimilarity(queryEmb, v);
          scored.push({ index: i, score: s, percent: Math.round(s * 100) });
        }
        scored.sort((a, b) => b.score - a.score);
        if (!cancelled) setSemanticResults(scored);
      } catch (e) {
        console.error('Semantic search failed:', e);
        if (!cancelled) setSemanticError(e?.message || 'Semantic search failed');
      } finally {
        if (!cancelled) setSemanticLoading(false);
      }
    }, SEMANTIC_QUERY_DEBOUNCE);

    return () => {
      cancelled = true;
      if (semanticTimerRef.current) clearTimeout(semanticTimerRef.current);
    };
  }, [searchMode, searchTerm, notes, connected]);

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
    // If there are unsaved changes in the current editor, prompt first
    setSelectedNote(prev => prev); // no-op to keep dependencies minimal
    const action = { type: 'edit', index };
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction(action);
      setUnsavedOpen(true);
    } else {
      setEditingNote({ ...notes[index], originalIndex: index });
      setIsCreating(false);
      setSelectedNote(null);
    }
  }, [notes, isCreating, editingNote, isEditorDirty]);

  const handleDeleteNote = useCallback((index) => {
    const moved = moveToTrash(index);
    if (moved) {
      setLastTrashedId(moved.id);
      setToastMessage('Note moved to trash. Undo?');
      setToastOpen(true);
    }
    if (selectedNote === index) setSelectedNote(null);
  }, [moveToTrash, selectedNote]);

  const handleUndo = useCallback(() => {
    if (lastTrashedId != null) {
      restoreFromTrash(lastTrashedId);
      setLastTrashedId(null);
    }
  }, [lastTrashedId, restoreFromTrash]);

  const handleNodeClick = useCallback((nodeId) => {
    const action = { type: 'nodeSelect', index: nodeId };
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction(action);
      setUnsavedOpen(true);
    } else {
      setSelectedNote(nodeId);
      setEditingNote(null);
      setIsCreating(false);
    }
  }, [isCreating, editingNote, isEditorDirty]);

  const handleNewNote = useCallback(() => {
    const action = { type: 'new' };
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction(action);
      setUnsavedOpen(true);
    } else {
      setIsCreating(true);
      setEditingNote(null);
      setSelectedNote(null);
    }
  }, [isCreating, editingNote, isEditorDirty]);

  const handleCancel = useCallback(() => {
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction({ type: 'cancel' });
      setUnsavedOpen(true);
    } else {
      setIsCreating(false);
      setEditingNote(null);
    }
  }, [isCreating, editingNote, isEditorDirty]);
 
  const handleImportClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);
 
  const handleFileSelected = useCallback((e) => {
    try {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      if (!file.name.toLowerCase().endsWith('.json')) {
        setError('Please select a JSON file');
        e.target.value = '';
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const text = String(reader.result || '');
          const parsed = JSON.parse(text);
          const incoming = Array.isArray(parsed) ? parsed : (parsed && Array.isArray(parsed.notes) ? parsed.notes : null);
          if (!incoming) {
            throw new Error('Invalid file format. Expected { "notes": [...] }');
          }
          if (!Array.isArray(incoming)) {
            throw new Error('Invalid notes format in file');
          }
          if (incoming.length === 0) {
            setError('No notes found in file');
            e.target.value = '';
            return;
          }
          setPendingImportedNotes(incoming);
          setShowImportModal(true);
          setError(null);
        } catch (err) {
          console.error('Import parse error:', err);
          setError(`Import failed: ${err.message}`);
        } finally {
          e.target.value = '';
        }
      };
      reader.onerror = () => {
        setError('Failed to read file');
        e.target.value = '';
      };
      reader.readAsText(file);
    } catch (err) {
      setError(`Import failed: ${err.message}`);
      if (e?.target) e.target.value = '';
    }
  }, []);
 
  const confirmReplace = useCallback(() => {
    try {
      const res = importNotes(pendingImportedNotes, 'replace');
      setSuccessMessage(`Imported ${res.imported} notes successfully`);
    } catch (err) {
      setError(`Import failed: ${err.message}`);
    } finally {
      setShowImportModal(false);
      setPendingImportedNotes([]);
      setSelectedNote(null);
      setEditingNote(null);
      setIsCreating(false);
    }
  }, [importNotes, pendingImportedNotes]);
 
  const confirmMerge = useCallback(() => {
    try {
      const res = importNotes(pendingImportedNotes, 'merge');
      setSuccessMessage(`Imported ${res.imported} notes successfully`);
    } catch (err) {
      setError(`Import failed: ${err.message}`);
    } finally {
      setShowImportModal(false);
      setPendingImportedNotes([]);
    }
  }, [importNotes, pendingImportedNotes]);
 
  // Navigation helpers that respect unsaved changes
  const executeAction = useCallback((action) => {
    if (!action) return;
    switch (action.type) {
      case 'cancel':
        setIsCreating(false);
        setEditingNote(null);
        break;
      case 'new':
        setIsCreating(true);
        setEditingNote(null);
        setSelectedNote(null);
        break;
      case 'edit': {
        const idx = action.index;
        if (idx != null && notes[idx]) {
          setEditingNote({ ...notes[idx], originalIndex: idx });
          setIsCreating(false);
          setSelectedNote(null);
        }
        break;
      }
      case 'selectNote':
      case 'nodeSelect': {
        const idx = action.index;
        if (idx != null) {
          setSelectedNote(idx);
          setEditingNote(null);
          setIsCreating(false);
        }
        break;
      }
      case 'tab':
        setActiveTab(action.tab);
        break;
      default:
        break;
    }
  }, [notes]);
  
  const requestNavigate = useCallback((action) => {
    const editingActive = isCreating || !!editingNote;
    if (editingActive && isEditorDirty) {
      setPendingAction(action);
      setUnsavedOpen(true);
    } else {
      executeAction(action);
    }
  }, [isCreating, editingNote, isEditorDirty, executeAction]);
  
  const handleTabChange = useCallback((tab) => {
    requestNavigate({ type: 'tab', tab });
  }, [requestNavigate]);

  // ---------- Find Similar Notes helpers ----------
  const buildDocText = useCallback((n) => {
    const t = String(n?.title || '').trim();
    const c = String(n?.content || '').trim();
    const g = String(n?.tags || '').trim();
    return `${t}. ${c} ${g}`.trim();
  }, []);

  const openSimilar = useCallback((baseDoc, baseTitle, excludeIndex = null) => {
    setSimilarBaseDoc(baseDoc);
    setSimilarBaseTitle(baseTitle);
    setSimilarExcludeIndex(excludeIndex);
    setSimilarOpen(true);
  }, []);

  const handleFindSimilarFromEditor = useCallback(() => {
    let data = null;
    if (editorRef.current && typeof editorRef.current.getCurrentData === 'function') {
      data = editorRef.current.getCurrentData();
    } else if (editingNote) {
      data = editingNote;
    }
    if (!data) {
      setError('No note data to analyze');
      return;
    }
    const doc = buildDocText(data);
    const title = String(data.title || 'This note');
    const exclude = editingNote?.originalIndex ?? null;
    openSimilar(doc, title, exclude);
  }, [editorRef, editingNote, buildDocText, openSimilar]);

  const handleFindSimilarFromList = useCallback((index) => {
    const n = notes[index];
    if (!n) return;
    const doc = buildDocText(n);
    openSimilar(doc, n.title || 'This note', index);
  }, [notes, buildDocText, openSimilar]);

  const LINKS_KEY = 'semantic-links-v1';
  const addLink = useCallback((aId, bId) => {
    try {
      if (aId == null || bId == null) {
        setError('Unable to link: missing note id(s). Save the note first.');
        return;
      }
      const raw = localStorage.getItem(LINKS_KEY);
      const arr = raw ? JSON.parse(raw) : [];
      const pair = aId < bId ? [aId, bId] : [bId, aId];
      const exists = Array.isArray(arr) && arr.some((p) => Array.isArray(p) && p[0] === pair[0] && p[1] === pair[1]);
      const next = exists ? arr : [...(Array.isArray(arr) ? arr : []), pair];
      localStorage.setItem(LINKS_KEY, JSON.stringify(next));
      setSuccessMessage('Notes linked');
    } catch (e) {
      setError('Failed to save link');
    }
  }, []);
 
  const saveAndContinue = useCallback(() => {
    if (!editorRef.current) return;
    const data = editorRef.current.getCurrentData();
    const titleOk = String(data.title || '').trim().length > 0;
    const contentOk = String(data.content || '').trim().length > 0;
    if (!titleOk || !contentOk) {
      alert('Title and Content are required');
      return;
    }
    handleSaveNote(data);
    setUnsavedOpen(false);
    if (pendingAction) {
      executeAction(pendingAction);
      setPendingAction(null);
    }
  }, [handleSaveNote, pendingAction, executeAction]);
  
  const discardChanges = useCallback(() => {
    setUnsavedOpen(false);
    // Close the editor and proceed
    setIsCreating(false);
    setEditingNote(null);
    if (pendingAction) {
      executeAction(pendingAction);
      setPendingAction(null);
    }
  }, [pendingAction, executeAction]);
  
  const cancelDialog = useCallback(() => {
    setUnsavedOpen(false);
  }, []);
  
  const stats = getStats();

  const { user, logout, isAuthenticated } = useAuth();

   // LocalStorage import modal logic
   const [showImportLocalModal, setShowImportLocalModal] = useState(false);
   useEffect(() => {
     if (isAuthenticated && user) {
       const hasCheckedImport = localStorage.getItem('import-checked');
       if (!hasCheckedImport) {
         const notesData = localStorage.getItem('semantic-notes-data');
         const trashData = localStorage.getItem('semantic-notes-trash');
         if (notesData || trashData) {
           setShowImportLocalModal(true);
         }
         localStorage.setItem('import-checked', 'true');
       }
     }
   }, [isAuthenticated, user]);

   const handleImportComplete = (importedCount) => {
     setShowImportLocalModal(false);
     window.location.reload();
   };

   const handleImportSkip = () => {
     setShowImportLocalModal(false);
   };

  return (
    <AuthGuard>
       {showImportLocalModal && (
         <ImportLocalNotesModal
           onClose={handleImportSkip}
           onImportComplete={handleImportComplete}
         />
       )}
      <div className="app">
        <header className="app-header">
        <h1>Semantic Notes</h1>
        
        <div className="header-actions">
          <div className="search-bar">
            <span className="search-icon">⌕</span>
            <input
              type="text"
              placeholder={searchMode === 'semantic' ? 'Semantic search…' : 'Search notes...'}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
            {searchMode === 'semantic' && semanticLoading && (
              <span className="search-inline-spinner" aria-label="Searching" />
            )}
          </div>

          <div className="search-controls">
            <div className="toggle-switch" role="group" aria-label="Search mode">
              <button
                className={`toggle-btn ${searchMode === 'text' ? 'active' : ''}`}
                onClick={() => setSearchMode('text')}
                title="Text Search"
              >
                Text
              </button>
              <button
                className={`toggle-btn ${searchMode === 'semantic' ? 'active' : ''}`}
                onClick={() => setSearchMode('semantic')}
                title="Semantic Search"
              >
                Semantic
              </button>
            </div>

            {searchMode === 'semantic' && (
              <div className="threshold-control" title="Minimum similarity threshold">
                <label className="threshold-label">
                  Min similarity: {minSimilarity}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={minSimilarity}
                  onChange={(e) => setMinSimilarity(Number(e.target.value))}
                  className="threshold-slider"
                />
              </div>
            )}
          </div>

          {searchMode === 'semantic' && semanticError && (
            <div className="search-error" title={semanticError}>⚠ {semanticError}</div>
          )}

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
 
          <button
            onClick={handleImportClick}
            className="btn btn-secondary btn-icon"
            title="Import Notes"
          >
            ↑
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            style={{ display: 'none' }}
            onChange={handleFileSelected}
          />

          <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
            <span className={`status-indicator ${connected ? '' : 'disconnected'}`} />
            {connected ? 'Connected' : 'Disconnected'}
          </div>
          {isAuthenticated && user && (
            <div className="user-section">
              <span className="username">👤 {user.username}</span>
              <button
                onClick={logout}
                className="logout-button"
                title="Logout"
              >
                Logout
              </button>
            </div>
          )}
        </div>
      </header>

      {(error || notesError) && (
        <div className="error-banner">
          ⚠ {error || notesError}
        </div>
      )}
 
      {successMessage && (
        <div className="success-banner">
          ✓ {successMessage}
        </div>
      )}
 
      <div className="main-content">
        <div className="sidebar">
          <div className="sidebar-tabs">
            <button
              className={`tab-btn ${activeTab === 'notes' ? 'active' : ''}`}
              onClick={() => handleTabChange('notes')}
              title="View Notes"
            >
              Notes
            </button>
            <button
              className={`tab-btn ${activeTab === 'trash' ? 'active' : ''}`}
              onClick={() => handleTabChange('trash')}
              title="View Trash"
            >
              Trash{trashedNotes.length ? ` (${trashedNotes.length})` : ''}
            </button>
          </div>

          {activeTab === 'notes' ? (
            isCreating || editingNote ? (
              <NoteEditor
                ref={editorRef}
                note={editingNote}
                onSave={handleSaveNote}
                onCancel={handleCancel}
                onDirtyChange={setIsEditorDirty}
                onFindSimilar={handleFindSimilarFromEditor}
              />
            ) : (
              <>
                <NotesList
                  notes={notes}
                  onSelect={(i) => requestNavigate({ type: 'selectNote', index: i })}
                  onEdit={handleEditNote}
                  onDelete={handleDeleteNote}
                  selectedNote={selectedNote}
                  searchTerm={searchTerm}
                  onFindSimilar={handleFindSimilarFromList}
                  searchMode={searchMode}
                  semanticResults={semanticResults}
                  minSimilarity={minSimilarity}
                  semanticLoading={semanticLoading}
                  semanticError={semanticError}
                />
                
                {(notes.length > 0 || trashedNotes.length > 0) && (
                  <div className="stats-bar">
                    <div>{stats.totalNotes} notes • {stats.totalWords} words</div>
                    <div>{stats.totalTags} unique tags • {stats.trashCount} in trash</div>
                  </div>
                )}
              </>
            )
          ) : (
            <TrashView
              trashedNotes={trashedNotes}
              onRestore={restoreFromTrash}
              onDeleteForever={permanentDelete}
              onEmptyTrash={emptyTrash}
            />
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
              controlsParams={graphParams}
              onControlsChange={handleControlsChange}
              onControlsReset={handleControlsReset}
              stats={{
                nodes: graphData?.nodes?.length || 0,
                edges: graphData?.edges?.length || 0
              }}
              loading={graphLoading}
              panelPosition="bottom-left"
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
 
      <SimilarNotesModal
        isOpen={similarOpen}
        baseDoc={similarBaseDoc}
        baseTitle={similarBaseTitle}
        notes={notes}
        excludeIndex={similarExcludeIndex}
        topK={8}
        onClose={() => setSimilarOpen(false)}
        onSelect={(idx) => {
          setSimilarOpen(false);
          requestNavigate({ type: 'selectNote', index: idx });
        }}
        onLink={similarExcludeIndex != null ? ((idx) => {
          const src = notes[similarExcludeIndex]?.id;
          const dst = notes[idx]?.id;
          addLink(src, dst);
        }) : undefined}
      />

      <ImportConfirmModal
        isOpen={showImportModal}
        count={pendingImportedNotes.length}
        onReplace={confirmReplace}
        onMerge={confirmMerge}
        onCancel={() => {
          setShowImportModal(false);
          setPendingImportedNotes([]);
        }}
      />

      <ToastNotification
        isOpen={toastOpen}
        message={toastMessage}
        actionLabel="Undo"
        onAction={handleUndo}
        onClose={() => setToastOpen(false)}
        duration={5000}
      />

      <UnsavedChangesDialog
        isOpen={unsavedOpen}
        onSaveAndContinue={saveAndContinue}
        onDiscard={discardChanges}
        onCancel={cancelDialog}
      />
    </div>
    </AuthGuard>
  );
}