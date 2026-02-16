import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from './contexts/AuthContext';
import AuthGuard from './components/AuthGuard';
import ErrorBoundary from './components/ErrorBoundary';
import NoteEditor from './components/NoteEditor';
import NotesList from './components/NotesList';
import GraphVisualization from './components/GraphVisualization';
import { useNotes } from './hooks/useNotes';
import { useConnectionStatus } from './hooks/useConnectionStatus';
import { useDimensions } from './hooks/useDimensions';
import { useGraph } from './hooks/useGraph';
import { useSemanticSearch } from './hooks/useSemanticSearch';
import { useUnsavedGuard } from './hooks/useUnsavedGuard';
import './styles/index.css';
import ImportConfirmModal from './components/ImportConfirmModal';
import ToastNotification from './components/ToastNotification';
import TrashView from './components/TrashView';
import UnsavedChangesDialog from './components/UnsavedChangesDialog';
import SimilarNotesModal from './components/SimilarNotesModal';
import ImportLocalNotesModal from './components/ImportLocalNotesModal';
import MarkdownPreview from './components/MarkdownPreview';

export default function App() {
  const { user, logout, isAuthenticated } = useAuth();
  const {
    notes, trashedNotes, loading: notesLoading, error: notesError,
    addNote, updateNote, moveToTrash, restoreFromTrash,
    permanentDelete, emptyTrash, getStats, exportNotes, importNotes,
  } = useNotes();
  const { connected, error, setError } = useConnectionStatus();
  const {
    searchMode, setSearchMode, searchTerm, setSearchTerm,
    minSimilarity, setMinSimilarity,
    semanticResults, semanticLoading, semanticError,
  } = useSemanticSearch(notes, connected);
  const {
    isEditorDirty, setIsEditorDirty,
    unsavedOpen, setUnsavedOpen,
    pendingAction, setPendingAction,
    guardedNavigate, cancelDialog,
  } = useUnsavedGuard();

  const graphRef = useRef(null);
  const fileInputRef = useRef(null);
  const editorRef = useRef(null);

  const { graphData, graphLoading, graphParams, handleControlsChange, handleControlsReset } =
    useGraph(notes, connected, setError);
  const dimensions = useDimensions(graphRef, { graphData, notesLength: notes.length });

  // --- UI state ---
  const [selectedNote, setSelectedNote] = useState(null);
  const [editingNote, setEditingNote] = useState(null);
  const [isCreating, setIsCreating] = useState(false);
  const [activeTab, setActiveTab] = useState('notes');
  const [showImportModal, setShowImportModal] = useState(false);
  const [pendingImportedNotes, setPendingImportedNotes] = useState([]);
  const [successMessage, setSuccessMessage] = useState('');
  const [toastOpen, setToastOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState('');
  const [lastTrashedId, setLastTrashedId] = useState(null);
  const [similarOpen, setSimilarOpen] = useState(false);
  const [similarBaseDoc, setSimilarBaseDoc] = useState('');
  const [similarBaseTitle, setSimilarBaseTitle] = useState('');
  const [similarExcludeIndex, setSimilarExcludeIndex] = useState(null);
  const [showImportLocalModal, setShowImportLocalModal] = useState(false);

  // Auto-clear success messages
  useEffect(() => {
    if (!successMessage) return;
    const t = setTimeout(() => setSuccessMessage(''), 4000);
    return () => clearTimeout(t);
  }, [successMessage]);

  // Check for local-storage notes on first authenticated load
  useEffect(() => {
    if (isAuthenticated && user) {
      const hasCheckedImport = localStorage.getItem('import-checked');
      if (!hasCheckedImport) {
        const notesData = localStorage.getItem('semantic-notes-data');
        const trashData = localStorage.getItem('semantic-notes-trash');
        if (notesData || trashData) setShowImportLocalModal(true);
        localStorage.setItem('import-checked', 'true');
      }
    }
  }, [isAuthenticated, user]);

  // --- Navigation / unsaved-guard helpers ---
  const editingActive = isCreating || !!editingNote;

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
    if (guardedNavigate(action, editingActive)) {
      executeAction(action);
    }
  }, [editingActive, guardedNavigate, executeAction]);

  // --- Event handlers ---
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
    requestNavigate({ type: 'edit', index });
  }, [requestNavigate]);

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
    requestNavigate({ type: 'nodeSelect', index: nodeId });
  }, [requestNavigate]);

  const handleNewNote = useCallback(() => {
    requestNavigate({ type: 'new' });
  }, [requestNavigate]);

  const handleCancel = useCallback(() => {
    requestNavigate({ type: 'cancel' });
  }, [requestNavigate]);

  const handleTabChange = useCallback((tab) => {
    requestNavigate({ type: 'tab', tab });
  }, [requestNavigate]);

  // --- Import / export ---
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
          const incoming = Array.isArray(parsed)
            ? parsed
            : (parsed && Array.isArray(parsed.notes) ? parsed.notes : null);
          if (!incoming) throw new Error('Invalid file format. Expected { "notes": [...] }');
          if (!Array.isArray(incoming)) throw new Error('Invalid notes format in file');
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
  }, [setError]);

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
  }, [importNotes, pendingImportedNotes, setError]);

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
  }, [importNotes, pendingImportedNotes, setError]);

  // --- Similar notes ---
  const LINKS_KEY = 'semantic-links-v1';

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
    if (!data) { setError('No note data to analyze'); return; }
    const doc = buildDocText(data);
    const title = String(data.title || 'This note');
    const exclude = editingNote?.originalIndex ?? null;
    openSimilar(doc, title, exclude);
  }, [editorRef, editingNote, buildDocText, openSimilar, setError]);

  const handleFindSimilarFromList = useCallback((index) => {
    const n = notes[index];
    if (!n) return;
    openSimilar(buildDocText(n), n.title || 'This note', index);
  }, [notes, buildDocText, openSimilar]);

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
  }, [setError]);

  // --- Unsaved-changes dialog handlers ---
  const saveAndContinue = useCallback(() => {
    if (!editorRef.current) return;
    const data = editorRef.current.getCurrentData();
    const titleOk = String(data.title || '').trim().length > 0;
    const contentOk = String(data.content || '').trim().length > 0;
    if (!titleOk || !contentOk) { alert('Title and Content are required'); return; }
    handleSaveNote(data);
    setUnsavedOpen(false);
    if (pendingAction) { executeAction(pendingAction); setPendingAction(null); }
  }, [handleSaveNote, pendingAction, executeAction, setUnsavedOpen, setPendingAction]);

  const discardChanges = useCallback(() => {
    setUnsavedOpen(false);
    setIsCreating(false);
    setEditingNote(null);
    if (pendingAction) { executeAction(pendingAction); setPendingAction(null); }
  }, [pendingAction, executeAction, setUnsavedOpen, setPendingAction]);

  const handleImportComplete = () => { setShowImportLocalModal(false); window.location.reload(); };
  const handleImportSkip = () => { setShowImportLocalModal(false); };
  const stats = getStats();

  // --- Render ---
  return (
    <AuthGuard>
      <ErrorBoundary>
      {showImportLocalModal && (
        <ImportLocalNotesModal onClose={handleImportSkip} onImportComplete={handleImportComplete} />
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
              <span className="username"> {user.username}</span>
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
            <ErrorBoundary>
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
            </ErrorBoundary>
          ) : (
            <div className="empty-state">
              <p>Unable to generate graph. Check connection.</p>
            </div>
          )}

            {selectedNote !== null && notes[selectedNote] && (
              <div className="selected-note-preview fade-in">
                <h3>{notes[selectedNote].title}</h3>

                {/* RENDER MARKDOWN IN PREVIEW PANE — CHANGED */}
                <MarkdownPreview
                  content={notes[selectedNote].content || ''}
                  className="markdown-preview-only"
                  style={{ maxHeight: '40vh', overflow: 'auto', marginTop: '0.5rem' }}
                />

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
    </ErrorBoundary>
    </AuthGuard>
  );
}
