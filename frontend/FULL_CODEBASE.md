# CODEBASE EXTRACTION

**Source Directory:** `C:\Users\XH673HG\OneDrive - EY\Desktop\9-APPLICATIONS\semantic-notes\frontend`
**Generated:** 2025-10-23 13:31:46
**Total Files:** 28

---

## Directory Structure

```
frontend/
├── .env
├── .env.example
├── index.html
├── src/
│   ├── App.css
│   ├── App.jsx
│   ├── components/
│   │   ├── AuthGuard.jsx
│   │   ├── ConfirmDialog.jsx
│   │   ├── ExportGraphModal.jsx
│   │   ├── GraphControlsPanel.jsx
│   │   ├── GraphVisualization.jsx
│   │   ├── ImportConfirmModal.jsx
│   │   ├── ImportLocalNotesModal.jsx
│   │   ├── LoginForm.jsx
│   │   ├── MarkdownCheatsheet.jsx
│   │   ├── MarkdownPreview.jsx
│   │   ├── NoteEditor.jsx
│   │   ├── NotesList.jsx
│   │   ├── SimilarNotesModal.jsx
│   │   ├── ToastNotification.jsx
│   │   ├── TrashView.jsx
│   │   └── UnsavedChangesDialog.jsx
│   ├── contexts/
│   │   └── AuthContext.jsx
│   ├── hooks/
│   │   └── useNotes.js
│   ├── main.jsx
│   ├── services/
│   │   ├── api.js
│   │   └── dbApi.js
│   └── utils/
│       └── graphExport.js
└── vite.config.js
```

## Code Files


### 📄 .env

```
# Example environment variables for the frontend
# Replace the base URL below with your backend API endpoint
VITE_API_BASE_URL=http://localhost:8000
```


### 📄 .env.example

```
# Example environment variables for the frontend
# Replace the base URL below with your backend API endpoint
VITE_API_BASE_URL=
```


### 📄 index.html

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Semantic Notes</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```


### 📄 src\App.css

```css
/* =================================================================
   SEMANTIC NOTES - MAIN STYLESHEET
   Professional Dark Theme
   ================================================================= */

/* =================================================================
   CSS CUSTOM PROPERTIES (DESIGN TOKENS)
   ================================================================= */
:root {
  /* Color Palette - Background */
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --bg-hover: #475569;
  
  /* Color Palette - Text */
  --text-primary: #f1f5f9;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  
  /* Color Palette - Accents */
  --accent-primary: #3b82f6;
  --accent-primary-hover: #2563eb;
  --accent-success: #10b981;
  --accent-danger: #ef4444;
  --accent-danger-hover: #dc2626;
  
  /* Borders */
  --border: #334155;
  --border-focus: #3b82f6;
  
  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.4);
  
  /* Border Radius */
  --radius-sm: 0.25rem;
  --radius-md: 0.375rem;
  --radius-lg: 0.5rem;
  
  /* Transitions */
  --transition: all 0.2s ease;
}

/* =================================================================
   BASE & RESET
   ================================================================= */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-primary);
  color: var(--text-secondary);
  height: 100vh;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
}

/* =================================================================
   LAYOUT - APP STRUCTURE
   ================================================================= */
.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.main-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 380px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* =================================================================
   COMPONENTS - HEADER
   ================================================================= */
.app-header {
  background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
  padding: 1rem 2rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: var(--shadow-md);
  z-index: 100;
}

.app-header h1 {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.025em;
}

.header-actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

/* =================================================================
   COMPONENTS - SEARCH
   ================================================================= */
.search-bar {
  position: relative;
}

.search-input {
  padding: 0.5rem 1rem 0.5rem 2.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  width: 280px;
  font-size: 0.875rem;
  transition: var(--transition);
}

.search-input:focus {
  outline: none;
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.search-input::placeholder {
  color: var(--text-dim);
}

.search-icon {
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-dim);
  pointer-events: none;
}

.search-controls {
  display: flex;
  gap: 0.75rem;
  align-items: center;
}

.search-inline-spinner {
  position: absolute;
  right: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  width: 16px;
  height: 16px;
  border: 2px solid var(--border);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.search-error {
  font-size: 0.75rem;
  color: var(--accent-danger);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid var(--accent-danger);
  padding: 0.25rem 0.5rem;
  border-radius: var(--radius-sm);
}

/* =================================================================
   COMPONENTS - CONNECTION STATUS
   ================================================================= */
.connection-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
  padding: 0.375rem 0.75rem;
  border-radius: var(--radius-md);
  background: rgba(0, 0, 0, 0.2);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent-success);
}

.status-indicator.disconnected {
  background: var(--accent-danger);
}

/* =================================================================
   COMPONENTS - BUTTONS
   ================================================================= */
.btn {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition);
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  white-space: nowrap;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--accent-primary);
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background: var(--accent-primary-hover);
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

.btn-secondary:hover:not(:disabled) {
  background: var(--bg-hover);
}

.btn-danger {
  background: var(--accent-danger);
  color: white;
}

.btn-danger:hover:not(:disabled) {
  background: var(--accent-danger-hover);
}

.btn-icon {
  padding: 0.5rem;
  min-width: 2rem;
}

.btn-sm {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
}

.logout-button {
  background: transparent;
  border: 1px solid var(--accent-primary);
  color: var(--text-dim);
  padding: 0.375rem 0.75rem;
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: 0.875rem;
  transition: var(--transition);
  margin: 10px;
}

.logout-button:hover {
  background: var(--accent-primary-hover);
  color: var(--text-secondary);
}

/* =================================================================
   COMPONENTS - TABS
   ================================================================= */
.sidebar-tabs {
  display: flex;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--border);
  background: var(--bg-secondary);
  position: sticky;
  top: 0;
  z-index: 5;
}

.tab-btn {
  padding: 0.375rem 0.75rem;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-muted);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: var(--transition);
  font-size: 0.8125rem;
  font-weight: 500;
}

.tab-btn:hover {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

.tab-btn.active {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
}

/* =================================================================
   COMPONENTS - TOGGLE SWITCH
   ================================================================= */
.toggle-switch {
  display: inline-flex;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: var(--bg-primary);
  box-shadow: var(--shadow-sm);
}

.toggle-btn {
  padding: 0.375rem 0.75rem;
  background: transparent;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.8125rem;
  font-weight: 500;
  transition: var(--transition);
}

.toggle-btn:hover {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

.toggle-btn.active {
  background: var(--accent-primary);
  color: white;
  box-shadow: var(--shadow-sm);
}

/* =================================================================
   COMPONENTS - FORM CONTROLS
   ================================================================= */
.form-group {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-muted);
}

.form-input,
.form-textarea {
  width: 100%;
  padding: 0.75rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: 0.875rem;
  transition: var(--transition);
}

.form-input:focus,
.form-textarea:focus {
  outline: none;
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-input::placeholder,
.form-textarea::placeholder {
  color: var(--text-dim);
}

.form-textarea {
  min-height: 300px;
  resize: vertical;
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  line-height: 1.6;
}

.form-select {
  padding: 0.375rem 0.75rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 0.75rem;
  cursor: pointer;
  transition: var(--transition);
}

.form-select:hover {
  background: var(--bg-tertiary);
}

.form-select:focus {
  outline: none;
  border-color: var(--border-focus);
}

.char-count {
  font-size: 0.75rem;
  color: var(--text-dim);
  text-align: right;
}

.threshold-control {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.threshold-label {
  font-size: 0.75rem;
  color: var(--text-dim);
}

.threshold-slider {
  width: 140px;
  accent-color: var(--accent-primary);
}

/* =================================================================
   COMPONENTS - NOTE EDITOR
   ================================================================= */
.note-editor {
  padding: 1.5rem;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  overflow-y: auto;
}

.editor-header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
}

.editor-header h2 {
  color: var(--text-primary);
  font-size: 1.125rem;
  font-weight: 600;
}

.unsaved-indicator {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  color: var(--text-muted);
  font-size: 0.8125rem;
}

.unsaved-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent-danger);
  box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
}

.editor-form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  flex: 1;
}

.editor-actions {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
}

.keyboard-hint {
  margin-left: auto;
  font-size: 0.75rem;
  color: var(--text-dim);
}

.content-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.content-toolbar-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* =================================================================
   COMPONENTS - NOTES LIST
   ================================================================= */
.notes-list {
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}

.list-header h3 {
  color: var(--text-primary);
  font-size: 1rem;
  font-weight: 600;
}

.list-controls {
  display: flex;
  gap: 0.5rem;
}

.notes-items {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.note-item {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1rem;
  cursor: pointer;
  transition: var(--transition);
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}

.note-item:hover {
  background: var(--bg-tertiary);
  transform: translateX(4px);
  box-shadow: var(--shadow-md);
}

.note-item.selected {
  border-color: var(--accent-primary);
  background: var(--bg-tertiary);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
}

.note-item-content {
  flex: 1;
  min-width: 0;
}

.note-item-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.5rem;
  gap: 0.75rem;
}

.note-item-title {
  color: var(--text-primary);
  font-size: 0.9375rem;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.note-date {
  font-size: 0.75rem;
  color: var(--text-dim);
  white-space: nowrap;
}

.note-preview {
  color: var(--text-muted);
  font-size: 0.8125rem;
  line-height: 1.5;
  margin-bottom: 0.5rem;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.note-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.375rem;
  margin-top: 0.5rem;
}

.tag {
  display: inline-block;
  padding: 0.125rem 0.5rem;
  background: rgba(59, 130, 246, 0.15);
  color: var(--accent-primary);
  border-radius: var(--radius-sm);
  font-size: 0.6875rem;
  font-weight: 500;
  border: 1px solid rgba(59, 130, 246, 0.2);
}

.note-item-actions {
  display: flex;
  gap: 0.25rem;
  opacity: 0;
  transition: opacity 0.2s;
}

.note-item:hover .note-item-actions {
  opacity: 1;
}

.action-btn {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  transition: var(--transition);
  color: var(--text-muted);
  font-weight: 500;
}

.action-btn:hover {
  background: var(--bg-hover);
  color: var(--text-secondary);
}

.action-btn.delete {
  color: var(--accent-danger);
}

.action-btn.delete:hover {
  background: rgba(239, 68, 68, 0.1);
  border-color: var(--accent-danger);
}

.similarity-badge {
  font-size: 0.75rem;
  color: var(--accent-success);
  background: rgba(16, 185, 129, 0.15);
  border: 1px solid rgba(16, 185, 129, 0.35);
  padding: 0.125rem 0.5rem;
  border-radius: 999px;
  margin-right: 0.5rem;
}

.why-match {
  color: var(--text-muted);
  font-size: 0.8125rem;
  line-height: 1.5;
  margin-bottom: 0.5rem;
}

/* =================================================================
   COMPONENTS - TRASH VIEW
   ================================================================= */
.trash-view {
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.trash-items {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.trash-item {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  transition: var(--transition);
}

.trash-item:hover {
  background: var(--bg-tertiary);
  transform: translateX(4px);
  box-shadow: var(--shadow-md);
}

.trash-item-content {
  flex: 1;
  min-width: 0;
}

.trash-item-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.5rem;
  gap: 0.75rem;
}

.trash-item .note-item-actions {
  opacity: 1;
}

/* =================================================================
   COMPONENTS - GRAPH VISUALIZATION
   ================================================================= */
.graph-container {
  flex: 1;
  background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary));
  position: relative;
  overflow: hidden;
}

.graph-visualization {
  width: 100%;
  height: 100%;
  position: relative;
}

.graph-svg {
  background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
  border-radius: var(--radius-lg);
}

.graph-controls {
  position: absolute;
  top: 1rem;
  right: 1rem;
  display: flex;
  gap: 0.5rem;
  z-index: 10;
}

.control-btn {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--text-secondary);
  transition: var(--transition);
  box-shadow: var(--shadow-md);
}

.control-btn:hover {
  background: var(--bg-tertiary);
  transform: scale(1.05);
}

.node-tooltip {
  position: absolute;
  top: 1rem;
  left: 1rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  color: var(--text-secondary);
  box-shadow: var(--shadow-lg);
  pointer-events: none;
  z-index: 100;
  max-width: 300px;
}

.tooltip-label {
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.tooltip-meta {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.selected-note-preview {
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  padding: 1rem;
  border-radius: var(--radius-lg);
  max-width: 350px;
  max-height: 800px;
  overflow-y: auto;
  box-shadow: var(--shadow-lg);
  z-index: 10;
}

.selected-note-preview h3 {
  color: var(--text-primary);
  margin-bottom: 0.5rem;
  font-size: 0.9375rem;
  font-weight: 600;
}

.selected-note-preview p {
  color: var(--text-muted);
  font-size: 0.8125rem;
  line-height: 1.6;
}

/* =================================================================
   COMPONENTS - GRAPH CONTROLS PANEL
   ================================================================= */
.graph-controls-panel {
  position: absolute;
  left: 1rem;
  bottom: 1rem;
  z-index: 20;
  width: 320px;
  max-width: calc(100% - 2rem);
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  color: var(--text-secondary);
  pointer-events: auto;
  overflow: hidden;
}

.graph-controls-panel.top-left {
  top: 1rem;
  bottom: auto;
}

.graph-controls-panel.bottom-left {
  bottom: 1rem;
  top: auto;
}

.gcp-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.5rem 0.5rem 0.5rem 0.375rem;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
}

.gcp-collapse-btn {
  flex: 0 0 auto;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg-primary);
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 0.875rem;
  line-height: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}

.gcp-collapse-btn:hover {
  background: var(--bg-tertiary);
  transform: scale(1.04);
}

.gcp-title {
  flex: 1 1 auto;
  font-weight: 600;
  color: var(--text-primary);
  font-size: 0.9375rem;
}

.gcp-meta {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

.gcp-stats {
  font-size: 0.75rem;
  color: var(--text-dim);
  white-space: nowrap;
}

.gcp-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid var(--border);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.gcp-body {
  display: grid;
  grid-template-rows: 1fr;
  transition: grid-template-rows 0.25s ease, opacity 0.25s ease;
  opacity: 1;
}

.graph-controls-panel.collapsed .gcp-body {
  grid-template-rows: 0fr;
  opacity: 0;
}

.gcp-body > * {
  min-height: 0;
}

.gcp-body-inner {
  padding: 0.75rem;
}

.gcp-row {
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
  margin-bottom: 0.75rem;
}

.gcp-label {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--text-muted);
}

.gcp-select {
  padding: 0.375rem 0.5rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 0.8125rem;
  transition: var(--transition);
  outline: none;
  cursor: pointer;
}

.gcp-select:hover {
  background: var(--bg-tertiary);
}

.gcp-select:focus {
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.gcp-range {
  width: 100%;
  accent-color: var(--accent-primary);
}

.gcp-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 0.25rem;
}

/* =================================================================
   COMPONENTS - SIMILAR NOTES MODAL
   ================================================================= */
.similar-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-height: 50vh;
  overflow: auto;
}

.similar-item {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.75rem;
  display: flex;
  justify-content: space-between;
  align-items: stretch;
  gap: 0.75rem;
  transition: var(--transition);
}

.similar-item:hover {
  background: var(--bg-tertiary);
  box-shadow: var(--shadow-md);
}

.similar-main {
  flex: 1;
  min-width: 0;
  cursor: pointer;
}

.similar-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.5rem;
  margin-bottom: 0.25rem;
}

.similar-title {
  color: var(--text-primary);
  font-weight: 600;
  font-size: 0.9375rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.similar-score {
  font-size: 0.75rem;
  color: var(--text-dim);
  white-space: nowrap;
}

.similar-preview {
  color: var(--text-muted);
  font-size: 0.8125rem;
  line-height: 1.5;
  margin: 0.25rem 0 0.5rem;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.similar-meter {
  height: 6px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
}

.similar-meter-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-success), var(--accent-primary));
}

.similar-actions {
  display: flex;
  gap: 0.375rem;
  align-items: center;
}

/* =================================================================
   COMPONENTS - MODAL
   ================================================================= */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  -webkit-backdrop-filter: blur(2px);
  backdrop-filter: blur(2px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 200;
  animation: fadeIn 0.2s ease-out;
}

.modal {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  width: min(520px, 92vw);
  box-shadow: var(--shadow-lg);
  overflow: hidden;
}

.modal-header {
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--border);
}

.modal-header h3 {
  color: var(--text-primary);
  font-size: 1.0625rem;
  font-weight: 600;
}

.modal-body {
  padding: 1rem 1.25rem;
  color: var(--text-secondary);
  line-height: 1.6;
}

.modal-list {
  margin: 0.5rem 0 0 1rem;
  color: var(--text-muted);
}

.modal-actions {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
  padding: 0.75rem 1.25rem;
  border-top: 1px solid var(--border);
}

/* =================================================================
   COMPONENTS - TOAST NOTIFICATION
   ================================================================= */
.toast-container {
  position: fixed;
  top: 1rem;
  right: 1rem;
  z-index: 300;
  pointer-events: none;
}

.toast-card {
  min-width: 280px;
  max-width: 420px;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent-primary);
  color: var(--text-secondary);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-lg);
  padding: 0.75rem 0.75rem 0.75rem 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  pointer-events: auto;
}

.toast-message {
  color: var(--text-primary);
  font-size: 0.875rem;
  font-weight: 500;
}

.toast-actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.toast-close {
  background: transparent;
  border: none;
  color: var(--text-dim);
  cursor: pointer;
  font-size: 1rem;
  padding: 0.25rem 0.5rem;
  border-radius: var(--radius-sm);
  transition: var(--transition);
}

.toast-close:hover {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

/* =================================================================
   COMPONENTS - BANNERS
   ================================================================= */
.error-banner {
  background: rgba(239, 68, 68, 0.1);
  border-bottom: 1px solid var(--accent-danger);
  color: var(--accent-danger);
  padding: 0.75rem 2rem;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.success-banner {
  background: rgba(16, 185, 129, 0.12);
  border-bottom: 1px solid var(--accent-success);
  color: var(--accent-success);
  padding: 0.75rem 2rem;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* =================================================================
   COMPONENTS - STATS BAR
   ================================================================= */
.stats-bar {
  padding: 1rem 1.5rem;
  border-top: 1px solid var(--border);
  font-size: 0.75rem;
  color: var(--text-dim);
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

/* =================================================================
   COMPONENTS - EMPTY & LOADING STATES
   ================================================================= */
.empty-message {
  text-align: center;
  padding: 3rem 1rem;
  color: var(--text-dim);
}

.empty-message-title {
  font-size: 1rem;
  font-weight: 500;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}

.empty-message-hint {
  font-size: 0.875rem;
  color: var(--text-dim);
}

.empty-state-hint {
  margin-top: 1rem;
  font-size: 0.875rem;
}

.loading,
.empty-state {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  color: var(--text-dim);
  padding: 2rem;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 1rem;
}

/* =================================================================
   COMPONENTS - MARKDOWN PREVIEW
   ================================================================= */
.markdown-preview,
.markdown-pane,
.markdown-preview-only {
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem;
  color: var(--text-secondary);
  font-size: 0.9375rem;
  line-height: 1.7;
}

.markdown-preview-only {
  min-height: 300px;
}

.markdown-preview h1,
.markdown-pane h1 {
  font-size: 1.625rem;
  color: var(--text-primary);
  margin: 0.5rem 0 0.75rem;
}

.markdown-preview h2,
.markdown-pane h2 {
  font-size: 1.375rem;
  color: var(--text-primary);
  margin: 0.75rem 0 0.5rem;
}

.markdown-preview h3,
.markdown-pane h3 {
  font-size: 1.125rem;
  color: var(--text-primary);
  margin: 0.75rem 0 0.5rem;
}

.markdown-preview h4,
.markdown-pane h4 {
  font-size: 1rem;
  color: var(--text-primary);
  margin: 0.75rem 0 0.5rem;
}

.markdown-preview h5,
.markdown-pane h5 {
  font-size: 0.9375rem;
  color: var(--text-primary);
  margin: 0.5rem 0;
}

.markdown-preview h6,
.markdown-pane h6 {
  font-size: 0.875rem;
  color: var(--text-muted);
  letter-spacing: 0.02em;
  margin: 0.5rem 0;
}

.markdown-preview p,
.markdown-pane p {
  margin: 0.5rem 0 0.75rem;
}

.markdown-preview strong,
.markdown-pane strong {
  color: var(--text-primary);
}

.markdown-preview em,
.markdown-pane em {
  color: var(--text-secondary);
}

.markdown-preview ul,
.markdown-pane ul,
.markdown-preview ol,
.markdown-pane ol {
  margin: 0.5rem 0 0.75rem 1.25rem;
}

.markdown-preview li,
.markdown-pane li {
  margin: 0.25rem 0;
}

.markdown-preview blockquote,
.markdown-pane blockquote {
  margin: 0.75rem 0;
  padding: 0.5rem 0.75rem;
  border-left: 3px solid var(--accent-primary);
  background: rgba(59, 130, 246, 0.08);
  color: var(--text-muted);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}

.inline-code,
.markdown-preview :not(pre) > code,
.markdown-pane :not(pre) > code {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.85em;
}

.markdown-preview pre,
.markdown-pane pre {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.75rem 1rem;
  overflow: auto;
  box-shadow: var(--shadow-sm);
}

.markdown-preview pre code,
.markdown-pane pre code {
  background: transparent;
  border: none;
  padding: 0;
  color: var(--text-secondary);
  font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
  font-size: 0.85rem;
  line-height: 1.6;
}

.markdown-preview a,
.markdown-pane a {
  color: var(--accent-primary);
  text-decoration: none;
}

.markdown-preview a:hover,
.markdown-pane a:hover {
  color: var(--accent-primary-hover);
  text-decoration: underline;
}

.markdown-preview table,
.markdown-pane table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.5rem 0 1rem;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.markdown-preview th,
.markdown-pane th,
.markdown-preview td,
.markdown-pane td {
  border: 1px solid var(--border);
  padding: 0.5rem 0.75rem;
  text-align: left;
}

.markdown-preview th,
.markdown-pane th {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.markdown-preview tr:nth-child(even) td,
.markdown-pane tr:nth-child(even) td {
  background: rgba(255, 255, 255, 0.02);
}

.markdown-preview img,
.markdown-pane img {
  max-width: 100%;
  display: block;
  margin: 0.5rem 0;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}

.markdown-preview hr,
.markdown-pane hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1rem 0;
}

.markdown-preview .hljs,
.markdown-pane .hljs {
  color: var(--text-secondary);
  background: transparent;
}

/* =================================================================
   COMPONENTS - SPLIT VIEW
   ================================================================= */
.split-container {
  display: flex;
  gap: 0.75rem;
  height: 40vh;
  min-height: 300px;
}

.split-pane {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.split-pane-editor .form-textarea {
  height: 100%;
  min-height: 0;
  resize: none;
}

.split-pane-preview .markdown-pane {
  height: 100%;
  overflow: auto;
}

/* =================================================================
   COMPONENTS - LOGIN FORM
   ================================================================= */
.login-container {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: var(--bg-primary);
}

.login-form {
  background: var(--bg-secondary);
  padding: 2rem;
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  max-width: 400px;
  width: 100%;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--border);
}

.login-form h2 {
  text-align: center;
  margin-bottom: 1.5rem;
  color: var(--text-primary);
  font-size: 1.5rem;
  font-weight: 600;
}

.login-form label {
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}

.login-form input {
  width: 100%;
  padding: 0.75rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  font-size: 0.875rem;
  margin-bottom: 1rem;
  transition: var(--transition);
}

.login-form input:focus {
  outline: none;
  border-color: var(--border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.login-form button[type="submit"] {
  width: 100%;
  padding: 0.75rem;
  background: var(--accent-primary);
  color: white;
  border: none;
  border-radius: var(--radius-md);
  cursor: pointer;
  font-size: 1rem;
  font-weight: 500;
  transition: var(--transition);
}

.login-form button[type="submit"]:hover:not(:disabled) {
  background: var(--accent-primary-hover);
}

.login-form button[type="submit"]:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.login-error {
  color: var(--accent-danger);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid var(--accent-danger);
  padding: 0.75rem;
  border-radius: var(--radius-md);
  margin-bottom: 1rem;
  text-align: center;
  font-size: 0.875rem;
}

.login-footer {
  text-align: center;
  margin-top: 1rem;
  font-size: 0.875rem;
  color: var(--text-muted);
}

.login-footer button {
  color: var(--accent-primary);
  border: none;
  background: none;
  cursor: pointer;
  font-weight: 500;
  transition: var(--transition);
}

.login-footer button:hover {
  color: var(--accent-primary-hover);
  text-decoration: underline;
}

/* =================================================================
   COMPONENTS - IMPORT LOCAL NOTES MODAL
   ================================================================= */
.modal-content {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  width: min(520px, 92vw);
  box-shadow: var(--shadow-lg);
  padding: 1.5rem;
  max-height: 90vh;
  overflow-y: auto;
}

.import-modal {
  color: var(--text-secondary);
}

.import-modal h2 {
  color: var(--text-primary);
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 1rem;
}

.import-modal p {
  margin-bottom: 0.75rem;
  line-height: 1.6;
}

.import-benefits {
  list-style: none;
  margin: 1rem 0;
  padding: 1rem;
  background: var(--bg-primary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
}

.import-benefits li {
  padding: 0.5rem 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
}

.import-note {
  font-size: 0.75rem;
  color: var(--text-dim);
  font-style: italic;
  margin-top: 1rem;
}

.error-message {
  color: var(--accent-danger);
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid var(--accent-danger);
  padding: 0.75rem;
  border-radius: var(--radius-md);
  margin: 1rem 0;
  font-size: 0.875rem;
}

.success-message {
  text-align: center;
  padding: 2rem 1rem;
}

.success-icon {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: var(--accent-success);
  color: white;
  font-size: 2rem;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 1rem;
}

.success-message p {
  color: var(--text-primary);
  font-weight: 500;
  margin-bottom: 0.5rem;
}

/* =================================================================
   UTILITIES - SCROLLBAR
   ================================================================= */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
  background: var(--bg-tertiary);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--bg-hover);
}

/* =================================================================
   UTILITIES - ANIMATIONS
   ================================================================= */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.fade-in {
  animation: fadeIn 0.3s ease-out;
}

/* =================================================================
   RESPONSIVE - MEDIA QUERIES
   ================================================================= */
@media (max-width: 900px) {
  .split-container {
    flex-direction: column;
    height: auto;
  }

  .split-pane-editor .form-textarea {
    resize: vertical;
  }

  .graph-controls-panel {
    width: 300px;
  }

  .threshold-slider {
    width: 110px;
  }
}

@media (max-width: 768px) {
  .sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--border);
  }

  .main-content {
    flex-direction: column;
  }

  .graph-container {
    min-height: 400px;
  }

  .selected-note-preview {
    max-width: calc(100% - 2rem);
  }

  .header-actions {
    gap: 0.5rem;
  }

  .search-input {
    width: 200px;
  }
}

@media (max-width: 600px) {
  .graph-controls-panel {
    width: calc(100% - 2rem);
  }
}
```


### 📄 src\App.jsx

```
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
  
  // Recalculate dimensions when graph data changes or notes length changes

  useEffect(() => {
  if (graphRef.current && (graphData || notes.length > 0)) {
    // Small delay to ensure DOM has fully rendered and settled
    const timer = setTimeout(() => {
      if (graphRef.current) {
        const newWidth = graphRef.current.clientWidth;
        const newHeight = graphRef.current.clientHeight;
        
        // Only update if dimensions actually changed to avoid unnecessary re-renders
        setDimensions(prev => {
          if (prev.width !== newWidth || prev.height !== newHeight) {
            return { width: newWidth, height: newHeight };
          }
          return prev;
        });
      }
    }, 100);
    
    return () => clearTimeout(timer);
  }
}, [graphData, notes.length]);

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
```


### 📄 src\components\AuthGuard.jsx

```
import { useAuth } from '../contexts/AuthContext';
import LoginForm from './LoginForm';

export default function AuthGuard({ children }) {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="loading">
        <div className="loading-spinner" />
        <div>Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginForm />;
  }

  return children;
}
```


### 📄 src\components\ConfirmDialog.jsx

```
import React from 'react';

export default function ConfirmDialog({
  isOpen = false,
  title = 'Confirm',
  message = '',
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
  danger = false
}) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
      <div className="modal">
        <div className="modal-header">
          <h3 id="confirm-title">{title}</h3>
        </div>
        <div className="modal-body">
          <p>{message}</p>
        </div>
        <div className="modal-actions">
          <button
            className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`}
            onClick={() => onConfirm && onConfirm()}
          >
            {confirmLabel}
          </button>
          <button className="btn btn-secondary" onClick={() => onCancel && onCancel()}>
            {cancelLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 src\components\ExportGraphModal.jsx

```
import React, { useEffect, useMemo, useState } from 'react';
import {
  defaultFilename,
  savePng,
  saveSvg,
  saveJson,
  copyPng,
  copySvg,
  copyJson,
  formatTimestamp,
} from '../utils/graphExport';

export default function ExportGraphModal({
  isOpen = false,
  onClose,
  svgRef,
  graphData,
  params = {},
  transform = { x: 0, y: 0, k: 1 },
  onNotify, // optional toast notifier: (msg: string) => void
}) {
  const [format, setFormat] = useState('png'); // 'png' | 'svg' | 'json'
  const [filename, setFilename] = useState(defaultFilename('png'));
  const [scale, setScale] = useState(2); // PNG resolution multiplier
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isOpen) return;
    setFormat('png');
    setScale(2);
    setError('');
    setBusy(false);
    setFilename(defaultFilename('png'));
  }, [isOpen]);

  const metadata = useMemo(() => ({
    exported_at: new Date().toISOString(),
    timestamp: formatTimestamp(),
    params,
    transform,
    stats: {
      nodes: graphData?.nodes?.length || 0,
      edges: graphData?.edges?.length || 0,
    },
  }), [params, transform, graphData]);

  const ensureExt = (name, ext) => {
    const lower = String(name || '').toLowerCase().trim();
    if (!lower.endsWith(`.${ext}`)) return `${name}.${ext}`;
    return name;
  };

  const handleDownload = async () => {
    try {
      setBusy(true);
      setError('');
      const svgEl = svgRef?.current;
      if (!svgEl) throw new Error('SVG not available');

      if (format === 'png') {
        const fname = ensureExt(filename || defaultFilename('png'), 'png');
        await savePng(svgEl, fname, scale);
        onNotify?.(`Exported PNG: ${fname}`);
      } else if (format === 'svg') {
        const fname = ensureExt(filename || defaultFilename('svg'), 'svg');
        saveSvg(svgEl, fname);
        onNotify?.(`Exported SVG: ${fname}`);
      } else if (format === 'json') {
        const fname = ensureExt(filename || defaultFilename('json'), 'json');
        saveJson(graphData, fname, metadata);
        onNotify?.(`Exported JSON: ${fname}`);
      }
      onClose?.();
    } catch (e) {
      setError(e?.message || 'Export failed');
    } finally {
      setBusy(false);
    }
  };

  const handleCopy = async () => {
    try {
      setBusy(true);
      setError('');
      const svgEl = svgRef?.current;
      if (format === 'png') {
        await copyPng(svgEl, scale);
        onNotify?.('PNG copied to clipboard');
      } else if (format === 'svg') {
        await copySvg(svgEl);
        onNotify?.('SVG copied to clipboard');
      } else if (format === 'json') {
        await copyJson(graphData, metadata);
        onNotify?.('JSON copied to clipboard');
      }
      onClose?.();
    } catch (e) {
      // Graceful fallback for clipboard limitations
      setError(e?.message || 'Copy failed. Your browser may not support clipboard for this format.');
    } finally {
      setBusy(false);
    }
  };

  if (!isOpen) return null;

  const stop = (e) => e.stopPropagation();

  return (
    <div className="modal-overlay" onClick={() => !busy && onClose?.()}>
      <div className="modal" role="dialog" aria-modal="true" onClick={stop}>
        <div className="modal-header">
          <h3>Export Graph</h3>
        </div>

        <div className="modal-body">
          <div className="gcp-row" style={{ marginBottom: '0.75rem' }}>
            <label className="gcp-label" htmlFor="export-format">Format</label>
            <div id="export-format" className="radio-group" role="radiogroup" aria-label="Export format">
              <label className="radio">
                <input
                  type="radio"
                  name="format"
                  value="png"
                  checked={format === 'png'}
                  onChange={() => {
                    setFormat('png');
                    setFilename(defaultFilename('png'));
                  }}
                  disabled={busy}
                />
                PNG (image)
              </label>
              <label className="radio" style={{ marginLeft: '1rem' }}>
                <input
                  type="radio"
                  name="format"
                  value="svg"
                  checked={format === 'svg'}
                  onChange={() => {
                    setFormat('svg');
                    setFilename(defaultFilename('svg'));
                  }}
                  disabled={busy}
                />
                SVG (vector)
              </label>
              <label className="radio" style={{ marginLeft: '1rem' }}>
                <input
                  type="radio"
                  name="format"
                  value="json"
                  checked={format === 'json'}
                  onChange={() => {
                    setFormat('json');
                    setFilename(defaultFilename('json'));
                  }}
                  disabled={busy}
                />
                JSON (data)
              </label>
            </div>
          </div>

          {format === 'png' && (
            <div className="gcp-row" style={{ marginBottom: '0.75rem' }}>
              <label className="gcp-label" htmlFor="export-scale">Resolution</label>
              <select
                id="export-scale"
                className="gcp-select"
                value={scale}
                onChange={(e) => setScale(parseInt(e.target.value, 10) || 2)}
                disabled={busy}
                title="Export scale factor"
              >
                <option value={1}>1x (standard)</option>
                <option value={2}>2x (high)</option>
                <option value={3}>3x (ultra)</option>
              </select>
            </div>
          )}

          <div className="gcp-row" style={{ marginBottom: '0.25rem' }}>
            <label className="gcp-label" htmlFor="export-filename">Filename</label>
            <input
              id="export-filename"
              className="gcp-select"
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              disabled={busy}
              placeholder={defaultFilename(format)}
              title="Choose a filename"
            />
          </div>

          <div className="small" style={{ opacity: 0.8, marginTop: '0.25rem' }}>
            Tip: Exported image reflects current zoom/pan and styling.
          </div>

          {error && (
            <div className="error-banner" style={{ marginTop: '0.5rem' }}>
              ⚠ {error}
            </div>
          )}
        </div>

        <div className="modal-actions">
          <button className="btn btn-primary" onClick={handleDownload} disabled={busy}>
            {busy ? 'Exporting…' : 'Download'}
          </button>
          <button className="btn btn-secondary" onClick={handleCopy} disabled={busy} title="Copy to clipboard">
            Copy
          </button>
          <button className="btn" onClick={() => !busy && onClose?.()} disabled={busy}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 src\components\GraphControlsPanel.jsx

```
import React, { useEffect, useMemo, useState } from 'react';

const COLLAPSE_LS_KEY = 'graph-controls-collapsed-v1';

function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

export default function GraphControlsPanel({
  params,
  onChange,
  onReset,
  stats = { nodes: 0, edges: 0 },
  loading = false,
  position = 'bottom-left', // 'top-left' | 'bottom-left'
}) {
  const [collapsed, setCollapsed] = useState(() => {
    try {
      const raw = localStorage.getItem(COLLAPSE_LS_KEY);
      return raw ? JSON.parse(raw) === true : false;
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSE_LS_KEY, JSON.stringify(collapsed));
    } catch {
      /* ignore */
    }
  }, [collapsed]);

  const isKnn = params?.connection === 'knn';
  const showClusters = !!params?.clustering;

  const panelClass = useMemo(() => {
    const base = 'graph-controls-panel';
    const pos = position === 'top-left' ? 'top-left' : 'bottom-left';
    return `${base} ${pos} ${collapsed ? 'collapsed' : ''}`;
  }, [position, collapsed]);

  return (
    <div className={panelClass} aria-live="polite">
      <div className="gcp-header">
        <button
          className="gcp-collapse-btn"
          onClick={() => setCollapsed((c) => !c)}
          aria-expanded={!collapsed}
          aria-controls="gcp-body"
          title={collapsed ? 'Expand controls' : 'Collapse controls'}
        >
          {collapsed ? '▸' : '▾'}
        </button>

        <div className="gcp-title" title="Graph Controls">Graph Controls</div>

        <div className="gcp-meta">
          <span className="gcp-stats" title="Current graph statistics">
            {stats.nodes} nodes, {stats.edges} edges
          </span>
          {loading && <span className="gcp-spinner" aria-label="Regenerating graph" />}
        </div>
      </div>

      <div id="gcp-body" className="gcp-body">
        <div className="gcp-body-inner">
          <div className="gcp-row">
            <label
              className="gcp-label"
              htmlFor="gcp-connection"
              title="How nodes are connected: k nearest neighbors or similarity threshold"
            >
              Connection
            </label>
            <select
              id="gcp-connection"
              className="gcp-select"
              value={params.connection}
              onChange={(e) => onChange({ connection: e.target.value === 'threshold' ? 'threshold' : 'knn' })}
              title="kNN: connect each node to its k most similar neighbors; Threshold: connect nodes whose similarity is above a set value"
            >
              <option value="knn">kNN</option>
              <option value="threshold">Threshold</option>
            </select>
          </div>

          {isKnn ? (
            <div className="gcp-row">
              <label className="gcp-label" htmlFor="gcp-k" title="Number of nearest neighbors per node (1-10)">
                k neighbors: {clamp(params.k_neighbors ?? 5, 1, 10)}
              </label>
              <input
                id="gcp-k"
                className="gcp-range"
                type="range"
                min="1"
                max="10"
                step="1"
                value={clamp(params.k_neighbors ?? 5, 1, 10)}
                onChange={(e) => onChange({ k_neighbors: clamp(parseInt(e.target.value, 10), 1, 10) })}
                title="Connect each node to its k most similar neighbors"
              />
            </div>
          ) : (
            <div className="gcp-row">
              <label
                className="gcp-label"
                htmlFor="gcp-threshold"
                title="Minimum cosine similarity (0.00-1.00) required to draw an edge"
              >
                Threshold: {(params.similarity_threshold ?? 0.7).toFixed(2)}
              </label>
              <input
                id="gcp-threshold"
                className="gcp-range"
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={clamp(Number(params.similarity_threshold ?? 0.7), 0, 1)}
                onChange={(e) => onChange({ similarity_threshold: clamp(parseFloat(e.target.value), 0, 1) })}
                title="Edges connect nodes with similarity equal to or greater than this value"
              />
            </div>
          )}

          <div className="gcp-row">
            <label
              className="gcp-label"
              htmlFor="gcp-dr"
              title="Dimensionality reduction for layout: project embeddings to 2D"
            >
              Dimensionality reduction
            </label>
            <select
              id="gcp-dr"
              className="gcp-select"
              value={params.dim_reduction ?? 'none'}
              onChange={(e) => {
                const v = e.target.value;
                onChange({ dim_reduction: v === 'none' ? null : v });
              }}
              title="Choose PCA (fast, linear), UMAP/t-SNE (nonlinear, capture local structure), or None"
            >
              <option value="tsne">t-SNE</option>
              <option value="pca">PCA</option>
              <option value="umap">UMAP</option>
              <option value="none">None</option>
            </select>
          </div>

          <div className="gcp-row">
            <label className="gcp-label" htmlFor="gcp-cluster" title="Cluster nodes into groups">
              Clustering
            </label>
            <select
              id="gcp-cluster"
              className="gcp-select"
              value={params.clustering ?? 'none'}
              onChange={(e) => {
                const v = e.target.value;
                onChange({ clustering: v === 'none' ? null : v });
              }}
              title="None: no clustering; k-means: partition into k clusters; Agglomerative: hierarchical clustering"
            >
              <option value="none">None</option>
              <option value="kmeans">k-means</option>
              <option value="agglomerative">Agglomerative</option>
            </select>
          </div>

          {showClusters && (
            <div className="gcp-row">
              <label className="gcp-label" htmlFor="gcp-nc" title="Number of clusters (2-20)">
                Clusters: {clamp(params.n_clusters ?? 5, 2, 20)}
              </label>
              <input
                id="gcp-nc"
                className="gcp-range"
                type="range"
                min="2"
                max="20"
                step="1"
                value={clamp(params.n_clusters ?? 5, 2, 20)}
                onChange={(e) => onChange({ n_clusters: clamp(parseInt(e.target.value, 10), 2, 20) })}
                title="Number of clusters for the selected clustering method"
              />
            </div>
          )}

          <div className="gcp-actions">
            <button
              className="btn btn-secondary btn-sm"
              onClick={onReset}
              title="Reset all controls to their default values"
            >
              Reset to defaults
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
```


### 📄 src\components\GraphVisualization.jsx

```
// GraphVisualization.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import GraphControlsPanel from './GraphControlsPanel';
import ExportGraphModal from './ExportGraphModal';
import ToastNotification from './ToastNotification';

const COLORS = ['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000'];

// Tunables
const NODE_R = 12;
const COLLIDE_R = 18;
const LABEL_ZOOM_THRESHOLD = 0.8;  // show link labels only when zoomed in
const MAX_CHARGE_DISTANCE = 600;   // cap n-body computations
const TARGET_FPS_MS = 16;          // ~60fps

export default function GraphVisualization({
  graphData,
  onNodeClick,
  selectedNote,
  width = 800,
  height = 600,
  controlsParams = {},
  onControlsChange = () => {},
  onControlsReset = () => {},
  stats = { nodes: 0, edges: 0 },
  loading = false,
  panelPosition = 'bottom-left'
}) {
  const svgRef = useRef(null);
  const gRootRef = useRef(null);
  const gLinksRef = useRef(null);
  const gLinkLabelsRef = useRef(null);
  const gNodesRef = useRef(null);

  const simulationRef = useRef(null);
  const zoomBehaviorRef = useRef(null);

  const [hoveredNode, setHoveredNode] = useState(null);
  const [isRunning, setIsRunning] = useState(true);

  const [exportOpen, setExportOpen] = useState(false);
  const [exportTransform, setExportTransform] = useState({ x: 0, y: 0, k: 1 });
  const [toastOpen, setToastOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState('');

  // --------- Data prep (memoized) ----------
  const { nodesMemo, linksMemo } = useMemo(() => {
    const nodes = (graphData?.nodes || []).map(n => ({
      ...n,
      id: String(n.id),
      // Normalize any precomputed [-1,1] space to actual px, else random seed
      x: typeof n.x === 'number' ? (n.x * width / 2 + width / 2) : Math.random() * width,
      y: typeof n.y === 'number' ? (n.y * height / 2 + height / 2) : Math.random() * height
    }));

    const edges = (graphData?.edges || []).map(e => ({
      ...e,
      source: String(e.source),
      target: String(e.target),
      weight: typeof e.weight === 'number' ? e.weight : 0.5
    }));

    return { nodesMemo: nodes, linksMemo: edges };
  }, [graphData, width, height]);

  // --------- Build / (re)build scene ----------
  useEffect(() => {
    if (!svgRef.current || !nodesMemo.length) return;

    const svg = d3.select(svgRef.current);

    // Clear once per graphData change
    svg.selectAll('*').remove();

    // Root group for zoom/pan
    const gRoot = svg.append('g').attr('class', 'graph-root');
    gRootRef.current = gRoot;

    // Layers (links under nodes)
    const gLinks = gRoot.append('g').attr('class', 'layer links');
    const gLinkLabels = gRoot.append('g').attr('class', 'layer link-labels').attr('opacity', 0);
    const gNodes = gRoot.append('g').attr('class', 'layer nodes');

    gLinksRef.current = gLinks;
    gLinkLabelsRef.current = gLinkLabels;
    gNodesRef.current = gNodes;

    // Zoom (persist between updates)
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        gRoot.attr('transform', event.transform);
        // toggle link-label visibility at zoom threshold to reduce DOM cost
        gLinkLabels.attr('opacity', event.transform.k >= LABEL_ZOOM_THRESHOLD ? 1 : 0);
      });

    zoomBehaviorRef.current = zoom;
    svg.call(zoom);

    // Scales (constant functions are cheaper than re-styling on every tick)
    const widthScale = d3.scaleLinear().domain([0, 1]).range([1, 4]);
    const alphaScale = d3.scaleLinear().domain([0, 1]).range([0.25, 0.9]);

    // Selections (stable across ticks)
    const linkSel = gLinks.selectAll('line')
      .data(linksMemo)
      .join('line')
      .attr('class', 'link')
      .attr('stroke', '#4a5568')
      .attr('stroke-linecap', 'round')
      .attr('stroke-opacity', d => alphaScale(d.weight))
      .attr('stroke-width', d => widthScale(d.weight))
      .style('pointer-events', 'none');

    // Only attach labels for heavier links; hide until zoomed in
    const labelSel = gLinkLabels.selectAll('text')
      .data(linksMemo.filter(d => d.weight > 0.4))
      .join('text')
      .attr('class', 'link-label')
      .text(d => d.weight.toFixed(2))
      .attr('font-size', 10)
      .attr('fill', '#9aa0a6')
      .attr('text-anchor', 'middle')
      .style('pointer-events', 'none');

    const drag = d3.drag()
      .on('start', (event, d) => {
        if (!event.active && simulationRef.current) simulationRef.current.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active && simulationRef.current) simulationRef.current.alphaTarget(0);
        // Hold with Shift, otherwise release
        if (!event.sourceEvent.shiftKey) {
          d.fx = null;
          d.fy = null;
        }
      });

    const nodeGSel = gNodes.selectAll('g.node')
      .data(nodesMemo, d => d.id)
      .join(enter => {
        const g = enter.append('g')
          .attr('class', 'node')
          .style('cursor', 'pointer')
          .call(drag);

        g.append('circle')
          .attr('r', NODE_R)
          .attr('fill', d => (d.cluster !== undefined ? COLORS[d.cluster % COLORS.length] : COLORS[0]))
          .attr('stroke', d => d.id === String(selectedNote) ? '#ffffff' : '#1f2937')
          .attr('stroke-width', d => d.id === String(selectedNote) ? 3 : 1.5);

        // Very small labels to keep DOM light; text is expensive
        g.append('text')
          .attr('y', -NODE_R - 4)
          .attr('text-anchor', 'middle')
          .attr('font-size', 8)
          .attr('font-weight', 500)
          .attr('fill', '#e5e7eb')
          .text(d => d.label || `Note ${d.id}`)
          .style('pointer-events', 'none');

        // Events (no transitions — cheaper)
        g.on('click', (event, d) => {
           event.stopPropagation();
           onNodeClick(parseInt(d.id, 10));
        })
        .on('mouseenter', (event, d) => {
           setHoveredNode(d);
           d3.select(event.currentTarget).select('circle')
             .attr('r', NODE_R + 3)
             .attr('stroke-width', 2.5);
        })
        .on('mouseleave', (event, d) => {
           setHoveredNode(null);
           d3.select(event.currentTarget).select('circle')
             .attr('r', NODE_R)
             .attr('stroke-width', d.id === String(selectedNote) ? 3 : 1.5);
        });

        return g;
      });

    // --------- Force simulation (tuned) ----------
    // link.distance/strength derived from weight (heavier => shorter/stronger)
    const distance = d3.scaleLinear().domain([0, 1]).range([100, 60]);
    const strength = d3.scaleLinear().domain([0, 1]).range([0.1, 1.0]);

    const sim = d3.forceSimulation(nodesMemo)
      .force('link', d3.forceLink(linksMemo)
        .id(d => d.id)
        .distance(d => distance(d.weight))
        .strength(d => strength(d.weight))
        .iterations(1)) // fewer iterations for speed
      .force('charge', d3.forceManyBody()
        .strength(-60)
        .theta(0.9)
        .distanceMax(Math.max(700, Math.min(MAX_CHARGE_DISTANCE, Math.max(width, height)))))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(COLLIDE_R).strength(0.1))
      .force('x', d3.forceX(width / 2).strength(0.01))
      .force('y', d3.forceY(height / 2).strength(0.01))
      // decay so it settles sooner
      .alpha(2.2)
      .alphaDecay(1 - Math.pow(0.001, 1 / 300));

    simulationRef.current = sim;

    // Throttle DOM writes to ~60 fps
    let lastRender = 0;
    let rafScheduled = false;

    const render = (now) => {
      rafScheduled = false;
      if (now - lastRender < TARGET_FPS_MS) return;
      lastRender = now;

      // Update link positions
      linkSel
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      // Midpoints for labels
      labelSel
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);

      // Node transforms
      nodeGSel.attr('transform', d => `translate(${d.x},${d.y})`);
    };

    sim.on('tick', () => {
      if (!rafScheduled) {
        rafScheduled = true;
        requestAnimationFrame(render);
      }
    });

    // Pause simulation when tab is hidden (battery/perf)
    const onVis = () => {
      if (document.hidden) {
        sim.stop();
      } else if (isRunning) {
        sim.alpha(0.3).restart();
      }
    };
    document.addEventListener('visibilitychange', onVis);

    // Cleanup
    return () => {
      document.removeEventListener('visibilitychange', onVis);
      sim.stop();
    };
  }, [nodesMemo, linksMemo, width, height, onNodeClick, isRunning]);

  // --------- Highlight selected node without full redraw ----------
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('g.node circle')
      .attr('stroke', function () {
        const d = d3.select(this.parentNode).datum();
        return d && d.id === String(selectedNote) ? '#ffffff' : '#1f2937';
      })
      .attr('stroke-width', function () {
        const d = d3.select(this.parentNode).datum();
        const isSel = d && d.id === String(selectedNote);
        return isSel ? 3 : 1.5;
      })
      .attr('r', function () {
        const d = d3.select(this.parentNode).datum();
        const isHover = false; // no persisted hover state on circles
        return isHover ? NODE_R + 3 : NODE_R;
      });
  }, [selectedNote]);

  // --------- Controls ----------
  const handleResetView = () => {
    const svg = d3.select(svgRef.current);
    if (!zoomBehaviorRef.current) return;
    svg.transition().duration(300).call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
  };

  const handleToggleSimulation = () => {
    const sim = simulationRef.current;
    if (!sim) return;
    if (isRunning) {
      sim.stop();
    } else {
      sim.alpha(0.3).restart();
    }
    setIsRunning(!isRunning);
  };

  const handleOpenExport = () => {
    try {
      const t = d3.zoomTransform(svgRef.current);
      setExportTransform({ x: t.x, y: t.y, k: t.k });
    } catch {
      setExportTransform({ x: 0, y: 0, k: 1 });
    }
    setExportOpen(true);
  };

  const handleNotify = (msg) => {
    setToastMessage(msg);
    setToastOpen(true);
  };

  return (
    <div className="graph-visualization">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="graph-svg"
        // minimal CSS via classes; avoid expensive filters
        style={{ background: 'transparent', display: 'block' }}
      />

      <GraphControlsPanel
        params={controlsParams}
        onChange={onControlsChange}
        onReset={onControlsReset}
        stats={stats}
        loading={loading}
        position={panelPosition}
      />

      <div className="graph-controls">
        <button onClick={handleResetView} className="control-btn" title="Reset View">⟲</button>
        <button onClick={handleToggleSimulation} className="control-btn" title="Toggle Physics">
          {isRunning ? '❚❚' : '▶'}
        </button>
        <button onClick={handleOpenExport} className="control-btn" title="Export Graph">⤓</button>
      </div>

      {hoveredNode && (
        <div className="node-tooltip">
          <div className="tooltip-label">{hoveredNode.label}</div>
          <div className="tooltip-meta">
            ID: {hoveredNode.id}
            {hoveredNode.cluster !== undefined && ` • Cluster: ${hoveredNode.cluster}`}
          </div>
        </div>
      )}

      <ExportGraphModal
        isOpen={exportOpen}
        onClose={() => setExportOpen(false)}
        svgRef={svgRef}
        graphData={graphData}
        params={controlsParams}
        transform={exportTransform}
        onNotify={handleNotify}
      />

      <ToastNotification
        isOpen={toastOpen}
        message={toastMessage}
        onClose={() => setToastOpen(false)}
        duration={4000}
      />
    </div>
  );
}

```


### 📄 src\components\ImportConfirmModal.jsx

```
import React, { useEffect } from 'react';

export default function ImportConfirmModal({ isOpen, count = 0, onReplace, onMerge, onCancel }) {
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e) => {
      if (e.key === 'Escape') onCancel?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onCancel]);

  if (!isOpen) return null;

  const stop = (e) => e.stopPropagation();

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal" role="dialog" aria-modal="true" onClick={stop}>
        <div className="modal-header">
          <h3>Import Notes</h3>
        </div>
        <div className="modal-body">
          <p>Detected {count} notes in the selected file.</p>
          <p>How would you like to import them?</p>
          <ul className="modal-list">
            <li><strong>Replace all notes</strong> — clears current notes and loads the imported ones.</li>
            <li><strong>Merge with existing</strong> — keeps current notes and adds imported notes. Conflicting IDs will be regenerated.</li>
          </ul>
        </div>
        <div className="modal-actions">
          <button className="btn btn-danger" onClick={onReplace} title="Replace current notes">
            Replace All
          </button>
          <button className="btn btn-primary" onClick={onMerge} title="Merge imported notes">
            Merge
          </button>
          <button className="btn btn-secondary" onClick={onCancel} title="Cancel">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 src\components\ImportLocalNotesModal.jsx

```
import { useState, useEffect } from 'react';
import apiService from '../services/api';

export default function ImportLocalNotesModal({ onClose, onImportComplete }) {
  const [localNotes, setLocalNotes] = useState([]);
  const [localTrash, setLocalTrash] = useState([]);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    // Check for localStorage notes
    const notesData = localStorage.getItem('semantic-notes-data');
    const trashData = localStorage.getItem('semantic-notes-trash');
    
    if (notesData) {
      try {
        const parsed = JSON.parse(notesData);
        setLocalNotes(parsed.notes || []);
      } catch (e) {
        console.error('Failed to parse notes:', e);
      }
    }
    
    if (trashData) {
      try {
        const parsed = JSON.parse(trashData);
        setLocalTrash(parsed.trash || []);
      } catch (e) {
        console.error('Failed to parse trash:', e);
      }
    }
  }, []);

  const handleImport = async () => {
    setImporting(true);
    setError('');
    
    try {
      const response = await apiService.request('/api/notes/import', {
        method: 'POST',
        body: JSON.stringify({
          notes: localNotes,
          trash: localTrash
        })
      });
      
      setSuccess(true);
      setTimeout(() => {
        onImportComplete(response.imported);
      }, 1500);
    } catch (err) {
      setError(err.message || 'Import failed. Please try again.');
      setImporting(false);
    }
  };

  const handleSkip = () => {
    onClose();
  };

  const totalToImport = localNotes.length + localTrash.length;

  if (totalToImport === 0) {
    return null; // Don't show modal if no notes to import
  }

  return (
    <div className="modal-overlay">
      <div className="modal-content import-modal">
        <h2>Import Your Local Notes</h2>
        
        {!success ? (
          <>
            <p>
              We found <strong>{localNotes.length} notes</strong>
              {localTrash.length > 0 && ` and ${localTrash.length} items in trash`} 
              {' '}stored locally on this device.
            </p>
            
            <p>
              Would you like to import them to your account? This will:
            </p>
            
            <ul className="import-benefits">
              <li>✓ Sync your notes across devices</li>
              <li>✓ Keep your data safe in the cloud</li>
              <li>✓ Preserve all timestamps and tags</li>
              <li>✓ Maintain your local cache for speed</li>
            </ul>
            
            {error && (
              <div className="error-message">{error}</div>
            )}
            
            <div className="modal-actions">
              <button
                onClick={handleImport}
                disabled={importing}
                className="btn-primary"
              >
                {importing ? 'Importing...' : `Import ${totalToImport} Notes`}
              </button>
              <button
                onClick={handleSkip}
                disabled={importing}
                className="btn-secondary"
              >
                Skip for Now
              </button>
            </div>
            
            <p className="import-note">
              Note: Your local notes will remain available even if you skip.
            </p>
          </>
        ) : (
          <div className="success-message">
            <div className="success-icon">✓</div>
            <p>Successfully imported {totalToImport} notes!</p>
            <p>Redirecting...</p>
          </div>
        )}
      </div>
    </div>
  );
}
```


### 📄 src\components\LoginForm.jsx

```
import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

export default function LoginForm() {
  const [mode, setMode] = useState('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  
  const { login, register } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (username.length < 3) {
      setError('Username must be at least 3 characters');
      return;
    }
    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }
    
    setLoading(true);
    try {
      if (mode === 'login') {
        await login(username, password);
      } else {
        await register(username, password, email);
      }
    } catch (err) {
      let message = 'Authentication failed';
      if (err instanceof Error) {
        message = err.message;
      } else if (typeof err === 'string') {
        message = err;
      } else if (err && err.detail) {
        message = err.detail;
      }
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <form onSubmit={handleSubmit} className="login-form">
        <h2>{mode === 'login' ? 'Login' : 'Register'}</h2>

        <label>Username</label>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />

        {mode === 'register' && (
          <>
            <label>Email (optional)</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </>
        )}

        <label>Password</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        {error && (
          <div className="login-error">
            {error}
          </div>
        )}

        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : mode === 'login' ? 'Login' : 'Register'}
        </button>

        <p className="login-footer">
          {mode === 'login' ? (
            <>
              Don't have an account?{' '}
              <button type="button" onClick={() => setMode('register')}>
                Register
              </button>
            </>
          ) : (
            <>
              Already have an account?{' '}
              <button type="button" onClick={() => setMode('login')}>
                Login
              </button>
            </>
          )}
        </p>
      </form>
    </div>
  );
}
```


### 📄 src\components\MarkdownCheatsheet.jsx

```
import React from 'react';

function Row({ label, example }) {
  return (
    <div style={{ display: 'flex', gap: '1rem', marginBottom: '0.5rem' }}>
      <div style={{ minWidth: 140, color: 'var(--text-muted)', fontWeight: 500 }}>{label}</div>
      <pre style={{
        margin: 0,
        padding: '0.5rem 0.75rem',
        background: 'var(--bg-primary)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        overflowX: 'auto',
        fontSize: '0.8125rem',
        lineHeight: 1.5
      }}>
        <code>{example}</code>
      </pre>
    </div>
  );
}

export default function MarkdownCheatsheet({ isOpen, onClose }) {
  if (!isOpen) return null;
  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="md-cheatsheet-title">
      <div className="modal">
        <div className="modal-header">
          <h3 id="md-cheatsheet-title">Markdown Cheatsheet</h3>
        </div>

        <div className="modal-body">
          <p style={{ marginBottom: '0.75rem' }}>
            Quick reference for common Markdown syntax supported in the editor and preview.
          </p>

          <Row label="Headers" example={
`# H1
## H2
### H3
#### H4
##### H5
###### H6`
          } />

          <Row label="Bold / Italic" example={
`**bold** or __bold__
*italic* or _italic_
~~strikethrough~~`
          } />

          <Row label="Lists" example={
`- item A
- item B
  - nested
1. First
2. Second`
          } />

          <Row label="Links / Images" example={
`[OpenAI](https://openai.com)
![Alt text](https://placehold.co/200x100)`
          } />

          <Row label="Inline code" example={
"`code` with backticks"
          } />

          <Row label="Code block" example={
"```js\nfunction hello(name) {\n  console.log('Hello ' + name);\n}\n```"
          } />

          <Row label="Blockquote" example={
`> A wise quote
> - Author`
          } />

          <Row label="Tables (GFM)" example={
`| Name | Role  |
|-----:|:-----:|
| Alice| Admin |
| Bob  | User  |`
          } />
        </div>

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 src\components\MarkdownPreview.jsx

```
import React, { forwardRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';

const MarkdownPreview = forwardRef(function MarkdownPreview({ content, className = '', style, ...divProps }, ref) {
  return (
    <div ref={ref} className={`markdown-preview ${className}`} style={style} {...divProps}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          a: ({ node, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer" />
          ),
          code({ node, inline, className, children, ...props }) {
            const text = String(children).replace(/\n$/, '');
            if (inline) {
              return (
                <code className={`inline-code ${className || ''}`.trim()} {...props}>
                  {text}
                </code>
              );
            }
            return (
              <pre className={`code-block ${className || ''}`.trim()}>
                <code {...props}>{text}</code>
              </pre>
            );
          },
          img: ({ node, ...props }) => <img {...props} loading="lazy" />,
        }}
      >
        {content || ''}
      </ReactMarkdown>
    </div>
  );
});

export default MarkdownPreview;
```


### 📄 src\components\NoteEditor.jsx

```
import React, { useState, useEffect, useMemo, useRef, forwardRef, useImperativeHandle } from 'react';
import MarkdownPreview from './MarkdownPreview';
import MarkdownCheatsheet from './MarkdownCheatsheet';

export default forwardRef(function NoteEditor({ note, onSave, onCancel, onDirtyChange, onFindSimilar }, ref) {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [tags, setTags] = useState('');
  const [original, setOriginal] = useState({ title: '', content: '', tags: '' });

  // Markdown editor modes and helpers
  const [viewMode, setViewMode] = useState('edit'); // 'edit' | 'preview' | 'split'
  const [cheatsheetOpen, setCheatsheetOpen] = useState(false);
  const textareaRef = useRef(null);
  const previewRef = useRef(null);
  const syncingRef = useRef(false);
 
  // Load note into form and establish original snapshot
  useEffect(() => {
    if (note) {
      const t = note.title || '';
      const c = note.content || '';
      const g = note.tags || '';
      setTitle(t);
      setContent(c);
      setTags(g);
      setOriginal({ title: t, content: c, tags: g });
    } else {
      setTitle('');
      setContent('');
      setTags('');
      setOriginal({ title: '', content: '', tags: '' });
    }
  }, [note]);

  // Dirty detection
  const isDirty = useMemo(() => {
    return title !== original.title || content !== original.content || tags !== original.tags;
  }, [title, content, tags, original]);

  // Notify parent when dirty state changes
  const lastDirty = useRef(isDirty);
  useEffect(() => {
    if (lastDirty.current !== isDirty) {
      lastDirty.current = isDirty;
      if (typeof onDirtyChange === 'function') onDirtyChange(isDirty);
    }
  }, [isDirty, onDirtyChange]);

  // Warn when trying to close/refresh tab with unsaved changes
  useEffect(() => {
    const handleBeforeUnload = (e) => {
      if (!isDirty) return;
      e.preventDefault();
      e.returnValue = '';
      return '';
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty]);

  const buildNoteData = () => {
    const noteData = {
      ...note,
      title: title.trim(),
      content: content.trim(),
      tags: tags.trim(),
      updatedAt: new Date().toISOString()
    };
    if (!note?.id) {
      noteData.createdAt = new Date().toISOString();
    }
    return noteData;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!title.trim()) {
      alert('Title is required');
      return;
    }
    if (!content.trim()) {
      alert('Content is required');
      return;
    }

    const noteData = buildNoteData();
    onSave(noteData);
    // Mark clean after successful save
    setOriginal({ title: noteData.title, content: noteData.content, tags: noteData.tags });
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && (e.key === 'Enter' || e.key.toLowerCase() === 's')) {
      e.preventDefault();
      handleSubmit(e);
    } else if (e.key === 'Escape') {
      onCancel();
    }
  };

  // Expose imperative API to parent (App) for Save & Continue flow
  useImperativeHandle(ref, () => ({
    isDirty: () => isDirty,
    getCurrentData: () => buildNoteData(),
    submit: () => {
      const fakeEvt = { preventDefault: () => {} };
      handleSubmit(fakeEvt);
    }
  }), [isDirty, title, content, tags, note]);

  // Scroll sync between editor and preview
  const syncScroll = (from) => {
    if (syncingRef.current) return;
    const ta = textareaRef.current;
    const pv = previewRef.current;
    if (!ta || !pv) return;

    const src = from === 'textarea' ? ta : pv;
    const dst = from === 'textarea' ? pv : ta;

    const srcScrollable = Math.max(1, src.scrollHeight - src.clientHeight);
    const ratio = src.scrollTop / srcScrollable;
    const dstScrollable = Math.max(1, dst.scrollHeight - dst.clientHeight);
    syncingRef.current = true;
    try {
      dst.scrollTop = ratio * dstScrollable;
    } finally {
      // release lock on next tick to avoid feedback loop
      setTimeout(() => { syncingRef.current = false; }, 0);
    }
  };

  const handleTextareaScroll = () => syncScroll('textarea');
  const handlePreviewScroll = () => syncScroll('preview');

  const isEditing = note?.id !== undefined;
 
  return (
    <div className="note-editor">
      <div className="editor-header">
        <h2>{isEditing ? 'Edit Note' : 'New Note'}</h2>
        {isDirty && (
          <div className="unsaved-indicator">
            <span className="unsaved-dot" />
            <span>Unsaved changes</span>
          </div>
        )}
      </div>
      
      <form onSubmit={handleSubmit} className="editor-form">
        <div className="form-group">
          <label className="form-label">Title</label>
          <input
            type="text"
            placeholder="Note title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={handleKeyDown}
            className="form-input"
            autoFocus
          />
        </div>

        <div className="form-group" style={{ flex: 1 }}>
          <div className="content-toolbar">
            <label className="form-label">Content</label>
            <div className="content-toolbar-actions">
              <div className="toggle-switch" role="group" aria-label="Editor mode">
                <button
                  type="button"
                  className={`toggle-btn ${viewMode === 'edit' ? 'active' : ''}`}
                  aria-pressed={viewMode === 'edit'}
                  onClick={() => setViewMode('edit')}
                  title="Edit markdown"
                >
                  Edit
                </button>
                <button
                  type="button"
                  className={`toggle-btn ${viewMode === 'preview' ? 'active' : ''}`}
                  aria-pressed={viewMode === 'preview'}
                  onClick={() => setViewMode('preview')}
                  title="Preview formatted markdown"
                >
                  Preview
                </button>
                <button
                  type="button"
                  className={`toggle-btn ${viewMode === 'split' ? 'active' : ''}`}
                  aria-pressed={viewMode === 'split'}
                  onClick={() => setViewMode('split')}
                  title="Edit and preview side-by-side"
                >
                  Split
                </button>
              </div>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setCheatsheetOpen(true)}
                title="Markdown syntax help"
              >
                Cheatsheet
              </button>
            </div>
          </div>

          {viewMode === 'edit' && (
            <textarea
              ref={textareaRef}
              placeholder="Write your note here..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
              onKeyDown={handleKeyDown}
              onScroll={handleTextareaScroll}
              className="form-input form-textarea"
            />
          )}

          {viewMode === 'preview' && (
            <MarkdownPreview
              ref={previewRef}
              content={content}
              className="form-input markdown-preview-only"
              style={{ minHeight: 300, overflow: 'auto' }}
              onScroll={handlePreviewScroll}
            />
          )}

          {viewMode === 'split' && (
            <div className="split-container">
              <div className="split-pane split-pane-editor">
                <textarea
                  ref={textareaRef}
                  placeholder="Write your note here..."
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onScroll={handleTextareaScroll}
                  className="form-input form-textarea"
                />
              </div>
              <div className="split-pane split-pane-preview">
                <MarkdownPreview
                  ref={previewRef}
                  content={content}
                  className="markdown-pane"
                  style={{ height: '100%', overflow: 'auto' }}
                  onScroll={handlePreviewScroll}
                />
              </div>
            </div>
          )}

          <div className="char-count">
            {content.length} characters
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Tags</label>
          <input
            type="text"
            placeholder="comma, separated, tags"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            onKeyDown={handleKeyDown}
            className="form-input"
          />
        </div>

        <div className="editor-actions">
          <button type="submit" className="btn btn-primary">
            {isEditing ? 'Update' : 'Create'}
          </button>
          <button type="button" onClick={onCancel} className="btn btn-secondary">
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onFindSimilar && onFindSimilar()}
            className="btn btn-secondary"
            title="Find notes similar to this one"
          >
            Find Similar
          </button>
          <span className="keyboard-hint">
            Ctrl+Enter or Ctrl+S to save • Esc to cancel
          </span>
        </div>
      </form>

      <MarkdownCheatsheet
        isOpen={cheatsheetOpen}
        onClose={() => setCheatsheetOpen(false)}
      />
    </div>
  );
});
```


### 📄 src\components\NotesList.jsx

```
import React, { useState, useMemo } from 'react';

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
          {processedNotes.map((note) => (
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
                  <div className="why-match">
                    {bestWhySnippet(note.content, searchTerm)}
                  </div>
                ) : (
                  <p className="note-preview">
                    {note.content.substring(0, PREVIEW_LENGTH)}
                    {note.content.length > PREVIEW_LENGTH && '...'}
                  </p>
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
          ))}
        </div>
      )}
    </div>
  );
}
```


### 📄 src\components\SimilarNotesModal.jsx

```
import React, { useEffect, useState } from 'react';
import apiService from '../services/api';

export default function SimilarNotesModal({
  isOpen = false,
  baseDoc = '',
  baseTitle = 'This note',
  notes = [],
  excludeIndex = null,
  topK = 8,
  onClose,
  onSelect,
  onLink
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState([]);

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError('');
      try {
        if (!baseDoc?.trim()) {
          setResults([]);
          setError('Note content is empty.');
          return;
        }
        const [embs, baseEmb] = await Promise.all([
          apiService.getEmbeddingsForNotes(notes),
          apiService.embedText(baseDoc),
        ]);
        const scored = [];
        for (let i = 0; i < notes.length; i++) {
          if (excludeIndex != null && i === excludeIndex) continue;
          const v = embs[i];
          if (!Array.isArray(v)) continue;
          const score = apiService.cosineSimilarity(baseEmb, v);
          scored.push({
            index: i,
            score,
            percent: Math.round(score * 100),
            title: notes[i]?.title || '(Untitled)',
            preview: (notes[i]?.content || '').substring(0, 160)
          });
        }
        scored.sort((a, b) => b.score - a.score);
        const top = scored.slice(0, topK);
        if (!cancelled) setResults(top);
      } catch (e) {
        console.error('Find similar failed:', e);
        if (!cancelled) setError(e?.message || 'Failed to compute similarities');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [isOpen, baseDoc, notes, excludeIndex, topK]);

  if (!isOpen) return null;

  const stop = (e) => e.stopPropagation();

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" onClick={stop}>
        <div className="modal-header">
          <h3>Notes similar to “{baseTitle || 'Untitled'}”</h3>
        </div>
        <div className="modal-body">
          {loading ? (
            <div className="loading" style={{ position: 'static', transform: 'none', padding: 0 }}>
              <div className="loading-spinner" />
              <div>Computing similarities...</div>
            </div>
          ) : error ? (
            <div className="error-banner" style={{ position: 'static' }}>
              ⚠ {error}
            </div>
          ) : results.length === 0 ? (
            <div className="empty-message">
              <div className="empty-message-title">No similar notes found</div>
              <div className="empty-message-hint">Try adding more content or different keywords</div>
            </div>
          ) : (
            <div className="similar-list">
              {results.map((r) => (
                <div key={r.index} className="similar-item">
                  <div
                    className="similar-main"
                    onClick={() => onSelect && onSelect(r.index)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter') onSelect && onSelect(r.index); }}
                  >
                    <div className="similar-header">
                      <div className="similar-title">{r.title}</div>
                      <div className="similar-score">{r.percent}% similar</div>
                    </div>
                    <div className="similar-preview">
                      {r.preview}{(notes[r.index]?.content || '').length > r.preview.length ? '…' : ''}
                    </div>
                    <div className="similar-meter">
                      <div className="similar-meter-fill" style={{ width: `${Math.min(100, Math.max(0, r.percent))}%` }} />
                    </div>
                  </div>
                  <div className="similar-actions">
                    <button className="btn btn-secondary btn-sm" onClick={() => onSelect && onSelect(r.index)} title="Open note">Open</button>
                    <button className="btn btn-primary btn-sm" onClick={() => onLink && onLink(r.index)} title="Link notes">Link</button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 src\components\ToastNotification.jsx

```
import React, { useEffect, useRef } from 'react';

export default function ToastNotification({
  isOpen = false,
  message = '',
  actionLabel = 'Undo',
  onAction,
  onClose,
  duration = 5000
}) {
  const timerRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      if (onClose) onClose();
    }, duration);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isOpen, message, duration, onClose]);

  if (!isOpen) return null;

  return (
    <div className="toast-container" role="status" aria-live="polite">
      <div className="toast-card fade-in">
        <div className="toast-message">{message}</div>

        <div className="toast-actions">
          {onAction && (
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => {
                if (onAction) onAction();
                if (onClose) onClose();
              }}
            >
              {actionLabel}
            </button>
          )}
          <button
            className="toast-close"
            aria-label="Dismiss notification"
            title="Dismiss"
            onClick={() => onClose && onClose()}
          >
            ×
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 src\components\TrashView.jsx

```
import React, { useMemo, useState } from 'react';
import ConfirmDialog from './ConfirmDialog';

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

export default function TrashView({
  trashedNotes = [],
  onRestore,
  onDeleteForever,
  onEmptyTrash
}) {
  const [confirmId, setConfirmId] = useState(null);
  const [confirmEmpty, setConfirmEmpty] = useState(false);

  const items = useMemo(() => {
    // Sort by deletedAt desc
    return [...trashedNotes].sort((a, b) =>
      new Date(b.deletedAt || 0) - new Date(a.deletedAt || 0)
    );
  }, [trashedNotes]);

  return (
    <div className="trash-view">
      <div className="list-header">
        <h3>Trash ({items.length})</h3>
        <div className="list-controls">
          <button
            className="btn btn-danger"
            disabled={items.length === 0}
            onClick={() => setConfirmEmpty(true)}
            title="Permanently delete all trashed notes"
          >
            Empty Trash
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="empty-message">
          <div className="empty-message-title">Trash is empty</div>
          <div className="empty-message-hint">Deleted notes will appear here</div>
        </div>
      ) : (
        <div className="trash-items">
          {items.map(note => (
            <div key={note.id} className="trash-item">
              <div className="trash-item-content">
                <div className="trash-item-header">
                  <h4 className="note-item-title">{note.title}</h4>
                  <span className="note-date">Deleted {formatRelativeTime(note.deletedAt)}</span>
                </div>
                <p className="note-preview">
                  {note.content.substring(0, PREVIEW_LENGTH)}
                  {note.content.length > PREVIEW_LENGTH && '...'}
                </p>
                {note.tags && (
                  <div className="note-tags">
                    {note.tags.split(',').map((tag, i) => (
                      <span key={i} className="tag">{tag.trim()}</span>
                    ))}
                  </div>
                )}
              </div>

              <div className="note-item-actions">
                <button
                  className="action-btn"
                  title="Restore"
                  onClick={() => onRestore && onRestore(note.id)}
                >
                  Restore
                </button>
                <button
                  className="action-btn delete"
                  title="Delete Forever"
                  onClick={() => setConfirmId(note.id)}
                >
                  Delete Forever
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <ConfirmDialog
        isOpen={confirmId !== null}
        title="Delete Forever?"
        message="This will permanently delete the note from trash. This action cannot be undone."
        confirmLabel="Delete Forever"
        cancelLabel="Cancel"
        danger
        onConfirm={() => {
          if (confirmId != null && onDeleteForever) {
            onDeleteForever(confirmId);
          }
          setConfirmId(null);
        }}
        onCancel={() => setConfirmId(null)}
      />

      <ConfirmDialog
        isOpen={confirmEmpty}
        title="Empty Trash?"
        message="All notes in the trash will be permanently deleted. This action cannot be undone."
        confirmLabel="Empty Trash"
        cancelLabel="Cancel"
        danger
        onConfirm={() => {
          if (onEmptyTrash) onEmptyTrash();
          setConfirmEmpty(false);
        }}
        onCancel={() => setConfirmEmpty(false)}
      />
    </div>
  );
}
```


### 📄 src\components\UnsavedChangesDialog.jsx

```
import React, { useEffect } from 'react';

export default function UnsavedChangesDialog({ 
  isOpen = false,
  onSaveAndContinue,
  onDiscard,
  onCancel
}) {
  if (!isOpen) return null;

  useEffect(() => {
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel && onCancel();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
        e.preventDefault();
        onSaveAndContinue && onSaveAndContinue();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        onSaveAndContinue && onSaveAndContinue();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onCancel, onSaveAndContinue]);

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-labelledby="unsaved-title">
      <div className="modal">
        <div className="modal-header">
          <h3 id="unsaved-title">Unsaved changes</h3>
        </div>
        <div className="modal-body">
          <p>You have unsaved changes. If you continue without saving, your edits will be lost.</p>
        </div>
        <div className="modal-actions">
          <button
            className="btn btn-primary"
            onClick={() => onSaveAndContinue && onSaveAndContinue()}
            autoFocus
          >
            Save & Continue
          </button>
          <button
            className="btn btn-danger"
            onClick={() => onDiscard && onDiscard()}
          >
            Discard
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => onCancel && onCancel()}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
```


### 📄 src\contexts\AuthContext.jsx

```
import { createContext, useContext, useState, useEffect } from 'react';
import apiService from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const validateAndRestoreSession = async () => {
      const storedToken = localStorage.getItem('auth_token');
      if (!storedToken) {
        setLoading(false);
        return;
      }

      apiService.setAuthToken(storedToken);

      try {
        const userResponse = await apiService.request('/api/auth/me');
        if (userResponse && userResponse.username) {
          setUser({ username: userResponse.username, userId: userResponse.user_id });
          setToken(storedToken);
          setIsAuthenticated(true);
        } else {
          throw new Error('Invalid token');
        }
      } catch (error) {
        console.error('Token validation failed:', error);
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_user');
        setIsAuthenticated(false);
      } finally {
        setLoading(false);
      }
    };

    validateAndRestoreSession();
  }, []);

  const login = async (username, password) => {
    const response = await apiService.login(username, password);
    const { access_token, username: user_name, user_id } = response;
    
    setToken(access_token);
    setUser({ username: user_name, userId: user_id });
    setIsAuthenticated(true);
    
    localStorage.setItem('auth_token', access_token);
    localStorage.setItem('auth_user', JSON.stringify({ username: user_name, userId: user_id }));
    apiService.setAuthToken(access_token);
    
    return response;
  };

  const register = async (username, password, email) => {
    const response = await apiService.register(username, password, email);
    const { access_token, username: user_name, user_id } = response;
    
    setToken(access_token);
    setUser({ username: user_name, userId: user_id });
    setIsAuthenticated(true);
    
    localStorage.setItem('auth_token', access_token);
    localStorage.setItem('auth_user', JSON.stringify({ username: user_name, userId: user_id }));
    apiService.setAuthToken(access_token);
    
    return response;
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    setIsAuthenticated(false);
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_user');
    apiService.setAuthToken(null);
  };

  const value = {
    user,
    token,
    isAuthenticated,
    loading,
    login,
    register,
    logout
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
```


### 📄 src\hooks\useNotes.js

```javascript
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
      saveAllToStorage([dbNote, ...notes], trashedNotes);
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
      saveAllToStorage(newList, trashedNotes);
    } catch (err) {
      console.error('Failed to update note:', err);
      setError('Failed to update note');
      throw err;
    }
  }, [isAuthenticated, notes, trashedNotes]);

  const deleteNote = useCallback((index) => {
    setNotes(prev => prev.filter((_, i) => i !== index));
  }, []);

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
      saveAllToStorage(notes.filter((_, i) => i !== index), [trashedNote, ...trashedNotes]);
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
      saveAllToStorage([restoredNote, ...notes], updatedTrash);
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
      saveAllToStorage(notes, updatedTrash);
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
      saveAllToStorage(notes, []);
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
      saveAllToStorage(updatedNotes, trashedNotes);

      return { success: true, imported: importedNotes.length };
    } catch (err) {
      console.error('Failed to import notes:', err);
      setError('Failed to import notes');
      throw err;
    }
  }, [trashedNotes]);
  
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
    deleteNote,
    moveToTrash,
    restoreFromTrash,
    permanentDelete,
    emptyTrash,
    searchNotes,
    getAllTags,
    exportNotes,
    importNotes,
    getStats
  };
}
```


### 📄 src\main.jsx

```
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

import { AuthProvider } from './contexts/AuthContext'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AuthProvider>
      <App />
    </AuthProvider>
  </React.StrictMode>,
)
```


### 📄 src\services\api.js

```javascript
const EMB_LS_KEY = 'semantic-emb-cache-v1';

class APIService {
  constructor(baseUrl = import.meta.env.VITE_API_BASE_URL) {
    this.baseUrl = (baseUrl || '').replace(/\/$/, '');
    // Embedding cache: { [noteId]: { h: number, v: number[] } }
    this._embCache = this._loadEmbCache();
  }

  // Add token storage
  static authToken = null;

  setAuthToken(token) {
    APIService.authToken = token;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (APIService.authToken) {
      headers['Authorization'] = `Bearer ${APIService.authToken}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  async checkHealth() {
    return this.request('/api/health');
  }

  async getStats() {
    return this.request('/api/stats');
  }

  async embedDocuments(documents) {
    if (!documents?.length) {
      throw new Error('No documents provided');
    }
    return this.request('/api/embed', {
      method: 'POST',
      body: JSON.stringify({ documents })
    });
  }

  async buildGraph(options) {
    const {
      documents,
      labels = null,

      // legacy params (kept for back-compat)
      mode = 'knn',
      top_k = 2,
      threshold = 0.3,
      dr_method = 'pca',
      n_components = 2,
      cluster = 'none',

      // new canonical params
      connection = undefined,           // 'knn' | 'threshold'
      k_neighbors = undefined,          // 1-10
      similarity_threshold = undefined, // 0-1
      dim_reduction = undefined,        // 'pca' | 'umap' | 'tsne' | null
      clustering = undefined,           // 'kmeans' | 'agglomerative' | null

      n_clusters = null,
      include_embeddings = false,
    } = options || {};

    if (!documents?.length) {
      throw new Error('No documents provided');
    }

    const conn = (connection ?? (mode === 'threshold' ? 'threshold' : 'knn')) === 'threshold' ? 'threshold' : 'knn';

    const k = k_neighbors ?? top_k ?? 2;
    const th = similarity_threshold ?? threshold ?? 0.3;

    // allow null to disable DR
    const dr = dim_reduction === undefined ? dr_method : dim_reduction;
    // normalize clustering
    const clust = clustering === undefined
      ? (cluster === 'none' ? null : cluster)
      : clustering;

    const payload = {
      documents,
      n_components: n_components ?? 2,
      include_embeddings
    };

    if (labels) payload.labels = labels;

    // Ensure explicit, backend-friendly values for DR and clustering
    payload.dr_method = dr === null ? 'none' : (dr ?? 'pca');
    payload.cluster = clust ? clust : 'none';
    if (clust && n_clusters != null) {
      payload.n_clusters = n_clusters;
    }

    if (conn === 'knn') {
      payload.top_k = k;
    } else {
      payload.threshold = th;
    }

    return this.request('/api/graph', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  // ---------- Embeddings: helpers, caching, similarity ----------

  getNoteText(note) {
    const title = String(note?.title || '').trim();
    const content = String(note?.content || '').trim();
    const tags = String(note?.tags || '').trim();
    return `${title}. ${content} ${tags}`.trim();
  }

  _hashString(str) {
    // FNV-1a 32-bit
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = (h >>> 0) * 0x01000193;
    }
    return h >>> 0;
  }

  hashNote(note) {
    return this._hashString(`${note?.id ?? 'new'}::${this.getNoteText(note)}`);
  }

  _loadEmbCache() {
    try {
      const raw = localStorage.getItem(EMB_LS_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') return parsed;
    } catch (e) {
      console.warn('Embedding cache load failed:', e);
    }
    return {};
  }

  _saveEmbCache() {
    try {
      localStorage.setItem(EMB_LS_KEY, JSON.stringify(this._embCache));
    } catch (e) {
      // If quota exceeded, drop cache silently
      console.warn('Embedding cache save failed:', e);
    }
  }

  async embedText(text) {
    const res = await this.embedDocuments([text]);
    const emb = res?.embeddings?.[0];
    if (!emb) throw new Error('Failed to compute embedding');
    return emb;
  }

  async saveEmbeddingsToDatabase(embeddings) {
    try {
      await this.request('/api/embeddings/batch', {
        method: 'POST',
        body: JSON.stringify({ embeddings }),
      });
    } catch (error) {
      console.error('Failed to save embeddings to database:', error);
    }
  }

  async fetchEmbeddingsFromDatabase(noteIds) {
    try {
      const response = await this.request(`/api/embeddings?note_ids=${noteIds.join(',')}`);
      return response.embeddings || {};
    } catch (error) {
      console.error('Failed to fetch embeddings from database:', error);
      return {};
    }
  }

  async getEmbeddingsForNotes(notes) {
    if (!notes || notes.length === 0) return {};

    const cache = this._loadEmbCache();
    const embeddings = {};
    const notesToCompute = [];
    const notesToFetchFromDb = [];
    const authToken = APIService.authToken;

    for (const note of notes) {
      const cached = cache[note.id];
      const noteHash = this.hashNote(note);
      if (cached && cached.h === noteHash) {
        embeddings[note.id] = cached.v;
      } else if (authToken) {
        notesToFetchFromDb.push(note);
      } else {
        notesToCompute.push(note);
      }
    }

    if (notesToFetchFromDb.length > 0 && authToken) {
      const noteIds = notesToFetchFromDb.map(n => n.id);
      const dbEmbeddings = await this.fetchEmbeddingsFromDatabase(noteIds);

      for (const note of notesToFetchFromDb) {
        const dbEmb = dbEmbeddings[note.id];
        const noteHash = this.hashNote(note);
        if (dbEmb && dbEmb.content_hash === noteHash) {
          embeddings[note.id] = dbEmb.embedding;
          cache[note.id] = { h: noteHash, v: dbEmb.embedding };
        } else {
          notesToCompute.push(note);
        }
      }
      this._saveEmbCache();
    }

    if (notesToCompute.length > 0) {
      const texts = notesToCompute.map(n => this.getNoteText(n));
      const res = await this.embedDocuments(texts);
      const vecs = res.embeddings || [];
      for (let i = 0; i < notesToCompute.length; i++) {
        const note = notesToCompute[i];
        embeddings[note.id] = vecs[i];
        cache[note.id] = { h: this.hashNote(note), v: vecs[i] };
      }
      this._saveEmbCache();

      if (authToken && notesToCompute.length > 0) {
        const embeddingsToSave = notesToCompute
          .filter(n => embeddings[n.id])
          .map(n => ({
            note_id: n.id,
            content_hash: this.hashNote(n),
            embedding: embeddings[n.id],
            model_name: 'sentence-transformers/all-MiniLM-L6-v2',
          }));

        if (embeddingsToSave.length > 0) {
          this.saveEmbeddingsToDatabase(embeddingsToSave).catch(err =>
            console.error('Background embedding sync failed:', err)
          );
        }
      }
    }

    return embeddings;
  }

  // Safe cosine similarity (normalizes if needed)
  cosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      const x = a[i];
      const y = b[i];
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }
  
  async register(username, password, email = null) {
    const response = await this.request('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, password, email }),
    });
    return response;
  }

  async login(username, password) {
    const response = await this.request('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
    return response;
  }

  async getCurrentUser() {
    const response = await this.request('/api/auth/me');
    return response;
  }
}

const apiService = new APIService();

export default apiService;
export { APIService };
```


### 📄 src\services\dbApi.js

```javascript
import apiService from './api';

const dbApi = {
  async fetchNotes() {
    return await apiService.request('/api/notes');
  },

  async createNote(noteData) {
    return await apiService.request('/api/notes', {
      method: 'POST',
      body: JSON.stringify(noteData),
    });
  },

  async updateNote(noteId, noteData) {
    return await apiService.request(`/api/notes/${noteId}`, {
      method: 'PUT',
      body: JSON.stringify(noteData),
    });
  },

  async moveToTrash(noteId) {
    return await apiService.request(`/api/notes/${noteId}/trash`, {
      method: 'POST',
    });
  },

  async restoreNote(noteId) {
    return await apiService.request(`/api/notes/${noteId}/restore`, {
      method: 'POST',
    });
  },

  async permanentDelete(noteId) {
    return await apiService.request(`/api/notes/${noteId}`, {
      method: 'DELETE',
    });
  },

  async fetchTrash() {
    return await apiService.request('/api/trash');
  },

  async emptyTrash() {
    return await apiService.request('/api/trash/empty', {
      method: 'POST',
    });
  },

  async importNotes(notes, trash) {
    return await apiService.request('/api/notes/import', {
      method: 'POST',
      body: JSON.stringify({ notes, trash }),
    });
  },
};

export default dbApi;
```


### 📄 src\utils\graphExport.js

```javascript
/**
 * Graph export utilities: PNG, SVG, JSON and clipboard helpers
 */

function pad2(n) {
  return String(n).padStart(2, '0');
}

export function formatTimestamp(date = new Date()) {
  const y = date.getFullYear();
  const m = pad2(date.getMonth() + 1);
  const d = pad2(date.getDate());
  const hh = pad2(date.getHours());
  const mm = pad2(date.getMinutes());
  return `${y}-${m}-${d}-${hh}-${mm}`;
}

export function defaultFilename(format = 'png') {
  const ts = formatTimestamp();
  if (format === 'json') return `semantic-graph-data-${ts}.json`;
  return `semantic-graph-${ts}.${format}`;
}

/**
 * Serialize an SVG element to string, ensuring necessary namespaces.
 */
export function getSvgString(svgEl) {
  if (!svgEl) throw new Error('SVG element is required');
  const cloned = svgEl.cloneNode(true);

  // Ensure width/height attributes exist (for canvas rasterization)
  const width = Number(svgEl.getAttribute('width') || svgEl.clientWidth || 800);
  const height = Number(svgEl.getAttribute('height') || svgEl.clientHeight || 600);
  cloned.setAttribute('width', String(width));
  cloned.setAttribute('height', String(height));

  // Add xmlns if missing
  if (!cloned.getAttribute('xmlns')) {
    cloned.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  }
  if (!cloned.getAttribute('xmlns:xlink')) {
    cloned.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
  }

  // Serialize
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(cloned);

  // Fix for some browsers that omit namespaces
  if (!source.match(/^<svg[^>]+xmlns="/)) {
    source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  if (!source.match(/^<svg[^>]+"http:\/\/www\.w3\.org\/1999\/xlink"/)) {
    source = source.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
  }

  // Ensure proper XML header
  return `<?xml version="1.0" standalone="no"?>${source}`;
}

/**
 * Download a Blob with the given filename.
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }, 0);
}

/**
 * Convert an SVG element to a PNG Blob by drawing into a high-DPI canvas.
 * scale: rasterization scale factor (2 for 2x resolution)
 */
export async function svgToPng(svgEl, scale = 2) {
  const width = Number(svgEl.getAttribute('width') || svgEl.clientWidth || 800);
  const height = Number(svgEl.getAttribute('height') || svgEl.clientHeight || 600);
  const svgString = getSvgString(svgEl);

  const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  try {
    const img = await new Promise((resolve, reject) => {
      const image = new Image();
      // Important: set crossOrigin to avoid taint if external resources are referenced
      image.crossOrigin = 'anonymous';
      image.onload = () => resolve(image);
      image.onerror = (e) => reject(new Error('Failed to load SVG for rasterization'));
      image.src = url;
    });

    const canvas = document.createElement('canvas');
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);

    const ctx = canvas.getContext('2d');
    // High quality rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Scale context and draw the SVG image
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, width, height);

    const pngBlob = await new Promise((resolve) => {
      canvas.toBlob((b) => resolve(b), 'image/png');
    });

    if (!pngBlob) throw new Error('Canvas export produced empty image');
    return pngBlob;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function savePng(svgEl, filename, scale = 2) {
  const blob = await svgToPng(svgEl, scale);
  downloadBlob(blob, filename);
  return blob;
}

export function saveSvg(svgEl, filename) {
  const svgString = getSvgString(svgEl);
  const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  downloadBlob(svgBlob, filename);
  return svgBlob;
}

export function buildJsonExport(graphData, metadata = {}) {
  const payload = {
    type: 'semantic-graph',
    version: 1,
    created_at: new Date().toISOString(),
    metadata,
    nodes: Array.isArray(graphData?.nodes) ? graphData.nodes : [],
    edges: Array.isArray(graphData?.edges) ? graphData.edges : [],
  };
  return JSON.stringify(payload, null, 2);
}

export function saveJson(graphData, filename, metadata = {}) {
  const json = buildJsonExport(graphData, metadata);
  const blob = new Blob([json], { type: 'application/json;charset=utf-8' });
  downloadBlob(blob, filename);
  return blob;
}

/**
 * Clipboard helpers
 */
export async function copyBlobToClipboard(blob, mime) {
  if (!navigator?.clipboard || !window.ClipboardItem) {
    throw new Error('Clipboard API not supported');
  }
  const item = new ClipboardItem({ [mime]: blob });
  await navigator.clipboard.write([item]);
}

export async function copyPng(svgEl, scale = 2) {
  const blob = await svgToPng(svgEl, scale);
  await copyBlobToClipboard(blob, 'image/png');
}

export async function copySvg(svgEl) {
  // Prefer writing as image/svg+xml if supported, otherwise as text
  const svgString = getSvgString(svgEl);
  if (navigator?.clipboard && window.ClipboardItem) {
    const blob = new Blob([svgString], { type: 'image/svg+xml' });
    try {
      await copyBlobToClipboard(blob, 'image/svg+xml');
      return;
    } catch {
      // fallback to text
    }
  }
  await navigator.clipboard.writeText(svgString);
}

export async function copyJson(graphData, metadata = {}) {
  const json = buildJsonExport(graphData, metadata);
  await navigator.clipboard.writeText(json);
}
```


### 📄 vite.config.js

```javascript
// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,        // <— add this
    open: true,
    // Proxy API requests to the FastAPI backend
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // Optimize chunks
    rollupOptions: {
      output: {
        manualChunks: {
          'd3': ['d3'],
          'react-vendor': ['react', 'react-dom'],
        }
      }
    }
  }
})
```
