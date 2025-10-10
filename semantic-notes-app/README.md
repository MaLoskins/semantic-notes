# Semantic Notes App

A powerful note-taking application with AI-powered semantic graph visualization using your FastAPI backend.

## Features

- 📝 **Rich Note Editor**: Create and edit notes with title, content, and tags
- 🧠 **Semantic Graph Visualization**: Visualize connections between notes based on semantic similarity
- 🔍 **Smart Search**: Search through notes by title, content, or tags
- 🎨 **Dark Theme**: Beautiful dark-themed interface optimized for long writing sessions
- 📊 **D3.js Powered Graphs**: Interactive force-directed graphs with zoom and drag capabilities
- 💾 **Local Storage**: Notes are automatically saved to browser local storage
- 📤 **Export/Import**: Export your notes as JSON for backup
- 🏷️ **Tag System**: Organize notes with tags for better categorization

## Prerequisites

1. **Backend Server**: Ensure your FastAPI backend is running on `http://localhost:8000`
   ```bash
   # In your backend directory
   python main.py
   ```

2. **Node.js**: Version 16.0 or higher
3. **npm** or **yarn**: Package manager

## Project Structure

```
semantic-notes-app/
├── src/
│   ├── components/
│   │   ├── NoteEditor.jsx
│   │   ├── NotesList.jsx
│   │   └── GraphVisualization.jsx
│   ├── hooks/
│   │   └── useNotes.js
│   ├── services/
│   │   └── api.js
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
├── package.json
├── vite.config.js
├── index.html
└── README.md
```

## Installation

1. **Create the project directory and file structure:**

```bash
mkdir semantic-notes-app
cd semantic-notes-app
```

2. **Create the directory structure:**

```bash
mkdir -p src/components src/hooks src/services
```

3. **Copy all the provided files into their respective locations**

4. **Install dependencies:**

```bash
npm install
# or
yarn install
```

5. **Create index.html in the root directory:**

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

6. **Create src/main.jsx:**

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

## Running the Application

1. **Start your FastAPI backend** (in a separate terminal):
```bash
cd /path/to/backend
python main.py
```

2. **Start the React development server:**
```bash
npm run dev
# or
yarn dev
```

The application will open automatically in your browser at `http://localhost:3000`

## Usage

### Creating Notes
1. Click the "➕ New Note" button in the header
2. Enter a title and content for your note
3. Optionally add comma-separated tags
4. Press "Create Note" or use `Ctrl+Enter` to save

### Viewing the Semantic Graph
- The graph automatically updates when you have 2 or more notes
- Each node represents a note
- Edge thickness represents semantic similarity (thicker = more similar)
- Drag nodes to reposition them
- Click nodes to preview the note content
- Use mouse wheel to zoom in/out
- Click 🎯 to reset the view
- Click ⏸️/▶️ to pause/resume physics simulation

### Managing Notes
- Click on a note in the sidebar to select it
- Use the ✏️ button to edit
- Use the 🗑️ button to delete
- Use the search bar to filter notes
- Sort notes by update time, creation time, or title
- Filter by tags using the dropdown

### Graph Behavior
- **KNN Mode**: Each note connects to its 2 nearest semantic neighbors (as configured)
- **PCA Dimensionality Reduction**: Positions nodes in 2D space based on semantic similarity

## API Configuration

The app is configured to use your backend with:
- **KNN mode** with `k=2` (each note connects to 2 most similar notes)
- **PCA** for dimensionality reduction
- Endpoint: `http://localhost:8000/api/graph`

To modify the backend URL, edit `src/services/api.js`:

```javascript
const apiService = new APIService('http://your-backend-url:port');
```

## Build for Production

```bash
npm run build
# or
yarn build
```

The built files will be in the `dist/` directory.

## Troubleshooting

### Backend Connection Issues
- Ensure the FastAPI server is running on port 8000
- Check CORS is enabled in your backend
- Verify the backend health endpoint: `http://localhost:8000/api/health`

### Graph Not Showing
- Need at least 2 notes for the graph to generate
- Check browser console for API errors
- Ensure the backend embedding service is properly initialized

### Performance Issues
- The app uses local storage for persistence
- Large numbers of notes (>100) may impact performance
- Consider adjusting the `top_k` parameter for very large note collections

## Features Roadmap

Potential enhancements that don't require backend changes:
- [ ] Markdown preview for notes
- [ ] Note templates
- [ ] Keyboard shortcuts for all actions
- [ ] Full-text search highlighting
- [ ] Note statistics and analytics
- [ ] Multiple graph layout options
- [ ] Note linking and references
- [ ] Color coding by tags/clusters

## License

MIT