# PostgreSQL Database Integration Plan
## Semantic Note-Taking Application

---

## Executive Summary

This plan outlines the integration of a PostgreSQL database into the semantic note-taking application. The integration maintains the current local-first architecture while enabling multi-user support with user-specific note isolation. The approach prioritizes minimal codebase changes, leveraging existing patterns and introducing database persistence without disrupting the working local functionality.

**Key Principles:**
- Preserve current local functionality and performance
- Add database persistence layer with minimal file changes
- Maintain embedding cache performance optimization
- Simple authentication (no complex security features)
- User-specific data isolation
- Lightweight, maintainable implementation

---

## Current State Analysis

### Data Currently Stored Locally

**1. Notes Data (localStorage key: `semantic-notes-data`)**
- Location: [`useNotes.js`](semantic-notes-app/src/hooks/useNotes.js:3) lines 6-29
- Structure: `{ notes: [...], lastUpdated: ISO timestamp }`
- Note fields: `id, title, content, tags, createdAt, updatedAt`

**2. Trashed Notes (localStorage key: `semantic-notes-trash`)**
- Location: [`useNotes.js`](semantic-notes-app/src/hooks/useNotes.js:4) lines 31-54
- Structure: `{ trash: [...], lastUpdated: ISO timestamp }`
- Additional field: `deletedAt` timestamp
- Auto-cleanup: Items older than 30 days removed automatically

**3. Embedding Cache (localStorage key: `semantic-emb-cache-v1`)**
- Location: [`api.js`](semantic-notes-app/src/services/api.js:1) lines 1-159
- Structure: `{ [noteId]: { h: hash, v: vector[] } }`
- Purpose: Performance optimization to avoid recomputing embeddings
- Contains: 384-dimensional float arrays (sentence-transformers/all-MiniLM-L6-v2)

**4. Graph Control Preferences (localStorage key: `graph-controls-prefs-v1`)**
- Location: [`App.jsx`](semantic-notes-app/src/App.jsx:18-26) lines 18-26, 87-100
- Structure: `{ connection, k_neighbors, similarity_threshold, dim_reduction, clustering, n_clusters }`
- Purpose: UI preferences for graph visualization

### Current Backend Architecture

**Backend Services:**
- [`main.py`](main.py): FastAPI server with `/api/health`, `/api/stats`, `/api/embed`, `/api/graph` endpoints
- [`embedding_service.py`](embedding_service.py): Singleton service for text embeddings (LRU cache, batch encoding)
- [`graph_service.py`](graph_service.py): Graph construction with similarity calculations, dimensionality reduction, clustering

**Frontend Architecture:**
- [`useNotes.js`](semantic-notes-app/src/hooks/useNotes.js) hook: Complete notes CRUD operations with localStorage persistence
- [`api.js`](semantic-notes-app/src/services/api.js) service: API communication layer with embedding cache management
- No existing authentication or user context

---

## Database Schema Design

### Core Tables

#### 1. users
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);
CREATE INDEX idx_users_username ON users(username);
```

**Reasoning:** Simple user identification. Password hashed (bcrypt), minimal fields for lightweight auth.

#### 2. notes
```sql
CREATE TABLE notes (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMP
);
CREATE INDEX idx_notes_user_id ON notes(user_id);
CREATE INDEX idx_notes_user_deleted ON notes(user_id, is_deleted);
```

**Reasoning:** 
- Soft deletes via `is_deleted` flag (supports trash/restore functionality)
- User isolation via `user_id` foreign key with CASCADE deletion
- Indexes for efficient user-specific queries
- TEXT type for `title` and `content` (no length limits, supports long notes)
- `tags` as TEXT (comma-separated, maintains current format)

#### 3. embeddings
```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    note_id INTEGER UNIQUE NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
    content_hash VARCHAR(64) NOT NULL,
    embedding_vector FLOAT4[] NOT NULL,
    model_name VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_embeddings_note_id ON embeddings(note_id);
CREATE INDEX idx_embeddings_hash ON embeddings(content_hash);
```

**Reasoning:**
- `FLOAT4[]` array type for 384-dimensional embeddings (4 bytes per float = ~1.5KB per embedding)
- `content_hash` enables cache invalidation when note content changes
- `model_name` tracks which embedding model generated the vector (future-proofing)
- One-to-one relationship with notes (UNIQUE constraint on `note_id`)
- Automatic cleanup via CASCADE when notes deleted

### Database Size Considerations

**Per User Estimates:**
- 100 notes: ~500KB (notes table)
- 100 embeddings: ~150KB (embeddings table)
- Total per 100 notes: ~650KB

**Storage Efficiency:**
- PostgreSQL efficiently stores array types
- Indexes add ~10-20% overhead
- Expected: 1000 users with 100 notes each = ~650MB

---

## Integration Phases

### Phase 1: Docker PostgreSQL Setup

**Objective:** Set up PostgreSQL database in Docker container with proper configuration for local development.

**Tasks:**

1. **Create Docker Compose configuration**
   - File: `docker-compose.yml` (new file at project root)
   - Service: PostgreSQL 15 (alpine for smaller size)
   - Exposed port: 5432
   - Volume: Persistent data storage in `./postgres-data`
   - Environment variables via `.env` file (never hardcoded)
   - Health check for container readiness

2. **Create environment configuration**
   - File: `.env` (new file at project root, added to `.gitignore`)
   - Variables: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `DATABASE_URL`
   - Template file: `.env.example` for documentation

3. **Create database initialization script**
   - File: `init-db.sql` (new file at project root)
   - Contains: All CREATE TABLE statements from schema design
   - Contains: All CREATE INDEX statements
   - Automatically executed by Docker on first run

4. **Verify Docker setup**
   - Start container: `docker-compose up -d`
   - Check logs: `docker-compose logs postgres`
   - Verify connection: `docker exec -it <container> psql -U <user> -d <db>`
   - Test tables exist: `\dt` command in psql

**Files Created:** `docker-compose.yml`, `.env`, `.env.example`, `init-db.sql`

**Dependencies Added:** None (Docker only)

---

### Phase 2: Backend Database Integration

**Objective:** Add database connectivity and ORM models to Python backend without disrupting existing embedding functionality.

**Tasks:**

1. **Add database dependencies**
   - Update: [`requirements.txt`](requirements.txt)
   - Add: `sqlalchemy`, `psycopg2-binary`, `python-jose[cryptography]`, `passlib[bcrypt]`, `python-multipart`
   - Reasoning: SQLAlchemy (ORM), psycopg2 (PostgreSQL driver), jose (JWT), passlib (password hashing), multipart (form data)

2. **Create database connection module**
   - File: `database.py` (new file at project root)
   - Purpose: Centralized database configuration and session management
   - Contents:
     - `get_db_url()`: Reads `DATABASE_URL` from environment
     - `engine`: SQLAlchemy engine with connection pooling
     - `SessionLocal`: Session factory for database operations
     - `get_db()`: Dependency injection function for FastAPI routes
     - `init_db()`: Initialize database tables (idempotent, safe to run multiple times)

3. **Create database models**
   - File: `models.py` (new file at project root)
   - Purpose: SQLAlchemy ORM models matching schema design
   - Contents:
     - `User` model: Maps to `users` table with relationship to notes
     - `Note` model: Maps to `notes` table with relationship to user and embeddings
     - `Embedding` model: Maps to `embeddings` table with relationship to note
     - Relationships configured for efficient queries (lazy loading, cascade deletes)

4. **Create authentication utilities**
   - File: `auth.py` (new file at project root)
   - Purpose: Simple authentication functions (no complex security)
   - Contents:
     - `hash_password(password: str) -> str`: Bcrypt hashing
     - `verify_password(plain: str, hashed: str) -> bool`: Verify password
     - `create_access_token(user_id: int) -> str`: Generate JWT with 7-day expiry
     - `decode_access_token(token: str) -> int`: Extract user_id from JWT
     - `get_current_user(token: str, db: Session) -> User`: FastAPI dependency for protected routes

5. **Create database service layer**
   - File: `db_service.py` (new file at project root)
   - Purpose: All database CRUD operations in one place (centralized, maintainable)
   - Contents:
     - **User operations**: `create_user()`, `get_user_by_username()`, `authenticate_user()`
     - **Note operations**: `get_user_notes()`, `create_note()`, `update_note()`, `soft_delete_note()`, `restore_note()`, `permanent_delete_note()`, `empty_trash()`
     - **Embedding operations**: `get_note_embedding()`, `save_note_embedding()`, `batch_get_embeddings()`, `invalidate_embedding_cache()`
   - Each function accepts `db: Session` parameter
   - Each function scoped to user_id for isolation
   - Efficient bulk operations where applicable

6. **Update main.py with new routes**
   - File: [`main.py`](main.py)
   - Changes:
     - Import new modules: `database`, `models`, `auth`, `db_service`
     - Add startup hook to `lifespan()`: Call `init_db()` to ensure tables exist
     - Add `/api/auth/register` endpoint (POST): Username/password → create user → return JWT
     - Add `/api/auth/login` endpoint (POST): Username/password → verify → return JWT
     - Add `/api/auth/me` endpoint (GET): Return current user info (requires JWT)
     - Add `/api/notes` endpoint (GET): Get all user notes (requires JWT)
     - Add `/api/notes` endpoint (POST): Create note (requires JWT)
     - Add `/api/notes/{note_id}` endpoint (PUT): Update note (requires JWT)
     - Add `/api/notes/{note_id}/trash` endpoint (POST): Move to trash (requires JWT)
     - Add `/api/notes/{note_id}/restore` endpoint (POST): Restore from trash (requires JWT)
     - Add `/api/notes/{note_id}` endpoint (DELETE): Permanent delete (requires JWT)
     - Add `/api/trash` endpoint (GET): Get trashed notes (requires JWT)
     - Add `/api/trash/empty` endpoint (POST): Empty trash (requires JWT)
   - Keep existing `/api/embed` and `/api/graph` endpoints unchanged (they don't need auth yet)

**Files Created:** `database.py`, `models.py`, `auth.py`, `db_service.py`

**Files Modified:** [`main.py`](main.py), [`requirements.txt`](requirements.txt)

**Breaking Changes:** None (new endpoints only, existing functionality unchanged)

---

### Phase 3: Frontend Authentication Layer

**Objective:** Add user authentication to frontend with minimal UI changes, enabling user-specific data loading.

**Tasks:**

1. **Create authentication context**
   - File: `semantic-notes-app/src/contexts/AuthContext.jsx` (new file)
   - Purpose: Global authentication state management
   - Contents:
     - State: `user` (username, userId), `token`, `isAuthenticated`, `loading`
     - Methods: `login(username, password)`, `register(username, password)`, `logout()`
     - Persistence: Store token in localStorage, auto-restore on mount
     - Auto-logout on token expiry (JWT decode to check exp)
   - Provider wraps entire app in [`main.jsx`](semantic-notes-app/src/main.jsx)

2. **Create authentication components**
   - File: `semantic-notes-app/src/components/LoginForm.jsx` (new file)
   - Purpose: Simple login/register form
   - Features: Toggle between login/register mode, basic validation, error display
   - Styling: Minimal, consistent with existing app design
   
   - File: `semantic-notes-app/src/components/AuthGuard.jsx` (new file)
   - Purpose: Redirect to login if not authenticated
   - Usage: Wrap main app content in [`App.jsx`](semantic-notes-app/src/App.jsx)

3. **Update API service for authentication**
   - File: [`semantic-notes-app/src/services/api.js`](semantic-notes-app/src/services/api.js)
   - Changes:
     - Add `setAuthToken(token)` method: Stores token for subsequent requests
     - Modify `request()` method: Include `Authorization: Bearer ${token}` header when token present
     - Add `register(username, password)` method: POST to `/api/auth/register`
     - Add `login(username, password)` method: POST to `/api/auth/login`
     - Add `getCurrentUser()` method: GET from `/api/auth/me`
   - Keep existing embedding and graph methods unchanged

4. **Create database API service**
   - File: `semantic-notes-app/src/services/dbApi.js` (new file)
   - Purpose: Dedicated service for database CRUD operations (separation of concerns)
   - Contents:
     - `fetchNotes()`: GET from `/api/notes`
     - `createNote(noteData)`: POST to `/api/notes`
     - `updateNote(noteId, noteData)`: PUT to `/api/notes/${noteId}`
     - `moveToTrash(noteId)`: POST to `/api/notes/${noteId}/trash`
     - `restoreNote(noteId)`: POST to `/api/notes/${noteId}/restore`
     - `permanentDelete(noteId)`: DELETE to `/api/notes/${noteId}`
     - `fetchTrash()`: GET from `/api/trash`
     - `emptyTrash()`: POST to `/api/trash/empty`
   - All methods use `apiService.request()` internally (reuses existing auth headers)

**Files Created:** `semantic-notes-app/src/contexts/AuthContext.jsx`, `semantic-notes-app/src/components/LoginForm.jsx`, `semantic-notes-app/src/components/AuthGuard.jsx`, `semantic-notes-app/src/services/dbApi.js`

**Files Modified:** [`semantic-notes-app/src/services/api.js`](semantic-notes-app/src/services/api.js), [`semantic-notes-app/src/main.jsx`](semantic-notes-app/src/main.jsx)

**UI Changes:** Login screen before main app (minimal design, matches existing style)

---

### Phase 4: Frontend-Database Synchronization

**Objective:** Update notes management to use database while maintaining local cache for performance and offline capability.

**Tasks:**

1. **Update useNotes hook for database integration**
   - File: [`semantic-notes-app/src/hooks/useNotes.js`](semantic-notes-app/src/hooks/useNotes.js)
   - Strategy: Hybrid approach - database as source of truth, localStorage as cache
   - Changes:
     - **Initial load**: Fetch from database on mount if authenticated
     - **Fallback**: Use localStorage if database unavailable (network error)
     - **Sync on change**: Every CRUD operation calls database first, then updates localStorage
     - **Error handling**: Display user-friendly errors, graceful degradation to localStorage
   - Modified functions:
     - `loadFromStorage()`: Add try-catch for network failures
     - `addNote()`: Call `dbApi.createNote()`, await response, then update state
     - `updateNote()`: Call `dbApi.updateNote()`, await response, then update state
     - `deleteNote()`: Permanent delete now calls database
     - `moveToTrash()`: Call `dbApi.moveToTrash()`, update both states
     - `restoreFromTrash()`: Call `dbApi.restoreNote()`, update both states
     - `permanentDelete()`: Call `dbApi.permanentDelete()`, remove from state
     - `emptyTrash()`: Call `dbApi.emptyTrash()`, clear trash state
   - Keep localStorage updates for cache and offline support
   - Add loading states for database operations

2. **Update App.jsx for authentication flow**
   - File: [`semantic-notes-app/src/App.jsx`](semantic-notes-app/src/App.jsx)
   - Changes:
     - Import `AuthContext` and `AuthGuard`
     - Wrap main app content with `AuthGuard`
     - Show `LoginForm` when not authenticated
     - Display username in header when authenticated
     - Add logout button in UI (simple, unobtrusive)
   - Keep all existing functionality unchanged when authenticated

3. **Handle embedding cache synchronization**
   - Approach: Keep localStorage cache for performance, sync to database in background
   - Changes to [`api.js`](semantic-notes-app/src/services/api.js):
     - After computing embeddings, send to database via new endpoint
     - On app load, fetch cached embeddings from database for notes missing in localStorage
     - This provides best of both worlds: fast local cache, reliable database backup
   - New backend endpoint needed: POST `/api/embeddings/batch` (accepts note_id + embedding pairs)

4. **Add loading and error states**
   - Update [`ToastNotification.jsx`](semantic-notes-app/src/components/ToastNotification.jsx): Display database operation errors
   - Add loading spinners for:
     - Initial notes fetch from database
     - Note creation/update/delete operations
     - Authentication operations
   - Graceful fallback messages when database unavailable

**Files Modified:** [`semantic-notes-app/src/hooks/useNotes.js`](semantic-notes-app/src/hooks/useNotes.js), [`semantic-notes-app/src/App.jsx`](semantic-notes-app/src/App.jsx), [`semantic-notes-app/src/services/api.js`](semantic-notes-app/src/services/api.js)

**User Experience:** Seamless transition, existing users see login screen, can register and import their localStorage notes

---

### Phase 5: Embedding Persistence

**Objective:** Store embeddings in database to enable cross-device access while maintaining localStorage performance cache.

**Tasks:**

1. **Add embedding persistence endpoint**
   - File: [`main.py`](main.py)
   - New endpoint: POST `/api/embeddings/batch` (requires JWT)
   - Purpose: Accept array of `{note_id, embedding}` pairs and store in database
   - Implementation:
     - Extract user_id from JWT
     - Verify all note_ids belong to user (security check)
     - For each pair: Check if embedding exists, upsert to database
     - Use `db_service.save_note_embedding()` function
     - Return success status

2. **Update db_service for embedding operations**
   - File: `db_service.py`
   - Add efficient batch operations:
     - `batch_upsert_embeddings(user_id, note_embeddings_list)`: Insert/update multiple embeddings
     - `get_user_embeddings(user_id)`: Fetch all embeddings for user's notes
     - `delete_embedding_by_note_id(note_id)`: Remove when note deleted
   - Use database transactions for batch operations (atomic, rollback on error)

3. **Update frontend embedding cache logic**
   - File: [`semantic-notes-app/src/services/api.js`](semantic-notes-app/src/services/api.js)
   - Changes to `getEmbeddingsForNotes()`:
     - Check localStorage cache first (fast path)
     - If miss, check database via new endpoint: GET `/api/embeddings?note_ids=1,2,3`
     - If still miss, compute via `/api/embed` and save to both caches
     - Background sync: After computing new embeddings, send to database asynchronously
   - Keep localStorage as primary cache for performance
   - Database as backup for cross-device sync

4. **Add embedding database endpoint for fetching**
   - File: [`main.py`](main.py)
   - New endpoint: GET `/api/embeddings` (requires JWT, accepts `note_ids` query param)
   - Purpose: Fetch embeddings for specific notes from database
   - Implementation:
     - Extract user_id from JWT
     - Parse note_ids from query string
     - Verify notes belong to user
     - Return embeddings with note_ids
     - Efficient single query via `batch_get_embeddings()`

**Files Modified:** [`main.py`](main.py), `db_service.py`, [`semantic-notes-app/src/services/api.js`](semantic-notes-app/src/services/api.js)

**Performance Impact:** Minimal - localStorage remains primary cache, database queries only on cache miss

---

### Phase 6: Testing and Validation

**Objective:** Ensure database integration works correctly with existing functionality, validate data isolation and performance.

**Tasks:**

1. **Backend API testing**
   - Test authentication endpoints:
     - Register new user (success case)
     - Register duplicate username (error case)
     - Login with correct credentials
     - Login with incorrect credentials
     - Access protected endpoint with valid token
     - Access protected endpoint with invalid token
   - Test notes CRUD endpoints:
     - Create note for user A
     - Fetch notes for user A (verify only user A's notes returned)
     - Update note for user A
     - Attempt to access user A's note as user B (verify rejection)
     - Soft delete note (verify is_deleted flag)
     - Restore note from trash
     - Permanent delete note
   - Test embedding endpoints:
     - Save embeddings for notes
     - Fetch embeddings for specific note_ids
     - Verify cache invalidation when note content changes

2. **Frontend integration testing**
   - Test authentication flow:
     - Register → login → access app
     - Logout → verify redirect to login
     - Token expiry → auto-logout
     - Token persistence across page refreshes
   - Test notes operations:
     - Create note → verify saved to database
     - Edit note → verify updated in database
     - Delete note → verify moved to trash
     - Restore note → verify restored from trash
     - Empty trash → verify permanent deletion
   - Test localStorage/database sync:
     - Create note offline (localStorage only)
     - Come online → verify sync to database
     - Load notes on new device → verify fetched from database

3. **User isolation validation**
   - Create two test users
   - Create notes for each user
   - Verify user A cannot see/modify user B's notes
   - Verify embedding isolation
   - Verify trash isolation

4. **Performance validation**
   - Measure time to load 100 notes from database
   - Measure time to load 100 notes from localStorage (cache hit)
   - Compare graph generation time (before/after database integration)
   - Verify no performance degradation for local operations

5. **Error handling validation**
   - Test behavior when database unavailable
   - Test behavior when network interrupted
   - Verify graceful degradation to localStorage
   - Verify user-friendly error messages displayed

**Success Criteria:**
- All tests pass
- User isolation working correctly
- No significant performance degradation
- Graceful fallback when database unavailable

---

### Phase 7: Migration Support (Optional)

**Objective:** Provide users with tool to import their existing localStorage notes into database after registering.

**Tasks:**

1. **Create import API endpoint**
   - File: [`main.py`](main.py)
   - New endpoint: POST `/api/notes/import` (requires JWT)
   - Purpose: Bulk import notes from localStorage to database
   - Implementation:
     - Accept array of note objects
     - Validate each note structure
     - Assign to current user
     - Use `db_service.bulk_create_notes()` for efficiency
     - Return count of imported notes

2. **Update db_service for bulk import**
   - File: `db_service.py`
   - Add `bulk_create_notes(user_id, notes_list)`:
     - Single transaction for all notes
     - Generate new IDs (don't trust localStorage IDs)
     - Preserve created_at and updated_at timestamps
     - Return mapping of old_id → new_id for embedding migration

3. **Create frontend import UI**
   - File: `semantic-notes-app/src/components/ImportLocalNotesModal.jsx` (new file)
   - Purpose: One-time import wizard for new users
   - Features:
     - Detect localStorage notes on first login
     - Show count of notes to import
     - Explain import process
     - Confirm button → trigger import
     - Progress indicator during import
     - Success message with imported count
   - Auto-show on first login if localStorage notes exist

4. **Handle embedding migration**
   - During import, also migrate embeddings:
     - Read embeddings from localStorage cache
     - Map to new database note IDs
     - Batch save to database
     - Keep localStorage cache intact for performance

**Files Created:** `semantic-notes-app/src/components/ImportLocalNotesModal.jsx`

**Files Modified:** [`main.py`](main.py), `db_service.py`, [`semantic-notes-app/src/App.jsx`](semantic-notes-app/src/App.jsx)

**User Experience:** Seamless transition from localStorage to database, no data loss

---

## File Structure Summary

### New Files (Backend)
```
project_root/
├── docker-compose.yml              # PostgreSQL Docker container config
├── init-db.sql                     # Database schema initialization
├── .env                            # Environment variables (not committed)
├── .env.example                    # Environment variables template
├── database.py                     # Database connection and session management
├── models.py                       # SQLAlchemy ORM models
├── auth.py                         # Authentication utilities (JWT, password hashing)
└── db_service.py                   # Database CRUD operations
```

### New Files (Frontend)
```
semantic-notes-app/src/
├── contexts/
│   └── AuthContext.jsx             # Global authentication state
├── components/
│   ├── LoginForm.jsx               # Login/register form
│   ├── AuthGuard.jsx               # Protected route wrapper
│   └── ImportLocalNotesModal.jsx  # One-time localStorage import
└── services/
    └── dbApi.js                    # Database API calls
```

### Modified Files
```
Backend:
├── main.py                         # Add auth and notes endpoints
└── requirements.txt                # Add database dependencies

Frontend:
├── src/main.jsx                    # Wrap app with AuthProvider
├── src/App.jsx                     # Add AuthGuard and logout button
├── src/hooks/useNotes.js           # Add database sync to CRUD operations
└── src/services/api.js             # Add auth headers and token management
```

---

## Dependencies Summary

### Backend (Add to requirements.txt)
```
sqlalchemy                          # ORM for database models
psycopg2-binary                     # PostgreSQL adapter for Python
python-jose[cryptography]           # JWT token generation/validation
passlib[bcrypt]                     # Password hashing
python-multipart                    # Form data parsing for FastAPI
```

**Total new dependencies: 5**
**Rationale:** These are standard, lightweight libraries for database and auth functionality. All are well-maintained and widely used in production.

### Frontend
No new npm dependencies required. All functionality implemented with existing packages:
- `axios` (already present) - HTTP requests
- `react` (already present) - UI components and hooks

---

## API Endpoints Reference

### Authentication Endpoints (New)
```
POST   /api/auth/register           # Create new user
POST   /api/auth/login              # Authenticate user, get JWT
GET    /api/auth/me                 # Get current user info (protected)
```

### Notes Endpoints (New)
```
GET    /api/notes                   # Get all user notes (protected)
POST   /api/notes                   # Create note (protected)
PUT    /api/notes/{id}              # Update note (protected)
POST   /api/notes/{id}/trash        # Move to trash (protected)
POST   /api/notes/{id}/restore      # Restore from trash (protected)
DELETE /api/notes/{id}              # Permanent delete (protected)
GET    /api/trash                   # Get trashed notes (protected)
POST   /api/trash/empty             # Empty trash (protected)
POST   /api/notes/import            # Bulk import notes (protected)
```

### Embedding Endpoints (New)
```
GET    /api/embeddings              # Get embeddings by note IDs (protected)
POST   /api/embeddings/batch        # Save embeddings (protected)
```

### Existing Endpoints (Unchanged)
```
GET    /api/health                  # Health check
GET    /api/stats                   # Service statistics
POST   /api/embed                   # Generate embeddings
POST   /api/graph                   # Build semantic graph
```

---

## Data Flow Diagrams

### Current Flow (Before Integration)
```
User → React App → localStorage
                 ↓
            FastAPI → Embeddings → Graph
```

### New Flow (After Integration)
```
User → LoginForm → AuthContext → JWT Token
                                    ↓
     → React App → useNotes Hook → Database (source of truth)
                                 ↘ localStorage (cache)
                 ↓
            FastAPI → Embeddings → Database (persistence)
                                 ↘ localStorage (cache)
                    ↓
                   Graph
```

### Authentication Flow
```mermaid
graph LR
    A[User] -->|username/password| B[LoginForm]
    B -->|POST /api/auth/login| C[FastAPI Auth]
    C -->|verify| D[Database]
    D -->|user found| C
    C -->|generate| E[JWT Token]
    E -->|store| F[localStorage]
    F -->|attach to requests| G[API Calls]
    G -->|verify token| C
    C -->|extract user_id| H[Protected Endpoints]
```

### Note Synchronization Flow
```mermaid
graph TD
    A[User Action] -->|create/edit note| B[useNotes Hook]
    B -->|call| C[dbApi.createNote]
    C -->|POST with JWT| D[FastAPI /api/notes]
    D -->|verify user| E[Database Insert]
    E -->|success| D
    D -->|response| C
    C -->|note with db_id| B
    B -->|update| F[React State]
    B -->|cache| G[localStorage]
```

---

## Migration Strategy

### For Existing Users

**First Time Login Flow:**
1. User visits app → sees LoginForm
2. User registers new account
3. System detects localStorage notes
4. ImportLocalNotesModal appears automatically
5. User confirms import
6. Notes migrated to database with preserved timestamps
7. User continues with database-backed experience
8. localStorage cache remains for performance

**Backward Compatibility:**
- If database unavailable, app falls back to localStorage-only mode
- No breaking changes to existing functionality
- Graph preferences remain in localStorage (minimal change)
- All UI components unchanged except auth layer

---

## Performance Considerations

### Database Query Optimization
- Indexes on `user_id` for fast user-specific queries
- Indexes on `is_deleted` for trash filtering
- Batch operations for bulk inserts/updates
- Connection pooling via SQLAlchemy

### Caching Strategy
- **Notes**: Database is source of truth, localStorage caches after fetch
- **Embeddings**: localStorage primary cache, database backup
- **Graph preferences**: localStorage only (UI state)
- Result: Minimal latency increase, most operations still local

### Expected Performance Impact
- First load after login: +200-500ms (database fetch)
- Subsequent loads: ~0ms (localStorage cache hit)
- Note creation: +100-200ms (database insert)
- Note updates: +100-200ms (database update)
- Graph generation: No impact (uses cached embeddings)

---

## Security Considerations

**Implemented (Simple Security):**
- Password hashing with bcrypt
- JWT tokens with expiry (7 days)
- User data isolation via foreign keys
- SQL injection prevention via SQLAlchemy ORM
- Protected API endpoints require valid JWT

**Not Implemented (As Per Requirements):**
- No password strength requirements
- No email verification
- No password reset flow
- No rate limiting
- No account lockout
- No 2FA/MFA
- No HTTPS enforcement (local development)
- No CORS restrictions (development mode)

---

## Environment Variables

### Backend (.env)
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/semantic_notes
POSTGRES_USER=semantic_user
POSTGRES_PASSWORD=secure_password_here
POSTGRES_DB=semantic_notes

# JWT
JWT_SECRET_KEY=your_secret_key_here_min_32_chars
JWT_ALGORITHM=HS256
JWT_EXPIRE_DAYS=7

# Embedding Service (existing)
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=64
```

### Docker Compose
```yaml
# PostgreSQL configuration (from .env)
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=${POSTGRES_DB}
```

---

## Testing Checklist

### Phase 1: Docker Setup
- [ ] PostgreSQL container starts successfully
- [ ] Database initialized with correct schema
- [ ] Tables and indexes created
- [ ] Connection from host machine works

### Phase 2: Backend Integration
- [ ] Dependencies install without conflicts
- [ ] Database connection established
- [ ] Models reflect database schema accurately
- [ ] Auth endpoints return expected responses
- [ ] Notes CRUD endpoints work correctly
- [ ] User isolation enforced

### Phase 3: Frontend Authentication
- [ ] Login form renders correctly
- [ ] Registration creates user and returns JWT
- [ ] Login authenticates and returns JWT
- [ ] Token stored in localStorage
- [ ] Token included in API requests
- [ ] AuthGuard redirects when not authenticated

### Phase 4: Frontend-Database Sync
- [ ] Notes fetched from database on load
- [ ] Note creation saves to database
- [ ] Note updates persist to database
- [ ] Trash operations sync to database
- [ ] localStorage cache updated alongside database
- [ ] Offline mode falls back gracefully

### Phase 5: Embedding Persistence
- [ ] Embeddings saved to database after computation
- [ ] Embeddings fetched from database on cache miss
- [ ] localStorage cache hits avoid database queries
- [ ] Cross-device embedding sync works

### Phase 6: Integration Testing
- [ ] Multiple users can use app simultaneously
- [ ] User A cannot access user B's data
- [ ] Performance within acceptable ranges
- [ ] Error messages user-friendly
- [ ] No console errors or warnings

### Phase 7: Migration Support
- [ ] Import modal detects localStorage notes
- [ ] Bulk import completes successfully
- [ ] Timestamps preserved during import
- [ ] Embeddings migrated correctly
- [ ] No data loss during import

---

## Success Metrics

- ✅ PostgreSQL running in Docker
- ✅ All database tables created correctly
- ✅ User registration and login working
- ✅ Notes CRUD operations through database
- ✅ User data isolation verified
- ✅ Embeddings persisted and retrievable
- ✅ Performance within acceptable ranges (<500ms for most operations)
- ✅ No breaking changes to existing functionality
- ✅ Graceful degradation when database unavailable
- ✅ All tests passing
- ✅ Zero data loss during migration
- 
---

## Estimated Implementation Scope

### Code Additions
- **Backend**: ~800-1000 lines (database.py, models.py, auth.py, db_service.py, main.py updates)
- **Frontend**: ~600-800 lines (AuthContext, LoginForm, AuthGuard, dbApi.js, useNotes updates)
- **Configuration**: ~100 lines (docker-compose.yml, init-db.sql, .env.example)

### Files Created: 14 new files
### Files Modified: 6 existing files
### Dependencies Added: 5 backend packages, 0 frontend packages

**Total Project Growth**: ~1500-1900 lines of code across 20 files

This represents a lightweight integration maintaining the principle of minimal codebase impact while enabling multi-user database persistence, authentication, and data isolation.