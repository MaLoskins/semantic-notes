# Phase 6: Testing and Validation Guide
## PostgreSQL Integration - Semantic Notes Application

**Version:** 1.0  
**Date:** 2025-10-22  
**Project Root:** `c:/Users/matth/Desktop/8-SANDBOX/notes/semantic-notes`

---

## Table of Contents

1. [Pre-Testing Setup Checklist](#1-pre-testing-setup-checklist)
2. [Backend API Test Cases](#2-backend-api-test-cases)
3. [Frontend Integration Test Scenarios](#3-frontend-integration-test-scenarios)
4. [User Isolation Test Plan](#4-user-isolation-test-plan)
5. [Performance Benchmarks](#5-performance-benchmarks)
6. [Error Handling Scenarios](#6-error-handling-scenarios)
7. [Test Execution Commands](#7-test-execution-commands)
8. [Validation Checklist](#8-validation-checklist)

---

## 1. Pre-Testing Setup Checklist

### 1.1 Environment Prerequisites

**Check installed tools:**
```bash
# Check Docker
docker --version
docker-compose --version

# Check Python
python --version  # Should be 3.10+

# Check Node.js
node --version    # Should be 16+
npm --version
```

### 1.2 Environment Configuration

**Create .env file from template:**
```bash
cp .env.example .env
```

**Edit .env and set required variables:**
```env
# Database Configuration
POSTGRES_USER=semantic_user
POSTGRES_PASSWORD=SecurePassword123!
POSTGRES_DB=semantic_notes
DATABASE_URL=postgresql://semantic_user:SecurePassword123!@localhost:5432/semantic_notes

# JWT Configuration
JWT_SECRET_KEY=your_very_long_and_secure_secret_key_here_min_32_characters_long
JWT_ALGORITHM=HS256
JWT_EXPIRE_DAYS=7

# Embedding Service
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=64
```

**Verify .gitignore includes:**
```
.env
postgres-data/
__pycache__/
*.pyc
node_modules/
```

### 1.3 Database Setup

**Start PostgreSQL container:**
```bash
# In WSL terminal (Windows)
wsl
cd /mnt/c/Users/matth/Desktop/8-SANDBOX/notes/semantic-notes
docker-compose up -d
```

**Verify container is running:**
```bash
docker ps | grep semantic-notes-db
```

**Expected output:**
```
CONTAINER ID   IMAGE               COMMAND                  STATUS          PORTS                    NAMES
a1b2c3d4e5f6   postgres:15-alpine  "docker-entrypoint.s…"   Up 10 seconds   0.0.0.0:5432->5432/tcp   semantic-notes-db
```

**Check container logs:**
```bash
docker-compose logs postgres
```

**Look for:**
```
database system is ready to accept connections
```

**Verify database tables created:**
```bash
docker exec -it semantic-notes-db psql -U semantic_user -d semantic_notes -c "\dt"
```

**Expected output:**
```
             List of relations
 Schema |    Name    | Type  |     Owner      
--------+------------+-------+----------------
 public | embeddings | table | semantic_user
 public | notes      | table | semantic_user
 public | users      | table | semantic_user
(3 rows)
```

**Verify indexes:**
```bash
docker exec -it semantic-notes-db psql -U semantic_user -d semantic_notes -c "\di"
```

### 1.4 Backend Setup

**Install Python dependencies:**
```bash
# In WSL terminal
pip install -r requirements.txt
```

**Verify imports work:**
```bash
python -c "import fastapi, sqlalchemy, jose, passlib; print('All imports successful')"
```

**Start FastAPI backend:**
```bash
# In WSL terminal
python main.py
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Verify health endpoint (open new terminal):**
```bash
curl http://localhost:8000/api/health
```

**Expected response:**
```json
{"status":"ok"}
```

### 1.5 Frontend Setup

**Install frontend dependencies:**
```bash
# In new WSL terminal
cd frontend
npm install
```

**Start React development server:**
```bash
npm run dev
```

**Expected output:**
```
  VITE v4.x.x  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### 1.6 Verification Checklist

- [ ] Docker container running
- [ ] Database accessible on port 5432
- [ ] All tables created (users, notes, embeddings)
- [ ] All indexes created
- [ ] Python dependencies installed
- [ ] Backend running on port 8000
- [ ] Health endpoint responds
- [ ] Frontend dependencies installed
- [ ] Frontend running on port 5173
- [ ] No error messages in any terminal

---

## 2. Backend API Test Cases

### Testing Tool Options

**Option 1: curl (Command line)**
```bash
# All examples use curl
```

**Option 2: Postman/Insomnia (GUI)**
- Import endpoints manually
- Set Authorization header: `Bearer <token>`

**Option 3: Python test script**
```bash
python test_client.py
```

### 2.1 Authentication Endpoints

#### Test 2.1.1: POST /api/auth/register (Success)

**Request:**
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser1",
    "password": "password123",
    "email": "testuser1@example.com"
  }'
```

**Expected Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "username": "testuser1",
  "user_id": 1
}
```

**Save token for subsequent tests:**
```bash
export TOKEN1="<access_token_from_response>"
```

#### Test 2.1.2: POST /api/auth/register (Duplicate Username)

**Request:**
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser1",
    "password": "differentpass",
    "email": "another@example.com"
  }'
```

**Expected Response (400 Bad Request):**
```json
{
  "detail": "Username already exists"
}
```

#### Test 2.1.3: POST /api/auth/login (Success)

**Request:**
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser1",
    "password": "password123"
  }'
```

**Expected Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "username": "testuser1",
  "user_id": 1
}
```

#### Test 2.1.4: POST /api/auth/login (Invalid Credentials)

**Request:**
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser1",
    "password": "wrongpassword"
  }'
```

**Expected Response (401 Unauthorized):**
```json
{
  "detail": "Invalid credentials"
}
```

#### Test 2.1.5: GET /api/auth/me (Valid Token)

**Request:**
```bash
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response (200 OK):**
```json
{
  "id": 1,
  "username": "testuser1",
  "email": "testuser1@example.com"
}
```

#### Test 2.1.6: GET /api/auth/me (Invalid Token)

**Request:**
```bash
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer invalid_token_here"
```

**Expected Response (401 Unauthorized):**
```json
{
  "detail": "Invalid or expired token"
}
```

#### Test 2.1.7: GET /api/auth/me (No Token)

**Request:**
```bash
curl http://localhost:8000/api/auth/me
```

**Expected Response (403 Forbidden):**
```json
{
  "detail": "Not authenticated"
}
```

### 2.2 Notes CRUD Endpoints

#### Test 2.2.1: POST /api/notes (Create Note)

**Request:**
```bash
curl -X POST http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My First Note",
    "content": "This is the content of my first note.",
    "tags": "test,sample"
  }'
```

**Expected Response (200 OK):**
```json
{
  "id": 1,
  "title": "My First Note",
  "content": "This is the content of my first note.",
  "tags": "test,sample",
  "created_at": "2025-10-22T04:30:00.000000",
  "updated_at": "2025-10-22T04:30:00.000000",
  "is_deleted": false
}
```

**Save note ID:**
```bash
export NOTE_ID1=1
```

#### Test 2.2.2: GET /api/notes (List Notes)

**Request:**
```bash
curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response (200 OK):**
```json
[
  {
    "id": 1,
    "title": "My First Note",
    "content": "This is the content of my first note.",
    "tags": "test,sample",
    "created_at": "2025-10-22T04:30:00.000000",
    "updated_at": "2025-10-22T04:30:00.000000",
    "is_deleted": false
  }
]
```

#### Test 2.2.3: PUT /api/notes/{id} (Update Note)

**Request:**
```bash
curl -X PUT http://localhost:8000/api/notes/$NOTE_ID1 \
  -H "Authorization: Bearer $TOKEN1" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Updated Note Title",
    "content": "This content has been updated.",
    "tags": "updated,modified"
  }'
```

**Expected Response (200 OK):**
```json
{
  "id": 1,
  "title": "Updated Note Title",
  "content": "This content has been updated.",
  "tags": "updated,modified",
  "created_at": "2025-10-22T04:30:00.000000",
  "updated_at": "2025-10-22T04:35:00.000000",
  "is_deleted": false
}
```

**Verify updated_at changed**

#### Test 2.2.4: PUT /api/notes/{id} (Note Not Found)

**Request:**
```bash
curl -X PUT http://localhost:8000/api/notes/99999 \
  -H "Authorization: Bearer $TOKEN1" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Non-existent",
    "content": "This should fail",
    "tags": ""
  }'
```

**Expected Response (404 Not Found):**
```json
{
  "detail": "Note not found"
}
```

### 2.3 Trash Operations

#### Test 2.3.1: POST /api/notes/{id}/trash (Move to Trash)

**Request:**
```bash
curl -X POST http://localhost:8000/api/notes/$NOTE_ID1/trash \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response (200 OK):**
```json
{
  "status": "moved to trash"
}
```

#### Test 2.3.2: GET /api/notes (Verify Not in Active Notes)

**Request:**
```bash
curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response (200 OK):**
```json
[]
```

#### Test 2.3.3: GET /api/trash (View Trashed Notes)

**Request:**
```bash
curl http://localhost:8000/api/trash \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response (200 OK):**
```json
[
  {
    "id": 1,
    "title": "Updated Note Title",
    "content": "This content has been updated.",
    "tags": "updated,modified",
    "created_at": "2025-10-22T04:30:00.000000",
    "updated_at": "2025-10-22T04:35:00.000000",
    "is_deleted": true
  }
]
```

#### Test 2.3.4: POST /api/notes/{id}/restore (Restore from Trash)

**Request:**
```bash
curl -X POST http://localhost:8000/api/notes/$NOTE_ID1/restore \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response (200 OK):**
```json
{
  "status": "restored"
}
```

#### Test 2.3.5: GET /api/notes (Verify Back in Active Notes)

**Request:**
```bash
curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response:** Note should appear in list with `is_deleted: false`

#### Test 2.3.6: DELETE /api/notes/{id} (Permanent Delete)

**Request:**
```bash
curl -X DELETE http://localhost:8000/api/notes/$NOTE_ID1 \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response (200 OK):**
```json
{
  "status": "permanently deleted"
}
```

#### Test 2.3.7: GET /api/notes (Verify Completely Gone)

**Request:**
```bash
curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response:** Empty array `[]`

#### Test 2.3.8: POST /api/trash/empty (Empty Trash)

**Setup: Create and trash multiple notes first**
```bash
# Create 3 notes
for i in {1..3}; do
  curl -X POST http://localhost:8000/api/notes \
    -H "Authorization: Bearer $TOKEN1" \
    -H "Content-Type: application/json" \
    -d "{\"title\":\"Note $i\",\"content\":\"Content $i\",\"tags\":\"\"}"
done

# Move all to trash (use actual IDs)
for id in {2..4}; do
  curl -X POST http://localhost:8000/api/notes/$id/trash \
    -H "Authorization: Bearer $TOKEN1"
done
```

**Request:**
```bash
curl -X POST http://localhost:8000/api/trash/empty \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response (200 OK):**
```json
{
  "deleted_count": 3
}
```

### 2.4 Embedding Endpoints

#### Test 2.4.1: POST /api/embeddings/batch (Save Embeddings)

**Setup: Create a note first**
```bash
NOTE_RESPONSE=$(curl -X POST http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Note for Embeddings",
    "content": "This note will have embeddings saved.",
    "tags": "embeddings,test"
  }')

# Extract note ID (requires jq or manual parsing)
export NOTE_ID_EMB=$(echo $NOTE_RESPONSE | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
```

**Request:**
```bash
curl -X POST http://localhost:8000/api/embeddings/batch \
  -H "Authorization: Bearer $TOKEN1" \
  -H "Content-Type: application/json" \
  -d "{
    \"embeddings\": [
      {
        \"note_id\": $NOTE_ID_EMB,
        \"content_hash\": \"abc123def456\",
        \"embedding\": [0.1, 0.2, 0.3, 0.4],
        \"model_name\": \"sentence-transformers/all-MiniLM-L6-v2\"
      }
    ]
  }"
```

**Expected Response (200 OK):**
```json
{
  "saved": 1,
  "total": 1
}
```

#### Test 2.4.2: GET /api/embeddings (Fetch Embeddings)

**Request:**
```bash
curl "http://localhost:8000/api/embeddings?note_ids=$NOTE_ID_EMB" \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response (200 OK):**
```json
{
  "embeddings": {
    "5": {
      "content_hash": "abc123def456",
      "embedding": [0.1, 0.2, 0.3, 0.4],
      "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    }
  }
}
```

#### Test 2.4.3: GET /api/embeddings (Multiple Note IDs)

**Request:**
```bash
curl "http://localhost:8000/api/embeddings?note_ids=1,2,3" \
  -H "Authorization: Bearer $TOKEN1"
```

**Expected Response:** Dictionary with embeddings for each note that exists

#### Test 2.4.4: POST /api/embeddings/batch (Unauthorized Access)

**Attempt to save embedding for another user's note**

**Setup: Create second user and note**
```bash
# Register second user
USER2_RESPONSE=$(curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser2",
    "password": "password456",
    "email": "testuser2@example.com"
  }')

export TOKEN2=$(echo $USER2_RESPONSE | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

# Create note as user2
NOTE2_RESPONSE=$(curl -X POST http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN2" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "User2 Note",
    "content": "This belongs to user2",
    "tags": ""
  }')

export NOTE_ID2=$(echo $NOTE2_RESPONSE | grep -o '"id":[0-9]*' | grep -o '[0-9]*')
```

**Request (user1 trying to save embedding for user2's note):**
```bash
curl -X POST http://localhost:8000/api/embeddings/batch \
  -H "Authorization: Bearer $TOKEN1" \
  -H "Content-Type: application/json" \
  -d "{
    \"embeddings\": [
      {
        \"note_id\": $NOTE_ID2,
        \"content_hash\": \"malicious\",
        \"embedding\": [0.1, 0.2],
        \"model_name\": \"test\"
      }
    ]
  }"
```

**Expected Response (403 Forbidden):**
```json
{
  "detail": "Note 6 not found or access denied"
}
```

---

## 3. Frontend Integration Test Scenarios

### 3.1 Authentication Flow

#### Test 3.1.1: Registration Flow

**Steps:**
1. Open browser to `http://localhost:5173`
2. Should see login form
3. Click "Register" or switch to register mode
4. Fill in:
   - Username: `frontenduser1`
   - Password: `testpass123`
   - Email: `frontend@test.com`
5. Click "Register" button

**Expected:**
- No error messages
- Redirect to main application
- See username displayed in header/navbar
- Token saved in localStorage
- Can create notes immediately

**Verify localStorage:**
```javascript
// In browser console
localStorage.getItem('auth_token')
// Should return JWT token
```

#### Test 3.1.2: Login Flow

**Steps:**
1. If logged in, logout first
2. Fill in login form:
   - Username: `frontenduser1`
   - Password: `testpass123`
3. Click "Login" button

**Expected:**
- Successful authentication
- Redirect to main app
- Previous notes loaded
- No error messages

#### Test 3.1.3: Token Persistence

**Steps:**
1. Login successfully
2. Refresh page (F5 or Ctrl+R)

**Expected:**
- Remain logged in
- Notes still visible
- No redirect to login
- Token still in localStorage

#### Test 3.1.4: Logout Flow

**Steps:**
1. Click logout button
2. Observe behavior

**Expected:**
- Redirect to login form
- Token removed from localStorage
- Cannot access protected routes
- Trying to navigate to main app redirects to login

#### Test 3.1.5: Token Expiry

**Manual Test (requires token manipulation):**
1. Login successfully
2. Open browser DevTools → Application → Local Storage
3. Modify token or wait 7 days
4. Try to perform action (create note)

**Expected:**
- Auto-logout
- Redirect to login
- Error message about expired session

### 3.2 Notes CRUD Operations

#### Test 3.2.1: Create Note

**Steps:**
1. Login as user
2. Click "New Note" or similar button
3. Fill in:
   - Title: "Frontend Test Note"
   - Content: "This is created from the frontend"
   - Tags: "frontend,test"
4. Save note

**Expected:**
- Note appears in notes list immediately
- Note has ID assigned by backend
- Timestamps present (created_at, updated_at)
- No errors in console

**Verify backend:**
```bash
curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer <token_from_localStorage>"
```

#### Test 3.2.2: Edit Note

**Steps:**
1. Click on existing note to edit
2. Modify:
   - Title: "Updated Frontend Note"
   - Content: "Content has been modified"
3. Save changes

**Expected:**
- Changes reflected immediately in UI
- updated_at timestamp changed
- Changes persisted to database
- Can refresh page and see updated content

#### Test 3.2.3: Multiple Notes List

**Steps:**
1. Create 5-10 notes with different titles
2. Observe notes list

**Expected:**
- All notes displayed
- Sorted by created date (newest first)
- Each note shows title preview
- Click any note to view/edit

### 3.3 Trash and Restore Operations

#### Test 3.3.1: Move to Trash

**Steps:**
1. Select a note
2. Click "Delete" or "Move to Trash" button
3. Confirm action if prompted

**Expected:**
- Note removed from active notes list
- Note appears in trash view
- can_deleted flag set to true
- Deleted timestamp recorded

#### Test 3.3.2: View Trash

**Steps:**
1. Navigate to Trash view
2. Observe trashed notes

**Expected:**
- All deleted notes visible
- Shows deletion date
- Option to restore or permanently delete
- Separate from active notes

#### Test 3.3.3: Restore from Trash

**Steps:**
1. In trash view, select a note
2. Click "Restore" button

**Expected:**
- Note removed from trash
- Note appears back in active notes list
- is_deleted flag set to false
- Original timestamps preserved

#### Test 3.3.4: Permanent Delete

**Steps:**
1. In trash view, select a note
2. Click "Permanently Delete" button
3. Confirm action

**Expected:**
- Note removed from trash
- Note completely deleted from database
- Cannot be recovered
- Embedding also deleted (cascade)

#### Test 3.3.5: Empty Trash

**Steps:**
1. Move multiple notes to trash
2. Click "Empty Trash" button
3. Confirm action

**Expected:**
- All trash notes permanently deleted
- Trash view empty
- Count shows 0 items
- All associated embeddings deleted

### 3.4 Cross-Device Sync

#### Test 3.4.1: Multi-Device Login

**Steps:**
1. Login on Device A (or Browser 1)
2. Create note "Device A Note"
3. Logout from Device A
4. Login on Device B (or Browser 2 in incognito mode)
5. Check notes list

**Expected:**
- Note "Device A Note" visible on Device B
- All data synced from database
- localStorage cache populated on Device B

#### Test 3.4.2: Concurrent Edits

**Steps:**
1. Login same user on two browsers
2. Browser A: Create note "Concurrent Test"
3. Browser B: Refresh to see note
4. Browser A: Edit content to "Version A"
5. Browser B: Edit content to "Version B" (without refreshing)
6. Both browsers save

**Expected:**
- Last write wins (Browser B's version)
- No data corruption
- Both browsers show same final version after refresh

### 3.5 Offline Mode Fallback

#### Test 3.5.1: Offline Note Creation

**Steps:**
1. Login successfully
2. Open DevTools → Network tab → Set to "Offline"
3. Try to create a new note

**Expected:**
- Note saved to localStorage
- User sees note in list
- Warning/notification about offline mode
- Note synced when connection restored

**Verify:**
```javascript
// In console while offline
localStorage.getItem('semantic-notes-data')
// Should contain the new note
```

#### Test 3.5.2: Database Unavailable

**Steps:**
1. Stop backend server: `Ctrl+C` in backend terminal
2. Try to perform operations

**Expected:**
- Graceful degradation
- User-friendly error messages
- localStorage cache still works
- Can view existing notes
- Clear indication of connection issue

### 3.6 Graph Visualization with Database Notes

#### Test 3.6.1: Generate Graph

**Steps:**
1. Create 5+ notes with related content
2. Navigate to graph view
3. Click "Generate Graph" or similar

**Expected:**
- Notes fetched from database
- Embeddings computed (or loaded from cache)
- Graph rendered with nodes and edges
- Similar notes connected
- No errors in console

#### Test 3.6.2: Embedding Cache Performance

**Steps:**
1. Generate graph for first time (measures embedding computation)
2. Note the time taken
3. Close and reopen application
4. Generate graph again (should use cache)

**Expected:**
- Second generation significantly faster
- Embeddings loaded from localStorage or database
- No re-computation for unchanged notes
- Console shows cache hits

---

## 4. User Isolation Test Plan

### 4.1 Setup Test Users

**Create two test users:**
```bash
# User A
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "userA",
    "password": "passA123",
    "email": "usera@test.com"
  }'

export TOKEN_A="<token_from_response>"

# User B
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "userB",
    "password": "passB456",
    "email": "userb@test.com"
  }'

export TOKEN_B="<token_from_response>"
```

### 4.2 Test Scenarios

#### Test 4.2.1: Notes Isolation

**Create notes for each user:**
```bash
# User A creates note
curl -X POST http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN_A" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "User A Private Note",
    "content": "This belongs to User A only",
    "tags": "private"
  }'

export NOTE_ID_A=<id_from_response>

# User B creates note
curl -X POST http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN_B" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "User B Private Note",
    "content": "This belongs to User B only",
    "tags": "secret"
  }'

export NOTE_ID_B=<id_from_response>
```

**Test: User A lists notes**
```bash
curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN_A"
```

**Expected:** Only User A's notes returned

**Test: User B lists notes**
```bash
curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN_B"
```

**Expected:** Only User B's notes returned

#### Test 4.2.2: Unauthorized Note Access

**User A tries to read User B's note:**
```bash
curl http://localhost:8000/api/notes/$NOTE_ID_B \
  -H "Authorization: Bearer $TOKEN_A"
```

**Expected Response (403 or 404):**
```json
{
  "detail": "Note not found"
}
```

**User A tries to update User B's note:**
```bash
curl -X PUT http://localhost:8000/api/notes/$NOTE_ID_B \
  -H "Authorization: Bearer $TOKEN_A" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Hacked",
    "content": "Attempting unauthorized access",
    "tags": ""
  }'
```

**Expected Response (404):**
```json
{
  "detail": "Note not found"
}
```

**User A tries to delete User B's note:**
```bash
curl -X DELETE http://localhost:8000/api/notes/$NOTE_ID_B \
  -H "Authorization: Bearer $TOKEN_A"
```

**Expected Response (404):**
```json
{
  "detail": "Note not found"
}
```

#### Test 4.2.3: Embeddings Isolation

**User A saves embedding:**
```bash
curl -X POST http://localhost:8000/api/embeddings/batch \
  -H "Authorization: Bearer $TOKEN_A" \
  -H "Content-Type: application/json" \
  -d "{
    \"embeddings\": [{
      \"note_id\": $NOTE_ID_A,
      \"content_hash\": \"hashA\",
      \"embedding\": [0.1, 0.2, 0.3],
      \"model_name\": \"test\"
    }]
  }"
```

**User B tries to access User A's embedding:**
```bash
curl "http://localhost:8000/api/embeddings?note_ids=$NOTE_ID_A" \
  -H "Authorization: Bearer $TOKEN_B"
```

**Expected Response (403):**
```json
{
  "detail": "Note <id> not found or access denied"
}
```

#### Test 4.2.4: Trash Isolation

**User A moves note to trash:**
```bash
curl -X POST http://localhost:8000/api/notes/$NOTE_ID_A/trash \
  -H "Authorization: Bearer $TOKEN_A"
```

**User B views their trash:**
```bash
curl http://localhost:8000/api/trash \
  -H "Authorization: Bearer $TOKEN_B"
```

**Expected:** User A's note NOT in User B's trash (only User B's deleted notes)

**User B tries to restore User A's trashed note:**
```bash
curl -X POST http://localhost:8000/api/notes/$NOTE_ID_A/restore \
  -H "Authorization: Bearer $TOKEN_B"
```

**Expected Response (404):**
```json
{
  "detail": "Note not found or not deleted"
}
```

### 4.3 Database Verification

**Connect to database and verify isolation:**
```bash
docker exec -it semantic-notes-db psql -U semantic_user -d semantic_notes
```

**Query to check user separation:**
```sql
-- Should show notes only for respective users
SELECT u.username, COUNT(n.id) as note_count
FROM users u
LEFT JOIN notes n ON u.id = n.user_id
GROUP BY u.username;

-- Verify no cross-user access
SELECT n.id, n.title, u.username as owner
FROM notes n
JOIN users u ON n.user_id = u.id;

-- Check embeddings isolation
SELECT e.id, e.note_id, n.user_id, u.username
FROM embeddings e
JOIN notes n ON e.note_id = n.id
JOIN users u ON n.user_id = u.id;
```

---

## 5. Performance Benchmarks

### 5.1 Notes Loading Performance

#### Benchmark 5.1.1: Load 10 Notes from Database

**Setup:**
```bash
# Create 10 notes for test user
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/notes \
    -H "Authorization: Bearer $TOKEN1" \
    -H "Content-Type: application/json" \
    -d "{\"title\":\"Note $i\",\"content\":\"Content for note number $i\",\"tags\":\"test\"}"
done
```

**Measure:**
```bash
time curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1"
```

**Target:** < 200ms

**Record Results:**
```
| Test | Time (ms) | Status |
|------|-----------|--------|
| Run 1|           |        |
| Run 2|           |        |
| Run 3|           |        |
| Avg  |           |        |
```

#### Benchmark 5.1.2: Load 50 Notes from Database

**Setup:**
```bash
# Create 50 notes
for i in {1..50}; do
  curl -X POST http://localhost:8000/api/notes \
    -H "Authorization: Bearer $TOKEN1" \
    -H "Content-Type: application/json" \
    -d "{\"title\":\"Note $i\",\"content\":\"Content for note number $i with more text to make it realistic. This note contains some sample content that would be typical in a real note-taking application.\",\"tags\":\"test,performance\"}"
done
```

**Measure:**
```bash
time curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1"
```

**Target:** < 500ms

#### Benchmark 5.1.3: Load 100 Notes from Database

**Setup:** Create 100 notes (same pattern)

**Measure:** Same as above

**Target:** < 1000ms (1 second)

### 5.2 localStorage Cache Performance

#### Benchmark 5.2.1: localStorage Read Performance

**In browser console:**
```javascript
// Measure time to read from localStorage
console.time('localStorage-read');
const data = localStorage.getItem('semantic-notes-data');
const parsed = JSON.parse(data);
console.timeEnd('localStorage-read');
console.log('Notes count:', parsed.notes.length);
```

**Expected:** < 10ms

#### Benchmark 5.2.2: Database vs Cache Comparison

**Test 1: First load (database)**
```javascript
// Clear localStorage first
localStorage.removeItem('semantic-notes-data');

// Measure fetch from database
console.time('db-fetch');
await fetch('/api/notes', {
  headers: { 'Authorization': 'Bearer ' + token }
})
  .then(r => r.json());
console.timeEnd('db-fetch');
```

**Test 2: Second load (cache)**
```javascript
// Measure localStorage read
console.time('cache-hit');
const cached = JSON.parse(localStorage.getItem('semantic-notes-data'));
console.timeEnd('cache-hit');
```

**Expected Ratio:** Cache should be 10-50x faster

### 5.3 Graph Generation Performance

#### Benchmark 5.3.1: Graph with 10 Notes

**Frontend timing:**
```javascript
// In browser console during graph generation
// Look for timing logs or measure manually

console.time('graph-generation');
// Click generate graph button
// After graph renders:
console.timeEnd('graph-generation');
```

**Target:** < 2 seconds (including embedding computation)

#### Benchmark 5.3.2: Graph with Cached Embeddings

**Steps:**
1. Generate graph once (embeddings computed)
2. Clear graph
3. Generate again (embeddings from cache)

**Measure time difference**

**Expected:** 50-80% faster with cache

### 5.4 Embedding Operations

#### Benchmark 5.4.1: Compute Embeddings (Cold)

**Request:**
```bash
time curl -X POST http://localhost:8000/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "This is a test document",
      "Another test document",
      "Third test document"
    ]
  }'
```

**Target:** < 500ms for 3 documents

#### Benchmark 5.4.2: Save Embeddings to Database

**Measure batch save operation:**
```bash
time curl -X POST http://localhost:8000/api/embeddings/batch \
  -H "Authorization: Bearer $TOKEN1" \
  -H "Content-Type: application/json" \
  -d '{
    "embeddings": [
      {"note_id": 1, "content_hash": "hash1", "embedding": [...]},
      {"note_id": 2, "content_hash": "hash2", "embedding": [...]},
      ...
    ]
  }'
```

**Target:** < 200ms per embedding

#### Benchmark 5.4.3: Fetch Embeddings from Database

**Request:**
```bash
time curl "http://localhost:8000/api/embeddings?note_ids=1,2,3,4,5" \
  -H "Authorization: Bearer $TOKEN1"
```

**Target:** < 100ms for 5 embeddings

### 5.5 Performance Test Results Template

**Create results spreadsheet:**

| Operation | Target | Run 1 | Run 2 | Run 3 | Avg | Pass/Fail |
|-----------|--------|-------|-------|-------|-----|-----------|
| Load 10 notes (DB) | <200ms | | | | | |
| Load 50 notes (DB) | <500ms | | | | | |
| Load 100 notes (DB) | <1000ms | | | | | |
| localStorage read | <10ms | | | | | |
| Compute 3 embeddings | <500ms | | | | | |
| Save 5 embeddings | <1000ms | | | | | |
| Fetch 5 embeddings | <100ms | | | | | |
| Graph gen (10 notes) | <2000ms | | | | | |
| Graph gen (cached) | <1000ms | | | | | |

---

## 6. Error Handling Scenarios

### 6.1 Database Connection Errors

#### Test 6.1.1: Database Unavailable

**Simulate:**
```bash
# Stop PostgreSQL container
docker-compose down
```

**Frontend Actions:**
1. Try to login
2. Try to create note
3. Try to load notes

**Expected:**
- Clear error messages displayed
- Fallback to localStorage if possible
- No application crash
- User informed of connection issue
- Ability to work offline

**Error Messages to Check:**
- "Unable to connect to database"
- "Working in offline mode"
- "Your changes will sync when connection is restored"

#### Test 6.1.2: Database Timeout

**Simulate (requires test configuration):**
- Add network delay to database

**Expected:**
- Timeout error after reasonable wait
- Graceful fallback
- Retry mechanism if applicable

### 6.2 Authentication Errors

#### Test 6.2.1: Invalid Token Format

**Request:**
```bash
curl http://localhost:8000/api/notes \
  -H "Authorization: Bearer malformed.token.here"
```

**Expected Response (401):**
```json
{
  "detail": "Invalid or expired token"
}
```

#### Test 6.2.2: Expired Token

**Manually expire token in database or wait 7 days**

**Expected:**
- Auto-logout on next request
- Redirect to login
- Clear error message
- Token removed from localStorage

### 6.3 Network Interruption

#### Test 6.3.1: Network Lost During Operation

**Simulate:**
1. Start creating/editing note
2. Open DevTools → Network → Set to "Offline"
3. Try to save

**Expected:**
- Save to localStorage
- Show "Working offline" indicator
- Queue changes for sync
- Don't lose user's work

#### Test 6.3.2: Network Restored

**Steps:**
1. While offline, make changes
2. Restore network connection
3. Trigger sync (or automatic)

**Expected:**
- Changes sync to database
- Success notification
- No data loss
- No duplicate entries

### 6.4 Malformed Requests

#### Test 6.4.1: Missing Required Fields

**Request:**
```bash
curl -X POST http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Missing content field"
  }'
```

**Expected Response (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "content"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

#### Test 6.4.2: Invalid Data Types

**Request:**
```bash
curl -X POST http://localhost:8000/api/notes \
  -H "Authorization: Bearer $TOKEN1" \
  -H "Content-Type: application/json" \
  -d '{
    "title": 12345,
    "content": {"invalid": "type"},
    "tags": []
  }'
```

**Expected:** Validation error response

### 6.5 Concurrent Edit Conflicts

#### Test 6.5.1: Simultaneous Updates

**Setup:**
1. Two users edit same note at same time
2. Both submit changes

**Expected:**
- Last write wins (standard behavior)
- OR conflict resolution UI
- No data corruption
- Clear indication of what happened

### 6.6 Rate Limiting (if implemented)

**Test excessive requests:**
```bash
# Send 100 requests rapidly
for i in {1..100}; do
  curl http://localhost:8000/api/notes \
    -H "Authorization: Bearer $TOKEN1" &
done
```

**Monitor for:**
- Rate limit responses (429)
- Server stability
- No crashes

---

## 7. Test Execution Commands

### 7.1 Starting Services

#### 7.1.1 Start All Services (One Command)

**Create start script: `start-all.sh`**
```bash
#!/bin/bash

echo "🐘 Starting PostgreSQL..."
cd /mnt/c/Users/matth/Desktop/8-SANDBOX/notes/semantic-notes
docker-compose up -d

echo "⏳ Waiting for database to be ready..."
sleep 5

echo "🐍 Starting FastAPI backend..."
python main.py &
BACKEND_PID=$!

echo "⏳ Waiting for backend to start..."
sleep 3

echo "⚛️  Starting React frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "✅ All services started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "📡 Services:"
echo "  - Database: postgresql://localhost:5432"
echo "  - Backend:  http://localhost:8000"
echo "  - Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
wait
```

**Make executable and run:**
```bash
chmod +x start-all.sh
./start-all.sh
```

#### 7.1.2 Start Services Individually

**Terminal 1: Database**
```bash
wsl
cd /mnt/c/Users/matth/Desktop/8-SANDBOX/notes/semantic-notes
docker-compose up
```

**Terminal 2: Backend**
```bash
wsl
cd /mnt/c/Users/matth/Desktop/8-SANDBOX/notes/semantic-notes
python main.py
```

**Terminal 3: Frontend**
```bash
wsl
cd /mnt/c/Users/matth/Desktop/8-SANDBOX/notes/semantic-notes/frontend
npm run dev
```

### 7.2 Manual API Testing Commands

**Create test script: `test-api.sh`**
```bash
#!/bin/bash

BASE_URL="http://localhost:8000"

echo "🧪 API Test Suite"
echo "================="

# Test 1: Health Check
echo ""
echo "1. Health Check"
curl -s $BASE_URL/api/health | jq .

# Test 2: Register User
echo ""
echo "2. Register User"
REGISTER_RESPONSE=$(curl -s -X POST $BASE_URL/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "test123",
    "email": "test@example.com"
  }')

echo $REGISTER_RESPONSE | jq .

TOKEN=$(echo $REGISTER_RESPONSE | jq -r .access_token)
echo "Token: $TOKEN"

# Test 3: Create Note
echo ""
echo "3. Create Note"
NOTE_RESPONSE=$(curl -s -X POST $BASE_URL/api/notes \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Note",
    "content": "This is a test",
    "tags": "test"
  }')

echo $NOTE_RESPONSE | jq .

NOTE_ID=$(echo $NOTE_RESPONSE | jq -r .id)
echo "Note ID: $NOTE_ID"

# Test 4: List Notes
echo ""
echo "4. List Notes"
curl -s $BASE_URL/api/notes \
  -H "Authorization: Bearer $TOKEN" | jq .

# Test 5: Update Note
echo ""
echo "5. Update Note"
curl -s -X PUT $BASE_URL/api/notes/$NOTE_ID \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Updated Test Note",
    "content": "Content has been updated",
    "tags": "test,updated"
  }' | jq .

# Test 6: Move to Trash
echo ""
echo "6. Move to Trash"
curl -s -X POST $BASE_URL/api/notes/$NOTE_ID/trash \
  -H "Authorization: Bearer $TOKEN" | jq .

# Test 7: View Trash
echo ""
echo "7. View Trash"
curl -s $BASE_URL/api/trash \
  -H "Authorization: Bearer $TOKEN" | jq .

echo ""
echo "✅ Tests Complete"
```

**Run:**
```bash
chmod +x test-api.sh
./test-api.sh
```

### 7.3 Database Query Commands

**Create database check script: `check-db.sh`**
```bash
#!/bin/bash

CONTAINER="semantic-notes-db"
DB_USER="semantic_user"
DB_NAME="semantic_notes"

echo "📊 Database Status Check"
echo "======================="

echo ""
echo "1. Tables"
docker exec -it $CONTAINER psql -U $DB_USER -d $DB_NAME -c "\dt"

echo ""
echo "2. Indexes"
docker exec -it $CONTAINER psql -U $DB_USER -d $DB_NAME -c "\di"

echo ""
echo "3. User Count"
docker exec -it $CONTAINER psql -U $DB_USER -d $DB_NAME -c "SELECT COUNT(*) as user_count FROM users;"

echo ""
echo "4. Notes Count"
docker exec -it $CONTAINER psql -U $DB_USER -d $DB_NAME -c "SELECT COUNT(*) as note_count FROM notes;"

echo ""
echo "5. Embeddings Count"
docker exec -it $CONTAINER psql -U $DB_USER -d $DB_NAME -c "SELECT COUNT(*) as embedding_count FROM embeddings;"

echo ""
echo "6. Recent Notes"
docker exec -it $CONTAINER psql -U $DB_USER -d $DB_NAME -c "SELECT id, title, user_id, created_at, is_deleted FROM notes ORDER BY created_at DESC LIMIT 5;"
```

**Run:**
```bash
chmod +x check-db.sh
./check-db.sh
```

### 7.4 Log Monitoring

**View all service logs:**
```bash
# Backend logs
tail -f backend.log

# Database logs
docker-compose logs -f postgres

# Frontend logs (in terminal running npm)
# Already visible

# Combined monitoring
# Terminal 1:
docker-compose logs -f

# Terminal 2:
tail -f backend.log
```

### 7.5 Stopping Services

**Create stop script: `stop-all.sh`**
```bash
#!/bin/bash

echo "🛑 Stopping all services..."

# Stop frontend (if started with start-all.sh)
pkill -f "vite"

# Stop backend
pkill -f "python main.py"
pkill -f "uvicorn"

# Stop database
cd /mnt/c/Users/matth/Desktop/8-SANDBOX/notes/semantic-notes
docker-compose down

echo "✅ All services stopped"
```

**Run:**
```bash
chmod +x stop-all.sh
./stop-all.sh
```

### 7.6 Database Reset Commands

**Complete reset (⚠️ DELETES ALL DATA):**
```bash
# Stop database
docker-compose down

# Remove data volume
rm -rf postgres-data/

# Restart (will reinitialize)
docker-compose up -d
```

**Selective cleanup:**
```bash
# Delete all notes (keep users)
docker exec -it semantic-notes-db psql -U semantic_user -d semantic_notes -c "TRUNCATE notes CASCADE;"

# Delete all users (cascades to notes and embeddings)
docker exec -it semantic-notes-db psql -U semantic_user -d semantic_notes -c "TRUNCATE users CASCADE;"

# Delete specific user and their data
docker exec -it semantic-notes-db psql -U semantic_user -d semantic_notes -c "DELETE FROM users WHERE username='testuser';"
```

---

## 8. Validation Checklist

### 8.1 Backend Validation

**Authentication & Authorization:**
- [ ] User registration creates user in database
- [ ] Password is hashed (not stored as plaintext)
- [ ] Login returns valid JWT token
- [ ] Token includes user_id and expiry
- [ ] Protected endpoints require valid token
- [ ] Invalid tokens return 401 Unauthorized
- [ ] Expired tokens are rejected
- [ ] Token persists for 7 days

**Notes CRUD Operations:**
- [ ] Create note assigns unique ID
- [ ] Create note sets created_at timestamp
- [ ] Update note changes updated_at timestamp
- [ ] Update note only works for note owner
- [ ] List notes returns only user's notes
- [ ] List notes excludes deleted notes
- [ ] Notes have correct structure (title, content, tags)

**Trash Operations:**
- [ ] Move to trash sets is_deleted=true
- [ ] Move to trash sets deleted_at timestamp
- [ ] Trashed notes not in active notes list
- [ ] Trashed notes appear in trash endpoint
- [ ] Restore sets is_deleted=false
- [ ] Restore clears deleted_at
- [ ] Permanent delete removes from database
- [ ] Empty trash deletes all user's trashed notes

**Embedding Operations:**
- [ ] Embeddings saved with note_id reference
- [ ] content_hash stored correctly
- [ ] embedding_vector is FLOAT4 array
- [ ] model_name defaults correctly
- [ ] Batch save works for multiple embeddings
- [ ] Batch get retrieves multiple embeddings efficiently
- [ ] Embeddings deleted when note deleted (CASCADE)

**User Isolation:**
- [ ] Users cannot access other users' notes
- [ ] Users cannot modify other users' notes
- [ ] Users cannot delete other users' notes
- [ ] Users cannot access other users' embeddings
- [ ] Trash is user-specific
- [ ] All queries filtered by user_id

### 8.2 Frontend Validation

**Authentication UI:**
- [ ] Login form renders correctly
- [ ] Register form renders correctly
- [ ] Form validation shows errors
- [ ] Successful auth redirects to app
- [ ] Token saved in localStorage
- [ ] Token included in API requests
- [ ] Logout clears token and redirects
- [ ] AuthGuard protects routes
- [ ] Username displayed when authenticated

**Notes Management:**
- [ ] Notes list displays all user notes
- [ ] Notes sorted by date (newest first)
- [ ] Create note form works
- [ ] New notes appear immediately
- [ ] Edit note updates content
- [ ] Changes saved to database
- [ ] Note deletion moves to trash
- [ ] UI responsive and no freezing

**Trash View:**
- [ ] Trash view accessible
- [ ] Shows deleted notes only
- [ ] Restore button works
- [ ] Permanent delete works
- [ ] Empty trash clears all
- [ ] Counts accurate

**Sync & Cache:**
- [ ] Notes loaded from database on login
- [ ] localStorage cache updated
- [ ] Offline mode falls back to cache
- [ ] Changes sync when online
- [ ] No data loss
- [ ] Cache invalidation works

**Graph Visualization:**
- [ ] Graph generates with database notes
- [ ] Embeddings computed or loaded
- [ ] Cache improves performance
- [ ] Similar notes connected
- [ ] Visual rendering works
- [ ] No console errors

### 8.3 Database Validation

**Schema Integrity:**
- [ ] All tables exist (users, notes, embeddings)
- [ ] All columns have correct types
- [ ] Foreign keys configured correctly
- [ ] CASCADE deletes work
- [ ] Indexes created

**Data Integrity:**
- [ ] No orphaned notes (user_id references exist)
- [ ] No orphaned embeddings (note_id references exist)
- [ ] Timestamps are UTC
- [ ] No duplicate usernames
- [ ] No duplicate note_id in embeddings

**Performance:**
- [ ] Queries use indexes (check EXPLAIN)
- [ ] No full table scans on large tables
- [ ] Connection pooling configured

### 8.4 Integration Validation

**End-to-End Flows:**
- [ ] Register → Create note → View → Edit → Delete flow works
- [ ] Login → Load notes → Generate graph flow works
- [ ] Multi-device: Create on A → View on B works
- [ ] Offline → Online sync works
- [ ] User isolation maintained throughout

**Error Handling:**
- [ ] Database down: graceful fallback
- [ ] Network error: user-friendly messages
- [ ] Invalid input: validation errors shown
- [ ] Auth errors: proper 401/403 responses
- [ ] No silent failures

**Performance:**
- [ ] Load time within targets
- [ ] No significant slowdown vs localStorage-only
- [ ] Embedding cache effective
- [ ] Database queries optimized

### 8.5 Security Validation

**Authentication:**
- [ ] Passwords hashed with bcrypt
- [ ] JWT tokens signed
- [ ] Token expiry enforced
- [ ] No sensitive data in tokens
- [ ] HTTPS ready (for production)

**Authorization:**
- [ ] All protected endpoints check token
- [ ] User_id extracted from token
- [ ] All queries filtered by user_id
- [ ] No SQL injection vulnerabilities
- [ ] No unauthorized data access

### 8.6 Final Sign-Off Checklist

**Critical Items:**
- [ ] All backend endpoints respond correctly
- [ ] User authentication works end-to-end
- [ ] Notes CRUD operations functional
- [ ] User isolation enforced everywhere
- [ ] Embeddings persist and load correctly
- [ ] localStorage fallback works
- [ ] No data loss in any scenario
- [ ] Performance acceptable (<1s for most operations)
- [ ] Error messages user-friendly
- [ ] No console errors or warnings

**Documentation:**
- [ ] API endpoints documented
- [ ] Database schema documented
- [ ] Testing procedures documented
- [ ] Troubleshooting guide available
- [ ] Environment setup guide clear

**Production Readiness:**
- [ ] Environment variables configured
- [ ] Database backups configured
- [ ] Monitoring in place
- [ ] Error logging configured
- [ ] Security best practices followed

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Database connection failed

**Symptoms:** Backend can't connect to PostgreSQL

**Solutions:**
```bash
# Check if container is running
docker ps | grep postgres

# Check container logs
docker-compose logs postgres

# Verify DATABASE_URL in .env
cat .env | grep DATABASE_URL

# Restart container
docker-compose restart postgres

# Check if port 5432 is available
netstat -an | grep 5432
```

#### Issue: JWT token invalid or expired

**Symptoms:** 401 Unauthorized on protected endpoints

**Solutions:**
```bash
# Check JWT_SECRET_KEY is set
cat .env | grep JWT_SECRET_KEY

# Verify token not expired
# Use jwt.io to decode token and check exp claim

# Clear localStorage and re-login
# In browser console:
localStorage.clear()
```

#### Issue: CORS errors in frontend

**Symptoms:** Network requests blocked by CORS policy

**Solutions:**
```python
# Verify CORS middleware in main.py
# Should allow localhost:5173

# Check backend is running on correct port
curl http://localhost:8000/api/health
```

#### Issue: Embeddings not persisting

**Symptoms:** Graph regenerates embeddings every time

**Solutions:**
```bash
# Check embeddings table has data
docker exec -it semantic-notes-db psql -U semantic_user -d semantic_notes \
  -c "SELECT COUNT(*) FROM embeddings;"

# Verify batch save endpoint works
# Check backend logs for errors

# Check localStorage cache
# In browser console:
localStorage.getItem('semantic-emb-cache-v1')
```

#### Issue: Notes not syncing across devices

**Symptoms:** Changes on one device not visible on another

**Solutions:**
```bash
# Verify both devices use same user account
# Check tokens are for same user_id

# Force refresh on second device
# Clear localStorage and re-login

# Check database has latest data
docker exec -it semantic-notes-db psql -U semantic_user -d semantic_notes \
  -c "SELECT id, title, updated_at FROM notes ORDER BY updated_at DESC LIMIT 5;"
```

#### Issue: Frontend build errors

**Symptoms:** npm run dev fails

**Solutions:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check Node version
node --version  # Should be 16+

# Check for port conflicts
# Vite uses port 5173 by default
```

---

## Test Results Documentation

### Test Execution Log Template

**Date:** ________________  
**Tester:** ________________  
**Environment:** Local Development  
**Backend Version:** _______  
**Frontend Version:** _______  

| Test ID | Test Name | Status | Notes | Time |
|---------|-----------|--------|-------|------|
| 2.1.1 | Register User | | | |
| 2.1.3 | Login User | | | |
| 2.2.1 | Create Note | | | |
| 2.2.2 | List Notes | | | |
| 2.2.3 | Update Note | | | |
| 2.3.1 | Move to Trash | | | |
| 2.3.4 | Restore Note | | | |
| 2.4.1 | Save Embeddings | | | |
| 2.4.2 | Fetch Embeddings | | | |
| 3.1.1 | Frontend Registration | | | |
| 3.2.1 | Frontend Create Note | | | |
| 4.2.1 | User Isolation | | | |
| 5.1.1 | Load 10 Notes | | | |
| 6.1.1 | Database Unavailable | | | |

**Overall Status:** ☐ PASS  ☐ FAIL  ☐ PARTIAL

**Issues Found:**
1. 
2. 
3. 

**Recommendations:**
1. 
2. 
3. 

---

## Conclusion

This testing guide provides comprehensive coverage of all PostgreSQL integration features. Follow the test procedures in order, document results, and address any failures before declaring Phase 6 complete.

**Success Criteria:**
- All validation checklist items marked complete
- No critical issues remaining
- Performance within acceptable ranges
- User isolation verified
- No data loss scenarios
- Error handling working correctly

**Next Steps After Testing:**
1. Document any issues found
2. Create bug tickets for failures
3. Retest after fixes
4. Get sign-off from stakeholders
5. Plan production deployment (if applicable)

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-22  
**Status:** Ready for Use