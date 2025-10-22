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