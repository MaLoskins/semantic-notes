const EMB_LS_KEY = 'semantic-emb-cache-v1';

class APIService {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
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