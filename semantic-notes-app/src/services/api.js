const EMB_LS_KEY = 'semantic-emb-cache-v1';

class APIService {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    // Embedding cache: { [noteId]: { h: number, v: number[] } }
    this._embCache = this._loadEmbCache();
  }

  async request(endpoint, options = {}) {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
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

  async getEmbeddingsForNotes(notes) {
    if (!Array.isArray(notes) || notes.length === 0) return [];

    const results = new Array(notes.length).fill(null);
    const toComputeIdxs = [];
    const toComputeDocs = [];

    notes.forEach((note, i) => {
      const id = note?.id;
      const h = this.hashNote(note);
      const cached = id != null ? this._embCache[id] : undefined;
      if (cached && cached.h === h && Array.isArray(cached.v)) {
        results[i] = cached.v;
      } else {
        toComputeIdxs.push(i);
        toComputeDocs.push(this.getNoteText(note));
      }
    });

    if (toComputeDocs.length > 0) {
      const res = await this.embedDocuments(toComputeDocs);
      const vecs = res?.embeddings || [];
      if (vecs.length !== toComputeDocs.length) {
        throw new Error('Embedding service returned mismatched vector count');
      }
      // Map back and update cache for notes with stable ids
      toComputeIdxs.forEach((noteIdx, j) => {
        const v = vecs[j];
        results[noteIdx] = v;
        const note = notes[noteIdx];
        const id = note?.id;
        if (id != null) {
          this._embCache[id] = { h: this.hashNote(note), v };
        }
      });
      this._saveEmbCache();
    }

    return results;
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
}

const apiService = new APIService();

export default apiService;
export { APIService };