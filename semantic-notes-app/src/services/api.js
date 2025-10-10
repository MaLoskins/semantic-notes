class APIService {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
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

  async buildGraph({
    documents,
    labels = null,
    mode = 'knn',
    top_k = 2,
    threshold = 0.3,
    dr_method = 'pca',
    n_components = 2,
    cluster = 'none',
    n_clusters = null,
    include_embeddings = false
  }) {
    if (!documents?.length) {
      throw new Error('No documents provided');
    }

    const payload = {
      documents,
      dr_method,
      n_components,
      cluster,
      include_embeddings
    };

    if (labels) payload.labels = labels;
    if (n_clusters) payload.n_clusters = n_clusters;

    if (mode === 'knn') {
      payload.top_k = top_k;
    } else {
      payload.threshold = threshold;
    }

    return this.request('/api/graph', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }
}

const apiService = new APIService();

export default apiService;
export { APIService };