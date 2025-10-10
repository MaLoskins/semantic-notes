// services/api.js
// API Service for interacting with the FastAPI backend

export class APIService {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.lastStats = null;
  }

  setBaseUrl(url) {
    this.baseUrl = url.replace(/\/$/, '');
  }

  async checkHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`);
      if (!response.ok) {
        throw new Error(`Health check failed: HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  async getStats() {
    try {
      const response = await fetch(`${this.baseUrl}/api/stats`);
      if (!response.ok) {
        throw new Error(`Stats fetch failed: HTTP ${response.status}`);
      }
      const stats = await response.json();
      this.lastStats = stats;
      return stats;
    } catch (error) {
      console.error('Stats fetch error:', error);
      throw error;
    }
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
    if (!documents || documents.length === 0) {
      throw new Error('No documents provided');
    }

    const payload = {
      documents,
      dr_method,
      n_components,
      cluster,
      include_embeddings
    };

    // Add optional parameters
    if (labels) payload.labels = labels;
    if (n_clusters) payload.n_clusters = n_clusters;

    // Set mode-specific parameters
    if (mode === 'knn') {
      payload.top_k = top_k;
    } else {
      payload.threshold = threshold;
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/graph`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Graph generation failed: HTTP ${response.status} - ${errorText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Graph generation error:', error);
      throw error;
    }
  }

  async embedDocuments(documents) {
    if (!documents || documents.length === 0) {
      throw new Error('No documents provided');
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/embed`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ documents }),
      });

      if (!response.ok) {
        throw new Error(`Embedding failed: HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Embedding error:', error);
      throw error;
    }
  }
}

// Create a singleton instance
const apiService = new APIService();

// Export both the instance and the class
export default apiService;
export { apiService };