// services/api.js
class APIService {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async request(endpoint, options = {}) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Request failed: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  async checkHealth() {
    return this.request('/api/health');
  }

  async getStats() {
    return this.request('/api/stats');
  }

  async buildGraph(params) {
    const { mode = 'knn', ...restParams } = params;
    
    const payload = {
      ...restParams,
      ...(mode === 'knn' 
        ? { top_k: restParams.top_k || 2 }
        : { threshold: restParams.threshold || 0.3 })
    };

    return this.request('/api/graph', {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }
}

export default new APIService();