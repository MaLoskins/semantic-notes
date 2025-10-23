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