import { useState, useEffect, useCallback, useRef } from 'react';
import apiService from '../services/api';

const GRAPH_UPDATE_DEBOUNCE = 500;

const GC_LS_KEY = 'graph-controls-prefs-v1';
const DEFAULT_GRAPH_PARAMS = {
  connection: 'knn',
  k_neighbors: 5,
  similarity_threshold: 0.7,
  dim_reduction: 'pca',
  clustering: null,
  n_clusters: 5,
};
const clamp = (n, min, max) => Math.min(max, Math.max(min, n));

export function useGraph(notes, connected, setError) {
  const [graphData, setGraphData] = useState(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const updateTimerRef = useRef(null);

  // Graph controls (persisted to localStorage)
  const [graphParams, setGraphParams] = useState(() => {
    try {
      const raw = localStorage.getItem(GC_LS_KEY);
      const parsed = raw ? JSON.parse(raw) : null;
      return { ...DEFAULT_GRAPH_PARAMS, ...(parsed || {}) };
    } catch {
      return DEFAULT_GRAPH_PARAMS;
    }
  });

  // Persist graph params to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(GC_LS_KEY, JSON.stringify(graphParams));
    } catch { /* ignore quota */ }
  }, [graphParams]);

  const handleControlsChange = useCallback((partial) => {
    setGraphParams((prev) => ({ ...prev, ...partial }));
  }, []);

  const handleControlsReset = useCallback(() => {
    setGraphParams(DEFAULT_GRAPH_PARAMS);
  }, []);

  // Generate graph with debouncing
  useEffect(() => {
    if (updateTimerRef.current) {
      clearTimeout(updateTimerRef.current);
    }

    if (!connected || notes.length < 2) {
      setGraphData(null);
      return;
    }

    updateTimerRef.current = setTimeout(async () => {
      setGraphLoading(true);
      try {
        const documents = notes.map((note) =>
          `${note.title}. ${note.content} ${note.tags || ''}`
        );

        const labels = notes.map((note) =>
          note.title.length > 30 ? `${note.title.substring(0, 30)}...` : note.title
        );

        const connection = graphParams.connection === 'threshold' ? 'threshold' : 'knn';
        const kNeighborsRaw = clamp(parseInt(graphParams.k_neighbors ?? 5, 10), 1, 10);
        const kNeighbors = Math.min(kNeighborsRaw, Math.max(1, notes.length - 1));
        const similarity_threshold = Math.max(0, Math.min(1, Number(graphParams.similarity_threshold ?? 0.7)));
        const dim_reduction = graphParams.dim_reduction === 'none' ? null : (graphParams.dim_reduction ?? 'pca');
        const clustering = graphParams.clustering ?? null;
        const n_clusters = clustering ? clamp(parseInt(graphParams.n_clusters ?? 5, 10), 2, 20) : undefined;

        const graph = await apiService.buildGraph({
          documents,
          labels,
          connection,
          k_neighbors: connection === 'knn' ? kNeighbors : undefined,
          similarity_threshold: connection === 'threshold' ? similarity_threshold : undefined,
          dim_reduction,
          clustering,
          n_clusters,
        });

        setGraphData(graph);
        setError(null);
      } catch (err) {
        console.error('Graph generation failed:', err);
        setError(`Graph generation failed: ${err.message}`);
      } finally {
        setGraphLoading(false);
      }
    }, GRAPH_UPDATE_DEBOUNCE);

    return () => {
      if (updateTimerRef.current) {
        clearTimeout(updateTimerRef.current);
      }
    };
  }, [notes, connected, graphParams, setError]);

  return {
    graphData,
    graphLoading,
    graphParams,
    handleControlsChange,
    handleControlsReset,
  };
}
