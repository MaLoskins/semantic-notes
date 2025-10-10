// hooks/useGraph.js
import { useState, useEffect } from 'react';
import apiService from '../services/api';

export function useGraph(notes, connected) {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const generateGraph = async () => {
      if (!connected || notes.length < 2) {
        setGraphData(null);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const documents = notes.map(note => 
          `${note.title}. ${note.content} ${note.tags || ''}`
        );
        
        const labels = notes.map(note => 
          note.title.length > 30 
            ? note.title.substring(0, 30) + '...' 
            : note.title
        );

        const graph = await apiService.buildGraph({
          documents,
          labels,
          mode: 'knn',
          top_k: Math.min(2, notes.length - 1),
          dr_method: 'pca'
        });

        setGraphData(graph);
      } catch (err) {
        console.error('Failed to generate graph:', err);
        setError('Failed to generate semantic graph');
      } finally {
        setLoading(false);
      }
    };

    generateGraph();
  }, [notes, connected]);

  return { graphData, loading, error };
}