import { useState, useEffect, useRef } from 'react';
import apiService from '../services/api';
import { cosineSimilarity } from '../utils/similarity';

const SEMANTIC_QUERY_DEBOUNCE = 500;
const MIN_SEM_QUERY_LEN = 3;

export function useSemanticSearch(notes, connected) {
  const [searchMode, setSearchMode] = useState('text'); // 'text' | 'semantic'
  const [searchTerm, setSearchTerm] = useState('');
  const [minSimilarity, setMinSimilarity] = useState(60); // 0-100%
  const [semanticResults, setSemanticResults] = useState([]); // [{index, score, percent}]
  const [semanticLoading, setSemanticLoading] = useState(false);
  const [semanticError, setSemanticError] = useState('');
  const semanticTimerRef = useRef(null);

  // Semantic Search (debounced)
  useEffect(() => {
    if (searchMode !== 'semantic') {
      setSemanticLoading(false);
      setSemanticError('');
      setSemanticResults([]);
      return;
    }
    const q = String(searchTerm || '').trim();
    if (!q) {
      setSemanticResults([]);
      setSemanticLoading(false);
      setSemanticError('');
      return;
    }
    if (q.length < MIN_SEM_QUERY_LEN) {
      setSemanticResults([]);
      setSemanticLoading(false);
      setSemanticError(`Type at least ${MIN_SEM_QUERY_LEN} characters for semantic search`);
      return;
    }
    if (!connected) {
      setSemanticResults([]);
      setSemanticLoading(false);
      setSemanticError('Semantic search requires backend connection');
      return;
    }

    if (semanticTimerRef.current) {
      clearTimeout(semanticTimerRef.current);
    }
    let cancelled = false;
    semanticTimerRef.current = setTimeout(async () => {
      setSemanticLoading(true);
      setSemanticError('');
      try {
        const [noteEmbs, queryEmb] = await Promise.all([
          apiService.getEmbeddingsForNotes(notes),
          apiService.embedText(q),
        ]);
        const scored = [];
        for (let i = 0; i < notes.length; i++) {
          const v = noteEmbs[i];
          if (!Array.isArray(v)) continue;
          const s = cosineSimilarity(queryEmb, v);
          scored.push({ index: i, score: s, percent: Math.round(s * 100) });
        }
        scored.sort((a, b) => b.score - a.score);
        if (!cancelled) setSemanticResults(scored);
      } catch (e) {
        console.error('Semantic search failed:', e);
        if (!cancelled) setSemanticError(e?.message || 'Semantic search failed');
      } finally {
        if (!cancelled) setSemanticLoading(false);
      }
    }, SEMANTIC_QUERY_DEBOUNCE);

    return () => {
      cancelled = true;
      if (semanticTimerRef.current) clearTimeout(semanticTimerRef.current);
    };
  }, [searchMode, searchTerm, notes, connected]);

  return {
    searchMode,
    setSearchMode,
    searchTerm,
    setSearchTerm,
    minSimilarity,
    setMinSimilarity,
    semanticResults,
    semanticLoading,
    semanticError,
  };
}
