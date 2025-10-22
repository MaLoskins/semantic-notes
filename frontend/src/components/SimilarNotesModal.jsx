import React, { useEffect, useState } from 'react';
import apiService from '../services/api';

export default function SimilarNotesModal({
  isOpen = false,
  baseDoc = '',
  baseTitle = 'This note',
  notes = [],
  excludeIndex = null,
  topK = 8,
  onClose,
  onSelect,
  onLink
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState([]);

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setError('');
      try {
        if (!baseDoc?.trim()) {
          setResults([]);
          setError('Note content is empty.');
          return;
        }
        const [embs, baseEmb] = await Promise.all([
          apiService.getEmbeddingsForNotes(notes),
          apiService.embedText(baseDoc),
        ]);
        const scored = [];
        for (let i = 0; i < notes.length; i++) {
          if (excludeIndex != null && i === excludeIndex) continue;
          const v = embs[i];
          if (!Array.isArray(v)) continue;
          const score = apiService.cosineSimilarity(baseEmb, v);
          scored.push({
            index: i,
            score,
            percent: Math.round(score * 100),
            title: notes[i]?.title || '(Untitled)',
            preview: (notes[i]?.content || '').substring(0, 160)
          });
        }
        scored.sort((a, b) => b.score - a.score);
        const top = scored.slice(0, topK);
        if (!cancelled) setResults(top);
      } catch (e) {
        console.error('Find similar failed:', e);
        if (!cancelled) setError(e?.message || 'Failed to compute similarities');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [isOpen, baseDoc, notes, excludeIndex, topK]);

  if (!isOpen) return null;

  const stop = (e) => e.stopPropagation();

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" onClick={stop}>
        <div className="modal-header">
          <h3>Notes similar to “{baseTitle || 'Untitled'}”</h3>
        </div>
        <div className="modal-body">
          {loading ? (
            <div className="loading" style={{ position: 'static', transform: 'none', padding: 0 }}>
              <div className="loading-spinner" />
              <div>Computing similarities...</div>
            </div>
          ) : error ? (
            <div className="error-banner" style={{ position: 'static' }}>
              ⚠ {error}
            </div>
          ) : results.length === 0 ? (
            <div className="empty-message">
              <div className="empty-message-title">No similar notes found</div>
              <div className="empty-message-hint">Try adding more content or different keywords</div>
            </div>
          ) : (
            <div className="similar-list">
              {results.map((r) => (
                <div key={r.index} className="similar-item">
                  <div
                    className="similar-main"
                    onClick={() => onSelect && onSelect(r.index)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter') onSelect && onSelect(r.index); }}
                  >
                    <div className="similar-header">
                      <div className="similar-title">{r.title}</div>
                      <div className="similar-score">{r.percent}% similar</div>
                    </div>
                    <div className="similar-preview">
                      {r.preview}{(notes[r.index]?.content || '').length > r.preview.length ? '…' : ''}
                    </div>
                    <div className="similar-meter">
                      <div className="similar-meter-fill" style={{ width: `${Math.min(100, Math.max(0, r.percent))}%` }} />
                    </div>
                  </div>
                  <div className="similar-actions">
                    <button className="btn btn-secondary btn-sm" onClick={() => onSelect && onSelect(r.index)} title="Open note">Open</button>
                    <button className="btn btn-primary btn-sm" onClick={() => onLink && onLink(r.index)} title="Link notes">Link</button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}