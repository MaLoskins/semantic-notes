import React, { useEffect, useMemo, useState } from 'react';
import {
  defaultFilename,
  savePng,
  saveSvg,
  saveJson,
  copyPng,
  copySvg,
  copyJson,
  formatTimestamp,
} from '../utils/graphExport';

export default function ExportGraphModal({
  isOpen = false,
  onClose,
  svgRef,
  graphData,
  params = {},
  transform = { x: 0, y: 0, k: 1 },
  onNotify, // optional toast notifier: (msg: string) => void
}) {
  const [format, setFormat] = useState('png'); // 'png' | 'svg' | 'json'
  const [filename, setFilename] = useState(defaultFilename('png'));
  const [scale, setScale] = useState(2); // PNG resolution multiplier
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isOpen) return;
    setFormat('png');
    setScale(2);
    setError('');
    setBusy(false);
    setFilename(defaultFilename('png'));
  }, [isOpen]);

  const metadata = useMemo(() => ({
    exported_at: new Date().toISOString(),
    timestamp: formatTimestamp(),
    params,
    transform,
    stats: {
      nodes: graphData?.nodes?.length || 0,
      edges: graphData?.edges?.length || 0,
    },
  }), [params, transform, graphData]);

  const ensureExt = (name, ext) => {
    const lower = String(name || '').toLowerCase().trim();
    if (!lower.endsWith(`.${ext}`)) return `${name}.${ext}`;
    return name;
  };

  const handleDownload = async () => {
    try {
      setBusy(true);
      setError('');
      const svgEl = svgRef?.current;
      if (!svgEl) throw new Error('SVG not available');

      if (format === 'png') {
        const fname = ensureExt(filename || defaultFilename('png'), 'png');
        await savePng(svgEl, fname, scale);
        onNotify?.(`Exported PNG: ${fname}`);
      } else if (format === 'svg') {
        const fname = ensureExt(filename || defaultFilename('svg'), 'svg');
        saveSvg(svgEl, fname);
        onNotify?.(`Exported SVG: ${fname}`);
      } else if (format === 'json') {
        const fname = ensureExt(filename || defaultFilename('json'), 'json');
        saveJson(graphData, fname, metadata);
        onNotify?.(`Exported JSON: ${fname}`);
      }
      onClose?.();
    } catch (e) {
      setError(e?.message || 'Export failed');
    } finally {
      setBusy(false);
    }
  };

  const handleCopy = async () => {
    try {
      setBusy(true);
      setError('');
      const svgEl = svgRef?.current;
      if (format === 'png') {
        await copyPng(svgEl, scale);
        onNotify?.('PNG copied to clipboard');
      } else if (format === 'svg') {
        await copySvg(svgEl);
        onNotify?.('SVG copied to clipboard');
      } else if (format === 'json') {
        await copyJson(graphData, metadata);
        onNotify?.('JSON copied to clipboard');
      }
      onClose?.();
    } catch (e) {
      // Graceful fallback for clipboard limitations
      setError(e?.message || 'Copy failed. Your browser may not support clipboard for this format.');
    } finally {
      setBusy(false);
    }
  };

  if (!isOpen) return null;

  const stop = (e) => e.stopPropagation();

  return (
    <div className="modal-overlay" onClick={() => !busy && onClose?.()}>
      <div className="modal" role="dialog" aria-modal="true" onClick={stop}>
        <div className="modal-header">
          <h3>Export Graph</h3>
        </div>

        <div className="modal-body">
          <div className="gcp-row" style={{ marginBottom: '0.75rem' }}>
            <label className="gcp-label" htmlFor="export-format">Format</label>
            <div id="export-format" className="radio-group" role="radiogroup" aria-label="Export format">
              <label className="radio">
                <input
                  type="radio"
                  name="format"
                  value="png"
                  checked={format === 'png'}
                  onChange={() => {
                    setFormat('png');
                    setFilename(defaultFilename('png'));
                  }}
                  disabled={busy}
                />
                PNG (image)
              </label>
              <label className="radio" style={{ marginLeft: '1rem' }}>
                <input
                  type="radio"
                  name="format"
                  value="svg"
                  checked={format === 'svg'}
                  onChange={() => {
                    setFormat('svg');
                    setFilename(defaultFilename('svg'));
                  }}
                  disabled={busy}
                />
                SVG (vector)
              </label>
              <label className="radio" style={{ marginLeft: '1rem' }}>
                <input
                  type="radio"
                  name="format"
                  value="json"
                  checked={format === 'json'}
                  onChange={() => {
                    setFormat('json');
                    setFilename(defaultFilename('json'));
                  }}
                  disabled={busy}
                />
                JSON (data)
              </label>
            </div>
          </div>

          {format === 'png' && (
            <div className="gcp-row" style={{ marginBottom: '0.75rem' }}>
              <label className="gcp-label" htmlFor="export-scale">Resolution</label>
              <select
                id="export-scale"
                className="gcp-select"
                value={scale}
                onChange={(e) => setScale(parseInt(e.target.value, 10) || 2)}
                disabled={busy}
                title="Export scale factor"
              >
                <option value={1}>1x (standard)</option>
                <option value={2}>2x (high)</option>
                <option value={3}>3x (ultra)</option>
              </select>
            </div>
          )}

          <div className="gcp-row" style={{ marginBottom: '0.25rem' }}>
            <label className="gcp-label" htmlFor="export-filename">Filename</label>
            <input
              id="export-filename"
              className="gcp-select"
              type="text"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              disabled={busy}
              placeholder={defaultFilename(format)}
              title="Choose a filename"
            />
          </div>

          <div className="small" style={{ opacity: 0.8, marginTop: '0.25rem' }}>
            Tip: Exported image reflects current zoom/pan and styling.
          </div>

          {error && (
            <div className="error-banner" style={{ marginTop: '0.5rem' }}>
              ⚠ {error}
            </div>
          )}
        </div>

        <div className="modal-actions">
          <button className="btn btn-primary" onClick={handleDownload} disabled={busy}>
            {busy ? 'Exporting…' : 'Download'}
          </button>
          <button className="btn btn-secondary" onClick={handleCopy} disabled={busy} title="Copy to clipboard">
            Copy
          </button>
          <button className="btn" onClick={() => !busy && onClose?.()} disabled={busy}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}