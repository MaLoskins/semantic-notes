/**
 * Graph export utilities: PNG, SVG, JSON and clipboard helpers
 */

function pad2(n) {
  return String(n).padStart(2, '0');
}

export function formatTimestamp(date = new Date()) {
  const y = date.getFullYear();
  const m = pad2(date.getMonth() + 1);
  const d = pad2(date.getDate());
  const hh = pad2(date.getHours());
  const mm = pad2(date.getMinutes());
  return `${y}-${m}-${d}-${hh}-${mm}`;
}

export function defaultFilename(format = 'png') {
  const ts = formatTimestamp();
  if (format === 'json') return `semantic-graph-data-${ts}.json`;
  return `semantic-graph-${ts}.${format}`;
}

/**
 * Serialize an SVG element to string, ensuring necessary namespaces.
 */
export function getSvgString(svgEl) {
  if (!svgEl) throw new Error('SVG element is required');
  const cloned = svgEl.cloneNode(true);

  // Ensure width/height attributes exist (for canvas rasterization)
  const width = Number(svgEl.getAttribute('width') || svgEl.clientWidth || 800);
  const height = Number(svgEl.getAttribute('height') || svgEl.clientHeight || 600);
  cloned.setAttribute('width', String(width));
  cloned.setAttribute('height', String(height));

  // Add xmlns if missing
  if (!cloned.getAttribute('xmlns')) {
    cloned.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  }
  if (!cloned.getAttribute('xmlns:xlink')) {
    cloned.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
  }

  // Serialize
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(cloned);

  // Fix for some browsers that omit namespaces
  if (!source.match(/^<svg[^>]+xmlns="/)) {
    source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
  }
  if (!source.match(/^<svg[^>]+"http:\/\/www\.w3\.org\/1999\/xlink"/)) {
    source = source.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
  }

  // Ensure proper XML header
  return `<?xml version="1.0" standalone="no"?>${source}`;
}

/**
 * Download a Blob with the given filename.
 */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }, 0);
}

/**
 * Convert an SVG element to a PNG Blob by drawing into a high-DPI canvas.
 * scale: rasterization scale factor (2 for 2x resolution)
 */
export async function svgToPng(svgEl, scale = 2) {
  const width = Number(svgEl.getAttribute('width') || svgEl.clientWidth || 800);
  const height = Number(svgEl.getAttribute('height') || svgEl.clientHeight || 600);
  const svgString = getSvgString(svgEl);

  const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(svgBlob);

  try {
    const img = await new Promise((resolve, reject) => {
      const image = new Image();
      // Important: set crossOrigin to avoid taint if external resources are referenced
      image.crossOrigin = 'anonymous';
      image.onload = () => resolve(image);
      image.onerror = (e) => reject(new Error('Failed to load SVG for rasterization'));
      image.src = url;
    });

    const canvas = document.createElement('canvas');
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);

    const ctx = canvas.getContext('2d');
    // High quality rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Scale context and draw the SVG image
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0, width, height);

    const pngBlob = await new Promise((resolve) => {
      canvas.toBlob((b) => resolve(b), 'image/png');
    });

    if (!pngBlob) throw new Error('Canvas export produced empty image');
    return pngBlob;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function savePng(svgEl, filename, scale = 2) {
  const blob = await svgToPng(svgEl, scale);
  downloadBlob(blob, filename);
  return blob;
}

export function saveSvg(svgEl, filename) {
  const svgString = getSvgString(svgEl);
  const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
  downloadBlob(svgBlob, filename);
  return svgBlob;
}

export function buildJsonExport(graphData, metadata = {}) {
  const payload = {
    type: 'semantic-graph',
    version: 1,
    created_at: new Date().toISOString(),
    metadata,
    nodes: Array.isArray(graphData?.nodes) ? graphData.nodes : [],
    edges: Array.isArray(graphData?.edges) ? graphData.edges : [],
  };
  return JSON.stringify(payload, null, 2);
}

export function saveJson(graphData, filename, metadata = {}) {
  const json = buildJsonExport(graphData, metadata);
  const blob = new Blob([json], { type: 'application/json;charset=utf-8' });
  downloadBlob(blob, filename);
  return blob;
}

/**
 * Clipboard helpers
 */
export async function copyBlobToClipboard(blob, mime) {
  if (!navigator?.clipboard || !window.ClipboardItem) {
    throw new Error('Clipboard API not supported');
  }
  const item = new ClipboardItem({ [mime]: blob });
  await navigator.clipboard.write([item]);
}

export async function copyPng(svgEl, scale = 2) {
  const blob = await svgToPng(svgEl, scale);
  await copyBlobToClipboard(blob, 'image/png');
}

export async function copySvg(svgEl) {
  // Prefer writing as image/svg+xml if supported, otherwise as text
  const svgString = getSvgString(svgEl);
  if (navigator?.clipboard && window.ClipboardItem) {
    const blob = new Blob([svgString], { type: 'image/svg+xml' });
    try {
      await copyBlobToClipboard(blob, 'image/svg+xml');
      return;
    } catch {
      // fallback to text
    }
  }
  await navigator.clipboard.writeText(svgString);
}

export async function copyJson(graphData, metadata = {}) {
  const json = buildJsonExport(graphData, metadata);
  await navigator.clipboard.writeText(json);
}