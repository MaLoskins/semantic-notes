// GraphVisualization.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import GraphControlsPanel from './GraphControlsPanel';
import ExportGraphModal from './ExportGraphModal';
import ToastNotification from './ToastNotification';

const COLORS = ['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000'];

// Tunables
const NODE_R = 12;
const COLLIDE_BASE = 20;            // base collide radius (topped up by degree)
const LABEL_ZOOM_THRESHOLD = 0.8;   // show link labels only when zoomed in
const MAX_CHARGE_DISTANCE = 700;    // cap n-body computations
const TARGET_FPS_MS = 16;           // ~60fps

// Core Physics Parameters
const INTRA_CLUSTER_LINK_DISTANCE_BASE = 40;   // base distance for links within same cluster
const INTER_CLUSTER_LINK_DISTANCE_BASE = 250;  // base distance for links between clusters
const BASE_CHARGE_STRENGTH = -35;              // base repulsion force for nodes
const CHARGE_PER_DEGREE = -10;                 // additional repulsion per connected edge
const CLUSTER_ANCHOR_STRENGTH = 0.15;          // how strongly nodes are pulled to cluster center
const RESTART_ALPHA = 0.1;                     // simulation "temperature" on restart/drag/visibility
const LINK_LABEL_WEIGHT_THRESHOLD = 0.4;       // minimum edge weight to show label
const ALPHA_DECAY_TICKS = 200;                 // number of ticks for simulation to cool down

export default function GraphVisualization({
  graphData,
  onNodeClick,
  selectedNote,
  width = 800,
  height = 600,
  controlsParams = {},
  onControlsChange = () => {},
  onControlsReset = () => {},
  stats = { nodes: 0, edges: 0 },
  loading = false,
  panelPosition = 'bottom-left'
}) {
  const svgRef = useRef(null);
  const gRootRef = useRef(null);
  const gLinksRef = useRef(null);
  const gLinkLabelsRef = useRef(null);
  const gNodesRef = useRef(null);

  const simulationRef = useRef(null);
  const zoomBehaviorRef = useRef(null);

  const [hoveredNode, setHoveredNode] = useState(null);
  const [isRunning, setIsRunning] = useState(true);

  const [exportOpen, setExportOpen] = useState(false);
  const [exportTransform, setExportTransform] = useState({ x: 0, y: 0, k: 1 });
  const [toastOpen, setToastOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState('');

  // --------- Data prep (memoized) ----------
  const { nodesMemo, linksMemo, degreeById, nodeById, clusterAnchors } = useMemo(() => {
    const nodes = (graphData?.nodes || []).map(n => ({
      ...n,
      id: String(n.id),
      // Normalize any precomputed [-1,1] space to actual px, else random seed
      x: typeof n.x === 'number' ? (n.x * width / 2 + width / 2) : Math.random() * width,
      y: typeof n.y === 'number' ? (n.y * height / 2 + height / 2) : Math.random() * height
    }));

    const edges = (graphData?.edges || []).map(e => ({
      ...e,
      source: String(e.source),
      target: String(e.target),
      weight: typeof e.weight === 'number' ? Math.max(0, Math.min(1, e.weight)) : 0.5
    }));

    // degree map
    const degree = new Map();
    edges.forEach(e => {
      degree.set(e.source, (degree.get(e.source) || 0) + 1);
      degree.set(e.target, (degree.get(e.target) || 0) + 1);
    });

    const byId = new Map(nodes.map(n => [n.id, n]));

    // cluster anchors: spread cluster centroids on a circle
    const clusters = Array.from(new Set(nodes.map(n => n.cluster).filter(v => v !== undefined)));
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.28; // ring radius for cluster anchors
    const anchors = new Map();
    const TWO_PI = Math.PI * 2;
    clusters.forEach((cl, i) => {
      const angle = (i / clusters.length) * TWO_PI;
      anchors.set(cl, { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) });
    });

    return {
      nodesMemo: nodes,
      linksMemo: edges,
      degreeById: degree,
      nodeById: byId,
      clusterAnchors: anchors
    };
  }, [graphData, width, height]);

  // --------- Build / (re)build scene ----------
  useEffect(() => {
    if (!svgRef.current || !nodesMemo.length) return;

    const svg = d3.select(svgRef.current);

    // Clear once per graphData change
    svg.selectAll('*').remove();

    // Root group for zoom/pan
    const gRoot = svg.append('g').attr('class', 'graph-root');
    gRootRef.current = gRoot;

    // Layers (links under nodes)
    const gLinks = gRoot.append('g').attr('class', 'layer links');
    const gLinkLabels = gRoot.append('g').attr('class', 'layer link-labels').attr('opacity', 0);
    const gNodes = gRoot.append('g').attr('class', 'layer nodes');

    gLinksRef.current = gLinks;
    gLinkLabelsRef.current = gLinkLabels;
    gNodesRef.current = gNodes;

    // Zoom (persist between updates)
    const zoom = d3.zoom()
      .scaleExtent([0.15, 6])
      .on('zoom', (event) => {
        gRoot.attr('transform', event.transform);
        // toggle link-label visibility at zoom threshold to reduce DOM cost
        gLinkLabels.attr('opacity', event.transform.k >= LABEL_ZOOM_THRESHOLD ? 1 : 0);
      });

    zoomBehaviorRef.current = zoom;
    svg.call(zoom);

    // Scales for link appearance
    const widthScale = d3.scaleLinear().domain([0, 1]).range([1, 3.5]);
    const alphaScale = d3.scaleLinear().domain([0, 1]).range([0.2, 0.85]);

    // Selections
    const linkSel = gLinks.selectAll('line')
      .data(linksMemo)
      .join('line')
      .attr('class', 'link')
      .attr('stroke', '#4a5568')
      .attr('stroke-linecap', 'round')
      .attr('stroke-opacity', d => alphaScale(d.weight))
      .attr('stroke-width', d => widthScale(d.weight))
      .style('pointer-events', 'none');

    // Only attach labels for heavier links; hide until zoomed in
    const labelSel = gLinkLabels.selectAll('text')
      .data(linksMemo.filter(d => d.weight > LINK_LABEL_WEIGHT_THRESHOLD))
      .join('text')
      .attr('class', 'link-label')
      .text(d => d.weight.toFixed(2))
      .attr('font-size', 10)
      .attr('fill', '#9aa0a6')
      .attr('text-anchor', 'middle')
      .style('pointer-events', 'none');

    const drag = d3.drag()
      .on('start', (event, d) => {
        if (!event.active && simulationRef.current) simulationRef.current.alphaTarget(RESTART_ALPHA).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on('end', (event, d) => {
        if (!event.active && simulationRef.current) simulationRef.current.alphaTarget(0);
        // Hold with Shift, otherwise release
        if (!event.sourceEvent.shiftKey) {
          d.fx = null;
          d.fy = null;
        }
      });

    const nodeGSel = gNodes.selectAll('g.node')
      .data(nodesMemo, d => d.id)
      .join(enter => {
        const g = enter.append('g')
          .attr('class', 'node')
          .style('cursor', 'pointer')
          .call(drag);

        g.append('circle')
          .attr('r', NODE_R)
          .attr('fill', d => (d.cluster !== undefined ? COLORS[d.cluster % COLORS.length] : COLORS[0]))
          .attr('stroke', d => d.id === String(selectedNote) ? '#ffffff' : '#1f2937')
          .attr('stroke-width', d => d.id === String(selectedNote) ? 3 : 1.5);

        g.append('text')
          .attr('y', -NODE_R - 4)
          .attr('text-anchor', 'middle')
          .attr('font-size', 8)
          .attr('font-weight', 500)
          .attr('fill', '#e5e7eb')
          .text(d => d.label || `Note ${d.id}`)
          .style('pointer-events', 'none');

        // Events
        g.on('click', (event, d) => {
           event.stopPropagation();
           onNodeClick(parseInt(d.id, 10));
        })
        .on('mouseenter', (event, d) => {
           setHoveredNode(d);
           d3.select(event.currentTarget).select('circle')
             .attr('r', NODE_R + 3)
             .attr('stroke-width', 2.5);
        })
        .on('mouseleave', (event, d) => {
           setHoveredNode(null);
           d3.select(event.currentTarget).select('circle')
             .attr('r', NODE_R)
             .attr('stroke-width', d.id === String(selectedNote) ? 3 : 1.5);
        });

        return g;
      });

    // --------- Force simulation (retuned) ----------
    const diag = Math.hypot(width, height);
    const chargeDistanceMax = Math.min(MAX_CHARGE_DISTANCE, diag * 0.9);

    // helpers that work before/after d3 mutates link endpoints into node objects
    const getCluster = (endpoint) => {
      if (endpoint && typeof endpoint === 'object') return endpoint.cluster;
      const id = String(endpoint);
      return nodeById.get(id)?.cluster;
    };

    // Cluster-aware link distance & strength
    const linkDistance = (d) => {
      const cA = getCluster(d.source);
      const cB = getCluster(d.target);
      const intra = cA !== undefined && cA === cB;
      const base = intra ? INTRA_CLUSTER_LINK_DISTANCE_BASE : INTER_CLUSTER_LINK_DISTANCE_BASE;
      const byWeight = intra ? 35 : 20;        // weight shortens more if intra
      return Math.max(20, base - byWeight * d.weight);
    };

    const linkStrength = (d) => {
      const cA = getCluster(d.source);
      const cB = getCluster(d.target);
      const intra = cA !== undefined && cA === cB;
      // stronger inside clusters, softer across
      const s = intra ? 0.6 + 0.4 * d.weight : 0.08 + 0.22 * d.weight;
      return Math.max(0.01, Math.min(1, s));
    };

    // Degree-aware repulsion and collision
    const chargeStrength = (node) => {
      const deg = degreeById.get(node.id) || 0;
      // More connected nodes repel a bit more to prevent hairballs
      const base = BASE_CHARGE_STRENGTH;
      const perDeg = CHARGE_PER_DEGREE; // each extra neighbor adds repulsion
      // clamp to avoid blowing apart the layout
      return Math.max(-260, base + perDeg * deg);
    };

    const collideRadius = (node) => {
      const deg = degreeById.get(node.id) || 0;
      // bigger collide radius for hubs
      return COLLIDE_BASE + Math.min(12, Math.sqrt(deg) * 2);
    };

    // Light cluster anchoring (keeps communities loosely coherent)
    const clusterX = d3.forceX(n => {
      if (n.cluster === undefined) return width / 2;
      return clusterAnchors.get(n.cluster)?.x ?? width / 2;
    }).strength(CLUSTER_ANCHOR_STRENGTH);

    const clusterY = d3.forceY(n => {
      if (n.cluster === undefined) return height / 2;
      return clusterAnchors.get(n.cluster)?.y ?? height / 2;
    }).strength(CLUSTER_ANCHOR_STRENGTH);

    // Mild global centering to avoid drift
    const center = d3.forceCenter(width / 2, height / 2);

    const sim = d3.forceSimulation(nodesMemo)
      .force('link', d3.forceLink(linksMemo)
        .id(d => d.id)
        .distance(linkDistance)
        .strength(linkStrength)
        .iterations(2))
      .force('charge', d3.forceManyBody()
        .strength(chargeStrength)
        .theta(0.9)
        .distanceMax(chargeDistanceMax))
      .force('collision', d3.forceCollide().radius(collideRadius).strength(0.7))
      .force('clusterX', clusterX)
      .force('clusterY', clusterY)
      .force('center', center)
      // gentle positional bias back to canvas (keeps things inside without hard walls)
      .force('x', d3.forceX(width / 2).strength(0.02))
      .force('y', d3.forceY(height / 2).strength(0.02))
      // realistic alpha settings
      .alpha(0.9)
      .alphaMin(0.001)
      .alphaDecay(1 - Math.pow(0.001, 1 / ALPHA_DECAY_TICKS));

    simulationRef.current = sim;

    // Throttle DOM writes to ~60 fps
    let lastRender = 0;
    let rafScheduled = false;

    const render = (now) => {
      rafScheduled = false;
      if (now - lastRender < TARGET_FPS_MS) return;
      lastRender = now;

      // Update link positions
      linkSel
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      // Midpoints for labels
      labelSel
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);

      // Node transforms
      nodeGSel.attr('transform', d => `translate(${d.x},${d.y})`);
    };

    sim.on('tick', () => {
      if (!rafScheduled) {
        rafScheduled = true;
        requestAnimationFrame(render);
      }
    });

    // Pause simulation when tab is hidden (battery/perf)
    const onVis = () => {
      if (document.hidden) {
        sim.stop();
      } else if (isRunning) {
        sim.alpha(RESTART_ALPHA).restart();
      }
    };
    document.addEventListener('visibilitychange', onVis);

    // Cleanup
    return () => {
      document.removeEventListener('visibilitychange', onVis);
      sim.stop();
    };
  }, [nodesMemo, linksMemo, width, height, onNodeClick, isRunning, degreeById, nodeById, clusterAnchors]);

  // --------- Highlight selected node without full redraw ----------
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('g.node circle')
      .attr('stroke', function () {
        const d = d3.select(this.parentNode).datum();
        return d && d.id === String(selectedNote) ? '#ffffff' : '#1f2937';
      })
      .attr('stroke-width', function () {
        const d = d3.select(this.parentNode).datum();
        const isSel = d && d.id === String(selectedNote);
        return isSel ? 3 : 1.5;
      })
      .attr('r', function () {
        const d = d3.select(this.parentNode).datum();
        const isHover = false; // no persisted hover state on circles
        return isHover ? NODE_R + 3 : NODE_R;
      });
  }, [selectedNote]);

  // --------- Controls ----------
  const handleResetView = () => {
    const svg = d3.select(svgRef.current);
    if (!zoomBehaviorRef.current) return;
    svg.transition().duration(300).call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
  };

  const handleToggleSimulation = () => {
    const sim = simulationRef.current;
    if (!sim) return;
    if (isRunning) {
      sim.stop();
    } else {
      sim.alpha(RESTART_ALPHA).restart();
    }
    setIsRunning(!isRunning);
  };

  const handleOpenExport = () => {
    try {
      const t = d3.zoomTransform(svgRef.current);
      setExportTransform({ x: t.x, y: t.y, k: t.k });
    } catch {
      setExportTransform({ x: 0, y: 0, k: 1 });
    }
    setExportOpen(true);
  };

  const handleNotify = (msg) => {
    setToastMessage(msg);
    setToastOpen(true);
  };

  return (
    <div className="graph-visualization">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="graph-svg"
        style={{ background: 'transparent', display: 'block' }}
      />

      <GraphControlsPanel
        params={controlsParams}
        onChange={onControlsChange}
        onReset={onControlsReset}
        stats={stats}
        loading={loading}
        position={panelPosition}
      />

      <div className="graph-controls">
        <button onClick={handleResetView} className="control-btn" title="Reset View">⟲</button>
        <button onClick={handleToggleSimulation} className="control-btn" title="Toggle Physics">
          {isRunning ? '❚❚' : '▶'}
        </button>
        <button onClick={handleOpenExport} className="control-btn" title="Export Graph">⤓</button>
      </div>

      {hoveredNode && (
        <div className="node-tooltip">
          <div className="tooltip-label">{hoveredNode.label}</div>
          <div className="tooltip-meta">
            ID: {hoveredNode.id}
            {hoveredNode.cluster !== undefined && ` • Cluster: ${hoveredNode.cluster}`}
          </div>
        </div>
      )}

      <ExportGraphModal
        isOpen={exportOpen}
        onClose={() => setExportOpen(false)}
        svgRef={svgRef}
        graphData={graphData}
        params={controlsParams}
        transform={exportTransform}
        onNotify={handleNotify}
      />

      <ToastNotification
        isOpen={toastOpen}
        message={toastMessage}
        onClose={() => setToastOpen(false)}
        duration={4000}
      />
    </div>
  );
}