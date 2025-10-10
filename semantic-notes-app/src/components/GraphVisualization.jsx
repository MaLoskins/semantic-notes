import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import GraphControlsPanel from './GraphControlsPanel';

const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];

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
  const simulationRef = useRef(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [isRunning, setIsRunning] = useState(true);

  useEffect(() => {
    if (!graphData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { nodes, edges } = graphData;
    if (!nodes || nodes.length === 0) return;

    const g = svg.append('g');
    
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => g.attr('transform', event.transform));
    
    svg.call(zoom);

    // Prepare data
    const nodeData = nodes.map(n => ({
      ...n,
      id: String(n.id),
      x: n.x ? (n.x * width/2 + width/2) : Math.random() * width,
      y: n.y ? (n.y * height/2 + height/2) : Math.random() * height
    }));

    const linkData = edges.map(e => ({
      ...e,
      source: String(e.source),
      target: String(e.target)
    }));

    const weightScale = d3.scaleLinear().domain([0, 1]).range([0.1, 1]);

    // Create simulation
    const simulation = d3.forceSimulation(nodeData)
      .force('link', d3.forceLink(linkData)
        .id(d => d.id)
        .distance(d => 120 * (1 - (d.weight || 0.5)))
        .strength(d => weightScale(d.weight || 0.5)))
      .force('charge', d3.forceManyBody().strength(-400).distanceMax(250))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(35).strength(0.7))
      .force('x', d3.forceX(width / 2).strength(0.05))
      .force('y', d3.forceY(height / 2).strength(0.05));

    simulationRef.current = simulation;

    // Gradients
    const defs = svg.append('defs');
    linkData.forEach((d, i) => {
      const gradient = defs.append('linearGradient')
        .attr('id', `grad-${i}`)
        .attr('gradientUnits', 'userSpaceOnUse');
      
      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', '#4a5568')
        .attr('stop-opacity', 0.2);
      gradient.append('stop')
        .attr('offset', '50%')
        .attr('stop-color', '#4a5568')
        .attr('stop-opacity', weightScale(d.weight || 0.5));
      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', '#4a5568')
        .attr('stop-opacity', 0.2);
    });

    // Links
    const link = g.append('g')
      .selectAll('line')
      .data(linkData)
      .enter().append('line')
      .attr('stroke', (d, i) => `url(#grad-${i})`)
      .attr('stroke-width', d => Math.max(1, (d.weight || 0.5) * 4))
      .style('pointer-events', 'none');

    // Link labels
    const linkLabels = g.append('g')
      .selectAll('text')
      .data(linkData.filter(d => d.weight > 0.5))
      .enter().append('text')
      .text(d => d.weight.toFixed(2))
      .style('font-size', '10px')
      .style('fill', '#6b7280')
      .style('text-anchor', 'middle')
      .style('pointer-events', 'none');

    // Nodes
    const node = g.append('g')
      .selectAll('g')
      .data(nodeData)
      .enter().append('g')
      .style('cursor', 'pointer')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    node.append('circle')
      .attr('r', d => d.id === String(selectedNote) ? 12 : 10)
      .attr('fill', d => {
        if (d.id === String(selectedNote)) return '#10b981';
        if (d.cluster !== undefined) return COLORS[d.cluster % COLORS.length];
        return COLORS[0];
      })
      .attr('stroke', '#1f2937')
      .attr('stroke-width', 2)
      .style('filter', 'drop-shadow(0 4px 6px rgba(0, 0, 0, 0.3))');

    node.append('text')
      .text(d => d.label || `Note ${d.id}`)
      .attr('y', -15)
      .style('font-size', '12px')
      .style('font-weight', '500')
      .style('fill', '#e5e7eb')
      .style('text-anchor', 'middle')
      .style('pointer-events', 'none')
      .style('filter', 'drop-shadow(0 1px 2px rgba(0, 0, 0, 0.8))');

    node.on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick(parseInt(d.id));
      })
      .on('mouseenter', (event, d) => {
        setHoveredNode(d);
        d3.select(event.currentTarget).select('circle')
          .transition().duration(200)
          .attr('r', 14)
          .attr('stroke-width', 3);
      })
      .on('mouseleave', (event, d) => {
        setHoveredNode(null);
        d3.select(event.currentTarget).select('circle')
          .transition().duration(200)
          .attr('r', d.id === String(selectedNote) ? 12 : 10)
          .attr('stroke-width', 2);
      });

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      linkLabels
        .attr('x', d => (d.source.x + d.target.x) / 2)
        .attr('y', d => (d.source.y + d.target.y) / 2);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      if (!event.sourceEvent.shiftKey) {
        d.fx = null;
        d.fy = null;
      }
    }

    return () => simulation.stop();
  }, [graphData, selectedNote, width, height, onNodeClick]);

  const handleResetView = () => {
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom().scaleExtent([0.1, 10]);
    svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
  };

  const handleToggleSimulation = () => {
    if (simulationRef.current) {
      if (isRunning) {
        simulationRef.current.stop();
      } else {
        simulationRef.current.alpha(0.3).restart();
      }
      setIsRunning(!isRunning);
    }
  };

  return (
    <div className="graph-visualization">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{
          background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
          borderRadius: '8px'
        }}
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
        <button onClick={handleResetView} className="control-btn" title="Reset View">
          ⟲
        </button>
        <button onClick={handleToggleSimulation} className="control-btn" title="Toggle Physics">
          {isRunning ? '❚❚' : '▶'}
        </button>
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
    </div>
  );
}