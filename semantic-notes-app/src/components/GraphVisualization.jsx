// components/GraphVisualization.jsx
import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const ResetIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
    <path d="M4 10a6 6 0 0112 0v1m0 0v3m0-3h-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" fill="none"/>
  </svg>
);

const PlayIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
    <path d="M7 5l8 5-8 5V5z" fill="currentColor"/>
  </svg>
);

const PauseIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
    <rect x="6" y="5" width="3" height="10" fill="currentColor"/>
    <rect x="11" y="5" width="3" height="10" fill="currentColor"/>
  </svg>
);

export default function GraphVisualization({ 
  graphData, 
  loading, 
  notes, 
  selectedNote,
  onNodeClick 
}) {
  const svgRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoveredNode, setHoveredNode] = useState(null);
  const [simulationRunning, setSimulationRunning] = useState(true);
  const simulationRef = useRef(null);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      const container = svgRef.current?.parentElement;
      if (container) {
        setDimensions({
          width: container.clientWidth,
          height: container.clientHeight
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Render graph
  useEffect(() => {
    if (!graphData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { nodes, edges } = graphData;
    if (!nodes || nodes.length === 0) return;

    const { width, height } = dimensions;

    // Setup
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

    // Scales
    const weightScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0.1, 1]);

    // Force simulation
    const simulation = d3.forceSimulation(nodeData)
      .force('link', d3.forceLink(linkData)
        .id(d => d.id)
        .distance(d => 100 * (1 - (d.weight || 0.5)))
        .strength(d => weightScale(d.weight || 0.5)))
      .force('charge', d3.forceManyBody().strength(-300).distanceMax(200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30).strength(0.5))
      .force('x', d3.forceX(width / 2).strength(0.02))
      .force('y', d3.forceY(height / 2).strength(0.02));

    simulationRef.current = simulation;

    // Links
    const link = g.append('g')
      .selectAll('line')
      .data(linkData)
      .enter().append('line')
      .attr('stroke', '#333')
      .attr('stroke-width', d => Math.max(0.5, (d.weight || 0.5) * 3))
      .attr('stroke-opacity', d => 0.3 + (d.weight || 0.5) * 0.4);

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

    // Node circles
    node.append('circle')
      .attr('r', d => d.id === String(selectedNote) ? 8 : 6)
      .attr('fill', d => {
        if (d.id === String(selectedNote)) return '#16a34a';
        if (d.cluster !== undefined) {
          const colors = ['#2563eb', '#7c3aed', '#dc2626', '#ea580c', '#16a34a'];
          return colors[d.cluster % colors.length];
        }
        return '#2563eb';
      })
      .attr('stroke', '#1a1a1a')
      .attr('stroke-width', 1.5);

    // Node labels
    node.append('text')
      .text(d => d.label || `Note ${d.id}`)
      .attr('x', 0)
      .attr('y', -12)
      .style('font-size', '11px')
      .style('font-weight', '500')
      .style('fill', '#e0e0e0')
      .style('text-anchor', 'middle')
      .style('pointer-events', 'none')
      .style('user-select', 'none');

    // Interactions
    node.on('click', (event, d) => {
        event.stopPropagation();
        onNodeClick(parseInt(d.id));
      })
      .on('mouseenter', (event, d) => {
        setHoveredNode(d);
        d3.select(event.currentTarget).select('circle')
          .transition().duration(150)
          .attr('r', 9);
      })
      .on('mouseleave', (event) => {
        setHoveredNode(null);
        d3.select(event.currentTarget).select('circle')
          .transition().duration(150)
          .attr('r', d => d.id === String(selectedNote) ? 8 : 6);
      });

    // Simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
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
  }, [graphData, selectedNote, dimensions, onNodeClick]);

  const handleResetView = () => {
    const svg = d3.select(svgRef.current);
    const zoom = d3.zoom().scaleExtent([0.1, 10]);
    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
  };

  const handleToggleSimulation = () => {
    if (simulationRef.current) {
      if (simulationRunning) {
        simulationRef.current.stop();
      } else {
        simulationRef.current.alpha(0.3).restart();
      }
      setSimulationRunning(!simulationRunning);
    }
  };

  if (loading) {
    return (
      <div className="graph-container">
        <div className="loading">
          <div className="spinner" />
          <span>Loading graph...</span>
        </div>
      </div>
    );
  }

  if (!notes || notes.length === 0) {
    return (
      <div className="graph-container">
        <div className="empty-state">
          <h3>Welcome to Semantic Notes</h3>
          <p>Create your first note to get started</p>
        </div>
      </div>
    );
  }

  if (notes.length === 1) {
    return (
      <div className="graph-container">
        <div className="empty-state">
          <p>Create at least 2 notes to visualize connections</p>
        </div>
      </div>
    );
  }

  return (
    <div className="graph-visualization">
      <svg 
        ref={svgRef} 
        width={dimensions.width} 
        height={dimensions.height}
        style={{ background: '#0f0f0f' }}
      />
      
      {graphData && (
        <div className="graph-controls">
          <button 
            onClick={handleResetView} 
            className="control-btn" 
            title="Reset View"
            aria-label="Reset view"
          >
            <ResetIcon />
          </button>
          <button 
            onClick={handleToggleSimulation} 
            className="control-btn" 
            title={simulationRunning ? 'Pause' : 'Play'}
            aria-label={simulationRunning ? 'Pause simulation' : 'Play simulation'}
          >
            {simulationRunning ? <PauseIcon /> : <PlayIcon />}
          </button>
        </div>
      )}

      {hoveredNode && (
        <div className="node-tooltip">
          <strong>{hoveredNode.label}</strong>
          {hoveredNode.cluster !== undefined && (
            <div>Cluster: {hoveredNode.cluster}</div>
          )}
        </div>
      )}

      {selectedNote !== null && notes[selectedNote] && (
        <div className="selected-note-preview">
          <h3>{notes[selectedNote].title}</h3>
          <p>{notes[selectedNote].content}</p>
          {notes[selectedNote].tags && (
            <div className="note-tags" style={{ marginTop: '0.5rem' }}>
              {notes[selectedNote].tags.split(',').map((tag, i) => (
                <span key={i} className="tag">{tag.trim()}</span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}