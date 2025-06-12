import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function CitationPage() {
  const svgRef = useRef();
  const [allData, setAllData] = useState(null);
  const [visibleData, setVisibleData] = useState(null);
  const [searchKeyword, setSearchKeyword] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    fetch('/citation_graph_with_cluster_v3.json')
      .then(res => res.json())
      .then(json => {
        const nodes = json.nodes;
        const nodeById = new Map(nodes.map(d => [d.id, d]));
        const links = json.links.map(d => ({
          source: nodeById.get(d.source),
          target: nodeById.get(d.target),
          weight: d.weight,
          similarity: d.similarity
        }));
        setAllData({ nodes, links });
      });
  }, []);

  const safeSelector = (id) => {
    let cleanedId = id.replace(/[^a-zA-Z0-9_-]/g, "_");
    if (!/^[a-zA-Z_]/.test(cleanedId)) {
      cleanedId = "_" + cleanedId;
    }
    return cleanedId;
  };

  const handleInputChange = (e) => {
    const keyword = e.target.value;
    setSearchKeyword(keyword);

    if (keyword.length < 2 || !allData) {
      setSuggestions([]);
      return;
    }

    const matched = allData.nodes
      .filter(n => n.name.toLowerCase().includes(keyword.toLowerCase()))
      .slice(0, 10);
    setSuggestions(matched);
  };

  const handleSearch = () => {
    if (!allData) return;
    const matchedNode = allData.nodes.find(n => n.name.toLowerCase() === searchKeyword.toLowerCase());
    if (matchedNode) {
      handleNodeFocus(matchedNode);
    } else {
      alert('해당 논문을 찾을 수 없습니다.');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleNodeFocus = (node) => {
    const paperId = node.id;
    const graph = buildGraph(allData.links);
    const visited = new Set();
    const queue = [paperId];
    const bfsResult = [];

    while (queue.length > 0 && bfsResult.length < 30) {
      const current = queue.shift();
      if (!visited.has(current)) {
        visited.add(current);
        bfsResult.push(current);
        const neighbors = graph.get(current) || [];
        queue.push(...neighbors);
      }
    }

    const filteredNodes = allData.nodes.filter(n => bfsResult.includes(n.id));
    const nodeById = new Map(filteredNodes.map(d => [d.id, d]));
    const filteredLinks = allData.links
      .filter(link => bfsResult.includes(link.source.id) && bfsResult.includes(link.target.id))
      .map(link => ({
        source: nodeById.get(link.source.id),
        target: nodeById.get(link.target.id),
        weight: link.weight,
        similarity: link.similarity
      }));

    const citedBy = allData.links
      .filter(l => l.target.id === node.id)
      .map(l => allData.nodes.find(n => n.id === l.source.id));

    const cites = allData.links
      .filter(l => l.source.id === node.id)
      .map(l => allData.nodes.find(n => n.id === l.target.id));

    setVisibleData({ nodes: filteredNodes, links: filteredLinks });
    setSuggestions([]);
    setSearchKeyword(node.name);
    setSelectedNode({ ...node, citedBy, cites });
  };

  const buildGraph = (links) => {
    const graph = new Map();
    links.forEach(link => {
      const sourceId = link.source.id;
      const targetId = link.target.id;
      if (!graph.has(sourceId)) graph.set(sourceId, []);
      if (!graph.has(targetId)) graph.set(targetId, []);
      graph.get(sourceId).push(targetId);
      graph.get(targetId).push(sourceId);
    });
    return graph;
  };

  const zoomInToNode = (nodeId) => {
    const svg = d3.select(svgRef.current);
    const width = 1000, height = 700;

    const node = d3.select(`g#${safeSelector(nodeId)}`);
    if (node.empty()) return;

    const d = node.datum();
    if (!d || d.x == null || d.y == null) return;

    const scale = 1.5;
    const transform = d3.zoomIdentity
      .translate(width / 2 - d.x * scale, height / 2 - d.y * scale)
      .scale(scale);

    const zoom = d3.zoom().on('zoom', (event) => {
      svg.select('g').attr('transform', event.transform);
    });
    svg.call(zoom);
    svg.transition().duration(750).call(zoom.transform, transform);
  };

  useEffect(() => {
    if (selectedNode) {
      zoomInToNode(selectedNode.id);
    }
  }, [selectedNode]);

  useEffect(() => {
    if (!visibleData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 1000;
    const height = 700;
    svg.attr('viewBox', [0, 0, width, height]);

    const svgGroup = svg.append("g");

    const allGroups = [...new Set(visibleData.nodes.map(d => d.group))];
    const color = d3.scaleOrdinal(d3.schemeCategory10).domain(allGroups);

    const simulation = d3.forceSimulation(visibleData.nodes)
      .force('link', d3.forceLink(visibleData.links)
        .id(d => d.id)
        .distance(d => 200 * (1 - (d.similarity || 0.5))))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(30));

    const link = svgGroup.append('g')
      .attr('stroke', '#aaa')
      .attr('stroke-opacity', 0.3)
      .selectAll('line')
      .data(visibleData.links)
      .join('line')
      .attr('stroke-width', d => Math.sqrt(d.weight));

    const nodeGroup = svgGroup.append('g')
      .selectAll('g')
      .data(visibleData.nodes)
      .join('g')
      .attr('id', d => safeSelector(d.id))
      .call(drag(simulation))
      .on("click", (event, d) => {
        handleNodeFocus(d);
      });

    nodeGroup.append('circle')
      .attr('r', d => (selectedNode && d.id === selectedNode.id) || d.name === searchKeyword ? 15 : 10)
      .attr('fill', d => color(d.group))
      .attr('stroke', d => (selectedNode && d.id === selectedNode.id) || d.name === searchKeyword ? 'red' : '#fff')
      .attr('stroke-width', d => (selectedNode && d.id === selectedNode.id) || d.name === searchKeyword ? 3 : 1)
      .style('cursor', 'pointer');

    nodeGroup.append('text')
      .text(d => d.name.length > 25 ? d.name.substring(0, 25) + "..." : d.name)
      .attr('text-anchor', 'middle')
      .attr('y', -15)
      .style('font-size', '10px')
      .style('cursor', 'pointer');

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      nodeGroup.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    const zoom = d3.zoom().on("zoom", (event) => {
      svgGroup.attr("transform", event.transform);
    });
    svg.call(zoom);

    function drag(simulation) {
      return d3.drag()
        .on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x; d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        });
    }


  }, [visibleData, selectedNode, searchKeyword]);

  const ToggleList = ({ title, items, onItemClick }) => {
    const [showAll, setShowAll] = useState(false);
    const [hoveredIndex, setHoveredIndex] = useState(null);
    const MAX = 5;

    if (!items || items.length === 0) return null;

    const visibleItems = showAll ? items : items.slice(0, MAX);

    return (
      <div style={{ marginBottom: '10px' }}>
        <strong>{title}:</strong>
        <ul style={{ marginTop: '5px', paddingLeft: '15px' }}>
          {visibleItems.map((paper, idx) => (
            <li
              key={idx}
              onClick={() => onItemClick(paper)}
              onMouseEnter={() => setHoveredIndex(idx)}
              onMouseLeave={() => setHoveredIndex(null)}
              style={{
                cursor: 'pointer',
                color: '#007bff',
                backgroundColor: hoveredIndex === idx ? '#e9f5ff' : 'transparent',
                borderRadius: '5px',
                padding: '3px 5px',
                transition: 'background-color 0.2s'
              }}
            >
              {paper.name || paper.id}
            </li>
          ))}
        </ul>
        {items.length > MAX && (
          <button
            onClick={() => setShowAll(!showAll)}
            style={{
              marginTop: '5px',
              background: 'none',
              border: 'none',
              color: '#555',
              cursor: 'pointer',
              fontSize: '13px',
              textDecoration: 'underline',
            }}
          >
            {showAll ? '간략히 보기' : '더 보기'}
          </button>
        )}
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ flex: 1, position: 'relative' }}>
        <div style={{ position: 'absolute', top: '50px', left: '50%', transform: 'translateX(-50%)', zIndex: 10, background: '#fff', padding: '10px', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)', display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="text"
            value={searchKeyword}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="논문 제목 검색"
            style={{ width: '300px', padding: '8px 12px', borderRadius: '5px', border: '1px solid #ccc' }}
          />
          <button onClick={handleSearch} style={{ padding: '8px 16px', borderRadius: '5px', border: 'none', backgroundColor: '#007bff', color: '#fff', cursor: 'pointer' }}>Search</button>
          {suggestions.length > 0 && (
            <ul style={{ position: 'absolute', top: '50px', background: '#fff', border: '1px solid #ddd', listStyle: 'none', padding: '5px', margin: 0, width: '300px', zIndex: 20, maxHeight: '200px', overflowY: 'auto', borderRadius: '5px', boxShadow: '0 2px 5px rgba(0,0,0,0.15)' }}>
              {suggestions.map(node => (
                <li key={node.id} style={{ padding: '5px', cursor: 'pointer' }} onClick={() => handleNodeFocus(node)}>{node.name}</li>
              ))}
            </ul>
          )}
        </div>
        <svg ref={svgRef} width="1000" height="700" />
      </div>

      <div style={{ background: '#f9f9f9', border: '1px solid #ddd', borderRadius: '10px', padding: '15px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)', width: '400px', margin: '10px', overflowY: 'auto' }}>
        {selectedNode ? (
          <div style={{ backgroundColor: '#fdfdfd', padding: '20px', borderRadius: '10px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)', lineHeight: 1.6 }}>
            <h2 style={{ fontSize: '18px', fontWeight: 'bold', color: '#007bff', marginBottom: '10px' }}>
              {selectedNode.name}
            </h2>

            <p><strong>📅 연도:</strong> {selectedNode.year}</p>
            <p>
              <strong>📈 인용 수:</strong>{' '}
              <span style={{ color: '#007bff', fontWeight: 'bold' }}>{selectedNode.citationCount}</span>
            </p>
            <p>
              <strong>📦 클러스터:</strong>{' '}
              <span style={{ color: '#28a745', fontWeight: 'bold' }}>{selectedNode.group}</span>
            </p>

            <div style={{ marginTop: '15px' }}>
              <strong>👥 저자:</strong>
              <ul style={{ marginTop: '8px', paddingLeft: '20px' }}>
                {selectedNode.authors &&
                  JSON.parse(selectedNode.authors).map((author, idx) => (
                    <li key={idx}>{author}</li>
                  ))}
              </ul>
            </div>

            <div style={{ marginTop: '20px' }}>
              <ToggleList
                title="📚 이 논문이 인용한 논문"
                items={selectedNode.cites}
                onItemClick={handleNodeFocus}
              />
              <ToggleList
                title="🔁 이 논문을 인용한 논문"
                items={selectedNode.citedBy}
                onItemClick={handleNodeFocus}
              />
            </div>
          </div>
        ) : (
          <p style={{ fontStyle: 'italic', color: '#777' }}>노드를 클릭하면 상세정보가 표시됩니다</p>
        )}
      </div>
    </div>
  );
}

export default CitationPage;
