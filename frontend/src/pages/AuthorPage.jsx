import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function AuthorPage() {
  const svgRef = useRef();
  const [allData, setAllData] = useState(null);
  const [visibleData, setVisibleData] = useState(null);
  const [searchKeyword, setSearchKeyword] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const simulationRef = useRef(null);
  const [externalClusterMap, setExternalClusterMap] = useState({});
  const [allPapers, setAllPapers] = useState([]);
  
  useEffect(() => {
    fetch('citation_graph_with_cluster_v3.json')
      .then(res => res.json())
      .then(data => setAllPapers(data.nodes || []))
      .catch(err => console.error('논문 데이터 로딩 실패:', err));
  }, []);

  const getAuthorPapers = (authorName) => {
    return allPapers.filter(paper => {
      let authors = [];
      try {
        authors = JSON.parse(paper.authors);
      } catch {
        authors = paper.authors?.split(/,\s*/) || [];
      }
      return authors.includes(authorName);
    });
  };

  useEffect(() => {
    Promise.all([
      fetch('/coauthorship_graph_with_year_cluster.json').then(res => res.json()),
      fetch('/external_cluster_links.json').then(res => res.json())
    ]).then(([graphData, externalMap]) => {
      const nodes = graphData.nodes.map(d => ({ ...d, name: d.id || '' }));
      const nodeById = new Map(nodes.map(d => [d.id, d]));
      const links = graphData.links.map(d => ({
        source: nodeById.get(d.source),
        target: nodeById.get(d.target),
        value: d.weight,
        papers: d.papers
      }));
      setAllData({ nodes, links, nodeById });
      setExternalClusterMap(externalMap);
    });
  }, []);

  const handleInputChange = (e) => {
    const keyword = e.target.value;
    setSearchKeyword(keyword);
    if (!keyword || !allData || !Array.isArray(allData.nodes)) {
      setSuggestions([]);
      return;
    }
    const matched = allData.nodes
      .filter(n => typeof n.name === 'string' && n.name.toLowerCase().includes(keyword.toLowerCase()))
      .slice(0, 10);
    setSuggestions(matched);
  };

  const handleSearch = () => {
    if (!allData) return;
    const matchedNode = allData.nodes.find(n => typeof n.name === 'string' && n.name.toLowerCase() === searchKeyword.toLowerCase());
    if (matchedNode) {
      handleNodeSelection(matchedNode);
    } else {
      alert('해당 저자를 찾을 수 없습니다.');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const handleNodeSelection = (node) => {
    const mainClusterId = node.cluster_id;
    const clusterNodes = allData.nodes.filter(n => n.cluster_id === mainClusterId);
    const clusterGraph = new Map();
    for (const link of allData.links) {
      if (link.source.cluster_id === mainClusterId && link.target.cluster_id === mainClusterId) {
        clusterGraph.set(link.source.id, (clusterGraph.get(link.source.id) || []).concat(link.target.id));
        clusterGraph.set(link.target.id, (clusterGraph.get(link.target.id) || []).concat(link.source.id));
      }
    }
    const bfsQueue = [node.id];
    const visited = new Set([node.id]);
    const bfsClusterIds = [node.id];
    while (bfsQueue.length > 0 && bfsClusterIds.length < 10) {
      const current = bfsQueue.shift();
      const neighbors = clusterGraph.get(current) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          bfsQueue.push(neighbor);
          bfsClusterIds.push(neighbor);
          if (bfsClusterIds.length >= 10) break;
        }
      }
    }
    const sameClusterNodes = bfsClusterIds.map(id => allData.nodeById.get(id));
    const nodeIdsInCluster = new Set(sameClusterNodes.map(n => n.id));

    const neighborScores = new Map();
    for (const link of allData.links) {
      const sourceId = link.source.id;
      const targetId = link.target.id;
      const sourceCluster = link.source.cluster_id;
      const targetCluster = link.target.cluster_id;
      const value = link.value;
      const srcInMain = nodeIdsInCluster.has(sourceId);
      const tgtInMain = nodeIdsInCluster.has(targetId);
      if (srcInMain && !tgtInMain && externalClusterMap[mainClusterId]?.some(c => c.cluster_id === targetCluster)) {
        neighborScores.set(targetId, (neighborScores.get(targetId) || 0) + value);
      } else if (!srcInMain && tgtInMain && externalClusterMap[mainClusterId]?.some(c => c.cluster_id === sourceCluster)) {
        neighborScores.set(sourceId, (neighborScores.get(sourceId) || 0) + value);
      }
    }

    const extraNodesFromCluster = [...neighborScores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([id]) => allData.nodeById.get(id));

    const visualizedIds = new Set([...sameClusterNodes, ...extraNodesFromCluster].map(n => n.id));
    const directlyConnected = allData.links.filter(link =>
      (link.source.id === node.id && !visualizedIds.has(link.target.id)) ||
      (link.target.id === node.id && !visualizedIds.has(link.source.id))
    );
    const bfsDirectNeighbors = directlyConnected
      .map(link => link.source.id === node.id ? link.target.id : link.source.id)
      .slice(0, 10)
      .map(id => allData.nodeById.get(id));

    const finalNodes = [...sameClusterNodes, ...extraNodesFromCluster, ...bfsDirectNeighbors];
    const nodeById = new Map(finalNodes.map(n => [n.id, n]));
    const finalLinks = allData.links
      .filter(link => nodeById.has(link.source.id) && nodeById.has(link.target.id))
      .map(link => ({
        source: nodeById.get(link.source.id),
        target: nodeById.get(link.target.id),
        value: link.value,
        papers: link.papers
      }));

    const collaborations = finalLinks
      .filter(l => l.source.id === node.id || l.target.id === node.id)
      .map(link => ({
        ...link,
        other: link.source.id === node.id ? link.target : link.source
      }));

    const papers = getAuthorPapers(node.name);

    setVisibleData({ nodes: finalNodes, links: finalLinks });
    setSuggestions([]);
    setSearchKeyword(node.name);
    setSelectedNode({ ...node, collaborations, papers });
  };

  const zoomInToNode = (node) => {
    const svg = d3.select(svgRef.current);
    const width = 1000, height = 700;
    if (!node || node.x == null || node.y == null) return;
    if (simulationRef.current) simulationRef.current.stop();

    const scale = 1.5;
    const transform = d3.zoomIdentity
      .translate(width / 2 - node.x * scale, height / 2 - node.y * scale)
      .scale(scale);

    const zoom = d3.zoom().on('zoom', (event) => {
      svg.select('g').attr('transform', event.transform);
    });
    svg.call(zoom);
    svg.transition().duration(750).call(zoom.transform, transform);
  };

  useEffect(() => {
    if (!visibleData) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 1000;
    const height = 700;
    svg.attr('viewBox', [0, 0, width, height]);
    const svgGroup = svg.append('g');

    const allClusters = [...new Set(visibleData.nodes.map(d => d.cluster_id))];
    const color = d3.scaleOrdinal(d3.schemeCategory10).domain(allClusters);

    const simulation = d3.forceSimulation(visibleData.nodes)
      .force('link', d3.forceLink(visibleData.links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-150))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(25));
    simulationRef.current = simulation;

    const link = svgGroup.append('g')
      .attr('stroke', '#aaa')
      .attr('stroke-opacity', 0.3)
      .selectAll('line')
      .data(visibleData.links)
      .join('line')
      .attr('stroke-width', d => Math.sqrt(d.value));

    const nodeGroup = svgGroup.append('g')
      .selectAll('g')
      .data(visibleData.nodes)
      .join('g')
      .call(drag(simulation))
      .on('click', (event, d) => {
        handleNodeSelection(d);
      });

    nodeGroup.append('circle')
      .attr('r', d => selectedNode && d.id === selectedNode.id ? 15 : 10)
      .attr('fill', d => color(d.cluster_id))
      .attr('stroke', d => selectedNode && d.id === selectedNode.id ? 'red' : '#fff')
      .attr('stroke-width', d => selectedNode && d.id === selectedNode.id ? 3 : 1)
      .style('cursor', 'pointer');

    nodeGroup.append('text')
      .text(d => typeof d.name === 'string' && d.name.length > 25 ? d.name.substring(0, 25) + "..." : d.name)
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
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        });
    }
  }, [visibleData]);

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ flex: 1, position: 'relative' }}>
        <div style={{ position: 'absolute', top: '50px', left: '50%', transform: 'translateX(-50%)', zIndex: 10, background: '#fff', padding: '10px', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0,0,0,0.1)', display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="text"
            value={searchKeyword}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="저자 이름 검색"
            style={{ width: '300px', padding: '8px 12px', borderRadius: '5px', border: '1px solid #ccc' }}
          />
          <button onClick={handleSearch} style={{ padding: '8px 16px', borderRadius: '5px', border: 'none', backgroundColor: '#007bff', color: '#fff', cursor: 'pointer' }}>Search</button>
          {suggestions.length > 0 && (
            <ul style={{ position: 'absolute', top: '50px', background: '#fff', border: '1px solid #ddd', listStyle: 'none', padding: '5px', margin: 0, width: '300px', zIndex: 20, maxHeight: '200px', overflowY: 'auto', borderRadius: '5px', boxShadow: '0 2px 5px rgba(0,0,0,0.15)' }}>
              {suggestions.map(node => (
                <li key={node.id} style={{ padding: '5px', cursor: 'pointer' }} onClick={() => handleNodeSelection(node)}>
                  {node.name}
                </li>
              ))}
            </ul>
          )}
        </div>
        <svg ref={svgRef} width="1000" height="700" />
      </div>
      <div style={{ background: '#f9f9f9', border: '1px solid #ddd', borderRadius: '10px', padding: '15px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)', width: '400px', margin: '10px', overflowY: 'auto' }}>
        {selectedNode && (
          <div>
            <h3 style={{ fontSize: '20px', color: '#333', marginBottom: '10px' }}>
              👤 저자 정보
            </h3>

            <div style={{ marginBottom: '12px', fontSize: '14px', lineHeight: '1.6' }}>
              <p><strong>Name:</strong> {selectedNode.id}</p>
              <p><strong>Cluster ID:</strong> {selectedNode.cluster_id}</p>
            </div>

            {/* 논문 목록 */}
            <h4 style={{ fontSize: '16px', margin: '16px 0 8px' }}>📝 작성한 논문</h4>
            <ul style={{
              listStyle: 'none',
              padding: 0,
              margin: 0,
              maxHeight: '300px',
              overflowY: 'auto'
            }}>
              {selectedNode.papers?.map((paper, i) => (
                <li key={i} style={{
                  backgroundColor: '#fff',
                  padding: '8px 12px',
                  marginBottom: '6px',
                  borderRadius: '6px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                  borderLeft: '4px solid #007bff'
                }}>
                  <div style={{ fontWeight: 'bold' }}>{paper.name}</div>
                  <div style={{ fontSize: '12px', color: '#666' }}>발표 연도: {paper.year}</div>
                </li>
              ))}
            </ul>

            {/* 협업 관계 */}
            <h4 style={{ fontSize: '16px', margin: '20px 0 8px' }}>🤝 협업 관계</h4>
            <ul style={{
              listStyle: 'none',
              padding: 0,
              margin: 0,
              maxHeight: '250px',
              overflowY: 'auto'
            }}>
              {selectedNode.collaborations?.map((link, i) => (
                <li
                  key={i}
                  style={{
                    cursor: 'pointer',
                    color: '#007bff',
                    padding: '8px 12px',
                    marginBottom: '5px',
                    borderRadius: '6px',
                    transition: 'background-color 0.2s ease',
                    backgroundColor: '#f5faff'
                  }}
                  onClick={() => handleNodeSelection(link.other)}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#e0f0ff'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f5faff'}
                >
                  <strong>{link.other.name || link.other.id}</strong> — 공동 논문 수: {link.value}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default AuthorPage;
