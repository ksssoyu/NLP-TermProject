// AuthorYearPage.js

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const availableWindows = [
  '2015_2017', '2016_2018', '2017_2019', '2018_2020',
  '2019_2021', '2020_2022', '2021_2023', '2022_2024'
];

const MAX_AUTHORS_PER_CLUSTER = 60;

function AuthorYearPage() {
  const svgRef = useRef();
  const color = d3.scaleOrdinal(d3.schemeCategory10); // Moved here for reuse

  const [selectedWindow, setSelectedWindow] = useState(availableWindows[0]);
  const [authorData, setAuthorData] = useState({});
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [selectedAuthor, setSelectedAuthor] = useState(null);
  const [authorPapers, setAuthorPapers] = useState([]);
  const [searchKeyword, setSearchKeyword] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [highlightedNodeId, setHighlightedNodeId] = useState(null);
  const [highlightedClusterNodeId, setHighlightedClusterNodeId] = useState(null);
  const simulationRef = useRef(null);
  const [allPapers, setAllPapers] = useState([]);
  const [authorRankings, setAuthorRankings] = useState([]);
  const [rankData, setRankData] = useState({ global_rank: [], cluster_rank: {} });

useEffect(() => {
  fetch(`/coauthorship_ranks_by_window/global_and_cluster_rank_${selectedWindow}.json`)
    .then(res => res.json())
    .then(data => setRankData(data))
    .catch(err => console.error('rank data load failed:', err));
}, [selectedWindow]);


useEffect(() => {
fetch('/top_authors_by_pagerank.json')
    .then(res => res.json())
    .then(data => setAuthorRankings(data))
    .catch(err => console.error('author pagerank load fail:', err));
}, []);

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

  const getAuthorRanks = (authorName) => {
  const global = rankData.global_rank.find(a => a.author === authorName);
  const clusterId = global?.cluster_id?.toString();
  const cluster = clusterId
    ? rankData.cluster_rank[clusterId]?.find(a => a.author === authorName)
    : null;
  return {
    globalRank: global?.rank ?? 'N/A',
    clusterRank: cluster?.rank ?? 'N/A',
    totalWeight: global?.total_weight ?? 'N/A'
  };
};


  const safeSelector = (id) => {
    let cleanedId = id.replace(/[^a-zA-Z0-9_-]/g, '_');
    if (!/^[a-zA-Z_]/.test(cleanedId)) {
      cleanedId = '_' + cleanedId;
    }
    return cleanedId;
  };

  const zoomInToNode = (nodeId) => {
    const svg = d3.select(svgRef.current);
    const width = 1000, height = 700;
    const node = d3.select(`g#${safeSelector(nodeId)}`);
    const d = node.datum();
    if (!d || d.x == null || d.y == null) return;
    if (simulationRef.current) simulationRef.current.stop();
    const scale = 1.5;
    const transform = d3.zoomIdentity.translate(width / 2 - d.x * scale, height / 2 - d.y * scale).scale(scale);
    const zoom = d3.zoom().on('zoom', (event) => {
      svg.select('g').attr('transform', event.transform);
    });
    svg.call(zoom);
    svg.transition().duration(750).call(zoom.transform, transform);
  };

  useEffect(() => {
    setSelectedCluster(null);
    setSelectedAuthor(null);
    setAuthorPapers([]);
    fetch(`/top60_authors_per_cluster/top60_authors_${selectedWindow}.json`)
      .then(res => res.json())
      .then(json => {
        setAuthorData(json.top5_authors_per_cluster || {});
      })
      .catch(() => setAuthorData({}));
  }, [selectedWindow]);

  const handleSearchInputChange = (e) => {
    const value = e.target.value;
    setSearchKeyword(value);
    if (!value || value.length < 2) {
      setSuggestions([]);
      return;
    }
    const results = [];
    Object.entries(authorData).forEach(([cid, authors]) => {
      authors.forEach(a => {
        if (a.author.toLowerCase().includes(value.toLowerCase())) {
          results.push({ ...a, clusterId: cid });
        }
      });
    });
    setSuggestions(results.slice(0, 10));
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      const matched = suggestions[0];
      if (matched) handleNodeSelect(matched);
    }
  };

  const handleNodeSelect = (nodeObj) => {
    if (nodeObj.type === 'author' || nodeObj.author) {
      const authorName = nodeObj.author || nodeObj.id;
      const clusterId = nodeObj.clusterId || Object.entries(authorData).find(([cid, authors]) =>
        authors.some(a => a.author === authorName)
      )?.[0];
      if (!clusterId) return;
      setSelectedCluster(null);
      setSelectedAuthor(authorName);
      setAuthorPapers(getAuthorPapers(authorName));
      setHighlightedNodeId(authorName);
      setHighlightedClusterNodeId(`cluster_${clusterId}`);
    } else if (nodeObj.type === 'cluster') {
      setSelectedCluster(nodeObj.clusterId);
      setSelectedAuthor(null);
      setAuthorPapers([]);
      setHighlightedClusterNodeId(nodeObj.id);
      setHighlightedNodeId(null);
    }
  };

  const handleSuggestionClick = (nodeObj) => {
    setSuggestions([]);
    handleNodeSelect(nodeObj);
};

  const top5AuthorsSet = new Set();

Object.entries(rankData.cluster_rank || {}).forEach(([clusterId, authors]) => {
  authors.slice(0, 5).forEach(a => top5AuthorsSet.add(a.author));
});


  useEffect(() => {
    if (!authorData) return;
    const width = 1000;
    const height = 700;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('viewBox', [0, 0, width, height]);
    const svgGroup = svg.append('g');
    const nodes = [];
    const links = [];

    Object.entries(authorData).forEach(([cid, authors]) => {
      const clusterId = `cluster_${cid}`;
      nodes.push({ id: clusterId, name: `C${cid}`, type: 'cluster', clusterId: cid });
      authors.slice(0, MAX_AUTHORS_PER_CLUSTER).forEach(author => {
        nodes.push({
            id: author.author,
            name: author.author,
            type: 'author',
            clusterId: cid,
            author: author.author
        });
        links.push({ source: clusterId, target: author.author });
        });
    });

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(d => d.type === 'cluster' ? -500 : -100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(d => d.type === 'cluster' ? 40 : 20))
      .alphaDecay(0.05);
    simulationRef.current = simulation;

    const zoom = d3.zoom().on('zoom', (event) => {
      svgGroup.attr('transform', event.transform);
    });
    svg.call(zoom).call(zoom.transform, d3.zoomIdentity);

    const link = svgGroup.append('g')
      .attr('stroke', '#aaa')
      .attr('stroke-opacity', 0.3)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-width', 1);

    const nodeGroup = svgGroup.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('id', d => safeSelector(d.id))
      .call(d3.drag()
        .on('start', event => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          event.subject.fx = event.subject.x;
          event.subject.fy = event.subject.y;
        })
        .on('drag', event => {
          event.subject.fx = event.x;
          event.subject.fy = event.y;
        })
        .on('end', event => {
          if (!event.active) simulation.alphaTarget(0);
          event.subject.fx = null;
          event.subject.fy = null;
        })
      )
      .on('click', (_, d) => {
        handleNodeSelect(d);
      });

    nodeGroup.append('circle')
      .attr('r', d => d.type === 'cluster' ? 25 : 8)
      .attr('fill', d => d.type === 'cluster' ? color(d.clusterId) : top5AuthorsSet.has(d.id) ? '#ff7f0e' : color(d.clusterId))
      .attr('stroke', d => (d.id === highlightedNodeId || d.id === highlightedClusterNodeId) ? 'red' : '#fff')
      .attr('stroke-width', d => (d.id === highlightedNodeId || d.id === highlightedClusterNodeId) ? 3 : 1.5)
      .style('cursor', 'pointer');

    nodeGroup.append('text')
      .text(d => d.name?.length > 20 ? d.name.slice(0, 20) + '…' : d.name)
      .attr('text-anchor', 'middle')
      .attr('y', d => d.type === 'cluster' ? 5 : -12)
      .style('font-size', d => d.type === 'cluster' ? '14px' : '9px')
      .style('font-weight', d => d.type === 'cluster' ? 'bold' : 'normal')
      .style('cursor', 'pointer');

    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      nodeGroup.attr('transform', d => `translate(${d.x},${d.y})`);
    });
  }, [authorData]);

  // 👇 AuthorYearPage.js 파일 안—다른 useEffect 아래 아무 곳에 추가
  useEffect(() => {
    // allPapers 가 준비된 뒤 selectedAuthor 가 지정돼 있으면 논문 재계산
    if (selectedAuthor && allPapers.length) {
        setAuthorPapers(getAuthorPapers(selectedAuthor));
    }
  }, [selectedAuthor, allPapers]);



  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const nodes = svg.selectAll('circle');

    nodes
      .attr('stroke', d => (d.id === highlightedNodeId || d.id === highlightedClusterNodeId) ? 'red' : '#fff')
      .attr('stroke-width', d => (d.id === highlightedNodeId || d.id === highlightedClusterNodeId) ? 3 : 1.5)
      .attr('r', d => d.type === 'cluster' ? 25 : (d.id === highlightedNodeId ? 12 : 8));

    if (highlightedNodeId) zoomInToNode(highlightedNodeId);
  }, [highlightedNodeId, highlightedClusterNodeId]);

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
        <div style={{ flex: 1, position: 'relative' }}>
        <div style={{ position: 'absolute', top: 10, left: 20, zIndex: 10 }}>
            <label style={{ marginRight: 10, fontWeight: 'bold' }}>기간 선택:</label>
            <select value={selectedWindow} onChange={e => setSelectedWindow(e.target.value)}>
            {availableWindows.map(w => <option key={w} value={w}>{w.replace('_', '-')}</option>)}
            </select>
        </div>

        <div style={{ 
            position: 'absolute', 
            top: '50px', 
            left: '50%', 
            transform: 'translateX(-50%)', 
            zIndex: 10, 
            background: '#fff', 
            padding: '10px', 
            borderRadius: '8px', 
            boxShadow: '0 2px 10px rgba(0,0,0,0.1)', 
            display: 'flex', 
            gap: '10px', 
            alignItems: 'center' 
            }}>
            <input
                type="text"
                value={searchKeyword}
                onChange={handleSearchInputChange}  // 또는 handleInputChange
                onKeyDown={handleKeyDown}
                placeholder="저자 이름 검색"
                style={{ width: '300px', padding: '8px 12px', borderRadius: '5px', border: '1px solid #ccc' }}
            />
            <button 
                onClick={() => handleSuggestionClick(suggestions[0])}  // 또는 handleSearch
                style={{ padding: '8px 16px', borderRadius: '5px', border: 'none', backgroundColor: '#007bff', color: '#fff', cursor: 'pointer' }}
            >
                Search
            </button>
            {suggestions.length > 0 && (
                <ul style={{
                position: 'absolute',
                top: '50px',
                background: '#fff',
                border: '1px solid #ddd',
                listStyle: 'none',
                padding: '5px',
                margin: 0,
                width: '300px',
                zIndex: 20,
                maxHeight: '200px',
                overflowY: 'auto',
                borderRadius: '5px',
                boxShadow: '0 2px 5px rgba(0,0,0,0.15)'
                }}>
                {suggestions.map((a, idx) => (
                    <li 
                    key={a.id || idx}
                    style={{ padding: '5px', cursor: 'pointer' }}
                    onClick={() => handleSuggestionClick(a)}
                    >
                    {a.name || a.author}
                    </li>
                ))}
                </ul>
            )}
        </div>
        <svg ref={svgRef} width={1000} height={700} />
        </div>

        <div style={{ background: '#f9f9f9', border: '1px solid #ddd', borderRadius: '10px', padding: '15px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)', width: '400px', margin: '10px', overflowY: 'auto' }}>
        {/* 저자 선택 시: 논문 목록 표시 */}
        {selectedAuthor && (
        <div>
            <h4 style={{ fontSize: '20px', marginBottom: '10px' }}>
            📚 <span style={{ color: '#007bff' }}>{selectedAuthor}</span>의 논문 목록
            </h4>
            
            {top5AuthorsSet.has(selectedAuthor) && (
            <div style={{ 
                backgroundColor: '#e6ffe6', 
                color: '#2e7d32', 
                padding: '6px 12px', 
                borderRadius: '6px', 
                fontWeight: 'bold', 
                marginBottom: '10px' 
            }}>
                ※ 이 저자는 <span style={{ color: '#1b5e20' }}>Top 5 저자</span>입니다!
            </div>
            )}
            {/* 랭크 정보 표시 */}
            <div style={{ marginBottom: '10px' }}>
            <strong>📊 협업 랭킹 정보:</strong>
            <ul style={{ paddingLeft: '20px', fontSize: '14px', lineHeight: 1.6 }}>
                {(() => {
                const ranks = getAuthorRanks(selectedAuthor);
                return (
                    <>
                    <li>Global Rank: <strong>{ranks.globalRank}</strong></li>
                    <li>Cluster Rank: <strong>{ranks.clusterRank}</strong></li>
                    <li>Total Collaboration Weight: <strong>{ranks.totalWeight}</strong></li>
                    </>
                );
                })()}
            </ul>
            </div>


            <ul style={{ 
            listStyle: 'none', 
            padding: 0, 
            margin: 0, 
            maxHeight: '300px', 
            overflowY: 'auto' 
            }}>
            {authorPapers.map((p, idx) => (
                <li key={idx} style={{ 
                backgroundColor: '#fff', 
                padding: '8px 12px', 
                marginBottom: '6px', 
                borderRadius: '6px', 
                boxShadow: '0 1px 3px rgba(0,0,0,0.1)', 
                borderLeft: '4px solid #007bff' 
                }}>
                <div style={{ fontWeight: 'bold' }}>{p.name}</div>
                <div style={{ fontSize: '12px', color: '#666' }}>발표 연도: {p.year}</div>
                </li>
            ))}
            </ul>
        </div>
        )}
        {/* 클러스터 선택 시: 기존 기능 유지 */}
        {!selectedAuthor && selectedCluster && authorData[selectedCluster] ? (
            <>
            <h4 style={{ fontSize: '20px' }}>Cluster {selectedCluster} 저자 목록</h4>
            <div style={{ maxHeight: 250, overflowY: 'auto', paddingLeft: 10, background: '#fff', border: '1px solid #ccc', borderRadius: 6, marginBottom: 10 }}>
                <ol style={{ paddingLeft: 20 }}>
                {authorData[selectedCluster].slice(0, MAX_AUTHORS_PER_CLUSTER).map((a, i) => (
                    <li key={i} style={{ cursor: 'pointer', padding: '4px 8px', borderRadius: '4px' }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#e0f0ff'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                    onClick={() => handleSuggestionClick(a)}>
                    {a.author}
                    </li>
                ))}
                </ol>
            </div>
            <h4 style={{ marginTop: 20 }}>Top 5 협업 저자</h4>
            <ul>
                {authorData[selectedCluster].slice(0, 5).map((a, i) => (
                <li key={i}>{a.author} - 협업 횟수: {a.total_internal_collab_weight}</li>
                ))}
            </ul>
            </>
        ) : null}

        {/* 아무 것도 선택되지 않았을 때 안내 문구 */}
        {!selectedAuthor && !selectedCluster && (
            <p>클러스터 또는 저자를 클릭하면 관련 정보가 표시됩니다.</p>
        )}
        </div>
    </div>
    );
}

export default AuthorYearPage;