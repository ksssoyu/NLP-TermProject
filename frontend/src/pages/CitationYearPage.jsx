import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function CitationYearPage() {
  const svgRef = useRef();
  const [yearlyEmergingKeywords, setYearlyEmergingKeywords] = useState({});
  const [allYearData, setAllYearData] = useState({});
  const [selectedYear, setSelectedYear] = useState('2015-2017');
  const [currentData, setCurrentData] = useState(null);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [searchKeyword, setSearchKeyword] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [citationLookup, setCitationLookup] = useState({});   // id → {authors, year}
  const [errorMessage, setErrorMessage] = useState('');
  const [emergingKeywords, setEmergingKeywords] = useState([]);
  const simulationRef = useRef(null);
  const availableYears = [
    { label: '2015-2017', file: 'cluster_2015_2017.json' },
    { label: '2016-2018', file: 'cluster_2016_2018.json' },
    { label: '2017-2019', file: 'cluster_2017_2019.json' },
    { label: '2018-2020', file: 'cluster_2018_2020.json' },
    { label: '2019-2021', file: 'cluster_2019_2021.json' },
    { label: '2020-2022', file: 'cluster_2020_2022.json' },
    { label: '2021-2023', file: 'cluster_2021_2023.json' },
    { label: '2022-2024', file: 'cluster_2022_2024.json' }
  ];
  const [pagerankLookup, setPagerankLookup] = useState({});  // 논문 ID → PageRank
  const [topPapersByCluster, setTopPapersByCluster] = useState({});
  const [globalTop10Set, setGlobalTop10Set] = useState(new Set());


  useEffect(() => {
    fetch('/papers_by_pagerank.json')
      .then(res => res.json())
      .then(json => {
        const map = {};
        json.forEach(p => {
          map[p.id] = p.pagerank;
        });
        setPagerankLookup(map);
      })
      .catch(err => console.error('Failed to load PageRank data:', err));
  }, []);

  useEffect(() => {
    // citation_graph_with_cluster_v3.json 파일 로드
    fetch('/citation_graph_with_cluster_v3.json')
      .then(res => res.json())
      .then(json => {
        // nodes 배열 → id 기반 메타데이터 맵으로 변환
        const map = {};
        json.nodes.forEach(n => {
          let authors = [];
          if (Array.isArray(n.authors)) authors = n.authors;
          else if (typeof n.authors === 'string') {
            try { authors = JSON.parse(n.authors); }      // '["A","B"]'
            catch { authors = n.authors.split(/,\s*/); }  // 'A, B, C'
          }
          map[n.id] = {
            authors,
            year: n.year,
            citationCount: n.citationCount ?? 0
          };
        });
        setCitationLookup(map);
      })
      .catch(err => console.error('Failed to load citation data:', err));
    
    // 모든 연도별 JSON 파일을 로드
    Promise.all(availableYears.map(year => 
      fetch(`/cluster_with_keywords_by_year/${year.file}`)
        .then(res => res.json())
        .catch(err => {
          console.error(`Failed to load ${year.file}:`, err);
          return null;
        })
    ))
    .then(dataArray => {
      const yearDataMap = {};
      availableYears.forEach((year, idx) => {
        if (dataArray[idx]) {
          yearDataMap[year.label] = dataArray[idx];
        }
      });
      setAllYearData(yearDataMap);
      // --- Emerging Topic Analysis ---
      const keywordTimeline = {};
      availableYears.forEach(({ label }) => {
        const yearData = yearDataMap[label];
        if (!yearData) return;
        Object.values(yearData.cluster_keywords).forEach(keywords => {
          keywords.forEach(keyword => {
            if (!keywordTimeline[keyword]) keywordTimeline[keyword] = {};
            keywordTimeline[keyword][label] = (keywordTimeline[keyword][label] || 0) + 1;
          });
        });
      });
      // Build year-specific emerging keyword list
      const yearToKeywordMap = {};
      availableYears.forEach(({ label }, idx) => {
        const currentData = yearDataMap[label];
        if (!currentData) return;

        const prev = availableYears[idx - 1]?.label;
        const prevData = yearDataMap[prev];

        const currentKeywordCounts = {};
        Object.values(currentData.cluster_keywords).forEach(kwList => {
          kwList.forEach(k => currentKeywordCounts[k] = (currentKeywordCounts[k] || 0) + 1);
        });

        const prevKeywordCounts = {};
        if (prevData) {
          Object.values(prevData.cluster_keywords).forEach(kwList => {
            kwList.forEach(k => prevKeywordCounts[k] = (prevKeywordCounts[k] || 0) + 1);
          });
        }

        const trendList = Object.entries(currentKeywordCounts)
          .map(([k, curCount]) => {
            const prevCount = prevKeywordCounts[k] || 0;
            return { keyword: k, prevCount, curCount };
          })
          .filter(({ prevCount, curCount }) => curCount >= 2 && curCount > prevCount * 1.5)
          .sort((a, b) => b.curCount - a.curCount)
          .slice(0, 8)
          .map(t => t.keyword);

        yearToKeywordMap[label] = trendList;
      });
      setYearlyEmergingKeywords(yearToKeywordMap);


      const latestYears = availableYears.slice(-3).map(y => y.label);
      const earlierYears = availableYears.slice(0, -3).map(y => y.label);

      const trends = Object.entries(keywordTimeline).map(([keyword, counts]) => {
        const past = earlierYears.reduce((sum, year) => sum + (counts[year] || 0), 0);
        const recent = latestYears.reduce((sum, year) => sum + (counts[year] || 0), 0);
        return { keyword, past, recent };
      });

      const trending = trends
        .filter(({ past, recent }) => recent >= 3 && recent > past * 1.5)
        .sort((a, b) => b.recent - a.recent)
        .slice(0, 10)
        .map(t => t.keyword);

      setEmergingKeywords(trending);

      // 첫 번째 사용 가능한 데이터로 초기화
      const firstAvailableYear = availableYears.find(year => yearDataMap[year.label]);
      if (firstAvailableYear) {
        setSelectedYear(firstAvailableYear.label);
        setCurrentData(yearDataMap[firstAvailableYear.label]);
      }
    })
    .catch(err => {
      console.error('Failed to load year data:', err);
    });
  }, []);

  const handleYearChange = (year) => {
    setSelectedYear(year);
    setCurrentData(allYearData[year]);
    setSelectedCluster(null);
    setSelectedPaper(null);
    setSearchKeyword('');
    setSuggestions([]);
    setErrorMessage('');
  };

  const handleInputChange = (e) => {
    const keyword = e.target.value;
    setSearchKeyword(keyword);

    if (keyword.length < 2 || !currentData) {
      setSuggestions([]);
      return;
    }

    const matched = [];
    Object.values(currentData.cluster_details).forEach(papers => {
      papers.forEach(paper => {
        if (paper.title.toLowerCase().includes(keyword.toLowerCase()) && matched.length < 10) {
          matched.push(paper);
        }
      });
    });
    setSuggestions(matched);
  };

  const handleSuggestionClick = (paper) => {
    const enrichedPaper = {
      ...paper,
      name: paper.title  // ← 이 한 줄만 추가하면 해결
    };
    setSelectedPaper(enrichedPaper);
    setSearchKeyword(paper.title);
    setSuggestions([]);

    // 클러스터 찾기
    Object.entries(currentData.cluster_details).forEach(([clusterId, papers]) => {
      if (papers.some(p => p.id === paper.id)) {
        setSelectedCluster(clusterId);
      }
    });
  };

  const handleSearch = () => {
    if (!currentData) return;

    // 논문 제목이 검색된 연도에 존재하는지 확인
    let foundPaper = null;
    let isPaperFoundInCurrentYear = false;

    Object.entries(currentData.cluster_details).forEach(([clusterId, papers]) => {
      const match = papers.find(p => p.title.toLowerCase() === searchKeyword.toLowerCase());
      if (match) {
        foundPaper = match;
        isPaperFoundInCurrentYear = true;
      }
    });

    if (isPaperFoundInCurrentYear) {
      handleSuggestionClick(foundPaper);
    } else {
      setErrorMessage('해당 연도에 존재하지 않는 논문입니다.');
      setSelectedPaper(null);
      setSelectedCluster(null);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const safeSelector = (id) => {
    // CSS 선택자 규칙에 맞게 변환 (예: . 대신 _로 변환)
    // 1. Replace all characters that are not alphanumeric, hyphens, or underscores with an underscore.
    let cleanedId = id.replace(/[^a-zA-Z0-9_-]/g, "_");

    // 2. Ensure the ID starts with a letter or an underscore.
    //    If it starts with a number or hyphen (which is allowed later but not at the start),
    //    prepend an underscore.
    if (!/^[a-zA-Z_]/.test(cleanedId)) {
      cleanedId = "_" + cleanedId;
    }

    return cleanedId;
  };

  const zoomInToNode = (nodeId) => {
    const svg = d3.select(svgRef.current);
    const width = 1000, height = 700;

    const node = d3.select(`g#${safeSelector(nodeId)}`);
    const d = node.datum();
    if (!d || d.x == null || d.y == null) return;

    // ① 시뮬레이션이 돌고 있으면 멈춤
    if (simulationRef.current) simulationRef.current.stop();

    const scale = 1.5;
    const transform = d3.zoomIdentity
        .translate(width / 2 - d.x * scale, height / 2 - d.y * scale)
        .scale(scale);

    const zoom = d3.zoom().on('zoom', (event) => {
      svg.select('g').attr('transform', event.transform);
    });
    svg.call(zoom);

    // ② 한 번만 부드럽게 이동
    svg.transition()
      .duration(750)
      .call(zoom.transform, transform);
  };


  useEffect(() => {
    if (!currentData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 1000;
    const height = 700;
    svg.attr('viewBox', [0, 0, width, height]);

    const svgGroup = svg.append("g");

    // 클러스터 데이터를 노드와 링크로 변환
    const nodes = [];
    const links = [];
    
    Object.entries(currentData.cluster_details).forEach(([clusterId, papers]) => {
      // 클러스터 중심 노드
      const centerNode = {
        id: `cluster_${clusterId}`,
        name: `Cluster ${clusterId}`,
        type: 'cluster',
        clusterId: clusterId,
        keywords: currentData.cluster_keywords[clusterId] || [],
        paperCount: papers.length
      };
      nodes.push(centerNode);

      // 논문 노드들
      papers.forEach(paper => {
        nodes.push({
          id: paper.id,
          name: paper.title,
          type: 'paper',
          clusterId: clusterId,
          // authors가 문자열이면 JSON.parse 해서 배열로 넣기
          detail: {
            ...paper,
          authors: (() => {
            if (Array.isArray(paper.authors)) return paper.authors;
            if (typeof paper.authors === 'string') {
              try { return JSON.parse(paper.authors); }   // '["A","B"]' 형태
              catch { return paper.authors.split(/,\s*/); } // 'A, B, C' 형태
            }
            return [];                                     // 정보 없을 때
          })(),

          },
        });
        
        // 클러스터 중심과 논문을 연결
        links.push({
          source: `cluster_${clusterId}`,
          target: paper.id
        });
      });
    });

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    const allPapers = Object.values(currentData.cluster_details).flat();

    const globalTop10 = [...allPapers]
      .map(p => ({ id: p.id, pagerank: pagerankLookup[p.id] || 0 }))
      .sort((a, b) => b.pagerank - a.pagerank)
      .slice(0, 10)
      .map(p => p.id);

    const globalTop10SetLocal = new Set(globalTop10); // ✅ local로 먼저 만들고

    setGlobalTop10Set(globalTop10SetLocal);           // state에도 저장

    /* ✅ 여기서부터 새로 추가 */
    const drawGraph = (data) => {
      if (!data) return;

      // === (원래 useEffect 안에 있던 그래프 생성 코드) ===
      const svg = d3.select(svgRef.current);
      svg.selectAll('*').remove();

      const width = 1000, height = 700;
      svg.attr('viewBox', [0, 0, width, height]);
      const svgGroup = svg.append('g');

      /* 1. nodes, links 구성 */
      const nodes = [];
      const links = [];
      Object.entries(data.cluster_details).forEach(([clusterId, papers]) => {
        nodes.push({
          id: `cluster_${clusterId}`,
          name: `Cluster ${clusterId}`,
          type: 'cluster',
          clusterId,
          keywords: data.cluster_keywords[clusterId] || [],
          paperCount: papers.length,
        });
        papers.forEach(paper => {
          nodes.push({
            id: paper.id,
            name: paper.title,
            type: 'paper',
            clusterId,
            detail: paper,
          });
          links.push({ source: `cluster_${clusterId}`, target: paper.id });
        });
      });

      /* 2. forceSimulation */
      const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(80))
        .force('charge', d3.forceManyBody().strength(d => d.type === 'cluster' ? -500 : -100))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide(d => d.type === 'cluster' ? 40 : 20))
        .alphaDecay(0.05);
      simulationRef.current = simulation;

      /* 3. link, nodeGroup, drag, tick 콜백 (기존 코드 그대로) */
      const color = d3.scaleOrdinal(d3.schemeCategory10);

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
        .call(drag(simulation))
        .on('click', (event, d) => {
          if (d.type === 'cluster') {
            setSelectedCluster(d.clusterId);
            setSelectedPaper(null);
          } else {
            setSelectedCluster(d.clusterId);
            setSelectedPaper(d);   // 줌은 선택-effect가 수행
          }
        });

      nodeGroup.append('circle')
        .attr('r', d => {
          if (d.type === 'cluster') return 25;
          if (topPapersByCluster[d.clusterId]?.has(d.id)) return 12;  // Top10 논문 크기 증가
          return 8;
        })
        .attr('fill', d => color(d.clusterId))
        .attr('stroke', d => {
          if (d.type === 'paper' && topPapersByCluster[d.clusterId]?.has(d.id)) return 'gold';
          if (d.type === 'cluster' && selectedCluster === d.clusterId) return 'red';
          if (d.type === 'paper' && selectedPaper?.id === d.id) return 'red';
          return '#fff';
        })
        .attr('stroke-width', d => 
          (selectedPaper?.id === d.id || selectedCluster === d.clusterId)
            ? 3 : 1.5)
        .style('cursor', 'pointer');

      nodeGroup.append('text')
        .text(d => d.type === 'cluster'
          ? `C${d.clusterId}`
          : (d.name.length > 20 ? d.name.slice(0, 20) + '…' : d.name))
        .attr('text-anchor', 'middle')
        .attr('y', d => d.type === 'cluster' ? 5 : -12)
        .style('font-size', d => d.type === 'cluster' ? '14px' : '8px')
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

      const zoom = d3.zoom().on('zoom', (event) => {
        svgGroup.attr('transform', event.transform);
      });
      svg.call(zoom);

      function drag(simulation) {
        function dragstarted(event, d) {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x; d.fy = d.y;
        }
        function dragged(event, d) {
          d.fx = event.x; d.fy = event.y;
        }
        function dragended(event, d) {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null; d.fy = null;
        }
        return d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended);
      }
    };  // ← drawGraph 끝

    if (selectedPaper) {
      // 현재 데이터의 노드 목록에 선택된 논문이 존재하는지 확인
      const paperExistsInCurrentData = nodes.some(node => node.id === selectedPaper.id);
      if (paperExistsInCurrentData) {
        zoomInToNode(selectedPaper.id);
      }
    }

    const computedTop = {};
    Object.entries(currentData.cluster_details).forEach(([clusterId, papers]) => {
      const sorted = [...papers]
        .map(p => ({ ...p, pagerank: pagerankLookup[p.id] || 0 }))
        .sort((a, b) => b.pagerank - a.pagerank)
        .slice(0, 10);
      computedTop[clusterId] = new Set(sorted.map(p => p.id));
    });
    setTopPapersByCluster(computedTop);

    drawGraph(currentData);   // 새 함수(아래 ①)로 분리
  }, [currentData]);

  useEffect(() => {
    const svg = d3.select(svgRef.current);

    svg.selectAll('circle')
        .attr('stroke', d => {
          if (d.type === 'paper') {
            if (selectedPaper?.id === d.id) return 'red'; // 1. 선택된 논문
            if (globalTop10Set.has(d.id)) return 'green'; // 2. 전체 Top10
            if (topPapersByCluster[d.clusterId]?.has(d.id)) return 'gold'; // 3. 클러스터 Top10
          }
          if (d.type === 'cluster' && selectedCluster === d.clusterId) return 'red';
          return '#fff';
        })
        .attr('stroke-width', d => {
          if (selectedPaper?.id === d.id || selectedCluster === d.clusterId) return 3;
          if (globalTop10Set.has(d.id)) return 2.5;
          if (topPapersByCluster[d.clusterId]?.has(d.id)) return 2;
          return 1.5;
        })
       .attr('r', d =>
         d.type === 'cluster'
           ? 25
           : (selectedPaper && d.id === selectedPaper.id ? 12 : 8));

    if (selectedPaper) zoomInToNode(selectedPaper.id);
  }, [selectedPaper, selectedCluster, globalTop10Set, topPapersByCluster]);

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <div style={{ flex: 1, position: 'relative' }}>
        {/* 연도 선택 */}
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '20px',
          zIndex: 10,
          background: '#fff',
          padding: '10px',
          borderRadius: '8px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
        }}>
          <label style={{ marginRight: '10px', fontWeight: 'bold' }}>연도 선택:</label>
          <select 
            value={selectedYear} 
            onChange={(e) => handleYearChange(e.target.value)}
            style={{ padding: '5px', borderRadius: '5px', border: '1px solid #ccc' }}
          >
            {availableYears.map(year => (
              <option key={year.label} value={year.label}>{year.label}</option>
            ))}
          </select>
        </div>
        {/* Emerging Highlight Toggle */}
        <div style={{
          position: 'absolute',
          top: '90px',
          left: '20px',
          zIndex: 10,
          background: '#fff',
          padding: '10px',
          borderRadius: '8px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
        }}>
        </div>


        {/* 검색 */}
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
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="논문 제목 검색"
            style={{ width: '300px', padding: '8px 12px', borderRadius: '5px', border: '1px solid #ccc' }}
          />
          <button onClick={handleSearch} style={{
            padding: '8px 16px',
            borderRadius: '5px',
            border: 'none',
            backgroundColor: '#007bff',
            color: '#fff',
            cursor: 'pointer'
          }}>Search</button>

          {errorMessage && (
            <p style={{ color: 'red', fontSize: '14px', marginTop: '10px' }}>{errorMessage}</p>
          )}

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
              {suggestions.map(paper => (
                <li key={paper.id} style={{ padding: '5px', cursor: 'pointer' }}
                  onClick={() => handleSuggestionClick(paper)}>
                  {paper.title}
                </li>
              ))}
            </ul>
          )}
        </div>

        <svg ref={svgRef} width="1000" height="700" />
      </div>

      {/* 사이드바 */}
      <div style={{ background: '#f9f9f9', border: '1px solid #ddd', borderRadius: '10px', padding: '15px', boxShadow: '0 2px 5px rgba(0,0,0,0.1)', width: '400px', margin: '10px', overflowY: 'auto' }}>

        {selectedPaper ? (
        <>
          <h4 style={{ fontSize: '20px', marginBottom: '10px', color: '#007bff' }}>{selectedPaper.name}</h4>

          {(() => {
            const meta = citationLookup[selectedPaper.id] || {};
            const rank = (() => {
              const clusterPapers = currentData.cluster_details[selectedCluster] || [];
              const sorted = [...clusterPapers]
                .map(p => ({ id: p.id, pagerank: pagerankLookup[p.id] || 0 }))
                .sort((a, b) => b.pagerank - a.pagerank);
              return sorted.findIndex(p => p.id === selectedPaper.id) + 1;  // 0-based → 1-based
            })();
            return (
              <div style={{ backgroundColor: '#fdfdfd', padding: '10px 12px', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.08)', marginBottom: '15px' }}>
                <p><strong>📅 출판연도:</strong> {meta.year ?? '정보 없음'}</p>
                <p><strong>👥 저자:</strong></p>
                  {meta.authors ? (
                    <ul style={{ paddingLeft: '20px', marginTop: '4px' }}>
                      {(Array.isArray(meta.authors) ? meta.authors : JSON.parse(meta.authors)).map((author, idx) => (
                        <li key={idx}>{author}</li>
                      ))}
                    </ul>
                  ) : (
                    <p>정보 없음</p>
                  )}
                <p><strong>🔁 인용 수:</strong> {meta.citationCount ?? '정보 없음'}</p>
                {rank && (
                  <p><strong>🏆 PageRank 순위 (클러스터 내):</strong> {rank}위</p>
                )}
                {(() => {
                  const allPapers = Object.values(currentData.cluster_details).flat();
                  const globalSorted = [...allPapers]
                    .map(p => ({ id: p.id, pagerank: pagerankLookup[p.id] || 0 }))
                    .sort((a, b) => b.pagerank - a.pagerank);
                  const globalRank = globalSorted.findIndex(p => p.id === selectedPaper.id) + 1;
                  return globalRank > 0 ? (
                    <p><strong>🌍 PageRank 순위 (전체 윈도우):</strong> {globalRank}위</p>
                  ) : null;
                })()}
              </div>
            );
          })()}
          <p style={{ fontWeight: 'bold', marginBottom: 5 }}>📌 클러스터 ID: {selectedCluster}</p>

          <div style={{ marginTop: '15px' }}>
            <h4 style={{ fontSize: '16px', marginBottom: '8px' }}>🔑 클러스터 {selectedCluster}의 키워드</h4>
            <ul style={{ paddingLeft: '20px', lineHeight: '1.6' }}>
              {currentData.cluster_keywords[selectedCluster]?.map((kw, i) =>
                <li key={i}>{kw}</li>)}
            </ul>
          </div>
        </>
      ) : selectedCluster && currentData ? (
        <>
          <h4 style={{ fontSize: '18px', color: '#007bff' }}>📚 클러스터 {selectedCluster} 내 논문들</h4>
          <div
            style={{
              maxHeight: 300,
              overflowY: 'auto',
              paddingLeft: 20,
              border: '1px solid #eee',
              borderRadius: 6,
              background: '#fff',
              padding: 8,
              marginTop: 5,
              boxShadow: '0 1px 3px rgba(0,0,0,0.08)'
            }}
          >
            <ul style={{ margin: 0, paddingLeft: '20px', listStyleType: 'disc' }}>
              {currentData.cluster_details[selectedCluster]?.map((paper, idx) => (
                <li
                  key={idx}
                  onClick={() => handleSuggestionClick(paper)}
                  style={{
                    marginBottom: '8px',
                    cursor: 'pointer',
                    padding: '6px 10px',
                    borderRadius: '6px',
                    transition: 'background-color 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#e9f5ff';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  {paper.title}
                </li>
              ))}
            </ul>
          </div>

          {currentData.cluster_keywords[selectedCluster]?.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <h4 style={{ fontSize: '16px', marginBottom: '15px' }}>🔑 클러스터 {selectedCluster}의 키워드</h4>
              <ul style={{
              listStyle: 'none',
              padding: 0,
              margin: 0,
              maxHeight: '300px',
              overflowY: 'auto'
            }}>
                {currentData.cluster_keywords[selectedCluster].map((kw, idx) => (
                  <li key={idx} style={{
                  backgroundColor: '#fff',
                  padding: '8px 12px',
                  marginBottom: '6px',
                  borderRadius: '6px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                  borderLeft: '4px solid #007bff'
                }}>{kw}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      ) : (
        <p style={{ fontStyle: 'italic', color: '#666' }}>논문 또는 클러스터를 클릭하여 상세 정보를 확인하세요.</p>
      )}
      </div>
  </div>
  );
}

export default CitationYearPage;