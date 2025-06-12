import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

function TrajectoryPage() {
  const svgRef = useRef();
  const [allData, setAllData] = useState(null);
  const [paperData, setPaperData] = useState([]);
  const [coauthorStats, setCoauthorStats] = useState([]);
  const [searchKeyword, setSearchKeyword] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [selectedAuthor, setSelectedAuthor] = useState(null);

  useEffect(() => {
    fetch('/coauthorship_graph_with_year.json')
      .then(res => res.json())
      .then(json => {
        const nodes = json.nodes.map(d => ({ ...d, name: d.id || '' }));
        const links = json.links.map(d => ({
          source: d.source,
          target: d.target,
          weight: d.weight,
          years: d.years
        }));
        setAllData({ nodes, links });
      });

    fetch('/citation_graph_with_cluster_v3.json')
      .then(res => res.json())
      .then(json => {
        const papers = json.nodes.map(d => ({
          ...d,
          authors: JSON.parse(d.authors || '[]')
        }));
        setPaperData(papers);
      });
  }, []);

  const handleInputChange = (e) => {
    const keyword = e.target.value;
    setSearchKeyword(keyword);
    if (!keyword || !allData) {
      setSuggestions([]);
      return;
    }
    const matched = allData.nodes
      .filter(n => n.name && typeof n.name === 'string' && n.name.toLowerCase().includes(keyword.toLowerCase()))
      .slice(0, 10);
    setSuggestions(matched);
  };

  const handleSearch = () => {
    if (!allData) return;
    const author = allData.nodes.find(n => n.name && n.name.toLowerCase() === searchKeyword.toLowerCase());
    if (!author) {
      alert('저자를 찾을 수 없습니다.');
      return;
    }
    computeTrajectory(author.name);
    setSelectedAuthor(author.name);
    setSuggestions([]);
  };

  const computeTrajectory = (authorName) => {
    if (!allData) return;

    const yearlySum = {};
    allData.links.forEach(link => {
      const years = typeof link.years === 'string' ? link.years.split(',').map(y => parseInt(y.trim())).filter(y => !isNaN(y)) : [];
      if (link.source === authorName || link.target === authorName) {
        years.forEach(year => {
          yearlySum[year] = (yearlySum[year] || 0) + link.weight;
        });
      }
    });

    const data = Object.entries(yearlySum)
      .sort((a, b) => a[0] - b[0])
      .map(([year, sum]) => ({ year: +year, coauthorshipSum: sum }));

    setCoauthorStats(data);
    drawLineChart(data);
  };

  const drawLineChart = (data) => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 1000;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 40, left: 60 };

    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    svg.attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // ✅ g를 margin.left, margin.top 만큼 이동시켜 차트를 중앙에 정렬
    const svgGroup = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // ✅ 내부 좌표 기준 (0~chartWidth)
    const x = d3.scaleLinear()
      .domain(d3.extent(data, d => d.year))
      .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.coauthorshipSum)]).nice()
      .range([chartHeight, 0]);

    // 선 그래프
    if (data.length > 1) {
      const line = d3.line()
        .x(d => x(d.year))
        .y(d => y(d.coauthorshipSum));

      svgGroup.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', 'steelblue')
        .attr('stroke-width', 2)
        .attr('d', line);
    }

    // 데이터 점
    svgGroup.append('g')
      .selectAll('circle')
      .data(data)
      .join('circle')
      .attr('cx', d => x(d.year))
      .attr('cy', d => y(d.coauthorshipSum))
      .attr('r', 4)
      .attr('fill', 'orange');

    // x축
    svgGroup.append('g')
      .attr('transform', `translate(0, ${chartHeight})`)
      .call(d3.axisBottom(x).tickFormat(d3.format('d')));

    // y축
    svgGroup.append('g')
      .call(d3.axisLeft(y));
  };

  const getPapersByAuthor = (author) => {
    return paperData.filter(p => p.authors.includes(author));
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '50px 20px',
      backgroundColor: '#fff', // 흰색 배경을 명확히 유지
      minHeight: '100vh'
    }}>
      <div style={{ flex: 1, position: 'relative' }}>
        <div style={{
          position: 'absolute',
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
            placeholder="저자 이름 검색"
            style={{ width: '300px', padding: '8px 12px', borderRadius: '5px', border: '1px solid #ccc' }}
          />
          <button onClick={handleSearch} style={{ padding: '8px 16px', borderRadius: '5px', border: 'none', backgroundColor: '#007bff', color: '#fff', cursor: 'pointer' }}>Search</button>
          {suggestions.length > 0 && (
            <ul style={{ position: 'absolute', top: '50px', background: '#fff', border: '1px solid #ddd', listStyle: 'none', padding: '5px', margin: 0, width: '300px', zIndex: 20, maxHeight: '200px', overflowY: 'auto', borderRadius: '5px', boxShadow: '0 2px 5px rgba(0,0,0,0.15)' }}>
              {suggestions.map(node => (
                <li key={node.id} style={{ padding: '5px', cursor: 'pointer' }}
                    onClick={() => {
                      setSearchKeyword(node.name);
                      setSuggestions([]);
                      setSelectedAuthor(node.name);
                      computeTrajectory(node.name);
                    }}>
                  {node.name}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div style={{
          width: '1000px',
          marginTop: '150px'
        }}>
          <svg ref={svgRef} width="1000" height="400" />
        </div>
        {selectedAuthor && (
          <div style={{
            width: '1000px',
            marginTop: '30px'
          }}>
            <h3>📄 논문 목록 - {selectedAuthor}</h3>
            <div style={{
              maxHeight: '500px',
              overflowY: 'auto',
              paddingRight: '10px',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
              border: '1px solid #ddd',
              borderRadius: '8px',
              padding: '10px',
              backgroundColor: '#fdfdfd'
            }}>
              {getPapersByAuthor(selectedAuthor).map(p => (
                <div key={p.id} style={{ borderBottom: '1px solid #eee', paddingBottom: '6px' }}>
                  <div style={{ fontSize: '14px', fontWeight: 'bold' }}>{p.name}</div>
                  <div style={{ fontSize: '13px', color: '#666' }}>{p.year}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TrajectoryPage;
