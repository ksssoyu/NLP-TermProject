import React, { useEffect, useState } from 'react';

function SearchPage() {
  const [papers, setPapers] = useState([]);
  const [keyword, setKeyword] = useState('');
  const [results, setResults] = useState([]);

  useEffect(() => {
    fetch('papers_by_pagerank.json')
      .then(res => res.json())
      .then(data => {
        setPapers(data);
        setResults(filterByKeyword(data, keyword));
      });
  }, []);

  const filterByKeyword = (data, keyword) => {
    const lowerKeyword = keyword.toLowerCase();
    return data
      .filter(p =>
        p.title.toLowerCase().includes(lowerKeyword) ||
        (p.abstract && p.abstract.toLowerCase().includes(lowerKeyword))
      )
      .sort((a, b) => b.pagerank - a.pagerank)
      .slice(0, 10);
  };

  const handleInputChange = (e) => {
    setKeyword(e.target.value);
  };

  const handleSearch = () => {
    if (!keyword.trim()) return;
    const filtered = filterByKeyword(papers, keyword);
    setResults(filtered);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', flexDirection: 'column', alignItems: 'center', height: '100vh', paddingTop: '60px' }}>
      <div style={{
        background: '#fff',
        padding: '12px',
        borderRadius: '8px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        display: 'flex',
        gap: '10px',
        alignItems: 'center',
        width: '600px'
      }}>
        <input
          type="text"
          value={keyword}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="논문 키워드를 입력하세요 (예: GPT)"
          style={{ flex: 1, padding: '8px 12px', borderRadius: '5px', border: '1px solid #ccc' }}
        />
        <button
          onClick={handleSearch}
          style={{ padding: '8px 16px', borderRadius: '5px', border: 'none', backgroundColor: '#007bff', color: '#fff', cursor: 'pointer' }}
        >
          Search
        </button>
      </div>

      <div style=
      {{ marginTop: '40px', width: '80%', maxWidth: '900px',flex: 1,               // 남은 세로공간을 차지
        maxHeight: '70vh',     // 혹은 원하는 고정값(e.g. 500)
        overflowY: 'auto',     // 내용이 넘칠 때만 세로 스크롤
        paddingRight: 8,       // 스크롤바로 인한 글자 가림 방지
        boxShadow: '0 2px 6px rgba(0,0,0,0.08)',
        background: '#f8f9fc',
        borderRadius: 8 }}>
        {results.length > 0 ? (
          <ul>
            {results.map((paper, idx) => (
              <li key={paper.id} style={{ marginBottom: '15px', lineHeight: 1.6 }}>
                <strong>{idx + 1}. {paper.title}</strong><br />
                <span>📅 Year: {paper.year}</span><br />
                <span>👥 Authors: {JSON.parse(paper.authors).join(', ')}</span><br />
                <span>📈 Citation Count: {paper.citationCount}</span><br />
                <span>⭐ PageRank Score: {paper.pagerank.toExponential(3)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p>🔎 키워드와 일치하는 논문이 없습니다.</p>
        )}
      </div>
    </div>
  );
}

export default SearchPage;
