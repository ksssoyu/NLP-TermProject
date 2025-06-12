import React, { useEffect, useState } from 'react';

function InfluencePage() {
  const [topPapers, setTopPapers] = useState([]);
  const [topAuthors, setTopAuthors] = useState([]);

  useEffect(() => {
    fetch('papers_by_pagerank.json')
      .then(res => res.json())
      .then(data => setTopPapers(data.slice(0, 10)))
      .catch(err => console.error('Failed to load paper influence data:', err));

    fetch('top10_authors_by_pagerank.json')
      .then(res => res.json())
      .then(data => setTopAuthors(data.slice(0, 10)))
      .catch(err => console.error('Failed to load author influence data:', err));
  }, []);

  return (
    <div style={{ padding: '40px', maxWidth: '1000px', margin: '0 auto' }}>
      <h2 style={{ fontSize: '24px', marginBottom: '20px', color: '#333' }}>
        📄 Top Influential Papers (Citation-based PageRank)
      </h2>
      <div style={{ borderRadius: '8px', overflow: 'hidden', boxShadow: '0 2px 10px rgba(0,0,0,0.08)', marginBottom: '50px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead style={{ backgroundColor: '#f0f4f8' }}>
            <tr>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Title</th>
              <th style={thStyle}>Year</th>
              <th style={thStyle}>Citations</th>
              <th style={thStyle}>PageRank</th>
              <th style={thStyle}>Authors</th>
            </tr>
          </thead>
          <tbody>
            {topPapers.map((paper, idx) => (
              <tr key={paper.id} style={idx % 2 === 0 ? rowEven : rowOdd}>
                <td style={tdStyle}>{idx + 1}</td>
                <td style={{ ...tdStyle, fontWeight: 'bold' }}>{paper.title}</td>
                <td style={tdStyle}>{paper.year}</td>
                <td style={tdStyle}>{paper.citationCount ?? 0}</td>
                <td style={tdStyle}>{(paper.pagerank ?? 0).toExponential(2)}</td>
                <td style={tdStyle}>
                  {Array.isArray(paper.authors)
                    ? paper.authors.join(', ')
                    : paper.authors}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 style={{ fontSize: '24px', marginBottom: '20px', color: '#333' }}>
        👤 Top Influential Authors (Co-authorship-based PageRank)
      </h2>
      <div style={{ borderRadius: '8px', overflow: 'hidden', boxShadow: '0 2px 10px rgba(0,0,0,0.08)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead style={{ backgroundColor: '#f0f4f8' }}>
            <tr>
              <th style={thStyle}>#</th>
              <th style={thStyle}>Author</th>
              <th style={thStyle}>PageRank</th>
              <th style={thStyle}>Cluster ID</th>
            </tr>
          </thead>
          <tbody>
            {topAuthors.map((author, idx) => (
              <tr key={author.author} style={idx % 2 === 0 ? rowEven : rowOdd}>
                <td style={tdStyle}>{idx + 1}</td>
                <td style={{ ...tdStyle, fontWeight: 'bold' }}>{author.author}</td>
                <td style={tdStyle}>{(author.pagerank ?? 0).toExponential(2)}</td>
                <td style={tdStyle}>{author.cluster_id ?? 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// 🔧 스타일 정의
const thStyle = {
  padding: '12px 10px',
  textAlign: 'left',
  fontWeight: '600',
  fontSize: '14px',
  color: '#333',
  borderBottom: '1px solid #ccc'
};

const tdStyle = {
  padding: '10px',
  fontSize: '14px',
  color: '#444',
  verticalAlign: 'top'
};

const rowEven = {
  backgroundColor: '#ffffff'
};

const rowOdd = {
  backgroundColor: '#f9f9f9'
};

export default InfluencePage;
