import React, { useEffect, useState } from 'react';
import * as d3 from 'd3';

function TopicPage() {
  const [windows, setWindows] = useState([]);
  const [selectedWindow, setSelectedWindow] = useState(null);
  const [globalStats, setGlobalStats] = useState({});
  const [openClusters, setOpenClusters] = useState({});

  useEffect(() => {
    fetch('analysis_data_final.json')
      .then(res => res.json())
      .then(data => {
        const filtered = data.filter(w => w.topics && w.topics.length > 0);
        setWindows(filtered);

        const stats = {};
        filtered.forEach((w, wi) => {
          w.topics.forEach(topic => {
            topic.keywords.forEach(kw => {
              const key = kw.toLowerCase();
              if (!stats[key]) stats[key] = { freq: 0, windows: new Set() };
              stats[key].freq += 1;
              stats[key].windows.add(wi);
            });
          });
        });

        setGlobalStats(stats);
      })
      .catch(err => console.error('Failed to load analysis data:', err));
  }, []);

  const width = 900;
  const height = 120;

  const stopwords = new Set([
    'model', 'language', 'data', 'system', 'task', 'approach', 'method',
    'performance', 'based', 'learning', 'use', 'using', 'information',
    'research', 'result', 'results', 'different', 'used', 'set'
  ]);

  const isValidKeyword = (w) => {
    if (stopwords.has(w)) return false;
    if (!isNaN(w) && /^[0-9]+$/.test(w)) return false;
    return true;
  };

  const getTopKeywords = (window) => {
    const freq = {};
    const clusterCount = {};

    window.topics.forEach(topic => {
      const seen = new Set();
      topic.keywords.forEach(word => {
        const w = word.toLowerCase();
        if (!isValidKeyword(w)) return;

        freq[w] = (freq[w] || 0) + 1;
        if (!seen.has(w)) {
          clusterCount[w] = (clusterCount[w] || 0) + 1;
          seen.add(w);
        }
      });
    });

    const totalWindows = windows.length;
    const scored = Object.entries(freq).map(([word, count]) => {
      const windowAppearances = globalStats[word]?.windows?.size || 1;
      const idf = Math.log(totalWindows / windowAppearances);
      return { word, score: count * idf };
    });

    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
      .map(k => k.word);
  };

  const getGlobalKeywordScores = (topics) => {
    const freq = {};

    topics.forEach(topic => {
      topic.keywords.forEach(kw => {
        const w = kw.toLowerCase();
        if (!isValidKeyword(w)) return;
        freq[w] = (freq[w] || 0) + 1;
      });
    });

    const totalWindows = windows.length;
    const scored = Object.entries(freq).map(([word, count]) => {
      const windowAppearances = globalStats[word]?.windows?.size || 1;
      const idf = Math.log(totalWindows / windowAppearances);
      return { word, score: count * idf };
    });

    return scored.sort((a, b) => b.score - a.score).slice(0, 20);
  };

  const getTopicKeywordScores = (topic) => {
    const freq = {};
    topic.keywords.forEach(kw => {
      const w = kw.toLowerCase();
      if (!isValidKeyword(w)) return;
      freq[w] = (freq[w] || 0) + 1;
    });

    const totalWindows = windows.length;
    const scored = Object.entries(freq).map(([word, count]) => {
      const windowAppearances = globalStats[word]?.windows?.size || 1;
      const idf = Math.log(totalWindows / windowAppearances);
      return { word, score: count * idf };
    });

    return scored.sort((a, b) => b.score - a.score);
  };

  const getEmergingKeywords = (currentIndex) => {
    if (currentIndex <= 0) return [];

    const prevKeywords = new Set();
    windows[currentIndex - 1].topics.forEach(topic => {
      topic.keywords.forEach(kw => {
        const w = kw.toLowerCase();
        if (isValidKeyword(w)) prevKeywords.add(w);
      });
    });

    const currentKeywords = {};
    windows[currentIndex].topics.forEach(topic => {
      topic.keywords.forEach(kw => {
        const w = kw.toLowerCase();
        if (isValidKeyword(w) && !prevKeywords.has(w)) {
          currentKeywords[w] = (currentKeywords[w] || 0) + 1;
        }
      });
    });

    const totalWindows = windows.length;
    const scored = Object.entries(currentKeywords).map(([word, count]) => {
      const windowAppearances = globalStats[word]?.windows?.size || 1;
      const idf = Math.log(totalWindows / windowAppearances);
      return { word, score: count * idf };
    });

    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, 10)
      .map(k => k.word);
  };

  const toggleCluster = (id) => {
    setOpenClusters(prev => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div style={{ padding: '30px', maxWidth: '1000px', margin: '0 auto' }}>
      <h2>📈 Topics Timeline</h2>

      <svg width={width} height={height} style={{ marginBottom: '30px' }}>
        <line x1={50} y1={height / 2} x2={width - 50} y2={height / 2} stroke="#ccc" strokeWidth={2} />
        {windows.map((win, i) => {
          const x = d3.scaleLinear()
            .domain([windows[0].window_start, windows[windows.length - 1].window_start])
            .range([60, width - 60])(win.window_start);
          return (
            <g key={i} transform={`translate(${x}, ${height / 2})`}>
              <circle
                r={selectedWindow === win ? 8 : 6}
                fill={selectedWindow === win ? '#007bff' : '#555'}
                onClick={() => setSelectedWindow(win)}
                style={{ cursor: 'pointer' }}
              />
              <text
                y={20}
                textAnchor="middle"
                style={{ fontSize: '12px', fill: '#333' }}
              >
                {win.window_start}-{win.window_end}
              </text>
            </g>
          );
        })}
      </svg>

      {selectedWindow && (
        <div style={{ background: '#fff', padding: '20px', borderRadius: '10px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
          <h3>{selectedWindow.window_start} - {selectedWindow.window_end}</h3>

          {/* 박스 스타일 키워드 */}
          <h4>🔥 Top Keywords in This Window</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '20px' }}>
            {getTopKeywords(selectedWindow).map((kw, i) => (
              <span key={i} style={{ background: '#ffe0b2', padding: '6px 10px', borderRadius: '6px', fontSize: '13px' }}>
                {kw}
              </span>
            ))}
          </div>

          {/* 전체 통합 키워드 바 */}
          <h4>🔥 Top Keywords (All Clusters)</h4>
          <div style={{ marginBottom: '20px' }}>
            {getGlobalKeywordScores(selectedWindow.topics).map(({ word, score }, i) => (
              <div key={i} style={{ marginBottom: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '13px' }}>{word}</span>
                  <span style={{ fontSize: '12px', color: '#666' }}>{score.toFixed(2)}</span>
                </div>
                <div style={{ background: '#eee', height: '6px', borderRadius: '3px' }}>
                  <div style={{
                    width: `${Math.min(score * 10, 100)}%`,
                    background: '#007bff',
                    height: '100%'
                  }}></div>
                </div>
              </div>
            ))}
          </div>

          {/* 클러스터별 */}
          <h4>📂 Clusters</h4>
          <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
            {selectedWindow.topics.map((topic, i) => {
              const clusterId = topic.cluster_id;
              const isOpen = openClusters[clusterId] || false;
              const scoredKeywords = getTopicKeywordScores(topic);
              return (
                <li key={i} style={{ marginBottom: '12px' }}>
                  <div
                    onClick={() => toggleCluster(clusterId)}
                    style={{ cursor: 'pointer', fontWeight: 'bold', marginBottom: '6px' }}
                  >
                    ▶ Cluster {clusterId} (size: {topic.size}) {isOpen ? '▲' : '▼'}
                  </div>
                  {isOpen && (
                    <div style={{ marginLeft: '10px' }}>
                      {scoredKeywords.map(({ word, score }, j) => (
                        <div key={j} style={{ marginBottom: '6px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span style={{ fontSize: '13px' }}>{word}</span>
                            <span style={{ fontSize: '12px', color: '#666' }}>{score.toFixed(2)}</span>
                          </div>
                          <div style={{ background: '#eee', height: '6px', borderRadius: '3px' }}>
                            <div style={{
                              width: `${Math.min(score * 10, 100)}%`,
                              background: '#007bff',
                              height: '100%'
                            }}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>

          {/* 이머징 키워드 */}
          <h4 style={{ marginTop: '20px' }}>🌱 Emerging Keywords</h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
            {getEmergingKeywords(windows.indexOf(selectedWindow)).map((kw, i) => (
              <span key={i} style={{ background: '#e0f7fa', padding: '6px 10px', borderRadius: '6px', fontSize: '13px' }}>{kw}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default TopicPage;
