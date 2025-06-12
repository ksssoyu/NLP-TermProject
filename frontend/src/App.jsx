import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import SearchPage from './pages/SearchPage';
import InfluencePage from './pages/InfluencePage';
import TrajectoryPage from './pages/TrajectoryPage';
import CitationPage from './pages/CitationPage';
import AuthorPage from './pages/AuthorPage';
import CitationYearPage  from './pages/CitationYearPage';
import AuthorYearPage  from './pages/AuthorYearPage';
import TopicPage from './pages/TopicPage';
import './App.css'; // css는 외부 파일로 빼자

function App() {
  return (
    <Router>
      <div>
        <div className="navbar">
          <Link className="nav-item" to="/">Search</Link>
          <Link className="nav-item" to="/influence">Influence</Link>
          <div className="nav-dropdown">
          <div className="nav-item">Citation ▾</div>
          <div className="dropdown-content">
            <Link to="/citation" className="dropdown-link">Main</Link>
            <Link to="/topic" className="dropdown-link">Topic</Link>
            <Link to="/citation-year" className="dropdown-link">By Year</Link>
          </div>
        </div>
          <div className="nav-dropdown">
          <div className="nav-item">Author ▾</div>
          <div className="dropdown-content">
            <Link to="/author" className="dropdown-link">Main</Link>
            <Link to="/author-year" className="dropdown-link">By Year</Link>
            <Link to="/trajectory" className="dropdown-link">Trajectory</Link>
          </div>
        </div>
        </div>


        <div className="dashboard-container">
          <h1>NLP Influence Dashboard</h1>

          <Routes>
            <Route path="/" element={<SearchPage />} />
            <Route path="/influence" element={<InfluencePage />} />
            <Route path="/trajectory" element={<TrajectoryPage />} />
            <Route path="/citation" element={<CitationPage />} />
            <Route path="/author" element={<AuthorPage />} />
            <Route path="/citation-year" element={<CitationYearPage />} />
            <Route path="/author-year" element={<AuthorYearPage />} />
            <Route path="/topic" element={<TopicPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
