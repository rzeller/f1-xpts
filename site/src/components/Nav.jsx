import './Nav.css';

export default function Nav({ page, setPage, race }) {
  return (
    <nav className="nav">
      <div className="nav-left">
        <span className="nav-logo">F1 xPts</span>
        <span className="nav-race">{race}</span>
      </div>
      <div className="nav-links">
        <button
          className={`nav-link ${page === 'dashboard' ? 'active' : ''}`}
          onClick={() => setPage('dashboard')}
        >
          Dashboard
        </button>
        <button
          className={`nav-link ${page === 'methodology' ? 'active' : ''}`}
          onClick={() => setPage('methodology')}
        >
          Methodology
        </button>
      </div>
    </nav>
  );
}
