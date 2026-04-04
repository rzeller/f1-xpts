import './Nav.css';

export default function Nav({ page, setPage, race, races, selectedRace, onRaceChange }) {
  return (
    <nav className="nav">
      <div className="nav-left">
        <span className="nav-logo">F1 xPts</span>
        {races.length > 1 ? (
          <select
            className="nav-race-select"
            value={selectedRace || ''}
            onChange={e => onRaceChange(e.target.value)}
          >
            {races.map(r => (
              <option key={r.slug} value={r.slug}>
                {r.name}{r.is_sprint ? ' (Sprint)' : ''}
              </option>
            ))}
          </select>
        ) : (
          <span className="nav-race">{race}</span>
        )}
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
