import { useState, useMemo } from 'react';
import DriverRow from '../components/DriverRow';
import ModelFit from './ModelFit';
import './Dashboard.css';

const SORT_MODES = [
  { key: 'ep_total', label: 'E[pts]', desc: true },
  { key: 'p_top10', label: 'Safest', desc: true },
  { key: 'p_win', label: 'Upside', desc: true },
  { key: 'std_dev', label: 'Low var', desc: false },
];

export default function Dashboard({ data }) {
  const [sortKey, setSortKey] = useState('ep_total');
  const [showModelFit, setShowModelFit] = useState(false);

  const sortMode = SORT_MODES.find(m => m.key === sortKey);

  const sorted = useMemo(() => {
    const arr = [...data.drivers];
    arr.sort((a, b) => sortMode.desc ? b[sortKey] - a[sortKey] : a[sortKey] - b[sortKey]);
    return arr;
  }, [data.drivers, sortKey, sortMode.desc]);

  const maxEP = Math.max(...data.drivers.map(d => d.ep_total));
  const minEP = Math.min(...data.drivers.map(d => d.ep_total));

  const date = new Date(data.meta.date + 'T00:00:00').toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });

  return (
    <div className="dashboard">
      <header className="dash-header">
        <div className="dash-title-row">
          <h1>{data.meta.race}</h1>
          {data.meta.is_sprint && <span className="sprint-badge">Sprint Weekend</span>}
          {data.meta.run_type === 'pre_qualifying' && <span className="picks-ready-badge">Ready for picks</span>}
        </div>
        <p className="dash-subtitle">{date}</p>
      </header>

      <div className="sort-controls">
        <span className="sort-label">Sort by</span>
        {SORT_MODES.map(mode => (
          <button
            key={mode.key}
            className={`sort-btn ${sortKey === mode.key ? 'active' : ''}`}
            onClick={() => setSortKey(mode.key)}
          >
            {mode.label}
          </button>
        ))}
      </div>

      <div className="table-wrap">
        <table className="driver-table">
          <thead>
            <tr>
              <th className="col-rank">#</th>
              <th className="col-driver">Driver</th>
              <th className="col-sparkline">Distribution</th>
              <th className="col-ep">E[pts]</th>
              <th className="col-pct">Win</th>
              <th className="col-pct">Podium</th>
              <th className="col-pct">Top 10</th>
              <th className="col-pct">DNF</th>
              <th className="col-num">Std Dev</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((driver, i) => (
              <DriverRow
                key={driver.abbr}
                driver={driver}
                rank={i + 1}
                teamColor={data.teams[driver.team_idx].color}
                maxEP={maxEP}
                minEP={minEP}
              />
            ))}
          </tbody>
        </table>
      </div>

      <footer className="dash-footer">
        <span>Model: Plackett-Luce | Devig: {data.meta.devig_method} | {data.meta.n_simulations.toLocaleString()} sims | Fit loss: {data.meta.fit_loss?.toFixed(5) ?? '—'}{data.meta.fit_converged === false && ' (not converged)'}</span>
        <span>Updated {new Date(data.meta.generated_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
      </footer>

      <button
        className="model-fit-toggle"
        onClick={() => setShowModelFit(v => !v)}
      >
        <span className={`chevron ${showModelFit ? 'open' : ''}`}>&#9662;</span>
        Model Fit Diagnostics
      </button>
      {showModelFit && <ModelFit data={data} />}
    </div>
  );
}
