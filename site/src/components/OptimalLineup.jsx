import { useState } from 'react';

/**
 * Top lineups panel.
 *
 * Shows a compact summary table of the top N lineups (one row each),
 * then a full per-driver breakdown for whichever lineup is selected.
 */
export default function OptimalLineup({ lineups, teams }) {
  const [selectedRank, setSelectedRank] = useState(1);
  if (!lineups?.length) return null;

  const selected = lineups.find(l => l.rank === selectedRank) ?? lineups[0];

  return (
    <div className="optimal-lineup">
      <div className="lineup-header">
        <h3 className="lineup-title">Top {lineups.length} Lineups</h3>
        <p className="lineup-subtitle">
          Ordered by expected points including the +{lineups[0].exact_pos_bonus} exact-position
          bonus (earned when a driver finishes in the same slot as their pick number).
          Click a row to see the full breakdown.
        </p>
      </div>

      {/* Summary table — one row per lineup */}
      <table className="lineup-summary-table">
        <thead>
          <tr>
            <th className="ls-rank">#</th>
            {[1, 2, 3, 4, 5].map(s => (
              <th key={s} className="ls-slot">Slot {s}</th>
            ))}
            <th className="ls-num">Base</th>
            <th className="ls-num">+Bonus</th>
            <th className="ls-num">Total</th>
          </tr>
        </thead>
        <tbody>
          {lineups.map(lineup => (
            <tr
              key={lineup.rank}
              className={`ls-row ${lineup.rank === selectedRank ? 'ls-row-selected' : ''}`}
              onClick={() => setSelectedRank(lineup.rank)}
            >
              <td className="ls-rank">{lineup.rank}</td>
              {lineup.picks.map(pick => {
                const tc = teams[pick.team_idx]?.color ?? '#888';
                return (
                  <td key={pick.slot} className="ls-slot">
                    <span className="ls-dot" style={{ background: tc }} />
                    {pick.abbr}
                  </td>
                );
              })}
              <td className="ls-num">{lineup.ep_base_total.toFixed(2)}</td>
              <td className="ls-num ls-bonus">+{lineup.ep_bonus_total.toFixed(2)}</td>
              <td className="ls-num ls-total">{lineup.ep_grand_total.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Full breakdown for selected lineup */}
      <div className="lineup-detail">
        <div className="lineup-detail-header">
          Lineup #{selected.rank} — full breakdown
        </div>
        <table className="lineup-table">
          <thead>
            <tr>
              <th className="lt-col-slot">Pick</th>
              <th className="lt-col-driver">Driver</th>
              <th className="lt-col-num">Base E[pts]</th>
              <th className="lt-col-num">Slot bonus E[pts]</th>
              <th className="lt-col-num">Total E[pts]</th>
            </tr>
          </thead>
          <tbody>
            {selected.picks.map(pick => {
              const teamColor = teams[pick.team_idx]?.color ?? '#888';
              return (
                <tr key={pick.slot}>
                  <td className="lt-col-slot">
                    <span className="pick-badge">Pick {pick.slot}</span>
                  </td>
                  <td className="lt-col-driver">
                    <span className="team-dot" style={{ background: teamColor }} />
                    <span className="driver-name">{pick.name}</span>
                    <span className="driver-abbr">{pick.abbr}</span>
                  </td>
                  <td className="lt-col-num">{pick.ep_base.toFixed(2)}</td>
                  <td className="lt-col-num bonus-cell">+{pick.slot_bonus_ev.toFixed(2)}</td>
                  <td className="lt-col-num total-cell">
                    {(pick.ep_base + pick.slot_bonus_ev).toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot>
            <tr className="lineup-total-row">
              <td colSpan={2} className="lt-total-label">Expected total</td>
              <td className="lt-col-num">{selected.ep_base_total.toFixed(2)}</td>
              <td className="lt-col-num bonus-cell">+{selected.ep_bonus_total.toFixed(2)}</td>
              <td className="lt-col-num total-cell grand-total">
                {selected.ep_grand_total.toFixed(2)}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}
