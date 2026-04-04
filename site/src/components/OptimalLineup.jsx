/**
 * Optimal 5-driver lineup panel.
 *
 * Displays the pick-slot assignments that maximize expected points
 * including the exact-position bonus (+10 when driver finishes in
 * the same slot as their pick number).
 */
export default function OptimalLineup({ lineup, teams }) {
  if (!lineup || !lineup.picks) return null;

  return (
    <div className="optimal-lineup">
      <div className="lineup-header">
        <h3 className="lineup-title">Optimal Lineup</h3>
        <p className="lineup-subtitle">
          Pick these drivers in this order to maximise expected points.
          The +10 exact-position bonus is earned when a driver finishes
          in the same slot as their pick number.
        </p>
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
          {lineup.picks.map(pick => {
            const teamColor = teams[pick.team_idx]?.color ?? '#888';
            const rowTotal = pick.ep_base + pick.slot_bonus_ev;
            return (
              <tr key={pick.slot}>
                <td className="lt-col-slot">
                  <span className="pick-badge">Pick {pick.slot}</span>
                </td>
                <td className="lt-col-driver">
                  <span
                    className="team-dot"
                    style={{ background: teamColor }}
                  />
                  <span className="driver-name">{pick.name}</span>
                  <span className="driver-abbr">{pick.abbr}</span>
                </td>
                <td className="lt-col-num">{pick.ep_base.toFixed(2)}</td>
                <td className="lt-col-num bonus-cell">
                  +{pick.slot_bonus_ev.toFixed(2)}
                </td>
                <td className="lt-col-num total-cell">
                  {rowTotal.toFixed(2)}
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="lineup-total-row">
            <td colSpan={2} className="lt-total-label">Expected total</td>
            <td className="lt-col-num">{lineup.ep_base_total.toFixed(2)}</td>
            <td className="lt-col-num bonus-cell">+{lineup.ep_bonus_total.toFixed(2)}</td>
            <td className="lt-col-num total-cell grand-total">{lineup.ep_grand_total.toFixed(2)}</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
