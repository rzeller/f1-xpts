// Sprint scoring is fixed — the JSON omits it for non-sprint weekends but the rules don't change.
const SPRINT_SCORING_FALLBACK = { 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1 };

const SPRINT_WEEKENDS = [
  'China', 'Miami', 'Canada', 'Great Britain', 'Netherlands', 'Singapore',
];

/**
 * Static scoring reference table — always shows both race and sprint tables.
 */
export default function ScoringTable({ scoring }) {
  const raceEntries = Object.entries(scoring.race)
    .sort((a, b) => Number(a[0]) - Number(b[0]));

  // Use sprint scoring from JSON if populated, otherwise use fallback constants.
  const sprintSource = scoring.sprint && Object.keys(scoring.sprint).length > 0
    ? scoring.sprint
    : SPRINT_SCORING_FALLBACK;

  const sprintEntries = Object.entries(sprintSource)
    .sort((a, b) => Number(a[0]) - Number(b[0]));

  return (
    <div>
      <div className="scoring-tables" style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
        <div>
          <h4 style={{ fontFamily: 'var(--font-display)', color: 'var(--text-bright)', marginBottom: 8, fontSize: '0.9rem' }}>
            Grand Prix
          </h4>
          <table style={{ borderCollapse: 'collapse', fontSize: '0.8rem' }}>
            <thead>
              <tr>
                <th style={thStyle}>Pos</th>
                <th style={thStyle}>Pts</th>
              </tr>
            </thead>
            <tbody>
              {raceEntries.map(([pos, pts]) => (
                <tr key={pos}>
                  <td style={tdStyle}>P{pos}</td>
                  <td style={{ ...tdStyle, color: 'var(--text-bright)' }}>{pts}</td>
                </tr>
              ))}
              <tr>
                <td style={tdStyle}>P11+</td>
                <td style={tdStyle}>0</td>
              </tr>
              <tr>
                <td style={tdStyle}>DNF</td>
                <td style={{ ...tdStyle, color: 'var(--red)' }}>{scoring.dnf_penalty}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div>
          <h4 style={{ fontFamily: 'var(--font-display)', color: 'var(--text-bright)', marginBottom: 8, fontSize: '0.9rem' }}>
            Sprint Race
          </h4>
          <table style={{ borderCollapse: 'collapse', fontSize: '0.8rem' }}>
            <thead>
              <tr>
                <th style={thStyle}>Pos</th>
                <th style={thStyle}>Pts</th>
              </tr>
            </thead>
            <tbody>
              {sprintEntries.map(([pos, pts]) => (
                <tr key={pos}>
                  <td style={tdStyle}>P{pos}</td>
                  <td style={{ ...tdStyle, color: 'var(--text-bright)' }}>{pts}</td>
                </tr>
              ))}
              <tr>
                <td style={tdStyle}>P9+</td>
                <td style={tdStyle}>0</td>
              </tr>
              <tr>
                <td style={tdStyle}>DNF</td>
                <td style={{ ...tdStyle, color: 'var(--red)' }}>{scoring.dnf_penalty}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <p style={{ marginTop: 16, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        Sprint weekends (6 of 24 rounds): {SPRINT_WEEKENDS.join(', ')}.
        Players pick the same 5 drivers for both the sprint and the Grand Prix.
        Sprint points stack on top of race points that weekend.
      </p>
    </div>
  );
}

const thStyle = {
  padding: '4px 16px 4px 0',
  textAlign: 'left',
  color: 'var(--text-dim)',
  fontWeight: 500,
  borderBottom: '1px solid var(--border)',
};

const tdStyle = {
  padding: '3px 16px 3px 0',
  color: 'var(--text-muted)',
  borderBottom: '1px solid var(--border)',
};
