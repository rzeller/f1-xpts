/**
 * Static scoring reference table.
 */
export default function ScoringTable({ scoring }) {
  const raceEntries = Object.entries(scoring.race)
    .sort((a, b) => Number(a[0]) - Number(b[0]));

  const sprintEntries = Object.entries(scoring.sprint || {})
    .sort((a, b) => Number(a[0]) - Number(b[0]));

  return (
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

      {sprintEntries.length > 0 && (
        <div>
          <h4 style={{ fontFamily: 'var(--font-display)', color: 'var(--text-bright)', marginBottom: 8, fontSize: '0.9rem' }}>
            Sprint
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
                <td style={tdStyle}>P{Number(sprintEntries[sprintEntries.length - 1]?.[0] || 8) + 1}+</td>
                <td style={tdStyle}>0</td>
              </tr>
              <tr>
                <td style={tdStyle}>DNF</td>
                <td style={{ ...tdStyle, color: 'var(--red)' }}>{scoring.dnf_penalty}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
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
