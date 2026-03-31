import './LambdaChart.css';

/**
 * Horizontal dot plot of driver strength (lambda) parameters.
 * Dots colored by team. Teammates connected by a thin line.
 */
export default function LambdaChart({ drivers, teams, selectedIdx, onSelect }) {
  const width = 600;
  const rowH = 22;
  const height = drivers.length * rowH + 20;
  const pad = { left: 100, right: 30, top: 10, bottom: 10 };
  const plotW = width - pad.left - pad.right;

  // Sort by lambda descending for display
  const sorted = [...drivers].map((d, i) => ({ ...d, origIdx: i }))
    .sort((a, b) => b.lambda - a.lambda);

  const lambdas = drivers.map(d => d.lambda);
  const minL = Math.min(...lambdas);
  const maxL = Math.max(...lambdas);
  const range = maxL - minL || 1;
  const toX = (l) => pad.left + ((l - minL) / range) * plotW;

  // Group teammates
  const teamDrivers = {};
  sorted.forEach((d, i) => {
    if (!teamDrivers[d.team_idx]) teamDrivers[d.team_idx] = [];
    teamDrivers[d.team_idx].push({ ...d, sortIdx: i });
  });

  return (
    <div className="lambda-chart">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {/* Zero line */}
        {minL < 0 && maxL > 0 && (
          <line
            x1={toX(0)} y1={pad.top} x2={toX(0)} y2={height - pad.bottom}
            stroke="var(--border-light)" strokeWidth={0.5} strokeDasharray="4,4"
          />
        )}

        {/* Teammate connectors */}
        {Object.values(teamDrivers).map(pair => {
          if (pair.length !== 2) return null;
          const [a, b] = pair;
          const color = teams[a.team_idx].color;
          const yA = pad.top + a.sortIdx * rowH + rowH / 2;
          const yB = pad.top + b.sortIdx * rowH + rowH / 2;
          const xA = toX(a.lambda);
          const xB = toX(b.lambda);
          const selectedTeamIdx = selectedIdx != null ? drivers[selectedIdx]?.team_idx : null;
          const pairSelected = a.team_idx === selectedTeamIdx;
          return (
            <line key={`${a.abbr}-${b.abbr}`}
              x1={xA} y1={yA} x2={xB} y2={yB}
              stroke={color} strokeWidth={pairSelected ? 2 : 1} opacity={pairSelected ? 0.7 : 0.25}
            />
          );
        })}

        {/* Dots and labels */}
        {sorted.map((d, i) => {
          const y = pad.top + i * rowH + rowH / 2;
          const x = toX(d.lambda);
          const color = teams[d.team_idx].color;
          const selectedTeamIdx = selectedIdx != null ? drivers[selectedIdx]?.team_idx : null;
          const isSelected = d.origIdx === selectedIdx;
          const isTeammate = !isSelected && selectedTeamIdx != null && d.team_idx === selectedTeamIdx;

          return (
            <g key={d.abbr}
              className="lambda-row"
              onClick={() => onSelect && onSelect(d.origIdx)}
              style={{ cursor: onSelect ? 'pointer' : 'default' }}
            >
              {/* Hover bg */}
              <rect x={0} y={y - rowH / 2} width={width} height={rowH}
                fill="transparent" className="lambda-hover-bg" />

              {/* Name */}
              <text x={pad.left - 8} y={y + 4} textAnchor="end"
                fill={isSelected ? 'var(--text-bright)' : isTeammate ? '#888' : 'var(--text-muted)'}
                fontSize={11} fontFamily="var(--font-data)" fontWeight={isSelected || isTeammate ? 600 : 400}>
                {d.abbr}
              </text>

              {/* Dot */}
              <circle cx={x} cy={y} r={isSelected || isTeammate ? 6 : 4.5}
                fill={color} opacity={isSelected || isTeammate ? 1 : 0.7}
                stroke={isSelected ? 'var(--text-bright)' : isTeammate ? '#888' : 'none'} strokeWidth={1.5}
              />

              {/* Lambda value */}
              <text x={x + 10} y={y + 4}
                fill={isSelected ? 'var(--text-bright)' : isTeammate ? '#888' : 'var(--text-dim)'}
                fontSize={10} fontFamily="var(--font-data)" fontWeight={isSelected || isTeammate ? 600 : 400}>
                {d.lambda.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
