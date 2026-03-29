import './CdfChart.css';

/**
 * CDF curve with overlay dots at key market cutoff points.
 */
export default function CdfChart({ distribution, color }) {
  const width = 600;
  const height = 200;
  const pad = { top: 10, right: 20, bottom: 32, left: 45 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  // Compute CDF (exclude DNF for the CDF — just finishing positions)
  const cdf = [];
  let cumulative = 0;
  for (let i = 0; i < 22; i++) {
    cumulative += distribution[i];
    cdf.push(cumulative);
  }

  // Build polyline points
  const points = cdf.map((c, i) => {
    const x = pad.left + ((i + 1) / 22) * plotW;
    const y = pad.top + plotH - c * plotH;
    return `${x},${y}`;
  }).join(' ');

  // Market cutoff positions (1-indexed)
  const markers = [
    { pos: 1, label: 'Win', prob: distribution[0] },
    { pos: 3, label: 'Podium', prob: cdf[2] },
    { pos: 6, label: 'Top 6', prob: cdf[5] },
    { pos: 10, label: 'Top 10', prob: cdf[9] },
  ];

  // Y ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <div className="cdf-chart">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {/* Y gridlines and labels */}
        {yTicks.map(v => {
          const y = pad.top + plotH - v * plotH;
          return (
            <g key={v}>
              <line x1={pad.left} y1={y} x2={width - pad.right} y2={y}
                stroke="var(--border)" strokeWidth={0.5} />
              <text x={pad.left - 6} y={y + 4} textAnchor="end"
                fill="var(--text-dim)" fontSize={10}>
                {(v * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}

        {/* X axis labels */}
        {[1, 5, 10, 15, 20, 22].map(pos => {
          const x = pad.left + (pos / 22) * plotW;
          return (
            <text key={pos} x={x} y={height - 6} textAnchor="middle"
              fill="var(--text-dim)" fontSize={10}>
              P{pos}
            </text>
          );
        })}

        {/* CDF line */}
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth={2}
          opacity={0.8}
        />

        {/* Market cutoff markers */}
        {markers.map(m => {
          const x = pad.left + (m.pos / 22) * plotW;
          const y = pad.top + plotH - m.prob * plotH;
          return (
            <g key={m.pos}>
              {/* Dashed lines to axes */}
              <line x1={pad.left} y1={y} x2={x} y2={y}
                stroke={color} strokeWidth={0.5} strokeDasharray="3,3" opacity={0.4} />
              <line x1={x} y1={y} x2={x} y2={pad.top + plotH}
                stroke={color} strokeWidth={0.5} strokeDasharray="3,3" opacity={0.4} />
              {/* Dot */}
              <circle cx={x} cy={y} r={5} fill={color} stroke="var(--bg)" strokeWidth={2} />
              {/* Label */}
              <text x={x + 8} y={y - 8} fill="var(--text)" fontSize={10}
                fontFamily="var(--font-data)">
                {m.label}: {(m.prob * 100).toFixed(1)}%
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
