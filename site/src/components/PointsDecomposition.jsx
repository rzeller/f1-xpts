/**
 * Stacked horizontal bar showing expected points contribution by position.
 * Each segment: P(pos) * points(pos). Red segment for DNF penalty.
 */
export default function PointsDecomposition({ distribution, scoring, dnfPenalty = -10, color }) {
  // Build segments: position 1-10 contribute points, 11-22 contribute 0, DNF contributes penalty
  const segments = [];
  let totalPositive = 0;

  for (let pos = 1; pos <= 10; pos++) {
    const pts = scoring[String(pos)] || 0;
    const contribution = distribution[pos - 1] * pts;
    if (contribution > 0.01) {
      segments.push({ label: `P${pos}`, value: contribution, color, pts });
      totalPositive += contribution;
    }
  }

  const dnfContrib = distribution[22] * dnfPenalty;
  const totalEP = totalPositive + dnfContrib;

  const width = 500;
  const height = 40;
  const pad = { left: 0, right: 0 };
  const plotW = width - pad.left - pad.right;

  // Scale: use total positive as the full bar width
  const scale = totalPositive > 0 ? plotW / totalPositive : 0;

  let xCursor = pad.left;

  return (
    <div className="points-decomp">
      <svg width={width} height={height + 30} viewBox={`0 0 ${width} ${height + 30}`}>
        {/* Positive segments */}
        {segments.map((seg, i) => {
          const w = seg.value * scale;
          const x = xCursor;
          xCursor += w;
          return (
            <g key={seg.label}>
              <rect x={x} y={4} width={w} height={height - 8}
                fill={seg.color} opacity={0.5 + (0.4 * seg.pts / 25)} rx={1} />
              {w > 28 && (
                <text x={x + w / 2} y={height / 2 + 3}
                  textAnchor="middle" fill="var(--text-bright)" fontSize={9}
                  fontFamily="var(--font-data)">
                  {seg.label}
                </text>
              )}
              {/* Value below */}
              {w > 22 && (
                <text x={x + w / 2} y={height + 12}
                  textAnchor="middle" fill="var(--text-dim)" fontSize={9}
                  fontFamily="var(--font-data)">
                  {seg.value.toFixed(1)}
                </text>
              )}
            </g>
          );
        })}

        {/* DNF penalty segment */}
        {dnfContrib < 0 && (
          <g>
            <rect x={xCursor + 4} y={4}
              width={Math.abs(dnfContrib) * scale} height={height - 8}
              fill="var(--red)" opacity={0.6} rx={1} />
            <text x={xCursor + 4 + Math.abs(dnfContrib) * scale / 2} y={height / 2 + 3}
              textAnchor="middle" fill="var(--text-bright)" fontSize={9}
              fontFamily="var(--font-data)">
              DNF
            </text>
            <text x={xCursor + 4 + Math.abs(dnfContrib) * scale / 2} y={height + 12}
              textAnchor="middle" fill="var(--red)" fontSize={9}
              fontFamily="var(--font-data)">
              {dnfContrib.toFixed(1)}
            </text>
          </g>
        )}
      </svg>

      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 4 }}>
        Total: <span style={{ color: totalEP >= 0 ? 'var(--text-bright)' : 'var(--red)' }}>
          {totalEP.toFixed(2)} expected points
        </span>
        {' '}({totalPositive.toFixed(2)} from positions, {dnfContrib.toFixed(2)} DNF penalty)
      </div>
    </div>
  );
}
