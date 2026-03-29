import { useState } from 'react';
import './PositionChart.css';

/**
 * Full-size position distribution bar chart with hover tooltips.
 * 23 bars: P1-P22 + DNF.
 */
export default function PositionChart({ distribution, color, scoring }) {
  const [hover, setHover] = useState(null);

  const n = distribution.length; // 23
  const max = Math.max(...distribution);
  const width = 600;
  const height = 220;
  const pad = { top: 10, right: 10, bottom: 32, left: 40 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const barW = plotW / n;
  const gap = 2;

  const labels = [...Array(22).keys()].map(i => `P${i + 1}`);
  labels.push('DNF');

  // Y axis: round max up for nice ticks
  const yMax = Math.ceil(max * 20) / 20; // round to nearest 5%
  const yTicks = [0, yMax / 2, yMax];

  return (
    <div className="position-chart">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {/* Y axis ticks */}
        {yTicks.map(v => {
          const y = pad.top + plotH - (v / yMax) * plotH;
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

        {/* Bars */}
        {distribution.map((p, i) => {
          const barH = yMax > 0 ? (p / yMax) * plotH : 0;
          const x = pad.left + i * barW + gap / 2;
          const y = pad.top + plotH - barH;
          const isDnf = i === 22;

          return (
            <g key={i}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              <rect
                x={x} y={y}
                width={Math.max(barW - gap, 2)}
                height={barH}
                fill={isDnf ? 'var(--red)' : color}
                opacity={hover === i ? 0.9 : 0.65}
                rx={1}
              />
              {/* X label — show every other to avoid crowding */}
              {(i % 2 === 0 || i === 22) && (
                <text
                  x={x + (barW - gap) / 2} y={height - 6}
                  textAnchor="middle" fill="var(--text-dim)" fontSize={9}
                >
                  {labels[i]}
                </text>
              )}
            </g>
          );
        })}

        {/* Tooltip */}
        {hover !== null && (
          <g>
            <rect
              x={pad.left + hover * barW + barW / 2 - 36}
              y={pad.top + plotH - (distribution[hover] / yMax) * plotH - 28}
              width={72} height={22} rx={4}
              fill="var(--bg-active)" stroke="var(--border-light)" strokeWidth={0.5}
            />
            <text
              x={pad.left + hover * barW + barW / 2}
              y={pad.top + plotH - (distribution[hover] / yMax) * plotH - 13}
              textAnchor="middle" fill="var(--text-bright)" fontSize={11}
              fontFamily="var(--font-data)"
            >
              {labels[hover]}: {(distribution[hover] * 100).toFixed(2)}%
            </text>
          </g>
        )}
      </svg>

      {/* Points context */}
      {scoring && hover !== null && hover < 22 && (
        <div className="position-chart-context">
          {scoring[String(hover + 1)]
            ? `${scoring[String(hover + 1)]} pts if ${labels[hover]}`
            : `0 pts (outside top 10)`}
        </div>
      )}
      {hover === 22 && (
        <div className="position-chart-context negative">-20 pts DNF penalty</div>
      )}
    </div>
  );
}
