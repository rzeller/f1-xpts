/**
 * Mini bar chart of position distribution. 22 bars (skips DNF).
 * SVG, ~120px wide x 28px tall.
 */
export default function Sparkline({ distribution, color, width = 90, height = 24 }) {
  // distribution has 23 values: P1-P22 + DNF. Skip DNF (index 22).
  const bars = distribution.slice(0, 22);
  const max = Math.max(...bars);
  if (max === 0) return <svg width={width} height={height} />;

  const barWidth = width / 22;
  const gap = 0.5;

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      {bars.map((p, i) => {
        const barH = (p / max) * (height - 2);
        return (
          <rect
            key={i}
            x={i * barWidth + gap / 2}
            y={height - barH - 1}
            width={Math.max(barWidth - gap, 1)}
            height={barH}
            fill={color}
            opacity={0.7}
            rx={0.5}
          />
        );
      })}
    </svg>
  );
}
