/**
 * Horizontal bar showing expected points relative to maxEP.
 * Handles negative values: positive goes right (team color), negative goes left (red).
 */
export default function BarGauge({ value, maxEP, minEP, color, width = 80, height = 14 }) {
  const range = maxEP - Math.min(minEP, 0);
  const zeroX = Math.min(minEP, 0) < 0 ? (Math.abs(minEP) / range) * width : 0;

  let barX, barW;
  if (value >= 0) {
    barX = zeroX;
    barW = (value / range) * width;
  } else {
    barW = (Math.abs(value) / range) * width;
    barX = zeroX - barW;
  }

  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      {/* Zero line when there are negatives */}
      {minEP < 0 && (
        <line
          x1={zeroX} y1={0} x2={zeroX} y2={height}
          stroke="var(--border-light)" strokeWidth={1}
        />
      )}
      <rect
        x={barX}
        y={2}
        width={Math.max(barW, 1)}
        height={height - 4}
        fill={value >= 0 ? color : 'var(--red)'}
        opacity={0.65}
        rx={2}
      />
    </svg>
  );
}
