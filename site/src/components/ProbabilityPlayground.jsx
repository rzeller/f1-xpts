import { useState, useMemo, useCallback } from 'react';
import { simulateRaces, computeExpectedPoints } from '../lib/simulation';
import './ProbabilityPlayground.css';

/**
 * Interactive probability playground.
 * Users adjust a driver's lambda (strength) via slider and see:
 * - Real-time position distribution changes
 * - Comparison of model probabilities vs. market constraints
 * - Fit error metric
 */
export default function ProbabilityPlayground({ data }) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const driver = data.drivers[selectedIdx];
  const teamColor = data.teams[driver.team_idx].color;

  // Lambda slider: range from -3 to +3, default = driver's fitted lambda
  const fittedLambda = driver.lambda;
  const [lambdaOffset, setLambdaOffset] = useState(0);
  const adjustedLambda = fittedLambda + lambdaOffset;

  // Market constraints from the fitted data
  const marketConstraints = [
    { label: 'Win', key: 'p_win', cutoff: 1 },
    { label: 'Podium', key: 'p_podium', cutoff: 3 },
    { label: 'Top 6', key: 'p_top6', cutoff: 6 },
    { label: 'Top 10', key: 'p_top10', cutoff: 10 },
  ];

  // Re-simulate with adjusted lambda
  const simResult = useMemo(() => {
    const logLambdas = data.drivers.map((d, i) =>
      i === selectedIdx ? adjustedLambda : d.lambda
    );
    const pDnfs = data.drivers.map(d => d.p_dnf);
    const teamIndices = data.drivers.map(d => d.team_idx);
    const correlation = data.meta?.correlation || null;
    const allDists = simulateRaces(logLambdas, pDnfs, 8000, 77, teamIndices, correlation);
    return allDists[selectedIdx];
  }, [data.drivers, selectedIdx, adjustedLambda, data.meta]);

  // Compute CDF from sim result
  const cdf = useMemo(() => {
    const c = [];
    let cum = 0;
    for (let i = 0; i < 22; i++) {
      cum += simResult[i];
      c.push(cum);
    }
    return c;
  }, [simResult]);

  // Model probabilities at market cutoffs
  const modelProbs = useMemo(() => ({
    p_win: simResult[0],
    p_podium: cdf[2],
    p_top6: cdf[5],
    p_top10: cdf[9],
  }), [simResult, cdf]);

  // Fit error: sum of squared differences between model and market
  const fitError = useMemo(() => {
    let err = 0;
    for (const mc of marketConstraints) {
      const diff = modelProbs[mc.key] - driver[mc.key];
      err += diff * diff;
    }
    return Math.sqrt(err);
  }, [modelProbs, driver]);

  // Expected points
  const ep = useMemo(() =>
    computeExpectedPoints(simResult, data.scoring.race, data.scoring.dnf_penalty),
    [simResult, data.scoring]
  );

  const handleDriverChange = useCallback((e) => {
    setSelectedIdx(Number(e.target.value));
    setLambdaOffset(0);
  }, []);

  const handleSliderChange = useCallback((e) => {
    setLambdaOffset(Number(e.target.value));
  }, []);

  const handleReset = useCallback(() => {
    setLambdaOffset(0);
  }, []);

  // Chart dimensions
  const width = 600;
  const height = 240;
  const pad = { top: 14, right: 10, bottom: 34, left: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const n = 23;
  const barW = plotW / n;
  const gap = 2;

  const max = Math.max(...simResult);
  const yMax = Math.ceil(max * 20) / 20;
  const yTicks = [0, yMax / 2, yMax];
  const labels = [...Array(22).keys()].map(i => `P${i + 1}`);
  labels.push('DNF');

  // Also get the original (fitted) distribution for overlay
  const originalDist = driver.position_distribution;

  return (
    <div className="playground">
      <div className="playground-controls">
        <div className="playground-driver-select">
          <label htmlFor="pg-driver">Driver</label>
          <select id="pg-driver" value={selectedIdx} onChange={handleDriverChange}>
            {data.drivers.map((d, i) => (
              <option key={d.abbr} value={i}>{d.name} ({d.abbr})</option>
            ))}
          </select>
        </div>

        <div className="playground-slider-group">
          <label htmlFor="pg-lambda">
            Strength (log {'\u03BB'}): <span className="playground-value" style={{ color: teamColor }}>
              {adjustedLambda.toFixed(2)}
            </span>
            {lambdaOffset !== 0 && (
              <span className="playground-offset">
                ({lambdaOffset > 0 ? '+' : ''}{lambdaOffset.toFixed(2)} from fitted)
              </span>
            )}
          </label>
          <div className="playground-slider-row">
            <input
              id="pg-lambda"
              type="range"
              min={-2}
              max={2}
              step={0.05}
              value={lambdaOffset}
              onChange={handleSliderChange}
              className="playground-slider"
            />
            <button onClick={handleReset} className="playground-reset" title="Reset to fitted value">
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* Distribution chart with overlay */}
      <div className="playground-chart">
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

          {/* Bars: adjusted distribution */}
          {simResult.map((p, i) => {
            const barH = yMax > 0 ? (p / yMax) * plotH : 0;
            const x = pad.left + i * barW + gap / 2;
            const y = pad.top + plotH - barH;
            const isDnf = i === 22;

            return (
              <g key={i}>
                <rect
                  x={x} y={y}
                  width={Math.max(barW - gap, 2)}
                  height={barH}
                  fill={isDnf ? 'var(--red)' : teamColor}
                  opacity={0.6}
                  rx={1}
                />
                {/* X label */}
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

          {/* Overlay: original fitted distribution as line */}
          {originalDist && (
            <polyline
              points={originalDist.map((p, i) => {
                const x = pad.left + i * barW + barW / 2;
                const y = pad.top + plotH - (p / yMax) * plotH;
                return `${x},${y}`;
              }).join(' ')}
              fill="none"
              stroke="var(--text-dim)"
              strokeWidth={1.5}
              strokeDasharray="4,3"
              opacity={0.7}
            />
          )}
        </svg>
        <div className="playground-legend">
          <span className="playground-legend-item">
            <span className="playground-legend-bar" style={{ background: teamColor, opacity: 0.6 }}></span>
            Adjusted
          </span>
          <span className="playground-legend-item">
            <span className="playground-legend-line"></span>
            Fitted model
          </span>
        </div>
      </div>

      {/* Market constraints comparison */}
      <div className="playground-constraints">
        <h4>Model vs. Market Probabilities</h4>
        <div className="playground-constraint-grid">
          {marketConstraints.map(mc => {
            const market = driver[mc.key];
            const model = modelProbs[mc.key];
            const diff = model - market;
            const pctDiff = market > 0 ? (diff / market * 100) : 0;

            return (
              <div key={mc.key} className="playground-constraint-card">
                <div className="playground-constraint-label">{mc.label}</div>
                <div className="playground-constraint-values">
                  <div className="playground-constraint-row">
                    <span className="playground-constraint-tag">Market</span>
                    <span className="playground-constraint-num">{(market * 100).toFixed(1)}%</span>
                  </div>
                  <div className="playground-constraint-row">
                    <span className="playground-constraint-tag" style={{ color: teamColor }}>Model</span>
                    <span className="playground-constraint-num" style={{ color: teamColor }}>
                      {(model * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className={`playground-constraint-diff ${Math.abs(diff) > 0.03 ? 'large-diff' : ''}`}>
                    {diff > 0 ? '+' : ''}{(diff * 100).toFixed(1)}pp
                  </div>
                </div>
                {/* Visual bar comparison */}
                <div className="playground-constraint-bars">
                  <div className="playground-bar-bg">
                    <div className="playground-bar-market" style={{ width: `${Math.min(market * 100, 100)}%` }}></div>
                    <div className="playground-bar-model" style={{
                      width: `${Math.min(model * 100, 100)}%`,
                      borderColor: teamColor
                    }}></div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="playground-summary">
          <div className="playground-summary-item">
            <span className="playground-summary-label">E[Points]</span>
            <span className="playground-summary-value" style={{ color: ep >= 0 ? 'var(--text-bright)' : 'var(--red)' }}>
              {ep.toFixed(2)}
            </span>
            <span className="playground-summary-sub">
              (fitted: {driver.ep_race.toFixed(2)})
            </span>
          </div>
          <div className="playground-summary-item">
            <span className="playground-summary-label">Fit Error (RMSE)</span>
            <span className={`playground-summary-value ${fitError > 0.05 ? 'negative' : fitError < 0.01 ? 'positive' : ''}`}>
              {fitError.toFixed(4)}
            </span>
            <span className="playground-summary-sub">
              {fitError < 0.01 ? 'Excellent fit' : fitError < 0.03 ? 'Good fit' : fitError < 0.08 ? 'Moderate' : 'Poor fit'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
