import { useMemo } from 'react';
import './ModelFit.css';

function formatTime(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

function formatRelativeTime(raceDate, generatedAt) {
  if (!raceDate || !generatedAt) return null;
  const race = new Date(raceDate + 'T14:00:00Z'); // approx race start
  const gen = new Date(generatedAt);
  const diffMs = race - gen;
  const diffHrs = Math.round(diffMs / (1000 * 60 * 60));
  if (diffHrs < 0) return `${Math.abs(diffHrs)}h after race start`;
  if (diffHrs < 24) return `${diffHrs}h before race start`;
  const diffDays = Math.round(diffHrs / 24);
  return `${diffDays} day${diffDays !== 1 ? 's' : ''} before race`;
}

// Simple SVG line/area chart
function LossChart({ lossHistory, stepLosses }) {
  if (!lossHistory || lossHistory.length === 0) return null;

  const w = 700, h = 200, pad = { t: 20, r: 20, b: 40, l: 60 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  const maxEval = lossHistory[lossHistory.length - 1].eval;
  const losses = lossHistory.map(d => d.loss);
  const maxLoss = Math.max(...losses);
  const minLoss = Math.min(...losses);
  const yRange = maxLoss - minLoss || 1;

  const toX = (ev) => pad.l + (ev / maxEval) * plotW;
  const toY = (loss) => pad.t + (1 - (loss - minLoss) / yRange) * plotH;

  // Total loss line
  const totalLine = lossHistory.map(d => `${toX(d.eval)},${toY(d.loss)}`).join(' ');
  // Data loss line
  const dataLine = lossHistory.map(d => `${toX(d.eval)},${toY(d.data)}`).join(' ');

  // Step markers
  const stepMarkers = (stepLosses || []).map(s => ({
    x: toX(s.eval),
    loss: s.best_loss,
  }));

  // Y-axis ticks
  const nTicks = 5;
  const yTicks = Array.from({ length: nTicks }, (_, i) => {
    const val = minLoss + (yRange * i) / (nTicks - 1);
    return { val, y: toY(val) };
  });

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="loss-chart">
      {/* Grid lines */}
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={w - pad.r} y1={t.y} y2={t.y} stroke="var(--border)" strokeWidth={0.5} />
          <text x={pad.l - 8} y={t.y + 4} textAnchor="end" fill="var(--text-dim)" fontSize={10}>
            {t.val.toFixed(2)}
          </text>
        </g>
      ))}

      {/* Data loss */}
      <polyline points={dataLine} fill="none" stroke="var(--text-dim)" strokeWidth={1} strokeDasharray="3,3" opacity={0.6} />

      {/* Total loss */}
      <polyline points={totalLine} fill="none" stroke="#22c55e" strokeWidth={1.5} />

      {/* Step markers */}
      {stepMarkers.map((s, i) => (
        <line key={i} x1={s.x} x2={s.x} y1={pad.t} y2={h - pad.b}
          stroke="var(--border-light)" strokeWidth={1} strokeDasharray="4,4" />
      ))}

      {/* Axes */}
      <line x1={pad.l} x2={w - pad.r} y1={h - pad.b} y2={h - pad.b} stroke="var(--border-light)" />
      <line x1={pad.l} x2={pad.l} y1={pad.t} y2={h - pad.b} stroke="var(--border-light)" />

      {/* Labels */}
      <text x={w / 2} y={h - 4} textAnchor="middle" fill="var(--text-dim)" fontSize={11}>Optimizer Evaluations</text>
      <text x={14} y={h / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={11}
        transform={`rotate(-90, 14, ${h / 2})`}>Loss</text>

      {/* Legend */}
      <line x1={pad.l + 10} x2={pad.l + 30} y1={pad.t + 8} y2={pad.t + 8} stroke="#22c55e" strokeWidth={1.5} />
      <text x={pad.l + 34} y={pad.t + 12} fill="var(--text-muted)" fontSize={10}>Total loss</text>
      <line x1={pad.l + 110} x2={pad.l + 130} y1={pad.t + 8} y2={pad.t + 8}
        stroke="var(--text-dim)" strokeWidth={1} strokeDasharray="3,3" />
      <text x={pad.l + 134} y={pad.t + 12} fill="var(--text-muted)" fontSize={10}>Data loss</text>
    </svg>
  );
}

// Scatter plot: model vs observed probability per driver per market
function ResidualsChart({ residuals, teams, drivers }) {
  if (!residuals || residuals.length === 0) return null;

  const w = 700, h = 400, pad = { t: 20, r: 20, b: 50, l: 60 };
  const plotW = w - pad.l - pad.r;
  const plotH = h - pad.t - pad.b;

  // Group by market for coloring
  const marketColors = {
    win: '#ef4444',
    podium: '#f59e0b',
    top6: '#22c55e',
    top10: '#3b82f6',
  };

  const allVals = residuals.flatMap(r => [r.observed, r.model]);
  const maxVal = Math.max(...allVals, 0.01);

  const toX = (v) => pad.l + (v / maxVal) * plotW;
  const toY = (v) => pad.t + (1 - v / maxVal) * plotH;

  // Grid
  const ticks = [0, 0.25, 0.5, 0.75, 1.0].filter(t => t <= maxVal * 1.1);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="residuals-chart">
      {/* Grid */}
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={pad.l} x2={w - pad.r} y1={toY(t)} y2={toY(t)} stroke="var(--border)" strokeWidth={0.5} />
          <line x1={toX(t)} x2={toX(t)} y1={pad.t} y2={h - pad.b} stroke="var(--border)" strokeWidth={0.5} />
          <text x={pad.l - 8} y={toY(t) + 4} textAnchor="end" fill="var(--text-dim)" fontSize={10}>
            {(t * 100).toFixed(0)}%
          </text>
          <text x={toX(t)} y={h - pad.b + 16} textAnchor="middle" fill="var(--text-dim)" fontSize={10}>
            {(t * 100).toFixed(0)}%
          </text>
        </g>
      ))}

      {/* Perfect fit line */}
      <line x1={toX(0)} y1={toY(0)} x2={toX(maxVal)} y2={toY(maxVal)}
        stroke="var(--text-dim)" strokeWidth={1} strokeDasharray="6,4" opacity={0.5} />

      {/* Data points */}
      {residuals.map((r, i) => {
        const color = marketColors[r.market] || 'var(--text-muted)';
        const driverData = drivers.find(d => d.abbr === r.driver);
        const teamColor = driverData ? teams[driverData.team_idx]?.color : color;
        return (
          <g key={i}>
            <circle
              cx={toX(r.observed)} cy={toY(r.model)} r={4}
              fill={color} opacity={0.8}
            />
            {Math.abs(r.residual) > 0.05 && (
              <text x={toX(r.observed) + 6} y={toY(r.model) + 3}
                fill="var(--text-muted)" fontSize={8}>
                {r.driver}
              </text>
            )}
          </g>
        );
      })}

      {/* Axes */}
      <line x1={pad.l} x2={w - pad.r} y1={h - pad.b} y2={h - pad.b} stroke="var(--border-light)" />
      <line x1={pad.l} x2={pad.l} y1={pad.t} y2={h - pad.b} stroke="var(--border-light)" />

      <text x={w / 2} y={h - 4} textAnchor="middle" fill="var(--text-dim)" fontSize={11}>Market Probability</text>
      <text x={14} y={h / 2} textAnchor="middle" fill="var(--text-dim)" fontSize={11}
        transform={`rotate(-90, 14, ${h / 2})`}>Model Probability</text>

      {/* Legend */}
      {Object.entries(marketColors).map(([mkt, color], i) => (
        <g key={mkt}>
          <circle cx={pad.l + 10 + i * 80} cy={pad.t + 8} r={4} fill={color} />
          <text x={pad.l + 18 + i * 80} y={pad.t + 12} fill="var(--text-muted)" fontSize={10}>{mkt}</text>
        </g>
      ))}
    </svg>
  );
}

// Bar chart showing the biggest residuals
function TopResiduals({ residuals }) {
  if (!residuals || residuals.length === 0) return null;

  const sorted = [...residuals]
    .sort((a, b) => Math.abs(b.residual) - Math.abs(a.residual))
    .slice(0, 12);

  const maxAbsResid = Math.max(...sorted.map(r => Math.abs(r.residual)));

  return (
    <div className="top-residuals">
      {sorted.map((r, i) => {
        const pct = (Math.abs(r.residual) / maxAbsResid) * 100;
        const isOver = r.residual > 0;
        return (
          <div key={i} className="resid-row">
            <span className="resid-label">
              {r.driver} <span className="resid-market">{r.market}</span>
            </span>
            <div className="resid-bar-wrap">
              <div
                className={`resid-bar ${isOver ? 'over' : 'under'}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span className={`resid-val ${isOver ? 'over' : 'under'}`}>
              {isOver ? '+' : ''}{(r.residual * 100).toFixed(1)}pp
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function ModelFit({ data }) {
  const fit = data.fit;
  const meta = data.meta;

  if (!fit) {
    return (
      <div className="model-fit">
        <h1>Model Fit</h1>
        <p className="muted">No fit diagnostics available. Re-run the pipeline to generate them.</p>
      </div>
    );
  }

  const relativeTime = formatRelativeTime(meta.date, meta.generated_at);

  // Compute summary stats from residuals
  const residualStats = useMemo(() => {
    if (!fit.residuals || fit.residuals.length === 0) return null;
    const absResids = fit.residuals.map(r => Math.abs(r.residual));
    const rmse = Math.sqrt(fit.residuals.reduce((s, r) => s + r.residual ** 2, 0) / fit.residuals.length);
    const maxResid = Math.max(...absResids);
    const meanResid = absResids.reduce((a, b) => a + b, 0) / absResids.length;
    const worstPoint = fit.residuals.reduce((w, r) =>
      Math.abs(r.residual) > Math.abs(w.residual) ? r : w
    );
    return { rmse, maxResid, meanResid, worstPoint };
  }, [fit.residuals]);

  return (
    <div className="model-fit">
      <header className="fit-header">
        <h1>Model Fit Details</h1>
        <p className="fit-subtitle">{meta.race} &mdash; {meta.date}</p>
      </header>

      {/* ====== OVERVIEW CARDS ====== */}
      <section className="fit-cards">
        <div className="fit-card">
          <div className="card-label">Status</div>
          <div className={`card-value ${fit.converged ? 'positive' : 'negative'}`}>
            {fit.converged ? 'Converged' : 'Did Not Converge'}
          </div>
          <div className="card-detail">{fit.message}</div>
        </div>
        <div className="fit-card">
          <div className="card-label">Final Loss</div>
          <div className="card-value">{fit.final_loss?.toFixed(4)}</div>
          <div className="card-detail">
            RMSE: {residualStats ? (residualStats.rmse * 100).toFixed(1) + 'pp' : '—'}
          </div>
        </div>
        <div className="fit-card">
          <div className="card-label">Optimizer</div>
          <div className="card-value">{fit.method}</div>
          <div className="card-detail">
            {fit.n_evals?.toLocaleString()} evals, {fit.n_steps} steps, {fit.elapsed_seconds ? formatTime(fit.elapsed_seconds) : '—'}
          </div>
        </div>
        <div className="fit-card">
          <div className="card-label">Generated</div>
          <div className="card-value" style={{ fontSize: '0.9rem' }}>
            {new Date(meta.generated_at).toLocaleString()}
          </div>
          <div className="card-detail">{relativeTime}</div>
        </div>
      </section>

      {/* ====== INPUT DATA ====== */}
      <section className="fit-section">
        <h2>Input Data</h2>
        <p>
          Market odds used as constraints for the model fit. Each market provides
          one probability constraint per driver listed.
        </p>
        <div className="market-grid">
          {(fit.market_inputs || []).map(m => (
            <div key={m.market} className="market-card">
              <div className="market-name">{m.market}</div>
              <div className="market-count">{m.n_drivers} drivers</div>
              <div className="market-drivers">
                {m.drivers.map(d => (
                  <span key={d} className="driver-chip">{d}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
        <div className="config-row">
          <span>Devig method: <strong>{meta.devig_method}</strong> (win market), rescaling (placement markets)</span>
          <span>Final sims: <strong>{meta.n_simulations?.toLocaleString()}</strong></span>
          <span>Fit sims/eval: <strong>{fit.n_sims_per_eval?.toLocaleString()}</strong></span>
          <span>Regularization: team={fit.team_reg}, shrink={fit.smoothness_reg}</span>
        </div>
      </section>

      {/* ====== CONVERGENCE ====== */}
      <section className="fit-section">
        <h2>Convergence</h2>
        <p>
          Loss over {fit.n_evals?.toLocaleString()} optimizer evaluations. The green line is total loss
          (data fit + regularization). Dashed line is data loss only. Vertical dashes mark optimizer steps.
        </p>
        <LossChart lossHistory={fit.loss_history} stepLosses={fit.step_losses} />
      </section>

      {/* ====== FIT QUALITY ====== */}
      <section className="fit-section">
        <h2>Fit Quality</h2>
        <p>
          Each dot is one driver in one market. A perfect fit puts every dot on the
          diagonal. Points below the line mean the model underestimates; above means overestimates.
        </p>
        <ResidualsChart residuals={fit.residuals} teams={data.teams} drivers={data.drivers} />
      </section>

      {/* ====== LARGEST RESIDUALS ====== */}
      <section className="fit-section">
        <h2>Largest Residuals</h2>
        <p>
          Where the model deviates most from the market. Blue bars = model underestimates,
          red bars = model overestimates.
          {residualStats && (
            <> Worst fit: <strong>{residualStats.worstPoint.driver}</strong> {residualStats.worstPoint.market}{' '}
            ({(residualStats.worstPoint.residual * 100).toFixed(1)}pp).</>
          )}
        </p>
        <TopResiduals residuals={fit.residuals} />
      </section>
    </div>
  );
}
