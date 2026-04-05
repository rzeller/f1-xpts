import { useState, useMemo } from 'react';
import PositionChart from '../components/PositionChart';
import CdfChart from '../components/CdfChart';
import LambdaChart from '../components/LambdaChart';
import PointsDecomposition from '../components/PointsDecomposition';
import ScoringTable from '../components/ScoringTable';
import ProbabilityPlayground from '../components/ProbabilityPlayground';
import { simulateRaces, computeExpectedPoints } from '../lib/simulation';
import './Methodology.css';

export default function Methodology({ data }) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const driver = data.drivers[selectedIdx];
  const teamColor = data.teams[driver.team_idx].color;

  // Run client-side simulation for the interactive distribution
  const simDistribution = useMemo(() => {
    const logLambdas = data.drivers.map(d => d.lambda);
    const pDnfs = data.drivers.map(d => d.p_dnf);
    const result = simulateRaces(logLambdas, pDnfs, 15000, 123);
    return result;
  }, [data.drivers]);

  const driverDist = simDistribution[selectedIdx];

  // Always show sprint scoring in the intro, even on non-sprint weekends
  const scoringWithSprint = {
    ...data.scoring,
    sprint: Object.keys(data.scoring.sprint || {}).length > 0
      ? data.scoring.sprint
      : { 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1 },
  };

  return (
    <div className="methodology">
      {/* ====== INTRODUCTION ====== */}
      <header className="meth-header">
        <h1>How We Pick Drivers</h1>
        <p className="meth-subtitle">
          Using betting markets and statistical modeling to make smarter fantasy picks.
        </p>
      </header>

      <section className="meth-intro">
        <h2>The Problem</h2>
        <p>
          Every race weekend, our family F1 fantasy league asks each player to pick 5 drivers
          <strong> in ranked order</strong> before qualifying. Points come from finishing in
          the top 10 (25 for a win, down to 1 for P10), a -10 penalty for a DNF, and a{' '}
          <strong>+10 exact-position bonus</strong> when a driver finishes in the same slot
          as their pick number (e.g., your Pick 3 finishes P3). No qualifying bonuses, no
          overtake points, no fastest lap.
        </p>
        <p>
          So the question is straightforward: which 5 drivers will score the most points this
          weekend? Gut feel gets you part of the way. Everyone knows the Mercedes and Ferrari
          drivers are strong this year. But how do you decide between a consistent midfield
          driver and a volatile frontrunner? Is a driver with a 15% chance of winning but an
          8% chance of DNF a better pick than someone who will reliably finish P6-P8?
        </p>
        <p>
          You need expected points — and to get those, you need probability distributions over
          finishing positions. That's where betting markets come in.
        </p>

        <h2>The Idea</h2>
        <p>
          Betting markets are remarkably good at aggregating information. Thousands of bettors,
          each with their own models, insider knowledge, and informed opinions, collectively
          set odds that reflect the true probabilities of race outcomes more accurately than
          any individual prediction.
        </p>
        <p>
          We can extract those probabilities — but it takes a few steps. Bookmaker odds aren't
          raw probabilities; they include a margin (the "vig") that guarantees the house a profit.
          And even after removing the vig, the odds only give us marginal probabilities for specific
          outcomes (who wins, who finishes top 3), not the full joint distribution we need to
          compute expected fantasy points.
        </p>

        <h2>The Pipeline</h2>
        <p>
          Here's how we go from raw betting odds to expected fantasy points:
        </p>

        <div className="pipeline-diagram">
          <div className="pipeline-step">
            <div className="pipeline-icon">1</div>
            <div className="pipeline-content">
              <strong>Betting Odds</strong>
              <span>Raw lines from bookmakers (e.g., Russell -150 to win)</span>
            </div>
          </div>
          <div className="pipeline-arrow">&darr;</div>
          <div className="pipeline-step">
            <div className="pipeline-icon">2</div>
            <div className="pipeline-content">
              <strong>Fair Probabilities</strong>
              <span>Remove the vig to get true implied probabilities</span>
            </div>
          </div>
          <div className="pipeline-arrow">&darr;</div>
          <div className="pipeline-step">
            <div className="pipeline-icon">3</div>
            <div className="pipeline-content">
              <strong>Plackett-Luce Model</strong>
              <span>Fit a ranking model to recover the full finishing distribution for all 22 drivers</span>
            </div>
          </div>
          <div className="pipeline-arrow">&darr;</div>
          <div className="pipeline-step">
            <div className="pipeline-icon">4</div>
            <div className="pipeline-content">
              <strong>Monte Carlo Simulation</strong>
              <span>Simulate {data.meta.n_simulations.toLocaleString()} races to get position probabilities and expected fantasy points</span>
            </div>
          </div>
          <div className="pipeline-arrow">&darr;</div>
          <div className="pipeline-step">
            <div className="pipeline-icon">5</div>
            <div className="pipeline-content">
              <strong>Lineup Optimizer</strong>
              <span>Solve a rectangular linear assignment to find the optimal 5-driver ordered lineup including the exact-position bonus</span>
            </div>
          </div>
        </div>

        <p>
          The inputs are whatever betting markets are available: race winner odds (almost always),
          plus podium, top 6, top 10, and DNF markets when we can find them. More markets mean
          tighter constraints on the model. The output is a full probability distribution over
          all 23 outcomes (P1 through P22, plus DNF) for every driver — and from that, expected
          fantasy points in our scoring system.
        </p>

        <h2>Scoring Rules</h2>
        <p>
          Each player picks <strong>5 drivers in ranked order</strong> before qualifying each
          weekend (same picks apply to both sprint and race on sprint weekends). No budget cap,
          no qualifying points, no fastest lap.
        </p>
        <ScoringTable scoring={scoringWithSprint} />
        <p style={{ marginTop: 12 }}>
          Sprint weekends (China, Miami, Canada, Great Britain, Netherlands, Singapore) add sprint
          race points on top of the Grand Prix points. A DNF in either race costs -10.
        </p>
        <p style={{ marginTop: 8 }}>
          The <strong>exact-position bonus</strong> rewards picking the right driver <em>and</em>{' '}
          placing them in the right slot. If your Pick 2 finishes P2, you earn +10 on top of
          their base points. If your Pick 2 finishes P1, no bonus — the slot matters. This means
          just picking the five best drivers isn't enough; you also need to order them optimally.
          That's what Step 5 solves.
        </p>
      </section>

      {/* ====== STEP 1: DEVIG ====== */}
      <section className="meth-section">
        <div className="step-num">1</div>
        <h2>Convert Odds to Fair Probabilities</h2>
        <p>
          Bookmaker odds embed a margin (the "vig"). A +200 line doesn't mean 33% — it means
          something less, once you account for the bookmaker's edge. How we remove the vig
          depends on the type of market.
        </p>
        <p>
          For the <strong>race winner</strong> market (exactly one driver wins), we use{' '}
          <strong>Shin's method</strong>. This is an outright market where the fair probabilities
          must sum to 1.0. Shin's method corrects for the favorite-longshot bias: bookmakers
          systematically underprice favorites and overprice longshots. The method solves for an
          "insider trading" parameter z, then backs out fair probabilities with the bias removed.
        </p>
        <p>
          For <strong>placement markets</strong> (podium, top 6, top 10), the structure is
          different. Each driver's odds answer a binary question: "Will this driver finish in
          the top N?" Since multiple drivers can place (3 podium, 6 top 6, etc.), the fair
          probabilities should sum to N, not 1. We devig these by rescaling the implied
          probabilities so they sum to the number of available slots. For example, the 22
          podium implied probabilities get rescaled to sum to 3.
        </p>
        <p>
          We pull odds from The Odds API (race winner market) and supplement with manual odds
          from Oddschecker for podium, top 6, top 10, and DNF markets when available. More
          markets = tighter constraints on the model.
        </p>

        <details className="deep-dive">
          <summary>Deep dive: Devig methods and why they matter</summary>
          <div className="deep-dive-content">
            <h4>What is the vig?</h4>
            <p>
              Suppose a bookmaker offers odds on a two-horse race. Horse A is -150
              (implied probability 60%) and Horse B is +120 (implied probability 45.5%).
              Those probabilities sum to 105.5%, not 100%. The extra 5.5% is the bookmaker's
              margin — the vig. They pay out less than the true odds would require, guaranteeing
              a profit regardless of the outcome.
            </p>
            <p>
              For F1, the vig is typically 15-30% across 22 drivers. To get fair probabilities,
              we need to remove it — but how we remove it matters a lot.
            </p>

            <h4>Approach 1: Multiplicative normalization</h4>
            <p>
              The simplest approach: divide each implied probability by the sum of all implied
              probabilities. If the implied probs sum to 1.20, just divide everything by 1.20.
              This is mathematically clean but assumes the vig is distributed proportionally
              across all outcomes. In reality, it isn't.
            </p>

            <h4>Approach 2: Power method</h4>
            <p>
              Find an exponent k such that the implied probabilities raised to the power k
              sum to 1.0. This allows the vig to be distributed non-uniformly — higher-probability
              outcomes get more of the adjustment. Better than multiplicative, but the functional
              form is arbitrary.
            </p>

            <h4>Approach 3: Shin's method (what we use for the win market)</h4>
            <p>
              In 1991 and 1992, Hyun Song Shin published a model of betting markets that treats
              the overround as a consequence of insider trading. The key insight: bookmakers
              set odds knowing that some fraction z of bettors have inside information. To protect
              themselves against these informed bettors, bookmakers shade the odds — but they shade
              favorites less and longshots more, because an insider bet on a longshot is more
              costly to the bookmaker than an insider bet on a favorite.
            </p>
            <p>
              This produces exactly the favorite-longshot bias observed in real betting markets:
              favorites tend to have odds that are closer to fair, while longshots have odds that
              are more inflated. Shin's method solves for the insider fraction z via bisection,
              then uses it to back out fair probabilities. The result naturally corrects for the
              bias without any arbitrary assumptions about the functional form.
            </p>
            <p>
              For F1, where the favorite might have 15% true win probability and the backmarker
              might have 0.1%, this distinction matters. Multiplicative normalization would
              underestimate the favorite's true probability and overestimate the longshot's.
              Shin's method gives more accurate results, especially in the tails.
            </p>

            <h4>Why placement markets are different</h4>
            <p>
              Shin's method (and the other approaches above) assume a mutually exclusive
              outright market: exactly one driver wins, so fair probabilities must sum to 1.
              But a podium market asks "Will Russell finish top 3?" — and three drivers can
              simultaneously podium. The fair probabilities should sum to 3, not 1.
            </p>
            <p>
              If you mistakenly apply Shin's method to a podium market, you normalize
              everything to sum to 1 — crushing a 75% podium favorite down to ~29%. This
              gives the optimizer contradictory targets (high win probability but low podium
              probability) and produces a poor fit.
            </p>
            <p>
              Instead, we treat placement markets as collections of binary yes/no questions.
              Each driver's implied probability carries individual vig, and the total vig is
              spread across all 22 binary sub-markets. We remove it by rescaling the implied
              probabilities so they sum to the correct target (3 for podium, 6 for top 6,
              10 for top 10).
            </p>

            <div className="deep-dive-ref">
              Shin, H. S. (1991). "Optimal Betting Odds Against Insider Traders."
              <em>Economic Journal</em>, 101(408), 1179-1185.
              <br />
              Shin, H. S. (1992). "Prices of State Contingent Claims with Insider Traders,
              and the Favourite-Longshot Bias." <em>Economic Journal</em>, 102(411), 426-435.
            </div>
          </div>
        </details>
      </section>

      {/* ====== STEP 2: PLACKETT-LUCE ====== */}
      <section className="meth-section">
        <div className="step-num">2</div>
        <h2>Fit the Plackett-Luce Model</h2>
        <p>
          The Plackett-Luce model assigns each driver a strength parameter <strong>{'\u03BB'}</strong> (lambda).
          The probability that driver i wins from a remaining set S is simply {'\u03BB'}_i / {'\u03A3\u03BB'}_j.
          A full finishing order is generated by drawing winners sequentially: pick the winner
          from all drivers, then pick 2nd from the remaining set, and so on.
        </p>
        <p>
          We decompose strength as <code>log({'\u03BB'}) = {'\u03BC'}_team + {'\u03B4'}_driver</code>, so teammates share
          a team component. This creates natural positive correlation — if one Mercedes is strong,
          both are.
        </p>
        <p>
          The tricky part is fitting. Given a candidate set of {'\u03BB'} values, there's no closed-form way
          to compute "what does the model predict for P(Russell finishes top 3)?" — you have to
          simulate it. So the optimizer works like this:
        </p>
        <ol style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: 1.7, marginLeft: 18, marginBottom: 10, maxWidth: 640 }}>
          <li style={{ marginBottom: 6 }}>Start with an initial guess for all 22 {'\u03BB'} values (based on the win probabilities).</li>
          <li style={{ marginBottom: 6 }}>Simulate ~20,000 races using those {'\u03BB'} values. From the simulated results, compute model-implied
            cumulative probabilities: P(top 1), P(top 3), P(top 6), P(top 10) for each driver.</li>
          <li style={{ marginBottom: 6 }}>Compare those model probabilities to the devigged market probabilities by computing the loss (details below).</li>
          <li style={{ marginBottom: 6 }}>The optimizer (scipy's <strong>Powell's method</strong> — a derivative-free
            direction-set method) uses the loss to search for better parameters. Powell's method
            doesn't need gradients: it optimizes along one coordinate direction at a time, then
            builds up conjugate directions from the pattern of improvements. This is well-suited
            to our stochastic loss surface where finite-difference gradients would be noisy.</li>
          <li style={{ marginBottom: 6 }}>Repeat: simulate another ~20K races with the updated {'\u03BB'} values, compute loss, take another step.
            Each evaluation uses a different random seed to smooth the stochastic loss surface.</li>
        </ol>
        <h3>The Objective Function</h3>
        <p>
          The loss function the optimizer minimizes has three terms:
        </p>
        <div className="objective-fn" style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '16px 20px',
          margin: '12px 0 16px',
          maxWidth: 640,
          fontSize: '0.85rem',
          lineHeight: 1.8,
          color: 'var(--text)',
          fontFamily: 'var(--font-data)',
        }}>
          <div style={{ marginBottom: 8 }}>
            <strong style={{ color: 'var(--text-bright)' }}>L</strong> ={'  '}
            <span style={{ color: 'var(--text-muted)' }}>
              {'\u03A3'}<sub>market, driver</sub>{' '}
              ( P<sub>model</sub>(driver, market) {'\u2212'} P<sub>market</sub>(driver, market) )<sup>2</sup>
            </span>
          </div>
          <div style={{ marginBottom: 8, paddingLeft: 24 }}>
            <span style={{ color: 'var(--text-dim)' }}>+{'  '}</span>
            <span style={{ color: 'var(--text-muted)' }}>
              {'\u03B1'}<sub>team</sub> {'\u00B7'} {'\u03A3'}<sub>teammates</sub>{' '}
              ( log{'\u03BB'}<sub>i</sub> {'\u2212'} log{'\u03BB'}<sub>j</sub> )<sup>2</sup>
            </span>
            <span style={{ color: 'var(--text-dim)', fontSize: '0.78rem' }}> — teammate similarity</span>
          </div>
          <div style={{ paddingLeft: 24 }}>
            <span style={{ color: 'var(--text-dim)' }}>+{'  '}</span>
            <span style={{ color: 'var(--text-muted)' }}>
              {'\u03B1'}<sub>shrink</sub> {'\u00B7'} {'\u03A3'}<sub>i</sub>{' '}
              ( log{'\u03BB'}<sub>i</sub> {'\u2212'} <span style={{ textDecoration: 'overline' }}>log{'\u03BB'}</span> )<sup>2</sup>
            </span>
            <span style={{ color: 'var(--text-dim)', fontSize: '0.78rem' }}> — shrinkage toward mean</span>
          </div>
        </div>
        <p>
          The first term is the data fit: for each market (win, podium, top 6, top 10, DNF) and each
          driver with odds in that market, the squared difference between what the model predicts
          (via simulation) and the devigged market probability. The second term penalizes large
          gaps between teammates — most performance difference is the car, not the driver. The third
          term is mild shrinkage toward the mean, preventing extreme {'\u03BB'} values when data is sparse.
        </p>
        <p>
          After ~100-200 iterations (2-4 million simulated races total), the optimizer converges.
          A lower fit loss means the model better reproduces the market odds.
        </p>

        <details className="deep-dive">
          <summary>Deep dive: Why Plackett-Luce?</summary>
          <div className="deep-dive-content">
            <h4>The core problem</h4>
            <p>
              After devigging, we have marginal probabilities: P(Russell wins) = 47%,
              P(Russell finishes top 3) = 75%, and so on. But fantasy points depend on the
              exact finishing position, not just "top N" cutoffs. We need the full joint
              distribution: what's the probability Russell finishes exactly P4? P7? P15?
            </p>
            <p>
              You can't just interpolate between the marginals. Finishing positions are
              dependent — if Russell finishes P1, Antonelli can't also finish P1. We need
              a model that respects this structure.
            </p>

            <h4>Plackett-Luce</h4>
            <p>
              The Plackett-Luce model (Luce, 1959; Plackett, 1975) is a sequential choice
              model for rankings. Each item (driver) has a strength {'\u03BB'}_i {'\u003E'} 0.
              A ranking is generated by repeatedly choosing from the remaining set:
            </p>
            <ul>
              <li>P(driver i wins) = {'\u03BB'}_i / ({'\u03BB'}_1 + {'\u03BB'}_2 + ... + {'\u03BB'}_22)</li>
              <li>P(driver j finishes 2nd | driver i won) = {'\u03BB'}_j / (sum of remaining {'\u03BB'}s)</li>
              <li>And so on, until all positions are filled</li>
            </ul>
            <p>
              This gives us a complete probability distribution over all 22! possible finishing
              orders, parameterized by just 22 numbers. From this, we can compute any marginal
              we want: P(finish exactly P4), P(finish top 6), P(beat a specific rival), etc.
            </p>

            <h4>Team structure</h4>
            <p>
              We decompose each driver's log-strength as {'\u03BC'}_team + {'\u03B4'}_driver, where
              teammates share the team component. This means if the optimizer determines that
              Mercedes has a strong car this weekend, both Russell and Antonelli benefit. The
              driver-specific {'\u03B4'} captures within-team differences. Regularization penalizes
              large teammate gaps, reflecting the reality that most performance difference is
              the car, not the driver.
            </p>

            <h4>Alternatives considered</h4>
            <p>
              <strong>Bradley-Terry model</strong> — works well for pairwise comparisons
              (who beats whom?) but doesn't naturally extend to full rankings of 22 items.
              You'd need to compute each finishing position by conditioning on all possible
              higher-position outcomes, which is computationally expensive and doesn't leverage
              the sequential structure.
            </p>
            <p>
              <strong>Thurstone model</strong> — assigns each driver a latent utility drawn
              from a normal distribution (driver i gets utility U_i ~ N({'\u03BC'}_i, {'\u03C3'}^2)).
              Rankings are determined by sorting utilities. This is similar to PL in practice
              but uses normal distributions instead of Gumbel (extreme value) distributions.
              The key downside: the normal distribution produces thinner tails, making upsets
              less likely than they actually are in F1. The PL/Gumbel setup better captures
              the occasional chaotic race.
            </p>
            <p>
              <strong>Copula approaches</strong> — model the dependence structure between
              finishing positions directly. Extremely flexible but massively overparameterized
              for our data. With only 4-5 market constraints per driver (win, podium, top 6,
              top 10, DNF), we can't identify all the correlation parameters. PL gives us
              a reasonable dependence structure (via the sequential elimination mechanism)
              from just one parameter per driver.
            </p>

            <div className="deep-dive-ref">
              Luce, R. D. (1959). <em>Individual Choice Behavior: A Theoretical Analysis.</em> Wiley.
              <br />
              Plackett, R. L. (1975). "The Analysis of Permutations."
              <em>Journal of the Royal Statistical Society, Series C</em>, 24(2), 193-202.
            </div>
          </div>
        </details>

        <h3>Driver Strength ({'\u03BB'})</h3>
        <p className="muted" style={{ marginBottom: 8 }}>
          Click a driver to explore their distribution below. Teammates are connected.
        </p>
        <LambdaChart
          drivers={data.drivers}
          teams={data.teams}
          selectedIdx={selectedIdx}
          onSelect={setSelectedIdx}
        />
      </section>

      {/* ====== STEP 3: SIMULATION ====== */}
      <section className="meth-section">
        <div className="step-num">3</div>
        <h2>Simulate {data.meta.n_simulations.toLocaleString()} Races</h2>
        <p>
          With the fitted {'\u03BB'} values and DNF probabilities, we simulate {data.meta.n_simulations.toLocaleString()} complete
          races. In each simulation: first, each driver independently rolls for DNF (based on
          team reliability + driver risk). Then the Plackett-Luce model draws a finishing order
          for all non-DNF drivers. The result is a full probability distribution over all 23
          outcomes (P1 through P22, plus DNF) for every driver.
        </p>

        <details className="deep-dive">
          <summary>Deep dive: Why Monte Carlo?</summary>
          <div className="deep-dive-content">
            <h4>Why not compute analytically?</h4>
            <p>
              In principle, you can compute P(driver i finishes position k) exactly from the
              Plackett-Luce model. For P(win), it's simple: {'\u03BB'}_i / {'\u03A3\u03BB'}_j.
              For P(finishes 2nd), you need to sum over all possible winners:
              {'\u03A3'}_{'{'}j{'\u2260'}i{'}'} [P(j wins) {'\u00D7'} P(i wins from remaining)].
              For P(finishes 3rd), you need to sum over all possible (winner, runner-up) pairs.
            </p>
            <p>
              By the time you get to P(finishes P11), you're summing over all possible
              orderings of the top 10 — that's 10! = 3.6 million terms. For P(finishes P22),
              you'd need 21! terms. This is computationally intractable for 22 drivers.
            </p>

            <h4>What simulation gives us</h4>
            <p>
              Monte Carlo simulation sidesteps this entirely. We just draw many random
              races and count how often each outcome occurs. With 50K simulations:
            </p>
            <ul>
              <li>The full probability distribution (P1 through P22 + DNF) for every driver</li>
              <li>Standard deviations and confidence intervals for free</li>
              <li>Any derived quantity: expected points, P(beat rival), P(score {'\u2265'} 10 pts), etc.</li>
              <li>Standard error of ~0.2 percentage points for the most common outcomes</li>
            </ul>
            <p>
              The 50K simulation count is a balance between precision and compute time. For
              the model fitting (which runs inside the optimizer loop), we use ~20K sims per
              evaluation to keep it fast while still being smooth enough for gradient-based
              optimization.
            </p>
          </div>
        </details>

        <h3>Position Distribution — {driver.name}</h3>
        <p className="muted" style={{ marginBottom: 8 }}>
          Hover bars for exact probabilities. Red bar = DNF.
          (Interactive: re-simulated with 15K races in your browser.)
        </p>
        <PositionChart
          distribution={driverDist}
          color={teamColor}
          scoring={data.scoring.race}
        />

        <h3 style={{ marginTop: 24 }}>Cumulative Distribution</h3>
        <p className="muted" style={{ marginBottom: 8 }}>
          Dots show the market cutoff points used for fitting.
        </p>
        <CdfChart distribution={driverDist} color={teamColor} />
      </section>

      {/* ====== STEP 4: SCORING ====== */}
      <section className="meth-section">
        <div className="step-num">4</div>
        <h2>Compute Expected Points</h2>
        <p>
          Expected points = {'\u03A3'} P(position k) {'\u00D7'} points(k) + P(DNF) {'\u00D7'} ({data.scoring.dnf_penalty}). Each finishing
          position contributes its probability times its point value. The DNF penalty of {data.scoring.dnf_penalty} is
          a significant drag — drivers with high DNF risk (20%+) can have negative expected points
          even if they occasionally finish in the top 10.
        </p>

        <details className="deep-dive">
          <summary>Deep dive: Why this scoring changes optimal picks</summary>
          <div className="deep-dive-content">
            <h4>The DNF penalty and exact position bonus</h4>
            <p>
              In our scoring system, a DNF costs -10 points. If a driver has a 10% DNF chance,
              that's an expected cost of -1.0 points per race. Additionally, if a driver finishes
              in the exact same slot as their pick position (e.g., your Pick 3 finishes P3), you
              earn a +10 bonus — equal in magnitude to the DNF penalty.
            </p>
            <p>
              This means reliability is far more important in our league than in the official
              F1 Fantasy game (which doesn't penalize DNFs as heavily). A consistent P6-P8
              finisher with low DNF risk can easily outscore a flashier driver who occasionally
              wins but retires more often.
            </p>

            <h4>Variance matters for picks</h4>
            <p>
              Since you pick 5 drivers, diversification matters. Two teammates are positively
              correlated (they share the same car), so picking both Mercedes drivers gives you
              less diversification than picking one Mercedes and one Ferrari. The standard
              deviation column on the dashboard helps identify high-variance picks — useful
              when you're behind in the standings and need upside.
            </p>
          </div>
        </details>

        <h3>Points Decomposition — {driver.name}</h3>
        <p className="muted" style={{ marginBottom: 8 }}>
          How much each finishing position contributes to expected points.
        </p>
        <PointsDecomposition
          distribution={driver.position_distribution}
          scoring={data.scoring.race}
          dnfPenalty={data.scoring.dnf_penalty}
          color={teamColor}
        />
      </section>

      {/* ====== STEP 5: LINEUP OPTIMIZER ====== */}
      <section className="meth-section">
        <div className="step-num">5</div>
        <h2>Find the Optimal Lineup</h2>
        <p>
          Once we have expected points and full position distributions for every driver, we
          need to answer: which 5 drivers should you pick, and in what order? The exact-position
          bonus means the ordering matters — placing a driver in the wrong slot loses you up to
          10 expected points.
        </p>
        <p>
          The naive approach — enumerate all ways to choose 5 drivers from 22 and try every
          ordering — would be C(22,5) × 5! = 3.2 million combinations. We can do much better.
        </p>

        <h3>Reducing to a Single Assignment Problem</h3>
        <p>
          Define a <strong>(22 × 5) score matrix</strong> where each entry is:
        </p>
        <div style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '14px 20px',
          margin: '12px 0 16px',
          maxWidth: 600,
          fontSize: '0.85rem',
          fontFamily: 'var(--font-data)',
          color: 'var(--text-muted)',
          lineHeight: 1.8,
        }}>
          score[driver][slot] = E[base points for driver] + P(driver finishes position slot) × 10
        </div>
        <p>
          Choosing 5 drivers and assigning each to a slot is exactly a{' '}
          <strong>rectangular linear assignment problem</strong>: pick one entry per column
          (slot), at most one entry per row (driver), to maximise the total. This is solved in
          a single pass by the <strong>Hungarian algorithm</strong> — no enumeration needed.
        </p>

        <details className="deep-dive">
          <summary>Deep dive: Why this formulation is exact</summary>
          <div className="deep-dive-content">
            <h4>The key insight</h4>
            <p>
              Total expected points for a lineup = Σ E[base points for driver k] + Σ P(driver k
              finishes in slot k) × 10. The base-points term depends only on <em>which</em>{' '}
              drivers are picked; the bonus term depends on both who is picked and how they are
              ordered. Combining them into a single score per (driver, slot) pair means the two
              decisions become one: choosing an entry in the matrix simultaneously selects the
              driver and assigns their slot.
            </p>
            <h4>Why C(22,5) enumeration is unnecessary</h4>
            <p>
              The earlier pruning approach kept only drivers whose base EP was within 10 points
              of the 5th-best driver (the max bonus), then enumerated subsets of those candidates.
              For this race that was still all 22 drivers. The assignment formulation avoids
              enumeration entirely: scipy's <code>linear_sum_assignment</code> solves a 22×5
              rectangular matrix in O(n·m²) ≈ O(22 × 25) operations — effectively instant.
            </p>
            <h4>The Hungarian algorithm</h4>
            <p>
              The Hungarian algorithm works by manufacturing zeros in the cost matrix. It subtracts
              row and column minimums (which preserves the optimal assignment), then finds a perfect
              matching through zero-cost cells. When no perfect matching exists, it adds a small
              delta to create new zeros, repeating until the matching covers all columns. Each step
              is O(n²); at most n steps are needed, giving O(n³) overall. For rectangular matrices,
              scipy pads the shorter dimension with zeros and strips dummy assignments from the result.
            </p>
          </div>
        </details>

        {data.optimal_lineup && (
          <div style={{ marginTop: 16 }}>
            <h3>Current Optimal Lineup — {data.meta.race}</h3>
            <p className="muted" style={{ marginBottom: 12 }}>
              The assignment that maximises expected points including the exact-position bonus.
              Slot bonus E[pts] = P(driver finishes in that exact position) × 10.
            </p>
            <table style={{ borderCollapse: 'collapse', fontSize: '0.85rem', fontFamily: 'var(--font-data)', width: '100%', maxWidth: 560 }}>
              <thead>
                <tr>
                  {['Pick', 'Driver', 'Base E[pts]', 'Slot bonus E[pts]', 'Total'].map(h => (
                    <th key={h} style={{ padding: '6px 14px 6px 0', textAlign: h === 'Pick' || h === 'Driver' ? 'left' : 'right', color: 'var(--text-dim)', fontWeight: 500, borderBottom: '1px solid var(--border)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.4px' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.optimal_lineup.picks.map(pick => {
                  const tc = data.teams[pick.team_idx]?.color ?? '#888';
                  return (
                    <tr key={pick.slot} style={{ borderBottom: '1px solid var(--border)' }}>
                      <td style={{ padding: '9px 14px 9px 0', color: 'var(--text-dim)' }}>Pick {pick.slot}</td>
                      <td style={{ padding: '9px 14px 9px 0' }}>
                        <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: tc, marginRight: 8 }} />
                        <span style={{ color: 'var(--text-bright)', fontWeight: 500 }}>{pick.name}</span>
                      </td>
                      <td style={{ padding: '9px 14px 9px 0', textAlign: 'right', color: 'var(--text-muted)' }}>{pick.ep_base.toFixed(2)}</td>
                      <td style={{ padding: '9px 14px 9px 0', textAlign: 'right', color: '#4ade80' }}>+{pick.slot_bonus_ev.toFixed(2)}</td>
                      <td style={{ padding: '9px 14px 9px 0', textAlign: 'right', color: 'var(--text-bright)', fontWeight: 600 }}>{(pick.ep_base + pick.slot_bonus_ev).toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr>
                  <td colSpan={2} style={{ padding: '9px 14px 9px 0', color: 'var(--text-dim)', fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.4px', borderTop: '2px solid var(--border-light)' }}>Expected total</td>
                  <td style={{ padding: '9px 14px 9px 0', textAlign: 'right', color: 'var(--text-muted)', borderTop: '2px solid var(--border-light)' }}>{data.optimal_lineup.ep_base_total.toFixed(2)}</td>
                  <td style={{ padding: '9px 14px 9px 0', textAlign: 'right', color: '#4ade80', borderTop: '2px solid var(--border-light)' }}>+{data.optimal_lineup.ep_bonus_total.toFixed(2)}</td>
                  <td style={{ padding: '9px 14px 9px 0', textAlign: 'right', color: 'var(--text-bright)', fontWeight: 700, fontSize: '1rem', borderTop: '2px solid var(--border-light)' }}>{data.optimal_lineup.ep_grand_total.toFixed(2)}</td>
                </tr>
              </tfoot>
            </table>
          </div>
        )}
      </section>

      {/* ====== PROBABILITY PLAYGROUND ====== */}
      <section className="meth-section">
        <h2>Probability Playground</h2>
        <p>
          Adjust a driver's strength parameter and see how it changes their position
          distribution in real time. The dashed line shows the fitted model; the solid bars
          show the result with your adjustment. Watch how the model probabilities diverge from
          the market constraints as you move further from the fitted value — that's the fit
          error increasing.
        </p>
        <ProbabilityPlayground data={data} />
      </section>

    </div>
  );
}
