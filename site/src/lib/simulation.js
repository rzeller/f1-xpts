/**
 * Client-side Plackett-Luce race simulator.
 * Port of pipeline/plackett_luce.py simulate_races().
 *
 * Runs in the browser for interactive Methodology page charts.
 * Uses a simple xoshiro128** PRNG for speed.
 */

// --- Seeded PRNG (xoshiro128**) ---
function xoshiro128ss(seed) {
  let s = [seed, seed ^ 0xdeadbeef, seed ^ 0xcafebabe, seed ^ 0x12345678];
  // Warm up
  for (let i = 0; i < 20; i++) next();

  function rotl(x, k) { return ((x << k) | (x >>> (32 - k))) >>> 0; }

  function next() {
    const result = (rotl((s[1] * 5) >>> 0, 7) * 9) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11);
    return result / 4294967296; // [0, 1)
  }

  return next;
}

/**
 * Box-Muller transform: generate a standard normal sample from uniform random.
 */
function boxMuller(rand) {
  const u1 = Math.max(rand(), 1e-10); // avoid log(0)
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Weighted random choice: given an array of weights, return the index.
 */
function weightedChoice(weights, rand) {
  let total = 0;
  for (let i = 0; i < weights.length; i++) total += weights[i];
  let r = rand() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

/**
 * Simulate races using the Plackett-Luce model with DNFs and optional correlation.
 *
 * @param {number[]} logLambdas - log-strength per driver (length 22)
 * @param {number[]} pDnfs - DNF probability per driver (length 22)
 * @param {number} nSims - number of simulations (default 10000)
 * @param {number} seed - random seed
 * @param {number[]|null} teamIndices - team index per driver (length 22), or null
 * @param {object|null} correlationParams - {sigma_team, sigma_global, sigma_dnf}, or null
 * @returns {number[][]} positionProbs - [driver][position] where position 0-21 = P1-P22, 22 = DNF
 */
export function simulateRaces(
  logLambdas, pDnfs, nSims = 10000, seed = 42,
  teamIndices = null, correlationParams = null
) {
  const n = logLambdas.length;
  const baseLambdas = logLambdas.map(ll => Math.exp(ll));
  const counts = Array.from({ length: n }, () => new Float64Array(n + 1));
  const rand = xoshiro128ss(seed);

  // Pre-compute correlation parameters
  const hasCorrelation = correlationParams != null && teamIndices != null;
  const sigmaTeam = hasCorrelation ? (correlationParams.sigma_team || 0) : 0;
  const sigmaGlobal = hasCorrelation ? (correlationParams.sigma_global || 0) : 0;
  const sigmaDnf = hasCorrelation ? (correlationParams.sigma_dnf || 0) : 0;
  const nTeams = hasCorrelation ? Math.max(...teamIndices) + 1 : 0;

  for (let sim = 0; sim < nSims; sim++) {
    // Per-sim lambda perturbation and DNF adjustment
    let simLambdas = baseLambdas;
    let simPDnfs = pDnfs;

    if (hasCorrelation) {
      // 1. Team race-day noise: one z per team, shared by teammates
      if (sigmaTeam > 0) {
        const zTeam = Array.from({ length: nTeams }, () => boxMuller(rand));
        simLambdas = baseLambdas.map((lam, i) =>
          lam * Math.exp(sigmaTeam * zTeam[teamIndices[i]])
        );
      }

      // 2. Chaos scaling: in sequential-draw form, scaling Gumbel noise by c
      //    is equivalent to raising lambdas to the power 1/c.
      //    chaos_scale = exp(sigma_global * z), so inv = exp(-sigma_global * z)
      if (sigmaGlobal > 0) {
        const zGlobal = boxMuller(rand);
        const invChaosScale = Math.exp(-sigmaGlobal * zGlobal);
        simLambdas = simLambdas.map(lam => Math.pow(lam, invChaosScale));
      }

      // 3. Correlated DNFs: log-normal multiplier on DNF probabilities
      if (sigmaDnf > 0) {
        const zDnf = boxMuller(rand);
        const mult = Math.exp(sigmaDnf * zDnf);
        simPDnfs = pDnfs.map(p => Math.min(p * mult, 0.5));
      }
    }

    // DNF rolls
    const finishers = [];
    const dnfs = [];
    for (let i = 0; i < n; i++) {
      if (rand() < simPDnfs[i]) {
        dnfs.push(i);
      } else {
        finishers.push(i);
      }
    }

    // PL sequential draw
    const remaining = [...finishers];
    const remLambdas = remaining.map(i => simLambdas[i]);

    for (let pos = 0; pos < remaining.length; pos++) {
      const chosenLocal = weightedChoice(remLambdas, rand);
      const chosenDriver = remaining[chosenLocal];
      counts[chosenDriver][pos] += 1;

      // Remove from remaining
      remaining.splice(chosenLocal, 1);
      remLambdas.splice(chosenLocal, 1);
    }

    // Record DNFs
    for (const d of dnfs) {
      counts[d][n] += 1;
    }
  }

  // Normalize to probabilities
  return counts.map(row => Array.from(row).map(c => c / nSims));
}

/**
 * Compute expected points from position distribution.
 */
export function computeExpectedPoints(positionProbs, scoring, dnfPenalty = -10) {
  let ep = 0;
  for (let pos = 0; pos < positionProbs.length - 1; pos++) {
    const pts = scoring[String(pos + 1)] || 0;
    ep += positionProbs[pos] * pts;
  }
  ep += positionProbs[positionProbs.length - 1] * dnfPenalty;
  return ep;
}
