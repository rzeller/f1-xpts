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
 * Simulate races using the Plackett-Luce model with DNFs.
 *
 * @param {number[]} logLambdas - log-strength per driver (length 22)
 * @param {number[]} pDnfs - DNF probability per driver (length 22)
 * @param {number} nSims - number of simulations (default 10000)
 * @param {number} seed - random seed
 * @returns {number[][]} positionProbs - [driver][position] where position 0-21 = P1-P22, 22 = DNF
 */
export function simulateRaces(logLambdas, pDnfs, nSims = 10000, seed = 42) {
  const n = logLambdas.length;
  const lambdas = logLambdas.map(ll => Math.exp(ll));
  const counts = Array.from({ length: n }, () => new Float64Array(n + 1));
  const rand = xoshiro128ss(seed);

  for (let sim = 0; sim < nSims; sim++) {
    // DNF rolls
    const finishers = [];
    const dnfs = [];
    for (let i = 0; i < n; i++) {
      if (rand() < pDnfs[i]) {
        dnfs.push(i);
      } else {
        finishers.push(i);
      }
    }

    // PL sequential draw
    const remaining = [...finishers];
    const remLambdas = remaining.map(i => lambdas[i]);

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
