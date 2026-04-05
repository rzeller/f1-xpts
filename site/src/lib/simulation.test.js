/**
 * Tests for simulation.js — client-side Plackett-Luce simulator.
 */

import { describe, it, expect } from "vitest";
import { simulateRaces, computeExpectedPoints } from "./simulation.js";

// --- simulateRaces ---

describe("simulateRaces", () => {
  it("returns correct shape for 3 drivers", () => {
    const logLambdas = [Math.log(3), Math.log(2), Math.log(1)];
    const pDnfs = [0.05, 0.05, 0.05];
    const result = simulateRaces(logLambdas, pDnfs, 1000, 42);
    expect(result.length).toBe(3);
    // Each driver has 4 entries: P1, P2, P3, DNF
    expect(result[0].length).toBe(4);
  });

  it("rows sum to approximately 1", () => {
    const logLambdas = [0.5, 0, -0.5];
    const pDnfs = [0.05, 0.05, 0.05];
    const result = simulateRaces(logLambdas, pDnfs, 5000, 42);
    for (const row of result) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 1);
    }
  });

  it("strongest driver wins most often", () => {
    const logLambdas = [Math.log(5), Math.log(2), Math.log(1)];
    const pDnfs = [0.0, 0.0, 0.0];
    const result = simulateRaces(logLambdas, pDnfs, 10000, 42);
    // Driver 0 should win more than driver 2
    expect(result[0][0]).toBeGreaterThan(result[2][0]);
  });

  it("respects DNF probabilities", () => {
    const logLambdas = [0, 0, 0];
    const pDnfs = [0.5, 0.5, 0.5];
    const result = simulateRaces(logLambdas, pDnfs, 10000, 42);
    for (let i = 0; i < 3; i++) {
      // DNF is last entry
      expect(result[i][3]).toBeCloseTo(0.5, 1);
    }
  });

  it("zero DNF means no DNFs", () => {
    const logLambdas = [0, 0, 0];
    const pDnfs = [0, 0, 0];
    const result = simulateRaces(logLambdas, pDnfs, 1000, 42);
    for (let i = 0; i < 3; i++) {
      expect(result[i][3]).toBe(0);
    }
  });

  it("is deterministic with same seed", () => {
    const logLambdas = [1, 0, -1];
    const pDnfs = [0.1, 0.1, 0.1];
    const r1 = simulateRaces(logLambdas, pDnfs, 1000, 42);
    const r2 = simulateRaces(logLambdas, pDnfs, 1000, 42);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        expect(r1[i][j]).toBe(r2[i][j]);
      }
    }
  });

  it("different seeds give different results", () => {
    const logLambdas = [1, 0, -1];
    const pDnfs = [0.1, 0.1, 0.1];
    const r1 = simulateRaces(logLambdas, pDnfs, 1000, 1);
    const r2 = simulateRaces(logLambdas, pDnfs, 1000, 2);
    // At least one value should differ
    let differs = false;
    for (let i = 0; i < 3 && !differs; i++) {
      for (let j = 0; j < 4 && !differs; j++) {
        if (r1[i][j] !== r2[i][j]) differs = true;
      }
    }
    expect(differs).toBe(true);
  });

  it("equal lambdas give roughly equal win rates", () => {
    const logLambdas = [0, 0, 0, 0];
    const pDnfs = [0, 0, 0, 0];
    const result = simulateRaces(logLambdas, pDnfs, 20000, 42);
    for (let i = 0; i < 4; i++) {
      expect(result[i][0]).toBeCloseTo(0.25, 1);
    }
  });

  it("works with correlation parameters", () => {
    const logLambdas = [1, 0.8, 0, -0.2, -1, -1.2];
    const pDnfs = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
    const teamIndices = [0, 0, 1, 1, 2, 2];
    const correlation = { sigma_team: 0.5, sigma_global: 0.3, sigma_dnf: 0.2 };
    const result = simulateRaces(logLambdas, pDnfs, 5000, 42, teamIndices, correlation);
    expect(result.length).toBe(6);
    expect(result[0].length).toBe(7); // 6 positions + DNF
    for (const row of result) {
      const sum = row.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 1);
    }
  });
});

// --- computeExpectedPoints ---

describe("computeExpectedPoints", () => {
  const raceScoring = {
    "1": 25, "2": 18, "3": 15, "4": 12, "5": 10,
    "6": 8, "7": 6, "8": 4, "9": 2, "10": 1,
  };

  it("certain winner gets 25 points", () => {
    // 22 positions + DNF = 23 entries
    const dist = new Array(23).fill(0);
    dist[0] = 1.0;
    expect(computeExpectedPoints(dist, raceScoring, -10)).toBe(25);
  });

  it("certain DNF gets penalty", () => {
    const dist = new Array(23).fill(0);
    dist[22] = 1.0;
    expect(computeExpectedPoints(dist, raceScoring, -10)).toBe(-10);
  });

  it("50/50 win or DNF", () => {
    const dist = new Array(23).fill(0);
    dist[0] = 0.5;
    dist[22] = 0.5;
    const ep = computeExpectedPoints(dist, raceScoring, -10);
    expect(ep).toBeCloseTo(0.5 * 25 + 0.5 * -10, 5);
  });

  it("P11 scores zero", () => {
    const dist = new Array(23).fill(0);
    dist[10] = 1.0; // P11
    expect(computeExpectedPoints(dist, raceScoring, -10)).toBe(0);
  });

  it("sprint scoring", () => {
    const sprintScoring = {
      "1": 8, "2": 7, "3": 6, "4": 5, "5": 4, "6": 3, "7": 2, "8": 1,
    };
    const dist = new Array(23).fill(0);
    dist[0] = 1.0;
    expect(computeExpectedPoints(dist, sprintScoring, -10)).toBe(8);
  });
});
