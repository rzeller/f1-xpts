# Backtesting Historical Race Predictions

Validates the F1 expected points pipeline against ~100 historical races (2020-2024) by fetching pre-race betting odds, running the full model pipeline, and comparing predictions to actual results.

## Prerequisites

- Python 3.10+
- `pip install -r pipeline/requirements.txt`
- A paid plan on [The Odds API](https://the-odds-api.com) ($30/month plan gives 20,000 credits)

## Step 1: Fetch Historical Odds

This downloads pre-race odds snapshots from The Odds API historical endpoint. It fetches odds at multiple time points per race weekend, in priority order:

| Priority | Time Point | Description |
|----------|-----------|-------------|
| P1 | `before_qualifying` | Before first qualifying event of the weekend |
| P2 | `after_sprint_qual` | After sprint qualifying (sprint weekends only) |
| P3 | `after_fp2` | After FP2 / before next session |
| P4 | `after_qualifying` | After race-grid qualifying / before race |
| P5 | `before_fp2` | Before FP2 |
| P6 | `before_fp1` | Before FP1 |

### How to run

From the repo root:

```bash
# Recommended: fetch P1 and P2 first (most important snapshots)
python -m pipeline.backtest.fetch_historical_odds \
    --api-key YOUR_API_KEY \
    --max-priority 2

# Then fetch remaining time points if budget allows
python -m pipeline.backtest.fetch_historical_odds \
    --api-key YOUR_API_KEY \
    --max-priority 6
```

### Dry run first

Always do a dry run to see what would be fetched and the estimated cost:

```bash
python -m pipeline.backtest.fetch_historical_odds \
    --dry-run \
    --max-priority 2
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--api-key` / `-k` | `$ODDS_API_KEY` env var | Your Odds API key |
| `--output-dir` / `-o` | `pipeline/historical_odds` | Where to save odds files |
| `--seasons 2023 2024` | all | Filter to specific seasons |
| `--max-priority` / `-p` | 6 | Only fetch time points up to this priority (1-6) |
| `--regions` | `us,uk,eu,au` | Bookmaker regions (more = sharper odds, no extra cost) |
| `--budget` | 20000 | Credit limit (stops before exceeding) |
| `--dry-run` | off | Print plan without making API calls |

### Cost estimate

Each time point per race costs ~20 credits (10 for outrights + 10 for placement markets).

| What | Races | Credits |
|------|-------|---------|
| P1 only (all races) | ~95 | ~1,900 |
| P1 + P2 (P2 is sprint-only) | ~95 + ~20 | ~2,300 |
| All 6 priorities | ~495 total time points | ~9,900 |

The $30/month plan (20,000 credits) comfortably covers all priorities in a single billing cycle.

### Checkpoint/resume

The script writes one JSON file per race per time point. If interrupted, re-running the same command skips already-downloaded files automatically. No data is lost.

### Output

Files are saved as `{season}_{round}_{slug}_{timepoint}.json` in the output directory:

```
pipeline/historical_odds/
  2024_01_bahrain-gp_before_qualifying.json
  2024_01_bahrain-gp_after_fp2.json
  2024_02_saudi-arabian-gp_before_qualifying.json
  ...
```

Each file contains American odds per driver per market (win, top4, top6, etc.) plus metadata (snapshot timestamp, session info, API event data).

### Important notes

- The historical odds directory is **committed to git** (not gitignored) since this is paid data you don't want to re-fetch.
- `pipeline/backtest_cache/` (fitted model cache) IS gitignored.
- The script fetches session schedules (FP1/FP2/Qualifying/Sprint times) from the free Jolpica API automatically -- no separate setup needed.
- Regions (us, uk, eu, au) don't multiply the API cost. Each call is 10 credits regardless of how many regions are included. More regions = more bookmakers = sharper odds.

## Step 2: Run Backtest

Once odds are downloaded:

```bash
# Single backtest with default parameters
python -m pipeline.backtest.cli run

# Filter to one season for a quick test
python -m pipeline.backtest.cli run --seasons 2024

# Use a specific time point's odds
python -m pipeline.backtest.cli run --time-point before_qualifying
```

## Step 3: Parameter Sweep

```bash
# Grid search across parameter combinations
python -m pipeline.backtest.cli sweep --n-workers 4

# Cross-validation (leave-one-season-out)
python -m pipeline.backtest.cli cv

# Fine-tune best parameters
python -m pipeline.backtest.cli optimize --n-evals 50
```

## Step 4: Visualize Results

```bash
python -m pipeline.backtest.cli plot --results-file pipeline/backtest_results/sweep_latest.json
```

## File Structure

```
pipeline/backtest/
  __init__.py
  __main__.py                   # python -m pipeline.backtest
  fetch_historical_odds.py      # Step 1: download historical odds
  config_historical.py          # Dynamic per-race driver configs
  metrics.py                    # Evaluation metrics (Brier, log-likelihood, etc.)
  runner.py                     # Core backtest loop
  sweep.py                      # Grid search + Bayesian optimization
  results_db.py                 # Results storage
  plot_results.py               # Visualization
  cli.py                        # Unified CLI
pipeline/historical_odds/       # Downloaded odds (committed)
pipeline/backtest_cache/        # Cached fitted models (gitignored)
pipeline/backtest_results/      # Sweep output (committed)
```
