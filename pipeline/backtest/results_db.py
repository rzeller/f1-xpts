"""
Results storage and loading for backtest sweep data.
"""

import json
from pathlib import Path
from typing import List, Optional


def save_sweep_results(results: dict, output_dir: str = "pipeline/backtest_results") -> str:
    """Save sweep results to JSON. Returns the filepath."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sweep_id = results.get("sweep_id", "unknown")
    filename = f"sweep_{sweep_id}.json"
    filepath = output_path / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    # Also save as latest
    latest = output_path / "sweep_latest.json"
    with open(latest, "w") as f:
        json.dump(results, f, indent=2)

    return str(filepath)


def load_sweep_results(filepath: str) -> dict:
    """Load sweep results from JSON."""
    with open(filepath) as f:
        return json.load(f)


def list_sweep_results(results_dir: str = "pipeline/backtest_results") -> List[dict]:
    """List all available sweep result files."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    files = []
    for f in sorted(results_path.glob("sweep_*.json")):
        if f.name == "sweep_latest.json":
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            files.append({
                "path": str(f),
                "sweep_id": data.get("sweep_id", ""),
                "n_configs": data.get("n_configs", 0),
                "elapsed": data.get("elapsed_seconds", 0),
            })
        except (json.JSONDecodeError, OSError):
            continue

    return files


def get_best_params(
    results: dict,
    metric: str = "fantasy_efficiency",
) -> Optional[dict]:
    """Extract the best parameters for a given metric from sweep results."""
    best = results.get("best_by_metric", {}).get(metric)
    if best:
        return best.get("params")
    return None
