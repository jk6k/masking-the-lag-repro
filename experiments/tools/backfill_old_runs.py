"""Backfill missing timeline_summary.csv and p0_p1_alignment.csv for old runs.

Re-runs phase1_runner.py for each run that is missing these files,
using the existing config_snapshot.yaml from each run directory.
This will overwrite run outputs with current-code-version results.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "results" / "runs"
RUNNER = ROOT / "experiments" / "tools" / "phase1_runner.py"


def find_incomplete_runs() -> list[Path]:
    """Return run directories missing timeline_summary.csv."""
    incomplete = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "master_metrics.csv").exists():
            continue
        if not (run_dir / "config_snapshot.yaml").exists():
            continue
        if not (run_dir / "timeline_summary.csv").exists():
            incomplete.append(run_dir)
    return incomplete


def main() -> None:
    incomplete = find_incomplete_runs()
    print(f"Found {len(incomplete)} runs missing timeline_summary.csv")

    if not incomplete:
        print("Nothing to do.")
        return

    for i, run_dir in enumerate(incomplete, 1):
        cfg_path = run_dir / "config_snapshot.yaml"
        print(f"\n[{i}/{len(incomplete)}] Re-running: {run_dir.name}")
        print(f"  Config: {cfg_path}")

        result = subprocess.run(
            [sys.executable, str(RUNNER), "--config", str(cfg_path)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )

        if result.returncode != 0:
            print(f"  ERROR (exit {result.returncode}):")
            for line in (result.stderr or "").strip().split("\n")[-5:]:
                print(f"    {line}")
        else:
            has_tl = (run_dir / "timeline_summary.csv").exists()
            has_p0 = (run_dir / "p0_p1_alignment.csv").exists()
            print(f"  OK  timeline_summary={has_tl}  p0_p1_alignment={has_p0}")

    # Final summary
    still_missing = find_incomplete_runs()
    print(f"\n=== DONE ===")
    print(f"Processed: {len(incomplete)}")
    print(f"Still missing: {len(still_missing)}")
    if still_missing:
        for d in still_missing:
            print(f"  {d.name}")


if __name__ == "__main__":
    main()
