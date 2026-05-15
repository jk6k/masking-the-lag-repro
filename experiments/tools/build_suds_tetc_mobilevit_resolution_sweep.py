#!/usr/bin/env python3
"""Build the R12e MobileViT-S resolution accuracy sweep artifact.

Orchestrates run_suds_eval_mlx.py across 4 resolutions x 3 seeds x 2 conditions
(e0_dense baseline + e8_overflow promoted policy), then aggregates results.

Full ImageNet val (50000 samples), governed MPS only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260514_r12_reinforcement"
DATE = "2026-05-14"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

RUN_SCRIPT = REPO_ROOT / "experiments/tools/run_suds_eval_mlx.py"

RESOLUTIONS = [160, 192, 224, 256]
SEEDS = [0, 1, 2]
CONDITIONS = ["e0_dense", "e8_overflow"]
MODEL = "mobilevit_s"
TAU_LOW = 0.30
TAU_HIGH = 0.95
MAX_EVAL_SAMPLES = 50000

CSV_OUT = REPORT_DATA / f"suds_tetc_mobilevit_resolution_accuracy_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_mobilevit_resolution_accuracy_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260514_suds_tetc_r12_deep_reinforcement.md"

IMAGENET_VAL = os.environ.get(
    "SUDS_IMAGENET_VAL",
    str(REPO_ROOT / "<private_imagenet_val>"),
)

ACCURACY_TARGET_PP = 1.0

CSV_FIELDS = [
    "tag", "roadmap_item", "model", "resolution", "seed", "condition",
    "top1", "top5", "processed_samples", "elapsed_s", "device",
    "mlx_default_device", "mlx_metal_available", "git_hash",
    "tau_low", "tau_high", "nominal_input_size", "eval_input_size",
    "keep_ratio", "degrade_ratio", "prune_ratio", "delta_vs_baseline_pp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--resolutions", type=int, nargs="+", default=RESOLUTIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--imagenet-val", default=IMAGENET_VAL)
    parser.add_argument("--python-bin", default=None,
                       help="Python binary (default: sys.executable)")
    parser.add_argument("--max-eval-samples", type=int, default=MAX_EVAL_SAMPLES,
                       help="Samples per run. Default is full ImageNet val (50000).")
    parser.add_argument("--skip-runs", action="store_true",
                       help="Skip actual runs, use existing JSON outputs")
    parser.add_argument("--run-output-dir", type=Path,
                       default=REPO_ROOT / "experiments/results/runs/suds_tetc_mobilevit_resolution_sweep_20260514_r12_reinforcement")
    return parser.parse_args()


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_single(
    *,
    python_bin: str,
    resolution: int,
    seed: int,
    condition: str,
    output_json: Path,
    imagenet_val: str,
    max_eval_samples: int,
) -> dict[str, Any]:
    """Run a single resolution/seed/condition evaluation."""
    cmd = [
        python_bin,
        str(RUN_SCRIPT),
        "--imagenet_val", imagenet_val,
        "--model", MODEL,
        "--input_size_override", str(resolution),
        "--seed", str(seed),
        "--condition", condition,
        "--tau_low", str(TAU_LOW),
        "--tau_high", str(TAU_HIGH),
        "--max_eval_samples", str(max_eval_samples),
        "--output_json", str(output_json),
        "--device", "mps",
    ]
    print(f"  [{condition} res={resolution} seed={seed}] {python_bin} ...")
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True,
                           capture_output=True)
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[-500:]}")
        raise SystemExit(f"Run failed with code {result.returncode}")
    if not output_json.is_file():
        raise SystemExit(f"Output JSON not found: {output_json}")
    return json.loads(output_json.read_text(encoding="utf-8"))


def extract_row(
    result: dict[str, Any],
    *,
    resolution: int,
    seed: int,
    condition: str,
) -> dict[str, Any]:
    """Extract a flat row from a run_suds_eval_mlx.py result."""
    config = result.get("config", {})
    cond_data = result.get(condition, {})
    perturb_stats = cond_data.get("perturb_stats", {})

    baseline_data = result.get("e0_dense", {})
    baseline_top1 = baseline_data.get("top1")

    top1 = cond_data.get("top1")
    delta = None
    if top1 is not None and baseline_top1 is not None and condition != "e0_dense":
        delta = round(top1 - baseline_top1, 6)
    elif condition == "e0_dense":
        delta = 0.0

    total_cols = (perturb_stats.get("total_pruned_columns", 0) +
                  perturb_stats.get("total_degraded_columns", 0) +
                  perturb_stats.get("total_kept_columns", 0))
    keep_ratio = perturb_stats.get("total_kept_columns", 0) / max(1, total_cols)
    degrade_ratio = perturb_stats.get("total_degraded_columns", 0) / max(1, total_cols)
    prune_ratio = perturb_stats.get("total_pruned_columns", 0) / max(1, total_cols)

    return {
        "tag": TAG,
        "roadmap_item": "R12e_mobilevit_resolution_sweep",
        "model": MODEL,
        "resolution": resolution,
        "seed": seed,
        "condition": condition,
        "top1": top1,
        "top5": cond_data.get("top5"),
        "processed_samples": cond_data.get("processed_samples"),
        "elapsed_s": cond_data.get("elapsed_s"),
        "device": config.get("device"),
        "mlx_default_device": config.get("mlx_default_device"),
        "mlx_metal_available": config.get("mlx_metal_available"),
        "git_hash": config.get("git_hash"),
        "tau_low": config.get("tau_low"),
        "tau_high": config.get("tau_high"),
        "nominal_input_size": config.get("nominal_input_size"),
        "eval_input_size": config.get("eval_input_size"),
        "keep_ratio": round(keep_ratio, 6) if total_cols > 0 else None,
        "degrade_ratio": round(degrade_ratio, 6) if total_cols > 0 else None,
        "prune_ratio": round(prune_ratio, 6) if total_cols > 0 else None,
        "delta_vs_baseline_pp": delta,
    }


def populate_pairwise_deltas(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Fill e8_overflow deltas from paired e0_dense rows.

    The per-run JSON emitted by run_suds_eval_mlx.py stores the baseline
    metrics only in the e0_dense file, while the promoted-policy file only
    stores the promoted condition. We therefore have to pair rows by
    (resolution, seed) before any acceptance or report logic can rely on the
    delta field.
    """
    baseline_by_pair = {
        (int(row["resolution"]), int(row["seed"])): row.get("top1")
        for row in rows
        if row["condition"] == "e0_dense" and row.get("top1") is not None
    }
    missing_pairs: list[tuple[int, int]] = []

    for row in rows:
        pair = (int(row["resolution"]), int(row["seed"]))
        if row["condition"] == "e0_dense":
            if row.get("delta_vs_baseline_pp") is None:
                row["delta_vs_baseline_pp"] = 0.0
            continue

        if row.get("delta_vs_baseline_pp") is not None:
            continue

        baseline_top1 = baseline_by_pair.get(pair)
        top1 = row.get("top1")
        if baseline_top1 is None or top1 is None:
            missing_pairs.append(pair)
            continue
        row["delta_vs_baseline_pp"] = round(top1 - baseline_top1, 6)

    return sorted(set(missing_pairs))


def load_run_json(run_dir: Path, resolution: int, seed: int, condition: str) -> dict[str, Any]:
    path = run_dir / f"run_res{resolution}_seed{seed}_{condition}.json"
    if not path.is_file():
        raise SystemExit(f"Run JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_acceptance(
    rows: list[dict[str, Any]],
    *,
    resolutions: list[int] | None = None,
    seeds: list[int] | None = None,
    conditions: list[str] | None = None,
    max_eval_samples: int = MAX_EVAL_SAMPLES,
) -> dict[str, Any]:
    """Check acceptance criteria."""
    missing_pairs = populate_pairwise_deltas(rows)
    resolutions = resolutions or RESOLUTIONS
    seeds = seeds or SEEDS
    conditions = conditions or CONDITIONS
    expected_count = len(resolutions) * len(seeds) * len(conditions)
    actual_count = len(rows)

    all_expected_samples = all(row.get("processed_samples") == max_eval_samples for row in rows)
    all_mps = all(
        str(row.get("device", "")).lower() == "mps" and
        "gpu" in str(row.get("mlx_default_device", "")).lower()
        for row in rows
    )

    # Compute worst mean delta across resolutions (for e8_overflow only)
    e8_rows = [row for row in rows if row["condition"] == "e8_overflow"]
    worst_delta = None
    worst_resolution = None
    worst_seed = None
    if e8_rows:
        # Per-resolution mean delta
        res_deltas = {}
        for res in resolutions:
            res_rows = [row for row in e8_rows if row["resolution"] == res]
            deltas = [row["delta_vs_baseline_pp"] for row in res_rows
                     if row["delta_vs_baseline_pp"] is not None]
            if deltas:
                res_deltas[res] = sum(deltas) / len(deltas)

        # Find worst individual run
        for row in e8_rows:
            d = row.get("delta_vs_baseline_pp")
            if d is not None and (worst_delta is None or d < worst_delta):
                worst_delta = d
                worst_resolution = row["resolution"]
                worst_seed = row["seed"]

        # Compute worst mean across resolutions
        worst_mean_delta = min(res_deltas.values()) if res_deltas else None
        worst_mean_res = min(res_deltas, key=res_deltas.get) if res_deltas else None
    else:
        worst_mean_delta = None
        worst_mean_res = None

    within_budget = (
        worst_mean_delta is not None and abs(worst_mean_delta) <= ACCURACY_TARGET_PP
    )

    blockers = []
    if actual_count != expected_count:
        blockers.append(f"row_count_{actual_count}_vs_expected_{expected_count}")
    if not all_expected_samples:
        blockers.append(f"not_all_{max_eval_samples}_samples")
    if not all_mps:
        blockers.append("not_all_mps_gpu")
    if missing_pairs:
        blockers.append(
            "missing_pairwise_deltas: "
            + ", ".join(f"res={res} seed={seed}" for res, seed in missing_pairs)
        )

    if not blockers:
        if within_budget:
            status = "pass"
        else:
            status = "boundary_recorded"
    else:
        status = "fail"

    return {
        "acceptance_state": status,
        "expected_rows": expected_count,
        "actual_rows": actual_count,
        "expected_samples_per_run": max_eval_samples,
        "all_expected_samples": all_expected_samples,
        "all_50000_samples": all_expected_samples if max_eval_samples == 50000 else False,
        "all_mps_gpu": all_mps,
        "accuracy_target_pp": ACCURACY_TARGET_PP,
        "worst_mean_delta_pp": worst_mean_delta,
        "worst_mean_resolution": worst_mean_res,
        "worst_single_delta_pp": worst_delta,
        "worst_single_resolution": worst_resolution,
        "worst_single_seed": worst_seed,
        "within_budget": within_budget,
        "blockers": blockers,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    *,
    tag: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    acceptance: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_tetc_mobilevit_resolution_accuracy_{tag}",
            "roadmap_item": "R12e_mobilevit_resolution_sweep",
            "evidence_label": "measured_mps_imagenet_resolution_sweep",
            "regeneration_command": "make suds-tetc-mobilevit-resolution-sweep",
            "git_hash": git_hash(),
            "model": MODEL,
            "resolutions": args.resolutions,
            "seeds": args.seeds,
            "conditions": CONDITIONS,
            "tau_low": TAU_LOW,
            "tau_high": TAU_HIGH,
            "max_eval_samples": args.max_eval_samples,
            "imagenet_val": args.imagenet_val,
        },
        "acceptance": acceptance,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    tag: str,
    rows: list[dict[str, Any]],
    acceptance: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    populate_pairwise_deltas(rows)

    e8_rows = [row for row in rows if row["condition"] == "e8_overflow"]
    e0_rows = [row for row in rows if row["condition"] == "e0_dense"]

    # Per-resolution summary
    res_lines = []
    report_resolutions = sorted({int(row["resolution"]) for row in rows}) or RESOLUTIONS
    report_seeds = sorted({int(row["seed"]) for row in rows}) or SEEDS
    for res in report_resolutions:
        res_e0 = [row for row in e0_rows if row["resolution"] == res]
        res_e8 = [row for row in e8_rows if row["resolution"] == res]
        e0_top1s = [row["top1"] for row in res_e0 if row["top1"] is not None]
        e8_top1s = [row["top1"] for row in res_e8 if row["top1"] is not None]
        e8_deltas = [row["delta_vs_baseline_pp"] for row in res_e8
                    if row["delta_vs_baseline_pp"] is not None]

        e0_mean = sum(e0_top1s) / len(e0_top1s) if e0_top1s else None
        e8_mean = sum(e8_top1s) / len(e8_top1s) if e8_top1s else None
        delta_mean = sum(e8_deltas) / len(e8_deltas) if e8_deltas else None

        if e0_mean is not None and e8_mean is not None and delta_mean is not None:
            res_lines.append(
                f"| `{res}` | {e0_mean:.2f}% | {e8_mean:.2f}% | "
                f"{delta_mean:+.4f} pp | {len(res_e0)}/{len(res_e8)} |"
            )
        else:
            res_lines.append(f"| `{res}` | n/a | n/a | n/a | {len(res_e0)}/{len(res_e8)} |")

    section = f"""## R12e MobileViT-S Resolution Sweep

Date: `{DATE}`
Tag: `{tag}`
Roadmap item: `R12e_mobilevit_resolution_sweep`

## MobileViT-S Resolution Accuracy Sweep

### Configuration

- Model: `{MODEL}` (nominal input size: 256)
- Resolutions: `{report_resolutions}`
- Seeds: `{report_seeds}`
- Conditions: `e0_dense` (baseline), `e8_overflow` (promoted SUDS policy)
- Promoted policy: tau_low=`{TAU_LOW}`, tau_high=`{TAU_HIGH}`
- Samples per run: `{acceptance['expected_samples_per_run']}` {"(full ImageNet val)" if acceptance['expected_samples_per_run'] == 50000 else "(smoke/partial run)"}
- Device: `mps` only, CPU fallback forbidden
- Total runs: `{len(rows)}`

### Per-Resolution Summary

| Resolution | e0_dense Top-1 (mean) | e8_overflow Top-1 (mean) | Mean Δ | Seeds (e0/e8) |
|---|---:|---:|---:|---|
{chr(10).join(res_lines)}

### Acceptance

- Acceptance state: `{acceptance['acceptance_state']}`
- Expected rows: `{acceptance['expected_rows']}`, Actual: `{acceptance['actual_rows']}`
- Expected samples per run: `{acceptance['expected_samples_per_run']}`
- All expected samples: `{acceptance['all_expected_samples']}`
- All 50000 samples: `{acceptance['all_50000_samples']}`
- All MPS/GPU: `{acceptance['all_mps_gpu']}`
- Accuracy target: `{ACCURACY_TARGET_PP}` pp
- Worst mean delta: `{acceptance.get('worst_mean_delta_pp')}` pp at resolution `{acceptance.get('worst_mean_resolution')}`
- Worst single delta: `{acceptance.get('worst_single_delta_pp')}` pp (res=`{acceptance.get('worst_single_resolution')}`, seed=`{acceptance.get('worst_single_seed')}`)
- Within budget: `{acceptance['within_budget']}`

### Interpretation

"""
    if acceptance["acceptance_state"] == "pass":
        section += (
            "MobileViT-S accuracy is resolution-stable under the promoted SUDS "
            "e8_overflow perturbation policy across all tested resolutions "
            f"({min(report_resolutions)}-{max(report_resolutions)}). The worst mean delta is within "
            f"the {ACCURACY_TARGET_PP} pp budget, supporting the claim that the "
            "SUDS policy does not depend on a single fixed input resolution."
        )
    else:
        section += (
            "MobileViT-S resolution sweep is recorded as boundary evidence. "
            "The worst mean delta exceeds or approaches the accuracy budget, "
            "indicating resolution sensitivity that should be acknowledged in "
            "the manuscript."
        )

    section += f"""

### Required Artifacts

- CSV: `experiments/results/report_data/suds_tetc_mobilevit_resolution_accuracy_{tag}.csv`
- JSON: `experiments/results/report_data/suds_tetc_mobilevit_resolution_accuracy_{tag}.json`

### Regeneration

```bash
make suds-tetc-mobilevit-resolution-sweep
```
"""
    merge_report_section(path, section)


def merge_report_section(path: Path, section: str) -> None:
    """Append or replace the R12e section without clobbering other R12 sections."""
    start = "<!-- R12E_MOBILEVIT_RESOLUTION_SWEEP_START -->"
    end = "<!-- R12E_MOBILEVIT_RESOLUTION_SWEEP_END -->"
    wrapped = f"{start}\n{section.rstrip()}\n{end}\n"
    if not path.exists():
        path.write_text(
            "# SUDS TETC R12 Deep Reinforcement\n\n" + wrapped,
            encoding="utf-8",
        )
        return
    text = path.read_text(encoding="utf-8")
    if start in text and end in text:
        before = text.split(start, 1)[0].rstrip()
        after = text.split(end, 1)[1].lstrip()
        path.write_text(f"{before}\n\n{wrapped}\n{after}".rstrip() + "\n", encoding="utf-8")
        return
    path.write_text(text.rstrip() + "\n\n" + wrapped, encoding="utf-8")


def main() -> int:
    args = parse_args()
    python_bin = args.python_bin or sys.executable

    args.run_output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_runs:
        print(f"R12e MobileViT-S resolution sweep: "
              f"{len(args.resolutions)} resolutions x {len(args.seeds)} seeds "
              f"x {len(CONDITIONS)} conditions = "
              f"{len(args.resolutions) * len(args.seeds) * len(CONDITIONS)} runs")
        print(f"Model: {MODEL}, tau=({TAU_LOW}, {TAU_HIGH}), "
              f"max_eval_samples={MAX_EVAL_SAMPLES}")
        print(f"Python: {python_bin}")
        print(f"ImageNet val: {args.imagenet_val}")
        print()

        for resolution in args.resolutions:
            for seed in args.seeds:
                for condition in CONDITIONS:
                    out_json = args.run_output_dir / f"run_res{resolution}_seed{seed}_{condition}.json"
                    if out_json.is_file():
                        print(f"  [{condition} res={resolution} seed={seed}] "
                              f"SKIP (existing: {out_json})")
                        continue
                    run_single(
                        python_bin=python_bin,
                        resolution=resolution,
                        seed=seed,
                        condition=condition,
                        output_json=out_json,
                        imagenet_val=args.imagenet_val,
                        max_eval_samples=args.max_eval_samples,
                    )

    # Aggregate results
    rows = []
    for resolution in args.resolutions:
        for seed in args.seeds:
            for condition in CONDITIONS:
                result = load_run_json(args.run_output_dir, resolution, seed, condition)
                rows.append(extract_row(result, resolution=resolution,
                                       seed=seed, condition=condition))

    acceptance = build_acceptance(
        rows,
        resolutions=args.resolutions,
        seeds=args.seeds,
        conditions=CONDITIONS,
        max_eval_samples=args.max_eval_samples,
    )

    write_csv(args.csv_out, rows)
    write_json(args.json_out, tag=args.tag, args=args, rows=rows,
              acceptance=acceptance)
    write_report(args.report_out, tag=args.tag, rows=rows,
                acceptance=acceptance)

    print(f"\nWrote {args.csv_out} ({len(rows)} rows)")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.report_out}")
    print(f"Acceptance state: {acceptance['acceptance_state']}")
    if acceptance.get("worst_mean_delta_pp") is not None:
        print(f"Worst mean delta: {acceptance['worst_mean_delta_pp']:.4f} pp "
              f"(res={acceptance['worst_mean_resolution']})")
    if acceptance.get("worst_single_delta_pp") is not None:
        print(f"Worst single delta: {acceptance['worst_single_delta_pp']:.4f} pp "
              f"(res={acceptance['worst_single_resolution']}, "
              f"seed={acceptance['worst_single_seed']})")

    if acceptance["acceptance_state"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
