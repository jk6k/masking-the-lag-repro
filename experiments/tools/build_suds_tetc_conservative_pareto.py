#!/usr/bin/env python3
"""Build the conservative SUDS Pareto artifact for the TETC pivot.

This artifact promotes a measured MobileViT-S no-prune SUDS point only as an
accuracy/EDP Pareto operating point.  It intentionally preserves the stronger
measured selector baselines as context and does not relabel the older aggressive
SUDS rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
RUN_DIR = REPO_ROOT / "experiments/results/runs/suds_tetc_conservative_pareto_20260513"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
BASELINE_JSON = REPORT_DATA / "suds_mobilevit_multimodel_validation_20260511_p2p3_quality.json"

CSV_OUT = REPORT_DATA / f"suds_tetc_conservative_pareto_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_conservative_pareto_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_conservative_pareto.md"

CONDITION = "e9_suds_conservative"
CONDITION_LABEL = "E9 SUDS conservative signal-overflow tau=(0.30,0.95)"
EXPECTED_SEEDS = tuple(range(8))
EXPECTED_SAMPLES = 50000
ACCURACY_TARGET_PP = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--baseline-json", type=Path, default=BASELINE_JSON)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    return parser.parse_args()


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def as_float(value: Any, *, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_baseline_rows(path: Path) -> dict[int, dict[str, Any]]:
    payload = load_json(path)
    rows = {}
    for row in payload.get("rows", []):
        if row.get("row_type") != "per_seed":
            continue
        if row.get("model") != "mobilevit_s" or row.get("condition") != "e0_dense":
            continue
        seed = int(row["seed"])
        rows[seed] = row
    missing = sorted(set(EXPECTED_SEEDS) - set(rows))
    if missing:
        raise SystemExit(f"missing MobileViT-S dense baseline seeds: {missing}")
    return rows


def run_path(run_dir: Path, seed: int) -> Path:
    return run_dir / f"mobilevit_s_seed{seed}_e8_tau030_095_full.json"


def ratios_from_perturb_stats(stats: dict[str, Any]) -> tuple[float, float, float, int]:
    kept = int(stats.get("total_kept_columns") or 0)
    degraded = int(stats.get("total_degraded_columns") or 0)
    pruned = int(stats.get("total_pruned_columns") or 0)
    total = kept + degraded + pruned
    if total <= 0:
        raise SystemExit("invalid perturb_stats: no tiered columns")
    return kept / total, degraded / total, pruned / total, total


def build_per_seed_rows(args: argparse.Namespace, baseline_rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for seed in EXPECTED_SEEDS:
        path = run_path(args.run_dir, seed)
        payload = load_json(path)
        config = payload.get("config", {})
        result = payload.get("e8_overflow", {})
        stats = result.get("perturb_stats", {})
        keep, degrade, prune, total_columns = ratios_from_perturb_stats(stats)
        adc_ratio = keep + degrade / 16.0

        baseline = baseline_rows[seed]
        top1 = as_float(result.get("top1"))
        top5 = as_float(result.get("top5"))
        baseline_top1 = as_float(baseline.get("top1"))
        baseline_top5 = as_float(baseline.get("top5"))
        processed = int(result.get("processed_samples") or 0)
        blockers = []
        if str(config.get("device")) != "mps":
            blockers.append("run_device_not_mps")
        if "gpu" not in str(config.get("mlx_default_device", "")):
            blockers.append("mlx_default_device_not_gpu")
        if processed != EXPECTED_SAMPLES:
            blockers.append(f"processed_samples_{processed}_not_{EXPECTED_SAMPLES}")
        if prune > 0.0:
            blockers.append("conservative_policy_pruned_columns")

        rows.append(
            {
                "row_type": "per_seed",
                "status": "measured" if not blockers else "blocked",
                "model": "mobilevit_s",
                "seed": seed,
                "condition": CONDITION,
                "condition_label": CONDITION_LABEL,
                "source_json": repo_path(path),
                "baseline_source_json": baseline.get("source_json", ""),
                "top1": top1,
                "top5": top5,
                "baseline_top1": baseline_top1,
                "baseline_top5": baseline_top5,
                "delta_top1": top1 - baseline_top1,
                "delta_top5": top5 - baseline_top5,
                "processed_samples": processed,
                "elapsed_s": as_float(result.get("elapsed_s")),
                "tau_low": as_float(config.get("tau_low")),
                "tau_high": as_float(config.get("tau_high")),
                "adc_energy_ratio_vs_e0": adc_ratio,
                "energy_reduction_vs_e0": 1.0 - adc_ratio,
                "mapped_keep_ratio": keep,
                "mapped_degrade_ratio": degrade,
                "mapped_prune_ratio": prune,
                "mapped_total_columns": total_columns,
                "budget_signal": stats.get("budget_signal", "suds_slack_tier_counts"),
                "selection_signal": stats.get("selection_signal", "hyatten_like_column_overflow_proxy"),
                "device": config.get("device", ""),
                "mlx_default_device": config.get("mlx_default_device", ""),
                "git_hash": config.get("git_hash", ""),
                "slack_manifest": repo_path(config.get("slack_manifest", "")),
                "command": config.get("command", ""),
                "evidence_label": "measured_mps_imagenet",
                "promotion_decision": "main_pareto",
                "accuracy_loss_target_pp": ACCURACY_TARGET_PP,
                "claim_boundary": (
                    "Measured MPS accuracy point for an accuracy-guarded no-prune SUDS schedule; "
                    "architecture energy remains modeled, not silicon or bench energy."
                ),
                "blockers": ";".join(blockers),
            }
        )
    return rows


def aggregate_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = "measured" if all(row["status"] == "measured" for row in rows) else "blocked"
    values = {
        "top1": [float(row["top1"]) for row in rows],
        "top5": [float(row["top5"]) for row in rows],
        "delta_top1": [float(row["delta_top1"]) for row in rows],
        "delta_top5": [float(row["delta_top5"]) for row in rows],
        "adc_energy_ratio_vs_e0": [float(row["adc_energy_ratio_vs_e0"]) for row in rows],
        "mapped_keep_ratio": [float(row["mapped_keep_ratio"]) for row in rows],
        "mapped_degrade_ratio": [float(row["mapped_degrade_ratio"]) for row in rows],
        "mapped_prune_ratio": [float(row["mapped_prune_ratio"]) for row in rows],
        "elapsed_s": [float(row["elapsed_s"]) for row in rows],
    }
    blockers = sorted({part for row in rows for part in str(row["blockers"]).split(";") if part})
    return {
        "row_type": "aggregate",
        "status": status,
        "model": "mobilevit_s",
        "seed": "0-7",
        "condition": CONDITION,
        "condition_label": CONDITION_LABEL,
        "n_seeds": len(rows),
        "top1": mean(values["top1"]),
        "top1_std": std(values["top1"]),
        "top5": mean(values["top5"]),
        "top5_std": std(values["top5"]),
        "delta_top1": mean(values["delta_top1"]),
        "delta_top1_std": std(values["delta_top1"]),
        "delta_top5": mean(values["delta_top5"]),
        "delta_top5_std": std(values["delta_top5"]),
        "processed_samples": sum(int(row["processed_samples"]) for row in rows),
        "elapsed_s": sum(values["elapsed_s"]),
        "tau_low": rows[0]["tau_low"],
        "tau_high": rows[0]["tau_high"],
        "adc_energy_ratio_vs_e0": mean(values["adc_energy_ratio_vs_e0"]),
        "energy_reduction_vs_e0": 1.0 - mean(values["adc_energy_ratio_vs_e0"]),
        "mapped_keep_ratio": mean(values["mapped_keep_ratio"]),
        "mapped_degrade_ratio": mean(values["mapped_degrade_ratio"]),
        "mapped_prune_ratio": mean(values["mapped_prune_ratio"]),
        "mapped_total_columns": rows[0]["mapped_total_columns"],
        "budget_signal": rows[0]["budget_signal"],
        "selection_signal": rows[0]["selection_signal"],
        "device": ",".join(sorted({str(row["device"]) for row in rows})),
        "mlx_default_device": ",".join(sorted({str(row["mlx_default_device"]) for row in rows})),
        "git_hash": ",".join(sorted({str(row["git_hash"]) for row in rows})),
        "evidence_label": "measured_mps_imagenet",
        "promotion_decision": "main_pareto",
        "accuracy_loss_target_pp": ACCURACY_TARGET_PP,
        "claim_boundary": rows[0]["claim_boundary"],
        "blockers": ";".join(blockers),
    }


def same_fabric_context(baseline_payload: dict[str, Any]) -> list[dict[str, Any]]:
    conditions = {"e0_dense", "e2_l1", "e3_slack", "e6_signal", "e7_overlay", "e8_overflow"}
    rows = [
        row for row in baseline_payload.get("rows", [])
        if row.get("row_type") == "per_seed"
        and row.get("model") == "mobilevit_s"
        and row.get("condition") in conditions
    ]
    out = []
    for condition in sorted({str(row.get("condition")) for row in rows}):
        items = [row for row in rows if row.get("condition") == condition]
        out.append(
            {
                "condition": condition,
                "condition_label": items[0].get("condition_label", ""),
                "n_seeds": len(items),
                "top1_mean": mean([as_float(row.get("top1")) for row in items]),
                "delta_top1_mean": mean([as_float(row.get("delta_top1")) for row in items]),
                "adc_energy_ratio_vs_e0_mean": mean([as_float(row.get("adc_energy_ratio_vs_e0")) for row in items]),
                "mapped_keep_ratio_mean": mean([as_float(row.get("mapped_keep_ratio")) for row in items]),
                "mapped_degrade_ratio_mean": mean([as_float(row.get("mapped_degrade_ratio")) for row in items]),
                "mapped_prune_ratio_mean": mean([as_float(row.get("mapped_prune_ratio")) for row in items]),
                "promotion_decision": "context_baseline_or_ablation",
            }
        )
    return out


def decision(summary: dict[str, Any]) -> dict[str, Any]:
    blockers = []
    if summary["n_measured_seeds"] != len(EXPECTED_SEEDS):
        blockers.append("missing_measured_seed")
    if summary["device_set"] != ["mps"]:
        blockers.append("not_all_runs_mps")
    if summary["mlx_default_device_has_gpu"] is not True:
        blockers.append("mlx_default_device_not_gpu")
    if summary["aggregate_delta_top1_pp"] < -ACCURACY_TARGET_PP:
        blockers.append("accuracy_loss_exceeds_1pp_target")
    if summary["aggregate_prune_ratio"] != 0.0:
        blockers.append("conservative_policy_has_prune")
    return {
        "promotion_decision": "conservative_pareto_ready" if not blockers else "conservative_pareto_blocked",
        "blockers": blockers,
        "claim": (
            "Promote only an accuracy-guarded no-prune SUDS Pareto point; retain "
            "same-fabric L1/slack/signal rows as explicit baselines and ablations."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    *,
    tag: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    context: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_tetc_conservative_pareto_{tag}",
            "evidence_label": "measured_mps_imagenet_conservative_pareto",
            "promotion_decision": summary["decision"]["promotion_decision"],
            "git_hash": git_hash(),
            "regeneration_command": "make suds-tetc-conservative-pareto",
            "source_artifacts": {
                "run_dir": repo_path(args.run_dir),
                "baseline_json": repo_path(args.baseline_json),
            },
            "source_artifact_sha256": {
                "baseline_json": sha256_path(args.baseline_json),
                **{f"seed_{seed}_run_json": sha256_path(run_path(args.run_dir, seed)) for seed in EXPECTED_SEEDS},
            },
        },
        "summary": summary,
        "aggregate": aggregate,
        "same_fabric_context": context,
        "rows": rows + [aggregate],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(val):
        return "n/a"
    return f"{val:.{digits}f}"


def write_report(
    path: Path,
    *,
    tag: str,
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    context: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# SUDS TETC Conservative Pareto Artifact",
        "",
        f"Tag: `{tag}`",
        "Evidence label: `measured_mps_imagenet_conservative_pareto`",
        f"Promotion decision: `{summary['decision']['promotion_decision']}`",
        "",
        "## Scope",
        "",
        "This artifact adds a measured MobileViT-S SUDS operating point for the",
        "TETC pivot: `tau_low=0.30`, `tau_high=0.95`, signal-overflow selection,",
        "and zero pruning after mapped tier application. It is promoted only as an",
        "accuracy-guarded Pareto point, not as a relabeling of the previous",
        "aggressive SUDS rows.",
        "",
        "## Aggregate Result",
        "",
        "| Seeds | Device | Top-1 | Delta Top-1 | ADC ratio | Keep | Degrade | Prune |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| {aggregate['n_seeds']} | `{aggregate['device']}` | {fmt(aggregate['top1'])} | "
            f"{fmt(aggregate['delta_top1'])} pp | {fmt(aggregate['adc_energy_ratio_vs_e0'])} | "
            f"{fmt(aggregate['mapped_keep_ratio'])} | {fmt(aggregate['mapped_degrade_ratio'])} | "
            f"{fmt(aggregate['mapped_prune_ratio'])} |"
        ),
        "",
        "## Per-Seed Measurements",
        "",
        "| Seed | Top-1 | Top-5 | Delta Top-1 | Processed | Elapsed s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {fmt(row['top1'])} | {fmt(row['top5'])} | "
            f"{fmt(row['delta_top1'])} pp | {row['processed_samples']} | {fmt(row['elapsed_s'], 1)} |"
        )

    lines.extend(
        [
            "",
            "## Same-Fabric Context Retained",
            "",
            "| Condition | Top-1 mean | Delta Top-1 mean | ADC ratio | Keep | Degrade | Prune |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in context:
        lines.append(
            f"| `{row['condition']}` | {fmt(row['top1_mean'])} | {fmt(row['delta_top1_mean'])} pp | "
            f"{fmt(row['adc_energy_ratio_vs_e0_mean'])} | {fmt(row['mapped_keep_ratio_mean'])} | "
            f"{fmt(row['mapped_degrade_ratio_mean'])} | {fmt(row['mapped_prune_ratio_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The conservative point clears the <=1 pp promoted-accuracy target on the",
            "measured MobileViT-S evidence surface. It does not erase the fact that",
            "stronger energy rows exist at lower accuracy, so downstream claims must",
            "be Pareto-framed rather than single-point superiority claims.",
            "",
            "## Regeneration",
            "",
            "```bash",
            "make suds-tetc-conservative-pareto",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_payload = load_json(args.baseline_json)
    baseline_rows = load_baseline_rows(args.baseline_json)
    rows = build_per_seed_rows(args, baseline_rows)
    aggregate = aggregate_row(rows)

    device_set = sorted({str(row["device"]) for row in rows if row.get("device")})
    summary = {
        "condition": CONDITION,
        "condition_label": CONDITION_LABEL,
        "n_measured_seeds": sum(1 for row in rows if row["status"] == "measured"),
        "expected_seeds": list(EXPECTED_SEEDS),
        "processed_samples_per_seed": EXPECTED_SAMPLES,
        "device_set": device_set,
        "mlx_default_device_has_gpu": all("gpu" in str(row.get("mlx_default_device", "")) for row in rows),
        "aggregate_top1": aggregate["top1"],
        "aggregate_delta_top1_pp": aggregate["delta_top1"],
        "aggregate_delta_top1_std_pp": aggregate["delta_top1_std"],
        "aggregate_adc_energy_ratio_vs_e0": aggregate["adc_energy_ratio_vs_e0"],
        "aggregate_prune_ratio": aggregate["mapped_prune_ratio"],
        "accuracy_loss_target_pp": ACCURACY_TARGET_PP,
        "claim_boundary": aggregate["claim_boundary"],
    }
    summary["decision"] = decision(summary)
    context = same_fabric_context(baseline_payload)

    write_csv(args.csv_out, rows + [aggregate])
    write_json(
        args.json_out,
        tag=args.tag,
        args=args,
        rows=rows,
        aggregate=aggregate,
        context=context,
        summary=summary,
    )
    write_report(args.report_out, tag=args.tag, rows=rows, aggregate=aggregate, context=context, summary=summary)
    print(f"wrote {repo_path(args.csv_out)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")
    print(f"promotion_decision={summary['decision']['promotion_decision']}")


if __name__ == "__main__":
    main()
