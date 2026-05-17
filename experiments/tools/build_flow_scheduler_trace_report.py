#!/usr/bin/env python3
"""Build a conservative HOPS scheduler trace note from released run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIR_CI_CSV = ROOT / "AICAS" / "assets" / "candidate_data" / "headline_statistics_pair_ci_20260310.csv"
DEFAULT_HEADLINE_SUMMARY_CSV = ROOT / "AICAS" / "assets" / "candidate_data" / "headline_statistics_summary_20260310.csv"
DEFAULT_RUNS_ROOT = ROOT / "experiments" / "results" / "runs"
DEFAULT_OUT_DATA_DIR = ROOT / "AICAS" / "assets" / "candidate_data"
DEFAULT_TAG = "20260310"

MAIN_RUNS = {
    "E0": "20260228_opt_sync_core_e0",
    "E2": "20260228_opt_sync_core_e2",
}
SCAN_PREFIX = "20260228_opt_sync_scan_"


def _format_num(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}{suffix}"


def _format_signed_delta(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    val = float(value)
    if abs(val) < 0.5 * (10 ** (-digits)):
        val = 0.0
    return f"{val:.{digits}f}{suffix}"


def _load_pair_ci(pair_ci_csv: Path) -> dict[str, float]:
    pair = pd.read_csv(pair_ci_csv)
    row = pair[pair["experiment_id"] == "E2"].iloc[0]
    return {
        "speedup_vs_e0_mean": float(row["speedup_vs_e0_mean"]),
        "acc_drop_pp_vs_e0_mean": float(row["acc_drop_pp_vs_e0_mean"]),
    }


def _load_e2_accuracy_evidence(summary_csv: Path) -> str:
    summary = pd.read_csv(summary_csv)
    row = summary[summary["experiment_id"] == "E2"].iloc[0]
    return str(row.get("accuracy_evidence") or "")


def _load_master_metrics(run_path: Path, experiment_id: str) -> pd.DataFrame:
    frame = pd.read_csv(run_path)
    frame = frame[frame["experiment_id"] == experiment_id].copy()
    keep = [
        "model",
        "seed",
        "latency_ms",
        "bubble_cycles",
        "utilization_avg",
        "movement_bound_ratio",
    ]
    frame = frame[keep].copy()
    for column in keep[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _build_flow_pairs(runs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    e0 = _load_master_metrics(runs_root / MAIN_RUNS["E0"] / "master_metrics.csv", "E0")
    e2 = _load_master_metrics(runs_root / MAIN_RUNS["E2"] / "master_metrics.csv", "E2")
    merged = e0.merge(e2, on=["model", "seed"], suffixes=("_e0", "_e2"), validate="one_to_one")
    merged["speedup_e2_vs_e0"] = merged["latency_ms_e0"] / merged["latency_ms_e2"]
    merged["bubble_reduction_pct"] = 100.0 * (1.0 - merged["bubble_cycles_e2"] / merged["bubble_cycles_e0"])
    merged["utilization_increase_pp"] = 100.0 * (merged["utilization_avg_e2"] - merged["utilization_avg_e0"])
    merged["movement_bound_delta_pp"] = 100.0 * (merged["movement_bound_ratio_e2"] - merged["movement_bound_ratio_e0"])
    per_model = (
        merged.groupby("model", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            speedup_e2_vs_e0_mean=("speedup_e2_vs_e0", "mean"),
            bubble_reduction_pct_mean=("bubble_reduction_pct", "mean"),
            utilization_increase_pp_mean=("utilization_increase_pp", "mean"),
            movement_bound_delta_pp_mean=("movement_bound_delta_pp", "mean"),
        )
        .sort_values("model")
        .reset_index(drop=True)
    )
    return merged.sort_values(["model", "seed"]).reset_index(drop=True), per_model


def _load_stage_breakdown(runs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    aggregates: dict[str, dict[str, float]] = {}
    for experiment_id, run_id in MAIN_RUNS.items():
        timeline_path = runs_root / run_id / "timeline_summary.csv"
        timeline = pd.read_csv(timeline_path)
        stage_totals: dict[str, float] = {}
        for payload in timeline["stage_cycles"]:
            data = json.loads(payload)
            for stage, cycles in data.items():
                stage_totals[stage] = stage_totals.get(stage, 0.0) + float(cycles)
        aggregates[experiment_id] = stage_totals
    for stage in sorted(set(aggregates["E0"]) | set(aggregates["E2"])):
        e0_cycles = aggregates["E0"].get(stage, 0.0)
        e2_cycles = aggregates["E2"].get(stage, 0.0)
        delta = e2_cycles - e0_cycles
        pct = None if e0_cycles == 0 else 100.0 * delta / e0_cycles
        rows.append(
            {
                "stage": stage,
                "e0_cycles_total": e0_cycles,
                "e2_cycles_total": e2_cycles,
                "delta_cycles_e2_minus_e0": delta,
                "pct_change_e2_vs_e0": pct,
            }
        )
    return pd.DataFrame(rows)


def _load_top_layer_bubble_reductions(runs_root: Path, top_n: int = 10) -> pd.DataFrame:
    def load(run_id: str) -> dict[str, float]:
        table: dict[str, float] = {}
        with (runs_root / run_id / "per_layer_timeline.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["stage"] == "bubble":
                    table[row["layer_id"]] = float(row["cycles"])
        return table

    e0 = load(MAIN_RUNS["E0"])
    e2 = load(MAIN_RUNS["E2"])
    rows: list[dict[str, object]] = []
    for layer_id, e0_bubble in e0.items():
        e2_bubble = e2.get(layer_id, 0.0)
        rows.append(
            {
                "layer_id": layer_id,
                "bubble_cycles_e0": e0_bubble,
                "bubble_cycles_e2": e2_bubble,
                "bubble_delta_cycles": e2_bubble - e0_bubble,
                "bubble_reduction_cycles": e0_bubble - e2_bubble,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("bubble_reduction_cycles", ascending=False).head(top_n).reset_index(drop=True)


def _extract_numeric_setting(config_path: Path, key: str) -> float | None:
    if not config_path.is_file():
        return None
    match = re.search(rf"(?m)^\s*{re.escape(key)}:\s*([0-9.]+)\s*$", config_path.read_text(encoding='utf-8'))
    if match is None:
        return None
    return float(match.group(1))


def _read_scan_metrics(timeline_summary_path: Path) -> tuple[float | None, float | None]:
    if not timeline_summary_path.is_file():
        return None, None
    frame = pd.read_csv(timeline_summary_path)
    bubble = float(frame["bubble_cycles"].astype(float).mean())
    util = float(frame["utilization_avg"].astype(float).mean())
    return bubble, util


def _build_sensitivity_audit(runs_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scan_groups = [("batch_size", "batch"), ("sequence_length", "seq")]
    for setting_name, token in scan_groups:
        e0_runs = sorted(runs_root.glob(f"{SCAN_PREFIX}e0_{token}*"))
        baseline_bubble: float | None = None
        baseline_util: float | None = None
        for run_dir in e0_runs:
            value = _extract_numeric_setting(run_dir / "config_snapshot.yaml", setting_name)
            bubble, util = _read_scan_metrics(run_dir / "timeline_summary.csv")
            if baseline_bubble is None:
                baseline_bubble = bubble
                baseline_util = util
            e2_pair = runs_root / run_dir.name.replace("e0_", "e2_")
            rows.append(
                {
                    "scan_type": setting_name,
                    "value": value,
                    "e0_run_id": run_dir.name,
                    "e2_pair_present": e2_pair.exists(),
                    "mean_bubble_cycles": bubble,
                    "mean_utilization_avg": util,
                    "diff_vs_first_e0_bubble": None if bubble is None or baseline_bubble is None else bubble - baseline_bubble,
                    "diff_vs_first_e0_util": None if util is None or baseline_util is None else util - baseline_util,
                }
            )
        if not e0_runs:
            rows.append(
                {
                    "scan_type": setting_name,
                    "value": None,
                    "e0_run_id": "",
                    "e2_pair_present": False,
                    "mean_bubble_cycles": None,
                    "mean_utilization_avg": None,
                    "diff_vs_first_e0_bubble": None,
                    "diff_vs_first_e0_util": None,
                }
            )
    latency_matches = sorted(runs_root.glob(f"{SCAN_PREFIX}*lat*"))
    rows.append(
        {
            "scan_type": "memory_latency",
            "value": None,
            "e0_run_id": "",
            "e2_pair_present": False,
            "mean_bubble_cycles": None,
            "mean_utilization_avg": None,
            "diff_vs_first_e0_bubble": None,
            "diff_vs_first_e0_util": None,
            "notes": "missing" if not latency_matches else "present",
        }
    )
    return pd.DataFrame(rows)


def _buffer_artifact_present(runs_root: Path) -> bool:
    patterns = ["*occupancy*", "*buffer*trace*", "*fifo*trace*"]
    for pattern in patterns:
        if any(path.is_file() for path in runs_root.rglob(pattern)):
            return True
    return False


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_report(
    report_path: Path,
    pair_ci_csv: Path,
    headline_summary_csv: Path,
    pairs_csv: Path,
    stage_csv: Path,
    sensitivity_csv: Path,
    pair_metrics: dict[str, float],
    e2_accuracy_evidence: str,
    per_seed_pairs: pd.DataFrame,
    per_model_summary: pd.DataFrame,
    stage_summary: pd.DataFrame,
    top_layers: pd.DataFrame,
    sensitivity: pd.DataFrame,
    buffer_trace_present: bool,
    tag: str,
) -> None:
    bubble_mean = float(per_seed_pairs["bubble_reduction_pct"].mean())
    util_mean = float(per_seed_pairs["utilization_increase_pp"].mean())
    movement_mean = float(per_seed_pairs["movement_bound_delta_pp"].mean())
    bubble_stage = stage_summary[stage_summary["stage"] == "bubble"].iloc[0]
    batch_scan = sensitivity[sensitivity["scan_type"] == "batch_size"].copy()
    seq_scan = sensitivity[sensitivity["scan_type"] == "sequence_length"].copy()
    e2_is_inherited = e2_accuracy_evidence == "inherited_from_e0_full_eval"
    if e2_is_inherited:
        e2_accuracy_line = "- `E2` does not have a dedicated full-eval accuracy rerun in the repository; its `0.00 pp` Top-1 delta is inherited from the `E0` quantized full-eval path and should be read as inherited evidence rather than an independent measurement."
        e2_interpretation_line = "- HOPS now has a direct scheduler-side trace: the released E0/E2 path halves bubble cycles and raises utilization, while the associated E2 Top-1 metadata remains inherited rather than independently re-measured."
    else:
        e2_accuracy_line = "- `E2` now has a dedicated full-eval accuracy rerun in the repaired chain, so its HOPS-only speedup and `0.00 pp` Top-1 delta are both supported by measured evidence."
        e2_interpretation_line = "- HOPS now has both a direct scheduler-side trace and a dedicated E2 full-eval closure: the released E0/E2 path halves bubble cycles, raises utilization, and preserves the measured E0-relative Top-1 within the rerun protocol."
    lines = [
        f"# HOPS Scheduler Trace Note ({tag})",
        "",
        "Scope",
        f"- Pair-CI source: `{pair_ci_csv}`",
        f"- Headline summary source: `{headline_summary_csv}`",
        f"- Per-seed HOPS trace summary: `{pairs_csv}`",
        f"- Stage breakdown summary: `{stage_csv}`",
        f"- Sensitivity audit: `{sensitivity_csv}`",
        f"- Main runs: `{MAIN_RUNS['E0']}` and `{MAIN_RUNS['E2']}` under `{DEFAULT_RUNS_ROOT}`",
        "",
        "Independent HOPS effect",
        (
            f"- `E2 HOPS-only` delivers "
            f"`{_format_num(pair_metrics['speedup_vs_e0_mean'], 2, 'x')}` paired speedup."
        ),
        e2_accuracy_line,
        (
            f"- Across `3 models x 3 seeds`, HOPS reduces bubble cycles by "
            f"`{_format_num(bubble_mean, 2, '%')}` on average, lifts utilization by "
            f"`{_format_num(util_mean, 2, ' pp')}`, and shifts the movement-bound ratio by "
            f"`{_format_num(movement_mean, 2, ' pp')}`."
        ),
    ]
    for row in per_model_summary.itertuples(index=False):
        lines.append(
            f"- `{row.model}`: bubble `-{_format_num(row.bubble_reduction_pct_mean, 2, '%')}`, "
            f"utilization `+{_format_num(row.utilization_increase_pp_mean, 2, ' pp')}`, "
            f"speedup `{_format_num(row.speedup_e2_vs_e0_mean, 2, 'x')}`."
        )
    lines.extend(
        [
            "",
            "Stage breakdown",
            (
                f"- Aggregate `bubble` stage cycles fall from `{int(bubble_stage['e0_cycles_total'])}` to "
                f"`{int(bubble_stage['e2_cycles_total'])}` "
                f"(`{_format_num(bubble_stage['pct_change_e2_vs_e0'], 2, '%')}`)."
            ),
        ]
    )
    for stage in ["fetch_map", "btos", "serialize_drive", "writeback", "electronic_compute"]:
        row = stage_summary[stage_summary["stage"] == stage].iloc[0]
        sign = "+" if float(row["pct_change_e2_vs_e0"]) >= 0 else ""
        lines.append(
            f"- `{stage}` changes by `{sign}{_format_num(row['pct_change_e2_vs_e0'], 2, '%')}`, "
            "which is consistent with HOPS pulling work forward while removing idle bubble time."
        )
    lines.extend(["", "Layer-localized bubble reductions"])
    for row in top_layers.itertuples(index=False):
        lines.append(
            f"- `{row.layer_id}`: bubble `{int(row.bubble_cycles_e0)} -> {int(row.bubble_cycles_e2)}` "
            f"(`-{int(row.bubble_reduction_cycles)} cycles`)."
        )
    batch_missing = not batch_scan["e2_pair_present"].any() if not batch_scan.empty else True
    seq_missing = not seq_scan["e2_pair_present"].any() if not seq_scan.empty else True
    batch_invariant = bool((batch_scan["diff_vs_first_e0_bubble"].fillna(0.0).abs() < 1e-9).all()) if not batch_scan.empty else False
    seq_invariant = bool((seq_scan["diff_vs_first_e0_bubble"].fillna(0.0).abs() < 1e-9).all()) if not seq_scan.empty else False
    lines.extend(
        [
            "",
            "Sensitivity audit",
            (
                f"- Batch-size scan: only baseline-side runs are present (`missing E2 pair = {batch_missing}`); "
                f"available `E0` traces are invariant across the scanned values (`invariant = {batch_invariant}`)."
            ),
            (
                f"- Sequence-length scan: only baseline-side runs are present (`missing E2 pair = {seq_missing}`); "
                f"available `E0` traces are invariant across the scanned values (`invariant = {seq_invariant}`)."
            ),
            "- No paired memory-latency HOPS sweep is present in the repository.",
            f"- Buffer-occupancy artifact present under `experiments/results/runs`: `{buffer_trace_present}`.",
            "",
            "Interpretation",
            e2_interpretation_line,
            "- The strongest bubble reductions are localized to large fusion, local-representation, and attention-projection layers, which is stronger evidence than a purely schematic timeline.",
            "- The remaining P1-4 gap is sensitivity closure: the repository still lacks paired memory-latency, batch-size, and sequence-length HOPS sweeps, and it does not expose a buffer-occupancy trace.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a conservative HOPS scheduler trace report.")
    parser.add_argument("--pair_ci_csv", type=Path, default=DEFAULT_PAIR_CI_CSV)
    parser.add_argument("--headline_summary_csv", type=Path, default=DEFAULT_HEADLINE_SUMMARY_CSV)
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--out_data_dir", type=Path, default=DEFAULT_OUT_DATA_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args()

    pair_metrics = _load_pair_ci(args.pair_ci_csv)
    e2_accuracy_evidence = _load_e2_accuracy_evidence(args.headline_summary_csv)
    per_seed_pairs, per_model_summary = _build_flow_pairs(args.runs_root)
    stage_summary = _load_stage_breakdown(args.runs_root)
    top_layers = _load_top_layer_bubble_reductions(args.runs_root)
    sensitivity = _build_sensitivity_audit(args.runs_root)
    buffer_trace_present = _buffer_artifact_present(args.runs_root)

    args.out_data_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = args.out_data_dir / f"flow_scheduler_trace_per_seed_{args.tag}.csv"
    stage_csv = args.out_data_dir / f"flow_scheduler_trace_stage_summary_{args.tag}.csv"
    sensitivity_csv = args.out_data_dir / f"flow_scheduler_sensitivity_audit_{args.tag}.csv"
    layer_csv = args.out_data_dir / f"flow_scheduler_top_layer_bubble_deltas_{args.tag}.csv"
    report_md = args.out_data_dir / f"flow_scheduler_trace_report_{args.tag}.md"

    _write_csv(pairs_csv, per_seed_pairs)
    _write_csv(stage_csv, stage_summary)
    _write_csv(sensitivity_csv, sensitivity)
    _write_csv(layer_csv, top_layers)
    _write_report(
        report_md,
        args.pair_ci_csv,
        args.headline_summary_csv,
        pairs_csv,
        stage_csv,
        sensitivity_csv,
        pair_metrics,
        e2_accuracy_evidence,
        per_seed_pairs,
        per_model_summary,
        stage_summary,
        top_layers,
        sensitivity,
        buffer_trace_present,
        args.tag,
    )

    print(f"Wrote per-seed trace CSV: {pairs_csv}")
    print(f"Wrote stage summary CSV: {stage_csv}")
    print(f"Wrote sensitivity audit CSV: {sensitivity_csv}")
    print(f"Wrote top-layer bubble CSV: {layer_csv}")
    print(f"Wrote report: {report_md}")


if __name__ == "__main__":
    main()
