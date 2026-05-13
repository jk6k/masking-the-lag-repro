#!/usr/bin/env python3
"""Build the R3 end-to-end perturbation accuracy artifact.

This generator does not rerun model evaluation.  It hardens the already
governed MPS accuracy runs by joining their forward-pass perturbation metadata
to the TETC architecture rows and R2 scheduler trace IDs.  The resulting
CSV/JSON/report is the audit surface for whether promoted accuracy deltas are
tied to the same KEEP/DEGRADE/PRUNE budget that drives modeled PPA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
DATE = "2026-05-13"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

ARCH_SUMMARY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_summary.csv"
ARCH_JSON = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}.json"
SCHEDULER_TRACE_CSV = REPORT_DATA / f"suds_tetc_scheduler_traces_{TAG}.csv"
GLUE_JSON = REPORT_DATA / "suds_glue_measured_validation_20260511_p2p3_quality.json"
GLUE_LINKAGE_CSV = REPORT_DATA / f"suds_glue_architecture_linkage_{TAG}.csv"
MOBILEVIT_JSON = REPORT_DATA / "suds_mobilevit_multimodel_validation_20260511_p2p3_quality.json"
CONSERVATIVE_PARETO_JSON = REPORT_DATA / f"suds_tetc_conservative_pareto_{TAG}.json"
CONSERVATIVE_PARETO_CSV = REPORT_DATA / f"suds_tetc_conservative_pareto_{TAG}.csv"

CSV_OUT = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_end_to_end_accuracy.md"

PROMOTED_CONDITION = "suds_pareto"
ACCURACY_TARGET_PP = 1.0
RATIO_TOLERANCE = 5e-4
DEFAULT_DEGRADE_NOISE_STD = 0.003
DEFAULT_PRUNE_NOISE_STD = 0.05

MOBILEVIT_SOURCE_CONDITION = {
    "lightening_dptc": "e0_dense",
    "uniform_8bit": "e0_dense",
    "random": "e5_random",
    "l1": "e2_l1",
    "slack_only": "e3_slack",
    "signal_only": "e6_signal",
    "suds_only": "e4_suds",
    "suds_pareto": "e9_suds_conservative",
    "suds_l1": "e7_overlay",
    "suds_signal": "e8_overflow",
}

BERT_FALLBACK_SOURCE_CONDITION = {
    "lightening_dptc": "e0_dense",
    "uniform_8bit": "e0_dense",
    "l1": "e2_l1",
    "slack_only": "e3_slack",
    "signal_only": "e6_signal",
    "suds_only": "e4_suds",
    "suds_pareto": "e2_l1",
    "suds_l1": "e7_overlay",
    "suds_signal": "e8_overflow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--architecture-summary-csv", type=Path, default=ARCH_SUMMARY_CSV)
    parser.add_argument("--architecture-json", type=Path, default=ARCH_JSON)
    parser.add_argument("--scheduler-trace-csv", type=Path, default=SCHEDULER_TRACE_CSV)
    parser.add_argument("--glue-json", type=Path, default=GLUE_JSON)
    parser.add_argument("--glue-linkage-csv", type=Path, default=GLUE_LINKAGE_CSV)
    parser.add_argument("--mobilevit-json", type=Path, default=MOBILEVIT_JSON)
    parser.add_argument("--conservative-pareto-json", type=Path, default=CONSERVATIVE_PARETO_JSON)
    parser.add_argument("--conservative-pareto-csv", type=Path, default=CONSERVATIVE_PARETO_CSV)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--accuracy-target-pp", type=float, default=ACCURACY_TARGET_PP)
    parser.add_argument("--ratio-tolerance", type=float, default=RATIO_TOLERANCE)
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
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(parts: dict[str, Any]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing required CSV artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return sum(finite) / len(finite) if finite else math.nan


def first_nonempty(values: list[Any], default: str = "") -> str:
    for value in values:
        text = str(value)
        if text and text.lower() != "nan":
            return text
    return default


def sorted_join(values: list[Any]) -> str:
    return ",".join(sorted({str(value) for value in values if str(value)}))


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(val) for val in value]
    return value


def collect_layer_signal(stats_rows: list[dict[str, Any]], key: str) -> str:
    values: set[str] = set()
    for row in stats_rows:
        stats = row.get("perturb_stats") or {}
        if stats.get(key):
            values.add(str(stats[key]))
        for layer in (stats.get("per_layer") or {}).values():
            if isinstance(layer, dict) and layer.get(key):
                values.add(str(layer[key]))
    return ",".join(sorted(values))


def condition_perturbation_family(condition: str, selection_signal: str, budget_signal: str) -> str:
    if condition in {"e0_dense"}:
        return "dense_8bit_reference"
    if condition in {"e2_l1", "e3_slack", "e5_random"}:
        return "binary_keep_prune_adc_proxy"
    if condition in {"e4_suds", "e7_overlay", "e8_overflow", "e9_suds_conservative"}:
        return "ternary_keep_degrade_prune_adc_proxy"
    if "overflow" in selection_signal:
        return "ternary_keep_degrade_prune_adc_proxy"
    if "suds" in budget_signal:
        return "ternary_keep_degrade_prune_adc_proxy"
    return "architecture_linked_accuracy_proxy"


def source_ratio_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "keep": mean([as_float(row.get("mapped_keep_ratio")) for row in rows]),
        "degrade": mean([as_float(row.get("mapped_degrade_ratio")) for row in rows]),
        "prune": mean([as_float(row.get("mapped_prune_ratio")) for row in rows]),
    }


def ratio_match_status(
    arch_row: dict[str, str],
    source_ratios: dict[str, float],
    *,
    tolerance: float,
) -> tuple[str, float]:
    diffs = [
        abs(as_float(arch_row.get("keep_ratio"), 0.0) - source_ratios["keep"]),
        abs(as_float(arch_row.get("degrade_ratio"), 0.0) - source_ratios["degrade"]),
        abs(as_float(arch_row.get("prune_ratio"), 0.0) - source_ratios["prune"]),
    ]
    max_diff = max(diffs)
    return ("pass" if max_diff <= tolerance else "fail", max_diff)


def rows_by_key(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    return grouped


def read_source_run_json(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    path = (REPO_ROOT / path_text).resolve() if not Path(path_text).is_absolute() else Path(path_text)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_trace_index(trace_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        if row.get("sensitivity_case") != "nominal":
            continue
        if row.get("scheduler") != "suds_aware":
            continue
        grouped[(row.get("workload", ""), row.get("condition", ""))].append(row)

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        trace_ids = sorted({row.get("schedule_trace_id", "") for row in rows if row.get("schedule_trace_id")})
        slack = [as_float(row.get("deadline_slack_norm")) for row in rows]
        waits = [as_float(row.get("queue_wait_ns"), 0.0) for row in rows]
        index[key] = {
            "schedule_trace_ids": trace_ids,
            "schedule_trace_id": trace_ids[0] if len(trace_ids) == 1 else ",".join(trace_ids),
            "scheduler": "suds_aware",
            "trace_kernel_rows": len(rows),
            "trace_link_roles": sorted_join([row.get("trace_link_role", "") for row in rows]),
            "min_deadline_slack_norm": min([value for value in slack if not math.isnan(value)], default=math.nan),
            "mean_deadline_slack_norm": mean(slack),
            "total_queue_wait_ns": sum(waits),
        }
    return index


def glue_source_context(
    *,
    arch_condition: str,
    glue_payload: dict[str, Any],
    glue_linkage_rows: list[dict[str, str]],
) -> dict[str, Any]:
    linked = [
        row for row in glue_linkage_rows
        if row.get("architecture_condition") == arch_condition
        and row.get("profile_link_status") == "pass"
    ]
    source_condition = (
        first_nonempty([row.get("condition", "") for row in linked])
        or BERT_FALLBACK_SOURCE_CONDITION.get(arch_condition, "")
    )
    per_seed = [
        row for row in glue_payload.get("per_seed", [])
        if row.get("condition") == source_condition
    ]
    if linked:
        linked_keys = {(row.get("task"), str(row.get("seed"))) for row in linked}
        source_rows = [
            row for row in per_seed
            if (row.get("task"), str(row.get("seed"))) in linked_keys
        ]
    else:
        source_rows = per_seed

    ratios = source_ratio_summary(source_rows)
    budget_signal = collect_layer_signal(source_rows, "budget_signal")
    selection_signal = collect_layer_signal(source_rows, "selection_signal")
    slack_source = sorted_join([row.get("slack_source", "") for row in source_rows])
    linked_schedule_source = sorted_join([row.get("linked_schedule_source", "") for row in linked])
    command = first_nonempty([row.get("command", "") for row in source_rows])
    return {
        "accuracy_source_artifact": repo_path(GLUE_JSON),
        "accuracy_source_rows": len(source_rows),
        "source_condition": source_condition,
        "source_row_type": "per_task_per_seed",
        "source_tasks": sorted_join([row.get("task", "") for row in source_rows]),
        "source_splits": sorted_join([row.get("split", "") for row in source_rows]),
        "source_seed_set": sorted_join([row.get("seed", "") for row in source_rows]),
        "source_device_set": sorted_join([row.get("device", "") for row in source_rows]),
        "source_git_hashes": sorted_join([row.get("git_hash", "") for row in source_rows]),
        "source_command": command or glue_payload.get("metadata", {}).get("regeneration_command", ""),
        "source_processed_samples": sum(as_int(row.get("processed_samples")) for row in source_rows),
        "source_accuracy_metric": "GLUE task primary metric aggregate",
        "source_ratios": ratios,
        "budget_signal": budget_signal or "none_dense_baseline",
        "selection_signal": selection_signal or "none_dense_baseline",
        "forward_slack_source": slack_source or "none_dense_baseline",
        "linked_schedule_source": linked_schedule_source or "dptc_photonic_tile_schedule",
        "linkage_rows": len(linked),
        "dataset_split_detail": "GLUE validation splits: " + sorted_join([row.get("task", "") for row in source_rows]),
    }


def mobilevit_source_context(
    *,
    arch_condition: str,
    mobilevit_payload: dict[str, Any],
    conservative_payload: dict[str, Any],
) -> dict[str, Any]:
    source_condition = MOBILEVIT_SOURCE_CONDITION.get(arch_condition, "")
    if source_condition == "e9_suds_conservative":
        source_rows = [
            row for row in conservative_payload.get("rows", [])
            if row.get("row_type") == "per_seed"
            and row.get("condition") == source_condition
        ]
        artifact = repo_path(CONSERVATIVE_PARETO_JSON)
        source_row_type = "per_seed_conservative_pareto"
    else:
        source_rows = [
            row for row in mobilevit_payload.get("rows", [])
            if row.get("row_type") == "per_seed"
            and row.get("model") == "mobilevit_s"
            and row.get("condition") == source_condition
        ]
        artifact = repo_path(MOBILEVIT_JSON)
        source_row_type = "per_seed_mobilevit_matrix"

    ratios = source_ratio_summary(source_rows)
    source_jsons = [str(row.get("source_json", "")) for row in source_rows if row.get("source_json")]
    run_payload = read_source_run_json(source_jsons[0]) if source_jsons else {}
    condition_payload = run_payload.get(source_condition)
    if source_condition == "e9_suds_conservative":
        condition_payload = run_payload.get("e8_overflow", condition_payload)
    stats_rows = []
    if isinstance(condition_payload, dict) and isinstance(condition_payload.get("perturb_stats"), dict):
        stats_rows.append({"perturb_stats": condition_payload["perturb_stats"]})

    budget_signal = collect_layer_signal(stats_rows, "budget_signal")
    selection_signal = collect_layer_signal(stats_rows, "selection_signal")
    command = first_nonempty([row.get("command", "") for row in source_rows])
    if not command:
        command = str(run_payload.get("config", {}).get("command", ""))
    return {
        "accuracy_source_artifact": artifact,
        "accuracy_source_rows": len(source_rows),
        "source_condition": source_condition,
        "source_row_type": source_row_type,
        "source_tasks": "imagenet_top1_top5",
        "source_splits": "ImageNet validation",
        "source_seed_set": sorted_join([row.get("seed", "") for row in source_rows]),
        "source_device_set": sorted_join([row.get("device", "") for row in source_rows]),
        "source_git_hashes": sorted_join([row.get("git_hash", "") for row in source_rows]),
        "source_command": command or mobilevit_payload.get("metadata", {}).get("regeneration_command", ""),
        "source_processed_samples": sum(as_int(row.get("processed_samples")) for row in source_rows),
        "source_accuracy_metric": "ImageNet top1 aggregate",
        "source_ratios": ratios,
        "budget_signal": budget_signal or first_nonempty([row.get("budget_signal", "") for row in source_rows], "none_dense_baseline"),
        "selection_signal": selection_signal or first_nonempty([row.get("selection_signal", "") for row in source_rows], "none_dense_baseline"),
        "forward_slack_source": first_nonempty([row.get("slack_manifest", "") for row in source_rows], "model_specific_slack_manifest"),
        "linked_schedule_source": "dptc_photonic_tile_schedule",
        "linkage_rows": 0,
        "dataset_split_detail": "ImageNet validation, 50000 deterministic samples per seed",
    }


def build_row(
    *,
    arch_row: dict[str, str],
    source: dict[str, Any],
    trace: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_ratios = source["source_ratios"]
    match_status, max_ratio_delta = ratio_match_status(
        arch_row,
        source_ratios,
        tolerance=args.ratio_tolerance,
    )
    condition = arch_row["condition"]
    selection_signal = source["selection_signal"]
    budget_signal = source["budget_signal"]
    perturbation_family = condition_perturbation_family(source["source_condition"], selection_signal, budget_signal)
    accuracy_delta = as_float(arch_row.get("delta_accuracy"))
    device_set = source["source_device_set"] or arch_row.get("device", "")
    source_rows = as_int(source["accuracy_source_rows"])
    trace_id = trace.get("schedule_trace_id", "")
    row_blockers = []
    if source_rows <= 0:
        row_blockers.append("missing_accuracy_source_rows")
    if "mps" not in {item.strip() for item in device_set.split(",") if item.strip()}:
        row_blockers.append("missing_mps_device_metadata")
    if match_status != "pass":
        row_blockers.append(f"policy_ratio_mismatch_max_delta={max_ratio_delta:.6f}")
    if not trace_id:
        row_blockers.append("missing_suds_aware_schedule_trace")
    if condition == PROMOTED_CONDITION and accuracy_delta < -args.accuracy_target_pp:
        row_blockers.append("promoted_accuracy_loss_exceeds_target")

    optical_noise_std = 0.0
    return {
        "tag": args.tag,
        "roadmap_item": "R3_end_to_end_perturbation_accuracy",
        "workload": arch_row.get("workload", ""),
        "workload_family": arch_row.get("workload_family", ""),
        "model": arch_row.get("model", ""),
        "dataset_or_split": arch_row.get("dataset_or_split", ""),
        "condition": condition,
        "condition_label": arch_row.get("condition_label", ""),
        "row_role": "promoted_suds_pareto" if condition == PROMOTED_CONDITION else "measured_baseline_or_ablation",
        "source_condition": source["source_condition"],
        "source_row_type": source["source_row_type"],
        "accuracy_metric": arch_row.get("accuracy_metric", ""),
        "accuracy": arch_row.get("accuracy", ""),
        "delta_accuracy_pp": arch_row.get("delta_accuracy", ""),
        "accuracy_evidence_label": arch_row.get("accuracy_evidence_label", ""),
        "accuracy_source_artifact": source["accuracy_source_artifact"],
        "accuracy_source_rows": source_rows,
        "source_tasks": source["source_tasks"],
        "source_splits": source["source_splits"],
        "source_seed_set": source["source_seed_set"],
        "source_device_set": device_set,
        "source_git_hashes": source["source_git_hashes"],
        "source_command": source["source_command"],
        "source_processed_samples": source["source_processed_samples"],
        "dataset_split_detail": source["dataset_split_detail"],
        "schedule_trace_id": trace_id,
        "scheduler": trace.get("scheduler", ""),
        "trace_kernel_rows": trace.get("trace_kernel_rows", 0),
        "trace_link_roles": trace.get("trace_link_roles", ""),
        "min_deadline_slack_norm": trace.get("min_deadline_slack_norm", ""),
        "mean_deadline_slack_norm": trace.get("mean_deadline_slack_norm", ""),
        "total_queue_wait_ns": trace.get("total_queue_wait_ns", ""),
        "linked_schedule_source": source["linked_schedule_source"],
        "glue_linkage_rows": source["linkage_rows"],
        "architecture_summary_artifact": repo_path(args.architecture_summary_csv),
        "architecture_json_artifact": repo_path(args.architecture_json),
        "architecture_evidence_label": arch_row.get("architecture_evidence_label", ""),
        "architecture_energy_pj": arch_row.get("energy_pj", ""),
        "architecture_latency_ns": arch_row.get("latency_ns", ""),
        "architecture_edp_pj_ns": arch_row.get("edp_pj_ns", ""),
        "energy_ratio_vs_lightening": arch_row.get("energy_ratio_vs_lightening", ""),
        "latency_ratio_vs_lightening": arch_row.get("latency_ratio_vs_lightening", ""),
        "edp_ratio_vs_lightening": arch_row.get("edp_ratio_vs_lightening", ""),
        "arch_keep_ratio": arch_row.get("keep_ratio", ""),
        "arch_degrade_ratio": arch_row.get("degrade_ratio", ""),
        "arch_prune_ratio": arch_row.get("prune_ratio", ""),
        "source_keep_ratio": source_ratios["keep"],
        "source_degrade_ratio": source_ratios["degrade"],
        "source_prune_ratio": source_ratios["prune"],
        "policy_match_status": match_status,
        "max_policy_ratio_delta": max_ratio_delta,
        "perturbation_family": perturbation_family,
        "budget_signal": budget_signal,
        "selection_signal": selection_signal,
        "forward_slack_source": source["forward_slack_source"],
        "forward_injection_mode": "weight_column_perturbation_before_model_forward",
        "adc_quantization_policy": "KEEP=8bit_identity;DEGRADE=4bit_equivalent_noise;PRUNE=2bit_or_removed_compute_proxy",
        "degrade_noise_std": DEFAULT_DEGRADE_NOISE_STD,
        "prune_noise_std": DEFAULT_PRUNE_NOISE_STD,
        "optical_noise_std": optical_noise_std,
        "overflow_fallback_policy": "hyatten_like_overflow_proxy" if "overflow" in selection_signal else "not_used",
        "optional_optical_noise_status": "disabled_for_promoted_accuracy;available_as_later_sensitivity",
        "accuracy_loss_target_pp": args.accuracy_target_pp,
        "claim_boundary": (
            "Accuracy is measured on MPS forward passes with tiered column perturbations. "
            "Energy, latency, memory, optical-link, and control terms remain architecture-modeled."
        ),
        "row_blockers": ";".join(row_blockers),
    }


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    architecture = load_json(args.architecture_json)
    arch_rows = [
        row for row in load_csv(args.architecture_summary_csv)
        if row.get("sensitivity_case") == "nominal"
        and row.get("accuracy_evidence_label") in {"measured_mps_glue", "measured_mps_imagenet"}
        and as_float(row.get("delta_accuracy")) is not None
    ]
    glue_payload = load_json(args.glue_json)
    glue_linkage_rows = load_csv(args.glue_linkage_csv)
    mobilevit_payload = load_json(args.mobilevit_json)
    conservative_payload = load_json(args.conservative_pareto_json)
    trace_index = build_trace_index(load_csv(args.scheduler_trace_csv))

    rows = []
    for arch_row in arch_rows:
        workload = arch_row.get("workload", "")
        condition = arch_row.get("condition", "")
        if workload == "bert_base_glue_seq128":
            source = glue_source_context(
                arch_condition=condition,
                glue_payload=glue_payload,
                glue_linkage_rows=glue_linkage_rows,
            )
        elif workload == "mobilevit_s_transformer_blocks_256":
            source = mobilevit_source_context(
                arch_condition=condition,
                mobilevit_payload=mobilevit_payload,
                conservative_payload=conservative_payload,
            )
        else:
            continue
        trace = trace_index.get((workload, condition), {})
        rows.append(build_row(arch_row=arch_row, source=source, trace=trace, args=args))

    promoted = [row for row in rows if row["condition"] == PROMOTED_CONDITION]
    promoted_workloads = sorted({row["workload"] for row in promoted})
    promoted_deltas = [as_float(row["delta_accuracy_pp"]) for row in promoted]
    worst_promoted_delta = min(promoted_deltas, default=math.nan)
    blockers = []
    if promoted_workloads != ["bert_base_glue_seq128", "mobilevit_s_transformer_blocks_256"]:
        blockers.append("promoted_rows_missing_workload")
    if any(row["row_blockers"] for row in promoted):
        blockers.append("promoted_row_has_blocker")
    if worst_promoted_delta < -args.accuracy_target_pp:
        blockers.append("promoted_accuracy_loss_exceeds_target")
    if not all(row["policy_match_status"] == "pass" for row in promoted):
        blockers.append("promoted_policy_ratio_mismatch")
    if not all(row["schedule_trace_id"] for row in promoted):
        blockers.append("promoted_schedule_trace_missing")
    if not all("mps" in row["source_device_set"].split(",") for row in promoted):
        blockers.append("promoted_mps_device_metadata_missing")

    summary = {
        "tag": args.tag,
        "date": DATE,
        "input_architecture_status": architecture.get("decision", {}).get("architecture_sim_status", ""),
        "n_rows": len(rows),
        "n_promoted_rows": len(promoted),
        "promoted_workloads": promoted_workloads,
        "worst_promoted_accuracy_delta_pp": worst_promoted_delta,
        "accuracy_loss_target_pp": args.accuracy_target_pp,
        "all_promoted_policy_matched": all(row["policy_match_status"] == "pass" for row in promoted),
        "all_promoted_mps": all("mps" in row["source_device_set"].split(",") for row in promoted),
        "all_promoted_trace_linked": all(bool(row["schedule_trace_id"]) for row in promoted),
        "measured_accuracy_labels": sorted({row["accuracy_evidence_label"] for row in rows}),
        "perturbation_scope": (
            "Model-forward MPS accuracy rows with KEEP/DEGRADE/PRUNE tiered "
            "column perturbations joined to the architecture PPA tier ratios."
        ),
    }
    summary["decision"] = {
        "r3_acceptance_state": "pass" if not blockers else "fail",
        "stop_condition_state": (
            "no R3 hard stop"
            if not math.isnan(worst_promoted_delta) and worst_promoted_delta >= -args.accuracy_target_pp
            else "R3 hard stop: promoted accuracy loss exceeds target"
        ),
        "blockers": blockers,
        "claim": (
            "Promoted rows have measured MPS accuracy deltas linked to the same "
            "architecture KEEP/DEGRADE/PRUNE ratios and SUDS-aware scheduler traces."
        ),
    }
    return rows, summary


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


def write_json(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_artifacts = {
        "architecture_summary_csv": repo_path(args.architecture_summary_csv),
        "architecture_json": repo_path(args.architecture_json),
        "scheduler_trace_csv": repo_path(args.scheduler_trace_csv),
        "glue_json": repo_path(args.glue_json),
        "glue_linkage_csv": repo_path(args.glue_linkage_csv),
        "mobilevit_json": repo_path(args.mobilevit_json),
        "conservative_pareto_json": repo_path(args.conservative_pareto_json),
        "conservative_pareto_csv": repo_path(args.conservative_pareto_csv),
    }
    payload = {
        "metadata": {
            "tag": args.tag,
            "artifact_id": f"suds_tetc_end_to_end_accuracy_{args.tag}",
            "roadmap_item": "R3_end_to_end_perturbation_accuracy",
            "evidence_label": "measured_mps_end_to_end_perturbation_accuracy",
            "promotion_decision": summary["decision"]["r3_acceptance_state"],
            "git_hash": git_hash(),
            "regeneration_command": "make suds-tetc-end-to-end-accuracy",
            "deterministic_config_id": config_hash(
                {
                    "tag": args.tag,
                    "accuracy_target_pp": args.accuracy_target_pp,
                    "ratio_tolerance": args.ratio_tolerance,
                    "source_artifacts": source_artifacts,
                }
            ),
            "source_artifacts": source_artifacts,
            "source_artifact_sha256": {
                key: sha256_path(REPO_ROOT / value)
                for key, value in source_artifacts.items()
                if (REPO_ROOT / value).is_file()
            },
        },
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    val = as_float(value)
    if math.isnan(val):
        return "n/a"
    return f"{val:.{digits}f}"


def write_report(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    promoted = [row for row in rows if row["condition"] == PROMOTED_CONDITION]
    lines = [
        "# SUDS TETC End-To-End Perturbation Accuracy",
        "",
        f"Tag: `{args.tag}`",
        "Roadmap item: `R3_end_to_end_perturbation_accuracy`",
        "Evidence label: `measured_mps_end_to_end_perturbation_accuracy`",
        f"Acceptance state: `{summary['decision']['r3_acceptance_state']}`",
        f"Stop-condition state: `{summary['decision']['stop_condition_state']}`",
        "",
        "## Scope",
        "",
        "This artifact joins governed MPS model-forward accuracy evidence to the",
        "TETC architecture PPA rows and R2 SUDS-aware scheduler traces. It checks",
        "that each promoted `suds_pareto` accuracy delta uses the same",
        "KEEP/DEGRADE/PRUNE tier ratios as the architecture row being promoted.",
        "",
        "The perturbation surface is a tiered column perturbation proxy: KEEP uses",
        "the unmodified 8-bit-equivalent path, DEGRADE uses a 4-bit-equivalent",
        "small-noise path, and PRUNE uses a stronger 2-bit/removal proxy. Optical",
        "noise is disabled for the promoted accuracy rows and remains a later",
        "sensitivity dimension; this report does not claim bit-exact ADC, optical",
        "device, or silicon behavior.",
        "",
        "## Promoted Rows",
        "",
        "| Workload | Accuracy delta | Device | Source condition | Trace ID | Policy match |",
        "|---|---:|---|---|---|---|",
    ]
    for row in promoted:
        lines.append(
            f"| `{row['workload']}` | {fmt(row['delta_accuracy_pp'])} pp | "
            f"`{row['source_device_set']}` | `{row['source_condition']}` | "
            f"`{row['schedule_trace_id']}` | `{row['policy_match_status']}` |"
        )

    lines.extend(
        [
            "",
            "## Measured Baseline And Ablation Rows",
            "",
            "| Workload | Condition | Delta | EDP ratio | Source rows | Policy match |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        if row["condition"] == PROMOTED_CONDITION:
            continue
        lines.append(
            f"| `{row['workload']}` | `{row['condition']}` | {fmt(row['delta_accuracy_pp'])} pp | "
            f"{fmt(row['edp_ratio_vs_lightening'])} | {row['accuracy_source_rows']} | "
            f"`{row['policy_match_status']}` |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Worst promoted accuracy delta: `{fmt(summary['worst_promoted_accuracy_delta_pp'])} pp`",
            f"- Target: `-{args.accuracy_target_pp:.3f} pp`",
            f"- Promoted policies matched: `{summary['all_promoted_policy_matched']}`",
            f"- Promoted rows on MPS: `{summary['all_promoted_mps']}`",
            f"- Promoted rows linked to R2 traces: `{summary['all_promoted_trace_linked']}`",
            f"- Blockers: `{','.join(summary['decision']['blockers']) or 'none'}`",
            "",
            "## Artifacts",
            "",
            f"- CSV: `{repo_path(args.csv_out)}`",
            f"- JSON: `{repo_path(args.json_out)}`",
            f"- Report: `{repo_path(args.report_out)}`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "make suds-tetc-end-to-end-accuracy",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, summary = build_rows(args)
    write_csv(args.csv_out, rows)
    write_json(args.json_out, args, rows, summary)
    write_report(args.report_out, args, rows, summary)
    print(f"wrote {repo_path(args.csv_out)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")
    print(f"r3_acceptance_state={summary['decision']['r3_acceptance_state']}")
    if summary["decision"]["r3_acceptance_state"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
