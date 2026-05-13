#!/usr/bin/env python3
"""Build the R4 same-simulator strong-baseline fairness artifact.

This generator does not rerun model evaluation. It audits the existing TETC
architecture simulator, R2 scheduler traces, and R3 MPS accuracy linkage under
one selected simulator configuration. The output is the fairness matrix used to
separate matched same-scope baselines from boundary rows whose assumptions
intentionally differ from the promoted SUDS DPTC fabric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
DATE = "2026-05-13"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

ARCH_SUMMARY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_summary.csv"
ARCH_PARAMETERS_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_parameters.csv"
ARCH_JSON = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}.json"
SCHEDULER_TRACE_CSV = REPORT_DATA / f"suds_tetc_scheduler_traces_{TAG}.csv"
END_TO_END_ACCURACY_CSV = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{TAG}.csv"
END_TO_END_ACCURACY_JSON = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{TAG}.json"

CSV_OUT = REPORT_DATA / f"suds_tetc_baseline_fairness_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_same_sim_baselines_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_same_simulator_baselines.md"

PROMOTED_CONDITION = "suds_pareto"
WORKLOADS_REQUIRED = ("bert_base_glue_seq128", "mobilevit_s_transformer_blocks_256")
EXPECTED_CONDITIONS = (
    "lightening_dptc",
    "uniform_8bit",
    "uniform_4bit",
    "random",
    "l1",
    "slack_only",
    "signal_only",
    "suds_only",
    "suds_l1",
    "suds_signal",
    "suds_pareto",
    "hyatten_style",
    "tempo_time_multiplexed",
    "astra_boundary",
)
SAME_SCOPE_BASELINES = {
    "lightening_dptc",
    "uniform_8bit",
    "uniform_4bit",
    "random",
    "l1",
    "slack_only",
    "signal_only",
}
SUDS_ABLATIONS = {"suds_only", "suds_l1", "suds_signal"}
BOUNDARY_BASELINES = {"hyatten_style", "tempo_time_multiplexed", "astra_boundary"}
MEASURED_ACCURACY_LABELS = {"measured_mps_glue", "measured_mps_imagenet"}
DEFAULT_EDP_TOLERANCE = 0.005
DEFAULT_ACCURACY_TOLERANCE_PP = 0.05

BOUNDARY_ASSUMPTION_DIFFERENCES = {
    "hyatten_style": (
        "uses HyAtten-style low-resolution signal-selection boundary; "
        "accuracy is literature/unmeasured in this local artifact"
    ),
    "tempo_time_multiplexed": (
        "uses TeMPO-style time-multiplexed readout boundary instead of the "
        "selected Lightening-style DPTC temporal-accumulation readout"
    ),
    "astra_boundary": (
        "uses ASTRA-style stochastic optical fabric and digital-fallback "
        "boundary instead of the selected deterministic DPTC fabric"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--architecture-summary-csv", type=Path, default=ARCH_SUMMARY_CSV)
    parser.add_argument("--architecture-parameters-csv", type=Path, default=ARCH_PARAMETERS_CSV)
    parser.add_argument("--architecture-json", type=Path, default=ARCH_JSON)
    parser.add_argument("--scheduler-trace-csv", type=Path, default=SCHEDULER_TRACE_CSV)
    parser.add_argument("--end-to-end-accuracy-csv", type=Path, default=END_TO_END_ACCURACY_CSV)
    parser.add_argument("--end-to-end-accuracy-json", type=Path, default=END_TO_END_ACCURACY_JSON)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--edp-tolerance", type=float, default=DEFAULT_EDP_TOLERANCE)
    parser.add_argument("--accuracy-tolerance-pp", type=float, default=DEFAULT_ACCURACY_TOLERANCE_PP)
    return parser.parse_args()


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing required CSV artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def config_hash(parts: dict[str, Any]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def parameter_lookup(rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        str(row.get("parameter", "")): str(row.get("value", ""))
        for row in rows
        if row.get("parameter")
    }


def nominal_arch_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("sensitivity_case") == "nominal"
        and row.get("condition") in EXPECTED_CONDITIONS
        and row.get("workload") in WORKLOADS_REQUIRED
    ]


def accuracy_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("workload", ""), row.get("condition", ""))
        if key[0] and key[1]:
            index[key] = row
    return index


def trace_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("sensitivity_case") != "nominal":
            continue
        if row.get("scheduler") != "suds_aware":
            continue
        key = (row.get("workload", ""), row.get("condition", ""))
        if key[0] and key[1]:
            grouped[key].append(row)

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group_rows in grouped.items():
        trace_ids = sorted({row.get("schedule_trace_id", "") for row in group_rows if row.get("schedule_trace_id")})
        queue_wait = sum(as_float(row.get("queue_wait_ns"), 0.0) for row in group_rows)
        slack = [
            as_float(row.get("deadline_slack_norm"))
            for row in group_rows
            if not math.isnan(as_float(row.get("deadline_slack_norm")))
        ]
        roles = sorted({row.get("trace_link_role", "") for row in group_rows if row.get("trace_link_role")})
        index[key] = {
            "schedule_trace_id": trace_ids[0] if len(trace_ids) == 1 else ",".join(trace_ids),
            "scheduler": "suds_aware",
            "trace_kernel_rows": len(group_rows),
            "trace_link_roles": ",".join(roles),
            "min_deadline_slack_norm": min(slack, default=math.nan),
            "mean_deadline_slack_norm": sum(slack) / len(slack) if slack else math.nan,
            "total_queue_wait_ns": queue_wait,
        }
    return index


def workload_reference(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("condition") == "lightening_dptc":
            refs[row.get("workload", "")] = row
    return refs


def comparison_scope(condition: str) -> str:
    if condition == PROMOTED_CONDITION:
        return "promoted_suds_pareto"
    if condition in SAME_SCOPE_BASELINES:
        return "same_scope_baseline"
    if condition in SUDS_ABLATIONS:
        return "same_simulator_suds_ablation"
    if condition in BOUNDARY_BASELINES:
        return "boundary_baseline"
    return "unknown"


def baseline_role(condition: str) -> str:
    if condition == PROMOTED_CONDITION:
        return "promoted_main_row"
    if condition in SUDS_ABLATIONS:
        return "suds_ablation"
    if condition in BOUNDARY_BASELINES:
        return "strong_boundary_context"
    return "strong_same_scope_baseline"


def accuracy_label_status(label: str) -> str:
    if label in MEASURED_ACCURACY_LABELS:
        return "measured_mps"
    if label:
        return "normalized_boundary_label"
    return "missing_accuracy_label"


def measured_for_dominance(row: dict[str, Any]) -> bool:
    return (
        row.get("accuracy_evidence_label") in MEASURED_ACCURACY_LABELS
        and not math.isnan(as_float(row.get("delta_accuracy_pp")))
        and not math.isnan(as_float(row.get("edp_ratio_vs_lightening")))
    )


def same_workload_shape(row: dict[str, str], ref: dict[str, str]) -> bool:
    fields = ("model", "dataset_or_split", "n_kernels", "macs", "output_values", "memory_moved_bytes")
    return all(str(row.get(field, "")) == str(ref.get(field, "")) for field in fields)


def same_selected_config(row: dict[str, str], ref: dict[str, str]) -> bool:
    fields = ("tile_dim", "tiles", "cores_per_tile", "parallel_cores", "sideband_group_cols", "adc_sharing_mode")
    return all(str(row.get(field, "")) == str(ref.get(field, "")) for field in fields)


def build_matrix_rows(
    *,
    arch_rows: list[dict[str, str]],
    params: dict[str, str],
    accuracy_rows: dict[tuple[str, str], dict[str, str]],
    traces: dict[tuple[str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    refs = workload_reference(arch_rows)
    rows: list[dict[str, Any]] = []
    for arch in sorted(arch_rows, key=lambda item: (item.get("workload", ""), item.get("condition", ""))):
        workload = arch.get("workload", "")
        condition = arch.get("condition", "")
        ref = refs.get(workload, {})
        acc = accuracy_rows.get((workload, condition), {})
        trace = traces.get((workload, condition), {})
        scope = comparison_scope(condition)
        selected_config_match = bool(ref) and same_selected_config(arch, ref)
        workload_shape_match = bool(ref) and same_workload_shape(arch, ref)
        accuracy_label = acc.get("accuracy_evidence_label") or arch.get("accuracy_evidence_label", "")
        delta_accuracy = acc.get("delta_accuracy_pp") or arch.get("delta_accuracy", "")
        accuracy_value = acc.get("accuracy") or arch.get("accuracy", "")
        source_device = acc.get("source_device_set") or arch.get("device", "")
        policy_match = acc.get("policy_match_status", "")
        if not policy_match:
            policy_match = "not_applicable_boundary_accuracy" if accuracy_label not in MEASURED_ACCURACY_LABELS else "architecture_summary_only"
        boundary_diff = BOUNDARY_ASSUMPTION_DIFFERENCES.get(condition, "")
        accuracy_boundary = ""
        if accuracy_label not in MEASURED_ACCURACY_LABELS:
            accuracy_boundary = f"accuracy label is {accuracy_label or 'missing'}"
        assumption_differences = "; ".join(item for item in [boundary_diff, accuracy_boundary] if item)
        row_blockers = []
        if not selected_config_match:
            row_blockers.append("selected_simulator_config_mismatch")
        if not workload_shape_match:
            row_blockers.append("workload_shape_mismatch")
        if not accuracy_label:
            row_blockers.append("missing_accuracy_evidence_label")
        if scope == "boundary_baseline" and not boundary_diff:
            row_blockers.append("boundary_assumption_difference_missing")
        if scope == "same_scope_baseline" and boundary_diff:
            row_blockers.append("same_scope_row_has_boundary_difference")
        if condition == PROMOTED_CONDITION and policy_match != "pass":
            row_blockers.append("promoted_policy_match_not_pass")
        if condition == PROMOTED_CONDITION and "mps" not in {part.strip() for part in source_device.split(",") if part.strip()}:
            row_blockers.append("promoted_mps_metadata_missing")
        if not trace.get("schedule_trace_id"):
            row_blockers.append("missing_suds_aware_schedule_trace")

        memory_settings_match = workload_shape_match and as_float(arch.get("memory_energy_pj"), 0.0) > 0.0
        simulator_config = {
            "tag": args.tag,
            "tile_dim": arch.get("tile_dim", ""),
            "tiles": arch.get("tiles", ""),
            "cores_per_tile": arch.get("cores_per_tile", ""),
            "parallel_cores": arch.get("parallel_cores", ""),
            "sideband_group_cols": arch.get("sideband_group_cols", ""),
            "adc_sharing_mode": arch.get("adc_sharing_mode", ""),
            "frequency_ghz": params.get("frequency_ghz", ""),
            "sram_global_kib": params.get("sram_global_kib", ""),
            "memory_energy_positive": memory_settings_match,
        }
        rows.append(
            {
                "tag": args.tag,
                "roadmap_item": "R4_same_simulator_strong_baselines",
                "workload": workload,
                "workload_family": arch.get("workload_family", ""),
                "model": arch.get("model", ""),
                "dataset_or_split": arch.get("dataset_or_split", ""),
                "condition": condition,
                "condition_label": arch.get("condition_label", ""),
                "baseline_role": baseline_role(condition),
                "comparison_scope": scope,
                "fairness_status": "pass" if not row_blockers else "fail",
                "simulator_config_id": f"r4_{config_hash(simulator_config)}",
                "same_simulator_config_match": selected_config_match,
                "matched_tile_count": str(arch.get("tiles", "")) == str(ref.get("tiles", "")),
                "matched_frequency": bool(params.get("frequency_ghz")),
                "matched_adc_dac_assumption": "same_calibrated_tier_table",
                "matched_memory_settings": memory_settings_match,
                "matched_workload_shape": workload_shape_match,
                "area_model_status": "matched_model_with_condition_terms" if selected_config_match else "mismatch",
                "tile_dim": arch.get("tile_dim", ""),
                "tiles": arch.get("tiles", ""),
                "cores_per_tile": arch.get("cores_per_tile", ""),
                "parallel_cores": arch.get("parallel_cores", ""),
                "sideband_group_cols": arch.get("sideband_group_cols", ""),
                "adc_sharing_mode": arch.get("adc_sharing_mode", ""),
                "frequency_ghz": params.get("frequency_ghz", ""),
                "n_kernels": arch.get("n_kernels", ""),
                "macs": arch.get("macs", ""),
                "memory_moved_bytes": arch.get("memory_moved_bytes", ""),
                "accuracy": accuracy_value,
                "delta_accuracy_pp": delta_accuracy,
                "accuracy_evidence_label": accuracy_label,
                "accuracy_label_status": accuracy_label_status(accuracy_label),
                "accuracy_source_artifact": acc.get("accuracy_source_artifact", ""),
                "source_device_set": source_device,
                "policy_match_status": policy_match,
                "schedule_trace_id": acc.get("schedule_trace_id") or trace.get("schedule_trace_id", ""),
                "scheduler": acc.get("scheduler") or trace.get("scheduler", ""),
                "trace_kernel_rows": acc.get("trace_kernel_rows") or trace.get("trace_kernel_rows", ""),
                "trace_link_roles": acc.get("trace_link_roles") or trace.get("trace_link_roles", ""),
                "min_deadline_slack_norm": acc.get("min_deadline_slack_norm") or trace.get("min_deadline_slack_norm", ""),
                "mean_deadline_slack_norm": acc.get("mean_deadline_slack_norm") or trace.get("mean_deadline_slack_norm", ""),
                "total_queue_wait_ns": acc.get("total_queue_wait_ns") or trace.get("total_queue_wait_ns", ""),
                "architecture_summary_artifact": repo_path(args.architecture_summary_csv),
                "architecture_json_artifact": repo_path(args.architecture_json),
                "scheduler_trace_artifact": repo_path(args.scheduler_trace_csv),
                "end_to_end_accuracy_artifact": repo_path(args.end_to_end_accuracy_json),
                "architecture_evidence_label": arch.get("architecture_evidence_label", ""),
                "energy_pj": arch.get("energy_pj", ""),
                "latency_ns": arch.get("latency_ns", ""),
                "edp_pj_ns": arch.get("edp_pj_ns", ""),
                "area_mm2": arch.get("area_mm2", ""),
                "energy_ratio_vs_lightening": arch.get("energy_ratio_vs_lightening", ""),
                "latency_ratio_vs_lightening": arch.get("latency_ratio_vs_lightening", ""),
                "edp_ratio_vs_lightening": arch.get("edp_ratio_vs_lightening", ""),
                "keep_ratio": arch.get("keep_ratio", ""),
                "degrade_ratio": arch.get("degrade_ratio", ""),
                "prune_ratio": arch.get("prune_ratio", ""),
                "assumption_differences": assumption_differences,
                "claim_boundary": (
                    "Same-simulator architecture PPA; accuracy is measured only where the "
                    "accuracy_evidence_label is measured_mps_*; boundary rows are not "
                    "promoted as same-scope SUDS dominators."
                ),
                "dominance_status": "pending",
                "dominance_reason": "",
                "row_blockers": ";".join(row_blockers),
            }
        )
    return rows


def annotate_dominance(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[str(row["workload"])].append(row)

    for workload_rows in by_workload.values():
        promoted = next((row for row in workload_rows if row["condition"] == PROMOTED_CONDITION), None)
        if not promoted:
            for row in workload_rows:
                row["dominance_status"] = "missing_promoted_suds_pareto"
                row["dominance_reason"] = "workload has no promoted SUDS Pareto row"
            continue
        promoted_edp = as_float(promoted.get("edp_ratio_vs_lightening"))
        promoted_delta = as_float(promoted.get("delta_accuracy_pp"))
        for row in workload_rows:
            condition = str(row["condition"])
            scope = str(row["comparison_scope"])
            if condition == PROMOTED_CONDITION:
                row["dominance_status"] = "promoted_reference"
                row["dominance_reason"] = (
                    f"promoted_edp={fmt(promoted_edp)}; promoted_delta={fmt(promoted_delta)} pp"
                )
                continue
            if scope != "same_scope_baseline":
                row["dominance_status"] = "not_same_scope_dominance_candidate"
                row["dominance_reason"] = "boundary row or SUDS ablation; retained for context"
                continue
            if not measured_for_dominance(row):
                row["dominance_status"] = "not_measured_for_equal_accuracy_dominance"
                row["dominance_reason"] = f"accuracy_evidence_label={row.get('accuracy_evidence_label', '')}"
                continue
            baseline_edp = as_float(row.get("edp_ratio_vs_lightening"))
            baseline_delta = as_float(row.get("delta_accuracy_pp"))
            edp_better = baseline_edp < promoted_edp - args.edp_tolerance
            accuracy_no_worse = baseline_delta >= promoted_delta - args.accuracy_tolerance_pp
            if edp_better and accuracy_no_worse:
                row["dominance_status"] = "dominates_promoted_suds_under_equal_accuracy"
                row["dominance_reason"] = (
                    f"baseline_edp={baseline_edp:.6f} < promoted_edp={promoted_edp:.6f} "
                    f"by more than tolerance={args.edp_tolerance:.6f}; "
                    f"baseline_delta={baseline_delta:.6f} pp is within "
                    f"{args.accuracy_tolerance_pp:.6f} pp of promoted_delta={promoted_delta:.6f} pp"
                )
            else:
                row["dominance_status"] = "not_dominating_promoted_suds"
                row["dominance_reason"] = (
                    f"edp_better={edp_better}; accuracy_no_worse={accuracy_no_worse}; "
                    f"baseline_edp={fmt(baseline_edp)}; baseline_delta={fmt(baseline_delta)} pp"
                )
    return rows


def build_summary(
    rows: list[dict[str, Any]],
    *,
    architecture: dict[str, Any],
    end_to_end_accuracy: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    present = {
        workload: sorted({row["condition"] for row in rows if row["workload"] == workload})
        for workload in WORKLOADS_REQUIRED
    }
    missing_conditions = {
        workload: sorted(set(EXPECTED_CONDITIONS) - set(conditions))
        for workload, conditions in present.items()
    }
    same_scope_rows = [row for row in rows if row["comparison_scope"] == "same_scope_baseline"]
    boundary_rows = [row for row in rows if row["comparison_scope"] == "boundary_baseline"]
    promoted_rows = [row for row in rows if row["condition"] == PROMOTED_CONDITION]
    dominators = [
        {
            "workload": row["workload"],
            "condition": row["condition"],
            "edp_ratio_vs_lightening": as_float(row["edp_ratio_vs_lightening"]),
            "delta_accuracy_pp": as_float(row["delta_accuracy_pp"]),
            "reason": row["dominance_reason"],
        }
        for row in rows
        if row["dominance_status"] == "dominates_promoted_suds_under_equal_accuracy"
    ]
    boundary_lower_edp = []
    for row in rows:
        if row["comparison_scope"] != "boundary_baseline":
            continue
        promoted = next(
            item for item in promoted_rows
            if item["workload"] == row["workload"]
        )
        if as_float(row["edp_ratio_vs_lightening"]) < as_float(promoted["edp_ratio_vs_lightening"]):
            boundary_lower_edp.append(
                {
                    "workload": row["workload"],
                    "condition": row["condition"],
                    "edp_ratio_vs_lightening": as_float(row["edp_ratio_vs_lightening"]),
                    "boundary_reason": row["assumption_differences"],
                }
            )

    same_scope_assumptions_matched = all(
        row["same_simulator_config_match"]
        and row["matched_workload_shape"]
        and row["matched_tile_count"]
        and row["matched_frequency"]
        and row["matched_memory_settings"]
        for row in same_scope_rows
    )
    boundary_assumptions_documented = all(bool(row["assumption_differences"]) for row in boundary_rows)
    accuracy_labels_normalized = all(bool(row["accuracy_evidence_label"]) for row in rows)
    promoted_rows_ok = (
        len(promoted_rows) == len(WORKLOADS_REQUIRED)
        and all(row["policy_match_status"] == "pass" for row in promoted_rows)
        and all(row["schedule_trace_id"] for row in promoted_rows)
        and all("mps" in {part.strip() for part in str(row["source_device_set"]).split(",") if part.strip()} for row in promoted_rows)
    )
    blockers = []
    if any(missing_conditions.values()):
        blockers.append("baseline_condition_matrix_incomplete")
    if not same_scope_assumptions_matched:
        blockers.append("same_scope_assumptions_not_matched")
    if not boundary_assumptions_documented:
        blockers.append("boundary_assumption_differences_not_documented")
    if not accuracy_labels_normalized:
        blockers.append("accuracy_evidence_labels_missing")
    if not promoted_rows_ok:
        blockers.append("promoted_rows_not_fully_linked")
    if dominators:
        blockers.append("same_scope_baseline_dominates_suds_under_equal_accuracy")

    acceptance_state = "pass" if not blockers else "fail"
    stop_condition_state = (
        "R4 hard stop: same-scope baseline dominates SUDS under equal accuracy"
        if dominators
        else "no R4 hard stop"
    )
    return {
        "tag": args.tag,
        "date": DATE,
        "n_rows": len(rows),
        "n_same_scope_baselines": len(same_scope_rows),
        "n_boundary_baselines": len(boundary_rows),
        "n_promoted_rows": len(promoted_rows),
        "expected_conditions": list(EXPECTED_CONDITIONS),
        "present_conditions_by_workload": present,
        "missing_conditions_by_workload": missing_conditions,
        "same_scope_assumptions_matched": same_scope_assumptions_matched,
        "boundary_assumptions_documented": boundary_assumptions_documented,
        "accuracy_labels_normalized": accuracy_labels_normalized,
        "promoted_rows_fully_linked": promoted_rows_ok,
        "dominators": dominators,
        "boundary_lower_edp_rows": boundary_lower_edp,
        "input_architecture_status": architecture.get("decision", {}).get("architecture_sim_status", ""),
        "input_r3_acceptance_state": end_to_end_accuracy.get("summary", {}).get("decision", {}).get("r3_acceptance_state", ""),
        "decision": {
            "r4_acceptance_state": acceptance_state,
            "stop_condition_state": stop_condition_state,
            "claim_update_required": bool(dominators),
            "blockers": blockers,
            "edp_tolerance": args.edp_tolerance,
            "accuracy_tolerance_pp": args.accuracy_tolerance_pp,
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, *, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": args.tag,
            "roadmap_item": "R4_same_simulator_strong_baselines",
            "artifact_id": f"suds_tetc_same_sim_baselines_{args.tag}",
            "evidence_label": "same_simulator_baseline_fairness",
            "promotion_decision": (
                "same_simulator_baseline_fairness_pass"
                if summary["decision"]["r4_acceptance_state"] == "pass"
                else "same_simulator_baseline_fairness_fail"
            ),
            "regeneration_command": "make suds-tetc-same-sim-baselines",
        },
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    lines = [
        "# SUDS TETC Same-Simulator Strong Baselines",
        "",
        f"Tag: `{args.tag}`",
        "Roadmap item: `R4_same_simulator_strong_baselines`",
        "Evidence label: `same_simulator_baseline_fairness`",
        f"Acceptance state: `{decision['r4_acceptance_state']}`",
        f"Stop-condition state: `{decision['stop_condition_state']}`",
        "",
        "## Scope",
        "",
        "This artifact rebuilds the baseline comparison surface from the same",
        "TETC architecture simulator configuration and joins it to R2 scheduler",
        "traces plus R3 MPS accuracy linkage where measured accuracy exists.",
        "It is a fairness and dominance audit, not a new model-evaluation run.",
        "",
        "The matrix keeps strong local selectors, uniform rows, SUDS ablations,",
        "and alternate-fabric rows visible. Same-scope rows must share the",
        "selected DPTC configuration. Boundary rows must state which assumption",
        "differs before they can be used as reviewer-facing context.",
        "",
        "## Decision",
        "",
        f"- R4 acceptance: `{decision['r4_acceptance_state']}`",
        f"- Blockers: `{';'.join(decision['blockers']) or 'none'}`",
        f"- Same-scope assumptions matched: `{summary['same_scope_assumptions_matched']}`",
        f"- Boundary assumptions documented: `{summary['boundary_assumptions_documented']}`",
        f"- Accuracy labels normalized: `{summary['accuracy_labels_normalized']}`",
        f"- Promoted rows fully linked: `{summary['promoted_rows_fully_linked']}`",
        f"- Same-scope dominators: `{len(summary['dominators'])}`",
        "",
        "## Fairness Matrix",
        "",
        "| Workload | Condition | Scope | Accuracy evidence | EDP ratio | Delta | Fairness | Dominance |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: (item["workload"], item["comparison_scope"], item["condition"])):
        lines.append(
            f"| `{row['workload']}` | `{row['condition']}` | `{row['comparison_scope']}` | "
            f"`{row['accuracy_evidence_label']}` | {fmt(row['edp_ratio_vs_lightening'], 3)} | "
            f"{fmt(row['delta_accuracy_pp'], 3)} | `{row['fairness_status']}` | "
            f"`{row['dominance_status']}` |"
        )

    lines.extend(
        [
            "",
            "## Boundary Assumptions",
            "",
            "| Workload | Boundary row | Stated difference |",
            "|---|---|---|",
        ]
    )
    for row in sorted([item for item in rows if item["comparison_scope"] == "boundary_baseline"], key=lambda item: (item["workload"], item["condition"])):
        lines.append(f"| `{row['workload']}` | `{row['condition']}` | {row['assumption_differences']} |")

    lines.extend(
        [
            "",
            "## Same-Scope Dominance Check",
            "",
            "| Workload | Condition | Result | Reason |",
            "|---|---|---|---|",
        ]
    )
    same_scope_rows = [row for row in rows if row["comparison_scope"] == "same_scope_baseline"]
    for row in sorted(same_scope_rows, key=lambda item: (item["workload"], item["condition"])):
        lines.append(
            f"| `{row['workload']}` | `{row['condition']}` | `{row['dominance_status']}` | "
            f"{row['dominance_reason']} |"
        )

    if summary["boundary_lower_edp_rows"]:
        lines.extend(
            [
                "",
                "## Boundary Lower-EDP Rows",
                "",
                "These rows are reported because they have lower modeled EDP than the",
                "promoted SUDS row, but they are not same-scope dominators.",
                "",
                "| Workload | Boundary row | EDP ratio | Boundary reason |",
                "|---|---|---:|---|",
            ]
        )
        for item in summary["boundary_lower_edp_rows"]:
            lines.append(
                f"| `{item['workload']}` | `{item['condition']}` | "
                f"{fmt(item['edp_ratio_vs_lightening'], 3)} | {item['boundary_reason']} |"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Fairness CSV: `{repo_path(args.csv_out)}`",
            f"- Same-simulator JSON: `{repo_path(args.json_out)}`",
            f"- Report: `{repo_path(args.report_out)}`",
            f"- Architecture summary input: `{repo_path(args.architecture_summary_csv)}`",
            f"- R2 scheduler trace input: `{repo_path(args.scheduler_trace_csv)}`",
            f"- R3 accuracy input: `{repo_path(args.end_to_end_accuracy_json)}`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "make suds-tetc-same-sim-baselines",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arch_rows = nominal_arch_rows(load_csv(args.architecture_summary_csv))
    params = parameter_lookup(load_csv(args.architecture_parameters_csv))
    architecture = load_json(args.architecture_json)
    r3 = load_json(args.end_to_end_accuracy_json)
    acc_rows = accuracy_index(load_csv(args.end_to_end_accuracy_csv))
    traces = trace_index(load_csv(args.scheduler_trace_csv))
    rows = build_matrix_rows(
        arch_rows=arch_rows,
        params=params,
        accuracy_rows=acc_rows,
        traces=traces,
        args=args,
    )
    rows = annotate_dominance(rows, args)
    summary = build_summary(rows, architecture=architecture, end_to_end_accuracy=r3, args=args)
    return rows, summary


def main() -> None:
    args = parse_args()
    rows, summary = build(args)
    write_csv(args.csv_out, rows)
    write_json(args.json_out, args=args, rows=rows, summary=summary)
    write_report(args.report_out, args=args, rows=rows, summary=summary)
    print(f"wrote {repo_path(args.csv_out)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")
    print(f"r4_acceptance_state={summary['decision']['r4_acceptance_state']}")
    print(f"stop_condition_state={summary['decision']['stop_condition_state']}")


if __name__ == "__main__":
    main()
