#!/usr/bin/env python3
"""Build the R5 multi-objective Pareto design-space artifact.

This generator extends the existing architecture design-space sweep with the
R3 measured-accuracy linkage and R4 same-simulator baseline audit. It keeps two
Pareto questions separate:

1. the raw architecture sweep over energy, latency, and area, and
2. the governed R5 sweep over energy, latency, EDP, area, memory pressure,
   control overhead, and calibration-sensitivity proxy under an accuracy
   budget.

The split is intentional. The selected operating point should not be promoted
as a raw single-objective optimum when nearby uncalibrated variants reduce one
or more PPA terms. R5 decides whether it is defensible as the bounded,
evidence-governed operating point used by the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
DATE = "2026-05-13"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

DESIGN_SPACE_CSV = REPORT_DATA / f"suds_transformer_architecture_design_space_{TAG}.csv"
DESIGN_SPACE_JSON = REPORT_DATA / f"suds_transformer_architecture_design_space_{TAG}.json"
ARCH_SENSITIVITY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_sensitivity.csv"
END_TO_END_ACCURACY_CSV = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{TAG}.csv"
END_TO_END_ACCURACY_JSON = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{TAG}.json"
SAME_SIM_BASELINES_CSV = REPORT_DATA / f"suds_tetc_baseline_fairness_{TAG}.csv"
SAME_SIM_BASELINES_JSON = REPORT_DATA / f"suds_tetc_same_sim_baselines_{TAG}.json"

CSV_OUT = REPORT_DATA / f"suds_tetc_pareto_design_space_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_pareto_design_space_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_pareto_design_space.md"

PROMOTED_CONDITION = "suds_pareto"
WORKLOADS_REQUIRED = ("bert_base_glue_seq128", "mobilevit_s_transformer_blocks_256")
DEFAULT_ACCURACY_BUDGET_PP = 1.0
DEFAULT_DOMINANCE_TOLERANCE = 1e-12

RAW_PPA_OBJECTIVES = ("energy_pj", "latency_ns", "area_mm2")
R5_OBJECTIVES = (
    "energy_pj",
    "latency_ns",
    "edp_pj_ns",
    "area_mm2",
    "memory_pressure_bytes_per_parallel_core",
    "control_overhead_energy_ratio",
    "calibration_sensitivity_energy_ratio",
)

THRESHOLD_POLICIES = {
    "lightening_dptc": "dense_keep_all",
    "hyatten_style": "low_resolution_signal_threshold_boundary",
    "tempo_time_multiplexed": "time_multiplexed_readout_boundary",
    "astra_boundary": "stochastic_optical_fabric_boundary",
    "suds_pareto": "scheduler_guarded_conservative_alpha_0.35_tau_0.1_0.7",
    "suds_l1": "scheduler_plus_l1_alpha_0.35_tau_0.1_0.7",
    "suds_signal": "scheduler_plus_signal_alpha_0.35_tau_0.1_0.7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--design-space-csv", type=Path, default=DESIGN_SPACE_CSV)
    parser.add_argument("--design-space-json", type=Path, default=DESIGN_SPACE_JSON)
    parser.add_argument("--architecture-sensitivity-csv", type=Path, default=ARCH_SENSITIVITY_CSV)
    parser.add_argument("--end-to-end-accuracy-csv", type=Path, default=END_TO_END_ACCURACY_CSV)
    parser.add_argument("--end-to-end-accuracy-json", type=Path, default=END_TO_END_ACCURACY_JSON)
    parser.add_argument("--same-sim-baselines-csv", type=Path, default=SAME_SIM_BASELINES_CSV)
    parser.add_argument("--same-sim-baselines-json", type=Path, default=SAME_SIM_BASELINES_JSON)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--accuracy-budget-pp", type=float, default=DEFAULT_ACCURACY_BUDGET_PP)
    parser.add_argument("--dominance-tolerance", type=float, default=DEFAULT_DOMINANCE_TOLERANCE)
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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or math.isnan(denominator):
        return math.nan
    return numerator / denominator


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


def accuracy_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        workload = row.get("workload", "")
        condition = row.get("condition", "")
        if workload and condition:
            index[(workload, condition)] = row
    return index


def r4_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        workload = row.get("workload", "")
        condition = row.get("condition", "")
        if workload and condition:
            index[(workload, condition)] = row
    return index


def sensitivity_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("sensitivity_case") != "pessimistic":
            continue
        workload = row.get("workload", "")
        condition = row.get("condition", "")
        if workload and condition:
            index[(workload, condition)] = row
    return index


def component_sum(row: dict[str, Any], fields: tuple[str, ...]) -> float:
    return sum(as_float(row.get(field), 0.0) for field in fields)


def enriched_row(
    row: dict[str, str],
    *,
    acc: dict[str, str],
    r4: dict[str, str],
    sens: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    energy = as_float(row.get("energy_pj"))
    parallel_cores = as_float(row.get("parallel_cores"))
    memory_moved = as_float(row.get("memory_moved_bytes"))
    control_energy = as_float(row.get("control_energy_pj"), 0.0)
    conversion_link_energy = component_sum(
        row,
        (
            "adc_energy_pj",
            "dac_mzm_energy_pj",
            "detector_tia_energy_pj",
            "laser_energy_pj",
            "optical_link_energy_pj",
        ),
    )
    memory_pressure = safe_ratio(memory_moved, parallel_cores)
    control_ratio = safe_ratio(control_energy, energy)
    calibration_proxy = safe_ratio(conversion_link_energy, energy)

    delta_accuracy = as_float(acc.get("delta_accuracy_pp", row.get("delta_accuracy")))
    accuracy_budget_pass = not math.isnan(delta_accuracy) and delta_accuracy >= -args.accuracy_budget_pp
    selected = parse_bool(row.get("selected_operating_point"))
    raw_pareto = parse_bool(row.get("pareto_front"))
    condition = row.get("condition", "")
    workload = row.get("workload", "")
    is_promoted_condition = condition == PROMOTED_CONDITION

    pessimistic_edp = as_float(sens.get("edp_ratio_vs_lightening"))
    nominal_edp = as_float(row.get("edp_ratio_vs_lightening"))
    pessimistic_edp_lift = (
        safe_ratio(pessimistic_edp, nominal_edp) if selected and not math.isnan(pessimistic_edp) else math.nan
    )

    fairness_status = r4.get("fairness_status", "")
    dominance_status = r4.get("dominance_status", "")
    policy_match_status = acc.get("policy_match_status", r4.get("policy_match_status", ""))
    source_device_set = acc.get("source_device_set", r4.get("source_device_set", row.get("device", "")))

    evidence_maturity = "simulated_design_variant_policy_accuracy_inherited"
    if not accuracy_budget_pass:
        evidence_maturity = "not_accuracy_budget_candidate"
    if selected and is_promoted_condition and fairness_status == "pass":
        evidence_maturity = "selected_point_with_r2_r3_r4_linkage"

    out: dict[str, Any] = {
        **row,
        "roadmap_item": "R5_multi_objective_pareto_design_space",
        "budget_policy": condition,
        "threshold_policy": THRESHOLD_POLICIES.get(condition, row.get("source_condition", "")),
        "accuracy": acc.get("accuracy", row.get("accuracy", "")),
        "delta_accuracy_pp": delta_accuracy,
        "accuracy_evidence_label": acc.get("accuracy_evidence_label", row.get("accuracy_evidence_label", "")),
        "accuracy_source_artifact": acc.get("accuracy_source_artifact", ""),
        "accuracy_budget_pp": args.accuracy_budget_pp,
        "accuracy_budget_pass": accuracy_budget_pass,
        "schedule_trace_id": acc.get("schedule_trace_id", r4.get("schedule_trace_id", "")),
        "scheduler": acc.get("scheduler", r4.get("scheduler", "")),
        "policy_match_status": policy_match_status,
        "source_device_set": source_device_set,
        "fairness_status": fairness_status,
        "r4_comparison_scope": r4.get("comparison_scope", ""),
        "r4_dominance_status": dominance_status,
        "r4_dominance_reason": r4.get("dominance_reason", ""),
        "memory_pressure_bytes_per_parallel_core": memory_pressure,
        "memory_energy_ratio": safe_ratio(as_float(row.get("memory_energy_pj"), 0.0), energy),
        "control_overhead_energy_ratio": control_ratio,
        "calibration_sensitivity_energy_ratio": calibration_proxy,
        "conversion_link_energy_pj": conversion_link_energy,
        "pessimistic_edp_ratio_vs_lightening_selected_point": pessimistic_edp,
        "pessimistic_edp_lift_vs_nominal_selected_point": pessimistic_edp_lift,
        "raw_energy_latency_area_pareto_front": raw_pareto,
        "r5_multiobjective_pareto_front": False,
        "r5_dominator_count": "",
        "r5_dominating_design_ids": "",
        "r5_pareto_reason": "not_accuracy_budget_candidate",
        "raw_energy_latency_area_dominator_count": "",
        "raw_energy_latency_area_dominating_design_ids": "",
        "edp_rank_within_workload_condition": "",
        "evidence_maturity": evidence_maturity,
        "promoted_selected_point_candidate": selected and is_promoted_condition,
        "claim_boundary": (
            "R5 evaluates modeled architecture design-space rows. Accuracy is "
            "policy-linked from R3 where measured, while non-selected design "
            "variants remain simulated architecture candidates rather than new "
            "measured model runs."
        ),
    }
    return out


def dominates(
    candidate: dict[str, Any],
    target: dict[str, Any],
    objectives: tuple[str, ...],
    *,
    tolerance: float,
) -> bool:
    candidate_values = [as_float(candidate.get(name)) for name in objectives]
    target_values = [as_float(target.get(name)) for name in objectives]
    if any(math.isnan(value) for value in candidate_values + target_values):
        return False
    no_worse = all(c <= t + tolerance for c, t in zip(candidate_values, target_values))
    strictly_better = any(c < t - tolerance for c, t in zip(candidate_values, target_values))
    return no_worse and strictly_better


def annotate_pareto(rows: list[dict[str, Any]], *, args: argparse.Namespace) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["workload"]), str(row["condition"]))].append(row)

    for group_rows in grouped.values():
        ranked = sorted(group_rows, key=lambda item: as_float(item.get("edp_pj_ns"), math.inf))
        for rank, row in enumerate(ranked, start=1):
            row["edp_rank_within_workload_condition"] = rank

        for row in group_rows:
            raw_dominators = [
                other
                for other in group_rows
                if other is not row
                and dominates(
                    other,
                    row,
                    RAW_PPA_OBJECTIVES,
                    tolerance=args.dominance_tolerance,
                )
            ]
            row["raw_energy_latency_area_dominator_count"] = len(raw_dominators)
            row["raw_energy_latency_area_dominating_design_ids"] = ";".join(
                str(item["design_id"]) for item in sorted(raw_dominators, key=lambda item: as_float(item["edp_pj_ns"]))[:8]
            )

        candidates = [row for row in group_rows if row["accuracy_budget_pass"]]
        for row in candidates:
            dominators = [
                other
                for other in candidates
                if other is not row
                and dominates(other, row, R5_OBJECTIVES, tolerance=args.dominance_tolerance)
            ]
            row["r5_dominator_count"] = len(dominators)
            row["r5_dominating_design_ids"] = ";".join(
                str(item["design_id"]) for item in sorted(dominators, key=lambda item: as_float(item["edp_pj_ns"]))[:8]
            )
            row["r5_multiobjective_pareto_front"] = not dominators
            if dominators:
                row["r5_pareto_reason"] = (
                    "dominated under R5 objectives: energy, latency, EDP, area, "
                    "memory pressure, control overhead, and calibration sensitivity"
                )
            else:
                row["r5_pareto_reason"] = (
                    "not dominated under the R5 accuracy-constrained multi-objective audit"
                )


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    design_json = load_json(args.design_space_json)
    r3_json = load_json(args.end_to_end_accuracy_json)
    r4_json = load_json(args.same_sim_baselines_json)
    acc_rows = accuracy_index(load_csv(args.end_to_end_accuracy_csv))
    r4_rows = r4_index(load_csv(args.same_sim_baselines_csv))
    sens_rows = sensitivity_index(load_csv(args.architecture_sensitivity_csv))

    rows: list[dict[str, Any]] = []
    for row in load_csv(args.design_space_csv):
        key = (row.get("workload", ""), row.get("condition", ""))
        rows.append(
            enriched_row(
                row,
                acc=acc_rows.get(key, {}),
                r4=r4_rows.get(key, {}),
                sens=sens_rows.get(key, {}),
                args=args,
            )
        )
    annotate_pareto(rows, args=args)
    return rows, design_json, r3_json, r4_json


def selected_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["condition"] == PROMOTED_CONDITION and parse_bool(row["selected_operating_point"])
    ]


def raw_dominator_examples(rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    by_key = {
        (row["workload"], row["condition"], row["design_id"]): row
        for row in rows
    }
    for selected in selected_rows:
        ids = [
            item for item in str(selected.get("raw_energy_latency_area_dominating_design_ids", "")).split(";")
            if item
        ]
        for design_id in ids[:3]:
            row = by_key.get((selected["workload"], selected["condition"], design_id))
            if not row:
                continue
            examples.append(
                {
                    "workload": row["workload"],
                    "condition": row["condition"],
                    "design_id": row["design_id"],
                    "edp_ratio_vs_lightening": as_float(row["edp_ratio_vs_lightening"]),
                    "energy_ratio_vs_lightening": as_float(row["energy_ratio_vs_lightening"]),
                    "latency_ratio_vs_lightening": as_float(row["latency_ratio_vs_lightening"]),
                    "area_mm2": as_float(row["area_mm2"]),
                    "sideband_group_cols": row["sideband_group_cols"],
                    "adc_sharing_mode": row["adc_sharing_mode"],
                }
            )
    return examples


def build_summary(
    rows: list[dict[str, Any]],
    *,
    design_json: dict[str, Any],
    r3_json: dict[str, Any],
    r4_json: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_rows = selected_summary_rows(rows)
    selected_workloads = sorted({row["workload"] for row in selected_rows})
    missing_workloads = sorted(set(WORKLOADS_REQUIRED) - set(selected_workloads))

    selected_accuracy_ok = all(row["accuracy_budget_pass"] for row in selected_rows)
    selected_r5_pareto_ok = all(row["r5_multiobjective_pareto_front"] for row in selected_rows)
    selected_raw_ppa_ok = all(row["raw_energy_latency_area_pareto_front"] for row in selected_rows)
    selected_r4_ok = all(
        row["fairness_status"] == "pass"
        and row["policy_match_status"] == "pass"
        and row["schedule_trace_id"]
        and "mps" in {part.strip() for part in str(row["source_device_set"]).split(",") if part.strip()}
        for row in selected_rows
    )

    r4_dominators = r4_json.get("summary", {}).get("dominators", [])
    blockers: list[str] = []
    if missing_workloads:
        blockers.append("selected_suds_pareto_workload_rows_missing")
    if not selected_accuracy_ok:
        blockers.append("selected_suds_pareto_accuracy_budget_fail")
    if not selected_r5_pareto_ok:
        blockers.append("selected_suds_pareto_not_r5_multiobjective_pareto_valid")
    if not selected_r4_ok:
        blockers.append("selected_suds_pareto_r3_r4_linkage_incomplete")
    if r4_dominators:
        blockers.append("r4_same_scope_baseline_dominator_present")

    claim_narrowing_required = not selected_raw_ppa_ok
    acceptance_state = "pass" if not blockers else "fail"
    stop_condition_state = (
        "R5 hard stop: selected SUDS Pareto point is not governed multi-objective Pareto-valid"
        if blockers
        else (
            "no R5 hard stop; raw energy-latency-area dominance is recorded and the claim is narrowed to the governed multi-objective evidence surface"
            if claim_narrowing_required
            else "no R5 hard stop"
        )
    )

    rows_by_condition: dict[str, int] = defaultdict(int)
    pareto_by_condition: dict[str, int] = defaultdict(int)
    for row in rows:
        condition = str(row["condition"])
        rows_by_condition[condition] += 1
        if row["r5_multiobjective_pareto_front"]:
            pareto_by_condition[condition] += 1

    selected_payload = [
        {
            "workload": row["workload"],
            "design_id": row["design_id"],
            "delta_accuracy_pp": as_float(row["delta_accuracy_pp"]),
            "edp_ratio_vs_lightening": as_float(row["edp_ratio_vs_lightening"]),
            "energy_ratio_vs_lightening": as_float(row["energy_ratio_vs_lightening"]),
            "latency_ratio_vs_lightening": as_float(row["latency_ratio_vs_lightening"]),
            "area_mm2": as_float(row["area_mm2"]),
            "memory_pressure_bytes_per_parallel_core": as_float(row["memory_pressure_bytes_per_parallel_core"]),
            "control_overhead_energy_ratio": as_float(row["control_overhead_energy_ratio"]),
            "calibration_sensitivity_energy_ratio": as_float(row["calibration_sensitivity_energy_ratio"]),
            "raw_energy_latency_area_pareto_front": bool(row["raw_energy_latency_area_pareto_front"]),
            "raw_energy_latency_area_dominator_count": int(row["raw_energy_latency_area_dominator_count"]),
            "r5_multiobjective_pareto_front": bool(row["r5_multiobjective_pareto_front"]),
            "r5_dominator_count": int(row["r5_dominator_count"]),
            "pessimistic_edp_lift_vs_nominal_selected_point": as_float(
                row["pessimistic_edp_lift_vs_nominal_selected_point"]
            ),
            "schedule_trace_id": row["schedule_trace_id"],
            "accuracy_evidence_label": row["accuracy_evidence_label"],
        }
        for row in selected_rows
    ]

    return {
        "tag": args.tag,
        "date": DATE,
        "rows": len(rows),
        "input_design_space_rows": design_json.get("summary", {}).get("rows"),
        "input_raw_pareto_rows": design_json.get("summary", {}).get("pareto_rows"),
        "r5_objectives": list(R5_OBJECTIVES),
        "raw_ppa_objectives": list(RAW_PPA_OBJECTIVES),
        "accuracy_budget_pp": args.accuracy_budget_pp,
        "required_workloads": list(WORKLOADS_REQUIRED),
        "selected_workloads": selected_workloads,
        "missing_selected_workloads": missing_workloads,
        "r5_pareto_rows_by_condition": dict(sorted(pareto_by_condition.items())),
        "rows_by_condition": dict(sorted(rows_by_condition.items())),
        "selected_rows": selected_payload,
        "raw_energy_latency_area_dominator_examples": raw_dominator_examples(rows, selected_rows),
        "input_r3_acceptance_state": r3_json.get("summary", {}).get("decision", {}).get("r3_acceptance_state", ""),
        "input_r4_acceptance_state": r4_json.get("summary", {}).get("decision", {}).get("r4_acceptance_state", ""),
        "decision": {
            "r5_acceptance_state": acceptance_state,
            "stop_condition_state": stop_condition_state,
            "blockers": blockers,
            "selected_accuracy_budget_ok": selected_accuracy_ok,
            "selected_r5_multiobjective_pareto_valid": selected_r5_pareto_ok,
            "selected_raw_energy_latency_area_pareto_valid": selected_raw_ppa_ok,
            "selected_r3_r4_linkage_ok": selected_r4_ok,
            "claim_narrowing_required_for_raw_unconstrained_pareto": claim_narrowing_required,
            "selected_point_freeze_state": (
                "freeze_selected_governed_multiobjective_point" if acceptance_state == "pass" else "do_not_freeze"
            ),
            "same_scope_r4_dominators": r4_dominators,
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
            "roadmap_item": "R5_multi_objective_pareto_design_space",
            "artifact_id": f"suds_tetc_pareto_design_space_{args.tag}",
            "evidence_label": "multiobjective_pareto_design_space",
            "promotion_decision": (
                "r5_pareto_design_space_pass"
                if summary["decision"]["r5_acceptance_state"] == "pass"
                else "r5_pareto_design_space_fail"
            ),
            "regeneration_command": "make suds-tetc-pareto-design-space",
        },
        "summary": summary,
        "selected_rows": summary["selected_rows"],
        "rows": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    lines = [
        "# SUDS TETC Pareto Design Space",
        "",
        f"Tag: `{args.tag}`",
        "Roadmap item: `R5_multi_objective_pareto_design_space`",
        "Evidence label: `multiobjective_pareto_design_space`",
        f"Acceptance state: `{decision['r5_acceptance_state']}`",
        f"Stop-condition state: `{decision['stop_condition_state']}`",
        "",
        "## Scope",
        "",
        "This artifact extends the existing architecture design-space sweep with",
        "R3 measured MPS accuracy linkage and R4 same-simulator baseline",
        "fairness metadata. It is not a new model-evaluation run. The purpose is",
        "to decide whether the selected `suds_pareto` operating point is a",
        "defensible governed Pareto choice under an accuracy budget.",
        "",
        "The audit deliberately separates raw energy/latency/area Pareto status",
        "from the R5 multi-objective status. Raw PPA variants that dominate the",
        "selected point stay visible, and the manuscript claim is narrowed away",
        "from global PPA optimality.",
        "",
        "## Decision",
        "",
        f"- R5 acceptance: `{decision['r5_acceptance_state']}`",
        f"- Blockers: `{';'.join(decision['blockers']) or 'none'}`",
        f"- Accuracy budget: `{args.accuracy_budget_pp:.3f}` pp",
        f"- Selected R5 multi-objective Pareto valid: `{decision['selected_r5_multiobjective_pareto_valid']}`",
        f"- Selected raw energy/latency/area Pareto valid: `{decision['selected_raw_energy_latency_area_pareto_valid']}`",
        f"- Claim narrowing required for raw unconstrained Pareto: `{decision['claim_narrowing_required_for_raw_unconstrained_pareto']}`",
        f"- Selected point freeze state: `{decision['selected_point_freeze_state']}`",
        "",
        "## Selected `suds_pareto` Rows",
        "",
        "| Workload | Design | Delta pp | EDP ratio | Area | Mem pressure | Control ratio | Calibration ratio | Raw PPA front | Raw dominators | R5 front |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in summary["selected_rows"]:
        lines.append(
            f"| `{row['workload']}` | `{row['design_id']}` | "
            f"{fmt(row['delta_accuracy_pp'], 3)} | {fmt(row['edp_ratio_vs_lightening'], 3)} | "
            f"{fmt(row['area_mm2'], 3)} | {fmt(row['memory_pressure_bytes_per_parallel_core'], 1)} | "
            f"{fmt(row['control_overhead_energy_ratio'], 6)} | "
            f"{fmt(row['calibration_sensitivity_energy_ratio'], 3)} | "
            f"`{row['raw_energy_latency_area_pareto_front']}` | "
            f"{row['raw_energy_latency_area_dominator_count']} | "
            f"`{row['r5_multiobjective_pareto_front']}` |"
        )

    lines.extend(
        [
            "",
            "## Raw PPA Dominance Pressure",
            "",
            "The following examples dominate the selected row on raw energy, latency,",
            "and area. They are not hidden; they are recorded as the reason the",
            "claim must be stated as a governed multi-objective selection rather",
            "than an unconstrained PPA optimum.",
            "",
            "| Workload | Design | EDP ratio | Energy ratio | Latency ratio | Area | Sideband | ADC sharing |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in summary["raw_energy_latency_area_dominator_examples"]:
        lines.append(
            f"| `{item['workload']}` | `{item['design_id']}` | "
            f"{fmt(item['edp_ratio_vs_lightening'], 3)} | "
            f"{fmt(item['energy_ratio_vs_lightening'], 3)} | "
            f"{fmt(item['latency_ratio_vs_lightening'], 3)} | "
            f"{fmt(item['area_mm2'], 3)} | {item['sideband_group_cols']} | "
            f"`{item['adc_sharing_mode']}` |"
        )

    lines.extend(
        [
            "",
            "## R5 Objectives",
            "",
            "| Objective | Meaning |",
            "|---|---|",
            "| `energy_pj` | Total modeled energy; lower is better. |",
            "| `latency_ns` | Total modeled latency; lower is better. |",
            "| `edp_pj_ns` | Energy-delay product; lower is better. |",
            "| `area_mm2` | Area proxy; lower is better. |",
            "| `memory_pressure_bytes_per_parallel_core` | Workload memory movement normalized by parallel cores; lower is better. |",
            "| `control_overhead_energy_ratio` | Sideband-control energy share; lower is better. |",
            "| `calibration_sensitivity_energy_ratio` | Conversion, laser, detector, and optical-link energy share used as a calibration-sensitivity proxy; lower is better. |",
            "",
            "## Design-Space Coverage",
            "",
            f"- Input design-space rows: `{summary['input_design_space_rows']}`",
            f"- R5 enriched rows: `{summary['rows']}`",
            f"- Input raw Pareto rows: `{summary['input_raw_pareto_rows']}`",
            f"- R3 input acceptance: `{summary['input_r3_acceptance_state']}`",
            f"- R4 input acceptance: `{summary['input_r4_acceptance_state']}`",
            "",
            "## Artifacts",
            "",
            f"- Extended design-space CSV: `{repo_path(args.csv_out)}`",
            f"- Extended design-space JSON: `{repo_path(args.json_out)}`",
            f"- Report: `{repo_path(args.report_out)}`",
            f"- Architecture design-space input: `{repo_path(args.design_space_json)}`",
            f"- R3 accuracy input: `{repo_path(args.end_to_end_accuracy_json)}`",
            f"- R4 baseline fairness input: `{repo_path(args.same_sim_baselines_json)}`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "make suds-tetc-pareto-design-space",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, design_json, r3_json, r4_json = build_rows(args)
    summary = build_summary(rows, design_json=design_json, r3_json=r3_json, r4_json=r4_json, args=args)
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
    print(f"r5_acceptance_state={summary['decision']['r5_acceptance_state']}")
    print(f"stop_condition_state={summary['decision']['stop_condition_state']}")


if __name__ == "__main__":
    main()
