#!/usr/bin/env python3
"""Build the R6 system sensitivity artifact for the SUDS TETC route.

R6 answers a specific architecture-review objection: memory movement,
conversion, optical links, laser power, and control signaling can erase
compute-side gains in photonic accelerators. This generator therefore keeps the
promoted SUDS point fixed and sweeps system parameters around the same-scope
Lightening-style DPTC reference.

The output is a bounded parametric audit. It is not a new hardware measurement,
device-solver result, or model-accuracy run.
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

ARCH_SUMMARY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_summary.csv"
END_TO_END_ACCURACY_JSON = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{TAG}.json"
PARETO_DESIGN_SPACE_JSON = REPORT_DATA / f"suds_tetc_pareto_design_space_{TAG}.json"

CSV_OUT = REPORT_DATA / f"suds_tetc_system_sensitivity_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_system_sensitivity_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_system_sensitivity.md"

PROMOTED_CONDITION = "suds_pareto"
REFERENCE_CONDITION = "lightening_dptc"
WORKLOADS_REQUIRED = ("bert_base_glue_seq128", "mobilevit_s_transformer_blocks_256")
BENEFIT_TOLERANCE = 1e-12
MATERIAL_EDP_IMPROVEMENT_PCT = 5.0

ENERGY_COMPONENTS = (
    "adc_energy_pj",
    "dac_mzm_energy_pj",
    "detector_tia_energy_pj",
    "laser_energy_pj",
    "memory_energy_pj",
    "optical_link_energy_pj",
    "control_energy_pj",
)

SCENARIO_FIELDS = (
    "memory_bandwidth_scale",
    "activation_reuse_scale",
    "batch_size_scale",
    "sequence_length_scale",
    "adc_energy_scale",
    "dac_energy_scale",
    "laser_multiplier",
    "optical_link_loss_scale",
    "sideband_control_overhead_scale",
)

DEFAULT_SCENARIO = {
    "memory_bandwidth_scale": 1.0,
    "activation_reuse_scale": 1.0,
    "batch_size_scale": 1.0,
    "sequence_length_scale": 1.0,
    "adc_energy_scale": 1.0,
    "dac_energy_scale": 1.0,
    "laser_multiplier": 1.0,
    "optical_link_loss_scale": 1.0,
    "sideband_control_overhead_scale": 1.0,
}

NAMED_REGIMES = {
    "optimistic": {
        "description": "Higher bandwidth/reuse and lower conversion/link/control costs.",
        "values": {
            "memory_bandwidth_scale": 2.0,
            "activation_reuse_scale": 1.4,
            "batch_size_scale": 4.0,
            "sequence_length_scale": 1.0,
            "adc_energy_scale": 0.75,
            "dac_energy_scale": 0.85,
            "laser_multiplier": 0.85,
            "optical_link_loss_scale": 0.75,
            "sideband_control_overhead_scale": 0.5,
        },
    },
    "nominal": {
        "description": "The selected architecture summary surface.",
        "values": dict(DEFAULT_SCENARIO),
    },
    "pessimistic": {
        "description": "Lower bandwidth/reuse with stressed ADC, DAC, laser, link, and sideband terms.",
        "values": {
            "memory_bandwidth_scale": 0.5,
            "activation_reuse_scale": 0.65,
            "batch_size_scale": 1.0,
            "sequence_length_scale": 2.0,
            "adc_energy_scale": 2.0,
            "dac_energy_scale": 1.25,
            "laser_multiplier": 1.15,
            "optical_link_loss_scale": 1.25,
            "sideband_control_overhead_scale": 3.0,
        },
    },
}

AXIS_SWEEPS = {
    "memory_bandwidth_scale": (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
    "activation_reuse_scale": (0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    "batch_size_scale": (1.0, 2.0, 4.0, 8.0, 16.0),
    "sequence_length_scale": (0.5, 1.0, 2.0, 4.0),
    "adc_energy_scale": (0.5, 1.0, 2.0, 4.0, 8.0),
    "dac_energy_scale": (0.5, 1.0, 1.25, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
    "laser_multiplier": (0.5, 0.85, 1.0, 1.15, 1.5, 2.0, 4.0),
    "optical_link_loss_scale": (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0),
    "sideband_control_overhead_scale": (0.5, 1.0, 3.0, 5.0, 10.0, 25.0, 100.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--architecture-summary-csv", type=Path, default=ARCH_SUMMARY_CSV)
    parser.add_argument("--end-to-end-accuracy-json", type=Path, default=END_TO_END_ACCURACY_JSON)
    parser.add_argument("--pareto-design-space-json", type=Path, default=PARETO_DESIGN_SPACE_JSON)
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


def slug(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def scenarios() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, spec in NAMED_REGIMES.items():
        values = dict(DEFAULT_SCENARIO)
        values.update(spec["values"])
        out.append(
            {
                "scenario_id": f"r6_named_{name}",
                "regime": name,
                "sweep_axis": "named_regime",
                "sweep_value": "",
                "realistic_regime": name in {"nominal", "pessimistic"},
                "description": spec["description"],
                **values,
            }
        )

    for axis, values in AXIS_SWEEPS.items():
        for value in values:
            scenario_values = dict(DEFAULT_SCENARIO)
            scenario_values[axis] = float(value)
            out.append(
                {
                    "scenario_id": f"r6_sweep_{axis}_{slug(float(value))}",
                    "regime": "one_at_a_time_sweep",
                    "sweep_axis": axis,
                    "sweep_value": float(value),
                    "realistic_regime": False,
                    "description": f"One-at-a-time R6 sweep for {axis}.",
                    **scenario_values,
                }
            )
    return out


def nominal_selected_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("sensitivity_case") != "nominal":
            continue
        key = (row.get("workload", ""), row.get("condition", ""))
        if key[0] and key[1]:
            index[key] = row
    return index


def estimate(row: dict[str, str], scenario: dict[str, Any]) -> dict[str, float]:
    energy = as_float(row.get("energy_pj"), 0.0)
    latency = as_float(row.get("latency_ns"), 0.0)
    components = {field: as_float(row.get(field), 0.0) for field in ENERGY_COMPONENTS}
    modeled_other = max(0.0, energy - sum(components.values()))

    batch = max(0.05, as_float(scenario["batch_size_scale"], 1.0))
    seq = max(0.05, as_float(scenario["sequence_length_scale"], 1.0))
    workload_scale = batch * seq
    reuse = max(0.05, as_float(scenario["activation_reuse_scale"], 1.0))
    bandwidth = max(0.05, as_float(scenario["memory_bandwidth_scale"], 1.0))
    adc_scale = max(0.01, as_float(scenario["adc_energy_scale"], 1.0))
    dac_scale = max(0.01, as_float(scenario["dac_energy_scale"], 1.0))
    laser_scale = max(0.01, as_float(scenario["laser_multiplier"], 1.0))
    link_loss = max(0.01, as_float(scenario["optical_link_loss_scale"], 1.0))
    sideband_scale = max(0.01, as_float(scenario["sideband_control_overhead_scale"], 1.0))

    # Larger batches improve weight/activation reuse but still increase total
    # moved bytes. Bandwidth is primarily a latency term; a small energy factor
    # records less efficient memory operation when bandwidth is stressed.
    memory_common = seq * batch / (reuse * batch**0.25)
    memory_energy_factor = memory_common * max(0.05, 1.0 + 0.10 * ((1.0 / bandwidth) - 1.0))
    memory_latency_factor = seq * batch / (reuse * bandwidth * batch**0.20)
    control_factor = seq * batch**0.85 * sideband_scale
    conversion_latency_factor = workload_scale * (0.65 * math.sqrt(adc_scale) + 0.35 * math.sqrt(dac_scale))
    link_latency_factor = workload_scale * math.sqrt(link_loss)

    adc_energy = components["adc_energy_pj"] * workload_scale * adc_scale
    dac_mzm_energy = components["dac_mzm_energy_pj"] * workload_scale * dac_scale
    detector_energy = components["detector_tia_energy_pj"] * workload_scale * link_loss
    laser_energy = components["laser_energy_pj"] * workload_scale * laser_scale * link_loss
    memory_energy = components["memory_energy_pj"] * memory_energy_factor
    optical_link_energy = components["optical_link_energy_pj"] * workload_scale * link_loss
    control_energy = components["control_energy_pj"] * control_factor
    other_energy = modeled_other * workload_scale

    estimated_energy = (
        other_energy
        + adc_energy
        + dac_mzm_energy
        + detector_energy
        + laser_energy
        + memory_energy
        + optical_link_energy
        + control_energy
    )
    latency_factor = (
        0.55 * workload_scale
        + 0.12 * conversion_latency_factor
        + 0.23 * memory_latency_factor
        + 0.06 * link_latency_factor
        + 0.04 * control_factor
    )
    estimated_latency = latency * latency_factor

    return {
        "estimated_energy_pj": estimated_energy,
        "estimated_latency_ns": estimated_latency,
        "estimated_edp_pj_ns": estimated_energy * estimated_latency,
        "estimated_adc_energy_pj": adc_energy,
        "estimated_dac_mzm_energy_pj": dac_mzm_energy,
        "estimated_detector_tia_energy_pj": detector_energy,
        "estimated_laser_energy_pj": laser_energy,
        "estimated_memory_energy_pj": memory_energy,
        "estimated_optical_link_energy_pj": optical_link_energy,
        "estimated_control_energy_pj": control_energy,
        "estimated_other_energy_pj": other_energy,
        "conversion_latency_factor": conversion_latency_factor,
        "memory_latency_factor": memory_latency_factor,
        "control_latency_factor": control_factor,
    }


def classify(edp_improvement_pct: float) -> str:
    if edp_improvement_pct <= BENEFIT_TOLERANCE:
        return "not_beneficial"
    if edp_improvement_pct < MATERIAL_EDP_IMPROVEMENT_PCT:
        return "thin_margin_boundary"
    return "beneficial"


def build_comparison_rows(
    index: dict[tuple[str, str], dict[str, str]],
    scenario_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scenario in scenario_rows:
        for workload in WORKLOADS_REQUIRED:
            reference = index.get((workload, REFERENCE_CONDITION))
            promoted = index.get((workload, PROMOTED_CONDITION))
            if reference is None or promoted is None:
                continue
            ref_est = estimate(reference, scenario)
            suds_est = estimate(promoted, scenario)
            energy_ratio = suds_est["estimated_energy_pj"] / ref_est["estimated_energy_pj"]
            latency_ratio = suds_est["estimated_latency_ns"] / ref_est["estimated_latency_ns"]
            edp_ratio = suds_est["estimated_edp_pj_ns"] / ref_est["estimated_edp_pj_ns"]
            energy_improvement_pct = (1.0 - energy_ratio) * 100.0
            edp_improvement_pct = (1.0 - edp_ratio) * 100.0
            benefit_preserved = edp_ratio < 1.0 - BENEFIT_TOLERANCE
            conversion_link_share = (
                suds_est["estimated_adc_energy_pj"]
                + suds_est["estimated_dac_mzm_energy_pj"]
                + suds_est["estimated_detector_tia_energy_pj"]
                + suds_est["estimated_laser_energy_pj"]
                + suds_est["estimated_optical_link_energy_pj"]
            ) / max(1e-12, suds_est["estimated_energy_pj"])
            memory_share = suds_est["estimated_memory_energy_pj"] / max(1e-12, suds_est["estimated_energy_pj"])
            control_share = suds_est["estimated_control_energy_pj"] / max(1e-12, suds_est["estimated_energy_pj"])

            row = {
                "tag": TAG,
                "roadmap_item": "R6_memory_bandwidth_conversion_link_sensitivity",
                "scenario_id": scenario["scenario_id"],
                "regime": scenario["regime"],
                "sweep_axis": scenario["sweep_axis"],
                "sweep_value": scenario["sweep_value"],
                "realistic_regime": scenario["realistic_regime"],
                "workload": workload,
                "promoted_condition": PROMOTED_CONDITION,
                "reference_condition": REFERENCE_CONDITION,
                "memory_bandwidth_scale": scenario["memory_bandwidth_scale"],
                "activation_reuse_scale": scenario["activation_reuse_scale"],
                "batch_size_scale": scenario["batch_size_scale"],
                "sequence_length_scale": scenario["sequence_length_scale"],
                "adc_energy_scale": scenario["adc_energy_scale"],
                "dac_energy_scale": scenario["dac_energy_scale"],
                "laser_multiplier": scenario["laser_multiplier"],
                "optical_link_loss_scale": scenario["optical_link_loss_scale"],
                "sideband_control_overhead_scale": scenario["sideband_control_overhead_scale"],
                "suds_energy_pj": suds_est["estimated_energy_pj"],
                "reference_energy_pj": ref_est["estimated_energy_pj"],
                "energy_ratio_vs_reference": energy_ratio,
                "energy_improvement_vs_reference_pct": energy_improvement_pct,
                "suds_latency_ns": suds_est["estimated_latency_ns"],
                "reference_latency_ns": ref_est["estimated_latency_ns"],
                "latency_ratio_vs_reference": latency_ratio,
                "suds_edp_pj_ns": suds_est["estimated_edp_pj_ns"],
                "reference_edp_pj_ns": ref_est["estimated_edp_pj_ns"],
                "edp_ratio_vs_reference": edp_ratio,
                "edp_improvement_vs_reference_pct": edp_improvement_pct,
                "benefit_preserved": benefit_preserved,
                "benefit_class": classify(edp_improvement_pct),
                "suds_conversion_link_energy_share": conversion_link_share,
                "suds_memory_energy_share": memory_share,
                "suds_control_energy_share": control_share,
                "suds_memory_latency_factor": suds_est["memory_latency_factor"],
                "suds_conversion_latency_factor": suds_est["conversion_latency_factor"],
                "suds_control_latency_factor": suds_est["control_latency_factor"],
                "nominal_accuracy_evidence_label": promoted.get("accuracy_evidence_label", ""),
                "nominal_device": promoted.get("device", ""),
                "claim_boundary": (
                    "Parametric architecture sensitivity from modeled R6 component scaling; "
                    "not a hardware measurement, layout signoff, device-solver result, or new accuracy run."
                ),
                "scenario_description": scenario["description"],
            }
            out.append(row)
    return out


def named_regime_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for regime in NAMED_REGIMES:
        regime_rows = [row for row in rows if row["regime"] == regime]
        if not regime_rows:
            continue
        summary[regime] = {
            "all_workloads_benefit_preserved": all(row["benefit_preserved"] for row in regime_rows),
            "min_edp_improvement_pct": min(as_float(row["edp_improvement_vs_reference_pct"]) for row in regime_rows),
            "max_edp_ratio_vs_reference": max(as_float(row["edp_ratio_vs_reference"]) for row in regime_rows),
            "rows": [
                {
                    "workload": row["workload"],
                    "edp_ratio_vs_reference": as_float(row["edp_ratio_vs_reference"]),
                    "edp_improvement_vs_reference_pct": as_float(row["edp_improvement_vs_reference_pct"]),
                    "energy_improvement_vs_reference_pct": as_float(row["energy_improvement_vs_reference_pct"]),
                    "benefit_class": row["benefit_class"],
                }
                for row in regime_rows
            ],
        }
    return summary


def axis_boundary_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    axis_rows = [row for row in rows if row["regime"] == "one_at_a_time_sweep"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in axis_rows:
        grouped[(str(row["sweep_axis"]), str(row["workload"]))].append(row)

    for (axis, workload), group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda item: as_float(item["sweep_value"]))
        beneficial = [row for row in ordered if row["benefit_preserved"]]
        not_beneficial = [row for row in ordered if not row["benefit_preserved"]]
        thin = [row for row in ordered if row["benefit_class"] == "thin_margin_boundary"]
        out.append(
            {
                "sweep_axis": axis,
                "workload": workload,
                "beneficial_values": [as_float(row["sweep_value"]) for row in beneficial],
                "thin_margin_values": [as_float(row["sweep_value"]) for row in thin],
                "not_beneficial_values": [as_float(row["sweep_value"]) for row in not_beneficial],
                "last_beneficial_value": as_float(beneficial[-1]["sweep_value"]) if beneficial else None,
                "first_not_beneficial_value": as_float(not_beneficial[0]["sweep_value"]) if not_beneficial else None,
                "min_edp_improvement_pct": min(as_float(row["edp_improvement_vs_reference_pct"]) for row in ordered),
                "max_edp_ratio_vs_reference": max(as_float(row["edp_ratio_vs_reference"]) for row in ordered),
            }
        )
    return out


def evidence_acceptance(r3_json: dict[str, Any], r5_json: dict[str, Any]) -> tuple[str, str]:
    r3 = r3_json.get("summary", {}).get("decision", {}).get("r3_acceptance_state", "")
    r5 = r5_json.get("summary", {}).get("decision", {}).get("r5_acceptance_state", "")
    return str(r3), str(r5)


def build_summary(
    rows: list[dict[str, Any]],
    *,
    index: dict[tuple[str, str], dict[str, str]],
    r3_json: dict[str, Any],
    r5_json: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    named = named_regime_summary(rows)
    boundaries = axis_boundary_summary(rows)
    r3_state, r5_state = evidence_acceptance(r3_json, r5_json)

    present_workloads = sorted(
        workload
        for workload in WORKLOADS_REQUIRED
        if (workload, PROMOTED_CONDITION) in index and (workload, REFERENCE_CONDITION) in index
    )
    missing_workloads = sorted(set(WORKLOADS_REQUIRED) - set(present_workloads))
    named_realistic_ok = all(
        named.get(regime, {}).get("all_workloads_benefit_preserved") is True
        for regime in ("nominal", "pessimistic")
    )
    optimistic_ok = named.get("optimistic", {}).get("all_workloads_benefit_preserved") is True
    not_beneficial_rows = [
        row for row in rows
        if row["regime"] == "one_at_a_time_sweep" and not row["benefit_preserved"]
    ]
    thin_rows = [
        row for row in rows
        if row["regime"] == "one_at_a_time_sweep" and row["benefit_class"] == "thin_margin_boundary"
    ]

    blockers: list[str] = []
    if missing_workloads:
        blockers.append("required_workload_pair_missing")
    if r3_state != "pass":
        blockers.append("r3_acceptance_state_not_pass")
    if r5_state != "pass":
        blockers.append("r5_acceptance_state_not_pass")
    if not named_realistic_ok:
        blockers.append("realistic_named_regime_erases_promoted_benefit")
    if not optimistic_ok:
        blockers.append("optimistic_named_regime_erases_promoted_benefit")

    acceptance_state = "pass" if not blockers else "fail"
    realistic_stop = not named_realistic_ok
    stop_condition_state = (
        "R6 hard stop: a nominal or pessimistic memory/conversion/link regime eliminates the promoted benefit"
        if realistic_stop
        else "no R6 hard stop; nominal and pessimistic named regimes preserve the bounded promoted EDP claim"
    )

    failure_examples = [
        {
            "workload": row["workload"],
            "sweep_axis": row["sweep_axis"],
            "sweep_value": as_float(row["sweep_value"]),
            "edp_ratio_vs_reference": as_float(row["edp_ratio_vs_reference"]),
            "edp_improvement_vs_reference_pct": as_float(row["edp_improvement_vs_reference_pct"]),
        }
        for row in sorted(not_beneficial_rows, key=lambda item: as_float(item["edp_ratio_vs_reference"], -math.inf), reverse=True)[:12]
    ]

    return {
        "tag": args.tag,
        "date": DATE,
        "roadmap_item": "R6_memory_bandwidth_conversion_link_sensitivity",
        "rows": len(rows),
        "required_workloads": list(WORKLOADS_REQUIRED),
        "present_workloads": present_workloads,
        "missing_workloads": missing_workloads,
        "named_regimes": named,
        "axis_boundaries": boundaries,
        "not_beneficial_sweep_rows": len(not_beneficial_rows),
        "thin_margin_sweep_rows": len(thin_rows),
        "not_beneficial_examples": failure_examples,
        "minimum_pessimistic_edp_improvement_pct": named.get("pessimistic", {}).get("min_edp_improvement_pct"),
        "input_r3_acceptance_state": r3_state,
        "input_r5_acceptance_state": r5_state,
        "decision": {
            "r6_acceptance_state": acceptance_state,
            "stop_condition_state": stop_condition_state,
            "blockers": blockers,
            "realistic_stop_condition_triggered": realistic_stop,
            "named_nominal_and_pessimistic_preserve_benefit": named_realistic_ok,
            "optimistic_preserves_benefit": optimistic_ok,
            "claim_narrowing_required_for_extreme_sweeps": bool(not_beneficial_rows or thin_rows),
            "valid_regime_statement": (
                "The promoted claim is valid for the nominal, optimistic, and pessimistic named R6 regimes. "
                "One-at-a-time extreme sweeps are boundary evidence and are not promoted as realistic operating regimes."
            ),
        },
        "artifacts": {
            "csv": repo_path(args.csv_out),
            "json": repo_path(args.json_out),
            "report": repo_path(args.report_out),
            "architecture_summary_csv": repo_path(args.architecture_summary_csv),
            "r3_json": repo_path(args.end_to_end_accuracy_json),
            "r5_json": repo_path(args.pareto_design_space_json),
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
            "roadmap_item": "R6_memory_bandwidth_conversion_link_sensitivity",
            "artifact_id": f"suds_tetc_system_sensitivity_{args.tag}",
            "evidence_label": "system_memory_conversion_link_sensitivity",
            "promotion_decision": (
                "r6_system_sensitivity_pass"
                if summary["decision"]["r6_acceptance_state"] == "pass"
                else "r6_system_sensitivity_fail"
            ),
            "regeneration_command": "make suds-tetc-system-sensitivity",
        },
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    named_rows = [row for row in rows if row["sweep_axis"] == "named_regime"]
    lines = [
        "# SUDS TETC System Sensitivity",
        "",
        f"Tag: `{args.tag}`",
        "Roadmap item: `R6_memory_bandwidth_conversion_link_sensitivity`",
        "Evidence label: `system_memory_conversion_link_sensitivity`",
        f"Acceptance state: `{decision['r6_acceptance_state']}`",
        f"Stop-condition state: `{decision['stop_condition_state']}`",
        "",
        "## Scope",
        "",
        "This artifact holds the promoted `suds_pareto` point fixed and sweeps",
        "memory bandwidth, activation reuse, batch size, sequence length, ADC",
        "energy, DAC/MZM energy, laser multiplier, optical-link loss, and",
        "sideband-control overhead against the same-scope Lightening-style DPTC",
        "reference. The rows are parametric architecture sensitivity rows, not",
        "new hardware measurements or model-accuracy runs.",
        "",
        "## Decision",
        "",
        f"- R6 acceptance: `{decision['r6_acceptance_state']}`",
        f"- Blockers: `{';'.join(decision['blockers']) or 'none'}`",
        f"- Nominal and pessimistic named regimes preserve benefit: `{decision['named_nominal_and_pessimistic_preserve_benefit']}`",
        f"- Claim narrowing required for extreme sweeps: `{decision['claim_narrowing_required_for_extreme_sweeps']}`",
        f"- Minimum pessimistic EDP improvement: `{fmt(summary['minimum_pessimistic_edp_improvement_pct'], 2)}%`",
        "",
        "## Named Regimes",
        "",
        "| Regime | Workload | Energy improvement | EDP improvement | EDP ratio | Class |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in named_rows:
        lines.append(
            f"| `{row['regime']}` | `{row['workload']}` | "
            f"{fmt(row['energy_improvement_vs_reference_pct'], 2)}% | "
            f"{fmt(row['edp_improvement_vs_reference_pct'], 2)}% | "
            f"{fmt(row['edp_ratio_vs_reference'], 3)} | `{row['benefit_class']}` |"
        )

    lines.extend(
        [
            "",
            "## Sweep Boundaries",
            "",
            "| Axis | Workload | Last beneficial value | First not-beneficial value | Min EDP improvement | Max EDP ratio |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["axis_boundaries"]:
        lines.append(
            f"| `{item['sweep_axis']}` | `{item['workload']}` | "
            f"{fmt(item['last_beneficial_value'], 3)} | "
            f"{fmt(item['first_not_beneficial_value'], 3)} | "
            f"{fmt(item['min_edp_improvement_pct'], 2)}% | "
            f"{fmt(item['max_edp_ratio_vs_reference'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Boundary Examples",
            "",
            "Rows below are one-at-a-time extreme sweeps where the promoted point is",
            "not beneficial. They bound the claim and are not used as promoted",
            "realistic regimes.",
            "",
            "| Workload | Axis | Value | EDP ratio | EDP improvement |",
            "|---|---|---:|---:|---:|",
        ]
    )
    if summary["not_beneficial_examples"]:
        for item in summary["not_beneficial_examples"]:
            lines.append(
                f"| `{item['workload']}` | `{item['sweep_axis']}` | "
                f"{fmt(item['sweep_value'], 3)} | {fmt(item['edp_ratio_vs_reference'], 3)} | "
                f"{fmt(item['edp_improvement_vs_reference_pct'], 2)}% |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- System sensitivity CSV: `{repo_path(args.csv_out)}`",
            f"- System sensitivity JSON: `{repo_path(args.json_out)}`",
            f"- Report: `{repo_path(args.report_out)}`",
            f"- Architecture summary input: `{repo_path(args.architecture_summary_csv)}`",
            f"- R3 accuracy input: `{repo_path(args.end_to_end_accuracy_json)}`",
            f"- R5 Pareto input: `{repo_path(args.pareto_design_space_json)}`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "make suds-tetc-system-sensitivity",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = nominal_selected_rows(load_csv(args.architecture_summary_csv))
    r3_json = load_json(args.end_to_end_accuracy_json)
    r5_json = load_json(args.pareto_design_space_json)
    rows = build_comparison_rows(index, scenarios())
    summary = build_summary(rows, index=index, r3_json=r3_json, r5_json=r5_json, args=args)
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
    print(f"r6_acceptance_state={summary['decision']['r6_acceptance_state']}")
    print(f"stop_condition_state={summary['decision']['stop_condition_state']}")


if __name__ == "__main__":
    main()
