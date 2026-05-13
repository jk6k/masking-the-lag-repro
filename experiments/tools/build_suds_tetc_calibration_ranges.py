#!/usr/bin/env python3
"""Build the R8 ADC and photonic calibration range artifact.

R8 converts single-point ADC and photonic assumptions into explicit nominal,
optimistic, pessimistic, and boundary ranges. The ranges are generated from the
existing local ADC macro sanity suite, PHY boundary sweep, R6 sensitivity audit,
and architecture parameter table. They are calibration and boundary evidence
only, not device-solver, foundry, extracted-layout, silicon, or bench evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
DATE = "2026-05-14"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

PARAMETERS_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_parameters.csv"
ADC_MACRO_CSV = REPORT_DATA / "suds_adc_macro_sanity_20260512_j1_quality_boost.csv"
ADC_MACRO_JSON = REPORT_DATA / "suds_adc_macro_sanity_20260512_j1_quality_boost.json"
PHY_BOUNDARY_CSV = REPORT_DATA / "suds_phy_circuit_boundary_20260511_p2p3_quality.csv"
PHY_BOUNDARY_JSON = REPORT_DATA / "suds_phy_circuit_boundary_20260511_p2p3_quality.json"
SYSTEM_SENSITIVITY_JSON = REPORT_DATA / f"suds_tetc_system_sensitivity_{TAG}.json"
RTL_CONTROL_JSON = REPORT_DATA / f"suds_tetc_rtl_control_plane_{TAG}.json"

CSV_OUT = REPORT_DATA / f"suds_tetc_calibration_ranges_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_calibration_ranges_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_calibration_ranges.md"

ROADMAP_ITEM = "R8_adc_and_photonic_calibration_deepening"
EVIDENCE_LABEL = "adc_photonic_calibration_ranges"
CLAIM_BOUNDARY = (
    "Calibration and boundary evidence only; not device-solver, foundry, "
    "extracted-layout, silicon, bench-energy, Lumerical, Spectre, or P&R closure."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--parameters-csv", type=Path, default=PARAMETERS_CSV)
    parser.add_argument("--adc-macro-csv", type=Path, default=ADC_MACRO_CSV)
    parser.add_argument("--adc-macro-json", type=Path, default=ADC_MACRO_JSON)
    parser.add_argument("--phy-boundary-csv", type=Path, default=PHY_BOUNDARY_CSV)
    parser.add_argument("--phy-boundary-json", type=Path, default=PHY_BOUNDARY_JSON)
    parser.add_argument("--system-sensitivity-json", type=Path, default=SYSTEM_SENSITIVITY_JSON)
    parser.add_argument("--rtl-control-json", type=Path, default=RTL_CONTROL_JSON)
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
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def finite_values(values: list[Any]) -> list[float]:
    out = [as_float(value) for value in values]
    return [value for value in out if not math.isnan(value)]


def fmt(value: Any, digits: int = 4) -> str:
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


def parameter_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("parameter", "")): row for row in rows if row.get("parameter")}


def nominal_param(params: dict[str, dict[str, str]], name: str, default: float = math.nan) -> float:
    row = params.get(name, {})
    return as_float(row.get("value"), default)


def pass_ratio(rows: list[dict[str, str]], **filters: Any) -> float:
    selected = [
        row
        for row in rows
        if all(str(row.get(key, "")) == str(value) for key, value in filters.items())
    ]
    if not selected:
        return math.nan
    passes = sum(1 for row in selected if row.get("boundary_status") == "pass")
    return passes / len(selected)


def best_worst_phy_group_ratio(rows: list[dict[str, str]]) -> tuple[float, float]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row.get("er_db", ""), row.get("xtalk_db", "")), []).append(row)
    ratios = [sum(1 for row in group if row.get("boundary_status") == "pass") / len(group) for group in groups.values() if group]
    return (max(ratios, default=math.nan), min(ratios, default=math.nan))


def r6_axis_boundary(r6: dict[str, Any], axis: str) -> dict[str, Any]:
    boundaries = r6.get("summary", {}).get("axis_boundaries", [])
    selected = [row for row in boundaries if row.get("sweep_axis") == axis]
    if not selected:
        return {}
    first_not = [
        as_float(row.get("first_not_beneficial_value"))
        for row in selected
        if as_float(row.get("first_not_beneficial_value")) == as_float(row.get("first_not_beneficial_value"))
    ]
    last_beneficial = [
        as_float(row.get("last_beneficial_value"))
        for row in selected
        if as_float(row.get("last_beneficial_value")) == as_float(row.get("last_beneficial_value"))
    ]
    return {
        "first_not_beneficial_value": min(first_not) if first_not else math.nan,
        "max_last_beneficial_value": max(last_beneficial) if last_beneficial else math.nan,
    }


def range_row(
    *,
    args: argparse.Namespace,
    range_id: str,
    parameter: str,
    parameter_group: str,
    unit: str,
    nominal_value: float,
    optimistic_value: float,
    pessimistic_value: float,
    boundary_value: float,
    worse_direction: str,
    nominal_label: str,
    pessimistic_label: str,
    range_source: str,
    source_artifact: str,
    source_rows: int,
    architecture_parameter_linked: bool,
    r6_boundary_linked: bool,
    evidence_label: str,
    claim_boundary: str = CLAIM_BOUNDARY,
) -> dict[str, Any]:
    has_nominal = not math.isnan(nominal_value)
    has_pessimistic = not math.isnan(pessimistic_value)
    status = "pass" if has_nominal and has_pessimistic and source_rows > 0 else "fail"
    return {
        "tag": args.tag,
        "date": DATE,
        "roadmap_item": ROADMAP_ITEM,
        "range_id": range_id,
        "parameter": parameter,
        "parameter_group": parameter_group,
        "unit": unit,
        "nominal_value": nominal_value,
        "optimistic_value": optimistic_value,
        "pessimistic_value": pessimistic_value,
        "boundary_value": boundary_value,
        "worse_direction": worse_direction,
        "nominal_label": nominal_label,
        "pessimistic_label": pessimistic_label,
        "range_source": range_source,
        "evidence_label": evidence_label,
        "source_artifact": source_artifact,
        "source_rows": source_rows,
        "architecture_parameter_linked": architecture_parameter_linked,
        "r6_boundary_linked": r6_boundary_linked,
        "claim_boundary": claim_boundary,
        "acceptance_status": status,
    }


def adc_rows(args: argparse.Namespace, params: dict[str, dict[str, str]], adc_rows_in: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    measured = [row for row in adc_rows_in if row.get("status", "measured") == "measured"]
    for bits in (4, 6, 8):
        bit_rows = [row for row in measured if int(as_float(row.get("adc_bits"), -1)) == bits]
        energy_values = finite_values([row.get("energy_per_conversion_pj") for row in bit_rows])
        latency_values = finite_values([row.get("latency_ps") for row in bit_rows])
        nominal_rows = [row for row in bit_rows if row.get("case_kind", row.get("mismatch_case", "")) == "nominal"]
        nominal_latency_values = finite_values([row.get("latency_ps") for row in nominal_rows])
        architecture_energy = nominal_param(params, f"adc{bits}_pj", median(energy_values) if energy_values else math.nan)
        nominal_latency = median(nominal_latency_values) if nominal_latency_values else (median(latency_values) if latency_values else math.nan)
        rows.append(
            range_row(
                args=args,
                range_id=f"r8_adc{bits}_energy_pj",
                parameter=f"adc{bits}_energy_pj",
                parameter_group="adc_tier_energy",
                unit="pJ/conversion",
                nominal_value=architecture_energy,
                optimistic_value=min(energy_values, default=math.nan),
                pessimistic_value=max(energy_values, default=math.nan),
                boundary_value=max(energy_values, default=math.nan),
                worse_direction="higher",
                nominal_label="architecture median ADC macro calibration",
                pessimistic_label="maximum measured J1 ADC macro energy corner",
                range_source="J1 ADC macro measured rows",
                source_artifact=repo_path(args.adc_macro_csv),
                source_rows=len(bit_rows),
                architecture_parameter_linked=f"adc{bits}_pj" in params,
                r6_boundary_linked=False,
                evidence_label="spice_macro",
                claim_boundary="ADC macro calibration only; not PDK, extracted-layout, silicon, measured hardware energy, or full SPICE closure.",
            )
        )
        rows.append(
            range_row(
                args=args,
                range_id=f"r8_adc{bits}_latency_ps",
                parameter=f"adc{bits}_latency_ps",
                parameter_group="adc_tier_latency",
                unit="ps/conversion",
                nominal_value=nominal_latency,
                optimistic_value=min(latency_values, default=math.nan),
                pessimistic_value=max(latency_values, default=math.nan),
                boundary_value=max(latency_values, default=math.nan),
                worse_direction="higher",
                nominal_label="nominal J1 ADC macro latency corner",
                pessimistic_label="slowest measured J1 ADC macro rate corner",
                range_source="J1 ADC macro measured low/nominal/high-rate rows",
                source_artifact=repo_path(args.adc_macro_csv),
                source_rows=len(bit_rows),
                architecture_parameter_linked=False,
                r6_boundary_linked=False,
                evidence_label="spice_macro",
                claim_boundary="ADC macro latency calibration only; not timing closure or hardware measurement.",
            )
        )
    return rows


def photonic_rows(
    args: argparse.Namespace,
    params: dict[str, dict[str, str]],
    phy_rows: list[dict[str, str]],
    r6_json: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    er_values = finite_values([row.get("er_db") for row in phy_rows])
    xtalk_values = finite_values([row.get("xtalk_db") for row in phy_rows])
    pass_rows = [row for row in phy_rows if row.get("boundary_status") == "pass"]
    p_laser_all = finite_values([row.get("p_laser_mw") for row in phy_rows])
    p_laser_pass = finite_values([row.get("p_laser_mw") for row in pass_rows])
    best_phy_ratio, worst_phy_ratio = best_worst_phy_group_ratio(phy_rows)
    nominal_pass_ratio = nominal_param(
        params,
        "phy_nominal_pass_ratio",
        sum(1 for row in phy_rows if row.get("boundary_status") == "pass") / max(1, len(phy_rows)),
    )
    laser_boundary = r6_axis_boundary(r6_json, "laser_multiplier")
    link_boundary = r6_axis_boundary(r6_json, "optical_link_loss_scale")
    dac_boundary = r6_axis_boundary(r6_json, "dac_energy_scale")

    rows.extend(
        [
            range_row(
                args=args,
                range_id="r8_modulator_extinction_ratio_db",
                parameter="modulator_extinction_ratio_db",
                parameter_group="photonic_noise",
                unit="dB",
                nominal_value=6.0 if 6.0 in er_values else median(er_values),
                optimistic_value=max(er_values, default=math.nan),
                pessimistic_value=min(er_values, default=math.nan),
                boundary_value=min(er_values, default=math.nan),
                worse_direction="lower",
                nominal_label="middle PHY ER sweep corner",
                pessimistic_label="lowest ER PHY stress corner",
                range_source="PHY boundary sweep extinction-ratio axis",
                source_artifact=repo_path(args.phy_boundary_csv),
                source_rows=len(phy_rows),
                architecture_parameter_linked=False,
                r6_boundary_linked=False,
                evidence_label="parametric_boundary",
            ),
            range_row(
                args=args,
                range_id="r8_detector_crosstalk_db",
                parameter="detector_crosstalk_db",
                parameter_group="photonic_noise",
                unit="dB",
                nominal_value=-25.0 if -25.0 in xtalk_values else median(xtalk_values),
                optimistic_value=min(xtalk_values, default=math.nan),
                pessimistic_value=max(xtalk_values, default=math.nan),
                boundary_value=max(xtalk_values, default=math.nan),
                worse_direction="higher",
                nominal_label="middle PHY crosstalk sweep corner",
                pessimistic_label="highest crosstalk PHY stress corner",
                range_source="PHY boundary sweep crosstalk axis",
                source_artifact=repo_path(args.phy_boundary_csv),
                source_rows=len(phy_rows),
                architecture_parameter_linked=False,
                r6_boundary_linked=False,
                evidence_label="parametric_boundary",
            ),
            range_row(
                args=args,
                range_id="r8_phy_pass_ratio",
                parameter="phy_pass_ratio",
                parameter_group="photonic_boundary",
                unit="pass fraction",
                nominal_value=nominal_pass_ratio,
                optimistic_value=best_phy_ratio,
                pessimistic_value=worst_phy_ratio,
                boundary_value=worst_phy_ratio,
                worse_direction="lower",
                nominal_label="overall PHY pass ratio tied to architecture table",
                pessimistic_label="worst ER/crosstalk grouped PHY pass ratio",
                range_source="PHY boundary sweep pass/fail summary",
                source_artifact=repo_path(args.phy_boundary_csv),
                source_rows=len(phy_rows),
                architecture_parameter_linked="phy_nominal_pass_ratio" in params,
                r6_boundary_linked=False,
                evidence_label="parametric_boundary",
            ),
            range_row(
                args=args,
                range_id="r8_phy_laser_power_mw",
                parameter="phy_laser_power_mw",
                parameter_group="photonic_boundary",
                unit="mW",
                nominal_value=median(p_laser_pass) if p_laser_pass else math.nan,
                optimistic_value=min(p_laser_all, default=math.nan),
                pessimistic_value=max(p_laser_pass, default=math.nan),
                boundary_value=max(p_laser_all, default=math.nan),
                worse_direction="higher",
                nominal_label="median passing PHY laser-power row",
                pessimistic_label="largest passing PHY laser-power row below the limit",
                range_source="PHY boundary sweep laser-power pass/fail rows",
                source_artifact=repo_path(args.phy_boundary_csv),
                source_rows=len(phy_rows),
                architecture_parameter_linked=False,
                r6_boundary_linked=False,
                evidence_label="parametric_boundary",
            ),
            range_row(
                args=args,
                range_id="r8_laser_multiplier",
                parameter="laser_multiplier",
                parameter_group="photonic_sensitivity",
                unit="x",
                nominal_value=1.0,
                optimistic_value=0.85,
                pessimistic_value=nominal_param(params, "phy_pessimistic_laser_multiplier", 1.15),
                boundary_value=as_float(laser_boundary.get("max_last_beneficial_value"), 4.0),
                worse_direction="higher",
                nominal_label="selected architecture nominal laser scale",
                pessimistic_label="R6 named pessimistic laser scale",
                range_source="R6 named-regime and one-at-a-time laser sweep",
                source_artifact=repo_path(args.system_sensitivity_json),
                source_rows=len(r6_json.get("rows", [])),
                architecture_parameter_linked="phy_pessimistic_laser_multiplier" in params,
                r6_boundary_linked=True,
                evidence_label="system_sensitivity_boundary",
            ),
            range_row(
                args=args,
                range_id="r8_optical_link_loss_scale",
                parameter="optical_link_loss_scale",
                parameter_group="photonic_sensitivity",
                unit="x",
                nominal_value=1.0,
                optimistic_value=0.75,
                pessimistic_value=1.25,
                boundary_value=as_float(link_boundary.get("max_last_beneficial_value"), 4.0),
                worse_direction="higher",
                nominal_label="selected architecture nominal optical-link loss scale",
                pessimistic_label="R6 named pessimistic optical-link loss scale",
                range_source="R6 named-regime and one-at-a-time optical-link sweep",
                source_artifact=repo_path(args.system_sensitivity_json),
                source_rows=len(r6_json.get("rows", [])),
                architecture_parameter_linked=False,
                r6_boundary_linked=True,
                evidence_label="system_sensitivity_boundary",
            ),
            range_row(
                args=args,
                range_id="r8_dac_mzm_energy_scale",
                parameter="dac_mzm_energy_scale",
                parameter_group="photonic_conversion",
                unit="x",
                nominal_value=1.0,
                optimistic_value=0.85,
                pessimistic_value=1.25,
                boundary_value=as_float(dac_boundary.get("first_not_beneficial_value"), 64.0),
                worse_direction="higher",
                nominal_label="selected architecture nominal DAC/MZM scale",
                pessimistic_label="R6 named pessimistic DAC/MZM scale",
                range_source="R6 named-regime and one-at-a-time DAC/MZM sweep",
                source_artifact=repo_path(args.system_sensitivity_json),
                source_rows=len(r6_json.get("rows", [])),
                architecture_parameter_linked=False,
                r6_boundary_linked=True,
                evidence_label="system_sensitivity_boundary",
            ),
        ]
    )
    return rows


def build_summary(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    adc_json: dict[str, Any],
    phy_json: dict[str, Any],
    r6_json: dict[str, Any],
    rtl_json: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if len([row for row in rows if row["parameter_group"].startswith("adc_")]) < 6:
        blockers.append("adc_energy_latency_range_rows_missing")
    if len([row for row in rows if row["parameter_group"].startswith("photonic")]) < 7:
        blockers.append("photonic_range_rows_missing")
    if any(row["acceptance_status"] != "pass" for row in rows):
        blockers.append("range_row_missing_nominal_or_pessimistic_value")
    if not any(row["architecture_parameter_linked"] for row in rows):
        blockers.append("architecture_parameter_linkage_missing")
    if not any(row["r6_boundary_linked"] for row in rows):
        blockers.append("r6_boundary_linkage_missing")
    if adc_json.get("metadata", {}).get("execution_status") != "measured":
        blockers.append("adc_macro_sanity_not_measured")
    if phy_json.get("metadata", {}).get("pass_rows", 0) <= 0 or phy_json.get("metadata", {}).get("fail_rows", 0) <= 0:
        blockers.append("phy_boundary_lacks_pass_fail_rows")
    if r6_json.get("summary", {}).get("decision", {}).get("r6_acceptance_state") != "pass":
        blockers.append("r6_acceptance_state_not_pass")
    if rtl_json.get("acceptance", {}).get("status") != "pass":
        blockers.append("r7_rtl_control_acceptance_not_pass")

    device_solver_required = False
    acceptance_state = "pass" if not blockers else "fail"
    stop_condition_state = (
        "R8 hard stop: calibration ranges require device-solver or foundry data"
        if device_solver_required
        else "no R8 hard stop; ranges remain calibration and boundary evidence only"
    )
    group_counts: dict[str, int] = {}
    for row in rows:
        group_counts[str(row["parameter_group"])] = group_counts.get(str(row["parameter_group"]), 0) + 1

    selected = [
        {
            "parameter": row["parameter"],
            "unit": row["unit"],
            "nominal_value": row["nominal_value"],
            "pessimistic_value": row["pessimistic_value"],
            "boundary_value": row["boundary_value"],
            "evidence_label": row["evidence_label"],
        }
        for row in rows
    ]
    return {
        "tag": args.tag,
        "date": DATE,
        "roadmap_item": ROADMAP_ITEM,
        "rows": len(rows),
        "group_counts": group_counts,
        "selected_ranges": selected,
        "adc_macro_execution_status": adc_json.get("metadata", {}).get("execution_status", "missing"),
        "phy_pass_rows": phy_json.get("metadata", {}).get("pass_rows", 0),
        "phy_fail_rows": phy_json.get("metadata", {}).get("fail_rows", 0),
        "input_r6_acceptance_state": r6_json.get("summary", {}).get("decision", {}).get("r6_acceptance_state", "missing"),
        "input_r7_acceptance_state": rtl_json.get("acceptance", {}).get("status", "missing"),
        "decision": {
            "r8_acceptance_state": acceptance_state,
            "stop_condition_state": stop_condition_state,
            "blockers": sorted(set(blockers)),
            "device_solver_required": device_solver_required,
            "architecture_parameters_have_nominal_and_pessimistic_values": all(
                row["acceptance_status"] == "pass" for row in rows
            ),
            "claim_boundary_calibration_only": True,
        },
        "artifacts": {
            "csv": repo_path(args.csv_out),
            "json": repo_path(args.json_out),
            "report": repo_path(args.report_out),
            "architecture_parameters_csv": repo_path(args.parameters_csv),
            "adc_macro_csv": repo_path(args.adc_macro_csv),
            "adc_macro_json": repo_path(args.adc_macro_json),
            "phy_boundary_csv": repo_path(args.phy_boundary_csv),
            "phy_boundary_json": repo_path(args.phy_boundary_json),
            "system_sensitivity_json": repo_path(args.system_sensitivity_json),
            "rtl_control_json": repo_path(args.rtl_control_json),
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
            "artifact_id": f"suds_tetc_calibration_ranges_{args.tag}",
            "roadmap_item": ROADMAP_ITEM,
            "evidence_label": EVIDENCE_LABEL,
            "promotion_decision": (
                "r8_calibration_ranges_pass"
                if summary["decision"]["r8_acceptance_state"] == "pass"
                else "r8_calibration_ranges_fail"
            ),
            "regeneration_command": "make suds-tetc-calibration-ranges",
            "claim_boundary_note": CLAIM_BOUNDARY,
        },
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    lines = [
        "# SUDS TETC Calibration Ranges",
        "",
        f"Tag: `{args.tag}`",
        f"Roadmap item: `{ROADMAP_ITEM}`",
        f"Evidence label: `{EVIDENCE_LABEL}`",
        f"Acceptance state: `{decision['r8_acceptance_state']}`",
        f"Stop-condition state: `{decision['stop_condition_state']}`",
        "",
        "## Scope",
        "",
        "R8 turns single-point ADC and photonic assumptions into explicit",
        "nominal, optimistic, pessimistic, and boundary ranges. The artifact",
        "uses the local ADC macro sanity suite, the PHY pass/fail boundary sweep,",
        "the R6 system-sensitivity audit, and the R7 RTL-control status as inputs.",
        "It does not claim device-solver, foundry, extracted-layout, silicon,",
        "bench-energy, Lumerical, Spectre, or P&R closure.",
        "",
        "## Decision",
        "",
        f"- R8 acceptance: `{decision['r8_acceptance_state']}`",
        f"- Blockers: `{';'.join(decision['blockers']) or 'none'}`",
        f"- Device-solver required: `{decision['device_solver_required']}`",
        f"- Calibration-only boundary retained: `{decision['claim_boundary_calibration_only']}`",
        "",
        "## Range Table",
        "",
        "| Parameter | Group | Nominal | Pessimistic | Boundary | Unit | Evidence |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{parameter}` | `{group}` | {nominal} | {pessimistic} | {boundary} | `{unit}` | `{evidence}` |".format(
                parameter=row["parameter"],
                group=row["parameter_group"],
                nominal=fmt(row["nominal_value"]),
                pessimistic=fmt(row["pessimistic_value"]),
                boundary=fmt(row["boundary_value"]),
                unit=row["unit"],
                evidence=row["evidence_label"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ADC tier energy and latency now have measured corner ranges from the J1 macro suite.",
            "- Modulator/detector noise is represented by ER and crosstalk axes from the PHY boundary sweep.",
            "- Laser, optical-link, and DAC/MZM ranges are tied to the R6 named regimes and boundary sweeps.",
            "- The architecture paper may cite these rows as calibration ranges and boundary evidence only.",
            "",
            "## Artifacts",
            "",
            f"- Calibration CSV: `{repo_path(args.csv_out)}`",
            f"- Calibration JSON: `{repo_path(args.json_out)}`",
            f"- Report: `{repo_path(args.report_out)}`",
            f"- Architecture parameters: `{repo_path(args.parameters_csv)}`",
            f"- ADC macro input: `{repo_path(args.adc_macro_csv)}`",
            f"- PHY boundary input: `{repo_path(args.phy_boundary_csv)}`",
            f"- R6 sensitivity input: `{repo_path(args.system_sensitivity_json)}`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "make suds-tetc-calibration-ranges",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = parameter_index(load_csv(args.parameters_csv))
    adc_csv_rows = load_csv(args.adc_macro_csv)
    phy_csv_rows = load_csv(args.phy_boundary_csv)
    adc_json = load_json(args.adc_macro_json)
    phy_json = load_json(args.phy_boundary_json)
    r6_json = load_json(args.system_sensitivity_json)
    rtl_json = load_json(args.rtl_control_json)
    rows = adc_rows(args, params, adc_csv_rows)
    rows.extend(photonic_rows(args, params, phy_csv_rows, r6_json))
    summary = build_summary(
        args=args,
        rows=rows,
        adc_json=adc_json,
        phy_json=phy_json,
        r6_json=r6_json,
        rtl_json=rtl_json,
    )
    return rows, summary


def main() -> None:
    args = parse_args()
    rows, summary = build_payload(args)
    write_csv(args.csv_out, rows)
    write_json(args.json_out, args=args, rows=rows, summary=summary)
    write_report(args.report_out, args=args, rows=rows, summary=summary)
    print(f"wrote {repo_path(args.csv_out)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")
    print(f"r8_acceptance_state={summary['decision']['r8_acceptance_state']}")
    print(f"stop_condition={summary['decision']['stop_condition_state']}")


if __name__ == "__main__":
    main()
