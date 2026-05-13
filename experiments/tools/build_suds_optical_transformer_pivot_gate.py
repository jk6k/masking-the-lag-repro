#!/usr/bin/env python3
"""Build the optical Transformer / TETC pivot readiness gate."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
CSV_OUT = REPORT_DATA / f"suds_optical_transformer_pivot_gate_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_optical_transformer_pivot_gate_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_optical_transformer_tetc_pivot_gate.md"


ARTIFACTS = {
    "pivot_plan": REPO_ROOT / "docs/coordination/active/SUDS_OPTICAL_TRANSFORMER_TETC_PIVOT_PLAN.md",
    "reframe": REPO_ROOT / "paper/suds_tetc_architecture_reframe.md",
    "tetc_manuscript": REPO_ROOT / "paper/suds_tetc_architecture_manuscript.tex",
    "fallback_manuscript": REPO_ROOT / "paper/suds_paper_acmart.tex",
    "architecture": REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}.json",
    "architecture_summary": REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_summary.csv",
    "architecture_params": REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_parameters.csv",
    "architecture_design_space": REPORT_DATA / f"suds_transformer_architecture_design_space_{TAG}.json",
    "architecture_design_space_csv": REPORT_DATA / f"suds_transformer_architecture_design_space_{TAG}.csv",
    "glue_arch_link": REPORT_DATA / f"suds_glue_architecture_linkage_{TAG}.csv",
    "g1_release": REPORT_DATA / f"suds_tetc_g1_release_artifacts_{TAG}.json",
    "red_team": REPORT_DATA / f"suds_tetc_internal_red_team_{TAG}.json",
    "public_repro_alignment": REPORT_DATA / f"suds_tetc_public_repro_alignment_{TAG}.json",
    "mobilevit": REPORT_DATA / "suds_mobilevit_multimodel_validation_20260511_p2p3_quality.json",
    "glue": REPORT_DATA / "suds_glue_measured_validation_20260511_p2p3_quality.json",
    "hyatten": REPORT_DATA / "suds_hyatten_composition_ablation_20260511_p2p3_quality.json",
    "adc_spice": REPORT_DATA / "suds_adc_macro_sanity_20260512_j1_quality_boost.json",
    "rtl": REPORT_DATA / "suds_rtl_control_overhead_20260512_j2_quality_boost.json",
    "phy": REPORT_DATA / "suds_phy_circuit_boundary_20260511_p2p3_quality.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def status_rank(status: str) -> int:
    return {"pass": 0, "partial": 1, "fail": 2}.get(status, 3)


def gate_row(
    gate_id: str,
    gate: str,
    status: str,
    evidence: str,
    blockers: list[str] | None = None,
    next_action: str = "",
    required: bool = True,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "gate": gate,
        "required": required,
        "status": status,
        "evidence": evidence,
        "blockers": ";".join(blockers or []),
        "next_action": next_action,
    }


def inspect_mobilevit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_json(ARTIFACTS["mobilevit"])
    meta = data.get("metadata", {})
    rows = data.get("rows", [])
    aggregate_rows = [row for row in rows if row.get("row_type") == "aggregate"]
    if not aggregate_rows:
        aggregate_rows = list(data.get("aggregates", []))
    model_condition = {
        (str(row.get("model", "")), str(row.get("condition", "")))
        for row in rows
        if row.get("status") == "measured"
    }
    mobilevit_s = [
        row for row in aggregate_rows
        if str(row.get("model", "")) == "mobilevit_s"
        and str(row.get("condition", "")).lower() in {"e6_signal", "e7_overlay", "e8_overflow"}
    ]
    max_abs_drop = max((abs(number(row.get("delta_top1"))) for row in mobilevit_s), default=0.0)
    min_adc_ratio = min((number(row.get("adc_energy_ratio_vs_e0"), 1.0) for row in mobilevit_s), default=1.0)
    summary = {
        "present": bool(data),
        "completion_ratio": number(meta.get("completion_ratio")),
        "measured_rows": int(number(meta.get("measured_rows"))),
        "expected_rows": int(number(meta.get("expected_rows"))),
        "models": unique([str(row.get("model", "")) for row in rows]),
        "conditions": unique([condition for _, condition in model_condition]),
        "mobilevit_s_max_abs_top1_drop_pp": round(max_abs_drop, 3),
        "mobilevit_s_min_adc_ratio": round(min_adc_ratio, 3),
    }
    return summary, rows


def inspect_glue() -> dict[str, Any]:
    data = load_json(ARTIFACTS["glue"])
    meta = data.get("metadata", {})
    per_seed = data.get("per_seed", [])
    aggregates = data.get("aggregates", [])
    tasks = meta.get("tasks") or unique([str(row.get("task", "")) for row in per_seed + aggregates])
    conditions = unique([str(row.get("condition", "")) for row in per_seed + aggregates])
    devices = unique([str(row.get("device", "")) for row in per_seed + aggregates])
    max_abs_delta = max((abs(number(row.get("delta_primary_metric"))) for row in aggregates), default=0.0)
    analytical_slack = any("analytical" in str(row.get("slack_source", "")) for row in per_seed)
    return {
        "present": bool(data),
        "tasks": tasks,
        "conditions": conditions,
        "devices": devices,
        "max_abs_delta_primary_pp": round(max_abs_delta, 3),
        "analytical_slack": analytical_slack,
        "promotion_decision": meta.get("promotion_decision", ""),
    }


def inspect_hyatten() -> dict[str, Any]:
    data = load_json(ARTIFACTS["hyatten"])
    rows = data.get("rows", [])
    conditions = unique([str(row.get("condition", "")) for row in rows])
    models = unique([str(row.get("model", "")) for row in rows])
    return {
        "present": bool(data),
        "models": models,
        "conditions": conditions,
    }


def inspect_architecture() -> dict[str, Any]:
    data = load_json(ARTIFACTS["architecture"])
    design_space = load_json(ARTIFACTS["architecture_design_space"])
    decision = data.get("decision", {})
    design_summary = data.get("design_space_summary", {})
    summary_rows = data.get("summary_rows", [])
    parameter_rows = data.get("parameter_rows", [])
    glue_link_rows = data.get("glue_link_rows", [])
    nominal_rows = [row for row in summary_rows if row.get("sensitivity_case") == "nominal"]
    workloads = unique([str(row.get("workload", "")) for row in nominal_rows])
    conditions = unique([str(row.get("condition", "")) for row in nominal_rows])
    system_terms = decision.get("system_cost_terms") or []
    advantage = decision.get("pessimistic_advantage_by_workload") or {}
    min_pessimistic_edp_gain = min(
        (
            number(row.get("pessimistic_edp_improvement_pct"), -999.0)
            for row in advantage.values()
        ),
        default=-999.0,
    )
    min_pessimistic_energy_gain = min(
        (
            number(row.get("pessimistic_energy_improvement_pct"), -999.0)
            for row in advantage.values()
        ),
        default=-999.0,
    )
    calibration_parameters = {str(row.get("parameter", "")) for row in parameter_rows}
    linked_glue_rows = [
        row for row in glue_link_rows
        if str(row.get("profile_link_status", "")) == "pass"
    ]
    return {
        "present": bool(data),
        "status": decision.get("architecture_sim_status", "missing"),
        "blockers": decision.get("blockers", ["architecture_artifact_missing"] if not data else []),
        "workloads": workloads,
        "conditions": conditions,
        "system_terms": system_terms,
        "min_pessimistic_edp_gain_pct": round(min_pessimistic_edp_gain, 3),
        "min_pessimistic_energy_gain_pct": round(min_pessimistic_energy_gain, 3),
        "calibration_parameters": sorted(calibration_parameters),
        "glue_link_rows": len(linked_glue_rows),
        "glue_link_conditions": unique([str(row.get("architecture_condition", "")) for row in linked_glue_rows]),
        "artifact": repo_path(ARTIFACTS["architecture"]),
        "summary_csv": repo_path(ARTIFACTS["architecture_summary"]),
        "parameters_csv": repo_path(ARTIFACTS["architecture_params"]),
        "design_space_rows": int(number(design_summary.get("design_space_rows") or design_space.get("summary", {}).get("rows"))),
        "design_space_pareto_rows": int(number(design_summary.get("pareto_rows") or design_space.get("summary", {}).get("pareto_rows"))),
        "design_space_json": repo_path(ARTIFACTS["architecture_design_space"]),
        "design_space_csv": repo_path(ARTIFACTS["architecture_design_space_csv"]),
        "glue_link_csv": repo_path(ARTIFACTS["glue_arch_link"]),
    }


def inspect_g1_release() -> dict[str, Any]:
    data = load_json(ARTIFACTS["g1_release"])
    red_team = load_json(ARTIFACTS["red_team"])
    public_repro = load_json(ARTIFACTS["public_repro_alignment"])
    summary = data.get("summary", {})
    manuscript = summary.get("manuscript", {})
    red_summary = summary.get("red_team", red_team.get("summary", {}))
    public_summary = summary.get("public_repro", public_repro.get("summary", {}))
    return {
        "present": bool(data),
        "status": summary.get("status", "missing"),
        "manuscript_status": manuscript.get("status", "missing"),
        "manuscript_line_count": int(number(manuscript.get("line_count"))),
        "red_team_status": red_summary.get("status", "missing"),
        "red_team_lens_count": int(number(red_summary.get("lens_count"))),
        "external_red_team_status": red_team.get("metadata", {}).get("external_red_team_status", ""),
        "external_equivalence": red_team.get("metadata", {}).get("external_equivalence", ""),
        "public_repro_status": public_summary.get("status", "missing"),
        "public_repro_manifest_status": public_summary.get("manifest_status", "missing"),
        "public_repro_validation_status": public_summary.get("validation_status", "missing"),
        "public_repro_validation_errors": int(number(public_summary.get("validation_error_count"))),
        "artifact": repo_path(ARTIFACTS["g1_release"]),
        "red_team_artifact": repo_path(ARTIFACTS["red_team"]),
        "public_repro_artifact": repo_path(ARTIFACTS["public_repro_alignment"]),
    }


def inspect_circuit() -> dict[str, Any]:
    adc = load_json(ARTIFACTS["adc_spice"])
    rtl = load_json(ARTIFACTS["rtl"])
    phy = load_json(ARTIFACTS["phy"])
    adc_rows = adc.get("rows", [])
    rtl_rows = rtl.get("rows", [])
    phy_rows = phy.get("rows", [])
    adc_bits = unique([str(row.get("adc_bits", "")) for row in adc_rows])
    rtl_pass = rtl.get("metadata", {}).get("yosys", {}).get("status") == "pass"
    phy_pass = number(phy.get("metadata", {}).get("pass_rows"))
    phy_fail = number(phy.get("metadata", {}).get("fail_rows"))
    phy_total = phy_pass + phy_fail
    return {
        "adc_present": bool(adc),
        "adc_status": adc.get("metadata", {}).get("execution_status", ""),
        "adc_bits": adc_bits,
        "rtl_present": bool(rtl),
        "rtl_yosys_pass": rtl_pass,
        "rtl_cells": rtl.get("metadata", {}).get("yosys", {}).get("stats", {}).get("cells", ""),
        "phy_present": bool(phy),
        "phy_pass_ratio": round(phy_pass / phy_total, 3) if phy_total else 0.0,
    }


def inspect_manuscript() -> dict[str, Any]:
    text = read_text(ARTIFACTS["fallback_manuscript"])
    tetc_text = read_text(ARTIFACTS["tetc_manuscript"])
    title_match = re.search(r"\\title\{([^}]*)\}", text)
    tetc_title_match = re.search(r"\\title\{([^}]*)\}", tetc_text)
    return {
        "present": bool(text),
        "title": title_match.group(1) if title_match else "",
        "still_jetc_comment": "JETC" in text[:500],
        "methodology_title": "The Missing Interface" in text[:2000],
        "mentions_tetc": "TETC" in text[:5000],
        "tetc_present": bool(tetc_text),
        "tetc_title": tetc_title_match.group(1) if tetc_title_match else "",
        "tetc_architecture_first": "Dynamic Photonic Transformer" in tetc_text[:5000]
        or "DPTC" in tetc_text[:5000],
        "tetc_forbidden_terms": unique(
            [
                term
                for term in forbidden_claim_terms()
                if term.lower() in tetc_text.lower()
            ]
        ),
    }


def forbidden_claim_terms() -> list[str]:
    return [
        "silicon-" + "validated",
        "SPICE " + "closure",
        "post-" + "layout",
        "measured hardware " + "energy",
        "deployment " + "readiness",
    ]


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mobilevit, mobilevit_rows = inspect_mobilevit()
    glue = inspect_glue()
    hyatten = inspect_hyatten()
    architecture = inspect_architecture()
    g1_release = inspect_g1_release()
    circuit = inspect_circuit()
    manuscript = inspect_manuscript()
    plan_exists = ARTIFACTS["pivot_plan"].is_file()
    reframe_exists = ARTIFACTS["reframe"].is_file()
    required_arch_conditions = {
        "uniform_8bit",
        "uniform_4bit",
        "random",
        "l1",
        "slack_only",
        "suds_only",
        "suds_l1",
        "suds_signal",
        "hyatten_style",
        "lightening_dptc",
        "tempo_time_multiplexed",
        "astra_boundary",
    }
    architecture_conditions_ok = required_arch_conditions.issubset(set(architecture["conditions"]))
    architecture_workloads_ok = {"bert_base_glue_seq128", "mobilevit_s_transformer_blocks_256"}.issubset(
        set(architecture["workloads"])
    )
    architecture_design_space_ok = architecture["design_space_rows"] > 0 and architecture["design_space_pareto_rows"] > 0
    glue_schedule_linked = architecture["glue_link_rows"] >= 108 and architecture_workloads_ok
    parameter_ties_ok = {"adc8_pj", "control_pj_per_sideband_group", "phy_nominal_pass_ratio"}.issubset(
        set(architecture["calibration_parameters"])
    )
    tetc_manuscript_ready = (
        manuscript["tetc_present"]
        and manuscript["tetc_architecture_first"]
        and not manuscript["tetc_forbidden_terms"]
    )
    g1_ready = tetc_manuscript_ready and g1_release["status"] == "pass"

    rows = [
        gate_row(
            "G0",
            "Route lock and protected fallback",
            "pass" if plan_exists and reframe_exists else "fail",
            f"plan={repo_path(ARTIFACTS['pivot_plan'])}; reframe={repo_path(ARTIFACTS['reframe'])}",
            [] if plan_exists and reframe_exists else ["pivot_plan_or_reframe_missing"],
            "Create the pivot plan and reframe blueprint.",
        ),
        gate_row(
            "G1",
            "Full TETC manuscript source",
            "pass" if g1_ready else ("partial" if tetc_manuscript_ready else "fail"),
            (
                f"fallback_title={manuscript['title']}; "
                f"tetc_source={repo_path(ARTIFACTS['tetc_manuscript']) if manuscript['tetc_present'] else 'missing'}; "
                f"tetc_title={manuscript['tetc_title']}; "
                f"g1_release={g1_release['status']}; "
                f"red_team={g1_release['red_team_status']}; "
                f"public_repro={g1_release['public_repro_status']}/{g1_release['public_repro_validation_status']}"
            ),
            (
                []
                if g1_ready
                else (
                    ["g1_release_artifacts_not_promoted"]
                    if tetc_manuscript_ready
                    else ["protected_fallback_manuscript_is_still_methodology_route", "new_tetc_source_missing_or_scaffold_only"]
                )
            )
            + ([f"forbidden_terms={','.join(manuscript['tetc_forbidden_terms'])}"] if manuscript["tetc_forbidden_terms"] else []),
            "Keep the architecture-first TETC source promoted only while manuscript, red-team, and public-repro alignment artifacts pass.",
        ),
        gate_row(
            "G2",
            "Two Transformer workloads on governed MPS",
            (
                "pass"
                if mobilevit["completion_ratio"] >= 1.0
                and len(glue["tasks"]) >= 6
                and "mps" in glue["devices"]
                and glue_schedule_linked
                else "partial"
            ),
            (
                f"MobileViT rows={mobilevit['measured_rows']}/{mobilevit['expected_rows']}; "
                f"GLUE tasks={','.join(glue['tasks'])}; devices={','.join(glue['devices'])}; "
                f"BERT architecture link rows={architecture['glue_link_rows']}"
            ),
            [] if glue_schedule_linked else ["bert_glue_rows_not_linked_to_dptc_schedule"],
            "Keep GLUE as measured MPS accuracy and architecture-modeled energy; rerun only if perturbation policy changes.",
        ),
        gate_row(
            "G3",
            "System-level PPA advantage",
            "pass" if architecture["status"] == "pass" and architecture_design_space_ok else "partial",
            (
                f"artifact={architecture['artifact']}; workloads={','.join(architecture['workloads'])}; "
                f"terms={','.join(architecture['system_terms'])}; "
                f"min_pessimistic_edp_gain={architecture['min_pessimistic_edp_gain_pct']}%; "
                f"design_space_rows={architecture['design_space_rows']}; "
                f"pareto_rows={architecture['design_space_pareto_rows']}"
            ),
            list(architecture["blockers"]) + ([] if architecture_design_space_ok else ["architecture_design_space_missing"]),
            "Use the architecture simulator summary table as the promoted system-level PPA surface.",
        ),
        gate_row(
            "G4",
            "Strong baselines and ablations",
            "pass" if architecture_conditions_ok and {"e0_dense", "e2_l1", "e3_slack", "e4_suds", "e5_random", "e6_signal", "e7_overlay", "e8_overflow"}.issubset(set(mobilevit["conditions"])) else "partial",
            (
                f"MobileViT conditions={','.join(mobilevit['conditions'])}; "
                f"architecture conditions={','.join(architecture['conditions'])}; "
                f"HyAtten artifact conditions={','.join(hyatten['conditions'])}"
            ),
            [] if architecture_conditions_ok else ["matched_lightening_hyatten_tempo_or_astra_architecture_baseline_missing"],
            "Keep signal-only wins as composition-boundary evidence, not slack-only superiority.",
        ),
        gate_row(
            "G5",
            "SPICE/RTL/PHY calibration tied to architecture parameters",
            "pass" if circuit["adc_status"] == "measured" and circuit["rtl_yosys_pass"] and circuit["phy_present"] and parameter_ties_ok else "partial",
            (
                f"ADC bits={','.join(circuit['adc_bits'])}; RTL cells={circuit['rtl_cells']}; "
                f"PHY pass ratio={circuit['phy_pass_ratio']}; architecture params={','.join(architecture['calibration_parameters'])}"
            ),
            [] if parameter_ties_ok else ["calibration_exists_but_not_yet_tied_to_new_architecture_parameter_table"],
            "Preserve ADC/RTL/PHY as calibration and boundary evidence only.",
        ),
        gate_row(
            "G6",
            "Target journal fit",
            "pass",
            "Primary route is IEEE TETC; TC is stretch; JSA/protected JETC-methodology package is fallback.",
            [],
            "Recheck CAS/JCR partition in institutional database before final submission.",
            required=False,
        ),
    ]

    decision = decide(rows)
    summary = {
        "mobilevit": mobilevit,
        "glue": glue,
        "hyatten": hyatten,
        "architecture": architecture,
        "g1_release": g1_release,
        "circuit": circuit,
        "manuscript": manuscript,
        "decision": decision,
    }
    return rows, summary


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_rows = [row for row in rows if row["required"]]
    failed = [row for row in required_rows if row["status"] == "fail"]
    partial = [row for row in required_rows if row["status"] == "partial"]
    if not failed and not partial:
        promotion = "tetc_submission_ready"
    elif rows[0]["status"] == "pass":
        promotion = "pivot_scaffolded_not_submission_ready"
    else:
        promotion = "pivot_not_ready"
    return {
        "promotion_decision": promotion,
        "required_failures": [row["gate_id"] for row in failed],
        "required_partials": [row["gate_id"] for row in partial],
        "highest_priority_next_step": highest_priority_next_step(failed, partial, promotion),
    }


def highest_priority_next_step(
    failed: list[dict[str, Any]],
    partial: list[dict[str, Any]],
    promotion: str,
) -> str:
    by_id = {row["gate_id"]: row for row in failed + partial}
    if "G3" in by_id:
        return "Build or repair the Transformer DPTC architecture simulator and matched Lightening/HyAtten baselines."
    if "G2" in by_id:
        return "Complete governed MPS workload linkage to hardware-derived photonic schedules."
    if "G4" in by_id:
        return "Complete matched baseline and ablation coverage under the architecture simulator."
    if "G5" in by_id:
        return "Tie ADC, RTL, and PHY calibration rows into the architecture parameter table."
    if "G1" in by_id:
        return "Finish G1 manuscript integration, red-team artifact, and public-repro alignment without weakening the protected fallback."
    if promotion == "tetc_submission_ready":
        return "Run make suds-tetc-science-gate before treating the route as a local submission candidate; external red-team remains advisory."
    return "Complete missing route-lock artifacts."


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, tag: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_optical_transformer_pivot_gate_{tag}",
            "evidence_label": "audit",
            "promotion_decision": summary["decision"]["promotion_decision"],
            "regeneration_command": "make suds-optical-transformer-pivot-gate",
        },
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, tag: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    report = f"""# SUDS Optical Transformer TETC Pivot Gate

Tag: `{tag}`
Evidence label: `audit`
Promotion decision: `{decision['promotion_decision']}`

## Decision

- Primary route: `IEEE TETC architecture-first optical Transformer accelerator`
- Stretch route: `IEEE Transactions on Computers`
- Fallback route: `JSA or protected methodology package`
- Required failures: `{','.join(decision['required_failures']) or 'none'}`
- Required partials: `{','.join(decision['required_partials']) or 'none'}`
- Highest-priority next step: {decision['highest_priority_next_step']}

## Gate Table

| Gate | Required | Status | Evidence | Blockers | Next action |
|---|---:|---|---|---|---|
"""
    for row in sorted(rows, key=lambda item: (status_rank(item["status"]), item["gate_id"])):
        report += (
            f"| {row['gate_id']} {row['gate']} | `{row['required']}` | `{row['status']}` | "
            f"{row['evidence']} | `{row['blockers']}` | {row['next_action']} |\n"
        )

    mobilevit = summary["mobilevit"]
    glue = summary["glue"]
    architecture = summary["architecture"]
    g1_release = summary["g1_release"]
    circuit = summary["circuit"]
    report += f"""
## Evidence Snapshot

- G1 release artifact: status `{g1_release['status']}`, manuscript `{g1_release['manuscript_status']}` with `{g1_release['manuscript_line_count']}` lines, red-team `{g1_release['red_team_status']}` across `{g1_release['red_team_lens_count']}` lenses, external status `{g1_release['external_red_team_status']}`, public-repro `{g1_release['public_repro_status']}` with validation `{g1_release['public_repro_validation_status']}`.
- MobileViT measured matrix: `{mobilevit['measured_rows']}/{mobilevit['expected_rows']}` rows, models `{','.join(mobilevit['models'])}`, conditions `{','.join(mobilevit['conditions'])}`.
- MobileViT-S composition boundary: max absolute Top-1 drop among E6/E7/E8 is `{mobilevit['mobilevit_s_max_abs_top1_drop_pp']}` pp at minimum ADC ratio `{mobilevit['mobilevit_s_min_adc_ratio']}`.
- BERT/GLUE measured validation: tasks `{','.join(glue['tasks'])}`, devices `{','.join(glue['devices'])}`, max aggregate delta `{glue['max_abs_delta_primary_pp']}` pp; slack source analytical = `{glue['analytical_slack']}`.
- Architecture simulator: status `{architecture['status']}`, workloads `{','.join(architecture['workloads'])}`, conditions `{','.join(architecture['conditions'])}`, minimum pessimistic EDP gain `{architecture['min_pessimistic_edp_gain_pct']}`%.
- Architecture design space: `{architecture['design_space_rows']}` rows, `{architecture['design_space_pareto_rows']}` Pareto rows, artifact `{architecture['design_space_json']}`.
- BERT GLUE architecture linkage: `{architecture['glue_link_rows']}` linked rows, conditions `{','.join(architecture['glue_link_conditions'])}`.
- Circuit calibration: ADC macro status `{circuit['adc_status']}`, ADC tiers `{','.join(circuit['adc_bits'])}`, RTL Yosys pass `{circuit['rtl_yosys_pass']}`, PHY pass ratio `{circuit['phy_pass_ratio']}`.

## Interpretation

The pivot route now has an architecture-level Transformer/DPTC simulator when
G3 is passing. This pivot gate is an artifact-pack gate: it checks that the
local architecture, manuscript, red-team substitute, public-repro alignment,
and calibration artifacts exist and agree on claim boundaries. It is still
not the final science-strength gate. Final local submission-candidate wording
must additionally pass `make suds-tetc-science-gate`. The simulator is modeled
system PPA, not bench-energy or circuit signoff. The protected methodology
manuscript remains available as fallback provenance, but it is no longer the
active route for this artifact gate.

## Regeneration

```bash
make suds-optical-transformer-pivot-gate
```
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, summary = build_rows()
    write_csv(args.csv_out, rows)
    write_json(args.json_out, args.tag, rows, summary)
    write_report(args.report_out, args.tag, rows, summary)
    print(f"wrote {repo_path(args.csv_out)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")
    print(f"promotion_decision={summary['decision']['promotion_decision']}")


if __name__ == "__main__":
    main()
