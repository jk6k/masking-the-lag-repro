#!/usr/bin/env python3
"""Build the non-bypassable science-strength gate for the SUDS TETC route."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

CSV_OUT = REPORT_DATA / f"suds_tetc_science_gate_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_science_gate_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_science_gate.md"

PIVOT_GATE_JSON = REPORT_DATA / f"suds_optical_transformer_pivot_gate_{TAG}.json"
ARCH_SUMMARY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_summary.csv"
ARCH_JSON = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}.json"
MANUSCRIPT = REPO_ROOT / "paper/suds_tetc_architecture_manuscript.tex"
INTERNAL_RED_TEAM_JSON = REPORT_DATA / f"suds_tetc_internal_red_team_{TAG}.json"

MAIN_SUDS_CONDITIONS = {"suds_pareto"}
MEASURED_SAME_FABRIC_BASELINES = {
    "lightening_dptc",
    "uniform_8bit",
    "random",
    "l1",
    "slack_only",
    "signal_only",
}
MEASURED_ACCURACY_LABELS = {"measured_mps_glue", "measured_mps_imagenet"}
EXTERNAL_RED_TEAM_REQUIRED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument(
        "--fail-on-not-ready",
        action="store_true",
        help="Exit non-zero unless the science gate passes as a local submission candidate.",
    )
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


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


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


def measured_row(row: dict[str, str]) -> bool:
    return row.get("accuracy_evidence_label") in MEASURED_ACCURACY_LABELS and as_float(row.get("delta_accuracy")) is not None


def nominal_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("sensitivity_case") == "nominal"]


def best_by_edp(rows: list[dict[str, str]]) -> dict[str, str] | None:
    scored = [(as_float(row.get("edp_ratio_vs_lightening")), row) for row in rows]
    scored = [(score, row) for score, row in scored if score is not None]
    return min(scored, default=(None, None), key=lambda item: item[0])[1]


def dominance_findings(summary_rows: list[dict[str, str]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for workload in sorted({row.get("workload", "") for row in summary_rows}):
        workload_rows = [row for row in summary_rows if row.get("workload") == workload]
        suds_rows = [
            row for row in workload_rows
            if row.get("condition") in MAIN_SUDS_CONDITIONS and measured_row(row)
        ]
        baseline_rows = [
            row for row in workload_rows
            if row.get("condition") in MEASURED_SAME_FABRIC_BASELINES and measured_row(row)
        ]
        best_suds = best_by_edp(suds_rows)
        if not best_suds:
            findings.append(
                {
                    "workload": workload,
                    "status": "fail",
                    "best_suds": "",
                    "dominators": ["missing_measured_suds_main_row"],
                }
            )
            continue
        best_suds_edp_value = as_float(best_suds.get("edp_ratio_vs_lightening"))
        best_suds_acc_value = as_float(best_suds.get("delta_accuracy"))
        best_suds_edp = best_suds_edp_value if best_suds_edp_value is not None else 999.0
        best_suds_acc = best_suds_acc_value if best_suds_acc_value is not None else -999.0
        dominators: list[str] = []
        for baseline in baseline_rows:
            baseline_edp = as_float(baseline.get("edp_ratio_vs_lightening"))
            baseline_acc = as_float(baseline.get("delta_accuracy"))
            if baseline_edp is None or baseline_acc is None:
                continue
            edp_no_worse = baseline_edp <= best_suds_edp + 0.005
            accuracy_no_worse = baseline_acc >= best_suds_acc - 0.05
            strictly_better = baseline_edp < best_suds_edp - 0.005 or baseline_acc > best_suds_acc + 0.05
            if edp_no_worse and accuracy_no_worse and strictly_better:
                dominators.append(
                    f"{baseline.get('condition')} edp={baseline_edp:.3f} delta={baseline_acc:.3f}"
                )
        findings.append(
            {
                "workload": workload,
                "status": "pass" if not dominators else "fail",
                "best_suds": (
                    f"{best_suds.get('condition')} edp={best_suds_edp:.3f} "
                    f"delta={best_suds_acc:.3f}"
                ),
                "dominators": dominators,
            }
        )
    return {
        "status": "pass" if findings and all(item["status"] == "pass" for item in findings) else "fail",
        "findings": findings,
    }


def accuracy_budget(summary_rows: list[dict[str, str]]) -> dict[str, Any]:
    suds_rows = [
        row for row in summary_rows
        if row.get("condition") in MAIN_SUDS_CONDITIONS and measured_row(row)
    ]
    deltas = [as_float(row.get("delta_accuracy")) for row in suds_rows]
    deltas = [delta for delta in deltas if delta is not None]
    min_delta = min(deltas, default=None)
    if min_delta is None:
        return {"status": "fail", "min_delta_pp": None, "blocker": "missing_measured_suds_accuracy_delta"}
    if min_delta >= -1.0:
        status = "pass"
        blocker = ""
    elif min_delta >= -2.0:
        status = "partial"
        blocker = "promoted_suds_accuracy_loss_exceeds_1pp_target"
    else:
        status = "fail"
        blocker = "promoted_suds_accuracy_loss_exceeds_2pp_boundary"
    return {"status": status, "min_delta_pp": round(min_delta, 3), "blocker": blocker}


def manuscript_maturity() -> dict[str, Any]:
    text = MANUSCRIPT.read_text(encoding="utf-8", errors="replace") if MANUSCRIPT.is_file() else ""
    has_references = "\\bibliography" in text or "\\begin{thebibliography}" in text
    has_figures = "\\includegraphics" in text
    line_count = len(text.splitlines())
    blockers = []
    if line_count < 900:
        blockers.append(f"manuscript_short_for_full_tetc_article_lines={line_count}")
    if not has_references:
        blockers.append("references_not_integrated")
    if not has_figures:
        blockers.append("no_main_text_figures_integrated")
    return {
        "status": "pass" if not blockers else "partial",
        "line_count": line_count,
        "has_references": has_references,
        "has_figures": has_figures,
        "blockers": blockers,
    }


def schedule_metadata(architecture: dict[str, Any]) -> dict[str, Any]:
    kernel_rows = [
        row for row in architecture.get("kernel_rows", [])
        if row.get("workload") == "bert_base_glue_seq128"
    ]
    glue_link_rows = [
        row for row in architecture.get("glue_link_rows", [])
        if row.get("architecture_workload") == "bert_base_glue_seq128"
        and row.get("profile_link_status") == "pass"
    ]
    slack_sources = sorted({str(row.get("slack_source", "")) for row in kernel_rows if row.get("slack_source")})
    linked_sources = sorted({str(row.get("linked_schedule_source", "")) for row in glue_link_rows if row.get("linked_schedule_source")})
    linked_conditions = sorted({str(row.get("architecture_condition", "")) for row in glue_link_rows if row.get("architecture_condition")})
    schedule_fields_ok = bool(kernel_rows) and all(
        as_float(row.get("schedule_start_ns")) is not None
        and as_float(row.get("schedule_deadline_ns")) is not None
        and as_float(row.get("scheduler_slack_norm")) is not None
        for row in kernel_rows
    )
    hardware_schedule_ok = (
        bool(kernel_rows)
        and slack_sources == ["dptc_photonic_tile_schedule"]
        and schedule_fields_ok
        and len(glue_link_rows) >= 108
        and linked_sources == ["dptc_photonic_tile_schedule"]
        and "suds_pareto" in linked_conditions
    )
    return {
        "status": "pass" if hardware_schedule_ok else "partial",
        "bert_kernel_rows": len(kernel_rows),
        "bert_slack_sources": slack_sources,
        "schedule_fields_ok": schedule_fields_ok,
        "glue_link_rows": len(glue_link_rows),
        "glue_link_conditions": linked_conditions,
        "linked_schedule_sources": linked_sources,
        "blockers": []
        if hardware_schedule_ok
        else [
            blocker
            for blocker, active in {
                "bert_dptc_kernel_schedule_missing": not kernel_rows,
                "bert_schedule_fields_incomplete": not schedule_fields_ok,
                "bert_slack_source_not_dptc_schedule": slack_sources != ["dptc_photonic_tile_schedule"],
                "bert_glue_schedule_link_rows_below_108": len(glue_link_rows) < 108,
                "bert_glue_link_not_dptc_schedule": linked_sources != ["dptc_photonic_tile_schedule"],
                "suds_pareto_not_linked_to_bert_schedule": "suds_pareto" not in linked_conditions,
            }.items()
            if active
        ],
    }


def workload_grounding(pivot: dict[str, Any], architecture: dict[str, Any]) -> dict[str, Any]:
    glue = pivot.get("summary", {}).get("glue", {})
    mobilevit = pivot.get("summary", {}).get("mobilevit", {})
    pivot_architecture = pivot.get("summary", {}).get("architecture", {})
    schedule = schedule_metadata(architecture)
    blockers = []
    if "mps" not in set(glue.get("devices", [])):
        blockers.append("glue_not_measured_on_mps")
    if len(glue.get("tasks", [])) < 6:
        blockers.append("glue_task_coverage_below_six")
    if mobilevit.get("completion_ratio", 0) < 1.0:
        blockers.append("mobilevit_matrix_incomplete")
    if glue.get("analytical_slack") and schedule["status"] != "pass":
        blockers.append("bert_slack_source_still_analytical_in_pivot_summary")
    blockers.extend(schedule["blockers"])
    if max(pivot_architecture.get("glue_link_rows", 0), schedule["glue_link_rows"]) < 108:
        blockers.append("bert_architecture_linkage_incomplete")
    status = "pass" if not blockers else "partial"
    return {
        "status": status,
        "blockers": blockers,
        "glue": glue,
        "mobilevit": mobilevit,
        "architecture": pivot_architecture,
        "schedule_metadata": schedule,
        "original_glue_analytical_slack_retained_as_provenance": bool(glue.get("analytical_slack")),
    }


def calibration_boundary(pivot: dict[str, Any]) -> dict[str, Any]:
    circuit = pivot.get("summary", {}).get("circuit", {})
    architecture = pivot.get("summary", {}).get("architecture", {})
    params = set(architecture.get("calibration_parameters", []))
    blockers = []
    if circuit.get("adc_status") != "measured":
        blockers.append("adc_macro_not_measured_or_surrogate_missing")
    if not circuit.get("rtl_yosys_pass"):
        blockers.append("rtl_sideband_yosys_not_passing")
    if "phy_nominal_pass_ratio" not in params:
        blockers.append("phy_boundary_not_tied_to_parameter_table")
    if not {"adc4_pj", "adc6_pj", "adc8_pj", "control_pj_per_sideband_group"}.issubset(params):
        blockers.append("adc_or_control_parameters_missing_from_architecture_table")
    return {"status": "pass" if not blockers else "partial", "blockers": blockers}


def red_team_advisory() -> dict[str, Any]:
    red_team = load_json(INTERNAL_RED_TEAM_JSON)
    metadata = red_team.get("metadata", {})
    summary = red_team.get("summary", {})
    external_status = metadata.get("external_red_team_status", "missing")
    return {
        "status": "pass" if summary.get("status") == "pass" else "partial",
        "internal_status": summary.get("status", "missing"),
        "external_status": external_status,
        "external_required": EXTERNAL_RED_TEAM_REQUIRED,
        "external_equivalence": metadata.get("external_equivalence", ""),
    }


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_rows = [row for row in rows if row["required"]]
    failures = [row for row in required_rows if row["status"] == "fail"]
    partials = [row for row in required_rows if row["status"] == "partial"]
    if failures:
        decision = "science_gate_fail_not_submission_ready"
    elif partials:
        decision = "science_gate_partial_not_submission_ready"
    else:
        decision = "science_gate_pass_local_submission_candidate"
    return {
        "promotion_decision": decision,
        "required_failures": [row["gate_id"] for row in failures],
        "required_partials": [row["gate_id"] for row in partials],
        "external_red_team_required": EXTERNAL_RED_TEAM_REQUIRED,
    }


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pivot = load_json(PIVOT_GATE_JSON)
    architecture = load_json(ARCH_JSON)
    rows_csv = nominal_rows(load_csv(ARCH_SUMMARY_CSV))
    dominance = dominance_findings(rows_csv)
    accuracy = accuracy_budget(rows_csv)
    workload = workload_grounding(pivot, architecture)
    maturity = manuscript_maturity()
    calibration = calibration_boundary(pivot)
    red_team = red_team_advisory()
    pivot_decision = pivot.get("summary", {}).get("decision", {}).get("promotion_decision") or pivot.get("metadata", {}).get("promotion_decision", "")
    arch_status = architecture.get("decision", {}).get("architecture_sim_status", "missing")

    dominance_blockers = [
        f"{item['workload']} dominated_by {', '.join(item['dominators'])}"
        for item in dominance["findings"]
        if item["dominators"]
    ]

    rows = [
        gate_row(
            "S0",
            "Artifact gate is necessary but not sufficient",
            "pass" if pivot_decision == "tetc_submission_ready" and arch_status == "pass" else "partial",
            f"pivot_decision={pivot_decision}; architecture_status={arch_status}",
            [] if pivot_decision == "tetc_submission_ready" and arch_status == "pass" else ["artifact_gate_not_green"],
            "Keep the pivot gate as artifact-pack readiness only.",
        ),
        gate_row(
            "S1",
            "Promoted SUDS Pareto row is competitive against measured same-fabric baselines",
            dominance["status"],
            "; ".join(f"{item['workload']}: {item['best_suds']}" for item in dominance["findings"]),
            dominance_blockers,
            "Rework the policy, operating point, or claim so SUDS is not dominated by measured same-fabric selectors.",
        ),
        gate_row(
            "S2",
            "Promoted accuracy loss stays within a TETC-grade budget",
            accuracy["status"],
            f"worst_promoted_suds_delta_pp={accuracy['min_delta_pp']}",
            [accuracy["blocker"]] if accuracy["blocker"] else [],
            "Target <=1 pp loss for promoted rows or show a full accuracy/EDP Pareto instead of a single headline.",
        ),
        gate_row(
            "S3",
            "Transformer workload grounding is hardware-derived enough for the main claim",
            workload["status"],
            (
                f"glue_tasks={','.join(workload['glue'].get('tasks', []))}; "
                f"devices={','.join(workload['glue'].get('devices', []))}; "
                f"mobilevit_completion={workload['mobilevit'].get('completion_ratio')}; "
                f"pivot_glue_link_rows={workload['architecture'].get('glue_link_rows')}; "
                f"schedule_kernel_rows={workload['schedule_metadata'].get('bert_kernel_rows')}; "
                f"schedule_link_rows={workload['schedule_metadata'].get('glue_link_rows')}; "
                f"schedule_sources={','.join(workload['schedule_metadata'].get('bert_slack_sources', []))}; "
                f"original_glue_analytical_slack={workload['original_glue_analytical_slack_retained_as_provenance']}"
            ),
            workload["blockers"],
            "Replace analytical BERT slack with hardware-derived schedule metadata before calling the route submission-ready.",
        ),
        gate_row(
            "S4",
            "Manuscript maturity reaches full journal-paper density",
            maturity["status"],
            (
                f"line_count={maturity['line_count']}; references={maturity['has_references']}; "
                f"figures={maturity['has_figures']}"
            ),
            maturity["blockers"],
            "Expand the TETC manuscript into a complete IEEE journal article with figures and references.",
        ),
        gate_row(
            "S5",
            "Calibration boundaries remain traceable without circuit overclaim",
            calibration["status"],
            "ADC/RTL/PHY parameters are checked against the pivot architecture summary.",
            calibration["blockers"],
            "Keep SPICE/RTL/PHY evidence as calibration, proxy, or boundary evidence only.",
        ),
        gate_row(
            "S6",
            "External red-team advisory",
            red_team["status"],
            (
                f"internal={red_team['internal_status']}; external={red_team['external_status']}; "
                f"external_required={red_team['external_required']}; equivalence={red_team['external_equivalence']}"
            ),
            [] if red_team["status"] == "pass" else ["internal_red_team_missing_or_failed"],
            "External red-team remains useful but is not a hard local blocker per user instruction.",
            required=False,
        ),
    ]

    summary = {
        "pivot_decision": pivot_decision,
        "architecture_status": arch_status,
        "dominance": dominance,
        "accuracy": accuracy,
        "workload_grounding": workload,
        "manuscript_maturity": maturity,
        "calibration_boundary": calibration,
        "red_team_advisory": red_team,
    }
    summary["decision"] = decide(rows)
    return rows, summary


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
            "artifact_id": f"suds_tetc_science_gate_{tag}",
            "evidence_label": "science_strength_gate",
            "promotion_decision": summary["decision"]["promotion_decision"],
            "regeneration_command": "make suds-tetc-science-gate",
        },
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, tag: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    decision = summary["decision"]
    lines = [
        "# SUDS TETC Science-Strength Gate",
        "",
        f"Tag: `{tag}`",
        "Evidence label: `science_strength_gate`",
        f"Promotion decision: `{decision['promotion_decision']}`",
        "",
        "## Decision",
        "",
        f"- Pivot artifact gate decision: `{summary['pivot_decision']}`",
        f"- Required failures: `{','.join(decision['required_failures']) or 'none'}`",
        f"- Required partials: `{','.join(decision['required_partials']) or 'none'}`",
        f"- External red-team required: `{decision['external_red_team_required']}`",
        "",
        "This gate is intentionally stricter than the artifact pivot gate. It treats",
        "buildable reports, public-repro alignment, and internal red-team artifacts",
        "as necessary but not sufficient for TETC submission readiness.",
        "",
        "## Gate Table",
        "",
        "| Gate | Required | Status | Evidence | Blockers | Next action |",
        "|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {gate_id} {gate} | `{required}` | `{status}` | {evidence} | `{blockers}` | {next_action} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Same-Fabric Dominance Findings",
            "",
            "| Workload | Promoted SUDS Pareto row | Dominating measured baseline rows |",
            "|---|---|---|",
        ]
    )
    for item in summary["dominance"]["findings"]:
        dominators = "; ".join(item["dominators"]) if item["dominators"] else "none"
        lines.append(f"| `{item['workload']}` | `{item['best_suds']}` | `{dominators}` |")

    remaining = [
        row["gate_id"] for row in rows
        if row["required"] and row["status"] in {"fail", "partial"}
    ]
    if not remaining:
        interpretation = (
            "All required science-strength gates pass locally. This is the only "
            "state in which the package may be called a local submission candidate."
        )
    else:
        interpretation = (
            "The current package is improved but still not submission-ready. "
            f"Remaining required gates: `{','.join(remaining)}`."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "External red-team review is advisory only in this gate and is not counted",
            "as a required failure or partial. The internal substitute is recorded",
            "as useful but not equivalent to independent external review.",
            "",
            "## Regeneration",
            "",
            "```bash",
            "make suds-tetc-science-gate",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    if args.fail_on_not_ready and summary["decision"]["promotion_decision"] != "science_gate_pass_local_submission_candidate":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
