#!/usr/bin/env python3
"""Build the R13 baseline-parameter crosswalk for TETC pre-submission review.

The crosswalk is a Markdown-mirror and artifact-only audit. It verifies that
the stated Lightening, HyAtten, TeMPO, and ASTRA assumptions are present in the
local KB mirrors or in the existing R4 simulator artifacts, then writes a
reviewer-facing CSV/JSON/report. It does not escalate to source PDFs and does
not run model evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-05-17"
TAG = "20260517_r13"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
KB_ROOT = Path("kb_root/markdown")

LIGHTENING_KB = KB_ROOT / "01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md"
HYATTEN_KB = KB_ROOT / (
    "01_transformer_attention_photonic/"
    "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md"
)
TEMPO_KB = KB_ROOT / (
    "01_transformer_attention_photonic/"
    "2402.07393_TeMPO_Transformer_Acceleration_with_Co-packaged_Silicon_Photonics.md"
)
ASTRA_KB = KB_ROOT / (
    "01_transformer_attention_photonic/"
    "ASTRA_Stochastic_Transformer_Silicon_Photonics_TECS2025.md"
)

ARCH_PARAMETERS_CSV = REPORT_DATA / "suds_transformer_architecture_sim_20260513_tetc_pivot_parameters.csv"
SAME_SIM_JSON = REPORT_DATA / "suds_tetc_same_sim_baselines_20260513_tetc_pivot.json"
R4_REPORT = REPO_ROOT / "docs/reports/20260513_suds_tetc_same_simulator_baselines.md"

CSV_OUT = REPORT_DATA / f"suds_tetc_baseline_parameter_crosswalk_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_baseline_parameter_crosswalk_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260517_suds_tetc_baseline_parameter_crosswalk.md"

BASELINE_TO_R4_CONDITION = {
    "Lightening": "lightening_dptc",
    "HyAtten": "hyatten_style",
    "TeMPO": "tempo_time_multiplexed",
    "ASTRA": "astra_boundary",
}
REQUIRED_PARAMS = {
    "tile_dim",
    "tiles",
    "cores_per_tile",
    "frequency_ghz",
    "sram_global_kib",
    "sram_subarray_kib",
    "lightening_adc_temporal_factor",
    "hyatten_low_resolution_fraction",
    "tempo_time_multiplexing_boundary",
    "astra_stochastic_boundary",
    "selected_adc_sharing",
}
MINIMUM_COLUMNS = (
    "baseline",
    "source_path",
    "source_assumption",
    "suds_mapping",
    "scope_label",
    "differing_assumption",
    "claim_permission",
)


@dataclass(frozen=True)
class SourceCheck:
    name: str
    path: Path
    anchors: tuple[str, ...]


@dataclass(frozen=True)
class CrosswalkSpec:
    baseline: str
    source_path: Path
    source_anchor: str
    source_assumption: str
    suds_mapping: str
    scope_label: str
    differing_assumption: str
    claim_permission: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-parameters-csv", type=Path, default=ARCH_PARAMETERS_CSV)
    parser.add_argument("--same-sim-json", type=Path, default=SAME_SIM_JSON)
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


def parameter_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["parameter"]: row for row in rows if row.get("parameter")}


def verify_source_checks(checks: tuple[SourceCheck, ...]) -> list[str]:
    verified: list[str] = []
    for check in checks:
        if not check.path.is_file():
            raise SystemExit(f"missing KB mirror for {check.name}: {check.path}")
        text = check.path.read_text(encoding="utf-8")
        missing = [anchor for anchor in check.anchors if anchor not in text]
        if missing:
            raise SystemExit(
                f"source assumption anchor missing in {check.name} mirror: "
                + "; ".join(missing)
            )
        verified.extend(f"{check.name}: {anchor}" for anchor in check.anchors)
    return verified


def r4_condition_rows(same_sim: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    return [
        row for row in same_sim.get("rows", [])
        if row.get("condition") == condition
    ]


def condition_summary(same_sim: dict[str, Any], baseline: str) -> dict[str, str]:
    condition = BASELINE_TO_R4_CONDITION[baseline]
    rows = r4_condition_rows(same_sim, condition)
    if not rows:
        raise SystemExit(f"missing R4 rows for condition: {condition}")
    scopes = sorted({str(row.get("comparison_scope", "")) for row in rows if row.get("comparison_scope")})
    labels = sorted({
        str(row.get("accuracy_evidence_label", ""))
        for row in rows
        if row.get("accuracy_evidence_label")
    })
    dominance = sorted({str(row.get("dominance_status", "")) for row in rows if row.get("dominance_status")})
    edp_values = [
        float(row["edp_ratio_vs_lightening"])
        for row in rows
        if row.get("edp_ratio_vs_lightening") not in (None, "", "nan")
    ]
    edp_range = "n/a" if not edp_values else f"{min(edp_values):.3f}-{max(edp_values):.3f}"
    differences = sorted({
        str(row.get("assumption_differences", ""))
        for row in rows
        if row.get("assumption_differences")
    })
    return {
        "r4_condition": condition,
        "r4_workloads": ",".join(sorted({str(row.get("workload", "")) for row in rows})),
        "r4_scope": ",".join(scopes),
        "r4_accuracy_evidence": ",".join(labels),
        "r4_edp_ratio_vs_lightening_range": edp_range,
        "r4_dominance_status": ",".join(dominance),
        "r4_assumption_differences": " | ".join(differences),
    }


def build_specs(params: dict[str, dict[str, str]]) -> list[CrosswalkSpec]:
    parameter_source = Path(repo_path(ARCH_PARAMETERS_CSV))
    same_sim_source = Path(repo_path(SAME_SIM_JSON))
    lightening_config = (
        f"tile_dim={params['tile_dim']['value']}, tiles={params['tiles']['value']}, "
        f"cores_per_tile={params['cores_per_tile']['value']}, "
        f"frequency_ghz={params['frequency_ghz']['value']}, "
        f"sram_global_kib={params['sram_global_kib']['value']}, "
        f"sram_subarray_kib={params['sram_subarray_kib']['value']}, "
        f"adc_temporal_factor={float(params['lightening_adc_temporal_factor']['value']):.6f}, "
        f"adc_sharing={params['selected_adc_sharing']['value']}"
    )
    hyatten_fraction = float(params["hyatten_low_resolution_fraction"]["value"])
    return [
        CrosswalkSpec(
            baseline="Lightening",
            source_path=LIGHTENING_KB,
            source_anchor="DPTC dynamic full-range matrix multiplication and temporal accumulation",
            source_assumption=(
                "Lightening uses a deterministic, coherent DPTC fabric for dynamic "
                "full-range Transformer matrix multiplication, with temporal "
                "accumulation before O-E conversion."
            ),
            suds_mapping=(
                "SUDS adopts this Lightening-style deterministic DPTC as the "
                "same-fabric reference row and applies scheduler-derived "
                "KEEP/DEGRADE/PRUNE policy decisions on that fabric."
            ),
            scope_label="same_scope_baseline",
            differing_assumption="none; this is the same-scope DPTC reference",
            claim_permission="main comparison",
        ),
        CrosswalkSpec(
            baseline="Lightening",
            source_path=parameter_source,
            source_anchor="R4 architecture parameter table selected Lightening-style configuration",
            source_assumption=(
                "The simulator parameter table fixes the selected DPTC operating "
                f"point: {lightening_config}."
            ),
            suds_mapping=(
                "R4 same-simulator rows use this common tile, frequency, memory, "
                "ADC/DAC, and workload-shape surface before dominance is checked."
            ),
            scope_label="same_scope_baseline",
            differing_assumption="none; shared simulator configuration",
            claim_permission="main comparison",
        ),
        CrosswalkSpec(
            baseline="HyAtten",
            source_path=HYATTEN_KB,
            source_anchor="over 85 percent low-resolution signals with digital fallback",
            source_assumption=(
                "HyAtten classifies attention signals by converter range: about "
                f"{hyatten_fraction:.0%} of analog signals use low-resolution "
                "4-bit conversion, while higher-resolution residuals are handled "
                "by digital circuits."
            ),
            suds_mapping=(
                "R4 keeps a HyAtten-style low-resolution signal-selection row "
                "visible with degrade_ratio=0.85 and keep_ratio=0.15, using the "
                "same calibrated ADC/DAC tier table but not the selected SUDS "
                "policy semantics."
            ),
            scope_label="boundary_baseline",
            differing_assumption=(
                "low-resolution selection and digital fallback differ from the "
                "selected deterministic DPTC scheduler policy"
            ),
            claim_permission="boundary context",
        ),
        CrosswalkSpec(
            baseline="HyAtten",
            source_path=same_sim_source,
            source_anchor="R4 hyatten_style accuracy label is literature_baseline_unmeasured_locally",
            source_assumption=(
                "The local R4 artifact records HyAtten-style rows with literature "
                "or unmeasured local accuracy evidence rather than local MPS "
                "accuracy linkage."
            ),
            suds_mapping=(
                "The row can contextualize conversion tradeoffs, but it is not "
                "eligible as an equal-accuracy dominance candidate."
            ),
            scope_label="not_dominance_candidate",
            differing_assumption="unmeasured local accuracy",
            claim_permission="no claim",
        ),
        CrosswalkSpec(
            baseline="TeMPO",
            source_path=TEMPO_KB,
            source_anchor="time-multiplexed dynamic PTC and hierarchical temporal integration",
            source_assumption=(
                "TeMPO uses a time-multiplexed dynamic photonic tensor accelerator "
                "with customized slow-light MZM devices and hierarchical "
                "photocurrent and temporal integration."
            ),
            suds_mapping=(
                "R4 includes a TeMPO-style time-multiplexed readout boundary row "
                "inside the same simulator accounting surface, but the local "
                "implementation remains the selected Lightening-style DPTC fabric."
            ),
            scope_label="boundary_baseline",
            differing_assumption=(
                "time multiplexing, readout mode, and customized device/circuit "
                "assumptions differ from the selected DPTC fabric"
            ),
            claim_permission="boundary context",
        ),
        CrosswalkSpec(
            baseline="TeMPO",
            source_path=same_sim_source,
            source_anchor="R4 tempo_time_multiplexed accuracy label is literature_architecture_boundary_unmeasured_locally",
            source_assumption=(
                "The local R4 artifact records TeMPO-style rows as architecture "
                "boundary rows with no local measured accuracy linkage."
            ),
            suds_mapping=(
                "The row can flag a lower-readout-cost architecture boundary, but "
                "it cannot be used as a same-fabric or equal-accuracy dominance "
                "claim."
            ),
            scope_label="not_dominance_candidate",
            differing_assumption="unmeasured local accuracy",
            claim_permission="no claim",
        ),
        CrosswalkSpec(
            baseline="ASTRA",
            source_path=ASTRA_KB,
            source_anchor="stochastic signed optical multipliers and temporal analog accumulation",
            source_assumption=(
                "ASTRA replaces deterministic DPTC operation with "
                "stochastic signed optical multiplication, compute-capable "
                "transducers, DAC-removal assumptions, and temporal analog "
                "accumulation."
            ),
            suds_mapping=(
                "R4 includes an ASTRA-style stochastic boundary row to expose "
                "alternate-fabric efficiency, while SUDS remains a deterministic "
                "DPTC scheduler/control proposal."
            ),
            scope_label="boundary_baseline",
            differing_assumption=(
                "stochastic fabric, stochastic format, DAC-removal assumptions, "
                "and digital fallback differ from the selected DPTC fabric"
            ),
            claim_permission="boundary context",
        ),
        CrosswalkSpec(
            baseline="ASTRA",
            source_path=same_sim_source,
            source_anchor="R4 astra_boundary accuracy label is literature_architecture_boundary_unmeasured_locally",
            source_assumption=(
                "The local R4 artifact records ASTRA-style rows as stochastic "
                "architecture boundary rows with no local measured accuracy linkage."
            ),
            suds_mapping=(
                "The row can show alternate-fabric modeled EDP pressure, but it "
                "is not a same-fabric SUDS dominator."
            ),
            scope_label="not_dominance_candidate",
            differing_assumption="stochastic format plus unmeasured local accuracy",
            claim_permission="no claim",
        ),
    ]


def build_rows(specs: list[CrosswalkSpec], same_sim: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in specs:
        r4 = condition_summary(same_sim, spec.baseline)
        rows.append(
            {
                "baseline": spec.baseline,
                "source_path": repo_path(spec.source_path),
                "source_anchor": spec.source_anchor,
                "source_assumption": spec.source_assumption,
                "suds_mapping": spec.suds_mapping,
                "scope_label": spec.scope_label,
                "differing_assumption": spec.differing_assumption,
                "claim_permission": spec.claim_permission,
                **r4,
            }
        )
    return rows


def acceptance_summary(rows: list[dict[str, str]], same_sim: dict[str, Any], verified_anchors: list[str]) -> dict[str, Any]:
    baselines = {row["baseline"] for row in rows}
    scope_counts = Counter(row["scope_label"] for row in rows)
    blockers: list[str] = []
    if baselines != set(BASELINE_TO_R4_CONDITION):
        blockers.append("missing_required_baseline")
    if not any(row["baseline"] == "Lightening" and row["scope_label"] == "same_scope_baseline" for row in rows):
        blockers.append("lightening_same_scope_reference_missing")
    for baseline in ("HyAtten", "TeMPO", "ASTRA"):
        if not any(row["baseline"] == baseline and row["scope_label"] == "boundary_baseline" for row in rows):
            blockers.append(f"{baseline.lower()}_boundary_row_missing")
        if not any(row["baseline"] == baseline and row["claim_permission"] == "no claim" for row in rows):
            blockers.append(f"{baseline.lower()}_not_dominance_candidate_missing")
    if not all(row.get(column) for row in rows for column in MINIMUM_COLUMNS):
        blockers.append("minimum_csv_column_empty")
    r4_summary = same_sim.get("summary", {})
    if r4_summary.get("decision", {}).get("r4_acceptance_state") != "pass":
        blockers.append("input_r4_fairness_matrix_not_pass")
    boundary_lower = r4_summary.get("boundary_lower_edp_rows", [])
    return {
        "tag": TAG,
        "date": DATE,
        "n_rows": len(rows),
        "baselines": sorted(baselines),
        "scope_counts": dict(sorted(scope_counts.items())),
        "lightening_same_scope_reference": "Lightening" in baselines and not any(
            row["baseline"] == "Lightening" and row["scope_label"] != "same_scope_baseline"
            for row in rows
        ),
        "alternate_fabric_boundaries_visible": all(
            any(row["baseline"] == baseline and row["scope_label"] == "boundary_baseline" for row in rows)
            for baseline in ("HyAtten", "TeMPO", "ASTRA")
        ),
        "not_dominance_candidates_visible": all(
            any(row["baseline"] == baseline and row["scope_label"] == "not_dominance_candidate" for row in rows)
            for baseline in ("HyAtten", "TeMPO", "ASTRA")
        ),
        "verified_anchor_count": len(verified_anchors),
        "verified_sources": verified_anchors,
        "r4_acceptance_state": r4_summary.get("decision", {}).get("r4_acceptance_state", ""),
        "r4_boundary_lower_edp_rows": boundary_lower,
        "decision": {
            "r13_2_acceptance_state": "pass" if not blockers else "fail",
            "blockers": blockers,
            "claim_update_required": False,
        },
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, *, args: argparse.Namespace, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": TAG,
            "roadmap_item": "R13-2_baseline_parameter_crosswalk",
            "evidence_label": "baseline_parameter_crosswalk",
            "regeneration_command": "make suds-tetc-baseline-crosswalk",
            "inputs": {
                "architecture_parameters_csv": repo_path(args.architecture_parameters_csv),
                "same_sim_json": repo_path(args.same_sim_json),
                "r4_report": repo_path(R4_REPORT),
            },
        },
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def markdown_table(rows: list[dict[str, str]], columns: tuple[str, ...]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("\n", " ") for column in columns) + " |")
    return lines


def write_report(path: Path, *, args: argparse.Namespace, rows: list[dict[str, str]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = summary["decision"]
    compact_rows = []
    for baseline in ("Lightening", "HyAtten", "TeMPO", "ASTRA"):
        baseline_rows = [row for row in rows if row["baseline"] == baseline]
        compact_rows.append(
            {
                "Baseline": baseline,
                "Scope": ", ".join(sorted({row["scope_label"] for row in baseline_rows})),
                "Key assumption": baseline_rows[0]["source_assumption"],
                "SUDS mapping": baseline_rows[0]["suds_mapping"],
                "Permission": ", ".join(sorted({row["claim_permission"] for row in baseline_rows})),
            }
        )
    lines = [
        "# SUDS TETC Baseline Parameter Crosswalk",
        "",
        f"Date: `{DATE}`",
        "Roadmap item: `R13-2_baseline_parameter_crosswalk`",
        "Evidence label: `baseline_parameter_crosswalk`",
        f"Acceptance state: `{decision['r13_2_acceptance_state']}`",
        "",
        "## Scope",
        "",
        "This report pre-answers baseline-fairness questions for Lightening,",
        "HyAtten, TeMPO, and ASTRA using only the local KB Markdown mirrors and",
        "the existing R4 same-simulator artifacts. It is a parameter and claim",
        "permission crosswalk, not a new model-evaluation or device-validation run.",
        "",
        "## Decision",
        "",
        f"- R13-2 acceptance: `{decision['r13_2_acceptance_state']}`",
        f"- Blockers: `{'; '.join(decision['blockers']) if decision['blockers'] else 'none'}`",
        f"- R4 input acceptance: `{summary['r4_acceptance_state']}`",
        f"- Verified source anchors: `{summary['verified_anchor_count']}`",
        f"- Lightening same-scope reference: `{summary['lightening_same_scope_reference']}`",
        f"- Alternate-fabric boundaries visible: `{summary['alternate_fabric_boundaries_visible']}`",
        f"- Not-dominance candidates visible: `{summary['not_dominance_candidates_visible']}`",
        "",
        "## Reviewer Crosswalk",
        "",
        *markdown_table(
            compact_rows,
            ("Baseline", "Scope", "Key assumption", "SUDS mapping", "Permission"),
        ),
        "",
        "## Full Crosswalk Rows",
        "",
        *markdown_table(
            rows,
            (
                "baseline",
                "source_path",
                "source_assumption",
                "suds_mapping",
                "scope_label",
                "differing_assumption",
                "claim_permission",
                "r4_accuracy_evidence",
                "r4_edp_ratio_vs_lightening_range",
                "r4_dominance_status",
            ),
        ),
        "",
        "## R4 Connection",
        "",
        "The crosswalk connects directly to the R4 fairness matrix. Lightening is",
        "the same-scope DPTC reference. HyAtten, TeMPO, and ASTRA remain visible",
        "as boundary rows when their converter-selection, time-multiplexed",
        "readout, stochastic-format, or local-accuracy assumptions differ from",
        "the selected deterministic SUDS DPTC fabric.",
        "",
        "R4 lower-EDP boundary rows remain boundary evidence only:",
        "",
        *markdown_table(
            [
                {
                    "workload": item.get("workload", ""),
                    "condition": item.get("condition", ""),
                    "edp_ratio_vs_lightening": f"{float(item.get('edp_ratio_vs_lightening', 0.0)):.3f}",
                    "boundary_reason": item.get("boundary_reason", ""),
                }
                for item in summary["r4_boundary_lower_edp_rows"]
            ],
            ("workload", "condition", "edp_ratio_vs_lightening", "boundary_reason"),
        ),
        "",
        "## Artifacts",
        "",
        f"- CSV: `{repo_path(args.csv_out)}`",
        f"- JSON: `{repo_path(args.json_out)}`",
        f"- Report: `{repo_path(args.report_out)}`",
        f"- R4 fairness report: `{repo_path(R4_REPORT)}`",
        "",
        "## Regeneration",
        "",
        "```bash",
        "make suds-tetc-baseline-crosswalk",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_checks = (
        SourceCheck(
            "Lightening",
            LIGHTENING_KB,
            (
                "dynamic full-range matrix multiplication",
                "analog-domain temporal accumulation",
            ),
        ),
        SourceCheck(
            "HyAtten",
            HYATTEN_KB,
            (
                "over 85% of the analog signals",
                "4-bit ADCs successfully process over 85% of signals",
                "high-resolution signals constitute less than 15%",
            ),
        ),
        SourceCheck(
            "TeMPO",
            TEMPO_KB,
            (
                "time-multiplexed dynamic photonic tensor accelerator",
                "hierarchically accumulated via parallel photocurrent aggregation",
                "customized slow-light MZM",
            ),
        ),
        SourceCheck(
            "ASTRA",
            ASTRA_KB,
            (
                "stochastic computing principles",
                "full-range optical stochastic multipliers",
                "temporal analog accumulation",
            ),
        ),
    )
    verified_anchors = verify_source_checks(source_checks)
    params = parameter_lookup(load_csv(args.architecture_parameters_csv))
    missing_params = sorted(REQUIRED_PARAMS - set(params))
    if missing_params:
        raise SystemExit("missing required architecture parameters: " + ", ".join(missing_params))
    same_sim = load_json(args.same_sim_json)
    specs = build_specs(params)
    rows = build_rows(specs, same_sim)
    summary = acceptance_summary(rows, same_sim, verified_anchors)
    if summary["decision"]["r13_2_acceptance_state"] != "pass":
        raise SystemExit("R13-2 crosswalk acceptance failed: " + "; ".join(summary["decision"]["blockers"]))
    write_csv(args.csv_out, rows)
    write_json(args.json_out, args=args, rows=rows, summary=summary)
    write_report(args.report_out, args=args, rows=rows, summary=summary)
    print(f"wrote {repo_path(args.csv_out)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")


if __name__ == "__main__":
    main()
