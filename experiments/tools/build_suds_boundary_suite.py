#!/usr/bin/env python3
"""Build the SUDS P1.2 boundary and counterexample suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260511_maxq"
SLACK_MANIFEST = REPO_ROOT / "experiments/results/runs/slack_manifest.json"
ABLATION_JSON = REPO_ROOT / f"experiments/results/report_data/suds_ablation_matrix_{TAG}.json"
PHASE_F_JSON = REPO_ROOT / "experiments/results/runs/phase_f/phase_f_summary.json"
CSV_OUT = REPO_ROOT / f"experiments/results/report_data/suds_boundary_suite_{TAG}.csv"
JSON_OUT = REPO_ROOT / f"experiments/results/report_data/suds_boundary_suite_{TAG}.json"
REPORT_OUT = REPO_ROOT / f"docs/reports/{TAG}_suds_boundary_suite.md"

FIELDS = [
    "case_id",
    "boundary_case",
    "evidence_label",
    "source_artifact",
    "stress_knob",
    "observed_value",
    "comparator_value",
    "pass_threshold",
    "outcome",
    "promotion_decision",
    "claim_boundary_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--slack-manifest", type=Path, default=SLACK_MANIFEST)
    parser.add_argument("--ablation-json", type=Path, default=ABLATION_JSON)
    parser.add_argument("--phase-f-json", type=Path, default=PHASE_F_JSON)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_ber(snr_db: float, er_db: float) -> float:
    er_linear = 10 ** (er_db / 10.0)
    er_penalty_db = 10 * math.log10((er_linear - 1) / (er_linear + 1)) if er_linear > 1 else -999.0
    if er_penalty_db < -900:
        return 0.5
    snr_eff_linear = 10 ** ((snr_db + er_penalty_db) / 10.0)
    q_factor = math.sqrt(snr_eff_linear)
    return 0.5 * math.erfc(q_factor / math.sqrt(2.0))


def estimate_snr(bits: int) -> float:
    return 6.02 * bits + 1.76


def compute_link_budget(phy_cfg: dict[str, Any], active_wdm_channels: int, xtalk_db: float) -> dict[str, float]:
    wdm_n = max(1, int(active_wdm_channels))
    er_db = float(phy_cfg.get("er_db", 6.0))
    er_linear = 10 ** (er_db / 10.0)
    loss_db = sum(float(value) for value in phy_cfg.get("loss_path_db", {}).values())
    xtalk_linear = 10 ** (xtalk_db / 10.0)
    interference = (wdm_n - 1) * xtalk_linear * (1.0 + 1.0 / er_linear) if er_linear > 0 else 0.0
    pp_crosstalk_db = 10.0 * math.log10(1.0 + interference) if interference >= 0 else 0.0
    p_laser_dbm = (
        float(phy_cfg.get("p_sensitivity_dbm", -22.0))
        + loss_db
        + pp_crosstalk_db
        + float(phy_cfg.get("pp_extinction_db", 1.5))
        + float(phy_cfg.get("margin_db", 4.0))
    )
    return {
        "active_wdm_channels": float(wdm_n),
        "xtalk_db": float(xtalk_db),
        "pp_crosstalk_db": float(pp_crosstalk_db),
        "p_laser_dbm": float(p_laser_dbm),
    }


def aggregate_by_condition(ablation: dict[str, Any], matrix: str = "main_ablation") -> dict[str, dict[str, Any]]:
    return {
        row["condition"]: row
        for row in ablation["aggregates"]
        if row.get("matrix") == matrix and row.get("row_type") == "aggregate"
    }


def tau_rows(ablation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in ablation["aggregates"]
        if row.get("matrix") == "tau_sensitivity" and row.get("row_type") == "aggregate"
    ]
    return sorted(rows, key=lambda row: (float(row.get("tau_low", 0)), float(row.get("tau_high", 0))))


def build_rows(slack: dict[str, Any], ablation: dict[str, Any], phase_f: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    global_stats = slack["_global"]
    current_cv = float(global_stats["cv"])
    current_dynamic_range = float(global_stats["max"]) - float(global_stats["min"])
    phase_f_best = phase_f["phy_scan"]["best_passing"]
    phy_cfg = phase_f["phy_config"]
    stressed_link = compute_link_budget(
        phy_cfg,
        int(phase_f_best["active_wdm_channels"]),
        xtalk_db=3.0,
    )
    ber_degrade_er1 = estimate_ber(estimate_snr(4), er_db=1.0)

    main = aggregate_by_condition(ablation)
    tau = tau_rows(ablation)
    default_tau = next(row for row in tau if row["condition"] == "e7_tau_010_070")
    aggressive_tau = next(row for row in tau if row["condition"] == "e7_tau_008_065")
    conservative_tau = next(row for row in tau if row["condition"] == "e7_tau_012_075")
    e6 = main["e6_signal"]
    e7 = main["e7_overlay"]

    rows: list[dict[str, Any]] = [
        {
            "case_id": "B1",
            "boundary_case": "fully_parallel_operand_broadcast",
            "evidence_label": "synthetic_boundary",
            "source_artifact": rel(SLACK_MANIFEST),
            "stress_knob": "column_serialization_removed",
            "observed_value": "column_slack_cv=0.000; column_dynamic_range=0.000",
            "comparator_value": f"current_total_slack_cv={current_cv:.3f}; current_range={current_dynamic_range:.3f}",
            "pass_threshold": "requires nonzero per-column timing gradient",
            "outcome": "fail_closed",
            "promotion_decision": "boundary",
            "claim_boundary_note": "If all operands are broadcast in parallel, column-level SUDS degenerates to layer-only or disabled control.",
        },
        {
            "case_id": "B2",
            "boundary_case": "low_slack_cv_architecture",
            "evidence_label": "synthetic_boundary",
            "source_artifact": rel(SLACK_MANIFEST),
            "stress_knob": "total_slack_cv_forced_to_0.05",
            "observed_value": "synthetic_cv=0.050",
            "comparator_value": f"current_cv={current_cv:.3f}",
            "pass_threshold": "cv>=0.10 for useful ternary separation",
            "outcome": "weak_budget_do_not_promote",
            "promotion_decision": "boundary",
            "claim_boundary_note": "Low-variation pipelines do not provide enough timing diversity for a strong ternary budget.",
        },
        {
            "case_id": "B3",
            "boundary_case": "high_crosstalk_laser_stress",
            "evidence_label": "parametric_boundary",
            "source_artifact": rel(PHASE_F_JSON),
            "stress_knob": "xtalk_db=+3.0 with SUDS best active-WDM count",
            "observed_value": f"p_laser_dbm={stressed_link['p_laser_dbm']:.2f}",
            "comparator_value": f"baseline_best_p_laser_dbm={phase_f_best['link_budget']['p_laser_dbm']:.2f}",
            "pass_threshold": "p_laser_dbm<=20.0",
            "outcome": "fails_phy_feasibility",
            "promotion_decision": "boundary",
            "claim_boundary_note": "The parametric PHY alignment claim fails under extreme crosstalk stress and remains non-SPICE evidence.",
        },
        {
            "case_id": "B4",
            "boundary_case": "high_ber_low_extinction_stress",
            "evidence_label": "parametric_boundary",
            "source_artifact": rel(PHASE_F_JSON),
            "stress_knob": "er_db=1.0 for DEGRADE-tier 4-bit path",
            "observed_value": f"ber_degrade={ber_degrade_er1:.3e}",
            "comparator_value": f"baseline_ber_degrade={phase_f_best['ber_degrade']:.3e}",
            "pass_threshold": "ber<=1e-12",
            "outcome": "fails_ber_gate",
            "promotion_decision": "boundary",
            "claim_boundary_note": "A weak extinction-ratio path can erase the PHY feasibility margin.",
        },
        {
            "case_id": "B5",
            "boundary_case": "shifted_tau_thresholds",
            "evidence_label": "measured_boundary",
            "source_artifact": rel(ABLATION_JSON),
            "stress_knob": "tau=(0.08,0.65) and tau=(0.12,0.75)",
            "observed_value": (
                f"default_delta={float(default_tau['delta_top1']):.2f}pp; "
                f"aggressive_delta={float(aggressive_tau['delta_top1']):.2f}pp; "
                f"conservative_delta={float(conservative_tau['delta_top1']):.2f}pp"
            ),
            "comparator_value": f"default_top1={float(default_tau['top1']):.2f}%",
            "pass_threshold": "nearby tau settings remain interpretable; aggressive tau is reported as degradation",
            "outcome": "passes_with_degradation_case",
            "promotion_decision": "appendix",
            "claim_boundary_note": "The operating point is not a single-point artifact, but aggressive tau shifting is visibly worse.",
        },
        {
            "case_id": "B6",
            "boundary_case": "transferred_slack_manifest_xxs_xs",
            "evidence_label": "audit_boundary",
            "source_artifact": "experiments/results/paper_sync/current_freeze.json",
            "stress_knob": "reuse_mobilevit_s_manifest_for_xxs_or_xs",
            "observed_value": "model_specific_mapping_absent",
            "comparator_value": "mobilevit_s_manifest_validated",
            "pass_threshold": "model-specific slack manifest and mapping audit required",
            "outcome": "fail_closed",
            "promotion_decision": "boundary",
            "claim_boundary_note": "XXS/XS remain context only until their own slack manifests are generated and audited.",
        },
        {
            "case_id": "B7",
            "boundary_case": "same_energy_signal_proxy",
            "evidence_label": "measured_boundary",
            "source_artifact": rel(ABLATION_JSON),
            "stress_knob": "same ADC ratio E6 vs E7",
            "observed_value": f"E7_delta={float(e7['delta_top1']):.2f}pp",
            "comparator_value": f"E6_delta={float(e6['delta_top1']):.2f}pp",
            "pass_threshold": "SUDS claim must be composition, not slack-only superiority",
            "outcome": "claim_narrowed",
            "promotion_decision": "main_text_boundary",
            "claim_boundary_note": "Signal proxy selection is stronger at the same mapped budget, so SUDS is promoted as a budget interface.",
        },
    ]

    summary = {
        "tag": TAG,
        "n_cases": len(rows),
        "n_failure_or_degradation_cases": sum(
            1
            for row in rows
            if row["outcome"] in {"fail_closed", "weak_budget_do_not_promote", "fails_phy_feasibility", "fails_ber_gate", "passes_with_degradation_case"}
        ),
        "current_slack_cv": current_cv,
        "current_slack_dynamic_range": current_dynamic_range,
        "stressed_link": stressed_link,
        "ber_degrade_er1": ber_degrade_er1,
        "default_tau_delta_top1": float(default_tau["delta_top1"]),
        "aggressive_tau_delta_top1": float(aggressive_tau["delta_top1"]),
        "conservative_tau_delta_top1": float(conservative_tau["delta_top1"]),
        "claim_boundary_note": (
            "The boundary suite promotes explicit fail-closed cases. It does not "
            "broaden the claim beyond MobileViT-S measured accuracy, SST-2/MRPC "
            "smoke validation, modeled ADC tiers, calibrated energy sensitivity, "
            "and parametric PHY support."
        ),
    }
    return rows, summary


def write_json(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "tag": TAG,
                    "evidence_label": "boundary_and_counterexample_suite",
                    "promotion_decision": "appendix_limitations_update",
                    "claim_boundary_note": summary["claim_boundary_note"],
                },
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], tag: str) -> None:
    table_lines = [
        "| {case_id} | {boundary_case} | {evidence_label} | {outcome} | {claim_boundary_note} |".format(**row)
        for row in rows
    ]
    report = f"""# SUDS Boundary And Counterexample Suite

Tag: `{tag}`
Evidence label: `audit + measured + parametric boundary`
Promotion decision: `appendix_limitations_update`

## Scope

This P1.2 gate records where SUDS fails, degenerates, or must be narrowed
before reviewers have to infer those cases. The suite is not a new positive
claim. It is a claim-control artifact.

## Boundary Cases

| Case | Boundary | Evidence | Outcome | Claim note |
|---|---|---|---|---|
{chr(10).join(table_lines)}

## Key Findings

- Fully parallel operand broadcast removes the column-serial timing gradient;
  SUDS must fail closed to layer-only or disabled control in that architecture.
- A low slack-CV architecture is not a good ternary-budget target.
- The PHY support result is parametric. Under high crosstalk or weak
  extinction-ratio stress, the feasibility gate can fail.
- The measured tau sweep is usable but not flat: default E7 is
  `{summary['default_tau_delta_top1']:.2f}` pp Top-1, aggressive tau shifts to
  `{summary['aggressive_tau_delta_top1']:.2f}` pp, and conservative tau is
  `{summary['conservative_tau_delta_top1']:.2f}` pp.
- Reusing the MobileViT-S slack manifest for XXS/XS remains fail-closed until
  model-specific manifests are audited.
- The same-energy E6 signal proxy remains the strongest measured ablation, so
  the manuscript must keep the composition claim.

## Acceptance Checks

- At least one explicit failure or degradation case is shown. This suite has
  `{summary['n_failure_or_degradation_cases']}` such cases.
- The limitations section must mention the boundary suite and preserve the
  fail-closed transferred-manifest rule.
- No row broadens the claim to hardware measurement, SPICE closure, broad GLUE,
  deployment readiness, or slack-only superiority.

## Required Artifacts

- CSV: `experiments/results/report_data/suds_boundary_suite_{tag}.csv`
- JSON: `experiments/results/report_data/suds_boundary_suite_{tag}.json`
- Report: `docs/reports/{tag}_suds_boundary_suite.md`

## Regeneration

```bash
.venv311-mps/bin/python experiments/tools/build_suds_boundary_suite.py
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    slack = load_json(args.slack_manifest)
    ablation = load_json(args.ablation_json)
    phase_f = load_json(args.phase_f_json)
    rows, summary = build_rows(slack, ablation, phase_f)
    write_csv(args.csv_out, rows)
    write_json(args.json_out, rows, summary)
    write_report(args.report_out, rows, summary, args.tag)
    print(f"[suds-boundary-suite] wrote {rel(args.csv_out)}")
    print(f"[suds-boundary-suite] wrote {rel(args.json_out)}")
    print(f"[suds-boundary-suite] wrote {rel(args.report_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
