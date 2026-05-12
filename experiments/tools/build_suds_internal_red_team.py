#!/usr/bin/env python3
"""Build the SUDS J5 internal multi-lens red-team evidence pack.

This is an internal quality gate, not external validation.  It records the
review lenses, required questions, dispositions, and claim-boundary decisions
used after the J1/J2 quality-boost evidence was added.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAG = "20260512_j5_quality_boost"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
CSV_OUT = REPORT_DATA / f"suds_internal_red_team_{DEFAULT_TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_internal_red_team_{DEFAULT_TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260512_j5_suds_internal_red_team.md"
PAPER_TEX = REPO_ROOT / "paper/suds_paper_acmart.tex"


REVIEW_PACKET = [
    "paper/suds_paper_acmart.pdf",
    "experiments/results/report_data/suds_c1_survey_manifest_20260511.json",
    "experiments/results/report_data/suds_mobilevit_measured_validation_20260511_maxq.json",
    "experiments/results/report_data/suds_mobilevit_multimodel_validation_20260511_p2p3_quality.json",
    "experiments/results/report_data/suds_hyatten_composition_ablation_20260511_p2p3_quality.json",
    "experiments/results/report_data/suds_adc_macro_sanity_20260512_j1_quality_boost.json",
    "docs/reports/20260512_j1_suds_adc_macro_spice_suite.md",
    "experiments/results/report_data/suds_rtl_control_overhead_20260512_j2_quality_boost.json",
    "docs/reports/20260512_j2_quality_boost_suds_rtl_control_overhead.md",
    "experiments/results/report_data/suds_phy_circuit_boundary_20260511_p2p3_quality.json",
    "configs/public_repro_manifest.json",
]


QUESTIONS = {
    "Q1": "Is novelty clear after acknowledging HyAtten, ENLighten, Lightening-Transformer, ASTRA, SCATTER, serving schedulers, and approximate computing?",
    "Q2": "Does the paper ever imply slack is semantic importance?",
    "Q3": "Does SPICE evidence reduce concern, or invite circuit-reviewer rejection by overclaiming?",
    "Q4": "Is E6 beating E7 handled as a strength of the composition story?",
    "Q5": "Are measured accuracy and modeled energy clearly separated?",
    "Q6": "Is the paper too long or too appendix-heavy for JETC?",
    "Q7": "Would the paper survive if C1 were downgraded from novelty proof to scoped motivation?",
    "Q8": "Does the ADC-Tier Calibration anchor make SPICE visible enough without inviting circuit-closure review?",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
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


def paper_scan() -> dict[str, Any]:
    text = PAPER_TEX.read_text(encoding="utf-8") if PAPER_TEX.is_file() else ""
    terms = [
        "ADC-Tier Calibration",
        "SPICE closure",
        "silicon measurement",
        "measured hardware energy",
        "slack-only superiority",
        "scoped motivation",
        "not slack-only superiority",
    ]
    return {
        "paper_path": repo_path(PAPER_TEX),
        "adc_tier_anchor_count": text.count("ADC-Tier Calibration"),
        "term_counts": {term: text.count(term) for term in terms},
        "has_compact_adc_anchor": "\\paragraph{ADC-Tier Calibration.}" in text,
        "has_p3_boundary_table": "\\label{tab:p3_boundary}" in text,
    }


def artifact_summary() -> dict[str, Any]:
    j1 = load_json(REPORT_DATA / "suds_adc_macro_sanity_20260512_j1_quality_boost.json")
    j2 = load_json(REPORT_DATA / "suds_rtl_control_overhead_20260512_j2_quality_boost.json")
    gate = load_json(REPORT_DATA / "suds_p2p3_submission_gate_20260511_p2p3_quality.json")
    return {
        "j1_adc_macro": {
            "present": bool(j1),
            "evidence_label": j1.get("metadata", {}).get("evidence_label", ""),
            "promotion_decision": j1.get("metadata", {}).get("promotion_decision", ""),
            "execution_status": j1.get("metadata", {}).get("execution_status", ""),
        },
        "j2_rtl": {
            "present": bool(j2),
            "evidence_label": j2.get("metadata", {}).get("evidence_label", ""),
            "promotion_decision": j2.get("metadata", {}).get("promotion_decision", ""),
            "yosys_status": j2.get("metadata", {}).get("yosys", {}).get("status", ""),
        },
        "p2p3_gate": {
            "present": bool(gate),
            "promotion_decision": gate.get("metadata", {}).get("promotion_decision", ""),
            "jetc_route": gate.get("gate", {}).get("jetc_route", ""),
        },
    }


def finding_rows() -> list[dict[str, Any]]:
    return [
        {
            "finding_id": "J5-A1",
            "lens": "architecture_ai_accelerator",
            "question_ids": "Q1;Q2;Q4;Q7",
            "severity": "medium",
            "finding": "Novelty is strongest when stated as a scheduler-to-accelerator budget interface, not as priority over slack/deadline scheduling, approximate computing, or photonic pruning.",
            "evidence_checked": "adjacent-work table; C1 scoped survey text; E6/E7 ablation text; conclusion boundary",
            "disposition": "fixed",
            "required_action": "Keep adjacent-work boundary table and composition wording; do not say slack is semantic importance or slack-only superiority.",
            "promotion_effect": "main_text wording is eligible; no broader novelty claim promoted",
            "accepted_risk_blocks_jetc": "false",
        },
        {
            "finding_id": "J5-A2",
            "lens": "architecture_ai_accelerator",
            "question_ids": "Q4",
            "severity": "low",
            "finding": "E6 L1-signal beating E7 could look like a failed SUDS result unless framed as evidence for composition with local selectors.",
            "evidence_checked": "abstract; C5 contribution; full MPS ablation figure caption; validation-scope limitation",
            "disposition": "fixed",
            "required_action": "Leave E6 stronger than E7 in the abstract and limitations; present the measured claim as composition, not slack-only superiority.",
            "promotion_effect": "strengthens honesty of measured validation",
            "accepted_risk_blocks_jetc": "false",
        },
        {
            "finding_id": "J5-C1",
            "lens": "photonic_circuit",
            "question_ids": "Q3;Q5;Q8",
            "severity": "high",
            "finding": "The ngspice ADC macro evidence reduces ADC-energy-model skepticism only if it remains macro calibration, not silicon, PDK, extracted-layout, measured hardware-energy, or SPICE closure evidence.",
            "evidence_checked": "J1 CSV/JSON/report; ADC-Tier Calibration anchor; P3 boundary table; limitations",
            "disposition": "fixed",
            "required_action": "Keep one compact ADC-Tier Calibration anchor in main text; keep decks, stress rows, traces, and regeneration command in appendix/report artifacts.",
            "promotion_effect": "appendix support only; no circuit-closure promotion",
            "accepted_risk_blocks_jetc": "false",
        },
        {
            "finding_id": "J5-C2",
            "lens": "photonic_circuit",
            "question_ids": "Q3;Q5",
            "severity": "medium",
            "finding": "Yosys synthesis materially improves RTL existence evidence, but timing, gate-equivalent area, and power are still proxy because there is no liberty/OpenROAD/place-and-route lane.",
            "evidence_checked": "J2 RTL synthesis JSON/report; P3 boundary table; limitations",
            "disposition": "boundary",
            "required_action": "Use rtl_synthesis for the block existence check and keep timing/GE/power wording as proxy overhead accounting.",
            "promotion_effect": "appendix overhead evidence; not implementation closure",
            "accepted_risk_blocks_jetc": "false",
        },
        {
            "finding_id": "J5-S1",
            "lens": "systems_serving",
            "question_ids": "Q1;Q2;Q7",
            "severity": "medium",
            "finding": "Serving-scheduler reviewers may read SUDS as a scheduling algorithm unless the interface boundary remains explicit.",
            "evidence_checked": "adjacent-work table; related-work boundary audit; generalisation procedure",
            "disposition": "accepted-risk",
            "required_action": "Maintain wording that serving schedulers manage requests/batches while SUDS exposes an accelerator-internal control surface they could call.",
            "promotion_effect": "non-blocking risk; paper remains methodology contribution",
            "accepted_risk_blocks_jetc": "false",
        },
        {
            "finding_id": "J5-S2",
            "lens": "systems_serving",
            "question_ids": "Q5;Q6",
            "severity": "low",
            "finding": "The appendix is dense, but moving full SPICE decks or stress matrices into the main body would hurt JETC fit.",
            "evidence_checked": "results section; appendix P2/P3 tables; J1 report",
            "disposition": "fixed",
            "required_action": "Keep only compact ADC-tier anchor in the main text and leave full deck/stress/regeneration details in report/supplement.",
            "promotion_effect": "main text stays focused",
            "accepted_risk_blocks_jetc": "false",
        },
        {
            "finding_id": "J5-R1",
            "lens": "reproducibility_artifact",
            "question_ids": "Q5;Q8",
            "severity": "high",
            "finding": "Public reproduction must regenerate the ADC macro suite without private paths, private data, weights, commercial EDA, or PDK dependencies.",
            "evidence_checked": "public repro manifest; J1 deck paths; public-repro-check result",
            "disposition": "fixed",
            "required_action": "Keep deck trace paths repo-relative; expose ngspice as an optional open-source dependency; record Xyce absence as a tool blocker rather than silent fallback.",
            "promotion_effect": "supports supplement/public repro inclusion",
            "accepted_risk_blocks_jetc": "false",
        },
        {
            "finding_id": "J5-R2",
            "lens": "reproducibility_artifact",
            "question_ids": "Q6;Q7",
            "severity": "medium",
            "finding": "C1 remains useful if downgraded from novelty proof to scoped motivation because the paper already argues an interface gap within a frozen technical subset.",
            "evidence_checked": "C1 survey manifest; systematic literature evidence text; adjacent-work boundary",
            "disposition": "accepted-risk",
            "required_action": "Do not strengthen C1 beyond the frozen 167-item SUDS technical subset; keep venue/routing cards out of C1 evidence.",
            "promotion_effect": "non-blocking positioning risk",
            "accepted_risk_blocks_jetc": "false",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    *,
    tag: str,
    rows: list[dict[str, Any]],
    scan: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    do_not_use = [row for row in rows if row["disposition"] == "do_not_use"]
    blocking_risks = [row for row in rows if row["accepted_risk_blocks_jetc"] == "true"]
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_internal_red_team_{tag}",
            "review_type": "internal_red_team",
            "evidence_label": "audit",
            "promotion_decision": "audit",
            "regeneration_command": (
                f".venv311-mps/bin/python experiments/tools/build_suds_internal_red_team.py --tag {tag}"
            ),
            "not_external_validation": True,
            "claim_boundary": (
                "Internal multi-lens quality-control review only; not independent external validation, "
                "not reviewer approval, and not a basis for circuit-closure or hardware-energy claims."
            ),
        },
        "acceptance": {
            "lens_count": len({row["lens"] for row in rows}),
            "finding_count": len(rows),
            "do_not_use_count": len(do_not_use),
            "accepted_risk_blocks_jetc_count": len(blocking_risks),
            "accepted": not do_not_use and not blocking_risks,
        },
        "review_packet": REVIEW_PACKET,
        "required_questions": QUESTIONS,
        "paper_scan": scan,
        "artifact_summary": artifacts,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    tag: str,
    rows: list[dict[str, Any]],
    scan: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    accepted = all(row["disposition"] != "do_not_use" for row in rows) and all(
        row["accepted_risk_blocks_jetc"] == "false" for row in rows
    )
    by_lens: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_lens.setdefault(row["lens"], []).append(row)

    report = f"""# SUDS J5 Internal Multi-Lens Red-Team

Tag: `{tag}`
Review type: `internal_red_team`
Evidence label: `audit`
Promotion decision: `audit`

This is an internal pre-submission quality-control gate. It must not be cited as
external validation, independent peer review, reviewer approval, silicon
validation, SPICE closure, measured hardware-energy evidence, or deployment
readiness.

## Acceptance

- Lenses completed: `{len(by_lens)}`
- Findings: `{len(rows)}`
- `do_not_use` findings: `{sum(1 for row in rows if row['disposition'] == 'do_not_use')}`
- Accepted-risk blockers: `{sum(1 for row in rows if row['accepted_risk_blocks_jetc'] == 'true')}`
- Gate result: `{"pass" if accepted else "fail"}`

## Claim Boundary Memo

SUDS remains a methodology and sideband-interface paper. Slack allocates quality
budgets; local model or physics proxies select exact columns. The SPICE macro
suite calibrates ADC-tier ordering only, while the RTL lane demonstrates a
synthesizable sideband block with proxy timing/area/power accounting. The paper
does not claim foundry/PDK evidence, extracted layout, silicon measurement,
measured hardware energy, SPICE closure, placed-and-routed implementation, or
workload-general deployment readiness.

## Required Questions

| ID | Question |
|---|---|
"""
    for qid, question in QUESTIONS.items():
        report += f"| {qid} | {question} |\n"

    report += "\n## Artifact Snapshot\n\n"
    report += f"- ADC-tier anchor count: `{scan['adc_tier_anchor_count']}`\n"
    report += f"- Compact ADC anchor present: `{scan['has_compact_adc_anchor']}`\n"
    report += f"- P3 boundary table present: `{scan['has_p3_boundary_table']}`\n"
    report += (
        "- J1 ADC macro: "
        f"`{artifacts['j1_adc_macro']['evidence_label']}` / "
        f"`{artifacts['j1_adc_macro']['promotion_decision']}` / "
        f"`{artifacts['j1_adc_macro']['execution_status']}`\n"
    )
    report += (
        "- J2 RTL: "
        f"`{artifacts['j2_rtl']['evidence_label']}` / "
        f"`{artifacts['j2_rtl']['promotion_decision']}` / "
        f"`{artifacts['j2_rtl']['yosys_status']}`\n"
    )

    for lens, lens_rows in by_lens.items():
        title = lens.replace("_", " ").title()
        report += f"\n## {title}\n\n"
        report += "| Finding | Questions | Severity | Disposition | Promotion effect |\n"
        report += "|---|---|---|---|---|\n"
        for row in lens_rows:
            report += (
                f"| {row['finding']} | `{row['question_ids']}` | `{row['severity']}` | "
                f"`{row['disposition']}` | {row['promotion_effect']} |\n"
            )
        report += "\nRequired actions:\n\n"
        for row in lens_rows:
            report += f"- `{row['finding_id']}`: {row['required_action']}\n"

    report += f"""

## Regeneration

```bash
.venv311-mps/bin/python experiments/tools/build_suds_internal_red_team.py --tag {tag}
```
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = finding_rows()
    scan = paper_scan()
    artifacts = artifact_summary()
    write_csv(args.csv_out, rows)
    write_json(args.json_out, tag=args.tag, rows=rows, scan=scan, artifacts=artifacts)
    write_report(args.report_out, tag=args.tag, rows=rows, scan=scan, artifacts=artifacts)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
