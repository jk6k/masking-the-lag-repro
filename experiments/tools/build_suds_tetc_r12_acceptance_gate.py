#!/usr/bin/env python3
"""Build the final R12 acceptance gate for SUDS TETC reinforcement."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260514_r12_reinforcement"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

CSV_OUT = REPORT_DATA / f"suds_tetc_r12_acceptance_gate_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_r12_acceptance_gate_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260514_suds_tetc_r12_acceptance_gate.md"

ARTIFACTS = {
    "R12a": REPORT_DATA / f"suds_tetc_rtl_simulation_{TAG}.json",
    "R12b": REPORT_DATA / f"suds_tetc_glue_task_expansion_{TAG}.json",
    "R12c": REPORT_DATA / f"suds_tetc_cross_workload_transfer_{TAG}.json",
    "R12d": REPORT_DATA / f"suds_tetc_internal_adversarial_review_{TAG}.json",
    "R12e": REPORT_DATA / f"suds_tetc_mobilevit_resolution_accuracy_{TAG}.json",
    "R12f": REPORT_DATA / f"suds_tetc_bert_multiseed_accuracy_{TAG}.json",
    "R12g": REPORT_DATA / f"suds_tetc_deit_tiny_accuracy_{TAG}.json",
    "R12h": REPORT_DATA / f"suds_tetc_adc_corner_cases_{TAG}.json",
}

ALLOWED_BOUNDARY = {
    "R12c": {"boundary_recorded", "accepted_boundary"},
    "R12f": {"boundary_recorded", "accepted_boundary"},
    "R12g": {"boundary_recorded", "review_boundary", "accepted_boundary"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def gate_row(item: str, status: str, evidence: str, blockers: list[str] | None = None, required: bool = True) -> dict[str, Any]:
    return {
        "item": item,
        "required": required,
        "status": status,
        "evidence": evidence,
        "blockers": ";".join(blockers or []),
    }


def artifact_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the artifact's acceptance-bearing view.

    Earlier R12 artifacts expose acceptance fields under ``summary``; the newer
    R12d/R12e/R12h tools expose the same class of fields under a dedicated
    top-level ``acceptance`` object. The gate must read both schemas or it can
    turn a real pass into a false fail.
    """
    summary = payload.get("summary", {})
    acceptance = payload.get("acceptance", {})
    view = dict(summary)
    view.update(acceptance)
    return view


def inspect_required(item: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary = artifact_view(payload)
    acceptance = summary.get("acceptance_state", "missing")
    blockers = list(summary.get("blockers") or [])
    if item == "R12a":
        status = "pass" if acceptance == "pass" and not blockers else "fail"
        evidence = f"acceptance={acceptance}; pass_count={summary.get('pass_count')}; fail_count={summary.get('fail_count')}"
        if acceptance != "pass":
            blockers.append("R12a_acceptance_not_pass")
    elif item == "R12b":
        status = "pass" if acceptance == "pass" and not blockers else "fail"
        evidence = f"acceptance={acceptance}; total_per_seed_rows={summary.get('total_per_seed_rows')}; tasks={len(summary.get('per_task_summary', {}))}"
        if acceptance != "pass":
            blockers.append("R12b_acceptance_not_pass")
    elif item == "R12c":
        source_blockers = list(blockers)
        status = "pass" if acceptance in ALLOWED_BOUNDARY["R12c"] else "fail"
        evidence = f"acceptance={acceptance}; transfer_rows={summary.get('transfer_rows')}; delta={summary.get('bert_to_mobilevit_delta_pp')}; boundary_reasons={source_blockers}"
        if status == "pass":
            blockers = []
        if acceptance not in ALLOWED_BOUNDARY["R12c"]:
            blockers.append("R12c_not_boundary_accepted")
    elif item == "R12d":
        meta = payload.get("metadata", {})
        input_hashes = meta.get("input_sha256", {})
        missing_hashes = [k for k, v in input_hashes.items() if not v or v == "missing"]
        status = "pass" if acceptance == "pass" and not blockers and not missing_hashes else "fail"
        evidence = f"acceptance={acceptance}; lenses={summary.get('total_lenses')}; unresolved={summary.get('unresolved_lenses', [])}; missing_hashes={missing_hashes}"
        if acceptance != "pass":
            blockers.append("R12d_acceptance_not_pass")
        if missing_hashes:
            blockers.append("missing_input_hashes: " + ",".join(missing_hashes))
    elif item == "R12e":
        status = "pass" if acceptance == "pass" and not blockers else "fail"
        evidence = f"acceptance={acceptance}; expected_rows={summary.get('expected_rows')}; worst_mean_delta={summary.get('worst_mean_delta_pp')}"
        if acceptance != "pass":
            blockers.append("R12e_acceptance_not_pass")
    elif item == "R12f":
        source_blockers = list(blockers)
        status = "pass" if acceptance in ALLOWED_BOUNDARY["R12f"] else "fail"
        evidence = f"acceptance={acceptance}; total_rows={summary.get('total_rows')}; seeds_per_condition={summary.get('seeds_per_condition')}; boundary_reasons={source_blockers}"
        if status == "pass":
            blockers = []
        if acceptance not in ALLOWED_BOUNDARY["R12f"]:
            blockers.append("R12f_not_boundary_accepted")
    elif item == "R12g":
        source_blockers = list(blockers)
        status = "pass" if acceptance in ALLOWED_BOUNDARY["R12g"] else "fail"
        evidence = f"acceptance={acceptance}; mean_delta={summary.get('mean_delta_top1_pp')}; max_abs_delta={summary.get('max_abs_delta_pp')}; boundary_reasons={source_blockers}"
        if status == "pass":
            blockers = []
        if acceptance not in ALLOWED_BOUNDARY["R12g"]:
            blockers.append("R12g_not_boundary_accepted")
    elif item == "R12h":
        status = "pass" if acceptance == "pass" and not blockers else "fail"
        evidence = f"acceptance={acceptance}; measured_rows={summary.get('measured_rows')}; energy_ordering={summary.get('energy_tier_ordering_all')}"
        if acceptance != "pass":
            blockers.append("R12h_acceptance_not_pass")
    else:
        status = "fail"
        evidence = f"unknown item {item}"
        blockers.append("unknown_item")
    return {
        "item": item,
        "required": True,
        "status": status,
        "acceptance_state": acceptance,
        "evidence": evidence,
        "blockers": blockers,
        "artifact": repo_path(ARTIFACTS[item]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["item", "required", "status", "acceptance_state", "evidence", "blockers", "artifact"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, *, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = "\n".join(
        f"| `{row['item']}` | `{row['acceptance_state']}` | `{row['status']}` | {row['evidence']} |"
        for row in rows
    )
    body = f"""# SUDS TETC R12 Acceptance Gate

Date: `2026-05-14`
Tag: `{TAG}`

## Summary

- Overall status: `{summary['status']}`
- Pass items: `{summary['pass_items']}`
- Fail items: `{summary['fail_items']}`
- Boundary-accepted items: `{summary['boundary_items']}`
- Required items: `{summary['required_items']}`

## Item Table

| Item | Acceptance | Status | Evidence |
|---|---|---|---|
{table}

## Blockers

{chr(10).join(f"- {blocker}" for blocker in summary['blockers']) if summary['blockers'] else "- none"}

## Required Artifacts

- CSV: `experiments/results/report_data/suds_tetc_r12_acceptance_gate_{TAG}.csv`
- JSON: `experiments/results/report_data/suds_tetc_r12_acceptance_gate_{TAG}.json`
- Report: `docs/reports/20260514_suds_tetc_r12_acceptance_gate.md`
"""
    path.write_text(body, encoding="utf-8")


def main() -> int:
    args = parse_args()
    artifacts = {item: load_json(path) for item, path in ARTIFACTS.items()}
    rows = [inspect_required(item, payload) for item, payload in artifacts.items()]
    pass_items = [row["item"] for row in rows if row["status"] == "pass"]
    fail_items = [row["item"] for row in rows if row["status"] != "pass"]
    boundary_items = [row["item"] for row in rows if row["status"] == "pass" and row["acceptance_state"] != "pass"]
    blockers = sorted({blocker for row in rows for blocker in row["blockers"]})
    status = "pass" if not fail_items else "fail"
    summary = {
        "status": status,
        "required_items": len(rows),
        "pass_items": len(pass_items),
        "fail_items": len(fail_items),
        "boundary_items": len(boundary_items),
        "blockers": blockers,
    }
    write_csv(args.csv_out, rows)
    write_json(args.json_out, rows=rows, summary=summary)
    write_report(args.report_out, rows=rows, summary=summary)
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.report_out}")
    print(f"Status: {status}")
    if blockers:
        print(f"Blockers: {blockers}")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
