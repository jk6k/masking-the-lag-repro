#!/usr/bin/env python3
"""Validate the unified FULLER report-pack contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .fuller_experiment_program_common import (
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        REPORT_CONTRACT_FIELDS,
        REPORT_DELIVERABLE_ORDER,
        ROOT,
        _resolve_path,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        REPORT_CONTRACT_FIELDS,
        REPORT_DELIVERABLE_ORDER,
        ROOT,
        _resolve_path,
        load_program_context,
    )


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_json_list(raw: str, field_name: str) -> list[str]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {field_name}: {raw!r}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Expected JSON list in {field_name}")
    return [str(item) for item in payload]


def check_fuller_report_pack_contract(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, str]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    report_csv = _resolve_path(root_dir, outputs["report_contract_csv"])
    report_json = _resolve_path(root_dir, outputs["report_contract_json"])
    report_md = _resolve_path(root_dir, outputs["report_contract_md"])
    for path in (report_csv, report_json, report_md):
        if not path.exists():
            raise SystemExit(f"Missing report-contract output: {path}")
    rows = _load_csv_rows(report_csv)
    json_rows = json.loads(report_json.read_text(encoding="utf-8")).get("rows")
    if not isinstance(json_rows, list):
        raise SystemExit("Report contract JSON must expose rows")
    if len(rows) != len(REPORT_DELIVERABLE_ORDER):
        raise SystemExit("Report contract must contain one row per deliverable")
    if [row["deliverable_id"] for row in rows] != REPORT_DELIVERABLE_ORDER:
        raise SystemExit("Report contract deliverable order drifted")
    for field in REPORT_CONTRACT_FIELDS:
        if field not in rows[0]:
            raise SystemExit(f"Report contract is missing field: {field}")
    valid_families = set(EXPERIMENT_FAMILY_ORDER)
    for row in rows:
        source_families = _parse_json_list(row["source_family_ids_json"], f"{row['deliverable_id']}.source_family_ids_json")
        if not source_families:
            raise SystemExit(f"{row['deliverable_id']} must name at least one source family")
        invalid = [family for family in source_families if family not in valid_families]
        if invalid:
            raise SystemExit(f"{row['deliverable_id']} references unknown source families: {invalid}")
    integrated_row = next(row for row in rows if row["deliverable_id"] == "integrated_fuller_evidence_summary")
    integrated_sources = _parse_json_list(integrated_row["source_family_ids_json"], "integrated_fuller_evidence_summary.source_family_ids_json")
    for family in ("analysis_grade_replay", "realism_calibration_support", "noise_robustness", "scaling_support", "device_compare", "holdout_audit"):
        if family not in integrated_sources:
            raise SystemExit(f"integrated_fuller_evidence_summary must cite {family}")
    if "lane_isolation_runtime_smoke" in integrated_sources or "anchor_validation" in integrated_sources:
        raise SystemExit("integrated_fuller_evidence_summary must not cite engineering smoke as claim-tier evidence")
    comparison_row = next(row for row in rows if row["deliverable_id"] == "lane_comparison_table")
    comparison_sources = _parse_json_list(comparison_row["source_family_ids_json"], "lane_comparison_table.source_family_ids_json")
    if comparison_sources != ["analysis_grade_replay"]:
        raise SystemExit("lane_comparison_table must remain analysis-grade-only in the v2 program")
    note_text = report_md.read_text(encoding="utf-8")
    for needle in (
        "No deliverable in this surface may cite an undeclared experiment family",
        "Engineering smoke families may appear in governance/status deliverables",
    ):
        if needle not in note_text:
            raise SystemExit(f"Report contract note must contain: {needle}")
    return {"status": "pass", "report_contract_csv": str(report_csv.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the FULLER report-pack contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = check_fuller_report_pack_contract(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
