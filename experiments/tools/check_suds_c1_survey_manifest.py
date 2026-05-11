#!/usr/bin/env python3
"""Acceptance checker for the SUDS C1 survey manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "experiments/results/report_data/suds_c1_survey_manifest_20260511.csv"
JSON_PATH = REPO_ROOT / "experiments/results/report_data/suds_c1_survey_manifest_20260511.json"
REPORT_PATH = REPO_ROOT / "docs/reports/20260511_suds_c1_survey_manifest_audit.md"

REQUIRED_FIELDS = (
    "item_id",
    "theme_bucket",
    "short_title",
    "venue",
    "year",
    "source_md_path",
    "source_pdf_relative_path",
    "paper_table_bucket",
    "resource_allocation_flag",
    "resource_allocation_mechanism",
    "allocation_signal",
    "scheduler_timing_signal_flag",
    "scoped_interface_flag",
    "evidence_location",
    "auditor_note",
)

BUCKET_TARGETS = {
    "Photonic Transformer accelerators": 12,
    "Photonic NN accelerators": 25,
    "Stochastic computing methods": 6,
    "System interconnect / scaling": 4,
    "Electronic LLM accelerators": 29,
    "LLM inference serving": 15,
    "Robustness / noise": 10,
    "Photonic core / platform": 16,
    "Baseline / reference / context items": 50,
}

RA_TARGETS = {
    "Photonic Transformer accelerators": 8,
    "Photonic NN accelerators": 12,
    "Stochastic computing methods": 5,
    "System interconnect / scaling": 4,
    "Electronic LLM accelerators": 15,
    "LLM inference serving": 10,
    "Robustness / noise": 6,
    "Photonic core / platform": 0,
    "Baseline / reference / context items": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--json", type=Path, default=JSON_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"not a manifest boolean: {value!r}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise SystemExit(f"CSV schema mismatch: {reader.fieldnames}")
        return list(reader)


def main() -> None:
    args = parse_args()
    rows = load_csv(args.csv)
    if len(rows) != 167:
        raise SystemExit(f"row count mismatch: {len(rows)} != 167")
    if not args.report.exists():
        raise SystemExit(f"missing report: {args.report}")

    item_ids = [row["item_id"] for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise SystemExit("duplicate item_id values found")

    source_paths = [row["source_md_path"] for row in rows]
    if len(source_paths) != len(set(source_paths)):
        raise SystemExit("duplicate source_md_path values found")

    missing = [path for path in source_paths if not Path(path).exists()]
    if missing:
        raise SystemExit(f"unreadable source_md_path values: {missing[:5]}")

    bucket_counts = Counter(row["paper_table_bucket"] for row in rows)
    if dict(bucket_counts) != BUCKET_TARGETS:
        raise SystemExit(f"bucket counts mismatch: {dict(bucket_counts)}")

    ra_counts = Counter(
        row["paper_table_bucket"]
        for row in rows
        if as_bool(row["resource_allocation_flag"])
    )
    for bucket, expected in RA_TARGETS.items():
        actual = ra_counts.get(bucket, 0)
        if actual != expected:
            raise SystemExit(f"RA count mismatch for {bucket}: {actual} != {expected}")

    for row in rows:
        if as_bool(row["scoped_interface_flag"]):
            raise SystemExit(f"scoped interface counterexample present: {row['item_id']}")
        if as_bool(row["resource_allocation_flag"]):
            if not row["resource_allocation_mechanism"].strip():
                raise SystemExit(f"RA row missing mechanism: {row['item_id']}")
            if not row["allocation_signal"].strip():
                raise SystemExit(f"RA row missing allocation signal: {row['item_id']}")
        if row["theme_bucket"] == "23_submission_venue_targets":
            raise SystemExit(f"venue/routing card leaked into C1: {row['item_id']}")

    data = json.loads(args.json.read_text(encoding="utf-8"))
    json_rows = data.get("rows", [])
    if len(json_rows) != len(rows):
        raise SystemExit("JSON row count does not match CSV")
    if data.get("metadata", {}).get("promotion_decision") != "promote":
        raise SystemExit("JSON promotion_decision is not promote")
    if len(data.get("spot_audit_item_ids", [])) < 25:
        raise SystemExit("spot audit has fewer than 25 rows")

    print("SUDS C1 survey manifest check passed")
    print(f"rows={len(rows)} ra_rows={sum(as_bool(row['resource_allocation_flag']) for row in rows)}")


if __name__ == "__main__":
    main()
