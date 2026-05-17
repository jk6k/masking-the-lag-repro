#!/usr/bin/env python3
"""Quarantine historically flagged original_papers entries and rewrite active metadata."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any


INDEX_FIELDS = [
    "theme_dir",
    "file_name",
    "relative_path",
    "sha256",
    "title_hint",
    "duplicate_group",
]
VALIDATION_FIELDS = [
    "relative_path",
    "theme_dir",
    "file_name",
    "filename_arxiv_id",
    "text_arxiv_id",
    "text_category",
    "has_photonic_keywords",
    "status",
    "issues",
]
QUARANTINE_FIELDS = [
    "original_relative_path",
    "quarantine_relative_path",
    "theme_dir",
    "file_name",
    "filename_arxiv_id",
    "text_arxiv_id",
    "text_category",
    "has_photonic_keywords",
    "historical_status",
    "historical_issues",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_if_missing(src: Path, dst: Path) -> None:
    if not dst.exists():
        shutil.copy2(src, dst)


def _write_summary_md(
    path: Path,
    *,
    active_count: int,
    quarantined_count: int,
    historical_flagged_count: int,
    quarantine_rows: list[dict[str, str]],
    quarantine_manifest_rel: str,
    historical_index_rel: str,
    historical_validation_csv_rel: str,
    historical_validation_md_rel: str,
) -> None:
    lines = [
        "# Paper Library Validation Report",
        "",
        "- Generated: 2026-03-21 local cleanup pass",
        "- Root: `original_papers`",
        f"- Active indexed PDFs: `{active_count}`",
        f"- Quarantined historical mismatches: `{quarantined_count}`",
        f"- Historical flagged entries carried forward from restored report: `{historical_flagged_count}`",
        "",
        "## Current State",
        "- Active library is restricted to non-quarantined indexed entries.",
        "- Historical mismatch PDFs have been moved to a quarantine surface.",
        "- Historical full-library metadata has been preserved separately.",
        "",
        "## Preserved Historical Artifacts",
        f"- `{historical_index_rel}`",
        f"- `{historical_validation_csv_rel}`",
        f"- `{historical_validation_md_rel}`",
        "",
        "## Quarantine Manifest",
        f"- `{quarantine_manifest_rel}`",
        "",
        "## Quarantined Files",
        "",
        "| original_relative_path | text_category | issues |",
        "|---|---|---|",
    ]
    for row in quarantine_rows:
        lines.append(
            f"| `{row['original_relative_path']}` | `{row['text_category']}` | `{row['historical_issues']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_library(root: Path, *, tag: str) -> dict[str, Any]:
    index_path = root / "paper_index.csv"
    validation_csv_path = root / "paper_validation_report.csv"
    validation_md_path = root / "paper_validation_report.md"

    historical_index_name = f"paper_index_historical_full_{tag}.csv"
    historical_validation_csv_name = f"paper_validation_report_historical_{tag}.csv"
    historical_validation_md_name = f"paper_validation_report_historical_{tag}.md"

    historical_index_path = root / historical_index_name
    historical_validation_csv_path = root / historical_validation_csv_name
    historical_validation_md_path = root / historical_validation_md_name

    _copy_if_missing(index_path, historical_index_path)
    _copy_if_missing(validation_csv_path, historical_validation_csv_path)
    _copy_if_missing(validation_md_path, historical_validation_md_path)

    index_rows = _read_csv(historical_index_path)
    validation_rows = _read_csv(historical_validation_csv_path)
    index_by_rel = {row["relative_path"]: row for row in index_rows}

    flagged_rows = [row for row in validation_rows if row.get("status") == "FLAG"]
    nonflagged_rows = [row for row in validation_rows if row.get("status") != "FLAG"]

    quarantine_dir = root / "quarantine" / f"{tag}_flagged_mismatches"
    quarantine_rows: list[dict[str, str]] = []
    active_index_rows: list[dict[str, str]] = []
    active_validation_rows: list[dict[str, str]] = []

    flagged_relpaths = {row["relative_path"] for row in flagged_rows}

    for row in flagged_rows:
        relpath = row["relative_path"]
        src = root / relpath
        if not src.is_file():
            raise SystemExit(f"Missing flagged file: {src}")
        dst = quarantine_dir / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        quarantine_rel = str(dst.relative_to(root))
        quarantine_rows.append(
            {
                "original_relative_path": relpath,
                "quarantine_relative_path": quarantine_rel,
                "theme_dir": row["theme_dir"],
                "file_name": row["file_name"],
                "filename_arxiv_id": row["filename_arxiv_id"],
                "text_arxiv_id": row["text_arxiv_id"],
                "text_category": row["text_category"],
                "has_photonic_keywords": row["has_photonic_keywords"],
                "historical_status": row["status"],
                "historical_issues": row["issues"],
            }
        )
        active_validation_rows.append(
            {
                **row,
                "relative_path": quarantine_rel,
                "status": "QUARANTINED",
                "issues": f"historical_flag_quarantined:{row['issues']}",
            }
        )

    for row in index_rows:
        relpath = row["relative_path"]
        if relpath not in flagged_relpaths:
            active_index_rows.append(row)

    for row in nonflagged_rows:
        if row["relative_path"] not in index_by_rel:
            raise SystemExit(f"Validation row not found in index: {row['relative_path']}")
        active_validation_rows.append(row)

    quarantine_manifest_path = root / f"paper_quarantine_manifest_{tag}.csv"
    _write_csv(quarantine_manifest_path, QUARANTINE_FIELDS, quarantine_rows)
    _write_csv(index_path, INDEX_FIELDS, active_index_rows)
    _write_csv(validation_csv_path, VALIDATION_FIELDS, active_validation_rows)
    _write_summary_md(
        validation_md_path,
        active_count=len(active_index_rows),
        quarantined_count=len(quarantine_rows),
        historical_flagged_count=len(flagged_rows),
        quarantine_rows=quarantine_rows,
        quarantine_manifest_rel=quarantine_manifest_path.name,
        historical_index_rel=historical_index_name,
        historical_validation_csv_rel=historical_validation_csv_name,
        historical_validation_md_rel=historical_validation_md_name,
    )

    return {
        "active_count": len(active_index_rows),
        "quarantined_count": len(quarantine_rows),
        "historical_flagged_count": len(flagged_rows),
        "quarantine_manifest": str(quarantine_manifest_path),
        "quarantine_dir": str(quarantine_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean original_papers by quarantining flagged mismatches.")
    parser.add_argument("--root", type=Path, default=Path("original_papers"))
    parser.add_argument("--tag", default="20260321")
    args = parser.parse_args()

    summary = clean_library(args.root, tag=args.tag)
    print(
        "[paper-cleanup] "
        f"active_count={summary['active_count']} "
        f"quarantined_count={summary['quarantined_count']} "
        f"quarantine_manifest={summary['quarantine_manifest']}"
    )


if __name__ == "__main__":
    main()
