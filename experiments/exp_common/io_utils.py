"""Shared IO helpers for experiment scripts."""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
REPO_ANCHORED_TOP_LEVELS = {
    "experiments",
    "figures",
    "configs",
    "original_papers",
    "docs",
    "scripts",
    "weights",
}


def resolve_workspace_path(
    raw_path: str | Path,
    *,
    anchor: str | Path | None = None,
    fallback_anchors: Iterable[str | Path] | None = None,
    prefer_existing: bool = False,
) -> Path:
    """Resolve repo-style and experiments-style relative paths consistently.

    This prevents accidental shadow paths such as ``experiments/experiments/results``
    when a script already anchors outputs under ``experiments/`` but the caller passes
    ``experiments/results/...``.
    """
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj

    parts = path_obj.parts
    if parts and parts[0] in REPO_ANCHORED_TOP_LEVELS:
        return REPO_ROOT.joinpath(*parts)

    anchors: list[Path] = []
    primary = Path(anchor) if anchor is not None else EXPERIMENTS_ROOT
    anchors.append(primary)
    for extra in fallback_anchors or []:
        extra_path = Path(extra)
        if extra_path not in anchors:
            anchors.append(extra_path)

    candidates = [base / path_obj for base in anchors]
    if prefer_existing:
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return candidates[0]


def backup_existing_file(path: str | Path) -> str | None:
    """Move an existing file to a timestamped .bak and return its new path."""
    path_obj = Path(path)
    if not path_obj.is_file():
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"{path_obj}.{timestamp}.bak")
    path_obj.replace(backup_path)
    return str(backup_path)


def _normalize_windows_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return raw_path
    path = raw_path.strip().strip('"').strip("'")
    match = re.match(r"^([a-zA-Z]):[\\/](.*)", path)
    if match:
        drive = match.group(1).lower()
        rest = match.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path


def _find_column(header: list[str], target: str) -> int | None:
    for i, col in enumerate(header):
        if col.strip() == target:
            return i
    for i, col in enumerate(header):
        if target in col:
            return i
    return None


def _read_hwinfo_values(
    path: str,
    column: str,
    *,
    empty_msg: str,
    column_missing_msg: str,
) -> list[float]:
    # Try a range of common encodings to handle HWiNFO CSV exports.
    encodings = [
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "utf-8",
        "gbk",
        "cp936",
        "cp1252",
        "latin1",
    ]
    last_header = None
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, errors="strict", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                last_header = header
                col_idx = _find_column(header, column)
                if col_idx is None:
                    continue
                values = []
                for row in reader:
                    if len(row) <= col_idx:
                        continue
                    try:
                        values.append(float(row[col_idx]))
                    except Exception:
                        continue
                if values:
                    return values
        except Exception:
            continue

    # Fallback pass with replace on decode errors.
    for encoding in ["utf-8", "gbk", "cp936", "latin1"]:
        try:
            with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                last_header = header
                col_idx = _find_column(header, column)
                if col_idx is None:
                    continue
                values = []
                for row in reader:
                    if len(row) <= col_idx:
                        continue
                    try:
                        values.append(float(row[col_idx]))
                    except Exception:
                        continue
                if values:
                    return values
        except Exception:
            continue

    if last_header is None:
        raise SystemExit(empty_msg)
    raise SystemExit(column_missing_msg.format(column=column))


def read_hwinfo_csv_avg(csv_path: str, column: str) -> dict[str, object]:
    """Read a HWiNFO CSV and compute avg/min/max for the target column."""
    path = _normalize_windows_path(csv_path)
    if not path or not Path(path).is_file():
        raise SystemExit(f"HWiNFO CSV not found: {csv_path}")
    values = _read_hwinfo_values(
        path,
        column,
        empty_msg="HWiNFO CSV is empty or unreadable.",
        column_missing_msg="Column not found in HWiNFO CSV: {column}",
    )
    avg = sum(values) / len(values)
    return {
        "avg_w": avg,
        "samples": len(values),
        "min_w": min(values),
        "max_w": max(values),
        "path": path,
        "column": column,
    }


def read_hwinfo_csv_values(csv_path: str, column: str) -> list[float]:
    """Read a CSV column into a float list."""
    path = _normalize_windows_path(csv_path)
    if not path or not Path(path).is_file():
        raise SystemExit(f"CSV not found: {csv_path}")
    return _read_hwinfo_values(
        path,
        column,
        empty_msg="CSV is empty or unreadable",
        column_missing_msg="Column not found: {column}",
    )


__all__ = [
    "EXPERIMENTS_ROOT",
    "REPO_ROOT",
    "backup_existing_file",
    "read_hwinfo_csv_avg",
    "read_hwinfo_csv_values",
    "resolve_workspace_path",
]
