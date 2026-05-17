#!/usr/bin/env python3
"""Restore the governed local knowledge base (`original_papers/`) from git history."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path("original_papers")
DEFAULT_SOURCE_COMMIT = "fe46295e916a73cf98ef303ba77521fcaae619e3"
DEFAULT_INVENTORY = "knowledge_base_inventory.csv"

INVENTORY_FIELDS = [
    "status",
    "theme_dir",
    "file_name",
    "relative_path",
    "source_relative_path",
    "sha256",
    "title_hint",
    "validation_status",
    "issues",
    "restored_from_commit",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_root(root: Path) -> tuple[Path, Path]:
    root_abs = root if root.is_absolute() else (REPO_ROOT / root)
    root_abs = root_abs.resolve()
    root_repo_rel = root_abs.relative_to(REPO_ROOT)
    return root_abs, root_repo_rel


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_read_blob(source_commit: str, repo_relpath: Path) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{source_commit}:{repo_relpath.as_posix()}"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"git show failed for {repo_relpath}: {stderr}")
    return result.stdout


def _restore_file(
    *,
    source_commit: str,
    source_repo_rel: Path,
    dest_abs: Path,
    expected_sha: str,
    overwrite: bool,
) -> str:
    if dest_abs.exists() and not overwrite:
        if _sha256_file(dest_abs) == expected_sha:
            return "skipped"
    payload = _git_read_blob(source_commit, source_repo_rel)
    actual_sha = _sha256_bytes(payload)
    if expected_sha and actual_sha != expected_sha:
        raise SystemExit(
            f"sha256 mismatch for {source_repo_rel}: expected {expected_sha}, got {actual_sha}"
        )
    dest_abs.parent.mkdir(parents=True, exist_ok=True)
    dest_abs.write_bytes(payload)
    return "restored"


def rebuild_local_knowledge_base(
    *,
    root: Path,
    source_commit: str,
    inventory_name: str,
    overwrite: bool,
) -> dict[str, int | str]:
    root_abs, root_repo_rel = _normalize_root(root)
    active_index_rows = _read_csv(root_abs / "paper_index.csv")
    historical_index_rows = _read_csv(root_abs / "paper_index_historical_full_20260321.csv")
    quarantine_rows = _read_csv(root_abs / "paper_quarantine_manifest_20260321.csv")

    historical_by_rel = {row["relative_path"]: row for row in historical_index_rows}
    inventory_rows: list[dict[str, str]] = []
    restored = 0
    skipped = 0

    for row in active_index_rows:
        relative_path = row["relative_path"]
        source_repo_rel = root_repo_rel / relative_path
        dest_abs = root_abs / relative_path
        action = _restore_file(
            source_commit=source_commit,
            source_repo_rel=source_repo_rel,
            dest_abs=dest_abs,
            expected_sha=row["sha256"],
            overwrite=overwrite,
        )
        restored += action == "restored"
        skipped += action == "skipped"
        inventory_rows.append(
            {
                "status": "ACTIVE",
                "theme_dir": row["theme_dir"],
                "file_name": row["file_name"],
                "relative_path": relative_path,
                "source_relative_path": relative_path,
                "sha256": row["sha256"],
                "title_hint": row["title_hint"],
                "validation_status": "OK",
                "issues": "",
                "restored_from_commit": source_commit,
            }
        )

    for row in quarantine_rows:
        historical_row = historical_by_rel.get(row["original_relative_path"])
        if historical_row is None:
            raise SystemExit(
                "quarantine source not found in historical index: "
                f"{row['original_relative_path']}"
            )
        source_repo_rel = root_repo_rel / row["original_relative_path"]
        dest_abs = root_abs / row["quarantine_relative_path"]
        action = _restore_file(
            source_commit=source_commit,
            source_repo_rel=source_repo_rel,
            dest_abs=dest_abs,
            expected_sha=historical_row["sha256"],
            overwrite=overwrite,
        )
        restored += action == "restored"
        skipped += action == "skipped"
        inventory_rows.append(
            {
                "status": "QUARANTINED",
                "theme_dir": row["theme_dir"],
                "file_name": row["file_name"],
                "relative_path": row["quarantine_relative_path"],
                "source_relative_path": row["original_relative_path"],
                "sha256": historical_row["sha256"],
                "title_hint": historical_row["title_hint"],
                "validation_status": row["historical_status"],
                "issues": row["historical_issues"],
                "restored_from_commit": source_commit,
            }
        )

    inventory_rows.sort(key=lambda row: (row["status"], row["relative_path"]))
    inventory_path = root_abs / inventory_name
    _write_csv(inventory_path, INVENTORY_FIELDS, inventory_rows)

    active_missing = sum(1 for row in active_index_rows if not (root_abs / row["relative_path"]).is_file())
    quarantine_missing = sum(
        1 for row in quarantine_rows if not (root_abs / row["quarantine_relative_path"]).is_file()
    )

    return {
        "active_count": len(active_index_rows),
        "quarantine_count": len(quarantine_rows),
        "inventory_count": len(inventory_rows),
        "restored_count": restored,
        "skipped_count": skipped,
        "active_missing": active_missing,
        "quarantine_missing": quarantine_missing,
        "inventory_path": str(inventory_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the current governed original_papers knowledge base from retained git history."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--inventory-name", default=DEFAULT_INVENTORY)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = rebuild_local_knowledge_base(
        root=args.root,
        source_commit=args.source_commit,
        inventory_name=args.inventory_name,
        overwrite=args.overwrite,
    )
    print(
        "[knowledge-base-rebuild] "
        f"active={summary['active_count']} "
        f"quarantine={summary['quarantine_count']} "
        f"inventory={summary['inventory_count']} "
        f"restored={summary['restored_count']} "
        f"skipped={summary['skipped_count']} "
        f"active_missing={summary['active_missing']} "
        f"quarantine_missing={summary['quarantine_missing']} "
        f"inventory_path={summary['inventory_path']}"
    )


if __name__ == "__main__":
    main()
