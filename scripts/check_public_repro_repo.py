#!/usr/bin/env python3
"""Validate the reader-facing reproduction repository surface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FREEZE_JSON = Path("experiments/results/paper_sync/current_freeze.json")
MANIFEST_JSON = Path("configs/public_repro_manifest.json")
DEFAULT_FREEZE_TAG = "20260430_full_figure_strict_remediated"
DEFAULT_MECHANISM_TAG = "20260426_fuller_phase4_mechanism_basis_rerun"


@dataclass
class Report:
    root: Path
    errors: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.errors.append(message)

    def render(self) -> str:
        lines = [f"[public-repro-check] root={self.root}"]
        for error in self.errors:
            lines.append(f"[public-repro-check] ERROR {error}")
        lines.append(f"[public-repro-check] summary: {len(self.errors)} error(s)")
        return "\n".join(lines)


def _run(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=root, text=True, capture_output=True, check=False)


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file():
            files.append(path)
    return files


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _list_field(manifest: dict[str, Any], key: str) -> set[str]:
    value = manifest.get(key)
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value}


def _load_manifest(report: Report) -> dict[str, Any]:
    path = report.root / MANIFEST_JSON
    if not path.is_file():
        report.add(f"missing required file: {MANIFEST_JSON.as_posix()}")
        return {
            "freeze_tag": DEFAULT_FREEZE_TAG,
            "mechanism_evidence_tag": DEFAULT_MECHANISM_TAG,
            "public_layers": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add(f"invalid public repro manifest: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _public_paths(manifest: dict[str, Any]) -> dict[str, Path | str]:
    freeze_tag = str(manifest.get("freeze_tag") or DEFAULT_FREEZE_TAG)
    mechanism_tag = str(manifest.get("mechanism_evidence_tag") or DEFAULT_MECHANISM_TAG)
    layers = manifest.get("public_layers")
    if not isinstance(layers, dict):
        layers = {}
    quick_dir = Path(str(layers.get("quick_reports_dir") or f"experiments/results/quick_reports/{freeze_tag}"))
    pack_dir = Path(str(layers.get("figures_dir") or f"figures/paper_figures_{freeze_tag}"))
    review_dir = Path(str(layers.get("review_dir") or f"experiments/results/review/{freeze_tag}"))
    return {
        "freeze_tag": freeze_tag,
        "mechanism_tag": mechanism_tag,
        "quick_dir": quick_dir,
        "pack_dir": pack_dir,
        "review_dir": review_dir,
    }


def _required_files(manifest: dict[str, Any]) -> list[Path]:
    configured = manifest.get("required_files")
    if isinstance(configured, list) and configured:
        return [Path(str(item)) for item in configured]

    paths = _public_paths(manifest)
    quick_dir = paths["quick_dir"]
    pack_dir = paths["pack_dir"]
    review_dir = paths["review_dir"]
    assert isinstance(quick_dir, Path)
    assert isinstance(pack_dir, Path)
    assert isinstance(review_dir, Path)
    return [
        Path("README.md"),
        Path("REPRODUCIBILITY.md"),
        Path("NOTICE.md"),
        Path("Makefile"),
        Path("requirements.txt"),
        MANIFEST_JSON,
        FREEZE_JSON,
        quick_dir / "compliance_report.json",
        pack_dir / "figure_numbering_registry.csv",
        pack_dir / "figure_traceability.csv",
        review_dir / "claim_contract_final_unreserved_20260430.csv",
        review_dir / "manuscript_evidence_map.csv",
        review_dir / "review_manifest.json",
        review_dir / "Fig2_HOPSTimeline_exact_trace.csv",
        Path("experiments/tools/check_figure_numbering_registry.py"),
        Path("experiments/tools/check_fuller_phase4_paper_data_figures.py"),
        Path("experiments/tools/render_fuller_phase4_paper_data_figures.py"),
    ]


def _allowed_local_roots(manifest: dict[str, Any]) -> set[str]:
    return _list_field(manifest, "allowed_local_roots") | {
        ".pytest_cache",
        ".venv",
        ".venv311-mps",
        ".venvs",
        "__pycache__",
        "build",
    }


def _is_allowed_local_path(rel: str, manifest: dict[str, Any]) -> bool:
    return rel == ".DS_Store" or any(
        rel == root or rel.startswith(f"{root}/")
        for root in _allowed_local_roots(manifest)
    )


def _metadata_text_paths(manifest: dict[str, Any]) -> list[Path]:
    paths = _public_paths(manifest)
    pack_dir = paths["pack_dir"]
    review_dir = paths["review_dir"]
    assert isinstance(pack_dir, Path)
    assert isinstance(review_dir, Path)
    return [
        Path("README.md"),
        Path("REPRODUCIBILITY.md"),
        Path("NOTICE.md"),
        FREEZE_JSON,
        pack_dir / "figure_traceability.csv",
        review_dir / "figure_traceability.csv",
        review_dir / "review_manifest.json",
        review_dir / "manuscript_evidence_map.csv",
        review_dir / "data_review_report.md",
    ]


def _check_required(report: Report, manifest: dict[str, Any]) -> None:
    for rel_path in _required_files(manifest):
        if not (report.root / rel_path).is_file():
            report.add(f"missing required file: {rel_path.as_posix()}")


def _check_banned_paths(report: Report, manifest: dict[str, Any]) -> None:
    banned_roots = _list_field(manifest, "banned_roots")
    banned_path_substrings = _list_field(manifest, "banned_path_substrings")
    banned_suffixes = _list_field(manifest, "banned_suffixes")

    for child in report.root.iterdir():
        if child.name in {".git", ".DS_Store"}:
            continue
        if child.name in banned_roots and child.name not in _allowed_local_roots(manifest):
            report.add(f"banned root present: {child.name}")

    for path in _iter_files(report.root):
        rel = _rel(report.root, path)
        if _is_allowed_local_path(rel, manifest):
            continue
        if path.suffix.lower() in banned_suffixes:
            report.add(f"banned binary/model suffix: {rel}")
        for token in banned_path_substrings:
            if token in rel:
                report.add(f"banned path token {token!r}: {rel}")


def _check_metadata_text(report: Report, manifest: dict[str, Any]) -> None:
    banned_tokens = _list_field(manifest, "banned_metadata_tokens")
    for rel_path in _metadata_text_paths(manifest):
        path = report.root / rel_path
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in banned_tokens:
            if token in text:
                report.add(f"banned metadata token {token!r} in {rel_path.as_posix()}")


def _check_freeze(report: Report, manifest: dict[str, Any]) -> None:
    path = report.root / FREEZE_JSON
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add(f"invalid current_freeze.json: {exc}")
        return

    paths = _public_paths(manifest)
    quick_dir = paths["quick_dir"]
    pack_dir = paths["pack_dir"]
    review_dir = paths["review_dir"]
    assert isinstance(quick_dir, Path)
    assert isinstance(pack_dir, Path)
    assert isinstance(review_dir, Path)
    expected = {
        "run_tag": paths["freeze_tag"],
        "freeze_tag": paths["freeze_tag"],
        "quick_reports_dir": quick_dir.as_posix(),
        "paper_figures_dir": pack_dir.as_posix(),
        "review_dir": review_dir.as_posix(),
        "mechanism_evidence_tag": paths["mechanism_tag"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            report.add(f"current_freeze.json {key} mismatch: observed={payload.get(key)!r} expected={value!r}")
    for key in ("quick_reports_dir", "paper_figures_dir", "review_dir"):
        rel_path = payload.get(key)
        if rel_path and not (report.root / rel_path).exists():
            report.add(f"current_freeze.json {key} target is missing: {rel_path}")


def _check_git(report: Report, manifest: dict[str, Any]) -> None:
    if not (report.root / ".git").is_dir():
        return
    status = _run(report.root, ["git", "status", "--short"])
    if status.returncode != 0:
        report.add(f"git status failed: {status.stderr.strip()}")
        return
    if status.stdout.strip():
        report.add(f"git working tree is not clean: {status.stdout.strip()}")
    tracked = _run(report.root, ["git", "ls-files"])
    if tracked.returncode != 0:
        report.add(f"git ls-files failed: {tracked.stderr.strip()}")
        return
    tracked_paths = set(tracked.stdout.splitlines())
    for path in _iter_files(report.root):
        rel = _rel(report.root, path)
        if _is_allowed_local_path(rel, manifest):
            continue
        if rel not in tracked_paths:
            report.add(f"untracked public file: {rel}")


def _run_subcheck(report: Report, args: list[str]) -> None:
    completed = _run(report.root, args)
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        report.add(f"subcheck failed: {' '.join(args)}\n{detail}")


def _check_artifact_tools(report: Report, manifest: dict[str, Any]) -> None:
    paths = _public_paths(manifest)
    quick_dir = paths["quick_dir"]
    pack_dir = paths["pack_dir"]
    review_dir = paths["review_dir"]
    assert isinstance(quick_dir, Path)
    assert isinstance(pack_dir, Path)
    assert isinstance(review_dir, Path)

    _run_subcheck(
        report,
        [
            sys.executable,
            "experiments/tools/check_figure_numbering_registry.py",
            "--pack_dir",
            pack_dir.as_posix(),
        ],
    )
    _run_subcheck(
        report,
        [
            sys.executable,
            "experiments/tools/check_fuller_phase4_paper_data_figures.py",
            "--quick_dir",
            quick_dir.as_posix(),
            "--mechanism_quick_dir",
            quick_dir.as_posix(),
            "--pack_dir",
            pack_dir.as_posix(),
            "--review_dir",
            review_dir.as_posix(),
            "--freeze_json",
            FREEZE_JSON.as_posix(),
            "--require_promoted",
        ],
    )


def validate(root: Path) -> Report:
    report = Report(root=root.resolve())
    manifest = _load_manifest(report)
    _check_required(report, manifest)
    _check_banned_paths(report, manifest)
    _check_metadata_text(report, manifest)
    _check_freeze(report, manifest)
    _check_artifact_tools(report, manifest)
    _check_git(report, manifest)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    report = validate(args.root)
    print(report.render())
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
