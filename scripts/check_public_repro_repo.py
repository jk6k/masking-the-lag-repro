#!/usr/bin/env python3
"""Validate the reader-facing reproduction repository surface."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


FREEZE_TAG = "20260430_full_figure_strict_remediated"
MECHANISM_TAG = "20260426_fuller_phase4_mechanism_basis_rerun"
QUICK_DIR = Path("experiments/results/quick_reports") / FREEZE_TAG
PACK_DIR = Path("figures") / f"paper_figures_{FREEZE_TAG}"
REVIEW_DIR = Path("experiments/results/review") / FREEZE_TAG
FREEZE_JSON = Path("experiments/results/paper_sync/current_freeze.json")
MANIFEST_JSON = Path("configs/public_repro_manifest.json")

REQUIRED_FILES = [
    Path("README.md"),
    Path("REPRODUCIBILITY.md"),
    Path("NOTICE.md"),
    Path("Makefile"),
    Path("requirements.txt"),
    MANIFEST_JSON,
    FREEZE_JSON,
    QUICK_DIR / "compliance_report.json",
    PACK_DIR / "figure_numbering_registry.csv",
    PACK_DIR / "figure_traceability.csv",
    REVIEW_DIR / "claim_contract_final_unreserved_20260430.csv",
    REVIEW_DIR / "manuscript_evidence_map.csv",
    REVIEW_DIR / "review_manifest.json",
    REVIEW_DIR / "Fig2_HOPSTimeline_exact_trace.csv",
    Path("experiments/tools/check_figure_numbering_registry.py"),
    Path("experiments/tools/check_fuller_phase4_paper_data_figures.py"),
    Path("experiments/tools/render_fuller_phase4_paper_data_figures.py"),
]

BANNED_ROOTS = {
    ".agents",
    ".venv",
    ".venv311-mps",
    ".venvs",
    "archives",
    "original_papers",
    "oral_pre",
    "peper_writing",
    "tmp",
    "weights",
}
BANNED_PATH_SUBSTRINGS = {
    "experiments/datasets",
    "experiments/results/accuracy/mlx_weights",
    "experiments/results/accuracy/progress",
    "progress/events",
    "draft_candidates",
    "task_briefs",
    "20260425_fuller_phase4_datafig_redesign_freeze",
    "20260429_fuller_final_paper_numbered",
}
BANNED_SUFFIXES = {".npz", ".onnx", ".pt", ".pth", ".pptx"}
PUBLIC_METADATA_TEXT = [
    Path("README.md"),
    Path("REPRODUCIBILITY.md"),
    Path("NOTICE.md"),
    FREEZE_JSON,
    PACK_DIR / "figure_traceability.csv",
    REVIEW_DIR / "figure_traceability.csv",
    REVIEW_DIR / "review_manifest.json",
    REVIEW_DIR / "manuscript_evidence_map.csv",
    REVIEW_DIR / "data_review_report.md",
]
BANNED_METADATA_TOKENS = {
    "original_papers/",
    "markdown_v1_backup",
    "draft_candidates",
    "task_briefs",
    "data_pack_worker",
    "20260425_fuller_phase4_datafig_redesign_freeze",
    "20260429_fuller_final_paper_numbered",
}


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


def _check_required(report: Report) -> None:
    for rel_path in REQUIRED_FILES:
        if not (report.root / rel_path).is_file():
            report.add(f"missing required file: {rel_path.as_posix()}")


def _check_banned_paths(report: Report) -> None:
    for child in report.root.iterdir():
        if child.name in {".git", ".DS_Store"}:
            continue
        if child.name in BANNED_ROOTS:
            report.add(f"banned root present: {child.name}")

    for path in _iter_files(report.root):
        rel = _rel(report.root, path)
        if path.suffix.lower() in BANNED_SUFFIXES:
            report.add(f"banned binary/model suffix: {rel}")
        for token in BANNED_PATH_SUBSTRINGS:
            if token in rel:
                report.add(f"banned path token {token!r}: {rel}")


def _check_metadata_text(report: Report) -> None:
    for rel_path in PUBLIC_METADATA_TEXT:
        path = report.root / rel_path
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in BANNED_METADATA_TOKENS:
            if token in text:
                report.add(f"banned metadata token {token!r} in {rel_path.as_posix()}")


def _check_freeze(report: Report) -> None:
    path = report.root / FREEZE_JSON
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add(f"invalid current_freeze.json: {exc}")
        return

    expected = {
        "run_tag": FREEZE_TAG,
        "freeze_tag": FREEZE_TAG,
        "quick_reports_dir": QUICK_DIR.as_posix(),
        "paper_figures_dir": PACK_DIR.as_posix(),
        "review_dir": REVIEW_DIR.as_posix(),
        "mechanism_evidence_tag": MECHANISM_TAG,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            report.add(f"current_freeze.json {key} mismatch: observed={payload.get(key)!r} expected={value!r}")
    for key in ("quick_reports_dir", "paper_figures_dir", "review_dir"):
        rel_path = payload.get(key)
        if rel_path and not (report.root / rel_path).exists():
            report.add(f"current_freeze.json {key} target is missing: {rel_path}")


def _check_git(report: Report) -> None:
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
        if rel.startswith("build/"):
            continue
        if rel not in tracked_paths:
            report.add(f"untracked public file: {rel}")


def _run_subcheck(report: Report, args: list[str]) -> None:
    completed = _run(report.root, args)
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        report.add(f"subcheck failed: {' '.join(args)}\n{detail}")


def _check_artifact_tools(report: Report) -> None:
    _run_subcheck(
        report,
        [
            sys.executable,
            "experiments/tools/check_figure_numbering_registry.py",
            "--pack_dir",
            PACK_DIR.as_posix(),
        ],
    )
    _run_subcheck(
        report,
        [
            sys.executable,
            "experiments/tools/check_fuller_phase4_paper_data_figures.py",
            "--quick_dir",
            QUICK_DIR.as_posix(),
            "--mechanism_quick_dir",
            QUICK_DIR.as_posix(),
            "--pack_dir",
            PACK_DIR.as_posix(),
            "--review_dir",
            REVIEW_DIR.as_posix(),
            "--freeze_json",
            FREEZE_JSON.as_posix(),
            "--require_promoted",
        ],
    )


def validate(root: Path) -> Report:
    report = Report(root=root.resolve())
    _check_required(report)
    _check_banned_paths(report)
    _check_metadata_text(report)
    _check_freeze(report)
    _check_artifact_tools(report)
    _check_git(report)
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
