#!/usr/bin/env python3
"""Validate the reader-facing reproduction repository surface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FREEZE_JSON = Path("experiments/results/paper_sync/current_freeze.json")
MANIFEST_JSON = Path("configs/public_repro_manifest.json")
CHECKSUMS_JSON = Path("checksums_manifest.json")
DEFAULT_FREEZE_TAG = "20260511_suds_maxq"
DEFAULT_MECHANISM_TAG = "20260511_suds_maxq"


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


def _built_in_private_metadata_tokens() -> set[str]:
    return {
        "/" + "Users" + "/",
        "jk" + "6k",
        "github.com/" + "jk" + "6k",
        "masking-the-lag-repro" + "." + "git",
    }


def _workflow(manifest: dict[str, Any]) -> str:
    return str(manifest.get("workflow") or "suds_q2")


def _is_suds(manifest: dict[str, Any]) -> bool:
    return _workflow(manifest) == "suds_q2"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    mechanism_tag = str(manifest.get("mechanism_evidence_tag") or (freeze_tag if _is_suds(manifest) else DEFAULT_MECHANISM_TAG))
    layers = manifest.get("public_layers")
    if not isinstance(layers, dict):
        layers = {}
    quick_dir = Path(str(layers.get("quick_reports_dir") or f"experiments/results/quick_reports/{freeze_tag}"))
    phase_dir = Path(str(layers.get("phase_dir") or "experiments/results/runs"))
    report_data_dir = Path(str(layers.get("report_data_dir") or "experiments/results/report_data"))
    pack_dir = Path(str(layers.get("figures_dir") or f"figures/paper_figures_{freeze_tag}"))
    review_default = f"experiments/results/review/{freeze_tag}_public" if _is_suds(manifest) else f"experiments/results/review/{freeze_tag}"
    review_dir = Path(str(layers.get("review_dir") or review_default))
    return {
        "freeze_tag": freeze_tag,
        "mechanism_tag": mechanism_tag,
        "quick_dir": quick_dir,
        "phase_dir": phase_dir,
        "report_data_dir": report_data_dir,
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
        CHECKSUMS_JSON,
        Path("Makefile"),
        Path("requirements.txt"),
        MANIFEST_JSON,
        FREEZE_JSON,
        Path("experiments/results/runs/phase_b/phase_b_summary.json"),
        Path("experiments/results/runs/phase_c/phase_c_summary.json"),
        Path("experiments/results/runs/phase_d/phase_d_summary.json"),
        Path("experiments/results/runs/phase_e/phase_e_summary.json"),
        Path("experiments/results/runs/phase_f/phase_f_summary.json"),
        Path("experiments/results/runs/slack_manifest.json"),
        Path("experiments/results/report_data/suds_ablation_matrix_20260511_maxq.csv"),
        pack_dir / "pack_metadata.json",
        pack_dir / "figure_numbering_registry.csv",
        pack_dir / "figure_traceability.csv",
        review_dir / "manuscript_evidence_map.csv",
        review_dir / "review_manifest.json",
        review_dir / "data_review_report.md",
        Path("experiments/tools/check_figure_numbering_registry.py"),
        Path("experiments/tools/render_suds_figures.py"),
        Path("scripts/check_public_repro_repo.py"),
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
    if _is_suds(manifest):
        return [
            Path("README.md"),
            Path("REPRODUCIBILITY.md"),
            Path("NOTICE.md"),
            Path("EXPORT_METADATA.json"),
            CHECKSUMS_JSON,
            FREEZE_JSON,
            pack_dir / "figure_traceability.csv",
            review_dir / "figure_traceability.csv",
            review_dir / "review_manifest.json",
            review_dir / "manuscript_evidence_map.csv",
            review_dir / "data_review_report.md",
        ]
    return [
        Path("README.md"),
        Path("REPRODUCIBILITY.md"),
        Path("NOTICE.md"),
        CHECKSUMS_JSON,
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
    allowed_binary_files = _list_field(manifest, "allowed_binary_files")

    for child in report.root.iterdir():
        if child.name in {".git", ".DS_Store"}:
            continue
        if child.name in banned_roots and child.name not in _allowed_local_roots(manifest):
            report.add(f"banned root present: {child.name}")

    for path in _iter_files(report.root):
        rel = _rel(report.root, path)
        if _is_allowed_local_path(rel, manifest):
            continue
        if rel in allowed_binary_files:
            continue
        if path.suffix.lower() in banned_suffixes:
            report.add(f"banned binary/model suffix: {rel}")
        for token in banned_path_substrings:
            if token in rel:
                report.add(f"banned path token {token!r}: {rel}")


def _check_metadata_text(report: Report, manifest: dict[str, Any]) -> None:
    banned_tokens = _list_field(manifest, "banned_metadata_tokens") | _built_in_private_metadata_tokens()
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
    phase_dir = paths["phase_dir"]
    report_data_dir = paths["report_data_dir"]
    pack_dir = paths["pack_dir"]
    review_dir = paths["review_dir"]
    assert isinstance(quick_dir, Path)
    assert isinstance(phase_dir, Path)
    assert isinstance(report_data_dir, Path)
    assert isinstance(pack_dir, Path)
    assert isinstance(review_dir, Path)
    if _is_suds(manifest):
        expected = {
            "run_tag": paths["freeze_tag"],
            "freeze_tag": paths["freeze_tag"],
            "workflow": "suds_q2",
            "phase_dir": phase_dir.as_posix(),
            "report_data_dir": report_data_dir.as_posix(),
            "paper_figures_dir": pack_dir.as_posix(),
            "review_dir": review_dir.as_posix(),
        }
    else:
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
    required_targets = ("phase_dir", "report_data_dir", "paper_figures_dir", "review_dir") if _is_suds(manifest) else ("quick_reports_dir", "paper_figures_dir", "review_dir")
    for key in required_targets:
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
        if rel in _list_field(manifest, "allowed_binary_files"):
            pass
        if rel not in tracked_paths:
            report.add(f"untracked public file: {rel}")


def _run_subcheck(report: Report, args: list[str]) -> None:
    completed = _run(report.root, args)
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        report.add(f"subcheck failed: {' '.join(args)}\n{detail}")


def _check_quick_report_inputs(report: Report, manifest: dict[str, Any]) -> None:
    paths = _public_paths(manifest)
    quick_dir = paths["quick_dir"]
    assert isinstance(quick_dir, Path)

    required = manifest.get("required_quick_report_files")
    if not isinstance(required, list) or not required:
        required = [
            "appf1_seed_range_variability.csv",
            "appf2_data_figure_compatibility_matrix.csv",
            "appf3_related_work_radar_scores.csv",
            "appf4_mechanism_ablation_context.csv",
            "appf5_mechanism_energy_breakdown.csv",
            "appf6_det_sparse_sweep_phase4_basis.csv",
            "compliance_report.json",
            "fig3_phase4_runtime_accuracy_boundary.csv",
            "fig4_runtime_accuracy_pareto.csv",
            "fig5_bounded_sensitivity_current_basis.csv",
            "fig6_broad_scaling_flow_buffer_current_basis.csv",
            "fig7_device_context.csv",
            "fig8_holdout_claim_boundary.csv",
            "final_numbering_mapping.csv",
        ]
    for filename in required:
        rel_path = quick_dir / str(filename)
        if not (report.root / rel_path).is_file():
            report.add(f"missing quick-report input: {rel_path.as_posix()}")


def _check_suds_inputs(report: Report, manifest: dict[str, Any]) -> None:
    paths = _public_paths(manifest)
    phase_dir = paths["phase_dir"]
    report_data_dir = paths["report_data_dir"]
    assert isinstance(phase_dir, Path)
    assert isinstance(report_data_dir, Path)

    required_phase_files = manifest.get("required_phase_summary_files")
    if not isinstance(required_phase_files, list) or not required_phase_files:
        required_phase_files = [
            "phase_b/phase_b_summary.json",
            "phase_c/phase_c_summary.json",
            "phase_d/phase_d_summary.json",
            "phase_e/phase_e_summary.json",
            "phase_f/phase_f_summary.json",
            "slack_manifest.json",
        ]
    for rel_name in required_phase_files:
        rel_path = phase_dir / str(rel_name)
        if not (report.root / rel_path).is_file():
            report.add(f"missing SUDS phase input: {rel_path.as_posix()}")

    required_report_data = manifest.get("required_report_data_files")
    if not isinstance(required_report_data, list) or not required_report_data:
        required_report_data = ["suds_ablation_matrix_20260511_maxq.csv"]
    for rel_name in required_report_data:
        rel_path = report_data_dir / str(rel_name)
        if not (report.root / rel_path).is_file():
            report.add(f"missing SUDS report-data input: {rel_path.as_posix()}")

    for rel_path in sorted(_list_field(manifest, "allowed_binary_files")):
        if not (report.root / rel_path).is_file():
            report.add(f"missing allowed AI schematic source asset: {rel_path}")


def _check_registry_metadata(report: Report, manifest: dict[str, Any]) -> None:
    paths = _public_paths(manifest)
    pack_dir = paths["pack_dir"]
    assert isinstance(pack_dir, Path)
    registry_path = report.root / pack_dir / "figure_numbering_registry.csv"
    traceability_path = report.root / pack_dir / "figure_traceability.csv"
    if not registry_path.is_file() or not traceability_path.is_file():
        return

    registry_rows = _csv_rows(registry_path)
    trace_rows = _csv_rows(traceability_path)
    registry_ids = [row.get("figure_id", "") for row in registry_rows if row.get("numbering_status") == "active"]
    expected_main = [str(item) for item in manifest.get("expected_main_figures", [])]
    expected_appendix = [str(item) for item in manifest.get("expected_appendix_figures", [])]
    if not expected_main:
        expected_main = [f"Fig{i}" for i in range(1, 7)] if _is_suds(manifest) else [f"Fig{i}" for i in range(1, 13)]
    if not expected_appendix:
        expected_appendix = [f"AppF{i}" for i in range(1, 5)] if _is_suds(manifest) else [f"AppF{i}" for i in range(1, 7)]
    for figure_id in expected_main + expected_appendix:
        if figure_id not in registry_ids:
            report.add(f"figure registry missing active figure_id: {figure_id}")
    trace_ids = {row.get("figure_id", "") for row in trace_rows}
    for figure_id in registry_ids:
        if figure_id not in trace_ids:
            report.add(f"traceability missing active figure_id: {figure_id}")


def _check_traceability_inputs(report: Report, manifest: dict[str, Any]) -> None:
    paths = _public_paths(manifest)
    pack_dir = paths["pack_dir"]
    quick_dir = paths["quick_dir"]
    assert isinstance(pack_dir, Path)
    assert isinstance(quick_dir, Path)
    traceability_path = report.root / pack_dir / "figure_traceability.csv"
    if not traceability_path.is_file():
        return
    if _is_suds(manifest):
        rendered_ids = manifest.get("rendered_by_public_make")
        if not isinstance(rendered_ids, list) or not rendered_ids:
            rendered_ids = ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "AppF1", "AppF2", "AppF3", "AppF4"]
        rendered_ids = [str(item) for item in rendered_ids]
        trace_by_id = {row.get("figure_id", ""): row for row in _csv_rows(traceability_path)}
        for figure_id in rendered_ids:
            row = trace_by_id.get(figure_id)
            if row is None:
                report.add(f"public SUDS traceability missing figure_id: {figure_id}")
                continue
            if row.get("render_command") != "make render-paper-figures":
                report.add(f"public SUDS render command mismatch for {figure_id}: {row.get('render_command')!r}")
            output = row.get("figure_file", "")
            if not output.startswith("build/rendered_figures/"):
                report.add(f"public SUDS render output should live under build/ for {figure_id}: {output}")
            try:
                outputs = json.loads(row.get("all_outputs", "{}"))
            except json.JSONDecodeError as exc:
                report.add(f"invalid all_outputs JSON for {figure_id}: {exc}")
                outputs = {}
            if not isinstance(outputs, dict) or not outputs:
                report.add(f"missing public SUDS all_outputs for {figure_id}")
            for rel_path in outputs.values():
                rel_text = str(rel_path)
                if not rel_text.startswith("build/rendered_figures/"):
                    report.add(f"public SUDS output should live under build/ for {figure_id}: {rel_text}")
            source_path = row.get("primary_source_path", "")
            if not source_path:
                report.add(f"public SUDS source path missing for {figure_id}")
            elif not (report.root / source_path).is_file():
                report.add(f"public SUDS source path does not exist for {figure_id}: {source_path}")
        return
    rendered_ids = manifest.get("rendered_by_public_make")
    if not isinstance(rendered_ids, list) or not rendered_ids:
        rendered_ids = ["Fig3", "Fig4", "Fig5", "Fig6", "Fig7", "Fig8", "AppF1", "AppF2", "AppF3", "AppF4", "AppF5", "AppF6"]
    rendered_ids = [str(item) for item in rendered_ids]
    trace_by_id = {row.get("figure_id", ""): row for row in _csv_rows(traceability_path)}
    for figure_id in rendered_ids:
        row = trace_by_id.get(figure_id)
        if row is None:
            report.add(f"public render traceability missing figure_id: {figure_id}")
            continue
        script_entry = row.get("script_entry", "")
        if script_entry and not (report.root / script_entry).is_file():
            report.add(f"public render script missing for {figure_id}: {script_entry}")
        if row.get("command") != "make render-paper-figures":
            report.add(f"public render command mismatch for {figure_id}: {row.get('command')!r}")
        output = row.get("figure_file", "")
        if not output.startswith("build/rendered_figures/"):
            report.add(f"public render output should live under build/ for {figure_id}: {output}")
        input_csvs = [item.strip() for item in row.get("input_csvs", "").split(";") if item.strip()]
        if not input_csvs:
            report.add(f"public render input missing for {figure_id}")
        for rel_path in input_csvs:
            if not (report.root / rel_path).is_file():
                report.add(f"public render input does not exist for {figure_id}: {rel_path}")
            if not Path(rel_path).as_posix().startswith(quick_dir.as_posix() + "/"):
                report.add(f"public render input must come from active quick reports for {figure_id}: {rel_path}")


def _check_review_metadata(report: Report, manifest: dict[str, Any]) -> None:
    paths = _public_paths(manifest)
    review_dir = paths["review_dir"]
    assert isinstance(review_dir, Path)
    review_manifest_path = report.root / review_dir / "review_manifest.json"
    if review_manifest_path.is_file():
        payload = json.loads(review_manifest_path.read_text(encoding="utf-8"))
        excluded = set(str(item) for item in payload.get("excluded", []))
        if "pre-rendered figure images" not in excluded and "pre-rendered final figure images" not in excluded:
            report.add("review_manifest.json should record exclusion of pre-rendered figure images")

    map_path = report.root / review_dir / "manuscript_evidence_map.csv"
    if map_path.is_file():
        rows = _csv_rows(map_path)
        expected_ids = [str(item) for item in manifest.get("rendered_by_public_make", [])]
        observed_ids = [row.get("figure_id", "") for row in rows]
        for figure_id in expected_ids:
            if figure_id not in observed_ids:
                report.add(f"manuscript evidence map missing public-rendered figure: {figure_id}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_checksums(report: Report, manifest: dict[str, Any]) -> None:
    checksum_path = report.root / CHECKSUMS_JSON
    if not checksum_path.is_file():
        report.add(f"missing required file: {CHECKSUMS_JSON.as_posix()}")
        return
    try:
        payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add(f"invalid checksum manifest: {exc}")
        return
    if payload.get("algorithm") != "sha256":
        report.add(f"checksum manifest algorithm mismatch: {payload.get('algorithm')!r}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        report.add("checksum manifest has no files")
        return

    listed: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            report.add("checksum manifest contains a non-object entry")
            continue
        rel = str(entry.get("path") or "")
        listed.add(rel)
        target = report.root / rel
        if not rel or not target.is_file():
            report.add(f"checksum target missing: {rel}")
            continue
        observed_size = target.stat().st_size
        if int(entry.get("size_bytes", -1)) != observed_size:
            report.add(
                f"checksum size mismatch for {rel}: "
                f"observed={observed_size} expected={entry.get('size_bytes')}"
            )
        if entry.get("sha256") != _sha256(target):
            report.add(f"checksum hash mismatch for {rel}")

    for path in _iter_files(report.root):
        rel = _rel(report.root, path)
        if rel == CHECKSUMS_JSON.as_posix() or _is_allowed_local_path(rel, manifest):
            continue
        if rel not in listed:
            report.add(f"file missing from checksum manifest: {rel}")


def _check_public_repro_inputs(report: Report, manifest: dict[str, Any]) -> None:
    if _is_suds(manifest):
        _check_suds_inputs(report, manifest)
    else:
        _check_quick_report_inputs(report, manifest)
    _check_registry_metadata(report, manifest)
    _check_traceability_inputs(report, manifest)
    _check_review_metadata(report, manifest)


def validate(root: Path) -> Report:
    report = Report(root=root.resolve())
    manifest = _load_manifest(report)
    _check_required(report, manifest)
    _check_banned_paths(report, manifest)
    _check_metadata_text(report, manifest)
    _check_freeze(report, manifest)
    _check_public_repro_inputs(report, manifest)
    _check_checksums(report, manifest)
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
