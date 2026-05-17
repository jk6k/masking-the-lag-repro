#!/usr/bin/env python3
"""Validate the canonical figure numbering registry for the active frozen pack."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK_PREFIX = "paper_figures_"
REGISTRY_NAME = "figure_numbering_registry.csv"
TRACEABILITY_NAME = "figure_traceability.csv"
EXPECTED_FIELDS = [
    "figure_id",
    "numbering_status",
    "manuscript_tier",
    "figure_family",
    "title",
    "canonical_stem",
    "source_kind",
    "source_record",
    "notes",
]
FIG_RE = re.compile(r"^Fig(\d+)$")
APP_RE = re.compile(r"^AppF(\d+)$")
PACK_FILE_RE = re.compile(r"^(Fig\d+|AppF\d+)_")
MECHANISM_FIGURES = {"AppF4", "AppF5", "AppF6"}
MECHANISM_RUN_TAG = "20260426_fuller_phase4_mechanism_basis_rerun"


@dataclass(frozen=True)
class RegistryRow:
    figure_id: str
    numbering_status: str
    manuscript_tier: str
    figure_family: str
    title: str
    canonical_stem: str
    source_kind: str
    source_record: str
    notes: str

    @property
    def is_active(self) -> bool:
        return self.numbering_status == "active"

    @property
    def is_reserved(self) -> bool:
        return self.numbering_status == "reserved_gap"


def _repo_path(path_str: str) -> Path:
    return (ROOT / path_str).resolve()


def _parse_sort_key(figure_id: str) -> tuple[int, int]:
    if match := FIG_RE.fullmatch(figure_id):
        return (0, int(match.group(1)))
    if match := APP_RE.fullmatch(figure_id):
        return (1, int(match.group(1)))
    raise ValueError(f"Unsupported figure_id format: {figure_id}")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_registry(path: Path) -> list[RegistryRow]:
    rows = _load_csv(path)
    if not rows:
        raise SystemExit(f"Empty registry: {path}")
    if list(rows[0].keys()) != EXPECTED_FIELDS:
        raise SystemExit(
            f"Registry fields mismatch for {path}: "
            f"expected {EXPECTED_FIELDS}, observed {list(rows[0].keys())}"
        )
    return [RegistryRow(**row) for row in rows]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_pack_dir(args: argparse.Namespace) -> Path:
    if args.pack_dir is not None:
        return args.pack_dir.resolve()
    freeze = _load_json(args.freeze_json.resolve())
    paper_figures_dir = str(freeze.get("paper_figures_dir") or "")
    if not paper_figures_dir:
        raise SystemExit(f"Missing paper_figures_dir in {args.freeze_json}")
    return _repo_path(paper_figures_dir)


def _extract_run_tag(pack_dir: Path) -> str:
    if not pack_dir.name.startswith(PACK_PREFIX):
        raise SystemExit(f"Unexpected frozen pack directory name: {pack_dir}")
    return pack_dir.name[len(PACK_PREFIX) :]


def _validate_registry_order(rows: list[RegistryRow], errors: list[str]) -> None:
    observed = [row.figure_id for row in rows]
    expected = [row.figure_id for row in sorted(rows, key=lambda row: _parse_sort_key(row.figure_id))]
    if observed != expected:
        errors.append(
            "Registry rows are not in canonical figure order: "
            f"observed={observed}, expected={expected}"
        )


def _validate_slot_coverage(rows: list[RegistryRow], errors: list[str]) -> None:
    main_rows = [row for row in rows if FIG_RE.fullmatch(row.figure_id)]
    active_nums = {int(FIG_RE.fullmatch(row.figure_id).group(1)) for row in main_rows if row.is_active}
    reserved_nums = {int(FIG_RE.fullmatch(row.figure_id).group(1)) for row in main_rows if row.is_reserved}
    if not active_nums:
        errors.append("Registry contains no active main-figure slots.")
        return
    max_main = max(active_nums | reserved_nums)
    covered = active_nums | reserved_nums
    missing = [num for num in range(1, max_main + 1) if num not in covered]
    if missing:
        errors.append(f"Main-figure numbering has unmanaged gaps: {missing}")
    appendix_rows = [row for row in rows if APP_RE.fullmatch(row.figure_id) and row.is_active]
    appendix_nums = sorted(int(APP_RE.fullmatch(row.figure_id).group(1)) for row in appendix_rows)
    expected_appendix = list(range(1, max(appendix_nums, default=0) + 1))
    if appendix_nums != expected_appendix:
        errors.append(
            "Appendix numbering is not contiguous from AppF1: "
            f"observed={appendix_nums}, expected={expected_appendix}"
        )


def _validate_registry_rows(rows: list[RegistryRow], pack_dir: Path, run_tag: str, errors: list[str]) -> None:
    seen_ids: set[str] = set()
    for row in rows:
        if row.figure_id in seen_ids:
            errors.append(f"Duplicate registry figure_id: {row.figure_id}")
            continue
        seen_ids.add(row.figure_id)
        try:
            _parse_sort_key(row.figure_id)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        source_path = _repo_path(row.source_record)
        if not source_path.is_file():
            errors.append(f"Missing source_record for {row.figure_id}: {row.source_record}")
        if row.is_active:
            if row.manuscript_tier not in {"main", "appendix"}:
                errors.append(f"Active figure has invalid manuscript_tier for {row.figure_id}: {row.manuscript_tier}")
            if not row.title:
                errors.append(f"Active figure is missing title: {row.figure_id}")
            if not row.canonical_stem:
                errors.append(f"Active figure is missing canonical_stem: {row.figure_id}")
            if row.canonical_stem and not row.canonical_stem.startswith(f"{row.figure_id}_"):
                errors.append(
                    f"canonical_stem must start with figure_id for {row.figure_id}: {row.canonical_stem}"
                )
            if row.source_kind not in {"manifest_json", "traceability_csv"}:
                errors.append(f"Unsupported active source_kind for {row.figure_id}: {row.source_kind}")
        elif row.is_reserved:
            if row.manuscript_tier != "unassigned":
                errors.append(f"Reserved slot must use manuscript_tier=unassigned: {row.figure_id}")
            if row.canonical_stem:
                errors.append(f"Reserved slot must not define canonical_stem: {row.figure_id}")
            if row.source_kind != "policy_doc":
                errors.append(f"Reserved slot must use source_kind=policy_doc: {row.figure_id}")
        else:
            errors.append(f"Unsupported numbering_status for {row.figure_id}: {row.numbering_status}")
    if pack_dir.name != f"{PACK_PREFIX}{run_tag}":
        errors.append(f"Pack directory/run_tag mismatch: {pack_dir.name} vs {run_tag}")


def _validate_manifest_backed_rows(
    rows: list[RegistryRow],
    pack_dir: Path,
    run_tag: str,
    errors: list[str],
) -> set[str]:
    registered_ids: set[str] = set()
    for row in rows:
        if not (row.is_active and row.source_kind == "manifest_json"):
            continue
        manifest_path = _repo_path(row.source_record)
        registered_ids.add(row.figure_id)
        if not manifest_path.is_file():
            continue
        payload = _load_json(manifest_path)
        if str(payload.get("figure_id") or "") != row.figure_id:
            errors.append(f"Manifest figure_id mismatch for {row.figure_id}: {manifest_path}")
        if str(payload.get("run_tag") or "") != run_tag:
            errors.append(f"Manifest run_tag mismatch for {row.figure_id}: {manifest_path}")
        frozen_outputs = payload.get("frozen_outputs") or {}
        if not isinstance(frozen_outputs, dict) or not frozen_outputs:
            errors.append(f"Manifest frozen_outputs missing for {row.figure_id}: {manifest_path}")
            continue
        for label, rel_path in frozen_outputs.items():
            out_path = _repo_path(str(rel_path))
            if not out_path.is_file():
                errors.append(f"Missing frozen output for {row.figure_id} ({label}): {rel_path}")
                continue
            if out_path.parent.resolve() != pack_dir.resolve():
                errors.append(f"Frozen output escapes pack_dir for {row.figure_id}: {rel_path}")
            if out_path.stem != row.canonical_stem:
                errors.append(
                    f"Manifest output stem mismatch for {row.figure_id}: "
                    f"{out_path.stem} vs {row.canonical_stem}"
                )
    return registered_ids


def _validate_traceability_backed_rows(
    rows: list[RegistryRow],
    traceability_path: Path,
    run_tag: str,
    mechanism_run_tag: str,
    errors: list[str],
) -> set[str]:
    trace_rows = _load_csv(traceability_path)
    trace_by_id = {str(row.get("figure_id") or ""): row for row in trace_rows}
    active_ids = {row.figure_id for row in rows if row.is_active}
    registered_ids: set[str] = set()
    for row in rows:
        if not (row.is_active and row.source_kind == "traceability_csv"):
            continue
        registered_ids.add(row.figure_id)
        trace_row = trace_by_id.get(row.figure_id)
        if trace_row is None:
            errors.append(f"Traceability row missing for {row.figure_id}")
            continue
        expected_run_tag = mechanism_run_tag if row.figure_id in MECHANISM_FIGURES else run_tag
        if str(trace_row.get("run_tag") or "") != expected_run_tag:
            errors.append(f"Traceability run_tag mismatch for {row.figure_id}")
        if str(trace_row.get("manuscript_tier") or "") != row.manuscript_tier:
            errors.append(f"Traceability manuscript_tier mismatch for {row.figure_id}")
        figure_file = str(trace_row.get("figure_file") or "")
        figure_path = _repo_path(figure_file)
        if not figure_path.is_file():
            errors.append(f"Missing traced figure file for {row.figure_id}: {figure_file}")
        else:
            if figure_path.stem != row.canonical_stem:
                errors.append(
                    f"Traceability stem mismatch for {row.figure_id}: "
                    f"{figure_path.stem} vs {row.canonical_stem}"
                )
    registry_traceability_sources = {
        row.source_record for row in rows if row.is_active and row.source_kind == "traceability_csv"
    }
    if registry_traceability_sources and registry_traceability_sources != {str(traceability_path.relative_to(ROOT))}:
        errors.append(
            "Traceability-backed rows must all point to the active figure_traceability.csv: "
            f"{sorted(registry_traceability_sources)}"
        )
    for figure_id in trace_by_id:
        if figure_id not in active_ids:
            errors.append(f"Traceability figure_id is unregistered in numbering registry: {figure_id}")
    return registered_ids


def _validate_manifest_inventory(
    rows: list[RegistryRow],
    pack_dir: Path,
    manifest_backed_ids: set[str],
    errors: list[str],
) -> None:
    registry_manifest_ids = {
        row.figure_id for row in rows if row.is_active and row.source_kind == "manifest_json"
    }
    for manifest_path in sorted(pack_dir.glob("*_manifest.json")):
        payload = _load_json(manifest_path)
        figure_id = str(payload.get("figure_id") or "")
        if figure_id not in registry_manifest_ids:
            errors.append(f"Pack manifest is not registered in figure_numbering_registry.csv: {manifest_path.name}")
            continue
        if figure_id not in manifest_backed_ids:
            errors.append(f"Pack manifest was not validated through registry linkage: {manifest_path.name}")


def _validate_pack_files(rows: list[RegistryRow], pack_dir: Path, errors: list[str]) -> None:
    active_rows = {row.figure_id: row for row in rows if row.is_active}
    for file_path in pack_dir.iterdir():
        if not file_path.is_file():
            continue
        match = PACK_FILE_RE.match(file_path.name)
        if match is None:
            continue
        figure_id = match.group(1)
        row = active_rows.get(figure_id)
        if row is None:
            errors.append(f"Pack contains file for unregistered or reserved figure_id: {file_path.name}")
            continue
        if row.canonical_stem and not file_path.name.startswith(row.canonical_stem):
            errors.append(
                f"Pack file stem does not match registry canonical_stem for {figure_id}: {file_path.name}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the canonical figure numbering registry.")
    parser.add_argument(
        "--pack_dir",
        type=Path,
        default=None,
        help="Optional figure pack directory. Defaults to the paper_sync current freeze.",
    )
    parser.add_argument(
        "--freeze_json",
        type=Path,
        default=ROOT / "experiments/results/paper_sync/current_freeze.json",
        help="Freeze pointer used when --pack_dir is omitted.",
    )
    args = parser.parse_args()

    pack_dir = _resolve_pack_dir(args)
    run_tag = _extract_run_tag(pack_dir)
    registry_path = pack_dir / REGISTRY_NAME
    traceability_path = pack_dir / TRACEABILITY_NAME
    if not registry_path.is_file():
        raise SystemExit(f"Missing registry: {registry_path}")
    if not traceability_path.is_file():
        raise SystemExit(f"Missing traceability file: {traceability_path}")

    rows = _load_registry(registry_path)
    errors: list[str] = []
    _validate_registry_order(rows, errors)
    _validate_slot_coverage(rows, errors)
    _validate_registry_rows(rows, pack_dir, run_tag, errors)
    manifest_backed_ids = _validate_manifest_backed_rows(rows, pack_dir, run_tag, errors)
    _validate_manifest_inventory(rows, pack_dir, manifest_backed_ids, errors)
    freeze_payload = _load_json(args.freeze_json.resolve()) if args.freeze_json.is_file() else {}
    mechanism_run_tag = str(freeze_payload.get("mechanism_evidence_tag") or MECHANISM_RUN_TAG)
    _validate_traceability_backed_rows(rows, traceability_path, run_tag, mechanism_run_tag, errors)
    _validate_pack_files(rows, pack_dir, errors)

    if errors:
        for item in errors:
            print(f"[figure-numbering-check][error] {item}", file=sys.stderr)
        return 1

    active_rows = [row for row in rows if row.is_active]
    reserved_rows = [row for row in rows if row.is_reserved]
    main_active = [row.figure_id for row in active_rows if FIG_RE.fullmatch(row.figure_id)]
    appendix_active = [row.figure_id for row in active_rows if APP_RE.fullmatch(row.figure_id)]
    reserved_main = [row.figure_id for row in reserved_rows if FIG_RE.fullmatch(row.figure_id)]
    print(
        "[figure-numbering-check] ok "
        f"run_tag={run_tag} active={len(active_rows)} reserved={len(reserved_rows)} "
        f"main_active={main_active[0]}..{main_active[-1]} appendix_active={appendix_active[0]}..{appendix_active[-1]}"
    )
    print(
        "[figure-numbering-check] reserved_main_slots="
        + (",".join(reserved_main) if reserved_main else "none")
    )
    print(
        "[figure-numbering-check] sources="
        f"manifest_json={sum(1 for row in active_rows if row.source_kind == 'manifest_json')} "
        f"traceability_csv={sum(1 for row in active_rows if row.source_kind == 'traceability_csv')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
