#!/usr/bin/env python3
"""Prepare conservative accuracy-source overlays and scaffolds for true-SC E0-E6."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml

try:
    from .fuller_phase1_registry import (
        normalize_variant_descriptor,
        variant_lookup_by_internal_id,
    )
except ImportError:
    from fuller_phase1_registry import (  # type: ignore
        normalize_variant_descriptor,
        variant_lookup_by_internal_id,
    )

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "configs" / "true_sc_e0_e6_canonical_bundle_20260421.yaml"
LEGACY_DEFAULT_BUNDLE_TAG = "20260418_true_sc_e0_e6_canonical"
LEGACY_DEFAULT_ACCURACY_LAUNCH_ROOT = (
    ROOT / "experiments" / "results" / "report_data" / "true_sc_e0_e6_accuracy_launch_20260418"
)
RUNTIME_SMOKE_TIER = "runtime_smoke"
LEGACY_RUNTIME_SMOKE_OVERLAY_TAG = "20260419_true_sc_e0_e6_runtime_smoke_current"
RUNTIME_SMOKE_EXPERIMENT_IDS = ("e1", "e2", "e3", "e4", "e5", "e6")
PATCHABLE_CONTEXT_FIELDS = (
    "split",
    "experiment_id",
    "det_policy",
    "det_k_signature",
    "det_k_global",
    "sparse_tau_global",
    "sparse_active_fraction",
    "gaussian_noise_std_ref",
    "crosstalk_alpha_ref",
    "config_snapshot",
)
DATE_TOKEN_PATTERN = re.compile(r"(20\d{2})(\d{2})(\d{2})")
WRITE_SCOPE_FULL = "full"
WRITE_SCOPE_REPORT_DATA_ONLY = "report-data-only"


def _bundle_tag(bundle_path: Path, bundle: dict[str, Any]) -> str:
    meta = bundle.get("meta") or {}
    candidate = str(meta.get("tag") or "").strip()
    if candidate:
        return candidate
    return LEGACY_DEFAULT_BUNDLE_TAG


def _dated_tag_parts(tag: str, *, fallback_date: str = "20260418") -> tuple[str, str]:
    cleaned = str(tag or "").strip()
    match = DATE_TOKEN_PATTERN.search(cleaned)
    if not match:
        return fallback_date, cleaned
    date_token = f"{match.group(1)}{match.group(2)}{match.group(3)}"
    prefix = f"{date_token}_"
    if cleaned.startswith(prefix):
        return date_token, cleaned[len(prefix) :]
    return date_token, cleaned.replace(prefix, "", 1)


def _derive_runtime_smoke_overlay_tag(bundle_path: Path, bundle: dict[str, Any]) -> str:
    canonical_tag = _bundle_tag(bundle_path, bundle)
    if canonical_tag == LEGACY_DEFAULT_BUNDLE_TAG:
        return LEGACY_RUNTIME_SMOKE_OVERLAY_TAG
    if canonical_tag.endswith("_canonical"):
        return canonical_tag[: -len("_canonical")] + "_runtime_smoke_current"
    return f"{canonical_tag}_runtime_smoke_current"


def _canonical_prefix(bundle_path: Path, bundle: dict[str, Any]) -> tuple[str, str]:
    canonical_tag = _bundle_tag(bundle_path, bundle)
    date_token, stem = _dated_tag_parts(canonical_tag)
    return date_token, stem.removesuffix("_canonical") or "true_sc_e0_e6"


def _derived_contract_token(bundle_path: Path, bundle: dict[str, Any]) -> str:
    _, canonical_prefix = _canonical_prefix(bundle_path, bundle)
    overlay_tag = _derive_runtime_smoke_overlay_tag(bundle_path, bundle)
    overlay_date, _ = _dated_tag_parts(overlay_tag, fallback_date="20260419")
    return f"{canonical_prefix}_contract_{overlay_date}"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return ROOT / path


def _sanitize_token(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value).strip()
    )
    return cleaned.strip("._") or "unnamed"


def _launch_artifact_stem(bundle_path: Path, bundle: dict[str, Any]) -> str:
    meta = bundle.get("meta") or {}
    candidate = str(meta.get("tag") or bundle_path.stem).strip()
    return _sanitize_token(candidate)


def _surface_display_name(meta: dict[str, Any], default: str) -> str:
    return str(meta.get("surface_display_name") or default).strip()


def _bundle_tool_command(meta: dict[str, Any]) -> str:
    return str(
        meta.get("bundle_tool_entrypoint")
        or "python3 experiments/tools/prepare_true_sc_e0_e6_bundle.py"
    ).strip()


def _phase1_runner_command(meta: dict[str, Any]) -> str:
    return str(meta.get("phase1_runner_entrypoint") or "python3 experiments/tools/phase1_runner.py").strip()


def _source_tool_command(meta: dict[str, Any]) -> str:
    return str(
        meta.get("source_tool_entrypoint")
        or "python3 experiments/tools/prepare_true_sc_e0_e6_accuracy_sources.py"
    ).strip()


def _variant_lookup(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants = bundle.get("variants") or []
    if not isinstance(variants, list):
        return {}
    return variant_lookup_by_internal_id(variants)


def _variant_descriptor_for_internal_id(
    experiment_id: str,
    *,
    variant_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    descriptor = variant_lookup.get(str(experiment_id or "").strip().upper())
    if descriptor is not None:
        return descriptor
    return normalize_variant_descriptor({"internal_experiment_id": experiment_id})


def _lane_stub_for_experiment(
    experiment_id: str,
    *,
    variant_lookup: dict[str, dict[str, Any]],
) -> str:
    descriptor = _variant_descriptor_for_internal_id(experiment_id, variant_lookup=variant_lookup)
    return str(descriptor.get("config_stub") or str(experiment_id).lower()).strip().lower()


def _lane_display_for_experiment(
    experiment_id: str,
    *,
    variant_lookup: dict[str, dict[str, Any]],
) -> str:
    descriptor = _variant_descriptor_for_internal_id(experiment_id, variant_lookup=variant_lookup)
    variant_id = str(descriptor.get("variant_id") or "").strip().upper()
    internal_id = str(descriptor.get("internal_experiment_id") or experiment_id).strip().upper()
    if variant_id and variant_id != internal_id:
        return f"{variant_id} ({internal_id})"
    return internal_id


def _runtime_smoke_overlay_paths(
    workspace_root: Path,
    *,
    bundle_path: Path,
    bundle: dict[str, Any],
) -> dict[str, Path]:
    overlay_tag = _derive_runtime_smoke_overlay_tag(bundle_path, bundle)
    overlay_date, overlay_stem = _dated_tag_parts(overlay_tag, fallback_date="20260419")
    generated_root = workspace_root / "experiments" / "results" / "generated_configs" / overlay_tag
    report_dir = workspace_root / "experiments" / "results" / "report_data"
    docs_dir = workspace_root / "docs" / "reports"
    return {
        "overlay_tag": Path(overlay_tag),
        "generated_config_dir": generated_root,
        "switch_matrix_csv": report_dir / f"{overlay_stem}_switch_matrix_{overlay_date}.csv",
        "switch_matrix_json": report_dir / f"{overlay_stem}_switch_matrix_{overlay_date}.json",
        "accuracy_contract_csv": report_dir / f"{overlay_stem}_accuracy_contract_{overlay_date}.csv",
        "accuracy_contract_json": report_dir / f"{overlay_stem}_accuracy_contract_{overlay_date}.json",
        "accuracy_scaffold_csv": report_dir / f"{overlay_stem}_accuracy_row_scaffold_{overlay_date}.csv",
        "preflight_json": report_dir / f"{overlay_stem}_preflight_{overlay_date}.json",
        "preflight_note_md": docs_dir / f"{overlay_date}_{overlay_stem}_preflight_note.md",
    }


def _resolve_bundle_note_date(meta: dict[str, Any], bundle_path: Path) -> str:
    for candidate in (meta.get("tag"), bundle_path.stem):
        match = DATE_TOKEN_PATTERN.search(str(candidate or "").strip())
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return "unknown"


def _read_note_date(note_path: Path | None) -> str:
    if note_path is None or not note_path.exists():
        return ""
    for line in note_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("Date: `") and stripped.endswith("`"):
            return stripped[len("Date: `") : -1]
    return ""


def _resolve_launch_artifact_paths(
    *,
    bundle_path: Path,
    bundle: dict[str, Any],
    evidence_tier: str,
) -> tuple[Path, Path, Path]:
    stem = _launch_artifact_stem(bundle_path, bundle)
    tier = _sanitize_token(evidence_tier)
    launch_root = ROOT / "experiments" / "results" / "report_data" / f"{stem}_accuracy_launch_{tier}"
    manifest_json = launch_root / f"{stem}_accuracy_launch_manifest_{tier}.json"
    note_md = ROOT / "docs" / "reports" / f"{stem}_accuracy_launch_prep_note_{tier}.md"
    return launch_root, manifest_json, note_md


def _read_note_launch_prep_reference(note_path: Path | None) -> str:
    if note_path is None or not note_path.exists():
        return ""
    for line in note_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("`")
            and stripped.endswith("`")
            and "accuracy_launch_prep_note" in stripped
        ):
            return stripped[1:-1]
    return ""


def _resolve_accuracy_source_paths(
    *,
    workspace_root: Path,
    generated_config_dir: Path,
    bundle_path: Path,
    bundle: dict[str, Any],
) -> dict[str, Path]:
    report_dir = workspace_root / "experiments" / "results" / "report_data"
    canonical_date, canonical_prefix = _canonical_prefix(bundle_path, bundle)
    e0_prefix = canonical_prefix.replace("_e0_e6", "_e0")
    overlay_tag = _derive_runtime_smoke_overlay_tag(bundle_path, bundle)
    overlay_date, overlay_stem = _dated_tag_parts(overlay_tag, fallback_date="20260419")
    lane_prefix = overlay_stem.removesuffix("_current")
    note_md = (
        workspace_root
        / "docs"
        / "reports"
        / f"{canonical_date}_{canonical_prefix}_accuracy_source_prep_note.md"
    )
    merged_bundle_overlay = workspace_root / "configs" / f"{overlay_stem}_bundle_{overlay_date}.yaml"
    merged_template_overlay = (
        workspace_root / "configs" / f"{overlay_stem}_template_{overlay_date}.yaml"
    )
    runtime_smoke_launch_root, runtime_smoke_launch_manifest_json, runtime_smoke_launch_note_md = (
        _resolve_launch_artifact_paths(
            bundle_path=merged_bundle_overlay,
            bundle={"meta": {"tag": overlay_tag}},
            evidence_tier=RUNTIME_SMOKE_TIER,
        )
    )
    return {
        "report_dir": report_dir,
        "lane_scaffold_dir": report_dir / f"{canonical_prefix}_accuracy_lane_scaffolds_{canonical_date}",
        "runtime_smoke_lane_dir": report_dir / f"{lane_prefix}_lane_candidates_{overlay_date}",
        "e0_backfill_patch_csv": report_dir / f"{e0_prefix}_accuracy_context_backfill_patch_{canonical_date}.csv",
        "e0_backfill_candidate_csv": report_dir / f"{e0_prefix}_accuracy_context_backfill_candidate_{canonical_date}.csv",
        "manifest_json": report_dir / f"{canonical_prefix}_accuracy_source_prep_manifest_{canonical_date}.json",
        "note_md": note_md,
        "e0_overlay_config": generated_config_dir / "e0_accuracy_backfill.yaml",
        "merged_candidate_csv": report_dir / f"{canonical_prefix}_current_runtime_smoke_candidate_{overlay_date}.csv",
        "merged_template_overlay": merged_template_overlay,
        "merged_bundle_overlay": merged_bundle_overlay,
        "runtime_smoke_lane_status_csv": report_dir / f"{lane_prefix}_lane_status_{overlay_date}.csv",
        "runtime_smoke_launch_root": runtime_smoke_launch_root,
        "runtime_smoke_launch_manifest_json": runtime_smoke_launch_manifest_json,
        "runtime_smoke_launch_note_md": runtime_smoke_launch_note_md,
    }

def _runtime_smoke_lane_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for child in path.iterdir()
        if child.is_dir() and (child / "annotated_accuracy.csv").exists()
    )


def _classify_accuracy_launch_root_candidate(
    candidate: Path,
    *,
    derived: Path,
    legacy: Path,
) -> str:
    if candidate == derived:
        return "derived"
    if candidate == legacy:
        return "legacy"
    name = candidate.name
    if (
        name.startswith("true_sc_e0_e6_runtime_smoke_recovery_")
        and name.endswith("_combined")
    ):
        return "runtime_smoke_recovery_combined"
    if (
        name.startswith("true_sc_e0_e6_runtime_smoke_recovery_")
        and "_run" in name
    ):
        return "runtime_smoke_recovery_run"
    if name.startswith("true_sc_all_target_postrepair_revalidation_"):
        return "postrepair_revalidation"
    return "other"


def _accuracy_launch_root_candidates(
    *,
    workspace_root: Path,
    bundle_path: Path,
    bundle: dict[str, Any],
) -> tuple[Path, Path, list[Path]]:
    report_dir = workspace_root / "experiments" / "results" / "report_data"
    derived = report_dir / f"{_launch_artifact_stem(bundle_path, bundle)}_accuracy_launch_{RUNTIME_SMOKE_TIER}"
    combined_roots = sorted(report_dir.glob("true_sc_e0_e6_runtime_smoke_recovery_*_combined"))
    recovery_run_roots = sorted(report_dir.glob("true_sc_e0_e6_runtime_smoke_recovery_*_run*"))
    postrepair_run_roots = sorted(report_dir.glob("true_sc_all_target_postrepair_revalidation_*"))
    legacy = report_dir / LEGACY_DEFAULT_ACCURACY_LAUNCH_ROOT.name

    candidates = [
        derived,
        *reversed(combined_roots),
        *reversed(recovery_run_roots),
        *reversed(postrepair_run_roots),
        legacy,
    ]

    deduped_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in deduped_candidates:
            continue
        deduped_candidates.append(candidate)

    return derived, legacy, deduped_candidates


def _resolve_accuracy_launch_root_details(
    *,
    workspace_root: Path,
    bundle_path: Path,
    bundle: dict[str, Any],
    accuracy_launch_root: Path | None,
) -> dict[str, Any]:
    if accuracy_launch_root is not None:
        selected_lane_count = (
            _runtime_smoke_lane_row_count(accuracy_launch_root)
            if accuracy_launch_root.exists()
            else 0
        )
        return {
            "selected_root": accuracy_launch_root,
            "selection_mode": "explicit_override",
            "selected_root_lane_count": selected_lane_count,
            "candidates": [
                {
                    "rank": 0,
                    "path": str(accuracy_launch_root),
                    "family": "explicit_override",
                    "exists": accuracy_launch_root.exists(),
                    "runtime_smoke_lane_row_count": selected_lane_count,
                    "selected": True,
                }
            ],
        }

    derived, legacy, candidates = _accuracy_launch_root_candidates(
        workspace_root=workspace_root,
        bundle_path=bundle_path,
        bundle=bundle,
    )

    best_populated_root: Path | None = None
    best_lane_count = -1
    first_existing_root: Path | None = None
    candidate_summaries: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates):
        exists = candidate.exists()
        lane_count = _runtime_smoke_lane_row_count(candidate) if exists else 0
        if exists and first_existing_root is None:
            first_existing_root = candidate
        if exists and lane_count > best_lane_count:
            best_lane_count = lane_count
            best_populated_root = candidate
        candidate_summaries.append(
            {
                "rank": rank,
                "path": str(candidate),
                "family": _classify_accuracy_launch_root_candidate(
                    candidate,
                    derived=derived,
                    legacy=legacy,
                ),
                "exists": exists,
                "runtime_smoke_lane_row_count": lane_count,
                "selected": False,
            }
        )

    if best_populated_root is not None and best_lane_count > 0:
        selected_root = best_populated_root
        selection_mode = "best_lane_coverage"
    elif first_existing_root is not None:
        selected_root = first_existing_root
        selection_mode = "first_existing_fallback"
    else:
        selected_root = derived
        selection_mode = "derived_default"

    selected_root_str = str(selected_root)
    selected_lane_count = 0
    for summary in candidate_summaries:
        if summary["path"] != selected_root_str:
            continue
        summary["selected"] = True
        selected_lane_count = int(summary["runtime_smoke_lane_row_count"])
        break

    return {
        "selected_root": selected_root,
        "selection_mode": selection_mode,
        "selected_root_lane_count": selected_lane_count,
        "candidates": candidate_summaries,
    }


def _resolve_accuracy_launch_root(
    *,
    workspace_root: Path,
    bundle_path: Path,
    bundle: dict[str, Any],
    accuracy_launch_root: Path | None,
) -> Path:
    resolution = _resolve_accuracy_launch_root_details(
        workspace_root=workspace_root,
        bundle_path=bundle_path,
        bundle=bundle,
        accuracy_launch_root=accuracy_launch_root,
    )
    return resolution["selected_root"]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _normalize_bool_str(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return "True"
    if text in {"false", "0", "no"}:
        return "False"
    return "True" if bool(value) else "False"


def _path_exists(path_value: str | Path | None) -> bool:
    path = _resolve_repo_path(path_value)
    return bool(path and path.exists())


def _resolve_config_snapshot_contract_state(
    *,
    expected_config_snapshot: str,
    selected_config_snapshot: str,
) -> str:
    expected_snapshot = str(expected_config_snapshot or "").strip()
    selected_snapshot = str(selected_config_snapshot or "").strip()
    expected_exists = _path_exists(expected_snapshot)
    selected_exists = _path_exists(selected_snapshot)

    if expected_snapshot and selected_snapshot and expected_snapshot != selected_snapshot:
        if expected_exists and selected_exists:
            return "expected_materialized_selected_historical"
        if expected_exists:
            return "expected_materialized_selected_missing"
        if selected_exists:
            return "future_placeholder_selected_historical"
        return "future_placeholder_selected_missing"
    if expected_snapshot:
        return "expected_materialized" if expected_exists else "expected_missing"
    if selected_snapshot:
        return "selected_present" if selected_exists else "selected_missing"
    return "unspecified"


def _contract_snapshot_contract_state(contract_row: dict[str, str]) -> str:
    explicit_state = str(contract_row.get("config_snapshot_contract_state") or "").strip()
    if explicit_state:
        return explicit_state
    return _resolve_config_snapshot_contract_state(
        expected_config_snapshot=str(contract_row.get("expected_config_snapshot") or ""),
        selected_config_snapshot=str(contract_row.get("selected_config_snapshot") or ""),
    )


def _contract_snapshot_exists_flag(
    contract_row: dict[str, str],
    *,
    field_name: str,
) -> str:
    explicit_value = str(contract_row.get(field_name) or "").strip()
    if explicit_value:
        return _normalize_bool_str(explicit_value)

    snapshot_field = field_name.removesuffix("_exists")
    snapshot_value = str(contract_row.get(snapshot_field) or "").strip()
    return _normalize_bool_str(_path_exists(snapshot_value))


def _summarize_config_snapshot_contract_rows(
    contract_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    summary_rows: list[dict[str, str]] = []
    state_counts: dict[str, int] = {}
    for row in contract_rows:
        state = _contract_snapshot_contract_state(row)
        summary_rows.append(
            {
                "experiment_id": str(row.get("experiment_id") or ""),
                "row_role": str(row.get("row_role") or ""),
                "row_status": str(row.get("row_status") or ""),
                "accuracy_context_run_id": str(row.get("accuracy_context_run_id") or ""),
                "selected_source_run_id": str(row.get("selected_source_run_id") or ""),
                "expected_config_snapshot": str(row.get("expected_config_snapshot") or ""),
                "expected_config_snapshot_exists": _contract_snapshot_exists_flag(
                    row,
                    field_name="expected_config_snapshot_exists",
                ),
                "selected_config_snapshot": str(row.get("selected_config_snapshot") or ""),
                "selected_config_snapshot_exists": _contract_snapshot_exists_flag(
                    row,
                    field_name="selected_config_snapshot_exists",
                ),
                "config_snapshot_contract_state": state,
            }
        )
        state_counts[state] = int(state_counts.get(state) or 0) + 1
    return summary_rows, state_counts


def _normalize_source_run_id(row: dict[str, Any]) -> str:
    source_run_id = str(row.get("source_run_id") or "").strip()
    if source_run_id:
        return source_run_id
    return str(row.get("run_id") or "").strip()


def _load_contract_rows(path: Path) -> list[dict[str, str]]:
    _, rows = _read_csv(path)
    return rows


def _contract_scaffold_to_row(contract_row: dict[str, str]) -> dict[str, str]:
    snapshot_state = _contract_snapshot_contract_state(contract_row)
    return {
        "run_id": str(contract_row.get("accuracy_context_run_id") or ""),
        "source_run_id": str(contract_row.get("accuracy_context_run_id") or ""),
        "variant_id": str(contract_row.get("variant_id") or ""),
        "experiment_id": str(contract_row.get("experiment_id") or ""),
        "internal_experiment_id": str(
            contract_row.get("internal_experiment_id") or contract_row.get("experiment_id") or ""
        ),
        "public_module_stack": str(contract_row.get("public_module_stack") or ""),
        "baseline": _normalize_bool_str(contract_row.get("expected_baseline")),
        "device": "mps",
        "measurement_window": "",
        "model": "mobilevit_s",
        "split": str(contract_row.get("expected_split") or ""),
        "workload": str(contract_row.get("expected_workload") or ""),
        "seed": str(contract_row.get("expected_seed") or ""),
        "config_snapshot": str(contract_row.get("expected_config_snapshot") or ""),
        "det_policy": str(contract_row.get("expected_det_policy") or ""),
        "det_k_signature": str(contract_row.get("expected_det_k_signature") or ""),
        "det_k_global": str(contract_row.get("expected_det_k_global") or ""),
        "sparse_tau_global": str(contract_row.get("expected_sparse_tau_global") or ""),
        "sparse_active_fraction": str(contract_row.get("expected_sparse_active_fraction") or ""),
        "gaussian_noise_std_ref": str(
            contract_row.get("expected_gaussian_noise_std_ref") or ""
        ),
        "crosstalk_alpha_ref": str(contract_row.get("expected_crosstalk_alpha_ref") or ""),
        "top1": "",
        "top5": "",
        "top1_delta": "",
        "top5_delta": "",
        "measured_pass_elapsed_s": "",
        "measured_processed_samples": "",
        "latency_ms_per_sample": "",
        "notes": (
            f"true_sc_e0_e6_contract_scaffold:{contract_row.get('lane_label') or ''};"
            f"row_role:{contract_row.get('row_role') or ''};"
            f"row_status:{contract_row.get('row_status') or ''};"
            f"config_snapshot_contract_state:{snapshot_state};"
            "fill_only_no_new_measurement"
        ),
    }


def _find_source_row(
    source_rows: list[dict[str, str]],
    *,
    selected_source_run_id: str,
    expected_baseline: str,
    selected_row_experiment_id: str,
) -> int | None:
    for index, row in enumerate(source_rows):
        if _normalize_source_run_id(row) != selected_source_run_id:
            continue
        if _normalize_bool_str(row.get("baseline")) != _normalize_bool_str(expected_baseline):
            continue
        row_experiment = str(row.get("experiment_id") or "").strip()
        if selected_row_experiment_id and row_experiment != selected_row_experiment_id:
            continue
        return index
    return None


def _append_note(existing: str, suffix: str) -> str:
    base = str(existing or "").strip()
    if suffix in base:
        return base
    if not base:
        return suffix
    return f"{base}; {suffix}"


def _expected_context_value(
    *,
    field: str,
    contract_row: dict[str, str],
    workspace_root: Path,
) -> str:
    expected_value = str(contract_row.get(f"expected_{field}") or "")
    if field == "config_snapshot":
        context_run_id = str(
            contract_row.get("accuracy_context_run_id")
            or contract_row.get("selected_source_run_id")
            or ""
        ).strip()
        if context_run_id:
            expected_value = str(
                (
                    workspace_root
                    / "experiments"
                    / "results"
                    / "runs"
                    / context_run_id
                    / "config_snapshot.yaml"
                ).resolve()
            )
    return expected_value


def _prepare_e0_backfill(
    *,
    workspace_root: Path,
    contract_rows: list[dict[str, str]],
    source_fieldnames: list[str],
    source_rows: list[dict[str, str]],
    contract_token: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    patch_rows: list[dict[str, Any]] = []
    candidate_rows = [dict(row) for row in source_rows]
    for row in contract_rows:
        if str(row.get("experiment_id") or "") != "E0":
            continue
        selected_source_run_id = str(row.get("selected_source_run_id") or "")
        selected_row_experiment_id = str(row.get("selected_row_experiment_id") or "")
        expected_baseline = str(row.get("expected_baseline") or "")
        matched_index = _find_source_row(
            source_rows,
            selected_source_run_id=selected_source_run_id,
            expected_baseline=expected_baseline,
            selected_row_experiment_id=selected_row_experiment_id,
        )
        if matched_index is None:
            patch_rows.append(
                {
                    "row_role": row.get("row_role") or "",
                    "selected_source_run_id": selected_source_run_id,
                    "expected_baseline": expected_baseline,
                    "status": "source_row_not_found",
                    "patched_fields": "",
                }
            )
            continue

        target = candidate_rows[matched_index]
        patched_fields: list[str] = []
        patched_values: dict[str, str] = {}
        for field in PATCHABLE_CONTEXT_FIELDS:
            expected_value = _expected_context_value(
                field=field,
                contract_row=row,
                workspace_root=workspace_root,
            )
            current_value = str(target.get(field) or "")
            if current_value == expected_value:
                continue
            target[field] = expected_value
            patched_fields.append(field)
            patched_values[field] = expected_value
        target["accuracy_measurement_contract_note"] = _append_note(
            target.get("accuracy_measurement_contract_note", ""),
            f"context_backfill_overlay:{contract_token};no_new_measurement",
        )
        target["notes"] = _append_note(
            target.get("notes", ""),
            f"metadata_backfill_overlay:{contract_token}",
        )
        patch_rows.append(
            {
                "row_role": row.get("row_role") or "",
                "selected_source_run_id": selected_source_run_id,
                "selected_row_id": row.get("selected_row_id") or "",
                "expected_baseline": expected_baseline,
                "status": "patched" if patched_fields else "no_patch_needed",
                "patched_fields": ";".join(patched_fields),
                **{
                    f"patched_{field}": patched_values.get(
                        field,
                        str(row.get(f"expected_{field}") or ""),
                    )
                    for field in PATCHABLE_CONTEXT_FIELDS
                },
            }
        )

    fieldnames = list(source_fieldnames)
    for extra in ("accuracy_measurement_contract_note",):
        if extra not in fieldnames:
            fieldnames.append(extra)
    return patch_rows, candidate_rows


def _write_note(
    *,
    workspace_root: Path,
    note_path: Path,
    note_date: str,
    bundle: dict[str, Any],
    bundle_path: Path,
    source_csv: Path,
    e0_patch_csv: Path,
    e0_candidate_csv: Path,
    e0_overlay_config: Path,
    lane_scaffold_dir: Path,
    merged_candidate_csv: Path,
    merged_template_overlay: Path,
    merged_bundle_overlay: Path,
    runtime_smoke_launch_note_md: Path,
    manifest: dict[str, Any],
) -> None:
    meta = bundle.get("meta") or {}
    variant_lookup = _variant_lookup(bundle)
    surface_display_name = _surface_display_name(meta, "True SC E0-E6")
    snapshot_state_counts = manifest.get("config_snapshot_contract_counts") or {}
    snapshot_contract_rows = manifest.get("config_snapshot_contract_rows") or []
    lines = [
        f"# {surface_display_name} Accuracy Source Preparation Note",
        "",
        f"Date: `{note_date}`",
        f"Bundle: `{bundle_path}`",
        f"Source CSV: `{source_csv}`",
        f"E0 backfill patch CSV: `{e0_patch_csv}`",
        f"E0 backfill candidate CSV: `{e0_candidate_csv}`",
        f"E0 overlay config: `{e0_overlay_config}`",
        f"Lane scaffold dir: `{lane_scaffold_dir}`",
        f"Merged candidate CSV: `{merged_candidate_csv}`",
        f"Merged template overlay: `{merged_template_overlay}`",
        f"Merged bundle overlay: `{merged_bundle_overlay}`",
        "",
        "## Outcome",
        "",
        "- `E0` now has a fill-only metadata backfill overlay candidate built from the existing measured row.",
        "- `E1-E6` now have per-lane scaffold source CSVs with required context fields but no fabricated measured values.",
        "- any completed runtime-smoke lane now gets lifted into a reusable current-row candidate source and overlay config.",
        "- the merged runtime-smoke overlay bundle now writes into a runtime-smoke-specific artifact family while preserving canonical run-id lineage for context matching.",
        "- No historical row was relabeled to a different experiment id or source run id.",
        "",
        "## Manifest",
        "",
        f"- `e0_patch_count={manifest['e0_patch_count']}`",
        f"- `lane_scaffold_files={manifest['lane_scaffold_file_count']}`",
        f"- `runtime_smoke_lane_files={manifest['runtime_smoke_lane_file_count']}`",
        "",
        "## Config Snapshot Contract",
        "",
    ]
    if snapshot_state_counts:
        for state in sorted(snapshot_state_counts):
            lines.append(f"- `{state}`: `{snapshot_state_counts[state]}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Config Snapshot Rows",
            "",
        ]
    )
    if snapshot_contract_rows:
        for row in snapshot_contract_rows:
            lane_display = _lane_display_for_experiment(
                str(row.get("experiment_id") or ""),
                variant_lookup=variant_lookup,
            )
            lines.append(
                f"- `{lane_display}` `{row['row_role']}`: "
                f"snapshot_state=`{row['config_snapshot_contract_state']}` "
                f"expected_exists=`{row['expected_config_snapshot_exists']}` "
                f"selected_exists=`{row['selected_config_snapshot_exists']}`"
            )
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- `E0` overlay config replay passed: `{e0_overlay_config}`",
            "- runtime-smoke current rows are merged conservatively without relabeling them as analysis-grade evidence",
            "- the scaffold CSVs remain fill-only preparation artifacts, not substitute evidence",
            "- runtime-smoke launch-prep note path resolves to:",
            f"  `{runtime_smoke_launch_note_md}`",
            "",
            "## Direct Entrypoints",
            "",
            f"- Refresh prep artifacts: `{_source_tool_command(meta)} --bundle {bundle_path}`",
            f"- Re-run ASTRA overlay preflight: `{_phase1_runner_command(meta)} --config {e0_overlay_config}`",
            f"- Re-run bundle preflight against merged current-row source: `{_bundle_tool_command(meta)} --bundle {merged_bundle_overlay}`",
        ]
    )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_existing_source_prep(
    *,
    bundle_path: Path,
    accuracy_launch_root: Path | None = None,
    write_scope: str = WRITE_SCOPE_FULL,
) -> dict[str, Any]:
    workspace_root = _workspace_like_root(bundle_path)
    bundle = _load_yaml(bundle_path)
    variant_lookup = _variant_lookup(bundle)
    launch_root_resolution = _resolve_accuracy_launch_root_details(
        workspace_root=workspace_root,
        bundle_path=bundle_path,
        bundle=bundle,
        accuracy_launch_root=accuracy_launch_root,
    )
    resolved_accuracy_launch_root = launch_root_resolution["selected_root"]
    paths = bundle.get("paths") or {}
    template_path = _resolve_repo_path(paths.get("template_yaml"))
    if template_path is None or not template_path.exists():
        raise SystemExit(f"Missing template_yaml for bundle: {bundle_path}")
    template_cfg = _load_yaml(template_path)
    source_csv = _resolve_repo_path(((template_cfg.get("accuracy") or {}).get("source_csv")))
    contract_csv = _resolve_repo_path(paths.get("accuracy_contract_csv"))
    generated_config_dir = _resolve_repo_path(paths.get("generated_config_dir"))
    if (
        source_csv is None
        or not source_csv.exists()
        or contract_csv is None
        or not contract_csv.exists()
        or generated_config_dir is None
    ):
        raise SystemExit(f"Bundle is missing accuracy source dependencies: {bundle_path}")

    artifact_paths = _resolve_accuracy_source_paths(
        workspace_root=workspace_root,
        generated_config_dir=generated_config_dir,
        bundle_path=bundle_path,
        bundle=bundle,
    )
    manifest_json = artifact_paths["manifest_json"]
    note_md = artifact_paths["note_md"]
    expected_note_date = _resolve_bundle_note_date(bundle.get("meta") or {}, bundle_path)
    expected_launch_note_reference = str(artifact_paths["runtime_smoke_launch_note_md"])
    contract_rows = _load_contract_rows(contract_csv)
    config_snapshot_contract_rows, config_snapshot_contract_counts = (
        _summarize_config_snapshot_contract_rows(contract_rows)
    )
    expected_fields = {
        "bundle_path": str(bundle_path),
        "accuracy_launch_root": str(resolved_accuracy_launch_root),
        "accuracy_launch_root_selection_mode": launch_root_resolution["selection_mode"],
        "accuracy_launch_root_lane_count": launch_root_resolution["selected_root_lane_count"],
        "source_csv": str(source_csv),
        "contract_csv": str(contract_csv),
        "e0_backfill_patch_csv": str(artifact_paths["e0_backfill_patch_csv"]),
        "e0_backfill_candidate_csv": str(artifact_paths["e0_backfill_candidate_csv"]),
        "e0_overlay_config": str(artifact_paths["e0_overlay_config"]),
        "merged_candidate_csv": str(artifact_paths["merged_candidate_csv"]),
        "merged_template_overlay": str(artifact_paths["merged_template_overlay"]),
        "merged_bundle_overlay": str(artifact_paths["merged_bundle_overlay"]),
        "lane_scaffold_dir": str(artifact_paths["lane_scaffold_dir"]),
        "runtime_smoke_lane_status_csv": str(artifact_paths["runtime_smoke_lane_status_csv"]),
        "runtime_smoke_lane_dir": str(artifact_paths["runtime_smoke_lane_dir"]),
        "manifest_json": str(manifest_json),
        "note_md": str(note_md),
        "source_prep_note_date": expected_note_date,
        "runtime_smoke_launch_root": str(artifact_paths["runtime_smoke_launch_root"]),
        "runtime_smoke_launch_manifest_json": str(
            artifact_paths["runtime_smoke_launch_manifest_json"]
        ),
        "runtime_smoke_launch_note_md": expected_launch_note_reference,
    }
    expected_structured_fields = {
        "accuracy_launch_root_candidates": launch_root_resolution["candidates"],
        "config_snapshot_contract_rows": config_snapshot_contract_rows,
        "config_snapshot_contract_counts": config_snapshot_contract_counts,
    }

    issues: list[str] = []
    out_of_scope_issues: list[str] = []
    existing_payload: dict[str, Any] = {}
    if manifest_json.exists():
        existing_payload = _load_json(manifest_json)
    else:
        issues.append("missing_source_prep_manifest")
    merged_bundle_overlay = artifact_paths["merged_bundle_overlay"]
    merged_template_overlay = artifact_paths["merged_template_overlay"]
    if (
        bundle_path.resolve() == merged_bundle_overlay.resolve()
        and template_path.resolve() != merged_template_overlay.resolve()
    ):
        issues.append("bundle_template_yaml_not_current_overlay")

    if existing_payload:
        for field, expected_value in expected_fields.items():
            stored_value = str(existing_payload.get(field) or "").strip()
            if stored_value != str(expected_value).strip():
                issues.append(f"{field}_mismatch")
        for field, expected_value in expected_structured_fields.items():
            if existing_payload.get(field) != expected_value:
                issues.append(f"{field}_mismatch")
        lane_scaffold_files = existing_payload.get("lane_scaffold_files") or []
        if not isinstance(lane_scaffold_files, list):
            lane_scaffold_files = []
        if int(existing_payload.get("lane_scaffold_file_count") or 0) != len(lane_scaffold_files):
            issues.append("lane_scaffold_file_count_mismatch")
        runtime_lane_files = existing_payload.get("runtime_smoke_lane_files") or []
        if not isinstance(runtime_lane_files, list):
            runtime_lane_files = []
        if int(existing_payload.get("runtime_smoke_lane_file_count") or 0) != len(runtime_lane_files):
            issues.append("runtime_smoke_lane_file_count_mismatch")

    existing_note_date = _read_note_date(note_md)
    existing_launch_note_reference = _read_note_launch_prep_reference(note_md)
    note_issue_target = (
        out_of_scope_issues
        if write_scope == WRITE_SCOPE_REPORT_DATA_ONLY
        else issues
    )
    if not note_md.exists():
        note_issue_target.append("missing_source_prep_note")
    elif existing_note_date != expected_note_date:
        note_issue_target.append("source_prep_note_body_date_mismatch")
    if note_md.exists() and existing_launch_note_reference != expected_launch_note_reference:
        note_issue_target.append("launch_prep_note_reference_mismatch")

    return {
        **expected_fields,
        **expected_structured_fields,
        "manifest_json_exists": manifest_json.exists(),
        "note_exists": note_md.exists(),
        "write_scope": write_scope,
        "existing_note_date": existing_note_date,
        "existing_launch_prep_note_reference": existing_launch_note_reference,
        "status": "aligned" if not issues else "stale",
        "issues": list(dict.fromkeys(issues)),
        "out_of_scope_issues": list(dict.fromkeys(out_of_scope_issues)),
    }


def _workspace_like_root(bundle_path: Path) -> Path:
    try:
        bundle_path.resolve().relative_to(ROOT.resolve())
        return ROOT
    except ValueError:
        return bundle_path.resolve().parent


def _read_rows_if_exists(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    return _read_csv(path)


def _match_contract_runtime_row(
    row: dict[str, str],
    *,
    contract_row: dict[str, str],
) -> bool:
    expected_run_id = str(contract_row.get("accuracy_context_run_id") or "").strip()
    if not expected_run_id:
        return False
    row_run_id = str(row.get("run_id") or "").strip()
    row_source_run_id = _normalize_source_run_id(row)
    if row_run_id != expected_run_id and row_source_run_id != expected_run_id:
        return False
    if _normalize_bool_str(row.get("baseline")) != _normalize_bool_str(
        contract_row.get("expected_baseline")
    ):
        return False
    expected_experiment_id = str(contract_row.get("experiment_id") or "").strip()
    row_experiment_id = str(row.get("experiment_id") or "").strip()
    if expected_experiment_id and row_experiment_id and row_experiment_id != expected_experiment_id:
        return False
    return True


def _patch_runtime_row_context(
    row: dict[str, str],
    *,
    contract_row: dict[str, str],
    contract_token: str,
) -> dict[str, str]:
    patched = dict(row)
    for field in (
        "split",
        "workload",
        "seed",
        "config_snapshot",
        "det_policy",
        "det_k_signature",
        "det_k_global",
        "sparse_tau_global",
        "sparse_active_fraction",
        "gaussian_noise_std_ref",
        "crosstalk_alpha_ref",
    ):
        expected = str(contract_row.get(f"expected_{field}") or "").strip()
        if expected:
            patched[field] = expected
    patched["experiment_id"] = str(contract_row.get("experiment_id") or patched.get("experiment_id") or "")
    if str(contract_row.get("variant_id") or "").strip():
        patched["variant_id"] = str(contract_row.get("variant_id") or "")
    patched["internal_experiment_id"] = str(
        contract_row.get("internal_experiment_id") or contract_row.get("experiment_id") or ""
    )
    if str(contract_row.get("public_module_stack") or "").strip():
        patched["public_module_stack"] = str(contract_row.get("public_module_stack") or "")
    patched["notes"] = _append_note(
        patched.get("notes", ""),
        f"runtime_smoke_current_lane_candidate:{contract_token}",
    )
    return patched


def _collect_runtime_smoke_lane_rows(
    *,
    contract_rows: list[dict[str, str]],
    accuracy_launch_root: Path,
    contract_token: str,
    variant_lookup: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, str]]], list[str], list[dict[str, str]]]:
    runtime_rows_by_experiment: dict[str, list[dict[str, str]]] = {}
    runtime_fieldnames: list[str] = []
    status_rows: list[dict[str, str]] = []
    for experiment_id in sorted(
        {
            str(row.get("experiment_id") or "").strip()
            for row in contract_rows
            if str(row.get("experiment_id") or "").strip() and str(row.get("experiment_id") or "").strip() != "E0"
        }
    ):
        contract_group = [
            row for row in contract_rows if str(row.get("experiment_id") or "").strip() == experiment_id
        ]
        lane_descriptor = _variant_descriptor_for_internal_id(
            experiment_id,
            variant_lookup=variant_lookup,
        )
        lane_stub = str(lane_descriptor.get("config_stub") or experiment_id.lower()).strip().lower()
        annotated_csv = accuracy_launch_root / lane_stub / "annotated_accuracy.csv"
        fieldnames, rows = _read_rows_if_exists(annotated_csv)
        matched_rows: list[dict[str, str]] = []
        missing_roles: list[str] = []
        if rows:
            for contract_row in contract_group:
                matched = next(
                    (
                        _patch_runtime_row_context(
                            row,
                            contract_row=contract_row,
                            contract_token=contract_token,
                        )
                        for row in rows
                        if _match_contract_runtime_row(row, contract_row=contract_row)
                    ),
                    None,
                )
                if matched is None:
                    missing_roles.append(str(contract_row.get("row_role") or ""))
                    continue
                matched_rows.append(matched)
        else:
            missing_roles = [str(row.get("row_role") or "") for row in contract_group]
        if matched_rows:
            runtime_rows_by_experiment[experiment_id] = matched_rows
            for field in fieldnames:
                if field not in runtime_fieldnames:
                    runtime_fieldnames.append(field)
            for row in matched_rows:
                for field in row:
                    if field not in runtime_fieldnames:
                        runtime_fieldnames.append(field)
        status_rows.append(
            {
                "variant_id": str(lane_descriptor.get("variant_id") or ""),
                "experiment_id": experiment_id,
                "internal_experiment_id": str(
                    lane_descriptor.get("internal_experiment_id") or experiment_id
                ),
                "lane_stub": lane_stub,
                "annotated_csv": str(annotated_csv),
                "status": (
                    "matched_complete"
                    if len(matched_rows) == len(contract_group) and len(contract_group) > 0
                    else ("matched_partial" if matched_rows else "missing")
                ),
                "matched_row_count": str(len(matched_rows)),
                "expected_row_count": str(len(contract_group)),
                "missing_row_roles": ";".join(role for role in missing_roles if role),
            }
        )
    return runtime_rows_by_experiment, runtime_fieldnames, status_rows


def build_accuracy_source_artifacts(
    *,
    bundle_path: Path,
    accuracy_launch_root: Path | None = None,
    write_scope: str = WRITE_SCOPE_FULL,
) -> dict[str, Any]:
    workspace_root = _workspace_like_root(bundle_path)
    bundle = _load_yaml(bundle_path)
    variant_lookup = _variant_lookup(bundle)
    launch_root_resolution = _resolve_accuracy_launch_root_details(
        workspace_root=workspace_root,
        bundle_path=bundle_path,
        bundle=bundle,
        accuracy_launch_root=accuracy_launch_root,
    )
    resolved_accuracy_launch_root = launch_root_resolution["selected_root"]
    paths = bundle.get("paths") or {}
    template_path = _resolve_repo_path(paths.get("template_yaml"))
    if template_path is None or not template_path.exists():
        raise SystemExit(f"Missing template_yaml for bundle: {bundle_path}")
    template_cfg = _load_yaml(template_path)
    source_csv = _resolve_repo_path(((template_cfg.get("accuracy") or {}).get("source_csv")))
    contract_csv = _resolve_repo_path(paths.get("accuracy_contract_csv"))
    generated_config_dir = _resolve_repo_path(paths.get("generated_config_dir"))
    if (
        source_csv is None
        or not source_csv.exists()
        or contract_csv is None
        or not contract_csv.exists()
        or generated_config_dir is None
    ):
        raise SystemExit(f"Bundle is missing accuracy source dependencies: {bundle_path}")

    source_fieldnames, source_rows = _read_csv(source_csv)
    contract_rows = _load_contract_rows(contract_csv)
    artifact_paths = _resolve_accuracy_source_paths(
        workspace_root=workspace_root,
        generated_config_dir=generated_config_dir,
        bundle_path=bundle_path,
        bundle=bundle,
    )
    scaffold_dir = artifact_paths["lane_scaffold_dir"]
    runtime_lane_dir = artifact_paths["runtime_smoke_lane_dir"]
    e0_patch_csv = artifact_paths["e0_backfill_patch_csv"]
    e0_candidate_csv = artifact_paths["e0_backfill_candidate_csv"]
    manifest_json = artifact_paths["manifest_json"]
    note_md = artifact_paths["note_md"]
    e0_overlay_config = artifact_paths["e0_overlay_config"]
    merged_candidate_csv = artifact_paths["merged_candidate_csv"]
    merged_template_overlay = artifact_paths["merged_template_overlay"]
    merged_bundle_overlay = artifact_paths["merged_bundle_overlay"]
    runtime_status_csv = artifact_paths["runtime_smoke_lane_status_csv"]
    source_prep_note_date = _resolve_bundle_note_date(bundle.get("meta") or {}, bundle_path)
    config_snapshot_contract_rows, config_snapshot_contract_counts = (
        _summarize_config_snapshot_contract_rows(contract_rows)
    )
    write_generated_configs = write_scope == WRITE_SCOPE_FULL
    write_docs = write_scope == WRITE_SCOPE_FULL
    write_owned_configs = write_scope == WRITE_SCOPE_FULL
    skipped_outputs: list[str] = []
    contract_token = _derived_contract_token(bundle_path, bundle)

    e0_patch_rows, e0_candidate_rows = _prepare_e0_backfill(
        workspace_root=workspace_root,
        contract_rows=contract_rows,
        source_fieldnames=source_fieldnames,
        source_rows=source_rows,
        contract_token=contract_token,
    )
    _write_csv(
        e0_patch_csv,
        [
            "row_role",
            "selected_source_run_id",
            "selected_row_id",
            "expected_baseline",
            "status",
            "patched_fields",
            *[f"patched_{field}" for field in PATCHABLE_CONTEXT_FIELDS],
        ],
        e0_patch_rows,
    )
    candidate_fieldnames = list(source_fieldnames)
    for extra in ("accuracy_measurement_contract_note", *PATCHABLE_CONTEXT_FIELDS):
        if extra not in candidate_fieldnames:
            candidate_fieldnames.append(extra)
    _write_csv(e0_candidate_csv, candidate_fieldnames, e0_candidate_rows)

    runtime_rows_by_experiment, runtime_fieldnames, runtime_status_rows = _collect_runtime_smoke_lane_rows(
        contract_rows=contract_rows,
        accuracy_launch_root=resolved_accuracy_launch_root,
        contract_token=contract_token,
        variant_lookup=variant_lookup,
    )
    _write_csv(
        runtime_status_csv,
        [
            "variant_id",
            "experiment_id",
            "internal_experiment_id",
            "lane_stub",
            "annotated_csv",
            "status",
            "matched_row_count",
            "expected_row_count",
            "missing_row_roles",
        ],
        runtime_status_rows,
    )

    lane_scaffold_files: list[str] = []
    runtime_lane_files: list[str] = []
    runtime_overlay_configs: list[str] = []
    grouped_rows: dict[str, list[dict[str, str]]] = {}
    for row in contract_rows:
        experiment_id = str(row.get("experiment_id") or "")
        if experiment_id == "E0":
            continue
        grouped_rows.setdefault(experiment_id, []).append(_contract_scaffold_to_row(row))
    scaffold_fieldnames = [
        "run_id",
        "source_run_id",
        "variant_id",
        "experiment_id",
        "internal_experiment_id",
        "public_module_stack",
        "baseline",
        "device",
        "measurement_window",
        "model",
        "split",
        "workload",
        "seed",
        "config_snapshot",
        "det_policy",
        "det_k_signature",
        "det_k_global",
        "sparse_tau_global",
        "sparse_active_fraction",
        "gaussian_noise_std_ref",
        "crosstalk_alpha_ref",
        "top1",
        "top5",
        "top1_delta",
        "top5_delta",
        "measured_pass_elapsed_s",
        "measured_processed_samples",
        "latency_ms_per_sample",
        "notes",
    ]
    for experiment_id, rows in sorted(grouped_rows.items()):
        lane_stub = _lane_stub_for_experiment(experiment_id, variant_lookup=variant_lookup)
        lane_path = scaffold_dir / f"{lane_stub}.csv"
        _write_csv(lane_path, scaffold_fieldnames, rows)
        lane_scaffold_files.append(str(lane_path))
    for experiment_id, rows in sorted(runtime_rows_by_experiment.items()):
        lane_stub = _lane_stub_for_experiment(experiment_id, variant_lookup=variant_lookup)
        lane_path = runtime_lane_dir / f"{lane_stub}.csv"
        lane_fieldnames = list(runtime_fieldnames)
        for row in rows:
            for field in row:
                if field not in lane_fieldnames:
                    lane_fieldnames.append(field)
        _write_csv(lane_path, lane_fieldnames, rows)
        runtime_lane_files.append(str(lane_path))
        lane_cfg_path = generated_config_dir / f"{lane_stub}_runtime_smoke_current.yaml"
        source_cfg_path = generated_config_dir / f"{lane_stub}.yaml"
        if source_cfg_path.exists():
            lane_cfg = _load_yaml(source_cfg_path)
            lane_accuracy_cfg = lane_cfg.setdefault("accuracy", {})
            if not isinstance(lane_accuracy_cfg, dict):
                raise SystemExit(f"Expected accuracy mapping in {source_cfg_path}")
            lane_accuracy_cfg["source_csv"] = str(lane_path)
            lane_contract = dict(lane_accuracy_cfg.get("measurement_contract") or {})
            lane_contract["source"] = str(lane_path)
            lane_contract["note"] = _append_note(
                lane_contract.get("note", ""),
                f"runtime_smoke_current_lane_candidate:{contract_token}",
            )
            lane_accuracy_cfg["measurement_contract"] = lane_contract
            lane_run_cfg = lane_cfg.setdefault("run", {})
            lane_run_cfg["notes"] = _append_note(
                lane_run_cfg.get("notes", ""),
                f"runtime_smoke_current_lane_overlay:{contract_token}",
            )
            if write_generated_configs:
                _write_yaml(lane_cfg_path, lane_cfg)
                runtime_overlay_configs.append(str(lane_cfg_path))
            else:
                skipped_outputs.append(str(lane_cfg_path))

    e0_cfg_path = generated_config_dir / f"{_lane_stub_for_experiment('E0', variant_lookup=variant_lookup)}.yaml"
    if not e0_cfg_path.exists():
        raise SystemExit(f"Missing generated E0 config: {e0_cfg_path}")
    e0_cfg = _load_yaml(e0_cfg_path)
    accuracy_cfg = e0_cfg.setdefault("accuracy", {})
    if not isinstance(accuracy_cfg, dict):
        raise SystemExit("Expected E0 config accuracy to be a mapping.")
    accuracy_cfg["source_csv"] = str(e0_candidate_csv)
    measurement_contract = dict(accuracy_cfg.get("measurement_contract") or {})
    measurement_contract["source"] = str(e0_candidate_csv)
    measurement_contract["note"] = _append_note(
        measurement_contract.get("note", ""),
        f"metadata_backfill_overlay:{contract_token};no_new_measurement",
    )
    accuracy_cfg["measurement_contract"] = measurement_contract
    run_cfg = e0_cfg.setdefault("run", {})
    run_cfg["notes"] = _append_note(
        run_cfg.get("notes", ""),
        f"accuracy_context_backfill_overlay:{contract_token}",
    )
    if write_generated_configs:
        _write_yaml(e0_overlay_config, e0_cfg)
    else:
        skipped_outputs.append(str(e0_overlay_config))

    merged_rows = list(e0_candidate_rows)
    for experiment_id in sorted(runtime_rows_by_experiment):
        merged_rows.extend(runtime_rows_by_experiment[experiment_id])
    merged_fieldnames = list(candidate_fieldnames)
    for field in runtime_fieldnames:
        if field not in merged_fieldnames:
            merged_fieldnames.append(field)
    _write_csv(merged_candidate_csv, merged_fieldnames, merged_rows)

    template_overlay_cfg = dict(template_cfg)
    template_overlay_accuracy = template_overlay_cfg.setdefault("accuracy", {})
    if not isinstance(template_overlay_accuracy, dict):
        raise SystemExit(f"Expected accuracy mapping in template {template_path}")
    template_overlay_accuracy["source_csv"] = str(merged_candidate_csv)
    if write_generated_configs:
        _write_yaml(merged_template_overlay, template_overlay_cfg)
    else:
        skipped_outputs.append(str(merged_template_overlay))
    bundle_overlay = dict(bundle)
    bundle_overlay_meta = dict(bundle_overlay.get("meta") or {})
    bundle_overlay_meta["tag"] = _derive_runtime_smoke_overlay_tag(bundle_path, bundle)
    bundle_overlay_meta["purpose"] = (
        f"current-row runtime-smoke overlay over canonical {_surface_display_name(bundle.get('meta') or {}, 'True SC E0-E6')} contract"
    )
    bundle_overlay["meta"] = bundle_overlay_meta
    bundle_overlay_paths = bundle_overlay.setdefault("paths", {})
    if not isinstance(bundle_overlay_paths, dict):
        raise SystemExit(f"Expected paths mapping in bundle {bundle_path}")
    bundle_overlay_paths.update(
        {
            key: str(value)
            for key, value in _runtime_smoke_overlay_paths(
                workspace_root,
                bundle_path=bundle_path,
                bundle=bundle,
            ).items()
            if key != "overlay_tag"
        }
    )
    bundle_overlay_paths["template_yaml"] = str(merged_template_overlay)
    if write_owned_configs:
        _write_yaml(merged_bundle_overlay, bundle_overlay)
    else:
        skipped_outputs.append(str(merged_bundle_overlay))

    manifest = {
        "bundle_path": str(bundle_path),
        "accuracy_launch_root": str(resolved_accuracy_launch_root),
        "accuracy_launch_root_selection_mode": launch_root_resolution["selection_mode"],
        "accuracy_launch_root_lane_count": launch_root_resolution["selected_root_lane_count"],
        "accuracy_launch_root_candidates": launch_root_resolution["candidates"],
        "source_csv": str(source_csv),
        "contract_csv": str(contract_csv),
        "e0_backfill_patch_csv": str(e0_patch_csv),
        "e0_backfill_candidate_csv": str(e0_candidate_csv),
        "e0_overlay_config": str(e0_overlay_config),
        "merged_candidate_csv": str(merged_candidate_csv),
        "merged_template_overlay": str(merged_template_overlay),
        "merged_bundle_overlay": str(merged_bundle_overlay),
        "lane_scaffold_dir": str(scaffold_dir),
        "lane_scaffold_files": lane_scaffold_files,
        "lane_scaffold_file_count": len(lane_scaffold_files),
        "runtime_smoke_lane_status_csv": str(runtime_status_csv),
        "runtime_smoke_lane_status_rows": runtime_status_rows,
        "runtime_smoke_lane_dir": str(runtime_lane_dir),
        "runtime_smoke_lane_files": runtime_lane_files,
        "runtime_smoke_lane_file_count": len(runtime_lane_files),
        "runtime_smoke_overlay_configs": runtime_overlay_configs,
        "runtime_smoke_launch_root": str(artifact_paths["runtime_smoke_launch_root"]),
        "runtime_smoke_launch_manifest_json": str(
            artifact_paths["runtime_smoke_launch_manifest_json"]
        ),
        "runtime_smoke_launch_note_md": str(artifact_paths["runtime_smoke_launch_note_md"]),
        "config_snapshot_contract_rows": config_snapshot_contract_rows,
        "config_snapshot_contract_counts": config_snapshot_contract_counts,
        "e0_patch_count": sum(1 for row in e0_patch_rows if row.get("status") == "patched"),
        "e0_patch_rows": e0_patch_rows,
        "manifest_json": str(manifest_json),
        "note_md": str(note_md),
        "source_prep_note_date": source_prep_note_date,
        "write_scope": write_scope,
        "skipped_outputs": sorted(dict.fromkeys(skipped_outputs)),
    }
    _write_json(manifest_json, manifest)
    if write_docs:
        _write_note(
            workspace_root=workspace_root,
            note_path=note_md,
            note_date=source_prep_note_date,
            bundle=bundle,
            bundle_path=bundle_path,
            source_csv=source_csv,
            e0_patch_csv=e0_patch_csv,
            e0_candidate_csv=e0_candidate_csv,
            e0_overlay_config=e0_overlay_config,
            lane_scaffold_dir=scaffold_dir,
            merged_candidate_csv=merged_candidate_csv,
            merged_template_overlay=merged_template_overlay,
            merged_bundle_overlay=merged_bundle_overlay,
            runtime_smoke_launch_note_md=artifact_paths["runtime_smoke_launch_note_md"],
            manifest=manifest,
        )
    else:
        skipped_outputs.append(str(note_md))
        manifest["skipped_outputs"] = sorted(dict.fromkeys(skipped_outputs))
        _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare conservative true-SC E0-E6 accuracy-source overlays and scaffolds."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--accuracy_launch_root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--inspect-existing-source-prep",
        action="store_true",
        help="Read-only comparison between the bundle contract and the stored source-prep manifest/note.",
    )
    parser.add_argument(
        "--write-scope",
        choices=(WRITE_SCOPE_FULL, WRITE_SCOPE_REPORT_DATA_ONLY),
        default=WRITE_SCOPE_FULL,
        help=(
            "Choose whether to materialize the full source-prep surface or only owned "
            "report-data artifacts."
        ),
    )
    args = parser.parse_args()

    bundle_path = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    accuracy_launch_root = None
    if args.accuracy_launch_root is not None:
        accuracy_launch_root = (
            args.accuracy_launch_root
            if args.accuracy_launch_root.is_absolute()
            else ROOT / args.accuracy_launch_root
        )
    if args.inspect_existing_source_prep:
        payload = inspect_existing_source_prep(
            bundle_path=bundle_path,
            accuracy_launch_root=accuracy_launch_root,
            write_scope=args.write_scope,
        )
    else:
        payload = build_accuracy_source_artifacts(
            bundle_path=bundle_path,
            accuracy_launch_root=accuracy_launch_root,
            write_scope=args.write_scope,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
