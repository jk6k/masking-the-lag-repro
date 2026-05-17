#!/usr/bin/env python3
"""Stage true-SC E0-E6 config snapshots and dry-run accuracy launch manifests."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from .fuller_phase1_registry import (
        normalize_variant_descriptor,
        variant_lookup_by_internal_id,
        variant_lookup_by_public_id,
    )
except ImportError:
    from fuller_phase1_registry import (  # type: ignore
        normalize_variant_descriptor,
        variant_lookup_by_internal_id,
        variant_lookup_by_public_id,
    )

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from accuracy.bitstream_runtime_safety import (
    BITSTREAM_RUNTIME_WORKING_SET_LIMIT_BYTES,
    build_bitstream_runtime_guardrail,
    default_full_surface_runtime_validation_root,
    governed_full_surface_runtime_validation_ready,
)

DEFAULT_BUNDLE = ROOT / "configs" / "true_sc_e0_e6_canonical_bundle_20260421.yaml"
DEFAULT_LANE_ORDER = ("E6", "E3", "E5", "E4", "E2", "E1")
RUNS_DIR = ROOT / "experiments" / "results" / "runs"
REPORT_DIR = ROOT / "experiments" / "results" / "report_data"
RUNTIME_SMOKE_TIER = "runtime_smoke"
ANALYSIS_GRADE_TIER = "analysis_grade"
RUNTIME_SMOKE_CONTRACT_NOTE = "runtime_smoke_model_level_measured_not_analysis_grade"
ANALYSIS_GRADE_CONTRACT_NOTE = "true_sc_e0_e6_current_generation_measured_candidate"
RUNTIME_SMOKE_SURFACE_SCOPE = "limited_linear_attention_pilot"
ANALYSIS_GRADE_SURFACE_SCOPE = "all"
DEFAULT_STREAM_LENGTH = 64
WRITE_SCOPE_FULL = "full"
WRITE_SCOPE_REPORT_DATA_ONLY = "report-data-only"
DEFAULT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS = 15.0
DEFAULT_STALL_TIMEOUT_SECONDS = 180.0
DEFAULT_PRELAUNCH_RUNTIME_SMOKE_SAMPLES = 16
DEFAULT_PRELAUNCH_MIN_SAMPLES_PER_HOUR = 30.0
DEFAULT_PRELAUNCH_MAX_SECONDS_PER_SAMPLE = 120.0
DEFAULT_RUNTIME_MIN_SAMPLES_PER_HOUR = 12.0
DEFAULT_RUNTIME_MAX_SECONDS_PER_SAMPLE = 300.0
DEFAULT_RUNTIME_MAX_ETA_CURRENT_RATE_SECONDS = 86400.0
DEFAULT_RUNTIME_MIN_PROCESSED_SAMPLES = 4
DEFAULT_RUNTIME_MIN_ELAPSED_SECONDS = 300.0
RUNTIME_SMOKE_TARGET_MODULE_KEYS = (
    "layer_4.1.global_rep.0.pre_norm_mha.1.attn_scores,"
    "layer_4.1.global_rep.0.pre_norm_mha.1.attn_output"
)
FULL_SURFACE_RUNTIME_VALIDATION_ROOT = default_full_surface_runtime_validation_root(ROOT)
DATE_TOKEN_PATTERN = re.compile(r"(20\d{2})(\d{2})(\d{2})")


def _resolve_governance(bundle_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    governance = bundle.get("governance") or {}
    if "measured_collection_enabled" in governance:
        raise SystemExit(
            "Legacy governance field 'measured_collection_enabled' is no longer supported. "
            "Migrate the bundle to runtime_smoke_enabled / analysis_grade_enabled / "
            "analysis_grade_required_seeds / analysis_grade_require_full_eval."
        )
    required_keys = (
        "runtime_smoke_enabled",
        "analysis_grade_enabled",
        "analysis_grade_required_seeds",
        "analysis_grade_require_full_eval",
    )
    missing = [key for key in required_keys if key not in governance]
    if missing:
        raise SystemExit(
            f"Bundle governance is missing required fields {missing} for {bundle_path}."
        )
    raw_required_seeds = governance.get("analysis_grade_required_seeds") or []
    if not isinstance(raw_required_seeds, list) or not raw_required_seeds:
        raise SystemExit(
            "governance.analysis_grade_required_seeds must be a non-empty list of integers."
        )
    try:
        required_seeds = [int(value) for value in raw_required_seeds]
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            "governance.analysis_grade_required_seeds must contain only integers."
        ) from exc
    return {
        "runtime_smoke_enabled": bool(governance.get("runtime_smoke_enabled")),
        "analysis_grade_enabled": bool(governance.get("analysis_grade_enabled")),
        "analysis_grade_required_seeds": required_seeds,
        "analysis_grade_require_full_eval": bool(
            governance.get("analysis_grade_require_full_eval")
        ),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
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


def _launch_tool_command(meta: dict[str, Any]) -> str:
    return str(
        meta.get("launch_tool_entrypoint")
        or "python3 experiments/tools/prepare_true_sc_e0_e6_accuracy_launch.py"
    ).strip()


def _variant_maps(
    bundle: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    variants = bundle.get("variants") or []
    if not isinstance(variants, list):
        return {}, {}
    return (
        variant_lookup_by_internal_id(variants),
        variant_lookup_by_public_id(variants),
    )


def _resolve_lane_descriptor(
    lane_token: str,
    *,
    by_internal: dict[str, dict[str, Any]],
    by_public: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    normalized = str(lane_token or "").strip().upper()
    if normalized in by_public:
        return by_public[normalized]
    if normalized in by_internal:
        return by_internal[normalized]
    return normalize_variant_descriptor({"internal_experiment_id": normalized})


def _lane_stub(descriptor: dict[str, Any]) -> str:
    return str(
        descriptor.get("config_stub")
        or descriptor.get("variant_id")
        or descriptor.get("internal_experiment_id")
        or "lane"
    ).strip().lower()


def _lane_display(descriptor: dict[str, Any]) -> str:
    variant_id = str(descriptor.get("variant_id") or "").strip().upper()
    internal_id = str(descriptor.get("internal_experiment_id") or "").strip().upper()
    if variant_id and variant_id != internal_id:
        return f"{variant_id} ({internal_id})"
    return internal_id or variant_id or "UNKNOWN"


def _resolve_launch_artifact_paths(
    *,
    bundle_path: Path,
    bundle: dict[str, Any],
    evidence_tier: str,
) -> tuple[Path, Path, Path]:
    stem = _launch_artifact_stem(bundle_path, bundle)
    tier = _sanitize_token(evidence_tier)
    launch_root = REPORT_DIR / f"{stem}_accuracy_launch_{tier}"
    manifest_json = launch_root / f"{stem}_accuracy_launch_manifest_{tier}.json"
    note_md = ROOT / "docs" / "reports" / f"{stem}_accuracy_launch_prep_note_{tier}.md"
    return launch_root, manifest_json, note_md


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _resolve_launch_note_date(meta: dict[str, Any], bundle_path: Path) -> str:
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


def _expected_lane_launch_paths(launch_root: Path, descriptor: dict[str, Any]) -> dict[str, str]:
    lane_root = launch_root / _lane_stub(descriptor)
    return {
        "results_csv": str(lane_root / "raw_accuracy.csv"),
        "annotated_results_csv": str(lane_root / "annotated_accuracy.csv"),
        "prepared_phase1_config_root": str(lane_root / "prepared_phase1_configs"),
        "prepared_eligibility_report_root": str(lane_root / "prepared_eligibility"),
        "authorization_root": str(lane_root / "authorization"),
        "progress_root": str(lane_root / "progress"),
        "progress_manifest_json": str(lane_root / "progress" / "manifest.json"),
    }


def _write_note(
    path: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    bundle: dict[str, Any],
    bundle_path: Path,
    evidence_tier: str,
    required_seeds: list[int],
    full_eval_required: bool,
    note_date: str,
) -> None:
    meta = bundle.get("meta") or {}
    surface_display_name = _surface_display_name(meta, "True SC E0-E6")
    lines = [
        f"# {surface_display_name} Accuracy Launch Prep Note",
        "",
        f"Date: `{note_date}`",
        f"Bundle: `{bundle_path}`",
        "",
        "## Launch Order",
        "",
    ]
    for row in manifest_rows:
        lane_display = _lane_display(
            {
                "variant_id": row.get("variant_id"),
                "internal_experiment_id": row.get("internal_experiment_id") or row.get("experiment_id"),
            }
        )
        lines.append(
            f"- `{lane_display}`: staged_run_id=`{row['run_id']}` "
            f"analysis_grade_ready=`{str(bool(row['analysis_grade_ready'])).lower()}` "
            f"runtime_launch_ready=`{str(bool(row['runtime_launch_ready'])).lower()}` "
            f"manifest=`{row['progress_manifest_json']}`"
        )
        blockers = list(row.get("analysis_grade_blockers") or [])
        if blockers:
            lines.append(f"  analysis blockers: `{'; '.join(str(item) for item in blockers)}`")
        runtime_blockers = list(row.get("runtime_launch_blockers") or [])
        if runtime_blockers:
            lines.append(f"  runtime blockers: `{'; '.join(str(item) for item in runtime_blockers)}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            f"- evidence_tier: `{evidence_tier}`",
            "- all commands are prepared with `--device mps`",
            f"- prelaunch smoke gate: `{DEFAULT_PRELAUNCH_RUNTIME_SMOKE_SAMPLES}` samples, "
            f"`>= {DEFAULT_PRELAUNCH_MIN_SAMPLES_PER_HOUR:.0f} samples/hour`, "
            f"`<= {DEFAULT_PRELAUNCH_MAX_SECONDS_PER_SAMPLE:.0f}s/sample` before the long run is allowed to continue",
            f"- runtime health gate: `>= {DEFAULT_RUNTIME_MIN_SAMPLES_PER_HOUR:.0f} samples/hour`, "
            f"`<= {DEFAULT_RUNTIME_MAX_SECONDS_PER_SAMPLE:.0f}s/sample`, "
            f"`<= {DEFAULT_RUNTIME_MAX_ETA_CURRENT_RATE_SECONDS:.0f}s` current-rate ETA",
            "- this tool only performs `--dry_run` launch preparation",
            "- any real long run must be launched separately under `caffeinate -dimsu` after governance approval",
            f"- refresh command: `{_launch_tool_command(meta)} --bundle {bundle_path} --evidence_tier {evidence_tier}`",
        ]
    )
    if evidence_tier == RUNTIME_SMOKE_TIER:
        lines.extend(
            [
                "- this manifest is runtime-smoke only and must not be used for thesis-facing architecture comparison",
                f"- analysis-grade requires seeds `{required_seeds}` and full eval=`{full_eval_required}`",
                f"- prepared bitstream surface scope: `{RUNTIME_SMOKE_SURFACE_SCOPE}`",
            ]
        )
    else:
        any_blocked = any(not bool(row.get("analysis_grade_ready")) for row in manifest_rows)
        any_runtime_blocked = any(not bool(row.get("runtime_launch_ready")) for row in manifest_rows)
        lines.extend(
            [
                "- this manifest is analysis-grade eligible by launch-shape construction",
                (
                    "- current governed analysis-grade readiness is still blocked on the listed lane blockers"
                    if any_blocked
                    else "- current governed analysis-grade readiness is clear on the staged lanes"
                ),
                (
                    "- current runtime launch safety is still blocked on the listed runtime blockers"
                    if any_runtime_blocked
                    else "- current runtime launch safety is clear on the staged lanes"
                ),
                f"- prepared bitstream surface scope: `{ANALYSIS_GRADE_SURFACE_SCOPE}`",
                f"- required seeds: `{required_seeds}`",
                f"- full eval required: `{full_eval_required}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inspect_existing_launch_prep(
    *,
    bundle_path: Path,
    lane_order: tuple[str, ...],
    evidence_tier: str,
    write_scope: str = WRITE_SCOPE_FULL,
) -> dict[str, Any]:
    bundle = _load_yaml(bundle_path)
    by_internal, by_public = _variant_maps(bundle)
    governance = _resolve_governance(bundle_path, bundle)
    if evidence_tier == RUNTIME_SMOKE_TIER:
        if not governance["runtime_smoke_enabled"]:
            raise SystemExit(
                f"Bundle does not allow runtime_smoke launch preparation: {bundle_path}"
            )
    elif evidence_tier == ANALYSIS_GRADE_TIER:
        if not governance["analysis_grade_enabled"]:
            raise SystemExit(
                f"Bundle does not allow analysis_grade launch preparation: {bundle_path}"
            )
    else:
        raise SystemExit(f"Unsupported evidence tier: {evidence_tier!r}")

    meta = bundle.get("meta") or {}
    generated_config_dir = _resolve_repo_path(((bundle.get("paths") or {}).get("generated_config_dir")))
    launch_root, manifest_json, note_md = _resolve_launch_artifact_paths(
        bundle_path=bundle_path,
        bundle=bundle,
        evidence_tier=evidence_tier,
    )
    expected_note_date = _resolve_launch_note_date(meta, bundle_path)
    expected_scope = _resolve_bitstream_surface_scope(evidence_tier)
    expected_target_module_keys = _resolve_bitstream_target_module_keys(evidence_tier) or ""

    issues: list[str] = []
    out_of_scope_issues: list[str] = []
    existing_payload: dict[str, Any] = {}
    if manifest_json.exists():
        existing_payload = _load_json(manifest_json)
    else:
        issues.append("missing_launch_manifest")

    existing_note_date = _read_note_date(note_md)
    if not note_md.exists():
        target_issues = (
            out_of_scope_issues
            if write_scope == WRITE_SCOPE_REPORT_DATA_ONLY
            else issues
        )
        target_issues.append("missing_launch_note")
    elif existing_note_date != expected_note_date:
        target_issues = (
            out_of_scope_issues
            if write_scope == WRITE_SCOPE_REPORT_DATA_ONLY
            else issues
        )
        target_issues.append("launch_note_date_mismatch")

    row_alignment: list[dict[str, Any]] = []
    if existing_payload:
        if str(existing_payload.get("bundle_path") or "").strip() != str(bundle_path):
            issues.append("bundle_path_mismatch")
        if str(existing_payload.get("manifest_json") or "").strip() != str(manifest_json):
            issues.append("manifest_json_path_mismatch")
        if str(existing_payload.get("note_md") or "").strip() != str(note_md):
            issues.append("note_md_path_mismatch")
        if list(existing_payload.get("lane_order") or []) != list(lane_order):
            issues.append("lane_order_mismatch")
        if str(existing_payload.get("accuracy_evidence_tier") or "").strip() != evidence_tier:
            issues.append("accuracy_evidence_tier_mismatch")
        if list(existing_payload.get("analysis_grade_required_seeds") or []) != list(
            governance["analysis_grade_required_seeds"]
        ):
            issues.append("analysis_grade_required_seeds_mismatch")
        if bool(existing_payload.get("analysis_grade_require_full_eval")) != bool(
            governance["analysis_grade_require_full_eval"]
        ):
            issues.append("analysis_grade_require_full_eval_mismatch")
        if str(existing_payload.get("bitstream_surface_scope") or "").strip() != expected_scope:
            issues.append("bitstream_surface_scope_mismatch")
        if str(existing_payload.get("bitstream_target_module_keys") or "").strip() != expected_target_module_keys:
            issues.append("bitstream_target_module_keys_mismatch")

        stored_rows = existing_payload.get("manifest_rows") or []
        if not isinstance(stored_rows, list):
            stored_rows = []
        rows_by_token: dict[str, dict[str, Any]] = {}
        for row in stored_rows:
            if not isinstance(row, dict):
                continue
            for key in (
                str(row.get("variant_id") or "").strip().upper(),
                str(row.get("internal_experiment_id") or row.get("experiment_id") or "").strip().upper(),
                str(row.get("experiment_id") or "").strip().upper(),
            ):
                if key:
                    rows_by_token[key] = row
        if len(stored_rows) != len(lane_order):
            issues.append("manifest_row_count_mismatch")

        for lane_token in lane_order:
            descriptor = _resolve_lane_descriptor(
                lane_token,
                by_internal=by_internal,
                by_public=by_public,
            )
            token_label = str(lane_token).strip().upper()
            expected_paths = _expected_lane_launch_paths(launch_root, descriptor)
            row = rows_by_token.get(token_label)
            if row is None:
                issues.append(f"{token_label}:manifest_row_missing")
                row_alignment.append(
                    {
                        "lane_order_token": token_label,
                        "variant_id": descriptor.get("variant_id"),
                        "experiment_id": descriptor.get("internal_experiment_id"),
                        "status": "missing",
                        "issues": ["manifest_row_missing"],
                    }
                )
                continue

            row_issues: list[str] = []
            for field, expected_value in expected_paths.items():
                stored_value = str(row.get(field) or "").strip()
                if stored_value != expected_value:
                    row_issues.append(f"{field}_mismatch")
            if generated_config_dir is None:
                row_issues.append("generated_config_dir_missing")
            else:
                expected_generated_config = str(
                    generated_config_dir / f"{_lane_stub(descriptor)}.yaml"
                )
                if str(row.get("generated_config") or "").strip() != expected_generated_config:
                    row_issues.append("generated_config_mismatch")
            progress_manifest_json = str(row.get("progress_manifest_json") or "").strip()
            if progress_manifest_json and not Path(progress_manifest_json).exists():
                row_issues.append("progress_manifest_missing")

            if row_issues:
                issues.extend(f"{token_label}:{issue}" for issue in row_issues)
            row_alignment.append(
                {
                    "lane_order_token": token_label,
                    "variant_id": descriptor.get("variant_id"),
                    "experiment_id": descriptor.get("internal_experiment_id"),
                    "status": "aligned" if not row_issues else "stale",
                    "issues": row_issues,
                    "progress_manifest_json": progress_manifest_json,
                }
            )

    return {
        "bundle_path": str(bundle_path),
        "accuracy_evidence_tier": evidence_tier,
        "lane_order": list(lane_order),
        "generated_config_dir": str(generated_config_dir) if generated_config_dir is not None else "",
        "generated_config_dir_exists": bool(generated_config_dir and generated_config_dir.exists()),
        "manifest_json": str(manifest_json),
        "manifest_json_exists": manifest_json.exists(),
        "note_md": str(note_md),
        "note_exists": note_md.exists(),
        "write_scope": write_scope,
        "expected_note_date": expected_note_date,
        "existing_note_date": existing_note_date,
        "expected_bitstream_surface_scope": expected_scope,
        "expected_bitstream_target_module_keys": expected_target_module_keys,
        "status": "aligned" if not issues else "stale",
        "issues": list(dict.fromkeys(issues)),
        "out_of_scope_issues": list(dict.fromkeys(out_of_scope_issues)),
        "row_alignment": row_alignment,
    }


def _resolve_contract_note(evidence_tier: str) -> str:
    if evidence_tier == RUNTIME_SMOKE_TIER:
        return RUNTIME_SMOKE_CONTRACT_NOTE
    if evidence_tier == ANALYSIS_GRADE_TIER:
        return ANALYSIS_GRADE_CONTRACT_NOTE
    raise SystemExit(f"Unsupported evidence tier: {evidence_tier!r}")


def _resolve_launch_analysis_grade_status(
    *,
    evidence_tier: str,
    required_seeds: list[int],
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if evidence_tier != ANALYSIS_GRADE_TIER:
        blockers.append("runtime_smoke_only")
    if sorted(set(required_seeds)) != [0, 1, 2] or len(required_seeds) != 3:
        blockers.append("missing_seeds012")
    return evidence_tier == ANALYSIS_GRADE_TIER and not blockers, blockers


def _resolve_bitstream_surface_scope(evidence_tier: str) -> str:
    if evidence_tier == RUNTIME_SMOKE_TIER:
        return RUNTIME_SMOKE_SURFACE_SCOPE
    if evidence_tier == ANALYSIS_GRADE_TIER:
        return ANALYSIS_GRADE_SURFACE_SCOPE
    raise SystemExit(f"Unsupported evidence tier: {evidence_tier!r}")


def _resolve_bitstream_target_module_keys(evidence_tier: str) -> str | None:
    if evidence_tier == RUNTIME_SMOKE_TIER:
        return RUNTIME_SMOKE_TARGET_MODULE_KEYS
    if evidence_tier == ANALYSIS_GRADE_TIER:
        return None
    raise SystemExit(f"Unsupported evidence tier: {evidence_tier!r}")


def _resolve_runtime_launch_status(
    *,
    model_key: str,
    surface_scope: str,
    target_module_keys_raw: str | None = None,
    explicit_eval_batch_size: int | None = None,
    stream_length: int = DEFAULT_STREAM_LENGTH,
) -> tuple[bool, list[str], dict[str, Any]]:
    guardrail = build_bitstream_runtime_guardrail(
        model_key=model_key,
        surface_scope=surface_scope,
        target_module_keys_raw=target_module_keys_raw,
        explicit_eval_batch_size=explicit_eval_batch_size,
        stream_length=int(stream_length),
        root_dir=ROOT,
    )
    blockers: list[str] = []
    full_surface_validation_ready = governed_full_surface_runtime_validation_ready(
        FULL_SURFACE_RUNTIME_VALIDATION_ROOT
    )
    if guardrail["all_target_surface"] and not full_surface_validation_ready:
        blockers.append("all_target_runtime_frozen_until_repaired_path_is_validated")
    if int(guardrail["estimated_working_set_bytes"]) > int(BITSTREAM_RUNTIME_WORKING_SET_LIMIT_BYTES):
        blockers.append("estimated_quantized_working_set_exceeds_launch_bound")
    guardrail["governed_full_surface_runtime_validation_ready"] = bool(
        full_surface_validation_ready
    )
    guardrail["governed_full_surface_runtime_validation_root"] = str(
        FULL_SURFACE_RUNTIME_VALIDATION_ROOT
    )
    return not blockers, blockers, guardrail


def _resolve_bundle_lane_analysis_grade_gates(bundle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    paths = bundle.get("paths") or {}
    preflight_json = _resolve_repo_path(paths.get("preflight_json"))
    if preflight_json is None or not preflight_json.exists():
        return {}
    payload = _load_json(preflight_json)
    lane_rows = payload.get("lane_rows") or []
    if not isinstance(lane_rows, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in lane_rows:
        if not isinstance(row, dict):
            continue
        experiment_id = str(row.get("experiment_id") or "").strip().upper()
        gate = row.get("analysis_grade_gate") or {}
        if not experiment_id or not isinstance(gate, dict):
            continue
        result[experiment_id] = {
            "status": str(gate.get("status") or "").strip(),
            "blockers": [str(item) for item in (gate.get("blockers") or [])],
            "accuracy_status": str(row.get("accuracy_status") or "").strip(),
        }
    return result


def _write_launch_prep_progress_manifest(
    path: Path,
    *,
    run_id: str,
    experiment_id: str,
    config_snapshot: str,
    config_snapshot_exists: bool,
    config_snapshot_stage_state: str,
    results_csv: Path,
    annotated_results_csv: Path,
    progress_root: Path,
    command: list[str],
    evidence_tier: str,
    required_seeds: list[int],
    full_eval_required: bool,
    bundle_analysis_grade_gate: dict[str, Any] | None = None,
    runtime_launch_ready: bool,
    runtime_launch_blockers: list[str],
    runtime_guardrail: dict[str, Any],
) -> tuple[bool, list[str]]:
    analysis_grade_ready, analysis_grade_blockers = _resolve_launch_analysis_grade_status(
        evidence_tier=evidence_tier,
        required_seeds=required_seeds,
    )
    gate = bundle_analysis_grade_gate or {}
    gate_blockers = [str(item) for item in (gate.get("blockers") or [])]
    governed_analysis_grade_ready = bool(analysis_grade_ready) and (
        str(gate.get("status") or "").strip() in {"", "pass"}
    )
    governed_analysis_grade_blockers = list(
        dict.fromkeys(list(analysis_grade_blockers) + gate_blockers)
    )
    payload = {
        "prep_only": True,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "config_snapshot": config_snapshot,
        "config_snapshot_exists": bool(config_snapshot_exists),
        "config_snapshot_stage_state": config_snapshot_stage_state,
        "results_csv": str(results_csv),
        "annotated_results_csv": str(annotated_results_csv),
        "progress_root": str(progress_root),
        "accuracy_evidence_tier": evidence_tier,
        "launch_shape_analysis_grade_ready": analysis_grade_ready,
        "launch_shape_analysis_grade_blockers": analysis_grade_blockers,
        "analysis_grade_ready": governed_analysis_grade_ready,
        "analysis_grade_blockers": governed_analysis_grade_blockers,
        "bundle_analysis_grade_gate": gate,
        "analysis_grade_required_seeds": list(required_seeds),
        "analysis_grade_require_full_eval": bool(full_eval_required),
        "runtime_launch_ready": bool(runtime_launch_ready),
        "runtime_launch_blockers": list(runtime_launch_blockers),
        "runtime_guardrail": runtime_guardrail,
        "runtime_health_gate": {
            "progress_heartbeat_interval_seconds": DEFAULT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
            "stall_timeout_seconds": DEFAULT_STALL_TIMEOUT_SECONDS,
            "prelaunch_runtime_smoke_samples": DEFAULT_PRELAUNCH_RUNTIME_SMOKE_SAMPLES,
            "prelaunch_min_samples_per_hour": DEFAULT_PRELAUNCH_MIN_SAMPLES_PER_HOUR,
            "prelaunch_max_seconds_per_sample": DEFAULT_PRELAUNCH_MAX_SECONDS_PER_SAMPLE,
            "pathological_min_samples_per_hour": DEFAULT_RUNTIME_MIN_SAMPLES_PER_HOUR,
            "pathological_max_seconds_per_sample": DEFAULT_RUNTIME_MAX_SECONDS_PER_SAMPLE,
            "pathological_max_eta_current_rate_seconds": DEFAULT_RUNTIME_MAX_ETA_CURRENT_RATE_SECONDS,
            "pathological_min_processed_samples": DEFAULT_RUNTIME_MIN_PROCESSED_SAMPLES,
            "pathological_min_elapsed_seconds": DEFAULT_RUNTIME_MIN_ELAPSED_SECONDS,
        },
        "launch_command": command,
    }
    _write_json(path, payload)
    return governed_analysis_grade_ready, governed_analysis_grade_blockers


def _stage_config_snapshot(
    src_cfg: Path,
    *,
    write_scope: str,
) -> dict[str, Any]:
    cfg = _load_yaml(src_cfg)
    run_cfg = cfg.get("run") or {}
    run_id = str(run_cfg.get("run_id") or src_cfg.stem).strip()
    experiment_id = str(run_cfg.get("experiment_id") or "").strip().upper()
    if not run_id:
        raise SystemExit(f"Config missing run.run_id: {src_cfg}")
    stage_dir = RUNS_DIR / run_id
    staged_cfg = stage_dir / "config_snapshot.yaml"
    if write_scope == WRITE_SCOPE_FULL:
        stage_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_cfg, staged_cfg)
        config_snapshot_exists = True
        config_snapshot_stage_state = "written"
    elif write_scope == WRITE_SCOPE_REPORT_DATA_ONLY:
        config_snapshot_exists = staged_cfg.exists()
        config_snapshot_stage_state = (
            "preexisting" if config_snapshot_exists else "predicted_unmaterialized"
        )
    else:
        raise SystemExit(f"Unsupported write scope: {write_scope!r}")
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "config_snapshot": str(staged_cfg),
        "config_snapshot_exists": config_snapshot_exists,
        "config_snapshot_stage_state": config_snapshot_stage_state,
    }


def _build_launch_command(
    *,
    python_bin: str,
    run_id: str,
    results_csv: Path,
    annotated_results_csv: Path,
    prepared_phase1_config_root: Path,
    prepared_eligibility_report_root: Path,
    authorization_root: Path,
    progress_root: Path,
    evidence_tier: str,
    required_seeds: list[int],
    full_eval_required: bool,
    bitstream_surface_scope: str,
    bitstream_target_module_keys: str | None,
) -> list[str]:
    contract_note = _resolve_contract_note(evidence_tier)
    command = [
        python_bin,
        str(ROOT / "experiments" / "tools" / "run_config_conditioned_accuracy_matrix.py"),
        "--run_ids",
        run_id,
        "--results_csv",
        str(results_csv),
        "--annotated_results_csv",
        str(annotated_results_csv),
        "--prepared_phase1_config_root",
        str(prepared_phase1_config_root),
        "--prepared_eligibility_report_root",
        str(prepared_eligibility_report_root),
        "--progress_root",
        str(progress_root),
        "--progress_heartbeat_interval_seconds",
        str(float(DEFAULT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS)),
        "--stall_timeout_seconds",
        str(float(DEFAULT_STALL_TIMEOUT_SECONDS)),
        "--prelaunch_runtime_smoke_samples",
        str(int(DEFAULT_PRELAUNCH_RUNTIME_SMOKE_SAMPLES)),
        "--prelaunch_min_samples_per_hour",
        str(float(DEFAULT_PRELAUNCH_MIN_SAMPLES_PER_HOUR)),
        "--prelaunch_max_seconds_per_sample",
        str(float(DEFAULT_PRELAUNCH_MAX_SECONDS_PER_SAMPLE)),
        "--pathological_min_samples_per_hour",
        str(float(DEFAULT_RUNTIME_MIN_SAMPLES_PER_HOUR)),
        "--pathological_max_seconds_per_sample",
        str(float(DEFAULT_RUNTIME_MAX_SECONDS_PER_SAMPLE)),
        "--pathological_max_eta_current_rate_seconds",
        str(float(DEFAULT_RUNTIME_MAX_ETA_CURRENT_RATE_SECONDS)),
        "--pathological_min_processed_samples",
        str(int(DEFAULT_RUNTIME_MIN_PROCESSED_SAMPLES)),
        "--pathological_min_elapsed_seconds",
        str(float(DEFAULT_RUNTIME_MIN_ELAPSED_SECONDS)),
        "--accuracy_backend",
        "mlx",
        "--models",
        "mobilevit_s",
        "--device",
        "mps",
        "--workers",
        "0",
        "--seeds",
        ",".join(str(seed) for seed in required_seeds)
        if evidence_tier == ANALYSIS_GRADE_TIER
        else "0",
        "--evidence_tier",
        evidence_tier,
        "--annotation_measurement_truth_class",
        "bitstream_model_level_measured",
        "--bitstream_measurement_truth_class",
        "bitstream_model_level_measured",
        "--bitstream_truth_class_authorization_root",
        str(authorization_root),
        "--annotation_contract_note",
        contract_note,
        "--bitstream_contract_note",
        contract_note,
        "--enable_bitstream_pilot",
        "--bitstream_surface_scope",
        bitstream_surface_scope,
        "--dry_run",
    ]
    if bitstream_target_module_keys:
        command.extend(["--bitstream_target_module_keys", bitstream_target_module_keys])
    if evidence_tier == ANALYSIS_GRADE_TIER and full_eval_required:
        # Intentionally omit --max_eval_samples to preserve full-eval launch shape.
        pass
    return command


def build_launch_artifacts(
    *,
    bundle_path: Path,
    python_bin: str,
    lane_order: tuple[str, ...],
    evidence_tier: str = RUNTIME_SMOKE_TIER,
    write_scope: str = WRITE_SCOPE_FULL,
) -> dict[str, Any]:
    bundle = _load_yaml(bundle_path)
    meta = bundle.get("meta") or {}
    by_internal, by_public = _variant_maps(bundle)
    governance = _resolve_governance(bundle_path, bundle)
    if evidence_tier == RUNTIME_SMOKE_TIER:
        if not governance["runtime_smoke_enabled"]:
            raise SystemExit(
                f"Bundle does not allow runtime_smoke launch preparation: {bundle_path}"
            )
    elif evidence_tier == ANALYSIS_GRADE_TIER:
        if not governance["analysis_grade_enabled"]:
            raise SystemExit(
                f"Bundle does not allow analysis_grade launch preparation: {bundle_path}"
            )
    else:
        raise SystemExit(f"Unsupported evidence tier: {evidence_tier!r}")
    generated_config_dir = _resolve_repo_path(((bundle.get("paths") or {}).get("generated_config_dir")))
    if generated_config_dir is None or not generated_config_dir.exists():
        raise SystemExit(f"Missing generated_config_dir for bundle: {bundle_path}")

    launch_root, manifest_json, note_md = _resolve_launch_artifact_paths(
        bundle_path=bundle_path,
        bundle=bundle,
        evidence_tier=evidence_tier,
    )
    note_date = _resolve_launch_note_date(meta, bundle_path)
    write_docs = write_scope == WRITE_SCOPE_FULL
    skipped_outputs: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    lane_analysis_grade_gates = _resolve_bundle_lane_analysis_grade_gates(bundle)
    launch_analysis_grade_ready, launch_analysis_grade_blockers = (
        _resolve_launch_analysis_grade_status(
            evidence_tier=evidence_tier,
            required_seeds=list(governance["analysis_grade_required_seeds"]),
        )
    )
    overall_analysis_grade_ready = launch_analysis_grade_ready
    overall_analysis_grade_blockers = list(launch_analysis_grade_blockers)
    overall_runtime_launch_ready = True
    overall_runtime_launch_blockers: list[str] = []
    bitstream_surface_scope = _resolve_bitstream_surface_scope(evidence_tier)
    bitstream_target_module_keys = _resolve_bitstream_target_module_keys(evidence_tier)

    for lane_token in lane_order:
        descriptor = _resolve_lane_descriptor(
            lane_token,
            by_internal=by_internal,
            by_public=by_public,
        )
        experiment_id = str(descriptor.get("internal_experiment_id") or lane_token).strip().upper()
        variant_id = str(descriptor.get("variant_id") or experiment_id).strip().upper()
        public_module_stack = list(descriptor.get("public_module_stack") or [])
        lane_stub = _lane_stub(descriptor)
        cfg_path = generated_config_dir / f"{lane_stub}.yaml"
        if not cfg_path.exists():
            raise SystemExit(f"Missing generated config for {lane_token}: {cfg_path}")
        staged = _stage_config_snapshot(cfg_path, write_scope=write_scope)
        if (
            write_scope == WRITE_SCOPE_REPORT_DATA_ONLY
            and not bool(staged["config_snapshot_exists"])
        ):
            skipped_outputs.append(str(staged["config_snapshot"]))
        lane_root = launch_root / lane_stub
        results_csv = lane_root / "raw_accuracy.csv"
        annotated_results_csv = lane_root / "annotated_accuracy.csv"
        prepared_phase1_config_root = lane_root / "prepared_phase1_configs"
        prepared_eligibility_report_root = lane_root / "prepared_eligibility"
        authorization_root = lane_root / "authorization"
        progress_root = lane_root / "progress"
        progress_manifest_json = progress_root / "manifest.json"
        command = _build_launch_command(
            python_bin=python_bin,
            run_id=staged["run_id"],
            results_csv=results_csv,
            annotated_results_csv=annotated_results_csv,
            prepared_phase1_config_root=prepared_phase1_config_root,
            prepared_eligibility_report_root=prepared_eligibility_report_root,
            authorization_root=authorization_root,
            progress_root=progress_root,
            evidence_tier=evidence_tier,
            required_seeds=list(governance["analysis_grade_required_seeds"]),
            full_eval_required=bool(governance["analysis_grade_require_full_eval"]),
            bitstream_surface_scope=bitstream_surface_scope,
            bitstream_target_module_keys=bitstream_target_module_keys,
        )
        runtime_launch_ready, runtime_launch_blockers, runtime_guardrail = _resolve_runtime_launch_status(
            model_key="mobilevit_s",
            surface_scope=bitstream_surface_scope,
            target_module_keys_raw=bitstream_target_module_keys,
        )
        lane_gate = lane_analysis_grade_gates.get(experiment_id, {})
        analysis_grade_ready, analysis_grade_blockers = _write_launch_prep_progress_manifest(
            progress_manifest_json,
            run_id=staged["run_id"],
            experiment_id=experiment_id,
            config_snapshot=str(staged["config_snapshot"]),
            config_snapshot_exists=bool(staged["config_snapshot_exists"]),
            config_snapshot_stage_state=str(staged["config_snapshot_stage_state"]),
            results_csv=results_csv,
            annotated_results_csv=annotated_results_csv,
            progress_root=progress_root,
            command=command,
            evidence_tier=evidence_tier,
            required_seeds=list(governance["analysis_grade_required_seeds"]),
            full_eval_required=bool(governance["analysis_grade_require_full_eval"]),
            bundle_analysis_grade_gate=lane_gate,
            runtime_launch_ready=runtime_launch_ready,
            runtime_launch_blockers=runtime_launch_blockers,
            runtime_guardrail=runtime_guardrail,
        )
        if not analysis_grade_ready:
            overall_analysis_grade_ready = False
            extra_lane_blockers = [
                blocker
                for blocker in analysis_grade_blockers
                if blocker not in launch_analysis_grade_blockers
            ]
            if extra_lane_blockers:
                overall_analysis_grade_blockers.extend(
                    f"{str(lane_token).strip().upper()}:{blocker}" for blocker in extra_lane_blockers
                )
        if not runtime_launch_ready:
            overall_runtime_launch_ready = False
            overall_runtime_launch_blockers.extend(
                f"{str(lane_token).strip().upper()}:{blocker}" for blocker in runtime_launch_blockers
            )
        manifest_rows.append(
            {
                **staged,
                "variant_id": variant_id,
                "internal_experiment_id": experiment_id,
                "public_module_stack": public_module_stack,
                "lane_order_token": str(lane_token).strip().upper(),
                "lane_stub": lane_stub,
                "accuracy_evidence_tier": evidence_tier,
                "launch_shape_analysis_grade_ready": launch_analysis_grade_ready,
                "launch_shape_analysis_grade_blockers": list(launch_analysis_grade_blockers),
                "analysis_grade_ready": analysis_grade_ready,
                "analysis_grade_blockers": analysis_grade_blockers,
                "runtime_launch_ready": runtime_launch_ready,
                "runtime_launch_blockers": list(runtime_launch_blockers),
                "runtime_guardrail": runtime_guardrail,
                "bitstream_surface_scope": bitstream_surface_scope,
                "bitstream_target_module_keys": bitstream_target_module_keys or "",
                "analysis_grade_required_seeds": list(
                    governance["analysis_grade_required_seeds"]
                ),
                "analysis_grade_require_full_eval": bool(
                    governance["analysis_grade_require_full_eval"]
                ),
                "bundle_analysis_grade_gate": lane_gate,
                "annotation_contract_note": _resolve_contract_note(evidence_tier),
                "bitstream_contract_note": _resolve_contract_note(evidence_tier),
                "generated_config": str(cfg_path),
                "results_csv": str(results_csv),
                "annotated_results_csv": str(annotated_results_csv),
                "prepared_phase1_config_root": str(prepared_phase1_config_root),
                "prepared_eligibility_report_root": str(prepared_eligibility_report_root),
                "authorization_root": str(authorization_root),
                "progress_root": str(progress_root),
                "progress_manifest_json": str(progress_manifest_json),
                "launch_command": command,
            }
        )

    overall_analysis_grade_blockers = list(dict.fromkeys(overall_analysis_grade_blockers))
    overall_runtime_launch_blockers = list(dict.fromkeys(overall_runtime_launch_blockers))
    payload = {
        "bundle_path": str(bundle_path),
        "lane_order": list(lane_order),
        "accuracy_evidence_tier": evidence_tier,
        "launch_shape_analysis_grade_ready": launch_analysis_grade_ready,
        "launch_shape_analysis_grade_blockers": launch_analysis_grade_blockers,
        "analysis_grade_ready": overall_analysis_grade_ready,
        "analysis_grade_blockers": overall_analysis_grade_blockers,
        "runtime_launch_ready": overall_runtime_launch_ready,
        "runtime_launch_blockers": overall_runtime_launch_blockers,
        "bitstream_surface_scope": bitstream_surface_scope,
        "bitstream_target_module_keys": bitstream_target_module_keys or "",
        "analysis_grade_required_seeds": list(governance["analysis_grade_required_seeds"]),
        "analysis_grade_require_full_eval": bool(
            governance["analysis_grade_require_full_eval"]
        ),
        "manifest_rows": manifest_rows,
        "manifest_json": str(manifest_json),
        "note_md": str(note_md),
        "write_scope": write_scope,
        "skipped_outputs": sorted(dict.fromkeys(skipped_outputs)),
    }
    _write_json(manifest_json, payload)
    if write_docs:
        _write_note(
            note_md,
            manifest_rows=manifest_rows,
            bundle=bundle,
            bundle_path=bundle_path,
            evidence_tier=evidence_tier,
            required_seeds=list(governance["analysis_grade_required_seeds"]),
            full_eval_required=bool(governance["analysis_grade_require_full_eval"]),
            note_date=note_date,
        )
    else:
        skipped_outputs.append(str(note_md))
        payload["skipped_outputs"] = sorted(dict.fromkeys(skipped_outputs))
        _write_json(manifest_json, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare dry-run accuracy launch manifests for true-SC E0-E6 lanes."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument(
        "--lane_order",
        default=",".join(DEFAULT_LANE_ORDER),
        help="Comma-separated launch order. Defaults to E6,E3,E5,E4,E2,E1.",
    )
    parser.add_argument(
        "--evidence_tier",
        choices=[RUNTIME_SMOKE_TIER, ANALYSIS_GRADE_TIER],
        default=RUNTIME_SMOKE_TIER,
        help="Launch evidence tier. Defaults to runtime_smoke.",
    )
    parser.add_argument(
        "--write-scope",
        choices=(WRITE_SCOPE_FULL, WRITE_SCOPE_REPORT_DATA_ONLY),
        default=WRITE_SCOPE_FULL,
        help="Write full outputs or only owned report-data artifacts.",
    )
    parser.add_argument("--inspect-existing-launch-prep", action="store_true")
    args = parser.parse_args()

    bundle_path = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    lane_order = tuple(item.strip().upper() for item in str(args.lane_order).split(",") if item.strip())
    if args.inspect_existing_launch_prep:
        payload = inspect_existing_launch_prep(
            bundle_path=bundle_path,
            lane_order=lane_order,
            evidence_tier=str(args.evidence_tier),
            write_scope=str(args.write_scope),
        )
    else:
        payload = build_launch_artifacts(
            bundle_path=bundle_path,
            python_bin=str(args.python_bin),
            lane_order=lane_order,
            evidence_tier=str(args.evidence_tier),
            write_scope=str(args.write_scope),
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
