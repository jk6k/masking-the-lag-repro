#!/usr/bin/env python3
"""Materialize and preflight the canonical true-SC E0-E6 config bundle."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from . import phase1_runner
    from .fuller_phase1_registry import (
        INTERNAL_EXPERIMENT_ORDER,
        normalize_variant_descriptor,
    )
except ImportError:
    import phase1_runner  # type: ignore
    from fuller_phase1_registry import (  # type: ignore
        INTERNAL_EXPERIMENT_ORDER,
        normalize_variant_descriptor,
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "configs" / "true_sc_e0_e6_canonical_bundle_20260421.yaml"
DEFAULT_PYTHON_BIN = sys.executable
EXPERIMENT_ORDER = INTERNAL_EXPERIMENT_ORDER
ACCURACY_CONTEXT_FIELDS = (
    "run_id",
    "experiment_id",
    "split",
    "workload",
    "seed",
    "det_policy",
    "det_k_signature",
    "det_k_global",
    "sparse_tau_global",
    "sparse_active_fraction",
    "gaussian_noise_std_ref",
    "crosstalk_alpha_ref",
    "config_snapshot",
)
RUNTIME_SMOKE_TIER = "runtime_smoke"
ANALYSIS_GRADE_TIER = "analysis_grade"
DATE_TOKEN_PATTERN = re.compile(r"(20\d{2})(\d{2})(\d{2})")
WRITE_SCOPE_FULL = "full"
WRITE_SCOPE_REPORT_DATA_ONLY = "report-data-only"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _csv_safe(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _resolve_repo_path(path_value: str | Path | None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return ROOT / path


def _path_exists(path_value: str | Path | None) -> bool:
    path = _resolve_repo_path(path_value)
    return bool(path and path.exists())


def _merge_nested_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _surface_display_name(meta: dict[str, Any], default: str) -> str:
    return str(meta.get("surface_display_name") or default).strip()


def _bundle_tool_command(meta: dict[str, Any]) -> str:
    return str(
        meta.get("bundle_tool_entrypoint")
        or "python3 experiments/tools/prepare_true_sc_e0_e6_bundle.py"
    ).strip()


def _phase1_runner_command(meta: dict[str, Any]) -> str:
    return str(meta.get("phase1_runner_entrypoint") or "python3 experiments/tools/phase1_runner.py").strip()


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


def _generated_config_family(path: Path | None) -> str:
    if path is None:
        return ""
    parts = path.parts
    marker = ("experiments", "results", "generated_configs")
    for index in range(len(parts) - len(marker)):
        if parts[index : index + len(marker)] == marker:
            family_index = index + len(marker)
            if family_index < len(parts):
                return str(parts[family_index]).strip()
            break
    return ""


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


def _summarize_template_provenance(
    *,
    template_path: Path,
    generated_root: Path,
) -> dict[str, Any]:
    bundle_generated_family = _generated_config_family(generated_root)
    template_generated_family = _generated_config_family(template_path)
    if template_generated_family:
        status = (
            "generated_template_aligned"
            if template_generated_family == bundle_generated_family
            else "generated_template_cross_family"
        )
    else:
        status = "repo_template"
    return {
        "status": status,
        "bundle_generated_family": bundle_generated_family,
        "template_generated_family": template_generated_family,
        "template_path": str(template_path),
        "template_is_generated_overlay": bool(template_generated_family),
        "matches_bundle_generated_family": bool(
            template_generated_family
            and bundle_generated_family
            and template_generated_family == bundle_generated_family
        ),
    }


def inspect_existing_preflight(
    *,
    bundle_path: Path,
    write_scope: str = WRITE_SCOPE_FULL,
) -> dict[str, Any]:
    if write_scope not in {WRITE_SCOPE_FULL, WRITE_SCOPE_REPORT_DATA_ONLY}:
        raise SystemExit(f"Unsupported write scope: {write_scope!r}")
    bundle = _load_yaml(bundle_path)
    meta = bundle.get("meta") or {}
    paths = bundle.get("paths") or {}

    template_path = _resolve_repo_path(paths.get("template_yaml"))
    if template_path is None or not template_path.exists():
        raise SystemExit(f"Missing template_yaml for bundle: {bundle_path}")
    generated_root = _resolve_repo_path(paths.get("generated_config_dir"))
    if generated_root is None:
        raise SystemExit(f"Missing generated_config_dir for bundle: {bundle_path}")
    preflight_json = _resolve_repo_path(paths.get("preflight_json"))
    if preflight_json is None:
        raise SystemExit(f"Missing preflight_json for bundle: {bundle_path}")
    preflight_note_md = _resolve_repo_path(paths.get("preflight_note_md"))

    current_template_provenance = _summarize_template_provenance(
        template_path=template_path,
        generated_root=generated_root,
    )
    expected_note_date = _resolve_bundle_note_date(meta, bundle_path)

    existing_payload: dict[str, Any] = {}
    issues: list[str] = []
    out_of_scope_issues: list[str] = []
    if preflight_json.exists():
        existing_payload = (
            _load_yaml(preflight_json)
            if preflight_json.suffix in {".yaml", ".yml"}
            else _load_json(preflight_json)
        )
    else:
        issues.append("missing_preflight_json")
    if current_template_provenance.get("status") == "generated_template_cross_family":
        issues.append("template_generated_family_cross_family")

    stored_template_provenance = existing_payload.get("template_provenance")
    stored_template_path = str(existing_payload.get("template_path") or "").strip()
    stored_generated_config_dir = str(existing_payload.get("generated_config_dir") or "").strip()
    stored_bundle_path = str(existing_payload.get("bundle_path") or "").strip()
    lane_rows = existing_payload.get("lane_rows") if isinstance(existing_payload, dict) else None
    stored_lane_row_count = len(lane_rows) if isinstance(lane_rows, list) else 0

    if existing_payload:
        if stored_bundle_path and stored_bundle_path != str(bundle_path):
            issues.append("bundle_path_mismatch")
        if stored_template_path and stored_template_path != str(template_path):
            issues.append("template_path_mismatch")
        if stored_generated_config_dir and stored_generated_config_dir != str(generated_root):
            issues.append("generated_config_dir_mismatch")
        if not stored_template_provenance:
            issues.append("missing_template_provenance")
        elif stored_template_provenance != current_template_provenance:
            issues.append("template_provenance_mismatch")

    existing_note_date = _read_note_date(preflight_note_md)
    note_issue_target = (
        out_of_scope_issues
        if write_scope == WRITE_SCOPE_REPORT_DATA_ONLY
        else issues
    )
    if preflight_note_md is None or not preflight_note_md.exists():
        note_issue_target.append("missing_preflight_note")
    elif existing_note_date != expected_note_date:
        note_issue_target.append("preflight_note_date_mismatch")

    return {
        "bundle_path": str(bundle_path),
        "preflight_json": str(preflight_json),
        "preflight_json_exists": preflight_json.exists(),
        "preflight_note_md": str(preflight_note_md) if preflight_note_md is not None else "",
        "preflight_note_exists": bool(preflight_note_md and preflight_note_md.exists()),
        "write_scope": write_scope,
        "expected_note_date": expected_note_date,
        "existing_note_date": existing_note_date,
        "current_template_provenance": current_template_provenance,
        "stored_template_provenance": stored_template_provenance,
        "stored_bundle_path": stored_bundle_path,
        "stored_template_path": stored_template_path,
        "stored_generated_config_dir": stored_generated_config_dir,
        "stored_lane_row_count": stored_lane_row_count,
        "status": "aligned" if not issues else "stale",
        "issues": list(dict.fromkeys(issues)),
        "out_of_scope_issues": list(dict.fromkeys(out_of_scope_issues)),
    }


def _parse_row_blocker_list(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    normalized = str(payload).strip()
    return [normalized] if normalized else []


def _resolve_governance(bundle_path: Path, governance: dict[str, Any]) -> dict[str, Any]:
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


def _normalize_switches(raw: dict[str, Any]) -> dict[str, bool]:
    return {
        "meso": bool(raw.get("meso")),
        "flow": bool(raw.get("flow")),
        "det": bool(raw.get("det")),
        "sparse": bool(raw.get("sparse")),
        "phy": bool(raw.get("phy")),
    }


def _load_accuracy_rows(template_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    accuracy_cfg = template_cfg.get("accuracy") or {}
    source_csv = accuracy_cfg.get("source_csv") or accuracy_cfg.get("csv")
    source_path = _resolve_repo_path(source_csv)
    if source_path is None or not source_path.exists():
        return []
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _inspect_accuracy_source(template_cfg: dict[str, Any]) -> dict[str, Any]:
    rows = _load_accuracy_rows(template_cfg)
    experiments = sorted(
        {
            str(row.get("experiment_id") or "").strip()
            for row in rows
            if str(row.get("experiment_id") or "").strip()
        }
    )
    source_run_ids = sorted(
        {
            str(row.get("source_run_id") or row.get("run_id") or "").strip()
            for row in rows
            if str(row.get("source_run_id") or row.get("run_id") or "").strip()
        }
    )
    return {
        "row_count": len(rows),
        "experiment_ids": experiments,
        "source_run_ids": source_run_ids,
    }


def _variant_accuracy_context_run_id(variant: dict[str, Any], run_id: str) -> str:
    explicit = str(variant.get("accuracy_context_run_id") or "").strip()
    if explicit:
        return explicit
    return f"{run_id}_acc_s0"


def _materialize_variant(
    *,
    template_cfg: dict[str, Any],
    variant: dict[str, Any],
    run_prefix: str,
    required_device: str,
    launch_prefix: list[str],
    require_accuracy_context_match: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_descriptor = normalize_variant_descriptor(variant)
    experiment_id = str(variant_descriptor["internal_experiment_id"]).strip().upper()
    variant_id = str(variant_descriptor["variant_id"]).strip().upper()
    config_stub = str(variant_descriptor["config_stub"]).strip().lower()
    run_id = f"{run_prefix}_{config_stub}"
    switches = _normalize_switches(variant_descriptor.get("switches") or {})
    cfg = copy.deepcopy(template_cfg)

    run_cfg = cfg.get("run") or {}
    run_cfg["run_id"] = run_id
    run_cfg["experiment_id"] = experiment_id
    run_cfg["internal_experiment_id"] = experiment_id
    run_cfg["variant_id"] = variant_id
    run_cfg["public_module_stack"] = copy.deepcopy(variant_descriptor["public_module_stack"])
    run_cfg["device"] = required_device
    run_cfg["long_run_launch_prefix"] = list(launch_prefix)
    notes = str(run_cfg.get("notes") or "").strip()
    lane_label = str(variant_descriptor.get("lane_label") or "").strip()
    mechanism_focus = str(variant_descriptor.get("mechanism_focus") or "").strip()
    suffix = (
        f"surface_variant:{variant_id.lower()};"
        f"lane_label:{lane_label or variant_id.lower()};"
        f"mechanism_focus:{mechanism_focus or 'unspecified'}"
    )
    run_cfg["notes"] = f"{notes} {suffix}".strip()
    cfg["run"] = run_cfg

    cfg["switches"] = dict(switches)
    phase1_runner._sync_section_enabled(cfg, switches)  # type: ignore[attr-defined]
    default_module_cfg = variant_descriptor.get("default_module_cfg") or {}
    if isinstance(default_module_cfg, dict) and default_module_cfg:
        cfg = _merge_nested_dict(cfg, default_module_cfg)
        cfg["switches"] = dict(switches)
        phase1_runner._sync_section_enabled(cfg, switches)  # type: ignore[attr-defined]

    accuracy_cfg = cfg.get("accuracy") or {}
    accuracy_cfg["require_context_match"] = require_accuracy_context_match
    accuracy_cfg["context_run_id"] = (
        str(variant_descriptor.get("accuracy_context_run_id") or "").strip()
        or _variant_accuracy_context_run_id(variant, run_id)
    )
    cfg["accuracy"] = accuracy_cfg

    return cfg, {
        "variant_id": variant_id,
        "experiment_id": experiment_id,
        "internal_experiment_id": experiment_id,
        "run_id": run_id,
        "config_stub": config_stub,
        "lane_label": lane_label,
        "mechanism_focus": mechanism_focus,
        "public_module_stack": copy.deepcopy(variant_descriptor["public_module_stack"]),
        "accuracy_context_run_id": accuracy_cfg["context_run_id"],
        "switches": switches,
    }


def _validate_variant_contract(
    *,
    cfg: dict[str, Any],
    expected_switches: dict[str, bool],
    required_device: str,
) -> dict[str, Any]:
    run_cfg = cfg.get("run") or {}
    switches = _normalize_switches(cfg.get("switches") or {})
    meso_cfg = cfg.get("meso") or {}
    flow_cfg = cfg.get("flow") or {}
    sparse_cfg = cfg.get("sparse") or {}
    phy_cfg = cfg.get("phy") or {}
    sc_det_cfg = cfg.get("sc_det") or {}
    early_stop = sc_det_cfg.get("early_stop") or {}
    accuracy_cfg = cfg.get("accuracy") or {}
    launch_prefix = run_cfg.get("long_run_launch_prefix") or []

    mismatches: list[str] = []
    if switches != expected_switches:
        mismatches.append("switch_matrix_mismatch")
    if str(run_cfg.get("device") or "").strip() != required_device:
        mismatches.append("device_not_mps")
    if list(launch_prefix) != ["caffeinate", "-dimsu"]:
        mismatches.append("long_run_launch_prefix_mismatch")
    if bool(meso_cfg.get("enabled")) != expected_switches["meso"]:
        mismatches.append("meso_wiring_mismatch")
    if bool(flow_cfg.get("enabled")) != expected_switches["flow"]:
        mismatches.append("flow_wiring_mismatch")
    if bool(sparse_cfg.get("enabled")) != expected_switches["sparse"]:
        mismatches.append("sparse_wiring_mismatch")
    if bool(phy_cfg.get("enabled")) != expected_switches["phy"]:
        mismatches.append("phy_wiring_mismatch")
    if bool(early_stop.get("enabled")) != expected_switches["det"]:
        mismatches.append("det_wiring_mismatch")
    if not bool(accuracy_cfg.get("require_context_match")):
        mismatches.append("accuracy_context_match_not_required")
    return {
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
    }


def _find_accuracy_matches(
    *,
    accuracy_rows: list[dict[str, Any]],
    experiment_id: str,
    context_run_id: str,
) -> dict[str, Any]:
    experiment_matches = [
        row
        for row in accuracy_rows
        if str(row.get("experiment_id") or "").strip().upper() == experiment_id
    ]
    context_matches = [
        row
        for row in experiment_matches
        if str(row.get("source_run_id") or row.get("run_id") or "").strip() == context_run_id
    ]
    return {
        "experiment_match_count": len(experiment_matches),
        "context_match_count": len(context_matches),
        "baseline_match_count": sum(
            1 for row in context_matches if str(row.get("baseline") or "").strip().lower() == "true"
        ),
        "target_match_count": sum(
            1 for row in context_matches if str(row.get("baseline") or "").strip().lower() == "false"
        ),
    }


def _build_accuracy_context_preview(cfg: dict[str, Any], switches: dict[str, bool]) -> dict[str, Any]:
    run_cfg = cfg.get("run") or {}
    data_cfg = cfg.get("data") or {}
    sc_det_cfg = cfg.get("sc_det") or {}
    sparse_cfg = cfg.get("sparse") or {}
    outputs_cfg = cfg.get("outputs") or {}
    p1_align_cfg = cfg.get("p1_align") or {}
    out_root = phase1_runner.resolve_workspace_path(  # type: ignore[attr-defined]
        outputs_cfg.get("out_dir") or "results/runs",
        anchor=phase1_runner.ROOT_DIR,
    )
    out_dir = out_root / str(run_cfg.get("run_id") or "")
    p1_align_resolved = {
        "gaussian_noise_std_ref": float(
            p1_align_cfg.get("gaussian_noise_std_ref")
            or p1_align_cfg.get("gaussian_noise_sigma_lsb_ref")
            or p1_align_cfg.get("sigma_lsb_ref")
            or 0.0
        ),
        "crosstalk_alpha_ref": float(p1_align_cfg.get("crosstalk_alpha_ref") or 0.0),
    }
    return phase1_runner._build_accuracy_context(  # type: ignore[attr-defined]
        run_id=str(run_cfg.get("run_id") or ""),
        experiment_id=str(run_cfg.get("experiment_id") or "E0"),
        out_dir=out_dir,
        run_cfg=run_cfg,
        data_cfg=data_cfg,
        switches=switches,
        sc_det_cfg=sc_det_cfg,
        sparse_cfg=sparse_cfg,
        p1_align_resolved=p1_align_resolved,
        accuracy_cfg=cfg.get("accuracy") or {},
    )


def _assess_accuracy_context_contract(
    *,
    cfg: dict[str, Any],
    accuracy_rows: list[dict[str, Any]],
    switches: dict[str, bool],
) -> dict[str, Any]:
    model = str(((cfg.get("models") or {}).get("keys") or ["mobilevit_s"])[0])
    accuracy_context = _build_accuracy_context_preview(cfg, switches)
    baseline_row = phase1_runner._pick_accuracy_row(  # type: ignore[attr-defined]
        rows=accuracy_rows,
        model=model,
        accuracy_context=accuracy_context,
        baseline=True,
    )
    target_row = phase1_runner._pick_accuracy_row(  # type: ignore[attr-defined]
        rows=accuracy_rows,
        model=model,
        accuracy_context=accuracy_context,
        baseline=False,
    )
    baseline_mismatches = phase1_runner._accuracy_context_mismatches(  # type: ignore[attr-defined]
        baseline_row,
        accuracy_context,
        expected_baseline=True,
    )
    target_mismatches = phase1_runner._accuracy_context_mismatches(  # type: ignore[attr-defined]
        target_row,
        accuracy_context,
        expected_baseline=False,
    )
    return {
        "baseline_row_id": baseline_row.get("run_id") if baseline_row else "",
        "target_row_id": target_row.get("run_id") if target_row else "",
        "baseline_mismatches": baseline_mismatches,
        "target_mismatches": target_mismatches,
        "target_analysis_grade_ready": str((target_row or {}).get("analysis_grade_ready") or "").strip(),
        "target_analysis_grade_blockers": _parse_row_blocker_list(
            (target_row or {}).get("analysis_grade_blockers")
        ),
        "target_accuracy_evidence_tier": str((target_row or {}).get("accuracy_evidence_tier") or "").strip(),
        "target_truth_class": str((target_row or {}).get("bitstream_measurement_truth_class") or "").strip(),
    }


def _build_accuracy_contract_rows(
    *,
    cfg: dict[str, Any],
    variant_meta: dict[str, Any],
    accuracy_rows: list[dict[str, Any]],
    accuracy_matches: dict[str, Any],
    deny_accuracy_fallback_relabel: bool,
) -> list[dict[str, Any]]:
    switches = variant_meta["switches"]
    model = str(((cfg.get("models") or {}).get("keys") or ["mobilevit_s"])[0])
    accuracy_context = _build_accuracy_context_preview(cfg, switches)
    rows: list[dict[str, Any]] = []

    for expected_baseline, row_role in ((True, "baseline"), (False, "target")):
        selected_row = phase1_runner._pick_accuracy_row(  # type: ignore[attr-defined]
            rows=accuracy_rows,
            model=model,
            accuracy_context=accuracy_context,
            baseline=expected_baseline,
        )
        mismatches = phase1_runner._accuracy_context_mismatches(  # type: ignore[attr-defined]
            selected_row,
            accuracy_context,
            expected_baseline=expected_baseline,
        )
        if accuracy_matches["experiment_match_count"] <= 0:
            row_status = "missing_current_lane_row"
        elif selected_row is None:
            row_status = "missing_selected_row"
        elif mismatches:
            row_status = "context_backfill_required"
        else:
            row_status = "ready"

        selected_config_snapshot = str((selected_row or {}).get("config_snapshot") or "")
        expected_config_snapshot = str(accuracy_context.get("config_snapshot") or "")
        config_snapshot_contract_state = _resolve_config_snapshot_contract_state(
            expected_config_snapshot=expected_config_snapshot,
            selected_config_snapshot=selected_config_snapshot,
        )

        row_payload: dict[str, Any] = {
            "variant_id": variant_meta["variant_id"],
            "experiment_id": variant_meta["experiment_id"],
            "internal_experiment_id": variant_meta["internal_experiment_id"],
            "public_module_stack": json.dumps(
                variant_meta["public_module_stack"],
                ensure_ascii=False,
            ),
            "lane_label": variant_meta["lane_label"],
            "mechanism_focus": variant_meta["mechanism_focus"],
            "run_id": variant_meta["run_id"],
            "accuracy_context_run_id": variant_meta["accuracy_context_run_id"],
            "row_role": row_role,
            "expected_baseline": expected_baseline,
            "row_status": row_status,
            "relabel_forbidden": deny_accuracy_fallback_relabel,
            "experiment_match_count": accuracy_matches["experiment_match_count"],
            "context_match_count": accuracy_matches["context_match_count"],
            "baseline_match_count": accuracy_matches["baseline_match_count"],
            "target_match_count": accuracy_matches["target_match_count"],
            "selected_row_id": str((selected_row or {}).get("run_id") or ""),
            "selected_source_run_id": str(
                (selected_row or {}).get("source_run_id") or (selected_row or {}).get("run_id") or ""
            ),
            "selected_row_experiment_id": str((selected_row or {}).get("experiment_id") or ""),
            "selected_config_snapshot": selected_config_snapshot,
            "selected_config_snapshot_exists": _csv_safe(_path_exists(selected_config_snapshot)),
            "selected_top1": _csv_safe((selected_row or {}).get("top1")),
            "selected_top5": _csv_safe((selected_row or {}).get("top5")),
            "selected_measurement_window": str((selected_row or {}).get("measurement_window") or ""),
            "selected_truth_class": str(
                (selected_row or {}).get("bitstream_measurement_truth_class") or ""
            ),
            "context_mismatches": ";".join(mismatches),
            "expected_config_snapshot_exists": _csv_safe(_path_exists(expected_config_snapshot)),
            "config_snapshot_contract_state": config_snapshot_contract_state,
        }
        for field in ACCURACY_CONTEXT_FIELDS:
            row_payload[f"expected_{field}"] = _csv_safe(accuracy_context.get(field))
            row_payload[f"selected_{field}"] = _csv_safe((selected_row or {}).get(field))
        rows.append(row_payload)
    return rows


def _build_accuracy_scaffold_rows(contract_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scaffold_rows: list[dict[str, Any]] = []
    for row in contract_rows:
        scaffold_row = {
            "run_id": row["accuracy_context_run_id"],
            "source_run_id": row["accuracy_context_run_id"],
            "experiment_id": row["experiment_id"],
            "baseline": "True" if row["row_role"] == "baseline" else "False",
            "device": "mps",
            "measurement_window": "",
            "model": "mobilevit_s",
            "split": row["expected_split"],
            "workload": row["expected_workload"],
            "seed": row["expected_seed"],
            "config_snapshot": row["expected_config_snapshot"],
            "det_policy": row["expected_det_policy"],
            "det_k_signature": row["expected_det_k_signature"],
            "det_k_global": row["expected_det_k_global"],
            "sparse_tau_global": row["expected_sparse_tau_global"],
            "sparse_active_fraction": row["expected_sparse_active_fraction"],
            "gaussian_noise_std_ref": row["expected_gaussian_noise_std_ref"],
            "crosstalk_alpha_ref": row["expected_crosstalk_alpha_ref"],
            "top1": "",
            "top5": "",
            "top1_delta": "",
            "top5_delta": "",
            "measured_pass_elapsed_s": "",
            "measured_processed_samples": "",
            "latency_ms_per_sample": "",
            "notes": (
                f"true_sc_e0_e6_contract_scaffold:{row['lane_label']};"
                f"row_role:{row['row_role']};row_status:{row['row_status']};"
                f"config_snapshot_contract_state:{row.get('config_snapshot_contract_state') or 'unspecified'};"
                "do_not_relabel_historical_row"
            ),
        }
        scaffold_rows.append(scaffold_row)
    return scaffold_rows


def _resolve_analysis_grade_gate(
    *,
    accuracy_contract: dict[str, Any],
    governance: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not governance["analysis_grade_enabled"]:
        blockers.append("analysis_grade_disabled")
    if accuracy_contract["baseline_mismatches"] or accuracy_contract["target_mismatches"]:
        blockers.append("missing_analysis_grade_accuracy_row")
    blockers.extend(str(item) for item in accuracy_contract.get("target_analysis_grade_blockers") or [])
    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
    }


def _resolve_runtime_smoke_gate(
    *,
    governance: dict[str, Any],
    contract_status: str,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not governance["runtime_smoke_enabled"]:
        blockers.append("runtime_smoke_disabled")
    if contract_status != "pass":
        blockers.append("wiring_contract_failed")
    return {
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
    }


def _build_summary_surface_contract() -> dict[str, Any]:
    disabled_flow = phase1_runner._resolve_flow_summary_fields(  # type: ignore[attr-defined]
        flow_cfg={},
        flow_enabled=False,
        flow_buffer_peak_cycles=123,
        flow_buffer_peak_frac=0.75,
    )
    active_flow = phase1_runner._resolve_flow_summary_fields(  # type: ignore[attr-defined]
        flow_cfg={
            "scheduler_mode": "elastic_residency_v3",
            "reuse_policy": "operand_factored",
            "admission_policy": "reuse_first",
            "service_policy": "reuse_first",
            "exception_lane_policy": "spill",
            "buffer_depth": 2,
            "overlap_efficiency": 0.75,
            "staging_cost_scale": 1.0,
            "sync_penalty_scale": 1.0,
        },
        flow_enabled=True,
        flow_buffer_peak_cycles=321,
        flow_buffer_peak_frac=0.5,
    )
    disabled_evidence = phase1_runner._resolve_switch_gated_evidence_surface(  # type: ignore[attr-defined]
        enabled=False,
        cfg={"evidence_type": "calibrated_model", "calibration_source": "/tmp/ignored.json"},
    )
    active_evidence = phase1_runner._resolve_switch_gated_evidence_surface(  # type: ignore[attr-defined]
        enabled=True,
        cfg={"evidence_type": "heuristic_proxy", "calibration_source": "/tmp/active.json"},
    )

    mismatches: list[str] = []
    if disabled_flow["flow_model_mode"] != "disabled" or disabled_flow["flow_buffer_peak_cycles"] != 0:
        mismatches.append("flow_disabled_surface_not_fail_closed")
    if disabled_flow["flow_buffer_depth"] != 0 or disabled_flow["flow_overlap_efficiency"] != 0.0:
        mismatches.append("flow_disabled_fields_not_zeroed")
    active_flow_mode = str(active_flow.get("flow_model_mode") or "").strip()
    if active_flow_mode != "elastic_residency_v3":
        mismatches.append("flow_active_surface_missing")
    if active_flow["flow_buffer_depth"] != 2 or active_flow["flow_overlap_efficiency"] != 0.75:
        mismatches.append("flow_active_controls_not_preserved")
    if active_flow["flow_staging_cost_scale"] != 1.0 or active_flow["flow_sync_penalty_scale"] != 1.0:
        mismatches.append("flow_active_cost_scales_not_preserved")
    if active_flow["flow_buffer_peak_cycles"] != 321 or active_flow["flow_buffer_peak_frac"] != 0.5:
        mismatches.append("flow_active_peak_surface_not_preserved")
    if disabled_evidence != ("disabled", ""):
        mismatches.append("disabled_evidence_surface_not_explicit")
    if active_evidence != ("heuristic_proxy", "/tmp/active.json"):
        mismatches.append("active_evidence_surface_not_preserved")
    return {
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
        "disabled_flow_fields": disabled_flow,
        "active_flow_fields": active_flow,
        "disabled_evidence_surface": {
            "evidence": disabled_evidence[0],
            "calibration_source": disabled_evidence[1],
        },
        "active_evidence_surface": {
            "evidence": active_evidence[0],
            "calibration_source": active_evidence[1],
        },
    }


def _run_phase1_preflight(
    *,
    cfg_path: Path,
    python_bin: str,
) -> dict[str, Any]:
    command = [python_bin, str(Path(phase1_runner.__file__).resolve()), "--config", str(cfg_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    summary_path = ""
    if result.returncode == 0:
        cfg = _load_yaml(cfg_path)
        run_id = str(((cfg.get("run") or {}).get("run_id")) or cfg_path.stem)
        out_root = _resolve_repo_path(((cfg.get("outputs") or {}).get("out_dir")) or "")
        if out_root is not None:
            summary_path = str(out_root / run_id / "phase1_summary.csv")
    return {
        "status": "pass" if result.returncode == 0 else "fail",
        "returncode": result.returncode,
        "command": command,
        "stdout_tail": result.stdout.strip().splitlines()[-10:],
        "stderr_tail": result.stderr.strip().splitlines()[-10:],
        "summary_path": summary_path,
    }


def _load_existing_phase1_preflight_rows(
    preflight_json: Path | None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    if preflight_json is None or not preflight_json.exists():
        return {}
    payload = _load_yaml(preflight_json) if preflight_json.suffix in {".yaml", ".yml"} else _load_json(preflight_json)
    lane_rows = payload.get("lane_rows") or []
    if not isinstance(lane_rows, list):
        return {}
    preserved: dict[tuple[str, str, str], dict[str, Any]] = {}
    for lane in lane_rows:
        if not isinstance(lane, dict):
            continue
        experiment_id = str(lane.get("experiment_id") or "").strip().upper()
        run_id = str(lane.get("run_id") or "").strip()
        config_path = str(lane.get("config_path") or "").strip()
        phase1_preflight = lane.get("phase1_preflight") or {}
        if not experiment_id or not run_id or not isinstance(phase1_preflight, dict):
            continue
        preserved[(experiment_id, run_id, config_path)] = copy.deepcopy(phase1_preflight)
    return preserved


def _resolve_phase1_preflight_result(
    *,
    run_phase1_preflight: bool,
    cfg_path: Path,
    python_bin: str,
    existing_phase1_preflight: dict[tuple[str, str, str], dict[str, Any]],
    experiment_id: str,
    run_id: str,
) -> dict[str, Any]:
    if run_phase1_preflight:
        return _run_phase1_preflight(cfg_path=cfg_path, python_bin=python_bin)
    preserved = existing_phase1_preflight.get((experiment_id, run_id, str(cfg_path)))
    if preserved is not None:
        return copy.deepcopy(preserved)
    return {"status": "not_run"}


def build_bundle_artifacts(
    *,
    bundle_path: Path,
    python_bin: str,
    run_phase1_preflight: bool,
    write_scope: str = WRITE_SCOPE_FULL,
) -> dict[str, Any]:
    if write_scope not in {WRITE_SCOPE_FULL, WRITE_SCOPE_REPORT_DATA_ONLY}:
        raise SystemExit(f"Unsupported write scope: {write_scope!r}")
    if write_scope == WRITE_SCOPE_REPORT_DATA_ONLY and run_phase1_preflight:
        raise SystemExit(
            "Phase1 preflight is not supported with --write-scope report-data-only."
        )
    bundle = _load_yaml(bundle_path)
    meta = bundle.get("meta") or {}
    paths = bundle.get("paths") or {}
    governance = bundle.get("governance") or {}
    resolved_governance = _resolve_governance(bundle_path, governance)
    variants = bundle.get("variants") or []
    if not isinstance(variants, list) or not variants:
        raise SystemExit(f"Bundle has no variants: {bundle_path}")

    template_path = _resolve_repo_path(paths.get("template_yaml"))
    if template_path is None or not template_path.exists():
        raise SystemExit(f"Missing template_yaml for bundle: {bundle_path}")
    template_cfg = _load_yaml(template_path)
    write_generated_configs = write_scope == WRITE_SCOPE_FULL
    write_docs = write_scope == WRITE_SCOPE_FULL
    skipped_outputs: list[str] = []
    required_device = str(governance.get("required_device") or "mps").strip()
    launch_prefix = list(governance.get("long_run_launch_prefix") or ["caffeinate", "-dimsu"])
    require_accuracy_context_match = bool(governance.get("require_accuracy_context_match", True))
    deny_accuracy_fallback_relabel = bool(governance.get("deny_accuracy_fallback_relabel", True))
    generated_root = _resolve_repo_path(paths.get("generated_config_dir"))
    if generated_root is None:
        raise SystemExit(f"Missing generated_config_dir for bundle: {bundle_path}")
    preflight_json = _resolve_repo_path(paths.get("preflight_json"))
    existing_phase1_preflight = _load_existing_phase1_preflight_rows(preflight_json)

    accuracy_rows = _load_accuracy_rows(template_cfg)
    accuracy_source_summary = _inspect_accuracy_source(template_cfg)
    template_provenance = _summarize_template_provenance(
        template_path=template_path,
        generated_root=generated_root,
    )
    summary_surface_contract = _build_summary_surface_contract()

    matrix_rows: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    accuracy_contract_rows: list[dict[str, Any]] = []
    blockers: list[str] = []

    for variant in sorted(
        variants,
        key=lambda item: EXPERIMENT_ORDER.index(
            str(
                normalize_variant_descriptor(item)["internal_experiment_id"]
            ).upper()
        ),
    ):
        cfg, variant_meta = _materialize_variant(
            template_cfg=template_cfg,
            variant=variant,
            run_prefix=str(meta.get("run_prefix") or meta.get("tag") or "true_sc_e0_e6"),
            required_device=required_device,
            launch_prefix=launch_prefix,
            require_accuracy_context_match=require_accuracy_context_match,
        )
        experiment_id = variant_meta["experiment_id"]
        cfg_path = generated_root / f"{variant_meta['config_stub']}.yaml"
        if write_generated_configs:
            _write_yaml(cfg_path, cfg)
        else:
            skipped_outputs.append(str(cfg_path))

        contract = _validate_variant_contract(
            cfg=cfg,
            expected_switches=variant_meta["switches"],
            required_device=required_device,
        )
        accuracy_contract = _assess_accuracy_context_contract(
            cfg=cfg,
            accuracy_rows=accuracy_rows,
            switches=variant_meta["switches"],
        )
        accuracy_matches = _find_accuracy_matches(
            accuracy_rows=accuracy_rows,
            experiment_id=experiment_id,
            context_run_id=str(variant_meta["accuracy_context_run_id"]),
        )
        accuracy_contract_rows.extend(
            _build_accuracy_contract_rows(
                cfg=cfg,
                variant_meta=variant_meta,
                accuracy_rows=accuracy_rows,
                accuracy_matches=accuracy_matches,
                deny_accuracy_fallback_relabel=deny_accuracy_fallback_relabel,
            )
        )
        if accuracy_matches["experiment_match_count"] <= 0:
            accuracy_status = "missing_current_accuracy_row"
        elif accuracy_contract["baseline_mismatches"] or accuracy_contract["target_mismatches"]:
            accuracy_status = "context_match_incomplete"
        elif accuracy_matches["context_match_count"] > 0 and accuracy_matches["target_match_count"] > 0:
            accuracy_status = "ready"
        else:
            accuracy_status = "context_match_incomplete"

        phase1_result = _resolve_phase1_preflight_result(
            run_phase1_preflight=run_phase1_preflight,
            cfg_path=cfg_path,
            python_bin=python_bin,
            existing_phase1_preflight=existing_phase1_preflight,
            experiment_id=experiment_id,
            run_id=variant_meta["run_id"],
        )
        runtime_smoke_gate = _resolve_runtime_smoke_gate(
            governance=resolved_governance,
            contract_status=contract["status"],
        )
        analysis_grade_gate = _resolve_analysis_grade_gate(
            accuracy_contract=accuracy_contract,
            governance=resolved_governance,
        )

        lane_blockers: list[str] = []
        if contract["status"] != "pass":
            lane_blockers.extend(contract["mismatches"])
        if summary_surface_contract["status"] != "pass":
            lane_blockers.extend(summary_surface_contract["mismatches"])
        if accuracy_status != "ready":
            lane_blockers.append(accuracy_status)
        if phase1_result.get("status") == "fail":
            lane_blockers.append("phase1_preflight_failed")
        if lane_blockers:
            blockers.append(f"{experiment_id}:{';'.join(lane_blockers)}")

        matrix_row = {
            "variant_id": variant_meta["variant_id"],
            "experiment_id": experiment_id,
            "internal_experiment_id": variant_meta["internal_experiment_id"],
            "public_module_stack": json.dumps(
                variant_meta["public_module_stack"],
                ensure_ascii=False,
            ),
            "lane_label": variant_meta["lane_label"],
            "mechanism_focus": variant_meta["mechanism_focus"],
            "run_id": variant_meta["run_id"],
            "config_path": str(cfg_path),
            "accuracy_context_run_id": variant_meta["accuracy_context_run_id"],
            "runtime_smoke_gate": runtime_smoke_gate["status"],
            "analysis_grade_gate": analysis_grade_gate["status"],
            "accuracy_status": accuracy_status,
            "wiring_status": contract["status"],
            **variant_meta["switches"],
        }
        matrix_rows.append(matrix_row)
        preflight_rows.append(
            {
                **matrix_row,
                "wiring_mismatches": contract["mismatches"],
                "accuracy_matches": accuracy_matches,
                "accuracy_context_contract": accuracy_contract,
                "runtime_smoke_gate": runtime_smoke_gate,
                "analysis_grade_gate": analysis_grade_gate,
                "phase1_preflight": phase1_result,
                "lane_blockers": lane_blockers,
            }
        )

    switch_matrix_csv = _resolve_repo_path(paths.get("switch_matrix_csv"))
    switch_matrix_json = _resolve_repo_path(paths.get("switch_matrix_json"))
    accuracy_contract_csv = _resolve_repo_path(paths.get("accuracy_contract_csv"))
    accuracy_contract_json = _resolve_repo_path(paths.get("accuracy_contract_json"))
    accuracy_scaffold_csv = _resolve_repo_path(paths.get("accuracy_scaffold_csv"))
    preflight_note_md = _resolve_repo_path(paths.get("preflight_note_md"))
    if (
        switch_matrix_csv is None
        or switch_matrix_json is None
        or accuracy_contract_csv is None
        or accuracy_contract_json is None
        or accuracy_scaffold_csv is None
        or preflight_json is None
        or preflight_note_md is None
    ):
        raise SystemExit(f"Bundle is missing output paths: {bundle_path}")

    _write_csv(
        switch_matrix_csv,
        matrix_rows,
        [
            "variant_id",
            "experiment_id",
            "internal_experiment_id",
            "public_module_stack",
            "lane_label",
            "mechanism_focus",
            "run_id",
            "config_path",
            "accuracy_context_run_id",
            "runtime_smoke_gate",
            "analysis_grade_gate",
            "accuracy_status",
            "wiring_status",
            "meso",
            "flow",
            "det",
            "sparse",
            "phy",
        ],
    )
    _write_json(switch_matrix_json, {"rows": matrix_rows})
    _write_csv(
        accuracy_contract_csv,
        accuracy_contract_rows,
        [
            "variant_id",
            "experiment_id",
            "internal_experiment_id",
            "public_module_stack",
            "lane_label",
            "mechanism_focus",
            "run_id",
            "accuracy_context_run_id",
            "row_role",
            "expected_baseline",
            "row_status",
            "relabel_forbidden",
            "experiment_match_count",
            "context_match_count",
            "baseline_match_count",
            "target_match_count",
            "selected_row_id",
            "selected_source_run_id",
            "selected_row_experiment_id",
            "selected_config_snapshot",
            "selected_config_snapshot_exists",
            "selected_top1",
            "selected_top5",
            "selected_measurement_window",
            "selected_truth_class",
            "context_mismatches",
            "expected_config_snapshot_exists",
            "config_snapshot_contract_state",
            *[f"expected_{field}" for field in ACCURACY_CONTEXT_FIELDS],
            *[f"selected_{field}" for field in ACCURACY_CONTEXT_FIELDS],
        ],
    )
    _write_json(accuracy_contract_json, {"rows": accuracy_contract_rows})
    _write_csv(
        accuracy_scaffold_csv,
        _build_accuracy_scaffold_rows(accuracy_contract_rows),
        [
            "run_id",
            "source_run_id",
            "experiment_id",
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
        ],
    )

    preflight_payload = {
        "bundle_path": str(bundle_path),
        "template_path": str(template_path),
        "generated_config_dir": str(generated_root),
        "governance": resolved_governance,
        "template_provenance": template_provenance,
        "accuracy_source_summary": accuracy_source_summary,
        "accuracy_contract_csv": str(accuracy_contract_csv),
        "accuracy_contract_json": str(accuracy_contract_json),
        "accuracy_scaffold_csv": str(accuracy_scaffold_csv),
        "accuracy_contract_rows": accuracy_contract_rows,
        "summary_surface_contract": summary_surface_contract,
        "lane_rows": preflight_rows,
        "global_blockers": blockers,
        "write_scope": write_scope,
        "skipped_outputs": sorted(dict.fromkeys(skipped_outputs)),
    }
    _write_json(preflight_json, preflight_payload)

    surface_display_name = _surface_display_name(meta, "True SC E0-E6")
    bundle_tool_command = _bundle_tool_command(meta)
    phase1_runner_command = _phase1_runner_command(meta)
    baseline_config_path = next(
        (
            row["config_path"]
            for row in preflight_rows
            if str(row.get("variant_id") or "").upper() == "ASTRA"
            or str(row.get("internal_experiment_id") or "").upper() == "E0"
        ),
        str(generated_root / "e0.yaml"),
    )
    note_lines = [
        f"# {surface_display_name} Preflight Note",
        "",
        f"Date: `{_resolve_bundle_note_date(meta, bundle_path)}`",
        f"Bundle: `{bundle_path}`",
        f"Template: `{template_path}`",
        f"Switch matrix CSV: `{switch_matrix_csv}`",
        f"Switch matrix JSON: `{switch_matrix_json}`",
        f"Accuracy contract CSV: `{accuracy_contract_csv}`",
        f"Accuracy contract JSON: `{accuracy_contract_json}`",
        f"Accuracy scaffold CSV: `{accuracy_scaffold_csv}`",
        "",
        "## Static Contract",
        "",
        f"- summary-surface contract: `{summary_surface_contract['status']}`",
        f"- template provenance: `{template_provenance['status']}`",
        f"- template generated family: `{template_provenance['template_generated_family'] or 'repo_template'}`",
        f"- bundle generated family: `{template_provenance['bundle_generated_family'] or 'none'}`",
        f"- accuracy source experiments: `{', '.join(accuracy_source_summary['experiment_ids']) or 'none'}`",
        f"- accuracy source rows: `{accuracy_source_summary['row_count']}`",
        "",
        "## Lane Status",
        "",
    ]
    for row in preflight_rows:
        note_lines.append(
            f"- `{row.get('variant_id') or row['experiment_id']}` "
            f"(internal=`{row.get('internal_experiment_id') or row['experiment_id']}`) "
            f"`{row['lane_label']}`: wiring=`{row['wiring_status']}` "
            f"accuracy=`{row['accuracy_status']}` runtime_smoke_gate=`{row['runtime_smoke_gate']['status']}` "
            f"analysis_grade_gate=`{row['analysis_grade_gate']['status']}` phase1=`{row['phase1_preflight']['status']}`"
        )
        if row["lane_blockers"]:
            note_lines.append(f"  blockers: `{'; '.join(row['lane_blockers'])}`")
        runtime_smoke_blockers = list(row["runtime_smoke_gate"].get("blockers") or [])
        analysis_grade_blockers = list(row["analysis_grade_gate"].get("blockers") or [])
        if runtime_smoke_blockers:
            note_lines.append(f"  runtime_smoke_blockers: `{'; '.join(runtime_smoke_blockers)}`")
        if analysis_grade_blockers:
            note_lines.append(f"  analysis_grade_blockers: `{'; '.join(analysis_grade_blockers)}`")
    note_lines.extend(["", "## Accuracy Row Contract", ""])
    for row in accuracy_contract_rows:
        note_lines.append(
            f"- `{row.get('variant_id') or row['experiment_id']}` "
            f"(internal=`{row['experiment_id']}`) `{row['row_role']}`: "
            f"status=`{row['row_status']}` "
            f"selected=`{row['selected_row_id'] or 'none'}` mismatches=`{row['context_mismatches'] or 'none'}` "
            f"snapshot_state=`{row['config_snapshot_contract_state']}`"
        )
    note_lines.extend(["", "## Formal Full-Rerun Blockers", ""])
    if blockers:
        for blocker in blockers:
            note_lines.append(f"- `{blocker}`")
    else:
        note_lines.append("- none")
    note_lines.extend(
        [
            "",
            "## Direct Run Entrypoints",
            "",
            f"- Materialize/update bundle: `{bundle_tool_command} --bundle {bundle_path}`",
            f"- Run static+phase1 preflight: `{bundle_tool_command} --bundle {bundle_path} --run-phase1-preflight`",
            f"- Run a single lane after contract closure: `{phase1_runner_command} --config {baseline_config_path}`",
        ]
    )
    if write_docs:
        preflight_note_md.parent.mkdir(parents=True, exist_ok=True)
        preflight_note_md.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    else:
        skipped_outputs.append(str(preflight_note_md))
        preflight_payload["skipped_outputs"] = sorted(dict.fromkeys(skipped_outputs))
        _write_json(preflight_json, preflight_payload)
    return preflight_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and preflight the canonical true-SC E0-E6 bundle."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--python_bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument(
        "--write-scope",
        choices=(WRITE_SCOPE_FULL, WRITE_SCOPE_REPORT_DATA_ONLY),
        default=WRITE_SCOPE_FULL,
        help="Write full outputs or only owned report-data artifacts.",
    )
    parser.add_argument("--inspect-existing-preflight", action="store_true")
    parser.add_argument("--run-phase1-preflight", action="store_true")
    args = parser.parse_args()

    bundle_path = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    if args.inspect_existing_preflight:
        payload = inspect_existing_preflight(
            bundle_path=bundle_path,
            write_scope=str(args.write_scope),
        )
    else:
        payload = build_bundle_artifacts(
            bundle_path=bundle_path,
            python_bin=str(args.python_bin),
            run_phase1_preflight=bool(args.run_phase1_preflight),
            write_scope=str(args.write_scope),
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
