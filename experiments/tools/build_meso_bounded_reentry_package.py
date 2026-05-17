#!/usr/bin/env python3
"""Build a calibrated bounded MESO re-entry package from retained freeze anchors."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from exp_common.meso_cost_model import (  # noqa: E402
    OVERHEAD_POLICY_BROADCAST_DRIVER_FRACTION,
    compute_meso_cost_model,
    resolve_meso_topology_dimension,
    summarize_meso_reuse_provenance,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _display_report_path(path_value: Any, *, repo_root: Path = ROOT) -> str:
    raw_value = str(path_value or "").strip()
    if not raw_value:
        return ""

    resolved_repo_root = repo_root.resolve()
    display_parts: list[str] = []
    for raw_part in raw_value.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        candidate = Path(part).expanduser()
        if not candidate.is_absolute():
            display_parts.append(part)
            continue
        try:
            display_parts.append(str(candidate.resolve().relative_to(resolved_repo_root)))
        except ValueError:
            display_parts.append(str(candidate))
    return ";".join(display_parts)


def _resolve_path_field(path_value: Any, *, cfg_path: Path, repo_root: Path = ROOT) -> str:
    raw_value = str(path_value or "").strip()
    if not raw_value:
        return ""

    resolved_parts: list[str] = []
    for raw_part in raw_value.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        candidate = Path(part).expanduser()
        if candidate.is_absolute():
            resolved_parts.append(str(candidate))
            continue
        # Repo-root-relative is the default for package inputs; explicit ./ or ../
        # keeps adjacent config fixtures usable in tests and local probes.
        if part.startswith("./") or part.startswith("../"):
            resolved_parts.append(str((cfg_path.parent / candidate).resolve()))
        else:
            resolved_parts.append(str((repo_root / candidate).resolve()))
    return ";".join(resolved_parts)


def _resolve_cfg_paths(
    cfg: dict[str, Any], *, cfg_path: Path, repo_root: Path = ROOT
) -> dict[str, Any]:
    resolved_cfg = dict(cfg)

    inputs = dict(cfg.get("inputs") or {})
    for key, value in list(inputs.items()):
        if isinstance(value, str):
            inputs[key] = _resolve_path_field(value, cfg_path=cfg_path, repo_root=repo_root)
    resolved_cfg["inputs"] = inputs

    meso_cfg = dict(cfg.get("meso") or {})
    for key in ("reuse_provenance_csv", "calibration_source"):
        value = meso_cfg.get(key)
        if isinstance(value, str):
            meso_cfg[key] = _resolve_path_field(value, cfg_path=cfg_path, repo_root=repo_root)
    resolved_cfg["meso"] = meso_cfg

    return resolved_cfg


def _row_by_experiment(rows: list[dict[str, str]], experiment_id: str) -> dict[str, str]:
    for row in rows:
        if str(row.get("experiment_id") or "").strip() == experiment_id:
            return row
    raise SystemExit(f"Missing experiment_id={experiment_id} row")


def _extract_markdown_section(markdown: str, heading: str) -> list[str]:
    lines = markdown.splitlines()
    capture = False
    heading_level = len(heading) - len(heading.lstrip("#"))
    section: list[str] = []
    for line in lines:
        if line.strip() == heading:
            capture = True
            continue
        if capture:
            stripped = line.strip()
            if stripped.startswith("#"):
                line_level = len(stripped) - len(stripped.lstrip("#"))
                if line_level <= heading_level:
                    break
            section.append(line.rstrip())
    return [line for line in section if line.strip()]


def _resolve_det_correlation_blocker_reason(
    *,
    note_present: bool,
    dedicated_lane_present: bool,
    det_correlation_closure_ready: bool,
) -> str:
    if not note_present:
        return "missing_det_correlation_closure_note"
    if det_correlation_closure_ready:
        return ""
    if not dedicated_lane_present:
        return "missing_dedicated_meso_det_correlation_lane"
    return "meso_det_correlation_not_closed"


def _build_det_correlation_intake(cfg: dict[str, Any]) -> dict[str, Any]:
    inputs = cfg.get("inputs") or {}
    note_value = str(inputs.get("det_correlation_closure_md") or "").strip()
    if not note_value:
        return {
            "source_note": "",
            "note_present": False,
            "runner_support_present": False,
            "independent_accuracy_closure_ready": False,
            "analysis_grade_backfill_executed": False,
            "dedicated_lane_present": False,
            "det_correlation_closure_ready": False,
            "det_correlation_intake_status": "blocked",
            "det_correlation_blocker_reason": "missing_det_correlation_closure_note",
            "det_correlation_detail": "DET correlation closure note was not provided.",
            "blocker_reasons": ["missing_det_correlation_closure_note"],
            "final_judgment": [],
            "next_work_package": [],
        }

    note_path = Path(note_value)
    if not note_path.exists():
        return {
            "source_note": note_value,
            "note_present": False,
            "runner_support_present": False,
            "independent_accuracy_closure_ready": False,
            "analysis_grade_backfill_executed": False,
            "dedicated_lane_present": False,
            "det_correlation_closure_ready": False,
            "det_correlation_intake_status": "blocked",
            "det_correlation_blocker_reason": "missing_det_correlation_closure_note",
            "det_correlation_detail": "DET correlation closure note path does not exist.",
            "blocker_reasons": ["missing_det_correlation_closure_note"],
            "final_judgment": [],
            "next_work_package": [],
        }

    markdown = _read_text(note_path)
    lowered = markdown.lower()
    det_section = _extract_markdown_section(
        markdown,
        "#### E. MESO x DET correlation evidence is still absent as a dedicated lane",
    )
    det_conclusion = next(
        (line for line in det_section if line.startswith("Conclusion:")),
        "",
    )
    if det_conclusion:
        det_detail = det_conclusion.replace("Conclusion:", "", 1).strip()
    else:
        fallback = next(
            (
                line.strip()
                for line in markdown.splitlines()
                if line.strip().startswith("4. The blocker is")
            ),
            "",
        )
        det_detail = fallback.replace("4. ", "", 1).strip()

    dedicated_lane_present = (
        "does not currently have a dedicated `meso+det` lane" not in det_detail.lower()
        and "no dedicated `meso+det` lane" not in det_detail.lower()
    )
    det_correlation_closure_ready = (
        "meso x det correlation evidence is also not closed" not in lowered
        and dedicated_lane_present
    )
    independent_accuracy_closure_ready = (
        "independent meso-specific evidence is still not closed" not in lowered
    )
    analysis_grade_backfill_executed = (
        "analysis-grade backfill was prepared but not executed" not in lowered
    )
    runner_support_present = "the blocker is not missing runner support" in lowered

    blocker_reasons: list[str] = []
    if not independent_accuracy_closure_ready:
        blocker_reasons.append("independent_meso_accuracy_not_closed")
    if not analysis_grade_backfill_executed:
        blocker_reasons.append("analysis_grade_e1_backfill_not_executed")
    if not dedicated_lane_present:
        blocker_reasons.append("no_dedicated_meso_det_lane")
    if not det_correlation_closure_ready:
        blocker_reasons.append("meso_det_correlation_not_closed")

    return {
        "source_note": str(note_path),
        "note_present": True,
        "runner_support_present": runner_support_present,
        "independent_accuracy_closure_ready": independent_accuracy_closure_ready,
        "analysis_grade_backfill_executed": analysis_grade_backfill_executed,
        "dedicated_lane_present": dedicated_lane_present,
        "det_correlation_closure_ready": det_correlation_closure_ready,
        "det_correlation_intake_status": (
            "ready" if det_correlation_closure_ready else "blocked"
        ),
        "det_correlation_blocker_reason": _resolve_det_correlation_blocker_reason(
            note_present=True,
            dedicated_lane_present=dedicated_lane_present,
            det_correlation_closure_ready=det_correlation_closure_ready,
        ),
        "det_correlation_detail": det_detail,
        "blocker_reasons": blocker_reasons,
        "final_judgment": _extract_markdown_section(markdown, "### Final Judgment"),
        "next_work_package": _extract_markdown_section(
            markdown,
            "### Smallest Credible Next Work Package",
        ),
    }


def _parse_seed_suffix(value: Any) -> int | None:
    match = re.search(r"_s(\d+)$", str(value or "").strip())
    if match is None:
        return None
    return int(match.group(1))


def _append_unique_blocker(blockers: list[str], blocker: str) -> None:
    blocker_text = str(blocker or "").strip()
    if blocker_text and blocker_text not in blockers:
        blockers.append(blocker_text)


def _resolve_job_artifact_path(
    value: Any,
    *,
    base_dir: Path,
    repo_root: Path = ROOT,
) -> Path | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None

    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate
    if raw_value.startswith("./") or raw_value.startswith("../"):
        return (base_dir / candidate).resolve()
    return (repo_root / candidate).resolve()


def _build_analysis_grade_job_blockers(
    jobs: list[dict[str, Any]],
    *,
    expected_seeds: list[int] | None = None,
    manifest_dir: Path,
) -> list[str]:
    blockers: list[str] = []
    resolved_seeds: list[int] = []

    for index, job in enumerate(jobs):
        job_label = str(job.get("eval_run_id") or job.get("step_id") or f"job_{index}")
        seed = job.get("seed")
        if seed is None:
            _append_unique_blocker(blockers, f"analysis_grade_job_missing_seed:{job_label}")
        else:
            if seed in resolved_seeds:
                _append_unique_blocker(blockers, f"duplicate_analysis_grade_job_seed:{seed}")
            resolved_seeds.append(int(seed))

        for field_name in (
            "prepared_phase1_config",
            "prepared_eligibility_report_json",
            "bitstream_truth_class_authorization_note",
            "command",
        ):
            if not str(job.get(field_name) or "").strip():
                _append_unique_blocker(
                    blockers,
                    f"analysis_grade_job_missing_{field_name}:{job_label}",
                )
        for field_name in (
            "prepared_phase1_config",
            "prepared_eligibility_report_json",
            "bitstream_truth_class_authorization_note",
        ):
            artifact_path = _resolve_job_artifact_path(
                job.get(field_name),
                base_dir=manifest_dir,
            )
            if artifact_path is not None and not artifact_path.exists():
                _append_unique_blocker(
                    blockers,
                    f"analysis_grade_job_missing_file:{field_name}:{job_label}",
                )

    if expected_seeds is not None:
        expected_seed_set = {int(seed) for seed in expected_seeds}
        actual_seed_set = set(resolved_seeds)
        missing = sorted(expected_seed_set - actual_seed_set)
        unexpected = sorted(actual_seed_set - expected_seed_set)
        if missing or unexpected:
            mismatch_parts: list[str] = []
            if missing:
                mismatch_parts.append("missing=" + ",".join(str(seed) for seed in missing))
            if unexpected:
                mismatch_parts.append("unexpected=" + ",".join(str(seed) for seed in unexpected))
            _append_unique_blocker(
                blockers,
                "analysis_grade_job_seed_mismatch:" + ";".join(mismatch_parts),
            )

    return blockers


def _build_analysis_grade_closure_package(
    cfg: dict[str, Any],
    *,
    expected_seeds: list[int] | None = None,
) -> dict[str, Any]:
    inputs = cfg.get("inputs") or {}
    manifest_value = str(inputs.get("analysis_grade_closure_manifest_json") or "").strip()
    if not manifest_value:
        return {
            "source_manifest": "",
            "status": "blocked",
            "analysis_grade_ready": False,
            "analysis_grade_blockers": ["missing_analysis_grade_closure_manifest"],
            "launch_prefix": ["caffeinate", "-dimsu"],
            "jobs": [],
        }

    manifest_path = Path(manifest_value)
    if not manifest_path.exists():
        return {
            "source_manifest": manifest_value,
            "status": "blocked",
            "analysis_grade_ready": False,
            "analysis_grade_blockers": ["missing_analysis_grade_closure_manifest"],
            "launch_prefix": ["caffeinate", "-dimsu"],
            "jobs": [],
        }

    manifest = _load_json(manifest_path)
    raw_jobs = manifest.get("jobs") or []
    jobs: list[dict[str, Any]] = []
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            continue
        eval_run_id = str(raw_job.get("eval_run_id") or "")
        jobs.append(
            {
                "step_id": str(raw_job.get("step_id") or ""),
                "run_id": str(raw_job.get("run_id") or ""),
                "eval_run_id": eval_run_id,
                "seed": _parse_seed_suffix(eval_run_id),
                "model": str(raw_job.get("model") or ""),
                "experiment_id": str(raw_job.get("experiment_id") or ""),
                "config_path": str(raw_job.get("config_path") or ""),
                "prepared_phase1_config": str(raw_job.get("prepared_phase1_config") or ""),
                "prepared_eligibility_report_json": str(
                    raw_job.get("prepared_eligibility_report_json") or ""
                ),
                "bitstream_truth_class_authorization_note": str(
                    raw_job.get("bitstream_truth_class_authorization_note") or ""
                ),
                "analysis_grade_ready": bool(
                    raw_job.get("analysis_grade_ready", manifest.get("analysis_grade_ready"))
                ),
                "planned_pass_count": int(raw_job.get("planned_pass_count") or 0),
                "command": str(raw_job.get("command") or ""),
            }
        )

    blockers = [str(item) for item in (manifest.get("analysis_grade_blockers") or []) if str(item)]
    if not jobs:
        _append_unique_blocker(blockers, "missing_analysis_grade_jobs")
    for blocker in _build_analysis_grade_job_blockers(
        jobs,
        expected_seeds=expected_seeds,
        manifest_dir=manifest_path.parent,
    ):
        _append_unique_blocker(blockers, blocker)
    analysis_grade_ready = (
        bool(manifest.get("analysis_grade_ready"))
        and bool(jobs)
        and all(bool(job["analysis_grade_ready"]) for job in jobs)
        and not blockers
    )

    return {
        "source_manifest": str(manifest_path),
        "status": "ready" if analysis_grade_ready else "blocked",
        "analysis_grade_ready": analysis_grade_ready,
        "analysis_grade_blockers": blockers,
        "launch_prefix": ["caffeinate", "-dimsu"],
        "results_csv": str(manifest.get("results_csv") or ""),
        "annotated_results_csv": str(manifest.get("annotated_results_csv") or ""),
        "prepared_phase1_config_root": str(manifest.get("prepared_phase1_config_root") or ""),
        "progress_root": str(manifest.get("progress_root") or ""),
        "jobs": jobs,
    }


def _append_run_notes(existing: Any, suffix: str) -> str:
    existing_text = str(existing or "").strip()
    suffix_text = suffix.strip()
    if not suffix_text:
        return existing_text
    if not existing_text:
        return suffix_text
    if suffix_text in existing_text:
        return existing_text
    return f"{existing_text}; {suffix_text}"


def _compose_phase3_handoff_run_notes(*, lane_kind: str, notes_suffix: str) -> str:
    lane_text_by_kind = {
        "measured_closure": "dedicated Phase3 E1 measured-closure lane; mechanism_focus:meso_only",
        "meso_det_correlation": (
            "dedicated Phase3 E1 MESO+DET correlation lane; mechanism_focus:meso_det"
        ),
    }
    base_text = lane_text_by_kind.get(lane_kind)
    if base_text is None:
        raise SystemExit(f"Unsupported phase3 handoff lane kind: {lane_kind}")
    return _append_run_notes(base_text, notes_suffix)


def _project_handoff_inputs(
    *,
    lane_cfg: dict[str, Any],
    cfg_inputs: dict[str, Any],
    effective_meso_cfg: dict[str, Any],
) -> None:
    inputs_cfg = dict(lane_cfg.get("inputs") or {})
    reuse_provenance_csv = str(
        effective_meso_cfg.get("reuse_provenance_csv")
        or cfg_inputs.get("reuse_provenance_csv")
        or ""
    ).strip()
    if reuse_provenance_csv:
        inputs_cfg["reuse_provenance_csv"] = reuse_provenance_csv
    calibration_source = str(effective_meso_cfg.get("calibration_source") or "").strip()
    if calibration_source:
        inputs_cfg["calibration_source"] = calibration_source
    if inputs_cfg:
        lane_cfg["inputs"] = inputs_cfg


def _build_meso_det_correlation_lane(
    *,
    cfg: dict[str, Any],
    effective_meso_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = cfg.get("inputs") or {}
    exec_cfg = dict((cfg.get("execution_package") or {}).get("correlation_lane") or {})
    base_config_value = str(inputs.get("correlation_base_config_yaml") or "").strip()
    if not base_config_value:
        raise SystemExit("Expected inputs.correlation_base_config_yaml for correlation lane packaging")

    base_config_path = Path(base_config_value)
    if not base_config_path.exists():
        raise SystemExit(f"Missing correlation base config: {base_config_path}")

    base_cfg = _load_yaml(base_config_path)
    lane_cfg = copy.deepcopy(base_cfg)

    run_cfg = dict(lane_cfg.get("run") or {})
    switches_cfg = dict(lane_cfg.get("switches") or {})
    accuracy_cfg = dict(lane_cfg.get("accuracy") or {})
    sc_det_cfg = dict(lane_cfg.get("sc_det") or {})
    early_stop_cfg = dict(sc_det_cfg.get("early_stop") or {})
    models_cfg = dict(lane_cfg.get("models") or {})

    run_id = str(exec_cfg.get("run_id") or "20260420_meso_det_correlation_e1").strip()
    experiment_id = str(exec_cfg.get("experiment_id") or "E1_MESO_DET").strip()
    eval_run_id_prefix = str(exec_cfg.get("eval_run_id_prefix") or f"{run_id}_acc").strip()
    if not eval_run_id_prefix:
        raise SystemExit("Expected non-empty correlation_lane.eval_run_id_prefix")
    notes_suffix = str(
        exec_cfg.get("notes_suffix") or "phase3_package:meso_det_correlation_minimal"
    ).strip()

    run_cfg["run_id"] = run_id
    run_cfg["experiment_id"] = experiment_id
    run_cfg["notes"] = _compose_phase3_handoff_run_notes(
        lane_kind="meso_det_correlation",
        notes_suffix=notes_suffix,
    )

    switches_cfg["meso"] = True
    switches_cfg["det"] = True

    accuracy_cfg["require_context_match"] = True
    accuracy_cfg["context_run_id"] = f"{eval_run_id_prefix}_s0"

    early_stop_cfg["enabled"] = True
    early_stop_cfg["k_global"] = int(exec_cfg.get("det_k_global") or 64)
    sc_det_cfg["early_stop"] = early_stop_cfg
    models_cfg["keys"] = [str(model) for model in (exec_cfg.get("models") or models_cfg.get("keys") or [])]

    lane_cfg["run"] = run_cfg
    lane_cfg["switches"] = switches_cfg
    lane_cfg["accuracy"] = accuracy_cfg
    lane_cfg["sc_det"] = sc_det_cfg
    lane_cfg["models"] = models_cfg

    effective_lane_meso_cfg = copy.deepcopy(effective_meso_cfg)
    effective_lane_meso_cfg["enabled"] = True
    lane_cfg["meso"] = effective_lane_meso_cfg
    _project_handoff_inputs(
        lane_cfg=lane_cfg,
        cfg_inputs=inputs,
        effective_meso_cfg=effective_lane_meso_cfg,
    )

    seeds = [int(seed) for seed in (exec_cfg.get("seeds") or [0, 1, 2])]
    output_config_name = str(exec_cfg.get("output_config_name") or "meso_det_correlation_config.yaml")
    if not output_config_name:
        raise SystemExit("Expected non-empty correlation_lane.output_config_name")

    return lane_cfg, {
        "status": "ready",
        "base_config": str(base_config_path),
        "output_config_name": output_config_name,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "eval_run_id_prefix": eval_run_id_prefix,
        "context_run_id_seed0": accuracy_cfg["context_run_id"],
        "seeds": seeds,
        "det_k_global": int(early_stop_cfg["k_global"]),
        "phase1_launch_required": True,
        "phase1_launch_prefix": ["caffeinate", "-dimsu"],
        "follow_on_accuracy_template_source": str(
            inputs.get("analysis_grade_closure_manifest_json") or ""
        ),
        }


def _strip_stale_det_lane_absence_detail(detail: str) -> str:
    cleaned = str(detail or "").strip()
    if not cleaned:
        return ""

    fragments = re.split(r"(?<=[.!?])\s+", cleaned)
    kept_fragments: list[str] = []
    for fragment in fragments:
        lowered = fragment.lower()
        if "dedicated `meso+det` lane" in lowered and (
            "does not currently have" in lowered
            or "no dedicated" in lowered
            or "still absent as a dedicated lane" in lowered
        ):
            continue
        kept_fragments.append(fragment.strip())
    return " ".join(fragment for fragment in kept_fragments if fragment)


def _build_packaged_det_correlation_detail(*, det_correlation_closure_ready: bool) -> str:
    if det_correlation_closure_ready:
        return (
            "packaged dedicated `MESO+DET` lane config is ready, and measured correlation "
            "evidence is closed."
        )
    return (
        "packaged dedicated `MESO+DET` lane config is ready, but measured correlation "
        "evidence is still not closed."
    )


_CLOSURE_LAUNCH_INPUT_LABELS = {
    "prepared_phase1_config": "prepared phase1 configs",
    "prepared_eligibility_report_json": "prepared eligibility reports",
    "bitstream_truth_class_authorization_note": "bitstream truth-class authorization notes",
}


def _format_human_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _extract_closure_launch_input_labels(closure_package: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for raw_blocker in closure_package.get("analysis_grade_blockers") or []:
        blocker = str(raw_blocker or "").strip()
        if not blocker:
            continue
        field_name = ""
        if blocker.startswith("analysis_grade_job_missing_file:"):
            parts = blocker.split(":", 2)
            if len(parts) >= 2:
                field_name = parts[1]
        elif blocker.startswith("analysis_grade_job_missing_"):
            field_name = blocker.split(":", 1)[0].replace("analysis_grade_job_missing_", "", 1)
        if not field_name:
            continue
        label = _CLOSURE_LAUNCH_INPUT_LABELS.get(field_name, field_name.replace("_", " "))
        if label not in labels:
            labels.append(label)
    return labels


def _reconcile_det_correlation_final_judgment(
    lines: list[Any],
    *,
    det_correlation_closure_ready: bool,
    closure_launch_input_labels: list[str],
) -> list[str]:
    reconciled_lines: list[str] = []
    for raw_line in lines:
        line = str(raw_line).strip()
        lowered = line.lower()
        if "analysis-grade backfill was prepared but not executed" in lowered:
            if closure_launch_input_labels:
                launch_inputs = _format_human_join(closure_launch_input_labels)
                reconciled_lines.append(
                    "- Independent MESO-specific evidence is still not closed because the "
                    "packaged analysis-grade `E1` closure lane is not launchable: the "
                    f"manifest-referenced {launch_inputs} are missing on disk."
                )
            else:
                reconciled_lines.append(line)
            continue
        if "there is no dedicated `meso+det` lane" in lowered:
            if det_correlation_closure_ready:
                reconciled_lines.append(
                    "- MESO x DET correlation evidence is closed on the packaged dedicated "
                    "`MESO+DET` lane."
                )
            else:
                reconciled_lines.append(
                    "- MESO x DET correlation evidence is also not closed because the packaged "
                    "dedicated `MESO+DET` lane is still unmeasured and the available MESO scans "
                    "keep `det=false` with `noise_correlation=0.0`."
                )
            continue
        if line:
            reconciled_lines.append(line)
    return reconciled_lines


def _reconcile_det_correlation_next_work_package(
    lines: list[Any],
    *,
    dedicated_lane_present: bool,
    det_correlation_closure_ready: bool,
    closure_launch_input_labels: list[str],
) -> list[str]:
    if det_correlation_closure_ready:
        return [str(line).rstrip() for line in lines if str(line).strip()]

    reconciled_lines: list[str] = []
    skip_stale_clone_block = False
    for raw_line in lines:
        line = str(raw_line).rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if dedicated_lane_present and stripped == "One bounded package should do both of the missing jobs:":
            reconciled_lines.append("The packaged handoff still needs two execution steps:")
            continue
        if (
            closure_launch_input_labels
            and stripped.startswith("1. Execute the prepared analysis-grade `E1` accuracy launch")
        ):
            launch_inputs = _format_human_join(closure_launch_input_labels)
            reconciled_lines.append(
                "1. Restore or regenerate the missing "
                f"{launch_inputs} referenced by the analysis-grade manifest for the "
                "packaged `E1` closure jobs, then execute the packaged measured-closure lane."
            )
            continue
        if dedicated_lane_present and stripped == "2. Clone `E1` into one minimal `MESO+DET` lane:":
            reconciled_lines.append(
                "2. Use the packaged minimal `MESO+DET` lane and run one "
                "correlation-focused accuracy check against that lane."
            )
            skip_stale_clone_block = True
            continue
        if skip_stale_clone_block:
            if stripped == "Why this is the minimum:":
                skip_stale_clone_block = False
                reconciled_lines.append(stripped)
            continue
        if (
            dedicated_lane_present
            and stripped == "- Step 2 creates the first isolatable MESO x DET correlation surface."
        ):
            reconciled_lines.append(
                "- Step 2 measures the already-packaged isolatable MESO x DET correlation "
                "surface."
            )
            continue
        reconciled_lines.append(line)
    return reconciled_lines


def _reconcile_det_correlation_intake(
    det_correlation_intake: dict[str, Any],
    *,
    correlation_package: dict[str, Any],
    closure_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reconciled = copy.deepcopy(det_correlation_intake)
    if not bool(reconciled.get("note_present")):
        return reconciled
    if str(correlation_package.get("status") or "").strip() != "ready":
        return reconciled
    closure_launch_input_labels = _extract_closure_launch_input_labels(closure_package or {})

    blockers = [
        str(blocker)
        for blocker in (reconciled.get("blocker_reasons") or [])
        if str(blocker) != "no_dedicated_meso_det_lane"
    ]
    if closure_launch_input_labels:
        blockers = [
            blocker
            for blocker in blockers
            if blocker != "analysis_grade_e1_backfill_not_executed"
        ]
        _append_unique_blocker(blockers, "analysis_grade_closure_launch_inputs_missing")
    dedicated_lane_present = True
    det_correlation_closure_ready = bool(reconciled.get("det_correlation_closure_ready"))
    if not det_correlation_closure_ready:
        _append_unique_blocker(blockers, "meso_det_correlation_not_closed")

    detail = _strip_stale_det_lane_absence_detail(
        str(reconciled.get("det_correlation_detail") or "").strip()
    )
    packaged_lane_detail = _build_packaged_det_correlation_detail(
        det_correlation_closure_ready=det_correlation_closure_ready
    )
    if not detail:
        detail = packaged_lane_detail
    elif packaged_lane_detail not in detail:
        detail = f"{packaged_lane_detail} Note context: {detail}"

    reconciled["dedicated_lane_present"] = dedicated_lane_present
    reconciled["det_correlation_closure_ready"] = det_correlation_closure_ready
    reconciled["det_correlation_intake_status"] = (
        "ready" if det_correlation_closure_ready else "blocked"
    )
    reconciled["det_correlation_blocker_reason"] = _resolve_det_correlation_blocker_reason(
        note_present=True,
        dedicated_lane_present=dedicated_lane_present,
        det_correlation_closure_ready=det_correlation_closure_ready,
    )
    reconciled["det_correlation_detail"] = detail
    reconciled["blocker_reasons"] = blockers
    reconciled["final_judgment"] = _reconcile_det_correlation_final_judgment(
        list(reconciled.get("final_judgment") or []),
        det_correlation_closure_ready=det_correlation_closure_ready,
        closure_launch_input_labels=closure_launch_input_labels,
    )
    reconciled["next_work_package"] = _reconcile_det_correlation_next_work_package(
        list(reconciled.get("next_work_package") or []),
        dedicated_lane_present=dedicated_lane_present,
        det_correlation_closure_ready=det_correlation_closure_ready,
        closure_launch_input_labels=closure_launch_input_labels,
    )
    return reconciled


def _build_meso_measured_closure_lane(
    *,
    cfg: dict[str, Any],
    effective_meso_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = cfg.get("inputs") or {}
    exec_cfg = dict((cfg.get("execution_package") or {}).get("closure_lane") or {})
    base_config_value = str(inputs.get("closure_base_config_yaml") or "").strip()
    if not base_config_value:
        raise SystemExit("Expected inputs.closure_base_config_yaml for closure lane packaging")

    base_config_path = Path(base_config_value)
    if not base_config_path.exists():
        raise SystemExit(f"Missing closure base config: {base_config_path}")

    base_cfg = _load_yaml(base_config_path)
    lane_cfg = copy.deepcopy(base_cfg)

    run_cfg = dict(lane_cfg.get("run") or {})
    switches_cfg = dict(lane_cfg.get("switches") or {})
    accuracy_cfg = dict(lane_cfg.get("accuracy") or {})

    run_id = str(exec_cfg.get("run_id") or "20260420_true_sc_meso_closure_e1").strip()
    experiment_id = str(exec_cfg.get("experiment_id") or run_cfg.get("experiment_id") or "E1").strip()
    eval_run_id_prefix = str(exec_cfg.get("eval_run_id_prefix") or f"{run_id}_acc").strip()
    if not eval_run_id_prefix:
        raise SystemExit("Expected non-empty closure_lane.eval_run_id_prefix")
    notes_suffix = str(
        exec_cfg.get("notes_suffix") or "phase3_package:meso_measured_closure"
    ).strip()

    run_cfg["run_id"] = run_id
    run_cfg["experiment_id"] = experiment_id
    run_cfg["notes"] = _compose_phase3_handoff_run_notes(
        lane_kind="measured_closure",
        notes_suffix=notes_suffix,
    )

    switches_cfg["meso"] = True
    switches_cfg["det"] = False
    switches_cfg["flow"] = bool(switches_cfg.get("flow", False))
    switches_cfg["sparse"] = bool(switches_cfg.get("sparse", False))
    switches_cfg["phy"] = bool(switches_cfg.get("phy", False))

    accuracy_cfg["require_context_match"] = True
    accuracy_cfg["context_run_id"] = f"{eval_run_id_prefix}_s0"

    lane_cfg["run"] = run_cfg
    lane_cfg["switches"] = switches_cfg
    lane_cfg["accuracy"] = accuracy_cfg

    effective_lane_meso_cfg = copy.deepcopy(effective_meso_cfg)
    effective_lane_meso_cfg["enabled"] = True
    lane_cfg["meso"] = effective_lane_meso_cfg
    _project_handoff_inputs(
        lane_cfg=lane_cfg,
        cfg_inputs=inputs,
        effective_meso_cfg=effective_lane_meso_cfg,
    )

    seeds = [int(seed) for seed in (exec_cfg.get("seeds") or [0, 1, 2])]
    output_config_name = str(exec_cfg.get("output_config_name") or "meso_measured_closure_config.yaml")
    if not output_config_name:
        raise SystemExit("Expected non-empty closure_lane.output_config_name")

    return lane_cfg, {
        "status": "ready",
        "base_config": str(base_config_path),
        "output_config_name": output_config_name,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "eval_run_id_prefix": eval_run_id_prefix,
        "context_run_id_seed0": accuracy_cfg["context_run_id"],
        "seeds": seeds,
        "phase1_launch_required": True,
        "phase1_launch_prefix": [
            str(item)
            for item in (run_cfg.get("long_run_launch_prefix") or ["caffeinate", "-dimsu"])
        ],
        "source_manifest": str(inputs.get("analysis_grade_closure_manifest_json") or ""),
    }


def _derive_power_calibration_from_sweep(fanout_sweep_csv: Path) -> dict[str, float | str]:
    rows = _read_csv(fanout_sweep_csv)
    if not rows:
        raise SystemExit(f"Empty MESO fanout sweep: {fanout_sweep_csv}")
    serializer_power_samples_mw: list[float] = []
    broadcast_power_samples_mw: list[float] = []
    for row in rows:
        serializers_saved = _to_float(row.get("serializers_saved"), 0.0) or 0.0
        net_gain_j = _to_float(row.get("net_energy_gain_j"), 0.0) or 0.0
        broadcast_driver_energy_j = _to_float(row.get("broadcast_driver_energy_j"), 0.0) or 0.0
        latency_ms = _to_float(row.get("latency_ms"), None)
        if latency_ms is None or latency_ms <= 0.0:
            continue
        latency_s = latency_ms / 1e3
        if serializers_saved > 0.0:
            serializer_energy_j = (net_gain_j + broadcast_driver_energy_j) / serializers_saved
            serializer_power_samples_mw.append(serializer_energy_j / latency_s * 1000.0)
        broadcast_power_samples_mw.append(broadcast_driver_energy_j / latency_s * 1000.0)
    if not serializer_power_samples_mw or not broadcast_power_samples_mw:
        raise SystemExit(f"Could not derive calibrated MESO power surface from {fanout_sweep_csv}")
    return {
        "cost_model_mode": "explicit_topology_v1",
        "evidence_type": "retained_model_calibrated",
        "calibration_source": str(fanout_sweep_csv),
        "serializer_power_mw": sum(serializer_power_samples_mw) / float(len(serializer_power_samples_mw)),
        "broadcast_driver_power_mw": sum(broadcast_power_samples_mw) / float(len(broadcast_power_samples_mw)),
    }


def _build_effective_meso_cfg(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = cfg.get("inputs") or {}
    meso_cfg = dict(cfg.get("meso") or {})
    calibration = _derive_power_calibration_from_sweep(
        Path(str(inputs["freeze_meso_fanout_sweep_csv"]))
    )
    meso_cfg.update(calibration)
    sources: list[str] = []
    for source in (
        calibration["calibration_source"],
        inputs.get("freeze_overview_csv"),
        meso_cfg.get("reuse_provenance_csv"),
        inputs.get("reuse_provenance_csv"),
    ):
        source_text = str(source or "").strip()
        if source_text and source_text not in sources:
            sources.append(source_text)
    meso_cfg["calibration_source"] = ";".join(sources)
    return meso_cfg, calibration


def _project_row(
    *,
    row: dict[str, str],
    meso_cfg: dict[str, Any],
    band_label: str,
    control_scale_vs_driver: float,
    buffer_scale_vs_driver: float,
) -> dict[str, Any]:
    current_energy_j = _to_float(row.get("energy_j"), 0.0) or 0.0
    current_latency_ms = _to_float(row.get("latency_ms"), 0.0) or 0.0
    current_tops_w = _to_float(row.get("tops_w"), 0.0) or 0.0
    current_net_gain_j = _to_float(row.get("net_energy_gain_j"), 0.0) or 0.0
    band_cfg = dict(meso_cfg)
    band_cfg["fabric_control_overhead_policy"] = OVERHEAD_POLICY_BROADCAST_DRIVER_FRACTION
    band_cfg["fabric_control_scale_vs_broadcast_driver"] = control_scale_vs_driver
    band_cfg["extra_buffering_overhead_policy"] = OVERHEAD_POLICY_BROADCAST_DRIVER_FRACTION
    band_cfg["extra_buffering_scale_vs_broadcast_driver"] = buffer_scale_vs_driver
    latency_s = current_latency_ms / 1e3
    metrics = compute_meso_cost_model(
        meso_cfg=band_cfg,
        meso_enabled=True,
        latency_s=latency_s,
    )
    projected_energy_j = current_energy_j - (
        float(metrics["net_energy_gain_j"]) - current_net_gain_j
    )
    projected_tops_w = (
        current_tops_w * current_energy_j / projected_energy_j
        if projected_energy_j > 0.0
        else math.nan
    )
    energy_improvement_pct = (
        (current_energy_j - projected_energy_j) / current_energy_j * 100.0
        if current_energy_j > 0.0
        else 0.0
    )
    tops_w_improvement_pct = (
        (projected_tops_w - current_tops_w) / current_tops_w * 100.0
        if current_tops_w > 0.0 and math.isfinite(projected_tops_w)
        else 0.0
    )
    return {
        "experiment_id": str(row["experiment_id"]),
        "band_label": band_label,
        "fanout": int(metrics["fanout"]),
        "topology_dimension": float(metrics["topology_dimension"]),
        "serializer_power_mw": float(_to_float(band_cfg.get("serializer_power_mw"), 0.0) or 0.0),
        "broadcast_driver_power_mw": float(
            _to_float(band_cfg.get("broadcast_driver_power_mw"), 0.0) or 0.0
        ),
        "control_scale_vs_driver": control_scale_vs_driver,
        "buffer_scale_vs_driver": buffer_scale_vs_driver,
        "current_energy_j": current_energy_j,
        "projected_energy_j": projected_energy_j,
        "current_latency_ms": current_latency_ms,
        "projected_latency_ms": current_latency_ms,
        "current_tops_w": current_tops_w,
        "projected_tops_w": projected_tops_w,
        "current_net_gain_j": current_net_gain_j,
        "projected_net_gain_j": float(metrics["net_energy_gain_j"]),
        "delta_net_gain_j": float(metrics["net_energy_gain_j"]) - current_net_gain_j,
        "broadcast_driver_energy_j": float(metrics["broadcast_driver_energy_j"]),
        "fabric_control_overhead_j": float(metrics["fabric_control_overhead_j"]),
        "extra_buffering_overhead_j": float(metrics["extra_buffering_overhead_j"]),
        "explicit_total_cost_j": float(metrics["explicit_total_cost_j"]),
        "explicit_total_savings_j": float(metrics["explicit_total_savings_j"]),
        "energy_improvement_pct": energy_improvement_pct,
        "tops_w_improvement_pct": tops_w_improvement_pct,
        "net_positive": bool(metrics["break_even"]),
    }


def _count_context_mismatches(value: Any) -> int:
    return len([item for item in str(value or "").split(";") if item.strip()])


def _row_status_priority(value: Any) -> int:
    status = str(value or "").strip()
    priorities = {
        "ready": 3,
        "context_backfill_required": 2,
        "missing": 1,
    }
    return priorities.get(status, 0)


def _truth_class_priority(value: Any) -> int:
    truth_class = str(value or "").strip()
    if truth_class == "bitstream_model_level_measured":
        return 3
    if truth_class:
        return 2
    return 0


def _row_role_priority(value: Any) -> int:
    row_role = str(value or "").strip().lower()
    if row_role == "target":
        return 2
    if row_role == "baseline":
        return 1
    return 0


def _select_accuracy_contract_row(contract_rows: list[dict[str, str]]) -> dict[str, str]:
    def sort_key(row: dict[str, str]) -> tuple[int, int, int, int, int]:
        truth_class = str(row.get("selected_truth_class") or "").strip()
        row_status = str(row.get("row_status") or "").strip()
        return (
            int(row_status == "ready" and truth_class == "bitstream_model_level_measured"),
            _truth_class_priority(truth_class),
            _row_status_priority(row_status),
            _row_role_priority(row.get("row_role")),
            -_count_context_mismatches(row.get("context_mismatches")),
        )

    return max(contract_rows, key=sort_key)


def _missing_accuracy_status_row(*, source: str, detail: str) -> dict[str, Any]:
    return {
        "source": source,
        "experiment_id": "E1",
        "accuracy_evidence": "",
        "accuracy_note": detail,
        "row_status": "missing",
        "truth_class": "",
        "context_mismatches": "",
        "independent_closure_ready": False,
    }


def _build_accuracy_status(cfg: dict[str, Any], overview_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    inputs = cfg.get("inputs") or {}
    e1_freeze = _row_by_experiment(overview_rows, "E1")
    rows: list[dict[str, Any]] = [
        {
            "source": "freeze_overview",
            "experiment_id": "E1",
            "accuracy_evidence": str(e1_freeze.get("accuracy_evidence") or ""),
            "accuracy_note": str(e1_freeze.get("accuracy_note") or ""),
            "row_status": "active_freeze",
            "truth_class": "",
            "context_mismatches": "",
            "independent_closure_ready": (
                str(e1_freeze.get("accuracy_evidence") or "") != "shared_e0_full_eval_reference"
            ),
        }
    ]
    for label, path_key in (
        ("contract_preflight", "accuracy_contract_csv"),
        ("analysis_grade_contract", "analysis_grade_accuracy_contract_csv"),
    ):
        path_value = str(inputs.get(path_key) or "").strip()
        if not path_value:
            rows.append(
                _missing_accuracy_status_row(
                    source=label,
                    detail=f"{path_key}_not_provided",
                )
            )
            continue
        path = Path(path_value)
        if not path.exists():
            rows.append(
                _missing_accuracy_status_row(
                    source=label,
                    detail=f"{path_key}_missing:{path_value}",
                )
            )
            continue
        contract_rows = [
            contract_row
            for contract_row in _read_csv(path)
            if str(contract_row.get("experiment_id") or "").strip() == "E1"
        ]
        if not contract_rows:
            rows.append(
                _missing_accuracy_status_row(
                    source=label,
                    detail=f"{path_key}_missing_e1_rows:{path_value}",
                )
            )
            continue
        contract_row = _select_accuracy_contract_row(contract_rows)
        rows.append(
            {
                "source": label,
                "experiment_id": "E1",
                "accuracy_evidence": "",
                "accuracy_note": str(contract_row.get("selected_source_run_id") or ""),
                "row_status": str(contract_row.get("row_status") or ""),
                "truth_class": str(contract_row.get("selected_truth_class") or ""),
                "context_mismatches": str(contract_row.get("context_mismatches") or ""),
                "independent_closure_ready": (
                    str(contract_row.get("row_status") or "").strip() == "ready"
                    and str(contract_row.get("selected_truth_class") or "").strip()
                    == "bitstream_model_level_measured"
                ),
            }
        )
    return rows


def _build_decision(
    *,
    cfg: dict[str, Any],
    provenance_summary: dict[str, Any],
    projections: list[dict[str, Any]],
    accuracy_status: list[dict[str, Any]],
    det_correlation_intake: dict[str, Any],
) -> dict[str, Any]:
    gate_cfg = cfg.get("promotion_gate") or {}
    fixed_fanout_threshold = int(gate_cfg.get("fixed_fanout_threshold") or 4)
    min_material_energy_gain_pct = float(gate_cfg.get("min_material_energy_gain_pct") or 0.5)
    min_material_tops_w_gain_pct = float(gate_cfg.get("min_material_tops_w_gain_pct") or 0.5)
    require_w1 = bool(gate_cfg.get("require_w1_for_mainline"))
    require_det_correlation = bool(gate_cfg.get("require_det_correlation_closure_for_mainline"))
    w1_present = bool(gate_cfg.get("w1_evidence_present"))

    nonzero_rows = [
        row for row in projections
        if row["control_scale_vs_driver"] > 0.0 or row["buffer_scale_vs_driver"] > 0.0
    ]
    e6_nonzero = [row for row in nonzero_rows if row["experiment_id"] == "E6"]
    defended_e6 = next(
        (
            row
            for row in projections
            if row["experiment_id"] == "E6" and str(row["band_label"]) == "defended"
        ),
        None,
    )
    max_e6_energy_gain = max((row["energy_improvement_pct"] for row in e6_nonzero), default=0.0)
    max_e6_tops_w_gain = max((row["tops_w_improvement_pct"] for row in e6_nonzero), default=0.0)
    defended_e6_energy_gain = (
        float(defended_e6["energy_improvement_pct"]) if defended_e6 is not None else 0.0
    )
    defended_e6_tops_w_gain = (
        float(defended_e6["tops_w_improvement_pct"]) if defended_e6 is not None else 0.0
    )
    nonzero_robust = bool(nonzero_rows) and all(bool(row["net_positive"]) for row in nonzero_rows)
    accuracy_independent = any(bool(row["independent_closure_ready"]) for row in accuracy_status)
    gate_status = {
        "fanout_breaks_fixed_fanout4": int(provenance_summary["resolved_fanout"]) > fixed_fanout_threshold,
        "nonzero_control_buffer_robust": nonzero_robust,
        # Promotion materiality is defined against the defended E6 band in the
        # mainline re-entry note; the broader max remains as a diagnostic only.
        "hops_gain_material": (
            defended_e6_energy_gain >= min_material_energy_gain_pct
            and defended_e6_tops_w_gain >= min_material_tops_w_gain_pct
        ),
        "independent_accuracy_closure": accuracy_independent,
        "det_correlation_closure": (
            (not require_det_correlation)
            or bool(det_correlation_intake["det_correlation_closure_ready"])
        ),
        "w1_requirement_satisfied": (not require_w1) or w1_present,
    }
    blocker_reasons: list[str] = []
    if not gate_status["fanout_breaks_fixed_fanout4"]:
        blocker_reasons.append("fanout_still_effectively_fixed_at_or_below_4")
    if not gate_status["nonzero_control_buffer_robust"]:
        blocker_reasons.append("meso_gain_not_robust_under_nonzero_control_buffer_band")
    if not gate_status["hops_gain_material"]:
        blocker_reasons.append("best_nonzero_e6_gain_below_materiality_threshold")
    if not gate_status["independent_accuracy_closure"]:
        blocker_reasons.append("no_independent_meso_specific_accuracy_closure")
    if not gate_status["det_correlation_closure"]:
        blocker_reasons.append(
            str(det_correlation_intake["det_correlation_blocker_reason"] or "")
            or "meso_det_correlation_not_closed"
        )
    if not gate_status["w1_requirement_satisfied"]:
        blocker_reasons.append("w1_missing_for_mainline_generalization")
    return {
        "date": str(date.today()),
        "resolved_fanout": int(provenance_summary["resolved_fanout"]),
        "resolved_topology_dimension": float(
            resolve_meso_topology_dimension(
                meso_cfg=cfg.get("meso") or {},
                fanout=int(provenance_summary["resolved_fanout"]),
            )
        ),
        "defended_e6_energy_gain_pct": defended_e6_energy_gain,
        "defended_e6_tops_w_gain_pct": defended_e6_tops_w_gain,
        "max_e6_energy_gain_pct_nonzero_band": max_e6_energy_gain,
        "max_e6_tops_w_gain_pct_nonzero_band": max_e6_tops_w_gain,
        "det_correlation_source_note": str(det_correlation_intake["source_note"]),
        "det_correlation_dedicated_lane_present": bool(det_correlation_intake["dedicated_lane_present"]),
        "det_correlation_closure_ready": bool(det_correlation_intake["det_correlation_closure_ready"]),
        "det_correlation_intake_status": str(det_correlation_intake["det_correlation_intake_status"]),
        "det_correlation_blocker_reason": str(
            det_correlation_intake["det_correlation_blocker_reason"]
        ),
        "det_correlation_detail": str(det_correlation_intake["det_correlation_detail"]),
        "det_correlation_runner_support_present": bool(
            det_correlation_intake["runner_support_present"]
        ),
        "det_correlation_blocker_reasons": list(det_correlation_intake["blocker_reasons"]),
        "gate_status": gate_status,
        "mainline_reentry": all(gate_status.values()),
        "bounded_lane_continues": gate_status["fanout_breaks_fixed_fanout4"] and gate_status["nonzero_control_buffer_robust"],
        "blocker_reasons": blocker_reasons,
        "stop_rule": (
            "Keep MESO off the HOPS mainline if any gate remains false; in practice stop promotion "
            "when the best defended nonzero E6 gain stays below "
            f"{min_material_energy_gain_pct:.2f}% energy and "
            f"{min_material_tops_w_gain_pct:.2f}% TOPS/W, or when E1 still lacks independent measured closure."
        ),
    }


def _format_accuracy_status_summary_line(row: dict[str, Any]) -> str:
    detail = str(row.get("accuracy_evidence") or row.get("accuracy_note") or "").strip()
    context_mismatches = str(row.get("context_mismatches") or "").strip()
    if not detail:
        detail = context_mismatches

    line = (
        f"- `{row['source']}`: row_status=`{row['row_status']}`, truth_class=`{row['truth_class']}`, "
        f"independent_closure_ready=`{str(bool(row['independent_closure_ready'])).lower()}`, "
        f"detail=`{detail}`"
    )
    if context_mismatches and context_mismatches != detail:
        line += f", context_mismatches=`{context_mismatches}`"
    return line


def _render_summary_md(
    *,
    cfg_path: Path,
    effective_cfg_path: Path,
    closure_config_path: Path,
    correlation_config_path: Path,
    provenance_summary: dict[str, Any],
    calibration: dict[str, Any],
    projections: list[dict[str, Any]],
    accuracy_status: list[dict[str, Any]],
    det_correlation_intake: dict[str, Any],
    closure_package: dict[str, Any],
    closure_config_package: dict[str, Any],
    correlation_package: dict[str, Any],
    decision: dict[str, Any],
    repo_root: Path = ROOT,
) -> str:
    lines = [
        "# MESO Bounded Re-entry Package",
        "",
        f"Date: `{decision['date']}`",
        f"Config: `{_display_report_path(cfg_path, repo_root=repo_root)}`",
        f"Effective config: `{_display_report_path(effective_cfg_path, repo_root=repo_root)}`",
        "",
        "## Provenance",
        "",
        f"- sample_count: `{provenance_summary['sample_count']}`",
        f"- filter-side p50 fanout: `{provenance_summary['sample_p50']}`",
        f"- clipped defended fanout: `{provenance_summary['resolved_fanout']}`",
        f"- derived topology_dimension: `{decision['resolved_topology_dimension']}`",
        "",
        "## Calibrated MESO Surface",
        "",
        f"- serializer_power_mw: `{float(calibration['serializer_power_mw']):.6f}`",
        f"- broadcast_driver_power_mw: `{float(calibration['broadcast_driver_power_mw']):.6f}`",
        f"- calibration_source: `{_display_report_path(calibration['calibration_source'], repo_root=repo_root)}`",
        "",
        "## Projection Bands",
        "",
    ]
    for experiment_id in ("E1", "E6"):
        lines.append(f"### {experiment_id}")
        lines.append("")
        for row in projections:
            if row["experiment_id"] != experiment_id:
                continue
            lines.append(
                f"- `{row['band_label']}`: energy `{row['projected_energy_j']:.9f} J`, "
                f"TOPS/W `{row['projected_tops_w']:.9f}`, energy gain `{row['energy_improvement_pct']:.3f}%`, "
                f"TOPS/W gain `{row['tops_w_improvement_pct']:.3f}%`, net_positive=`{str(bool(row['net_positive'])).lower()}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Accuracy Status",
            "",
        ]
    )
    for row in accuracy_status:
        lines.append(_format_accuracy_status_summary_line(row))
    lines.extend(
        [
            "",
            "## Phase3 Execution Package",
            "",
            "### Measured Closure Package",
            "",
            f"- source_manifest: `{_display_report_path(closure_package['source_manifest'], repo_root=repo_root)}`",
            f"- status: `{closure_package['status']}`",
            f"- analysis_grade_ready: `{str(bool(closure_package['analysis_grade_ready'])).lower()}`",
            f"- launch_prefix: `{' '.join(closure_package['launch_prefix'])}`",
            f"- job_count: `{len(closure_package['jobs'])}`",
            f"- closure_config: `{_display_report_path(closure_config_path, repo_root=repo_root)}`",
            f"- closure_base_config: `{_display_report_path(closure_config_package['base_config'], repo_root=repo_root)}`",
            f"- closure_run_id: `{closure_config_package['run_id']}`",
            f"- closure_context_run_id_seed0: `{closure_config_package['context_run_id_seed0']}`",
        ]
    )
    if closure_package["analysis_grade_blockers"]:
        lines.append(
            f"- analysis_grade_blockers: `{'; '.join(closure_package['analysis_grade_blockers'])}`"
        )
    for job in closure_package["jobs"]:
        seed_label = "unknown" if job["seed"] is None else str(job["seed"])
        lines.append(
            f"- closure_job seed `{seed_label}`: eval_run_id=`{job['eval_run_id']}`, "
            f"prepared_phase1_config=`{_display_report_path(job['prepared_phase1_config'], repo_root=repo_root)}`"
        )
    lines.extend(
        [
            "",
            "### Minimal MESO+DET Correlation Lane",
            "",
            f"- correlation_config: `{_display_report_path(correlation_config_path, repo_root=repo_root)}`",
            f"- base_config: `{_display_report_path(correlation_package['base_config'], repo_root=repo_root)}`",
            f"- status: `{correlation_package['status']}`",
            f"- run_id: `{correlation_package['run_id']}`",
            f"- experiment_id: `{correlation_package['experiment_id']}`",
            f"- context_run_id_seed0: `{correlation_package['context_run_id_seed0']}`",
            f"- seeds: `{', '.join(str(seed) for seed in correlation_package['seeds'])}`",
            f"- det_k_global: `{correlation_package['det_k_global']}`",
            f"- phase1_launch_prefix: `{' '.join(correlation_package['phase1_launch_prefix'])}`",
            f"- follow_on_accuracy_template_source: `{_display_report_path(correlation_package['follow_on_accuracy_template_source'], repo_root=repo_root)}`",
            "",
            "## DET Correlation Intake",
            "",
            f"- source_note: `{_display_report_path(det_correlation_intake['source_note'], repo_root=repo_root)}`",
            f"- runner_support_present: `{str(bool(det_correlation_intake['runner_support_present'])).lower()}`",
            f"- dedicated_lane_present: `{str(bool(det_correlation_intake['dedicated_lane_present'])).lower()}`",
            f"- det_correlation_closure_ready: `{str(bool(det_correlation_intake['det_correlation_closure_ready'])).lower()}`",
            f"- det_correlation_intake_status: `{det_correlation_intake['det_correlation_intake_status']}`",
            f"- det_correlation_blocker_reason: `{det_correlation_intake['det_correlation_blocker_reason']}`",
            f"- det_correlation_detail: {det_correlation_intake['det_correlation_detail']}",
            f"- blocker_reasons: `{'; '.join(det_correlation_intake['blocker_reasons'])}`",
            "",
            "### Final Judgment Intake",
            "",
        ]
    )
    for line in det_correlation_intake["final_judgment"]:
        lines.append(line)
    lines.extend(
        [
            "",
            "### Minimum Next Work Package Intake",
            "",
        ]
    )
    for line in det_correlation_intake["next_work_package"]:
        lines.append(line)
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- bounded_lane_continues: `{str(bool(decision['bounded_lane_continues'])).lower()}`",
            f"- mainline_reentry: `{str(bool(decision['mainline_reentry'])).lower()}`",
            "",
            "### Gate Status",
            "",
        ]
    )
    for gate_name, gate_value in (decision.get("gate_status") or {}).items():
        lines.append(f"- {gate_name}: `{str(bool(gate_value)).lower()}`")
    lines.extend(
        [
            "",
            f"- defended_e6_gain: energy `{float(decision['defended_e6_energy_gain_pct']):.3f}%`, "
            f"TOPS/W `{float(decision['defended_e6_tops_w_gain_pct']):.3f}%`",
            f"- det_correlation_intake_status: `{decision['det_correlation_intake_status']}`",
            f"- det_correlation_blocker_reason: `{decision['det_correlation_blocker_reason']}`",
            f"- det_correlation_detail: {decision['det_correlation_detail']}",
            f"- blocker_reasons: `{'; '.join(decision['blocker_reasons'])}`",
            f"- stop_rule: {decision['stop_rule']}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, repo_root: Path = ROOT) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "phase1_meso_bounded_package_20260420.yaml",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=repo_root / "experiments" / "results" / "analysis" / "20260420_meso_bounded_package",
    )
    args = parser.parse_args(argv)

    resolved_cfg_path = args.config.resolve()
    cfg = _resolve_cfg_paths(
        _load_yaml(resolved_cfg_path),
        cfg_path=resolved_cfg_path,
        repo_root=repo_root,
    )
    inputs = cfg.get("inputs") or {}
    overview_rows = _read_csv(Path(str(inputs["freeze_overview_csv"])))
    effective_meso_cfg, calibration = _build_effective_meso_cfg(cfg)
    provenance_summary = summarize_meso_reuse_provenance(effective_meso_cfg)
    det_correlation_intake = _build_det_correlation_intake(cfg)
    closure_lane_cfg, closure_config_package = _build_meso_measured_closure_lane(
        cfg=cfg,
        effective_meso_cfg=effective_meso_cfg,
    )
    closure_package = _build_analysis_grade_closure_package(
        cfg,
        expected_seeds=closure_config_package["seeds"],
    )
    correlation_lane_cfg, correlation_package = _build_meso_det_correlation_lane(
        cfg=cfg,
        effective_meso_cfg=effective_meso_cfg,
    )
    det_correlation_intake = _reconcile_det_correlation_intake(
        det_correlation_intake,
        correlation_package=correlation_package,
        closure_package=closure_package,
    )

    band_rows: list[dict[str, Any]] = []
    bands = cfg.get("sensitivity_bands") or {}
    if not isinstance(bands, dict) or not bands:
        raise SystemExit("Expected non-empty sensitivity_bands mapping")
    for experiment_id in ("E1", "E6"):
        row = _row_by_experiment(overview_rows, experiment_id)
        for band_label, band_cfg in bands.items():
            band_rows.append(
                _project_row(
                    row=row,
                    meso_cfg=effective_meso_cfg,
                    band_label=str(band_label),
                    control_scale_vs_driver=float(
                        band_cfg.get("control_scale_vs_broadcast_driver") or 0.0
                    ),
                    buffer_scale_vs_driver=float(
                        band_cfg.get("buffer_scale_vs_broadcast_driver") or 0.0
                    ),
                )
            )

    accuracy_status = _build_accuracy_status(cfg, overview_rows)
    decision = _build_decision(
        cfg=cfg,
        provenance_summary=provenance_summary,
        projections=band_rows,
        accuracy_status=accuracy_status,
        det_correlation_intake=det_correlation_intake,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    effective_cfg_path = args.out_dir / "meso_effective_config.yaml"
    closure_config_path = args.out_dir / closure_config_package["output_config_name"]
    correlation_config_path = args.out_dir / correlation_package["output_config_name"]
    _write_yaml(
        effective_cfg_path,
        {
            "source_config": str(args.config),
            "inputs": inputs,
            "meso": effective_meso_cfg,
            "sensitivity_bands": bands,
            "promotion_gate": cfg.get("promotion_gate") or {},
        },
    )
    _write_yaml(closure_config_path, closure_lane_cfg)
    _write_yaml(correlation_config_path, correlation_lane_cfg)
    _write_csv(
        args.out_dir / "meso_projection_summary.csv",
        band_rows,
        fieldnames=list(band_rows[0].keys()),
    )
    _write_csv(
        args.out_dir / "meso_accuracy_status.csv",
        accuracy_status,
        fieldnames=list(accuracy_status[0].keys()),
    )
    _write_json(
        args.out_dir / "meso_promotion_gate.json",
        {
            "provenance_summary": provenance_summary,
            "calibration": calibration,
            "det_correlation_intake": det_correlation_intake,
            "closure_package": closure_package,
            "closure_config_package": closure_config_package,
            "correlation_package": correlation_package,
            "decision": decision,
        },
    )
    _write_json(
        args.out_dir / "meso_execution_package.json",
        {
            "closure_package": {
                **closure_package,
                "config_path": str(closure_config_path),
                "config_package": closure_config_package,
            },
            "correlation_package": {
                **correlation_package,
                "config_path": str(correlation_config_path),
            },
        },
    )
    _write_text(
        args.out_dir / "meso_reentry_summary.md",
        _render_summary_md(
            cfg_path=args.config,
            effective_cfg_path=effective_cfg_path,
            closure_config_path=closure_config_path,
            correlation_config_path=correlation_config_path,
            provenance_summary=provenance_summary,
            calibration=calibration,
            projections=band_rows,
            accuracy_status=accuracy_status,
            det_correlation_intake=det_correlation_intake,
            closure_package=closure_package,
            closure_config_package=closure_config_package,
            correlation_package=correlation_package,
            decision=decision,
            repo_root=repo_root,
        ),
    )
    print(f"[meso-bounded-package] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
