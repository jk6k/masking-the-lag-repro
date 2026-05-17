#!/usr/bin/env python3
"""Build a governed runtime-smoke report layer for the current True-SC E0-E6 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "configs" / "true_sc_e0_e6_runtime_smoke_current_bundle_20260419.yaml"
DEFAULT_LAUNCH_ROOT = (
    ROOT / "experiments" / "results" / "report_data" / "true_sc_e0_e6_runtime_smoke_recovery_20260419_combined"
)
DEFAULT_OUT_DIR = (
    ROOT / "experiments" / "results" / "report_data" / "true_sc_e0_e6_runtime_smoke_report_20260419"
)


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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _to_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_float(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _derive_lane_summary_path(lane: dict[str, Any]) -> Path | None:
    config_path = _resolve_repo_path(lane.get("config_path"))
    if config_path is None or not config_path.exists():
        return None
    cfg = _load_yaml(config_path)
    run_cfg = cfg.get("run") or {}
    outputs_cfg = cfg.get("outputs") or {}
    run_id = str(run_cfg.get("run_id") or lane.get("run_id") or "").strip()
    out_dir = _resolve_repo_path(outputs_cfg.get("out_dir"))
    if out_dir is None or not run_id:
        return None
    return out_dir / run_id / "phase1_summary.csv"


def _resolve_lane_phase1_summary_path(lane: dict[str, Any]) -> Path:
    experiment_id = str(lane.get("experiment_id") or "")
    phase1_result = lane.get("phase1_preflight") or {}
    status = str(phase1_result.get("status") or "").strip()
    summary_path_text = str(phase1_result.get("summary_path") or "").strip()
    summary_path = Path(summary_path_text) if summary_path_text else None

    if status == "pass":
        if summary_path is not None and summary_path.exists():
            return summary_path
        derived = _derive_lane_summary_path(lane)
        if derived is not None and derived.exists():
            return derived
        missing = summary_path if summary_path is not None else derived
        raise SystemExit(f"Missing Phase-1 summary for lane {experiment_id}: {missing}")

    if status in {"", "not_run"}:
        if summary_path is not None and summary_path.exists():
            return summary_path
        derived = _derive_lane_summary_path(lane)
        if derived is not None and derived.exists():
            return derived
    raise SystemExit(f"Phase-1 preflight is not passing for lane {experiment_id}")


def _normalize_bool_str(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return "true"
    if text in {"false", "0", "no"}:
        return "false"
    return ""


def _analysis_blockers(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, list):
        return ";".join(str(item) for item in payload)
    return text


def _claimability_note(
    *,
    experiment_id: str,
    target_row: dict[str, str],
    eligibility: dict[str, Any],
) -> str:
    if experiment_id == "E0":
        return "current E0 anchor row; quantized path remains the model-level measured baseline reference"
    if bool(eligibility.get("promotable_measured_row_eligible")):
        return "promotable measured row eligible"
    truth_class = str(target_row.get("bitstream_measurement_truth_class") or "").strip()
    scope = str(target_row.get("bitstream_runtime_claim_surface_status") or "").strip()
    parts = ["runtime_smoke only"]
    if truth_class:
        parts.append(f"truth_class={truth_class}")
    if scope:
        parts.append(f"claim_surface={scope}")
    blockers = eligibility.get("blockers") or []
    if blockers:
        parts.append("blockers=" + ";".join(str(item) for item in blockers))
    return "; ".join(parts)


def _eligibility_report_path(launch_root: Path, experiment_id: str, context_run_id: str) -> Path | None:
    lane_root = launch_root / experiment_id.lower()
    candidates = [
        lane_root / "prepared_eligibility_reports" / f"{context_run_id}_measured_row_eligibility.json",
        lane_root / "prepared_eligibility" / f"{context_run_id}_measured_row_eligibility.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_accuracy_rows(
    *,
    candidate_rows: list[dict[str, str]],
    experiment_id: str,
    context_run_id: str,
) -> tuple[dict[str, str], dict[str, str]]:
    baseline = {}
    target = {}
    for row in candidate_rows:
        row_experiment = str(row.get("experiment_id") or "").strip()
        row_run_id = str(row.get("run_id") or "").strip()
        row_source_run_id = str(row.get("source_run_id") or "").strip()
        if row_experiment != experiment_id:
            continue
        if row_run_id != context_run_id and row_source_run_id != context_run_id:
            continue
        baseline_flag = _normalize_bool_str(row.get("baseline"))
        if baseline_flag == "true":
            baseline = row
        elif baseline_flag == "false":
            target = row
    if not baseline or not target:
        raise SystemExit(
            f"Missing runtime-smoke accuracy rows for experiment={experiment_id} context_run_id={context_run_id}"
        )
    return baseline, target


def _overview_row(
    *,
    lane: dict[str, Any],
    summary_row: dict[str, str],
    baseline_row: dict[str, str],
    target_row: dict[str, str],
    eligibility: dict[str, Any],
    e0_latency_ms: float | None,
    summary_path: Path,
    eligibility_path: Path | None,
) -> dict[str, Any]:
    latency_ms = _to_float(summary_row.get("latency_ms"))
    energy_j = _to_float(summary_row.get("energy_j"))
    throughput_images_s = 1000.0 / latency_ms if latency_ms and latency_ms > 0 else None
    speedup_vs_e0 = (e0_latency_ms / latency_ms) if latency_ms and e0_latency_ms else None
    baseline_top1 = _to_float(baseline_row.get("top1"))
    quantized_top1 = _to_float(target_row.get("top1"))
    quantized_drop = (
        baseline_top1 - quantized_top1
        if baseline_top1 is not None and quantized_top1 is not None
        else None
    )
    analysis_grade_blockers = _analysis_blockers(target_row.get("analysis_grade_blockers"))
    return {
        "experiment_id": lane["experiment_id"],
        "lane_label": lane["lane_label"],
        "mechanism_focus": lane["mechanism_focus"],
        "run_id": lane["run_id"],
        "accuracy_context_run_id": lane["accuracy_context_run_id"],
        "latency_ms": _format_float(latency_ms, digits=6),
        "energy_j": _format_float(energy_j, digits=9),
        "tops_w": summary_row.get("tops_w", ""),
        "throughput_images_s": _format_float(throughput_images_s, digits=6),
        "speedup_vs_E0": _format_float(speedup_vs_e0, digits=6),
        "avg_effective_bsl": summary_row.get("avg_effective_bsl", ""),
        "duty_cycle_avg": summary_row.get("duty_cycle_avg", ""),
        "baseline_top1": baseline_row.get("top1", ""),
        "quantized_top1": target_row.get("top1", ""),
        "quantized_top1_drop_pp_vs_baseline": _format_float(quantized_drop, digits=6),
        "quantized_truth_class": target_row.get("bitstream_measurement_truth_class", ""),
        "quantized_evidence_tier": target_row.get("accuracy_evidence_tier", ""),
        "quantized_claim_surface_status": target_row.get("bitstream_runtime_claim_surface_status", ""),
        "bitstream_surface_scope": target_row.get("bitstream_surface_scope", ""),
        "analysis_grade_ready": target_row.get("analysis_grade_ready", ""),
        "analysis_grade_blockers": analysis_grade_blockers,
        "promotable_measured_row_eligible": str(
            bool(eligibility.get("promotable_measured_row_eligible"))
        ).lower(),
        "eligibility_blockers": ";".join(str(item) for item in (eligibility.get("blockers") or [])),
        "phase1_summary_path": str(summary_path),
        "eligibility_report_path": str(eligibility_path) if eligibility_path else "",
        "claimability_note": _claimability_note(
            experiment_id=lane["experiment_id"],
            target_row=target_row,
            eligibility=eligibility,
        ),
    }


def _mechanism_row(
    *,
    lane: dict[str, Any],
    summary_row: dict[str, str],
    target_row: dict[str, str],
    eligibility: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": lane["experiment_id"],
        "lane_label": lane["lane_label"],
        "mechanism_focus": lane["mechanism_focus"],
        "meso_enabled": str(lane.get("meso", "")).lower(),
        "flow_enabled": str(lane.get("flow", "")).lower(),
        "det_enabled": str(lane.get("det", "")).lower(),
        "sparse_enabled": str(lane.get("sparse", "")).lower(),
        "phy_enabled": str(lane.get("phy", "")).lower(),
        "fanout": summary_row.get("fanout", ""),
        "serializers_saved": summary_row.get("serializers_saved", ""),
        "net_energy_gain_j": summary_row.get("net_energy_gain_j", ""),
        "flow_buffer_depth": summary_row.get("flow_buffer_depth", ""),
        "flow_overlap_efficiency": summary_row.get("flow_overlap_efficiency", ""),
        "flow_buffer_peak_cycles": summary_row.get("flow_buffer_peak_cycles", ""),
        "det_k_global": summary_row.get("det_k_global", "") or target_row.get("det_k_global", ""),
        "det_prefix_error_mean": summary_row.get("det_prefix_error_mean", "") or target_row.get("det_prefix_error_mean", ""),
        "det_prefix_error_p95": summary_row.get("det_prefix_error_p95", "") or target_row.get("det_prefix_error_p95", ""),
        "sparse_tau_global": summary_row.get("sparse_tau_global", "") or target_row.get("sparse_tau_global", ""),
        "sparse_active_fraction": summary_row.get("sparse_active_fraction", "") or target_row.get("sparse_active_fraction", ""),
        "sparse_measured_activity_fraction": summary_row.get("sparse_measured_activity_fraction", ""),
        "N_wdm": summary_row.get("N_wdm", ""),
        "PP_crosstalk_db": summary_row.get("PP_crosstalk_db", ""),
        "P_laser_dbm": summary_row.get("P_laser_dbm", ""),
        "quantized_truth_class": target_row.get("bitstream_measurement_truth_class", ""),
        "quantized_claim_surface_status": target_row.get("bitstream_runtime_claim_surface_status", ""),
        "promotable_measured_row_eligible": str(
            bool(eligibility.get("promotable_measured_row_eligible"))
        ).lower(),
        "eligibility_blockers": ";".join(str(item) for item in (eligibility.get("blockers") or [])),
    }


def _write_note(
    *,
    note_path: Path,
    bundle_path: Path,
    launch_root: Path,
    out_dir: Path,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# True SC E0-E6 Runtime-Smoke Reintegration Note",
        "",
        "Date: `2026-04-19`",
        f"Bundle: `{bundle_path}`",
        f"Launch root: `{launch_root}`",
        f"Report dir: `{out_dir}`",
        "",
        "## Status",
        "",
        "- The current `E0-E6` runtime-smoke matrix is fully populated and context-matched.",
        "- All accelerator-backed collection in this report stayed on `mps` under `caffeinate -dimsu`.",
        "- Every lane now has a Phase-1 summary and a bounded runtime-smoke accuracy row pair.",
        "- This report is still not `analysis_grade` and must not be used as a measured full-eval promotion surface.",
        "",
        "## Current Boundary",
        "",
        "- `E0` remains anchored to the current model-level measured baseline row.",
        "- `E1-E6` quantized rows are still `runtime_smoke` bounded artifacts.",
        "- Quantized `E1-E6` rows remain `bridge_only_nonbitstream_measured` / limited-claim-surface evidence with analysis-grade blockers.",
        "",
        "## Report Artifacts",
        "",
        f"- Overview CSV: `{manifest['overview_csv']}`",
        f"- Mechanism status CSV: `{manifest['mechanism_csv']}`",
        f"- Manifest JSON: `{manifest['manifest_json']}`",
        "",
        "## Next Honest Step",
        "",
        "- If we stay on the governed recovery branch, the next promotion is an analysis-grade launch plan or a thesis-facing synthesis note that explicitly says it is runtime-smoke only.",
    ]
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_runtime_smoke_report(
    *,
    bundle_path: Path,
    launch_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    bundle = _load_yaml(bundle_path)
    paths = bundle.get("paths") or {}
    preflight_json = _resolve_repo_path(paths.get("preflight_json"))
    template_yaml = _resolve_repo_path(paths.get("template_yaml"))
    if preflight_json is None or not preflight_json.exists() or template_yaml is None or not template_yaml.exists():
        raise SystemExit(f"Bundle is missing preflight/template outputs: {bundle_path}")

    preflight = _load_json_if_exists(preflight_json)
    lane_rows = list(preflight.get("lane_rows") or [])
    if not lane_rows:
        raise SystemExit(f"No lane_rows found in {preflight_json}")

    template_cfg = _load_yaml(template_yaml)
    accuracy_source_csv = _resolve_repo_path(((template_cfg.get("accuracy") or {}).get("source_csv")))
    if accuracy_source_csv is None or not accuracy_source_csv.exists():
        raise SystemExit(f"Missing merged accuracy source CSV for {bundle_path}")
    candidate_rows = _read_csv_rows(accuracy_source_csv)

    summary_rows: dict[str, dict[str, str]] = {}
    summary_paths: dict[str, Path] = {}
    e0_latency_ms: float | None = None
    overview_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []

    for lane in lane_rows:
        summary_path = _resolve_lane_phase1_summary_path(lane)
        summary_row = _read_csv_rows(summary_path)[0]
        experiment_id = str(lane["experiment_id"])
        summary_rows[experiment_id] = summary_row
        summary_paths[experiment_id] = summary_path
        if experiment_id == "E0":
            e0_latency_ms = _to_float(summary_row.get("latency_ms"))

    for lane in lane_rows:
        experiment_id = str(lane["experiment_id"])
        summary_path = summary_paths[experiment_id]
        summary_row = summary_rows[experiment_id]
        baseline_row, target_row = _find_accuracy_rows(
            candidate_rows=candidate_rows,
            experiment_id=experiment_id,
            context_run_id=str(lane["accuracy_context_run_id"]),
        )
        eligibility_path = _eligibility_report_path(
            launch_root=launch_root,
            experiment_id=experiment_id,
            context_run_id=str(lane["accuracy_context_run_id"]),
        )
        eligibility = _load_json_if_exists(eligibility_path) if eligibility_path else {}
        overview_rows.append(
            _overview_row(
                lane=lane,
                summary_row=summary_row,
                baseline_row=baseline_row,
                target_row=target_row,
                eligibility=eligibility,
                e0_latency_ms=e0_latency_ms,
                summary_path=summary_path,
                eligibility_path=eligibility_path,
            )
        )
        mechanism_rows.append(
            _mechanism_row(
                lane=lane,
                summary_row=summary_row,
                target_row=target_row,
                eligibility=eligibility,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    overview_csv = out_dir / "quickpack_e0_e6_overview_runtime_smoke.csv"
    mechanism_csv = out_dir / "mechanism_runtime_smoke_status.csv"
    manifest_json = out_dir / "manifest.json"
    note_md = out_dir / "runtime_smoke_reintegration_note.md"

    _write_csv(
        overview_csv,
        overview_rows,
        [
            "experiment_id",
            "lane_label",
            "mechanism_focus",
            "run_id",
            "accuracy_context_run_id",
            "latency_ms",
            "energy_j",
            "tops_w",
            "throughput_images_s",
            "speedup_vs_E0",
            "avg_effective_bsl",
            "duty_cycle_avg",
            "baseline_top1",
            "quantized_top1",
            "quantized_top1_drop_pp_vs_baseline",
            "quantized_truth_class",
            "quantized_evidence_tier",
            "quantized_claim_surface_status",
            "bitstream_surface_scope",
            "analysis_grade_ready",
            "analysis_grade_blockers",
            "promotable_measured_row_eligible",
            "eligibility_blockers",
            "phase1_summary_path",
            "eligibility_report_path",
            "claimability_note",
        ],
    )
    _write_csv(
        mechanism_csv,
        mechanism_rows,
        [
            "experiment_id",
            "lane_label",
            "mechanism_focus",
            "meso_enabled",
            "flow_enabled",
            "det_enabled",
            "sparse_enabled",
            "phy_enabled",
            "fanout",
            "serializers_saved",
            "net_energy_gain_j",
            "flow_buffer_depth",
            "flow_overlap_efficiency",
            "flow_buffer_peak_cycles",
            "det_k_global",
            "det_prefix_error_mean",
            "det_prefix_error_p95",
            "sparse_tau_global",
            "sparse_active_fraction",
            "sparse_measured_activity_fraction",
            "N_wdm",
            "PP_crosstalk_db",
            "P_laser_dbm",
            "quantized_truth_class",
            "quantized_claim_surface_status",
            "promotable_measured_row_eligible",
            "eligibility_blockers",
        ],
    )

    manifest = {
        "bundle_path": str(bundle_path),
        "launch_root": str(launch_root),
        "accuracy_source_csv": str(accuracy_source_csv),
        "preflight_json": str(preflight_json),
        "overview_csv": str(overview_csv),
        "mechanism_csv": str(mechanism_csv),
        "note_md": str(note_md),
        "lane_count": len(overview_rows),
        "analysis_grade_ready_lanes": [
            row["experiment_id"] for row in overview_rows if row["analysis_grade_ready"] == "true"
        ],
    }
    _write_json(manifest_json, manifest)
    manifest["manifest_json"] = str(manifest_json)
    _write_note(
        note_path=note_md,
        bundle_path=bundle_path,
        launch_root=launch_root,
        out_dir=out_dir,
        manifest=manifest,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a governed True-SC E0-E6 runtime-smoke report pack.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--launch_root", type=Path, default=DEFAULT_LAUNCH_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    bundle_path = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    launch_root = args.launch_root if args.launch_root.is_absolute() else ROOT / args.launch_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    payload = build_runtime_smoke_report(
        bundle_path=bundle_path,
        launch_root=launch_root,
        out_dir=out_dir,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
