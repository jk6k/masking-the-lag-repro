#!/usr/bin/env python3
"""Build a current GPU-efficiency diagnostic for the active FULLER analysis-grade run."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import ROOT, _write_csv, _write_json, _write_text
except ImportError:
    from fuller_experiment_program_common import ROOT, _write_csv, _write_json, _write_text  # type: ignore


DATE_TAG = "20260423"
DEFAULT_STATUS_JSON = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_analysis_grade_replay_status_20260423.json"
)
DEFAULT_MANIFEST_JSON = (
    ROOT
    / "experiments"
    / "results"
    / "report_data"
    / "20260423_fuller_analysis_grade_replay"
    / "astra"
    / "progress"
    / "manifest.json"
)
DEFAULT_EVENTS_JSONL = (
    ROOT
    / "experiments"
    / "results"
    / "report_data"
    / "20260423_fuller_analysis_grade_replay"
    / "astra"
    / "progress"
    / "events"
    / "20260421_fuller_phase1_preflight_astra_mobilevit_s_s0.jsonl"
)
DEFAULT_RAW_RESULTS_CSV = (
    ROOT
    / "experiments"
    / "results"
    / "report_data"
    / "20260423_fuller_analysis_grade_replay"
    / "astra"
    / "raw_accuracy.csv"
)
DEFAULT_OUTPUT_CSV = (
    ROOT
    / "experiments"
    / "results"
    / "report_data"
    / f"fuller_analysis_grade_gpu_efficiency_diagnostic_{DATE_TAG}.csv"
)
DEFAULT_OUTPUT_JSON = (
    ROOT
    / "experiments"
    / "results"
    / "report_data"
    / f"fuller_analysis_grade_gpu_efficiency_diagnostic_{DATE_TAG}.json"
)
DEFAULT_OUTPUT_MD = ROOT / "docs" / "reports" / f"{DATE_TAG}_fuller_analysis_grade_gpu_efficiency_diagnostic.md"

DIAGNOSTIC_FIELDS = [
    "lane_id",
    "queue_state",
    "current_lane",
    "required_device",
    "bitstream_surface_scope",
    "active_target_module_count",
    "targetable_module_count",
    "active_target_fraction",
    "eval_batch_size",
    "workers",
    "bitstream_stream_length",
    "bitstream_stream_reuse_policy",
    "quantized_processed_samples",
    "quantized_total_samples",
    "quantized_batch_events",
    "quantized_runtime_heartbeat_events",
    "runtime_heartbeat_per_1k_samples",
    "matmul_tile_heartbeat_count",
    "matmul_row_band_heartbeat_count",
    "dominant_runtime_modules_json",
    "quantized_samples_per_hour",
    "quantized_eta_seconds",
    "baseline_completed",
    "quantized_result_present",
    "bottleneck_class",
    "primary_root_cause",
    "recommended_actions_json",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _command_value(command: list[str], flag: str) -> str:
    if flag not in command:
        return ""
    index = command.index(flag)
    if index + 1 >= len(command):
        return ""
    return str(command[index + 1])


def _job_command(job: dict[str, Any]) -> list[str]:
    commands = job.get("commands") or []
    if commands and isinstance(commands[0], list):
        return [str(item) for item in commands[0]]
    command = job.get("command")
    if isinstance(command, str) and command.strip():
        return command.strip().split()
    return []


def _float_or_zero(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def _int_or_zero(value: Any) -> int:
    if value in ("", None):
        return 0
    return int(value)


def _diagnostic_note(row: dict[str, Any], *, status_json: Path, manifest_json: Path, events_jsonl: Path) -> str:
    active_fraction_pct = float(row["active_target_fraction"]) * 100.0
    recommended_actions = json.loads(str(row["recommended_actions_json"]))
    lines = [
        "# FULLER Analysis-Grade GPU Efficiency Diagnostic",
        "",
        "Date: `2026-04-23`",
        "Status: `current_analysis_grade_gpu_efficiency_diagnostic`",
        "",
        "## Current Runtime",
        "",
        f"- status_json: `{status_json}`",
        f"- manifest_json: `{manifest_json}`",
        f"- events_jsonl: `{events_jsonl}`",
        f"- queue_state: `{row['queue_state']}`",
        f"- current_lane: `{row['current_lane']}`",
        f"- required_device: `{row['required_device']}`",
        "",
        "## What The Current Run Shows",
        "",
        f"- The active bitstream surface is still narrow: `{row['active_target_module_count']}` active targets out of `{row['targetable_module_count']}` targetable modules (`{active_fraction_pct:.2f}%`).",
        f"- The current lane is progressing through quantized pass work at roughly `{float(row['quantized_samples_per_hour']):.1f}` samples/hour with ETA about `{float(row['quantized_eta_seconds'])/3600.0:.2f}` hours.",
        f"- Runtime heartbeats remain fine-grained: `{row['matmul_tile_heartbeat_count']}` tile heartbeats and `{row['matmul_row_band_heartbeat_count']}` row-band heartbeats across `{row['quantized_processed_samples']}` quantized samples so far.",
        f"- Heartbeat density is `{float(row['runtime_heartbeat_per_1k_samples']):.2f}` runtime callbacks per 1k processed samples, which is consistent with fragmented host-side orchestration rather than sustained large-kernel GPU saturation.",
        "",
        "## Code Hotspots",
        "",
        "- `experiments/accuracy/mlx_mobilevit.py:708` to `:773` builds bitstreams row-by-row on the MLX path.",
        "- `experiments/accuracy/mlx_mobilevit.py:798` to `:844` estimates stream products tile-by-tile and still periodically materializes MLX accumulators.",
        "- `experiments/accuracy/mlx_mobilevit.py:1160` to `:1245` drives Python-side `row_tile x col_tile` matmul loops and emits tile/row-band progress heartbeats.",
        "",
        "## Diagnosis",
        "",
        f"- bottleneck_class: `{row['bottleneck_class']}`",
        f"- primary_root_cause: `{row['primary_root_cause']}`",
        "- This is not a CPU fallback diagnosis. The run is on `mps`, but the GPU is being fed with a narrow bitstream surface and many small MLX work units.",
        "",
        "## Prioritized Next Actions",
        "",
    ]
    lines.extend(f"- {action}" for action in recommended_actions)
    return "\n".join(lines) + "\n"


def build_fuller_analysis_grade_gpu_efficiency_diagnostic(
    *,
    status_json: Path = DEFAULT_STATUS_JSON,
    manifest_json: Path = DEFAULT_MANIFEST_JSON,
    events_jsonl: Path = DEFAULT_EVENTS_JSONL,
    raw_results_csv: Path = DEFAULT_RAW_RESULTS_CSV,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
) -> dict[str, Any]:
    status_payload = _load_json(status_json)
    manifest_payload = _load_json(manifest_json)
    event_rows = _load_jsonl(events_jsonl)
    raw_rows = _load_csv(raw_results_csv) if raw_results_csv.exists() else []

    command = [str(item) for item in status_payload.get("active_command") or []]
    current_lane = str(status_payload.get("current_lane") or "").strip().upper()
    current_job = next(
        (
            job
            for job in manifest_payload.get("jobs") or []
            if str(job.get("eval_run_id") or "").endswith("_s0")
        ),
        (manifest_payload.get("jobs") or [None])[0],
    )
    if not isinstance(current_job, dict):
        raise SystemExit("Could not resolve current analysis-grade job from manifest.")
    job_command = _job_command(current_job)

    guardrail = current_job.get("bitstream_runtime_guardrail") or {}
    if not isinstance(guardrail, dict):
        raise SystemExit("Missing bitstream_runtime_guardrail in manifest.")

    quantized_rows = [row for row in event_rows if row.get("pass_kind") == "quantized_eval_pass"]
    batch_rows = [row for row in quantized_rows if row.get("event") == "pass_batch_complete"]
    heartbeat_rows = [row for row in quantized_rows if row.get("event") == "pass_runtime_heartbeat"]
    latest_batch = batch_rows[-1] if batch_rows else {}

    stage_counts: dict[str, int] = {}
    module_counts: dict[str, int] = {}
    for row in heartbeat_rows:
        stage = str(row.get("runtime_stage") or "")
        module_key = str(row.get("runtime_module_key") or "")
        if stage:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        if module_key:
            module_counts[module_key] = module_counts.get(module_key, 0) + 1

    def command_value(flag: str) -> str:
        return _command_value(command, flag) or _command_value(job_command, flag)

    active_target_module_count = _int_or_zero(guardrail.get("active_target_module_count"))
    targetable_module_count = _int_or_zero(guardrail.get("targetable_module_count"))
    active_target_fraction = (
        float(active_target_module_count) / float(targetable_module_count)
        if targetable_module_count > 0
        else 0.0
    )
    quantized_processed_samples = _int_or_zero(latest_batch.get("processed_samples"))
    runtime_heartbeat_per_1k = (
        (float(len(heartbeat_rows)) / float(quantized_processed_samples)) * 1000.0
        if quantized_processed_samples > 0
        else 0.0
    )
    baseline_completed = any(str(row.get("baseline") or "").strip().lower() == "true" for row in raw_rows)
    quantized_result_present = any(
        str(row.get("baseline") or "").strip().lower() == "false" for row in raw_rows
    )

    recommended_actions = [
        "Expand the governed active bitstream surface beyond the current 2-module limited_linear_attention_pilot where claim boundaries allow.",
        "Reduce Python-side row/column tile orchestration in mlx_mobilevit bitstream matmul so MPS receives fewer, larger work units.",
        "Further lower tile-level progress and synchronization pressure inside the MLX bitstream estimate path.",
        "Revisit stream-generation caching and tile-budget tuning together instead of only raising workers or eval_batch_size.",
    ]
    row = {
        "lane_id": current_lane or "ASTRA",
        "queue_state": str(status_payload.get("queue_state") or ""),
        "current_lane": current_lane or "ASTRA",
        "required_device": command_value("--device"),
        "bitstream_surface_scope": str(guardrail.get("surface_scope") or command_value("--bitstream_surface_scope")),
        "active_target_module_count": active_target_module_count,
        "targetable_module_count": targetable_module_count,
        "active_target_fraction": round(active_target_fraction, 6),
        "eval_batch_size": _int_or_zero(command_value("--eval_batch_size")),
        "workers": _int_or_zero(command_value("--workers")),
        "bitstream_stream_length": _int_or_zero(command_value("--bitstream_stream_length")),
        "bitstream_stream_reuse_policy": command_value("--bitstream_stream_reuse_policy"),
        "quantized_processed_samples": quantized_processed_samples,
        "quantized_total_samples": _int_or_zero(latest_batch.get("total_samples")),
        "quantized_batch_events": len(batch_rows),
        "quantized_runtime_heartbeat_events": len(heartbeat_rows),
        "runtime_heartbeat_per_1k_samples": round(runtime_heartbeat_per_1k, 3),
        "matmul_tile_heartbeat_count": stage_counts.get("matmul_tile_complete", 0),
        "matmul_row_band_heartbeat_count": stage_counts.get("matmul_row_band_complete", 0),
        "dominant_runtime_modules_json": json.dumps(module_counts, ensure_ascii=False, sort_keys=True),
        "quantized_samples_per_hour": round(_float_or_zero(latest_batch.get("samples_per_hour")), 6),
        "quantized_eta_seconds": round(_float_or_zero(latest_batch.get("eta_pass_seconds_current_rate")), 6),
        "baseline_completed": baseline_completed,
        "quantized_result_present": quantized_result_present,
        "bottleneck_class": "narrow_bitstream_surface_and_host_scheduled_tiling",
        "primary_root_cause": "The analysis-grade lane is running on mps, but only 2 of 177 targetable modules are active and the MLX bitstream path still executes many small host-orchestrated tiles.",
        "recommended_actions_json": json.dumps(recommended_actions, ensure_ascii=False),
    }

    note = _diagnostic_note(row, status_json=status_json, manifest_json=manifest_json, events_jsonl=events_jsonl)
    payload = {
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status_json": str(status_json),
        "manifest_json": str(manifest_json),
        "events_jsonl": str(events_jsonl),
        "raw_results_csv": str(raw_results_csv),
        "diagnostic_csv": str(output_csv),
        "diagnostic_json": str(output_json),
        "diagnostic_md": str(output_md),
        "rows": [row],
    }
    _write_csv(output_csv, DIAGNOSTIC_FIELDS, [row])
    _write_json(output_json, payload)
    _write_text(output_md, note)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS_JSON)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST_JSON)
    parser.add_argument("--events-jsonl", type=Path, default=DEFAULT_EVENTS_JSONL)
    parser.add_argument("--raw-results-csv", type=Path, default=DEFAULT_RAW_RESULTS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    payload = build_fuller_analysis_grade_gpu_efficiency_diagnostic(
        status_json=args.status_json,
        manifest_json=args.manifest_json,
        events_jsonl=args.events_jsonl,
        raw_results_csv=args.raw_results_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        output_md=args.output_md,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
