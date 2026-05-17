#!/usr/bin/env python3
"""Prepare and run current Phase4-basis mechanism evidence.

This utility keeps the restored mechanism narrative on the same Phase4 basis
as the 2026-04-25 intake: ASTRA/MESO/HOPS/DET come from the 20260423 replay,
while repaired SPARSE/FULLER come from the 20260425 sparse-fixed replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from exp_common.det_prefix import compute_prefix_error_stats  # noqa: E402

RUN_TAG = "20260426_fuller_phase4_mechanism_basis_rerun"
REPORT_DIR = ROOT / "experiments/results/report_data" / RUN_TAG
QUICK_DIR = ROOT / "experiments/results/quick_reports" / RUN_TAG
RUNS_DIR = ROOT / "experiments/results/runs"
PREPARED_CONFIG_DIR = REPORT_DIR / "prepared_phase1_configs"
PROGRESS_DIR = REPORT_DIR / "progress"
MPS_PYTHON = ROOT / ".venv311-mps/bin/python"

ASTRA_ROOT = ROOT / "experiments/results/report_data/20260423_fuller_analysis_grade_replay/astra"
REPLAY_20260423 = ROOT / "experiments/results/report_data/20260423_fuller_analysis_grade_replay"
SPARSE_FIXED_20260425 = ROOT / "experiments/results/report_data/20260425_sparse_fixed_analysis_grade_replay"

DET_GRID = [4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 129]
SPARSE_TAU_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
SEEDS = [0, 1, 2]
FULL_SWEEP_EXPECTED_SAMPLES = 45_000

MECHANISM_LABELS = {
    "E0": "ASTRA baseline",
    "E1": "MESO broadcast",
    "E2": "HOPS scheduling",
    "E3": "DET early stop",
    "E4": "SPARSE gating",
    "E6": "FULLER integrated",
}


@dataclass(frozen=True)
class Lane:
    lane: str
    experiment_id: str
    source_root: Path
    config_name: str
    mechanism_focus: str

    @property
    def source_config(self) -> Path:
        return self.source_root / "prepared_phase1_configs" / self.config_name

    @property
    def accuracy_csv(self) -> Path:
        return self.source_root / "annotated_accuracy.csv"

    @property
    def raw_accuracy_csv(self) -> Path:
        return self.source_root / "raw_accuracy.csv"

    @property
    def run_id(self) -> str:
        return f"{RUN_TAG}_{self.lane.lower()}_mechanism"

    @property
    def prepared_config(self) -> Path:
        return PREPARED_CONFIG_DIR / f"{self.lane.lower()}_mechanism_phase1.yaml"

    @property
    def phase1_summary(self) -> Path:
        return RUNS_DIR / self.run_id / "phase1_summary.csv"


LANES = [
    Lane(
        "ASTRA",
        "E0",
        REPLAY_20260423 / "astra",
        "20260421_fuller_phase1_preflight_astra_acc_s0_prepared_phase1_config.yaml",
        "astra_quantized_baseline",
    ),
    Lane(
        "MESO",
        "E1",
        REPLAY_20260423 / "meso",
        "20260421_fuller_phase1_preflight_meso_acc_s0_prepared_phase1_config.yaml",
        "meso_broadcast",
    ),
    Lane(
        "HOPS",
        "E2",
        REPLAY_20260423 / "hops",
        "20260421_fuller_phase1_preflight_hops_acc_s0_prepared_phase1_config.yaml",
        "hops_scheduling",
    ),
    Lane(
        "DET",
        "E3",
        REPLAY_20260423 / "det",
        "20260421_fuller_phase1_preflight_det_acc_s0_prepared_phase1_config.yaml",
        "det_early_stop",
    ),
    Lane(
        "SPARSE",
        "E4",
        SPARSE_FIXED_20260425 / "sparse",
        "20260421_fuller_phase1_preflight_sparse_acc_s0_prepared_phase1_config.yaml",
        "sparse_gating",
    ),
    Lane(
        "FULLER",
        "E6",
        SPARSE_FIXED_20260425 / "fuller",
        "20260421_fuller_phase1_preflight_fuller_acc_s0_prepared_phase1_config.yaml",
        "meso_hops_det_sparse_integrated",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return math.nan
    return statistics.fmean(clean)


def stdev(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return 0.0
    return statistics.stdev(clean)


def format_float(value: Any, digits: int = 8) -> str:
    number = to_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}g}"


def selected_accuracy_rows(lane: Lane) -> list[dict[str, str]]:
    rows = read_csv(lane.accuracy_csv)
    selected: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("device", "")).strip().lower() != "mps":
            continue
        if to_bool(row.get("baseline")):
            continue
        if str(row.get("experiment_id", "")).strip() != lane.experiment_id:
            continue
        if not to_bool(row.get("analysis_grade_ready")):
            continue
        selected.append(row)
    if not selected:
        raise SystemExit(f"No Phase4 analysis-grade MPS quantized rows selected for {lane.lane}: {lane.accuracy_csv}")
    return selected


def accuracy_stats(lane: Lane) -> dict[str, Any]:
    rows = selected_accuracy_rows(lane)
    top1 = [to_float(row.get("top1")) for row in rows]
    top5 = [to_float(row.get("top5")) for row in rows]
    latency = [to_float(row.get("latency_ms_per_sample")) for row in rows]
    sparse_active = [to_float(row.get("sparse_active_fraction")) for row in rows]
    det_k = [to_float(row.get("det_k_global")) for row in rows]
    sparse_tau = [to_float(row.get("sparse_tau_global")) for row in rows]
    return {
        "accuracy_rows": len(rows),
        "top1_mean": mean(top1),
        "top1_std": stdev(top1),
        "top1_min": min(v for v in top1 if math.isfinite(v)),
        "top1_max": max(v for v in top1 if math.isfinite(v)),
        "top5_mean": mean(top5),
        "top5_std": stdev(top5),
        "accuracy_latency_ms_per_sample_mean": mean(latency),
        "det_k_global": mean(det_k),
        "sparse_tau_global": mean(sparse_tau),
        "sparse_active_fraction_mean": mean(sparse_active),
        "source_run_ids": ";".join(row.get("run_id", "") for row in rows),
        "accuracy_source_csv": rel(lane.accuracy_csv),
    }


def phase1_command(config_path: Path) -> list[str]:
    return [
        "caffeinate",
        "-dimsu",
        str(MPS_PYTHON),
        "experiments/tools/phase1_runner.py",
        "--config",
        rel(config_path),
    ]


def prepare_configs() -> None:
    PREPARED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    command_rows: list[dict[str, Any]] = []
    manifest_lanes: list[dict[str, Any]] = []
    for lane in LANES:
        if not lane.source_config.is_file():
            raise SystemExit(f"Missing source config for {lane.lane}: {lane.source_config}")
        cfg = yaml.safe_load(lane.source_config.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise SystemExit(f"Invalid YAML config: {lane.source_config}")

        run_cfg = dict(cfg.get("run") or {})
        old_context_run_id = str((cfg.get("accuracy") or {}).get("context_run_id") or "").strip()
        run_cfg.update(
            {
                "run_id": lane.run_id,
                "experiment_id": lane.experiment_id,
                "internal_experiment_id": lane.experiment_id,
                "variant_id": lane.lane,
                "seed": 0,
                "device": "mps",
                "execution_surface": "host_unsandboxed_caffeinate_required",
                "long_run_launch_prefix": ["caffeinate", "-dimsu"],
            }
        )
        notes = str(run_cfg.get("notes") or "").strip()
        run_cfg["notes"] = (
            f"{notes}; " if notes else ""
        ) + f"phase4_mechanism_basis_rerun:{RUN_TAG}; mechanism_focus:{lane.mechanism_focus}"
        cfg["run"] = run_cfg

        outputs_cfg = dict(cfg.get("outputs") or {})
        outputs_cfg.update(
            {
                "out_dir": "experiments/results/runs",
                "append_master": False,
                "save_config_snapshot": True,
                "save_layer_tables": True,
                "save_calibration_log": True,
            }
        )
        cfg["outputs"] = outputs_cfg

        accuracy_cfg = dict(cfg.get("accuracy") or {})
        accuracy_cfg["source_csv"] = rel(lane.accuracy_csv)
        accuracy_cfg["require_context_match"] = False
        accuracy_cfg["context_run_id"] = lane.run_id
        accuracy_cfg["phase4_basis_original_context_run_id"] = old_context_run_id
        accuracy_cfg["phase4_basis_selection_rule"] = (
            "device=mps; baseline=False; experiment_id matches lane; analysis_grade_ready=true"
        )
        contract = dict(accuracy_cfg.get("measurement_contract") or {})
        contract["source"] = rel(lane.accuracy_csv)
        contract["phase4_basis_original_context_run_id"] = old_context_run_id
        contract["note"] = f"{contract.get('note', '')}; current_phase4_mechanism_basis_rerun:{RUN_TAG}".strip("; ")
        accuracy_cfg["measurement_contract"] = contract
        cfg["accuracy"] = accuracy_cfg

        lane.prepared_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command = phase1_command(lane.prepared_config)
        command_rows.append(
            {
                "run_tag": RUN_TAG,
                "lane": lane.lane,
                "experiment_id": lane.experiment_id,
                "phase1_run_id": lane.run_id,
                "prepared_config": rel(lane.prepared_config),
                "source_config": rel(lane.source_config),
                "source_accuracy_csv": rel(lane.accuracy_csv),
                "command": shlex.join(command),
            }
        )
        manifest_lanes.append(
            {
                "lane": lane.lane,
                "experiment_id": lane.experiment_id,
                "mechanism_focus": lane.mechanism_focus,
                "phase1_run_id": lane.run_id,
                "prepared_config": rel(lane.prepared_config),
                "source_config": rel(lane.source_config),
                "accuracy_csv": rel(lane.accuracy_csv),
                "original_context_run_id": old_context_run_id,
            }
        )

    write_csv(
        REPORT_DIR / "mechanism_phase1_commands.csv",
        command_rows,
        [
            "run_tag",
            "lane",
            "experiment_id",
            "phase1_run_id",
            "prepared_config",
            "source_config",
            "source_accuracy_csv",
            "command",
        ],
    )
    write_json(
        REPORT_DIR / "mechanism_basis_manifest.json",
        {
            "run_tag": RUN_TAG,
            "generated_at": utc_now(),
            "phase4_basis": "20260423 analysis-grade replay + 20260425 sparse-fixed SPARSE/FULLER repair",
            "accuracy_selection_rule": "device=mps; baseline=False; experiment_id matches lane; analysis_grade_ready=true",
            "mps_policy": "All local accelerator-backed runs use --device mps and caffeinate -dimsu.",
            "lanes": manifest_lanes,
        },
    )


def run_phase1(force: bool = False) -> None:
    for lane in LANES:
        if lane.phase1_summary.is_file() and not force:
            print(f"[phase1] skip existing {rel(lane.phase1_summary)}")
            continue
        command = phase1_command(lane.prepared_config)
        print(f"[phase1] {lane.lane}: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def read_phase1_row(lane: Lane) -> dict[str, str]:
    if not lane.phase1_summary.is_file():
        raise SystemExit(f"Missing phase1 summary for {lane.lane}: {lane.phase1_summary}")
    rows = read_csv(lane.phase1_summary)
    if not rows:
        raise SystemExit(f"Empty phase1 summary for {lane.lane}: {lane.phase1_summary}")
    return rows[0]


def claim_boundary(lane: Lane) -> str:
    if lane.lane in {"SPARSE", "FULLER"}:
        return "current_phase4_basis_accuracy_preservation_blocked; mechanism/runtime evidence only"
    if lane.lane == "DET":
        return "current_phase4_basis_det_tradeoff_evidence; not a positive preservation claim"
    return "current_phase4_basis_mechanism_runtime_evidence"


def closed_energy_components(phase: dict[str, str]) -> dict[str, float | str]:
    """Return AppF5 components that close to total energy.

    The Phase1 raw breakdown fields are diagnostic estimates and are not always
    mutually exclusive. For stacked figure use, keep the raw total audit columns
    while scaling/folding components so the visible stack does not exceed the
    modeled per-inference total.
    """
    total_mj = to_float(phase.get("energy_j")) * 1000.0
    raw_memory_mj = to_float(phase.get("energy_breakdown_memory_move_j")) * 1000.0
    raw_conversion_mj = to_float(phase.get("energy_breakdown_conversion_control_j")) * 1000.0
    raw_optical_static_mj = (
        to_float(phase.get("energy_breakdown_oe_j"), 0.0)
        + to_float(phase.get("energy_breakdown_adc_pca_j"), 0.0)
        + to_float(phase.get("energy_breakdown_laser_optical_j"), 0.0)
        + to_float(phase.get("energy_breakdown_other_static_j"), 0.0)
    ) * 1000.0
    raw_sum_mj = raw_memory_mj + raw_conversion_mj + raw_optical_static_mj
    if not math.isfinite(total_mj) or total_mj <= 0.0 or raw_sum_mj <= 0.0:
        return {
            "total_energy_mj": total_mj,
            "memory_move_mj": raw_memory_mj,
            "conversion_control_mj": raw_conversion_mj,
            "optical_static_mj": raw_optical_static_mj,
            "raw_component_sum_mj": raw_sum_mj,
            "component_scale_factor": "",
            "component_closure_status": "unclosed_missing_total_or_components",
        }

    if raw_sum_mj > total_mj:
        scale = total_mj / raw_sum_mj
        memory_mj = raw_memory_mj * scale
        conversion_mj = raw_conversion_mj * scale
        optical_static_mj = raw_optical_static_mj * scale
        status = "raw_overlap_scaled_to_total"
    else:
        scale = 1.0
        memory_mj = raw_memory_mj
        conversion_mj = raw_conversion_mj
        optical_static_mj = raw_optical_static_mj + (total_mj - raw_sum_mj)
        status = "residual_folded_into_optical_static"

    return {
        "total_energy_mj": total_mj,
        "memory_move_mj": memory_mj,
        "conversion_control_mj": conversion_mj,
        "optical_static_mj": optical_static_mj,
        "raw_component_sum_mj": raw_sum_mj,
        "component_scale_factor": scale,
        "component_closure_status": status,
    }


def aggregate_quick_reports() -> None:
    phase_rows: dict[str, dict[str, str]] = {lane.lane: read_phase1_row(lane) for lane in LANES}
    acc_rows: dict[str, dict[str, Any]] = {lane.lane: accuracy_stats(lane) for lane in LANES}
    astra_phase = phase_rows["ASTRA"]
    astra_acc = acc_rows["ASTRA"]
    astra_latency = to_float(astra_phase.get("latency_ms"))
    astra_energy = to_float(astra_phase.get("energy_j"))
    astra_top1 = to_float(astra_acc.get("top1_mean"))

    lane_summary_rows: list[dict[str, Any]] = []
    appf4_rows: list[dict[str, Any]] = []
    appf5_rows: list[dict[str, Any]] = []
    for lane in LANES:
        phase = phase_rows[lane.lane]
        acc = acc_rows[lane.lane]
        latency = to_float(phase.get("latency_ms"))
        energy = to_float(phase.get("energy_j"))
        top1 = to_float(acc.get("top1_mean"))
        speedup = astra_latency / latency if latency > 0 and astra_latency > 0 else math.nan
        latency_ratio = latency / astra_latency if astra_latency > 0 else math.nan
        energy_ratio = energy / astra_energy if astra_energy > 0 else math.nan
        acc_drop = astra_top1 - top1 if math.isfinite(astra_top1) and math.isfinite(top1) else math.nan
        base = {
            "run_tag": RUN_TAG,
            "lane": lane.lane,
            "experiment_id": lane.experiment_id,
            "mechanism_label": MECHANISM_LABELS[lane.experiment_id],
            "mechanism_focus": lane.mechanism_focus,
            "phase1_run_id": lane.run_id,
            "phase1_summary_csv": rel(lane.phase1_summary),
            "phase4_accuracy_csv": rel(lane.accuracy_csv),
            "accuracy_rows": acc["accuracy_rows"],
            "top1_mean": acc["top1_mean"],
            "top1_std": acc["top1_std"],
            "top1_min": acc["top1_min"],
            "top1_max": acc["top1_max"],
            "top5_mean": acc["top5_mean"],
            "top5_std": acc["top5_std"],
            "accuracy_latency_ms_per_sample_mean": acc["accuracy_latency_ms_per_sample_mean"],
            "phase1_latency_ms": latency,
            "phase1_energy_j": energy,
            "phase1_avg_power_w": to_float(phase.get("avg_power_w")),
            "speedup_vs_astra": speedup,
            "latency_ratio_vs_astra": latency_ratio,
            "energy_ratio_vs_astra": energy_ratio,
            "energy_saving_vs_astra_pct": (1.0 - energy_ratio) * 100.0 if math.isfinite(energy_ratio) else math.nan,
            "acc_drop_vs_astra_pp": acc_drop,
            "det_k_global": acc["det_k_global"],
            "sparse_tau_global": acc["sparse_tau_global"],
            "sparse_active_fraction_mean": acc["sparse_active_fraction_mean"],
            "accuracy_source_run_ids": acc["source_run_ids"],
            "source_status": "current_phase4_basis_phase1_rerun",
            "claim_boundary": claim_boundary(lane),
            "notes": "Phase1 mechanism/energy rerun paired with current Phase4 analysis-grade MPS quantized accuracy rows.",
        }
        lane_summary_rows.append(base)
        appf4_rows.append(
            {
                "run_tag": RUN_TAG,
                "experiment_id": lane.experiment_id,
                "mechanism_label": MECHANISM_LABELS[lane.experiment_id],
                "speedup_vs_E0": speedup,
                "energy_ratio_vs_E0": energy_ratio,
                "latency_ratio_vs_E0": latency_ratio,
                "energy_j": energy,
                "latency_ms": latency,
                "acc_delta_vs_E0_pp": acc_drop,
                "acc_drop_vs_fp32_pp": "",
                "accuracy_evidence": f"phase4_analysis_grade_mps_quantized_mean_n={acc['accuracy_rows']}; source={rel(lane.accuracy_csv)}",
                "source_status": "current_phase4_basis_phase1_rerun",
                "compatibility_status": "current_basis_replaces_retained_context",
                "claim_boundary": claim_boundary(lane),
                "notes": "Accuracy delta is ASTRA quantized Phase4 mean minus lane Phase4 mean; lower is better.",
            }
        )
        energy_components = closed_energy_components(phase)
        appf5_rows.append(
            {
                "run_tag": RUN_TAG,
                "experiment_id": lane.experiment_id,
                "mechanism_label": MECHANISM_LABELS[lane.experiment_id],
                **energy_components,
                "source_status": "current_phase4_basis_phase1_rerun",
                "compatibility_status": "current_basis_replaces_retained_context",
                "claim_boundary": claim_boundary(lane),
                "notes": (
                    "Visible stack is closed to energy_j; raw_component_sum_mj audits Phase1 diagnostic "
                    "breakdown overlap, and optical_static_mj may include folded residual/system static."
                ),
            }
        )

    lane_fields = [
        "run_tag",
        "lane",
        "experiment_id",
        "mechanism_label",
        "mechanism_focus",
        "phase1_run_id",
        "phase1_summary_csv",
        "phase4_accuracy_csv",
        "accuracy_rows",
        "top1_mean",
        "top1_std",
        "top1_min",
        "top1_max",
        "top5_mean",
        "top5_std",
        "accuracy_latency_ms_per_sample_mean",
        "phase1_latency_ms",
        "phase1_energy_j",
        "phase1_avg_power_w",
        "speedup_vs_astra",
        "latency_ratio_vs_astra",
        "energy_ratio_vs_astra",
        "energy_saving_vs_astra_pct",
        "acc_drop_vs_astra_pp",
        "det_k_global",
        "sparse_tau_global",
        "sparse_active_fraction_mean",
        "accuracy_source_run_ids",
        "source_status",
        "claim_boundary",
        "notes",
    ]
    write_csv(QUICK_DIR / "mechanism_phase4_lane_summary.csv", lane_summary_rows, lane_fields)
    write_csv(
        QUICK_DIR / "appf4_mechanism_ablation_context.csv",
        appf4_rows,
        [
            "run_tag",
            "experiment_id",
            "mechanism_label",
            "speedup_vs_E0",
            "energy_ratio_vs_E0",
            "latency_ratio_vs_E0",
            "energy_j",
            "latency_ms",
            "acc_delta_vs_E0_pp",
            "acc_drop_vs_fp32_pp",
            "accuracy_evidence",
            "source_status",
            "compatibility_status",
            "claim_boundary",
            "notes",
        ],
    )
    write_csv(
        QUICK_DIR / "appf5_mechanism_energy_breakdown.csv",
        appf5_rows,
        [
            "run_tag",
            "experiment_id",
            "mechanism_label",
            "total_energy_mj",
            "memory_move_mj",
            "conversion_control_mj",
            "optical_static_mj",
            "raw_component_sum_mj",
            "component_scale_factor",
            "component_closure_status",
            "source_status",
            "compatibility_status",
            "claim_boundary",
            "notes",
        ],
    )
    write_json(
        QUICK_DIR / "compliance_report.json",
        {
            "run_tag": RUN_TAG,
            "generated_at": utc_now(),
            "build_allowed": True,
            "phase4_basis": "current Phase4 basis, not retained mechanism context",
            "quick_reports": [
                rel(QUICK_DIR / "mechanism_phase4_lane_summary.csv"),
                rel(QUICK_DIR / "appf4_mechanism_ablation_context.csv"),
                rel(QUICK_DIR / "appf5_mechanism_energy_breakdown.csv"),
            ],
        },
    )


def set_arg(command: list[str], flag: str, value: Any) -> list[str]:
    command = list(command)
    text = str(value)
    if flag in command:
        idx = command.index(flag)
        if idx + 1 >= len(command):
            command.append(text)
        else:
            command[idx + 1] = text
    else:
        command.extend([flag, text])
    return command


def remove_arg(command: list[str], flag: str, takes_value: bool = True) -> list[str]:
    out: list[str] = []
    skip = False
    for token in command:
        if skip:
            skip = False
            continue
        if token == flag:
            skip = takes_value
            continue
        out.append(token)
    return out


def ensure_switch(command: list[str], flag: str, enabled: bool) -> list[str]:
    command = [token for token in command if token != flag]
    if enabled:
        command.append(flag)
    return command


def first_command(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") or []
    if not jobs:
        raise SystemExit(f"No jobs in manifest: {manifest_path}")
    commands = jobs[0].get("commands") or []
    if not commands:
        raise SystemExit(f"No command array in first job: {manifest_path}")
    return [str(token) for token in commands[0]]


def prefix_stats_by_k() -> dict[int, dict[str, Any]]:
    rows = compute_prefix_error_stats(
        bsl_max=129,
        k_grid=DET_GRID,
        num_prob_points=129,
        p_min=0.001,
        p_max=0.999,
        enforce_monotonic=False,
    )
    return {int(row["k"]): row for row in rows}


def base_eval_command(lane: str) -> list[str]:
    if lane == "DET":
        path = REPLAY_20260423 / "det/progress/manifest.json"
    elif lane == "SPARSE":
        path = SPARSE_FIXED_20260425 / "sparse/progress/manifest.json"
    else:
        raise ValueError(lane)
    command = first_command(path)
    command[0] = str(MPS_PYTHON)
    return command


def base_policy_profile(lane: str) -> Path:
    if lane == "DET":
        return REPLAY_20260423 / "det/progress/policy_profiles/20260421_fuller_phase1_preflight_det_acc_s0.json"
    if lane == "SPARSE":
        return SPARSE_FIXED_20260425 / "sparse/progress/policy_profiles/20260421_fuller_phase1_preflight_sparse_acc_s0.json"
    raise ValueError(lane)


def write_policy_profile(lane: str, run_id: str) -> Path:
    src = base_policy_profile(lane)
    if not src.is_file():
        raise SystemExit(f"Missing base policy profile for {lane}: {src}")
    payload = json.loads(src.read_text(encoding="utf-8"))
    payload["run_tag"] = RUN_TAG
    payload["lane_id"] = lane
    payload["experiment_family_id"] = "phase4_mechanism_basis_rerun"
    payload["phase4_basis_source_policy_profile"] = rel(src)
    path = PROGRESS_DIR / "policy_profiles" / f"{run_id}.json"
    write_json(path, payload)
    return path


def command_with_common_updates(
    command: list[str],
    *,
    lane: str,
    run_id: str,
    seed: int,
    results_csv: Path,
    append: bool,
    smoke_samples: int | None = None,
) -> list[str]:
    command = list(command)
    command = set_arg(command, "--seed", seed)
    command = set_arg(command, "--device", "mps")
    command = set_arg(command, "--results_csv", rel(results_csv))
    command = set_arg(command, "--run_id", run_id)
    command = set_arg(command, "--source_run_id", run_id)
    command = set_arg(command, "--config_snapshot", rel(RUNS_DIR / run_id / "config_snapshot.yaml"))
    command = set_arg(command, "--progress_jsonl", rel(PROGRESS_DIR / "events" / f"{run_id}.jsonl"))
    command = set_arg(command, "--progress_label", f"{RUN_TAG}:{lane}:seed{seed}")
    command = set_arg(command, "--accuracy_policy_profile_json", rel(write_policy_profile(lane, run_id)))
    command = set_arg(command, "--baseline_reference_csv", rel(ASTRA_ROOT / "raw_accuracy.csv"))
    command = set_arg(command, "--baseline_reference_run_id", "20260421_fuller_phase1_preflight_astra_acc_s0")
    command = set_arg(command, "--bitstream_contract_note", f"current_phase4_mechanism_basis_rerun:{RUN_TAG}")
    command = ensure_switch(command, "--append", append)
    command = ensure_switch(command, "--resume", True)
    if smoke_samples is None:
        command = remove_arg(command, "--max_eval_samples")
    else:
        command = set_arg(command, "--max_eval_samples", smoke_samples)
    return command


def det_sweep_command(k: int, seed: int, append: bool, smoke_samples: int | None = None) -> tuple[list[str], dict[str, Any]]:
    stats = prefix_stats_by_k()[int(k)]
    run_id = f"{RUN_TAG}_det_k{k}_s{seed}" + (f"_smoke{smoke_samples}" if smoke_samples else "")
    results_csv = REPORT_DIR / ("det_sweep/smoke_raw_accuracy.csv" if smoke_samples else "det_sweep/raw_accuracy.csv")
    command = command_with_common_updates(
        base_eval_command("DET"),
        lane="DET",
        run_id=run_id,
        seed=seed,
        results_csv=results_csv,
        append=append,
        smoke_samples=smoke_samples,
    )
    command = set_arg(command, "--experiment_id", "E3")
    command = set_arg(command, "--det_k_global", k)
    command = set_arg(command, "--det_prefix_error_mean", stats["prefix_error_mean"])
    command = set_arg(command, "--det_prefix_error_p95", stats["prefix_error_p95"])
    command = ensure_switch(command, "--apply_det_perturbation", True)
    command = remove_arg(command, "--sparse_tau_global")
    command = remove_arg(command, "--sparse_active_fraction")
    command = ensure_switch(command, "--apply_sparse_perturbation", False)
    return command, stats


def sparse_sweep_command(tau: float, seed: int, append: bool, smoke_samples: int | None = None) -> tuple[list[str], dict[str, Any]]:
    active_fraction = max(0.0, min(1.0, 1.0 - float(tau)))
    tau_label = f"{int(round(tau * 100)):02d}"
    run_id = f"{RUN_TAG}_sparse_t{tau_label}_s{seed}" + (f"_smoke{smoke_samples}" if smoke_samples else "")
    results_csv = REPORT_DIR / ("sparse_sweep/smoke_raw_accuracy.csv" if smoke_samples else "sparse_sweep/raw_accuracy.csv")
    command = command_with_common_updates(
        base_eval_command("SPARSE"),
        lane="SPARSE",
        run_id=run_id,
        seed=seed,
        results_csv=results_csv,
        append=append,
        smoke_samples=smoke_samples,
    )
    command = set_arg(command, "--experiment_id", "E4")
    command = set_arg(command, "--sparse_tau_global", tau)
    command = set_arg(command, "--sparse_active_fraction", active_fraction)
    command = ensure_switch(command, "--apply_sparse_perturbation", True)
    command = remove_arg(command, "--det_k_global")
    command = remove_arg(command, "--det_prefix_error_mean")
    command = remove_arg(command, "--det_prefix_error_p95")
    command = ensure_switch(command, "--apply_det_perturbation", False)
    return command, {"sparse_tau_global": tau, "sparse_active_fraction": active_fraction}


def write_command_file(path: Path, commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        "",
    ]
    lines.extend("caffeinate -dimsu " + shlex.join(command) for command in commands)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def write_queue_script(commands_path: Path, queue_path: Path) -> None:
    log_dir = REPORT_DIR / "logs"
    status_jsonl = REPORT_DIR / "det_sparse_sweep_queue_status.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {shlex.quote(str(ROOT))}",
                f"mkdir -p {shlex.quote(str(log_dir))} {shlex.quote(str(PROGRESS_DIR / 'events'))} {shlex.quote(str(PROGRESS_DIR / 'policy_profiles'))}",
                f"STATUS={shlex.quote(str(status_jsonl))}",
                f"COMMANDS={shlex.quote(str(commands_path))}",
                "echo \"{\\\"event\\\":\\\"queue_start\\\",\\\"ts\\\":\\\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\\\"}\" >> \"$STATUS\"",
                "job_index=0",
                "while IFS= read -r cmd; do",
                "  [[ -z \"$cmd\" || \"$cmd\" == \\#* || \"$cmd\" == set\\ * || \"$cmd\" == cd\\ * || \"$cmd\" == '#!'* ]] && continue",
                "  job_index=$((job_index + 1))",
                "  echo \"{\\\"event\\\":\\\"job_start\\\",\\\"job_index\\\":$job_index,\\\"ts\\\":\\\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\\\"}\" >> \"$STATUS\"",
                "  bash -lc \"$cmd\"",
                "  rc=$?",
                "  echo \"{\\\"event\\\":\\\"job_end\\\",\\\"job_index\\\":$job_index,\\\"rc\\\":$rc,\\\"ts\\\":\\\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\\\"}\" >> \"$STATUS\"",
                "  if [[ $rc -ne 0 ]]; then exit $rc; fi",
                "done < \"$COMMANDS\"",
                "echo \"{\\\"event\\\":\\\"queue_end\\\",\\\"jobs\\\":$job_index,\\\"ts\\\":\\\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\\\"}\" >> \"$STATUS\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    queue_path.chmod(0o755)


def write_sweep_manifest() -> None:
    (REPORT_DIR / "det_sweep").mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "sparse_sweep").mkdir(parents=True, exist_ok=True)
    (PROGRESS_DIR / "events").mkdir(parents=True, exist_ok=True)
    (PROGRESS_DIR / "policy_profiles").mkdir(parents=True, exist_ok=True)

    plan_rows: list[dict[str, Any]] = []
    full_commands: list[list[str]] = []
    smoke_commands: list[list[str]] = []

    for lane_name in ("DET", "SPARSE"):
        lane_first = True
        if lane_name == "DET":
            for k in DET_GRID:
                for seed in SEEDS:
                    command, stats = det_sweep_command(k, seed, append=not lane_first)
                    lane_first = False
                    full_commands.append(command)
                    plan_rows.append(
                        {
                            "run_tag": RUN_TAG,
                            "sweep_lane": "DET",
                            "experiment_id": "E3",
                            "seed": seed,
                            "det_k_global": k,
                            "det_prefix_error_mean": stats["prefix_error_mean"],
                            "det_prefix_error_p95": stats["prefix_error_p95"],
                            "sparse_tau_global": "",
                            "sparse_active_fraction": "",
                            "results_csv": rel(REPORT_DIR / "det_sweep/raw_accuracy.csv"),
                            "run_id": f"{RUN_TAG}_det_k{k}_s{seed}",
                            "command": shlex.join(["caffeinate", "-dimsu", *command]),
                        }
                    )
            smoke, _ = det_sweep_command(64, 0, append=False, smoke_samples=256)
            smoke_commands.append(smoke)
        else:
            for tau in SPARSE_TAU_GRID:
                for seed in SEEDS:
                    command, stats = sparse_sweep_command(tau, seed, append=not lane_first)
                    lane_first = False
                    full_commands.append(command)
                    tau_label = f"{int(round(tau * 100)):02d}"
                    plan_rows.append(
                        {
                            "run_tag": RUN_TAG,
                            "sweep_lane": "SPARSE",
                            "experiment_id": "E4",
                            "seed": seed,
                            "det_k_global": "",
                            "det_prefix_error_mean": "",
                            "det_prefix_error_p95": "",
                            "sparse_tau_global": stats["sparse_tau_global"],
                            "sparse_active_fraction": stats["sparse_active_fraction"],
                            "results_csv": rel(REPORT_DIR / "sparse_sweep/raw_accuracy.csv"),
                            "run_id": f"{RUN_TAG}_sparse_t{tau_label}_s{seed}",
                            "command": shlex.join(["caffeinate", "-dimsu", *command]),
                        }
                    )
            smoke, _ = sparse_sweep_command(0.25, 0, append=False, smoke_samples=256)
            smoke_commands.append(smoke)

    write_csv(
        REPORT_DIR / "det_sparse_sweep_plan.csv",
        plan_rows,
        [
            "run_tag",
            "sweep_lane",
            "experiment_id",
            "seed",
            "det_k_global",
            "det_prefix_error_mean",
            "det_prefix_error_p95",
            "sparse_tau_global",
            "sparse_active_fraction",
            "results_csv",
            "run_id",
            "command",
        ],
    )

    commands_path = REPORT_DIR / "det_sparse_sweep_commands.sh"
    smoke_path = REPORT_DIR / "det_sparse_sweep_smoke_commands.sh"
    queue_path = REPORT_DIR / "run_det_sparse_sweep_queue.sh"
    write_command_file(commands_path, full_commands)
    write_command_file(smoke_path, smoke_commands)
    write_queue_script(commands_path, queue_path)

    prefix_rows = [
        {
            "run_tag": RUN_TAG,
            "det_k_global": k,
            "bsl_max": stats["bsl_max"],
            "prefix_error_mean": stats["prefix_error_mean"],
            "prefix_error_p95": stats["prefix_error_p95"],
            "energy_saved_pct": stats["energy_saved_pct"],
            "source_status": "current_phase4_basis_sweep_input",
        }
        for k, stats in prefix_stats_by_k().items()
    ]
    write_csv(
        QUICK_DIR / "det_prefix_k_grid_phase4_basis.csv",
        prefix_rows,
        [
            "run_tag",
            "det_k_global",
            "bsl_max",
            "prefix_error_mean",
            "prefix_error_p95",
            "energy_saved_pct",
            "source_status",
        ],
    )
    sparse_rows = [
        {
            "run_tag": RUN_TAG,
            "sparse_tau_global": tau,
            "sparse_active_fraction": max(0.0, min(1.0, 1.0 - tau)),
            "source_status": "current_phase4_basis_sweep_input",
        }
        for tau in SPARSE_TAU_GRID
    ]
    write_csv(
        QUICK_DIR / "sparse_tau_grid_phase4_basis.csv",
        sparse_rows,
        ["run_tag", "sparse_tau_global", "sparse_active_fraction", "source_status"],
    )

    write_json(
        REPORT_DIR / "det_sparse_sweep_manifest.json",
        {
            "run_tag": RUN_TAG,
            "generated_at": utc_now(),
            "phase4_basis": "current Phase4 basis; MPS-only; caffeinate-wrapped",
            "accuracy_evidence_tier_target": "analysis_grade",
            "det_grid": DET_GRID,
            "sparse_tau_grid": SPARSE_TAU_GRID,
            "seeds": SEEDS,
            "full_command_count": len(full_commands),
            "smoke_command_count": len(smoke_commands),
            "plan_csv": rel(REPORT_DIR / "det_sparse_sweep_plan.csv"),
            "full_commands_sh": rel(commands_path),
            "smoke_commands_sh": rel(smoke_path),
            "queue_script": rel(queue_path),
            "status_jsonl": rel(REPORT_DIR / "det_sparse_sweep_queue_status.jsonl"),
            "det_results_csv": rel(REPORT_DIR / "det_sweep/raw_accuracy.csv"),
            "sparse_results_csv": rel(REPORT_DIR / "sparse_sweep/raw_accuracy.csv"),
            "notes": "Full sweep is intentionally separated from Phase1 mechanism rerun because it is long-running ImageNet accuracy work.",
        },
    )


def sweep_raw_path(lane: str, *, smoke: bool = False) -> Path:
    if lane == "DET":
        return REPORT_DIR / ("det_sweep/smoke_raw_accuracy.csv" if smoke else "det_sweep/raw_accuracy.csv")
    if lane == "SPARSE":
        return REPORT_DIR / ("sparse_sweep/smoke_raw_accuracy.csv" if smoke else "sparse_sweep/raw_accuracy.csv")
    raise ValueError(lane)


def sweep_annotated_path(lane: str, *, smoke: bool = False) -> Path:
    if lane == "DET":
        return REPORT_DIR / ("det_sweep/smoke_annotated_accuracy.csv" if smoke else "det_sweep/annotated_accuracy.csv")
    if lane == "SPARSE":
        return REPORT_DIR / ("sparse_sweep/smoke_annotated_accuracy.csv" if smoke else "sparse_sweep/annotated_accuracy.csv")
    raise ValueError(lane)


def annotate_sweep_rows(lane: str, rows: list[dict[str, str]], *, smoke: bool = False) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        blockers: list[str] = []
        if str(row.get("device") or "").strip().lower() != "mps":
            blockers.append("device_not_mps")
        if str(row.get("experiment_id") or "").strip() not in {"E3", "E4"}:
            blockers.append("unexpected_experiment_id")
        if to_bool(row.get("baseline")):
            blockers.append("baseline_row")
        processed = int(to_float(row.get("measured_processed_samples"), 0.0))
        expected = 256 if smoke else FULL_SWEEP_EXPECTED_SAMPLES
        if processed < expected:
            blockers.append(f"processed_samples_lt_{expected}")
        if lane == "DET" and not str(row.get("det_k_global") or "").strip():
            blockers.append("missing_det_k_global")
        if lane == "SPARSE" and not str(row.get("sparse_tau_global") or "").strip():
            blockers.append("missing_sparse_tau_global")
        ready = not blockers
        status = "current_phase4_basis_smoke" if smoke else "current_phase4_basis_full_sweep"
        annotated.append(
            {
                **row,
                "run_tag": RUN_TAG,
                "sweep_lane": lane,
                "accuracy_evidence_tier": "mps_smoke_path_validation" if smoke else "analysis_grade_sweep",
                "analysis_grade_ready": str(bool(ready and not smoke)).lower(),
                "sweep_smoke_ready": str(bool(ready and smoke)).lower(),
                "analysis_grade_blockers": json.dumps(blockers, ensure_ascii=False),
                "source_status": status,
                "claim_boundary": (
                    "smoke_path_validation_not_paper_evidence"
                    if smoke
                    else "current_phase4_basis_sweep_measured; inspect accuracy drop before claims"
                ),
            }
        )
    return annotated


def write_annotated_sweep(lane: str, *, smoke: bool = False) -> list[dict[str, Any]]:
    raw_path = sweep_raw_path(lane, smoke=smoke)
    if not raw_path.is_file():
        return []
    rows = read_csv(raw_path)
    annotated = annotate_sweep_rows(lane, rows, smoke=smoke)
    if not annotated:
        return []
    fieldnames = list(rows[0].keys())
    for extra in [
        "run_tag",
        "sweep_lane",
        "accuracy_evidence_tier",
        "analysis_grade_ready",
        "sweep_smoke_ready",
        "analysis_grade_blockers",
        "source_status",
        "claim_boundary",
    ]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    write_csv(sweep_annotated_path(lane, smoke=smoke), annotated, fieldnames)
    return annotated


def group_mean(rows: list[dict[str, Any]], field: str) -> float:
    return mean([to_float(row.get(field)) for row in rows])


def group_std(rows: list[dict[str, Any]], field: str) -> float:
    return stdev([to_float(row.get(field)) for row in rows])


def summarize_sweep_group(
    *,
    lane: str,
    key_field: str,
    key_value: Any,
    rows: list[dict[str, Any]],
    astra_top1: float,
    smoke: bool,
) -> dict[str, Any]:
    top1_mean = group_mean(rows, "top1")
    top5_mean = group_mean(rows, "top5")
    processed = [to_float(row.get("measured_processed_samples")) for row in rows]
    seeds = sorted({str(int(to_float(row.get("seed"), -1))) for row in rows if math.isfinite(to_float(row.get("seed"), math.nan))})
    ready_field = "sweep_smoke_ready" if smoke else "analysis_grade_ready"
    ready = all(to_bool(row.get(ready_field)) for row in rows)
    target_count = 1 if smoke else len(SEEDS)
    complete = len(seeds) >= target_count and ready
    return {
        "run_tag": RUN_TAG,
        "sweep_lane": lane,
        key_field: key_value,
        "rows": len(rows),
        "seeds": ";".join(seeds),
        "seed_count": len(seeds),
        "complete": str(bool(complete)).lower(),
        "top1_mean": top1_mean,
        "top1_std": group_std(rows, "top1"),
        "top5_mean": top5_mean,
        "top5_std": group_std(rows, "top5"),
        "delta_vs_astra_pp": top1_mean - astra_top1 if math.isfinite(top1_mean) else math.nan,
        "acc_drop_vs_astra_pp": astra_top1 - top1_mean if math.isfinite(top1_mean) else math.nan,
        "latency_ms_per_sample_mean": group_mean(rows, "latency_ms_per_sample"),
        "measured_processed_samples_min": min([v for v in processed if math.isfinite(v)] or [math.nan]),
        "det_prefix_error_mean": group_mean(rows, "det_prefix_error_mean") if lane == "DET" else "",
        "det_prefix_error_p95": group_mean(rows, "det_prefix_error_p95") if lane == "DET" else "",
        "sparse_active_fraction_mean": group_mean(rows, "sparse_active_fraction") if lane == "SPARSE" else "",
        "evidence_tier": "mps_smoke_path_validation" if smoke else "analysis_grade_sweep",
        "source_status": "current_phase4_basis_smoke" if smoke else "current_phase4_basis_full_sweep",
        "claim_boundary": (
            "smoke_path_validation_not_paper_evidence"
            if smoke
            else "current_phase4_basis_sweep_measured; preserve mechanism narrative only where accuracy tradeoff supports it"
        ),
    }


def summarize_sweep_rows(annotated: list[dict[str, Any]], *, lane: str, smoke: bool) -> list[dict[str, Any]]:
    if not annotated:
        return []
    astra_summary = read_csv(QUICK_DIR / "mechanism_phase4_lane_summary.csv")
    astra_row = next(row for row in astra_summary if row.get("lane") == "ASTRA")
    astra_top1 = to_float(astra_row.get("top1_mean"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    key_field = "det_k_global" if lane == "DET" else "sparse_tau_global"
    for row in annotated:
        key = format_float(row.get(key_field), digits=12)
        grouped.setdefault(key, []).append(row)
    rows = [
        summarize_sweep_group(
            lane=lane,
            key_field=key_field,
            key_value=key,
            rows=group,
            astra_top1=astra_top1,
            smoke=smoke,
        )
        for key, group in grouped.items()
    ]
    rows.sort(key=lambda row: to_float(row.get(key_field), math.inf))
    return rows


def postprocess_sweep(*, smoke: bool = False) -> None:
    det_rows = write_annotated_sweep("DET", smoke=smoke)
    sparse_rows = write_annotated_sweep("SPARSE", smoke=smoke)
    det_summary = summarize_sweep_rows(det_rows, lane="DET", smoke=smoke)
    sparse_summary = summarize_sweep_rows(sparse_rows, lane="SPARSE", smoke=smoke)
    suffix = "_smoke" if smoke else ""
    det_fields = [
        "run_tag",
        "sweep_lane",
        "det_k_global",
        "rows",
        "seeds",
        "seed_count",
        "complete",
        "top1_mean",
        "top1_std",
        "top5_mean",
        "top5_std",
        "delta_vs_astra_pp",
        "acc_drop_vs_astra_pp",
        "latency_ms_per_sample_mean",
        "measured_processed_samples_min",
        "det_prefix_error_mean",
        "det_prefix_error_p95",
        "evidence_tier",
        "source_status",
        "claim_boundary",
    ]
    sparse_fields = [
        "run_tag",
        "sweep_lane",
        "sparse_tau_global",
        "rows",
        "seeds",
        "seed_count",
        "complete",
        "top1_mean",
        "top1_std",
        "top5_mean",
        "top5_std",
        "delta_vs_astra_pp",
        "acc_drop_vs_astra_pp",
        "latency_ms_per_sample_mean",
        "measured_processed_samples_min",
        "sparse_active_fraction_mean",
        "evidence_tier",
        "source_status",
        "claim_boundary",
    ]
    write_csv(QUICK_DIR / f"det_k_sweep_phase4_basis{suffix}.csv", det_summary, det_fields)
    write_csv(QUICK_DIR / f"sparse_tau_sweep_phase4_basis{suffix}.csv", sparse_summary, sparse_fields)

    appf6_rows: list[dict[str, Any]] = []
    for row in det_summary:
        appf6_rows.append(
            {
                "run_tag": RUN_TAG,
                "row_type": "det_k_sweep",
                "experiment_id": "E3",
                "mechanism_label": "DET k sweep",
                "det_k_global": row.get("det_k_global", ""),
                "sparse_tau_global": "",
                "speedup_vs_E0": "",
                "prefix_error_mean": row.get("det_prefix_error_mean", ""),
                "paired_delta_vs_e0_quant_pp": row.get("delta_vs_astra_pp", ""),
                "paired_delta_vs_fp32_pp": "",
                "top1_mean": row.get("top1_mean", ""),
                "top5_mean": row.get("top5_mean", ""),
                "seed_count": row.get("seed_count", ""),
                "complete": row.get("complete", ""),
                "source_status": row.get("source_status", ""),
                "claim_boundary": row.get("claim_boundary", ""),
                "notes": "Current Phase4-basis DET sweep; delta is lane top1 minus ASTRA quantized top1 mean.",
            }
        )
    for row in sparse_summary:
        appf6_rows.append(
            {
                "run_tag": RUN_TAG,
                "row_type": "sparse_tau_sweep",
                "experiment_id": "E4",
                "mechanism_label": "SPARSE tau sweep",
                "det_k_global": "",
                "sparse_tau_global": row.get("sparse_tau_global", ""),
                "speedup_vs_E0": "",
                "prefix_error_mean": "",
                "paired_delta_vs_e0_quant_pp": row.get("delta_vs_astra_pp", ""),
                "paired_delta_vs_fp32_pp": "",
                "top1_mean": row.get("top1_mean", ""),
                "top5_mean": row.get("top5_mean", ""),
                "seed_count": row.get("seed_count", ""),
                "complete": row.get("complete", ""),
                "source_status": row.get("source_status", ""),
                "claim_boundary": row.get("claim_boundary", ""),
                "notes": "Current Phase4-basis SPARSE tau sweep; delta is lane top1 minus ASTRA quantized top1 mean.",
            }
        )
    write_csv(
        QUICK_DIR / f"appf6_det_sparse_sweep_phase4_basis{suffix}.csv",
        appf6_rows,
        [
            "run_tag",
            "row_type",
            "experiment_id",
            "mechanism_label",
            "det_k_global",
            "sparse_tau_global",
            "speedup_vs_E0",
            "prefix_error_mean",
            "paired_delta_vs_e0_quant_pp",
            "paired_delta_vs_fp32_pp",
            "top1_mean",
            "top5_mean",
            "seed_count",
            "complete",
            "source_status",
            "claim_boundary",
            "notes",
        ],
    )
    write_json(
        REPORT_DIR / f"det_sparse_sweep_postprocess_report{suffix}.json",
        {
            "run_tag": RUN_TAG,
            "generated_at": utc_now(),
            "smoke": smoke,
            "det_raw_csv": rel(sweep_raw_path("DET", smoke=smoke)),
            "sparse_raw_csv": rel(sweep_raw_path("SPARSE", smoke=smoke)),
            "det_annotated_rows": len(det_rows),
            "sparse_annotated_rows": len(sparse_rows),
            "det_summary_rows": len(det_summary),
            "sparse_summary_rows": len(sparse_summary),
            "full_sweep_expected_samples": FULL_SWEEP_EXPECTED_SAMPLES,
            "outputs": [
                rel(QUICK_DIR / f"det_k_sweep_phase4_basis{suffix}.csv"),
                rel(QUICK_DIR / f"sparse_tau_sweep_phase4_basis{suffix}.csv"),
                rel(QUICK_DIR / f"appf6_det_sparse_sweep_phase4_basis{suffix}.csv"),
            ],
        },
    )


def launch_sweep_queue() -> None:
    queue_path = REPORT_DIR / "run_det_sparse_sweep_queue.sh"
    if not queue_path.is_file():
        raise SystemExit(f"Missing queue script: {queue_path}")
    log_path = REPORT_DIR / "logs/det_sparse_sweep_queue.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            ["bash", str(queue_path)],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write_json(
        REPORT_DIR / "det_sparse_sweep_queue_launch.json",
        {
            "run_tag": RUN_TAG,
            "launched_at": utc_now(),
            "pid": process.pid,
            "queue_script": rel(queue_path),
            "log_path": rel(log_path),
            "status_jsonl": rel(REPORT_DIR / "det_sparse_sweep_queue_status.jsonl"),
        },
    )
    print(f"[sweep] launched queue pid={process.pid} log={rel(log_path)}")


def write_next_prompt() -> None:
    prompt = f"""继续在 /Users/jk6k/Desktop/fyp 工作。遵守 AGENTS.md：本机加速实验只用 MPS，长跑用 caffeinate -dimsu，不要 CUDA/CPU fallback。

当前目标：把机制叙事补回当前 Phase4 basis，而不是旧 retained-context。优先检查并使用这些产物：
- report manifest: experiments/results/report_data/{RUN_TAG}/mechanism_basis_manifest.json
- Phase1 机制/能量 quick reports: experiments/results/quick_reports/{RUN_TAG}/mechanism_phase4_lane_summary.csv, appf4_mechanism_ablation_context.csv, appf5_mechanism_energy_breakdown.csv
- DET/SPARSE sweep manifest: experiments/results/report_data/{RUN_TAG}/det_sparse_sweep_manifest.json
- sweep 状态: experiments/results/report_data/{RUN_TAG}/det_sparse_sweep_queue_status.jsonl
- 后处理命令: python3 experiments/tools/build_fuller_phase4_mechanism_basis_rerun.py --postprocess-sweep
- smoke 后处理已验证: experiments/results/quick_reports/{RUN_TAG}/appf6_det_sparse_sweep_phase4_basis_smoke.csv

下一步：
1. 确认 DET/SPARSE full sweep 是否已完成；若未启动，用 experiments/results/report_data/{RUN_TAG}/run_det_sparse_sweep_queue.sh 启动，或先跑 smoke_commands。
2. 完成后把 DET/SPARSE raw_accuracy.csv 注释/聚合成 current Phase4 sweep quick reports。
3. 将 AppF4/AppF5/AppF6 从旧 retained-context 改为 current_phase4_basis，并重渲染/验证数据图集。
4. 保留机制叙事但明确 claim boundary：SPARSE/FULLER 当前仍不能写正向 accuracy-preservation claim，除非新 sweep 证明。"""
    out = ROOT / f"docs/reports/{RUN_TAG}_next_thread_prompt.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(prompt + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Write current-basis Phase1 configs.")
    parser.add_argument("--run-phase1", action="store_true", help="Run Phase1 mechanism/energy lanes with MPS caffeinate.")
    parser.add_argument("--force-phase1", action="store_true", help="Rerun Phase1 even if summaries exist.")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate Phase1 + Phase4 accuracy quick reports.")
    parser.add_argument("--write-sweep-manifest", action="store_true", help="Write DET/SPARSE sweep commands and manifest.")
    parser.add_argument("--launch-sweep", action="store_true", help="Launch the full DET/SPARSE sweep queue in the background.")
    parser.add_argument("--postprocess-sweep", action="store_true", help="Annotate and summarize full DET/SPARSE sweep rows.")
    parser.add_argument("--postprocess-smoke", action="store_true", help="Annotate and summarize DET/SPARSE smoke rows.")
    parser.add_argument("--write-next-prompt", action="store_true", help="Write a compact next-thread prompt.")
    parser.add_argument("--all", action="store_true", help="Prepare, run Phase1, aggregate, and write sweep manifest/prompt.")
    args = parser.parse_args()

    if args.all:
        args.prepare = True
        args.run_phase1 = True
        args.aggregate = True
        args.write_sweep_manifest = True
        args.write_next_prompt = True

    if args.prepare:
        prepare_configs()
    if args.run_phase1:
        run_phase1(force=args.force_phase1)
    if args.aggregate:
        aggregate_quick_reports()
    if args.write_sweep_manifest:
        write_sweep_manifest()
    if args.launch_sweep:
        launch_sweep_queue()
    if args.postprocess_sweep:
        postprocess_sweep(smoke=False)
    if args.postprocess_smoke:
        postprocess_sweep(smoke=True)
    if args.write_next_prompt:
        write_next_prompt()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
