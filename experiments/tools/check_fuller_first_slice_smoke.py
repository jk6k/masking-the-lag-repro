#!/usr/bin/env python3
"""Validate the active fuller slice template with a synthetic integrated smoke gate."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

try:
    from .path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path
    from .phase1_runner import (
        EXPERIMENT_SWITCH_MATRIX,
        _aggregate_buffer_trace_for_model,
        _aggregate_timeline_for_model,
        _build_per_layer_timeline_rows,
        _compute_integrated_system_costs,
        _resolve_existing_path,
        _resolve_switches,
        _sync_section_enabled,
        _timeline_latency_ms_from_stage_cycles,
    )
except ImportError:
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path  # type: ignore
    from phase1_runner import (  # type: ignore
        EXPERIMENT_SWITCH_MATRIX,
        _aggregate_buffer_trace_for_model,
        _aggregate_timeline_for_model,
        _build_per_layer_timeline_rows,
        _compute_integrated_system_costs,
        _resolve_existing_path,
        _resolve_switches,
        _sync_section_enabled,
        _timeline_latency_ms_from_stage_cycles,
    )

try:
    from ..exp_common.meso_cost_model import compute_meso_cost_model
except ImportError:
    from exp_common.meso_cost_model import compute_meso_cost_model  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "configs" / "fuller_det_sparse_reentry_slice_template_20260331.yaml"
DEFAULT_OUT_DIR = MAIN_PROJECT_REPORT_DATA_DIR


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return data


def _record(rows: list[dict[str, str]], check_id: str, status: str, detail: str) -> None:
    rows.append({"check_id": check_id, "status": status, "detail": detail})


def _example_op() -> dict[str, float | str]:
    return {
        "name": "layer_0",
        "type": "gemm",
        "latency_ms": 1.0,
        "energy_mj_load_x": 2.0,
        "energy_mj_load_y": 2.0,
        "energy_mj_oe": 1.0,
        "energy_mj_adc_pca": 1.0,
        "energy_mj_laser": 1.0,
        "energy_mj_mem": 3.0,
        "energy_mj_static": 0.5,
    }


def _summarize_flow(*, flow_enabled: bool, flow_cfg: dict[str, Any]) -> tuple[float, int, float, int, float]:
    timeline_rows, buffer_rows = _build_per_layer_timeline_rows(
        model="mobilevit_s",
        scaled_ops=[_example_op()],
        sample_rate_gsps=1.0,
        flow_enabled=flow_enabled,
        det_enabled=False,
        flow_cfg=flow_cfg,
    )
    stage_cycles, bubble_cycles, utilization_avg = _aggregate_timeline_for_model(
        model="mobilevit_s",
        per_layer_timeline_rows=timeline_rows,
    )
    peak_cycles, peak_frac = _aggregate_buffer_trace_for_model(
        model="mobilevit_s",
        per_layer_buffer_rows=buffer_rows,
    )
    latency_ms = _timeline_latency_ms_from_stage_cycles(stage_cycles, 1.0)
    return latency_ms, bubble_cycles, utilization_avg, peak_cycles, peak_frac


def evaluate_template(template: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checks: list[dict[str, str]] = []

    baseline_substrate = template.get("baseline_substrate") or {}
    run_cfg = template.get("run") or {}
    experiment_id = str(run_cfg.get("experiment_id") or "").strip()
    _record(
        checks,
        "baseline_substrate.astra_role",
        "pass"
        if (
            str(baseline_substrate.get("family")) == "ASTRA-style"
            and str(baseline_substrate.get("role")) == "baseline_execution_substrate"
            and baseline_substrate.get("deterministic_structure_explicit") is True
            and baseline_substrate.get("shared_conversion_serialization_present") is True
            and str(baseline_substrate.get("novelty_guardrail")) == "baseline_only_not_active_layer"
        )
        else "fail",
        repr(baseline_substrate),
    )
    _record(
        checks,
        "run.experiment_id",
        "pass" if experiment_id and experiment_id not in EXPERIMENT_SWITCH_MATRIX else "fail",
        f"experiment_id={experiment_id!r}",
    )
    _record(
        checks,
        "run.device",
        "pass" if str(run_cfg.get("device")) == "mps" else "fail",
        f"device={run_cfg.get('device')!r}",
    )
    _record(
        checks,
        "run.execution_surface",
        "pass"
        if str(run_cfg.get("execution_surface")) == "host_unsandboxed_caffeinate_required"
        else "fail",
        f"execution_surface={run_cfg.get('execution_surface')!r}",
    )

    resolved_switches = _resolve_switches(template, experiment_id)
    expected_switches = {"meso": True, "flow": True, "det": True, "sparse": True, "phy": True}
    _record(
        checks,
        "switches.resolved",
        "pass" if resolved_switches == expected_switches else "fail",
        f"switches={resolved_switches!r}",
    )

    synced = copy.deepcopy(template)
    _sync_section_enabled(synced, resolved_switches)
    section_ok = (
        bool((synced.get("meso") or {}).get("enabled")) is True
        and bool((synced.get("flow") or {}).get("enabled")) is True
        and bool((synced.get("phy") or {}).get("enabled")) is True
        and bool((synced.get("sparse") or {}).get("enabled")) is True
        and bool((((synced.get("sc_det") or {}).get("early_stop") or {}).get("enabled")) is True)
    )
    _record(checks, "switches.section_sync", "pass" if section_ok else "fail", repr({
        "meso": (synced.get("meso") or {}).get("enabled"),
        "flow": (synced.get("flow") or {}).get("enabled"),
        "phy": (synced.get("phy") or {}).get("enabled"),
        "sparse": (synced.get("sparse") or {}).get("enabled"),
        "det": (((synced.get("sc_det") or {}).get("early_stop") or {}).get("enabled")),
    }))

    accuracy_source_raw = str((synced.get("accuracy") or {}).get("source_csv") or "").strip()
    baseline_ref_raw = str(((synced.get("baseline_ref") or {}).get("e0_latency_csv")) or "").strip()
    if accuracy_source_raw:
        accuracy_source = _resolve_existing_path(accuracy_source_raw)
        _record(checks, "paths.accuracy_source_csv", "pass" if accuracy_source.exists() else "fail", str(accuracy_source))
    else:
        _record(checks, "paths.accuracy_source_csv", "pass", "null (reduced-workspace optional)")
    if baseline_ref_raw:
        baseline_ref = _resolve_existing_path(baseline_ref_raw)
        _record(checks, "paths.baseline_ref_e0_latency_csv", "pass" if baseline_ref.exists() else "fail", str(baseline_ref))
    else:
        _record(checks, "paths.baseline_ref_e0_latency_csv", "pass", "null (reduced-workspace optional)")
    accuracy_cfg = synced.get("accuracy") or {}
    accuracy_anchor_ok = (
        str(accuracy_cfg.get("context_run_id") or "") in {
            "",
            "20260314_canonical_mainchain_v1_e0",
            "20260228_opt_sync_core_e6",
        }
        and bool(accuracy_cfg.get("require_context_match")) is False
    )
    _record(
        checks,
        "accuracy.anchor_context",
        "pass" if accuracy_anchor_ok else "fail",
        (
            f"context_run_id={accuracy_cfg.get('context_run_id')!r} "
            f"require_context_match={accuracy_cfg.get('require_context_match')!r}"
        ),
    )

    det_hook = synced.get("det_reentry_hook")
    det_hook_ok = det_hook in (None, {})
    _record(
        checks,
        "det_reentry_hook.off_active_surface",
        "pass" if det_hook_ok else "fail",
        f"det_reentry_hook={det_hook!r}",
    )

    baseline_flow = _summarize_flow(
        flow_enabled=False,
        flow_cfg={
            "buffer_depth": 0,
            "overlap_efficiency": 0.0,
            "staging_cost_scale": 1.0,
            "sync_penalty_scale": 1.0,
        },
    )
    flowed = _summarize_flow(
        flow_enabled=True,
        flow_cfg=synced.get("flow") or {},
    )
    flow_ok = flowed[0] < baseline_flow[0] and flowed[1] < baseline_flow[1] and flowed[2] > baseline_flow[2]
    _record(
        checks,
        "flow.synthetic_smoke",
        "pass" if flow_ok and flowed[4] >= 0.0 else "fail",
        f"baseline_latency_ms={baseline_flow[0]} flowed_latency_ms={flowed[0]} baseline_bubbles={baseline_flow[1]} flowed_bubbles={flowed[1]} baseline_util={baseline_flow[2]} flowed_util={flowed[2]} peak_frac={flowed[4]}",
    )

    meso_metrics = compute_meso_cost_model(
        meso_cfg=synced.get("meso") or {},
        meso_enabled=resolved_switches["meso"],
        latency_s=max(flowed[0] / 1e3, 1e-9),
    )
    meso_ok = (
        meso_metrics["cost_model_mode"] == "explicit_topology_v1"
        and meso_metrics["load_scale_applied"] is False
        and all(
            float(meso_metrics[key]) >= 0.0
            for key in (
                "serializer_energy_j",
                "broadcast_driver_energy_j",
                "fabric_control_overhead_j",
                "extra_buffering_overhead_j",
                "explicit_total_cost_j",
                "explicit_total_savings_j",
            )
        )
    )
    _record(
        checks,
        "meso.synthetic_smoke",
        "pass" if meso_ok else "fail",
        (
            f"cost_model_mode={meso_metrics['cost_model_mode']!r} "
            f"load_scale_applied={meso_metrics['load_scale_applied']!r} "
            f"net_energy_gain_j={meso_metrics['net_energy_gain_j']!r}"
        ),
    )

    cost_cfg = synced.get("integrated_system_costs") or {}
    integrated_on = _compute_integrated_system_costs(
        conversion_control_j=10.0,
        memory_move_j=20.0,
        thermal_energy_j=4.0,
        flow_enabled=resolved_switches["flow"],
        phy_enabled=resolved_switches["phy"],
        noise_enabled=False,
        cost_cfg=cost_cfg,
    )
    integrated_off = _compute_integrated_system_costs(
        conversion_control_j=10.0,
        memory_move_j=20.0,
        thermal_energy_j=4.0,
        flow_enabled=resolved_switches["flow"],
        phy_enabled=False,
        noise_enabled=False,
        cost_cfg=cost_cfg,
    )
    integrated_ok = (
        float(integrated_on["integrated_host_staging_j"]) > 0.0
        and float(integrated_off["integrated_calibration_monitoring_j"]) == 0.0
        and (
            (
                float(cost_cfg.get("calibration_monitoring_scale_vs_thermal") or 0.0) > 0.0
                and float(integrated_on["integrated_calibration_monitoring_j"]) > 0.0
            )
            or (
                float(cost_cfg.get("calibration_monitoring_scale_vs_thermal") or 0.0) == 0.0
                and float(integrated_on["integrated_calibration_monitoring_j"]) == 0.0
            )
        )
    )
    _record(
        checks,
        "phy_bounded.synthetic_smoke",
        "pass" if integrated_ok else "fail",
        (
            f"host_staging_on={integrated_on['integrated_host_staging_j']!r} "
            f"calibration_on={integrated_on['integrated_calibration_monitoring_j']!r} "
            f"calibration_off={integrated_off['integrated_calibration_monitoring_j']!r}"
        ),
    )

    summary = {
        "experiment_id": experiment_id,
        "resolved_switches": resolved_switches,
        "flow_latency_ms": flowed[0],
        "meso_cost_model_mode": str(meso_metrics["cost_model_mode"]),
        "astra_family": str(baseline_substrate.get("family") or ""),
        "overall_ok": all(row["status"] == "pass" for row in checks),
    }
    return checks, summary


def _write_report(path: Path, checks: list[dict[str, str]], summary: dict[str, Any], template_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fuller First-Slice Synthetic Smoke Report",
        "",
        "Scope",
        f"- Template: `{template_path}`",
        f"- Experiment id: `{summary['experiment_id']}`",
        f"- Resolved switches: `{summary['resolved_switches']}`",
        "",
        "Checks",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the synthetic smoke gate for the active fuller slice template.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    template_path = assert_main_project_path(resolve_repo_path(args.template), arg_name="--template")
    out_dir = assert_main_project_path(resolve_repo_path(args.out_dir), arg_name="--out_dir")
    template = _load_yaml(template_path)
    checks, summary = evaluate_template(template)
    tag = (template.get("canonical_chain") or {}).get("tag") or "unknown"
    report_path = out_dir / f"fuller_first_slice_smoke_{tag}.md"
    _write_report(report_path, checks, summary, template_path)
    overall = "OK" if summary["overall_ok"] else "FAIL"
    print(f"[fuller-first-slice-smoke] overall={overall} report={report_path}")
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
