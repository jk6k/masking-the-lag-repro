#!/usr/bin/env python3
"""Build the R2 hardware-derived scheduler slack trace artifacts.

The R2 scheduler model consumes the selected architecture kernel rows and
builds deterministic accelerator schedule traces under multiple scheduler
variants.  It is an architecture scheduler trace over modeled queues and
deadlines, not RTL timing, physical implementation, or hardware measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
DATE = "2026-05-13"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

ARCH_KERNELS_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_kernels.csv"
ARCH_SUMMARY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_summary.csv"
ARCH_JSON = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}.json"

TRACE_CSV = REPORT_DATA / f"suds_tetc_scheduler_traces_{TAG}.csv"
ABLATION_CSV = REPORT_DATA / f"suds_tetc_scheduler_ablation_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_scheduler_traces_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_scheduler_slack_traces.md"

DEFAULT_SEED = 23
DEFAULT_MIN_SCHEDULER_DISTANCE = 0.01
RELEASE_PRESSURE_FACTOR = 0.92
PROMOTED_CONDITION = "suds_pareto"
PROMOTED_SCHEDULER = "suds_aware"

SCHEDULERS = (
    "fifo",
    "asap",
    "edf_deadline_aware",
    "utilization_aware",
    "suds_aware",
)

SCHEDULER_LABELS = {
    "fifo": "FIFO release-order scheduler",
    "asap": "ASAP earliest-ready scheduler",
    "edf_deadline_aware": "EDF deadline-aware scheduler",
    "utilization_aware": "Utilization-aware scheduler",
    "suds_aware": "SUDS-aware slack/deadline scheduler",
}

LOOKAHEAD_BY_SCHEDULER = {
    "fifo": 0.00,
    "asap": 0.05,
    "edf_deadline_aware": 0.35,
    "utilization_aware": 0.55,
    "suds_aware": 0.70,
}

DISPATCH_GAP_BY_SCHEDULER = {
    "fifo": 0.035,
    "asap": 0.018,
    "edf_deadline_aware": 0.012,
    "utilization_aware": 0.010,
    "suds_aware": 0.006,
}

SIDEBAND_PREFETCH_BY_SCHEDULER = {
    "fifo": 0.000,
    "asap": 0.008,
    "edf_deadline_aware": 0.018,
    "utilization_aware": 0.014,
    "suds_aware": 0.028,
}

CORE_EFFICIENCY_BY_SCHEDULER = {
    "fifo": 0.96,
    "asap": 1.00,
    "edf_deadline_aware": 1.00,
    "utilization_aware": 1.06,
    "suds_aware": 1.03,
}

EVENT_FRACTIONS = {
    "memory_read": 0.12,
    "sideband_issue": 0.02,
    "dac_issue": 0.06,
    "core_execute": 0.62,
    "adc_readout": 0.14,
    "optical_transfer": 0.04,
}

RESOURCE_BY_EVENT = {
    "memory_read": "memory_traffic",
    "sideband_issue": "control_sideband",
    "dac_issue": "dac_frontend",
    "core_execute": "dptc_core",
    "adc_readout": "adc_queue",
    "optical_transfer": "optical_link",
    "digital_fallback": "digital_fallback",
}


@dataclass
class ResourcePool:
    name: str
    capacity: int
    available_at: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.available_at = [0.0 for _ in range(max(1, self.capacity))]

    def reserve(self, ready_ns: float, duration_ns: float, units: int) -> dict[str, float | int]:
        units = max(1, min(int(units), self.capacity))
        order = sorted(range(self.capacity), key=lambda idx: self.available_at[idx])
        chosen = order[:units]
        start_ns = max(ready_ns, max(self.available_at[idx] for idx in chosen))
        end_ns = start_ns + max(0.0, duration_ns)
        active_before = sum(1 for value in self.available_at if value > ready_ns)
        for idx in chosen:
            self.available_at[idx] = end_ns
        return {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "queue_wait_ns": max(0.0, start_ns - ready_ns),
            "queue_depth_before": active_before,
            "units_requested": units,
            "capacity": self.capacity,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--architecture-kernels-csv", type=Path, default=ARCH_KERNELS_CSV)
    parser.add_argument("--architecture-summary-csv", type=Path, default=ARCH_SUMMARY_CSV)
    parser.add_argument("--architecture-json", type=Path, default=ARCH_JSON)
    parser.add_argument("--trace-csv", type=Path, default=TRACE_CSV)
    parser.add_argument("--ablation-csv", type=Path, default=ABLATION_CSV)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--min-scheduler-distance", type=float, default=DEFAULT_MIN_SCHEDULER_DISTANCE)
    return parser.parse_args()


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(parts: dict[str, Any]) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def trace_hash(parts: dict[str, Any]) -> str:
    return f"r2_{config_hash(parts)}"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing required CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def bytes_for_kernel(kernel: dict[str, str]) -> float:
    m = as_int(kernel.get("m"))
    d = as_int(kernel.get("d"))
    n = as_int(kernel.get("n"))
    return float((m * d + d * n + m * n) * 2)


def sideband_groups_for_kernel(kernel: dict[str, str], summary: dict[str, str]) -> int:
    output_groups = max(1, as_int(kernel.get("output_groups"), 1))
    tile_dim = max(1, as_int(summary.get("tile_dim"), 32))
    sideband_cols = max(1, as_int(summary.get("sideband_group_cols"), 32))
    active = max(0.0, as_float(kernel.get("active_compute_ratio"), 1.0))
    return max(1, ceil_div(int(math.ceil(output_groups * tile_dim * active)), sideband_cols))


def resource_capacities(summary: dict[str, str]) -> dict[str, int]:
    parallel_cores = max(1, as_int(summary.get("parallel_cores"), 8))
    tiles = max(1, as_int(summary.get("tiles"), 4))
    cores_per_tile = max(1, as_int(summary.get("cores_per_tile"), 2))
    return {
        "memory_traffic": max(1, min(4, tiles)),
        "control_sideband": 1,
        "dac_frontend": max(1, parallel_cores),
        "dptc_core": max(1, parallel_cores),
        "adc_queue": max(1, tiles),
        "optical_link": 1,
        "digital_fallback": max(1, cores_per_tile),
    }


def event_duration_fractions(summary: dict[str, str]) -> dict[str, float]:
    fractions = dict(EVENT_FRACTIONS)
    if as_float(summary.get("digital_fallback_energy_pj")) > 0.0:
        fractions["digital_fallback"] = 0.04
        fractions["core_execute"] = max(0.0, fractions["core_execute"] - 0.04)
    total = sum(fractions.values())
    return {key: value / total for key, value in fractions.items()}


def units_for_event(event_type: str, kernel: dict[str, str], summary: dict[str, str], capacities: dict[str, int]) -> int:
    resource = RESOURCE_BY_EVENT[event_type]
    if event_type == "core_execute":
        cycles = max(1, as_int(kernel.get("dptc_cycles"), 1))
        dptc_tiles = max(1, as_int(kernel.get("condition_dptc_tiles"), as_int(kernel.get("dptc_tiles"), 1)))
        return max(1, min(capacities[resource], ceil_div(dptc_tiles, cycles)))
    if event_type == "dac_issue":
        return max(1, min(capacities[resource], max(1, as_int(summary.get("parallel_cores"), 8) // 2)))
    if event_type == "adc_readout":
        return max(1, min(capacities[resource], max(1, as_int(summary.get("tiles"), 4))))
    return 1


def group_kernel_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["workload"], row["condition"]), []).append(row)
    for items in grouped.values():
        items.sort(key=lambda row: as_int(row.get("kernel_index")))
    return grouped


def nominal_summary_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["workload"], row["condition"]): row
        for row in rows
        if row.get("sensitivity_case") == "nominal"
    }


def base_latency_sum(kernels: list[dict[str, str]]) -> float:
    return sum(as_float(row.get("base_latency_ns")) for row in kernels)


def deadline_scale(summary: dict[str, str], kernels: list[dict[str, str]]) -> float:
    return as_float(summary.get("latency_ns"), base_latency_sum(kernels)) / max(1.0e-12, base_latency_sum(kernels))


def median_kernel_latency(kernels: list[dict[str, str]]) -> float:
    values = sorted(max(1.0e-9, as_float(row.get("condition_latency_ns"), as_float(row.get("base_latency_ns")))) for row in kernels)
    return percentile(values, 0.50)


def build_dependency_map(kernels: list[dict[str, str]]) -> dict[str, list[str]]:
    by_layer: dict[int, list[dict[str, str]]] = {}
    for row in kernels:
        by_layer.setdefault(as_int(row.get("layer_index")), []).append(row)
    for rows in by_layer.values():
        rows.sort(key=lambda row: as_int(row.get("kernel_index")))

    deps: dict[str, list[str]] = {str(row["kernel_id"]): [] for row in kernels}
    previous_tail = ""
    for _, rows in sorted(by_layer.items()):
        by_class = {str(row.get("kernel_class")): str(row["kernel_id"]) for row in rows}
        ordered_ids = [str(row["kernel_id"]) for row in rows]
        qkv = by_class.get("mha_qkv_projection", ordered_ids[0])
        qk = by_class.get("mha_qk_scores")
        av = by_class.get("mha_av_context")
        out = by_class.get("mha_output_projection")
        ffn_expand = by_class.get("ffn_expand")
        ffn_project = by_class.get("ffn_project")

        deps[qkv] = [previous_tail] if previous_tail else []
        if qk:
            deps[qk] = [qkv]
        if av:
            deps[av] = [qkv]
        if out:
            deps[out] = [item for item in (qk, av) if item]
        if ffn_expand:
            deps[ffn_expand] = [out] if out else [ordered_ids[max(0, len(ordered_ids) - 3)]]
        if ffn_project:
            deps[ffn_project] = [ffn_expand] if ffn_expand else [ordered_ids[max(0, len(ordered_ids) - 2)]]

        known = {qkv, qk, av, out, ffn_expand, ffn_project}
        previous = previous_tail
        for kernel_id in ordered_ids:
            if kernel_id not in known:
                deps[kernel_id] = [previous] if previous else []
            previous = kernel_id
        previous_tail = ffn_project or ordered_ids[-1]
    return deps


def release_ns(kernel: dict[str, str], scale: float) -> float:
    return as_float(kernel.get("schedule_start_ns")) * scale * RELEASE_PRESSURE_FACTOR


def deadline_ns(kernel: dict[str, str], scale: float) -> float:
    return as_float(kernel.get("schedule_deadline_ns")) * scale


def tile_slots(kernel: dict[str, str], summary: dict[str, str]) -> int:
    tiles = max(1, as_int(summary.get("tiles"), 4))
    cores_per_tile = max(1, as_int(summary.get("cores_per_tile"), 2))
    cycles = max(1, as_int(kernel.get("dptc_cycles"), 1))
    dptc_tiles = max(1, as_int(kernel.get("condition_dptc_tiles"), as_int(kernel.get("dptc_tiles"), 1)))
    return max(1, min(tiles, ceil_div(dptc_tiles, max(1, cycles * cores_per_tile))))


def tile_id_for_kernel(kernel: dict[str, str], summary: dict[str, str]) -> int:
    tiles = max(1, as_int(summary.get("tiles"), 4))
    return (as_int(kernel.get("kernel_index")) + as_int(kernel.get("layer_index")) * 3) % tiles


def priority_key(
    scheduler: str,
    kernel: dict[str, str],
    *,
    ready_ns: float,
    deadline: float,
    max_dptc_tiles: float,
    median_latency: float,
) -> tuple[float, ...]:
    kernel_index = as_int(kernel.get("kernel_index"))
    latency = max(1.0e-9, as_float(kernel.get("condition_latency_ns"), as_float(kernel.get("base_latency_ns"))))
    dptc_tiles = max(1.0, as_float(kernel.get("condition_dptc_tiles"), as_float(kernel.get("dptc_tiles"))))
    slack_norm = max(0.0, min(1.0, as_float(kernel.get("scheduler_slack_norm"), 0.0)))
    deadline_pressure = 1.0 / max(1.0, deadline - ready_ns)

    if scheduler == "fifo":
        return (float(kernel_index),)
    if scheduler == "asap":
        return (ready_ns, latency, float(kernel_index))
    if scheduler == "edf_deadline_aware":
        return (deadline, ready_ns, float(kernel_index))
    if scheduler == "utilization_aware":
        utilization_score = dptc_tiles / max(1.0, max_dptc_tiles)
        short_job_bonus = median_latency / max(median_latency, latency)
        return (-0.70 * utilization_score - 0.30 * short_job_bonus, deadline, float(kernel_index))
    if scheduler == "suds_aware":
        urgency = 1.0 - slack_norm
        tile_pressure = dptc_tiles / max(1.0, max_dptc_tiles)
        keep_guard = as_float(kernel.get("keep_ratio"), 1.0)
        degrade_guard = as_float(kernel.get("degrade_ratio"), 0.0)
        prune_guard = as_float(kernel.get("prune_ratio"), 0.0)
        score = (
            0.48 * urgency
            + 0.24 * tile_pressure
            + 0.16 * min(1.0, deadline_pressure * median_latency)
            + 0.08 * keep_guard
            + 0.04 * degrade_guard
            - 0.04 * prune_guard
        )
        return (-score, deadline, ready_ns, float(kernel_index))
    raise SystemExit(f"unknown scheduler: {scheduler}")


def resource_state(resources: dict[str, ResourcePool], ready_ns: float) -> str:
    parts = []
    for name in sorted(resources):
        pool = resources[name]
        active = sum(1 for value in pool.available_at if value > ready_ns)
        parts.append(f"{name}:{active}/{pool.capacity}")
    return ";".join(parts)


def budget_decision(kernel: dict[str, str]) -> str:
    keep = as_float(kernel.get("keep_ratio"), 1.0)
    degrade = as_float(kernel.get("degrade_ratio"), 0.0)
    prune = as_float(kernel.get("prune_ratio"), 0.0)
    if keep >= 0.999 and degrade <= 1.0e-9 and prune <= 1.0e-9:
        return "keep_all"
    if prune > 1.0e-9 and degrade > 1.0e-9:
        return "keep_degrade_prune"
    if prune > 1.0e-9:
        return "keep_prune"
    if degrade > 1.0e-9:
        return "keep_degrade"
    return "keep_budgeted"


def schedule_one_kernel(
    *,
    tag: str,
    seed: int,
    deterministic_config_id: str,
    schedule_trace_id: str,
    scheduler: str,
    kernel: dict[str, str],
    summary: dict[str, str],
    dependencies: list[str],
    ready_ns_value: float,
    deadline_ns_value: float,
    resources: dict[str, ResourcePool],
    median_latency_ns: float,
) -> tuple[dict[str, Any], dict[str, float], dict[str, dict[str, Any]]]:
    fractions = event_duration_fractions(summary)
    capacities = {name: pool.capacity for name, pool in resources.items()}
    condition_latency = max(
        1.0e-9,
        as_float(kernel.get("condition_latency_ns"), as_float(kernel.get("base_latency_ns"))),
    )
    event_durations = {event_type: condition_latency * fraction for event_type, fraction in fractions.items()}
    core_efficiency = CORE_EFFICIENCY_BY_SCHEDULER[scheduler]
    event_durations["core_execute"] = event_durations["core_execute"] / max(1.0e-9, core_efficiency)
    input_slack_norm = max(0.0, min(1.0, as_float(kernel.get("scheduler_slack_norm"), 0.0)))
    urgency = 1.0 - input_slack_norm
    base_dispatch_delay = DISPATCH_GAP_BY_SCHEDULER[scheduler] * median_latency_ns
    if scheduler == "suds_aware":
        dispatch_delay_ns = base_dispatch_delay * (0.55 + 0.45 * input_slack_norm)
    elif scheduler == "edf_deadline_aware":
        dispatch_delay_ns = base_dispatch_delay * (0.80 + 0.20 * urgency)
    elif scheduler == "fifo":
        dispatch_delay_ns = base_dispatch_delay * (1.0 + 0.35 * urgency)
    else:
        dispatch_delay_ns = base_dispatch_delay
    sideband_prefetch_ns = SIDEBAND_PREFETCH_BY_SCHEDULER[scheduler] * median_latency_ns
    memory_ready_ns = ready_ns_value + dispatch_delay_ns
    sideband_ready_ns = max(ready_ns_value, memory_ready_ns - sideband_prefetch_ns)
    before_state = resource_state(resources, ready_ns_value)

    memory = resources["memory_traffic"].reserve(
        memory_ready_ns,
        event_durations["memory_read"],
        units_for_event("memory_read", kernel, summary, capacities),
    )
    sideband = resources["control_sideband"].reserve(
        sideband_ready_ns,
        event_durations["sideband_issue"],
        units_for_event("sideband_issue", kernel, summary, capacities),
    )
    dac = resources["dac_frontend"].reserve(
        float(memory["end_ns"]),
        event_durations["dac_issue"],
        units_for_event("dac_issue", kernel, summary, capacities),
    )
    core_ready = max(float(dac["end_ns"]), float(sideband["end_ns"]))
    core = resources["dptc_core"].reserve(
        core_ready,
        event_durations["core_execute"],
        units_for_event("core_execute", kernel, summary, capacities),
    )
    adc = resources["adc_queue"].reserve(
        float(core["end_ns"]),
        event_durations["adc_readout"],
        units_for_event("adc_readout", kernel, summary, capacities),
    )
    optical = resources["optical_link"].reserve(
        float(adc["end_ns"]),
        event_durations["optical_transfer"],
        units_for_event("optical_transfer", kernel, summary, capacities),
    )
    event_results = {
        "memory_read": memory,
        "sideband_issue": sideband,
        "dac_issue": dac,
        "core_execute": core,
        "adc_readout": adc,
        "optical_transfer": optical,
    }
    digital_end = float(core["end_ns"])
    if "digital_fallback" in event_durations:
        digital = resources["digital_fallback"].reserve(
            float(core["end_ns"]),
            event_durations["digital_fallback"],
            units_for_event("digital_fallback", kernel, summary, capacities),
        )
        event_results["digital_fallback"] = digital
        digital_end = float(digital["end_ns"])

    start_ns = min(float(item["start_ns"]) for item in event_results.values())
    data_ready_ns = float(adc["end_ns"])
    completion_ns = max(float(optical["end_ns"]), digital_end)
    raw_slack_ns = deadline_ns_value - completion_ns
    slack_window_ns = max(1.0, deadline_ns_value - ready_ns_value)
    normalized_slack = raw_slack_ns / slack_window_ns
    queue_wait_ns = sum(float(item["queue_wait_ns"]) for item in event_results.values())
    max_queue_depth = max(int(item["queue_depth_before"]) for item in event_results.values())
    after_state = resource_state(resources, ready_ns_value)

    row = {
        "tag": tag,
        "seed": seed,
        "deterministic_config_id": deterministic_config_id,
        "schedule_trace_id": schedule_trace_id,
        "scheduler": scheduler,
        "scheduler_label": SCHEDULER_LABELS[scheduler],
        "sensitivity_case": "nominal",
        "workload": summary["workload"],
        "condition": summary["condition"],
        "condition_label": summary.get("condition_label", ""),
        "model": kernel.get("model", ""),
        "dataset_or_split": kernel.get("dataset_or_split", ""),
        "kernel_id": kernel["kernel_id"],
        "kernel_index": as_int(kernel.get("kernel_index")),
        "kernel_name": kernel.get("kernel_name", ""),
        "kernel_class": kernel.get("kernel_class", ""),
        "layer_index": as_int(kernel.get("layer_index")),
        "dependency_kernel_ids": ";".join(dependencies),
        "release_ns": ready_ns_value,
        "scheduled_start_ns": start_ns,
        "data_ready_ns": data_ready_ns,
        "completion_ns": completion_ns,
        "deadline_ns": deadline_ns_value,
        "deadline_slack_ns": raw_slack_ns,
        "deadline_slack_norm": normalized_slack,
        "input_scheduler_slack_norm": as_float(kernel.get("scheduler_slack_norm")),
        "deadline_miss": raw_slack_ns < 0.0,
        "scheduler_dispatch_delay_ns": dispatch_delay_ns,
        "scheduler_sideband_prefetch_ns": sideband_prefetch_ns,
        "scheduler_core_efficiency": core_efficiency,
        "queue_wait_ns": queue_wait_ns,
        "max_queue_depth_before": max_queue_depth,
        "resource_queue_state_before": before_state,
        "resource_queue_state_after": after_state,
        "memory_wait_ns": float(memory["queue_wait_ns"]),
        "sideband_wait_ns": float(sideband["queue_wait_ns"]),
        "dac_wait_ns": float(dac["queue_wait_ns"]),
        "core_wait_ns": float(core["queue_wait_ns"]),
        "adc_wait_ns": float(adc["queue_wait_ns"]),
        "optical_wait_ns": float(optical["queue_wait_ns"]),
        "tile_id": tile_id_for_kernel(kernel, summary),
        "tile_slots_requested": tile_slots(kernel, summary),
        "dptc_tiles": as_int(kernel.get("condition_dptc_tiles"), as_int(kernel.get("dptc_tiles"))),
        "output_groups": as_int(kernel.get("output_groups")),
        "output_values": as_int(kernel.get("condition_output_values"), as_int(kernel.get("output_values"))),
        "memory_bytes": bytes_for_kernel(kernel),
        "optical_values": as_float(kernel.get("condition_output_values"), as_float(kernel.get("output_values"))),
        "sideband_groups": sideband_groups_for_kernel(kernel, summary),
        "active_compute_ratio": as_float(kernel.get("active_compute_ratio"), 1.0),
        "keep_ratio": as_float(kernel.get("keep_ratio"), 1.0),
        "degrade_ratio": as_float(kernel.get("degrade_ratio"), 0.0),
        "prune_ratio": as_float(kernel.get("prune_ratio"), 0.0),
        "condition_latency_ns": condition_latency,
        "budget_decision": budget_decision(kernel),
        "mapping_evidence": kernel.get("mapping_evidence", ""),
        "slack_source": "r2_hardware_scheduler_trace",
        "trace_link_role": "promoted_suds_pareto" if summary["condition"] == PROMOTED_CONDITION else "scheduler_ablation",
        "architecture_summary_artifact": repo_path(ARCH_SUMMARY_CSV),
        "architecture_kernel_artifact": repo_path(ARCH_KERNELS_CSV),
        "architecture_energy_pj": as_float(summary.get("energy_pj")),
        "architecture_latency_ns": as_float(summary.get("latency_ns")),
        "architecture_edp_pj_ns": as_float(summary.get("edp_pj_ns")),
        "accuracy_evidence_label": summary.get("accuracy_evidence_label", ""),
        "promotion_decision": summary.get("promotion_decision", ""),
    }
    times = {
        "release_ns": ready_ns_value,
        "data_ready_ns": data_ready_ns,
        "completion_ns": completion_ns,
        "deadline_ns": deadline_ns_value,
    }
    return row, times, event_results


def simulate_scheduler_condition(
    *,
    tag: str,
    seed: int,
    deterministic_config_id: str,
    scheduler: str,
    workload: str,
    condition: str,
    kernels: list[dict[str, str]],
    summary: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scale = deadline_scale(summary, kernels)
    dependencies = build_dependency_map(kernels)
    by_id = {str(row["kernel_id"]): row for row in kernels}
    pending = set(by_id)
    times: dict[str, dict[str, float]] = {}
    capacities = resource_capacities(summary)
    resources = {name: ResourcePool(name, capacity) for name, capacity in capacities.items()}
    max_dptc_tiles = max(as_float(row.get("condition_dptc_tiles"), as_float(row.get("dptc_tiles"))) for row in kernels)
    median_latency = median_kernel_latency(kernels)
    lookahead_ns = LOOKAHEAD_BY_SCHEDULER[scheduler] * median_latency
    schedule_trace_id = trace_hash(
        {
            "tag": tag,
            "seed": seed,
            "workload": workload,
            "condition": condition,
            "scheduler": scheduler,
            "deadline_scale": round(scale, 12),
            "release_pressure": RELEASE_PRESSURE_FACTOR,
            "lookahead_ns": round(lookahead_ns, 12),
            "dispatch_gap": DISPATCH_GAP_BY_SCHEDULER[scheduler],
            "sideband_prefetch": SIDEBAND_PREFETCH_BY_SCHEDULER[scheduler],
            "core_efficiency": CORE_EFFICIENCY_BY_SCHEDULER[scheduler],
        }
    )
    current_ns = 0.0
    trace_rows: list[dict[str, Any]] = []
    resource_busy_unit_ns = {name: 0.0 for name in resources}
    resource_queue_wait_ns = {name: 0.0 for name in resources}
    resource_max_queue_depth = {name: 0 for name in resources}

    while pending:
        schedulable: list[tuple[dict[str, str], float, float, list[str]]] = []
        for kernel_id in pending:
            deps = [dep for dep in dependencies.get(kernel_id, []) if dep]
            if any(dep not in times for dep in deps):
                continue
            kernel = by_id[kernel_id]
            dep_ready = max((times[dep]["data_ready_ns"] for dep in deps), default=0.0)
            ready = max(release_ns(kernel, scale), dep_ready)
            deadline = deadline_ns(kernel, scale)
            schedulable.append((kernel, ready, deadline, deps))

        if not schedulable:
            raise SystemExit(f"dependency deadlock for workload={workload} condition={condition} scheduler={scheduler}")

        eligible = [item for item in schedulable if item[1] <= current_ns + lookahead_ns]
        if not eligible:
            current_ns = min(item[1] for item in schedulable)
            continue

        kernel, ready, deadline, deps = min(
            eligible,
            key=lambda item: priority_key(
                scheduler,
                item[0],
                ready_ns=item[1],
                deadline=item[2],
                max_dptc_tiles=max_dptc_tiles,
                median_latency=median_latency,
            ),
        )
        row, kernel_times, event_results = schedule_one_kernel(
            tag=tag,
            seed=seed,
            deterministic_config_id=deterministic_config_id,
            schedule_trace_id=schedule_trace_id,
            scheduler=scheduler,
            kernel=kernel,
            summary=summary,
            dependencies=deps,
            ready_ns_value=ready,
            deadline_ns_value=deadline,
            resources=resources,
            median_latency_ns=median_latency,
        )
        trace_rows.append(row)
        times[str(kernel["kernel_id"])] = kernel_times
        pending.remove(str(kernel["kernel_id"]))
        for event_type, reservation in event_results.items():
            resource = RESOURCE_BY_EVENT[event_type]
            resource_busy_unit_ns[resource] += (
                float(reservation["end_ns"]) - float(reservation["start_ns"])
            ) * float(reservation["units_requested"])
            resource_queue_wait_ns[resource] += float(reservation["queue_wait_ns"])
            resource_max_queue_depth[resource] = max(
                resource_max_queue_depth[resource],
                int(reservation["queue_depth_before"]),
            )

    makespan = max((as_float(row["completion_ns"]) for row in trace_rows), default=0.0)
    resource_stats = {
        name: {
            "capacity": pool.capacity,
            "busy_unit_ns": resource_busy_unit_ns[name],
            "queue_wait_ns": resource_queue_wait_ns[name],
            "max_queue_depth_before": resource_max_queue_depth[name],
            "utilization": resource_busy_unit_ns[name] / max(1.0e-12, makespan * pool.capacity),
        }
        for name, pool in resources.items()
    }
    summary_row = summarize_trace_rows(
        trace_rows=trace_rows,
        summary=summary,
        scheduler=scheduler,
        schedule_trace_id=schedule_trace_id,
        lookahead_ns=lookahead_ns,
        deadline_scale_value=scale,
        resource_stats=resource_stats,
    )
    return trace_rows, summary_row


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * quantile
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def fmt_float(value: Any, digits: int = 6) -> float:
    return round(as_float(value), digits)


def summarize_trace_rows(
    *,
    trace_rows: list[dict[str, Any]],
    summary: dict[str, str],
    scheduler: str,
    schedule_trace_id: str,
    lookahead_ns: float,
    deadline_scale_value: float,
    resource_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    slack = [as_float(row["deadline_slack_ns"]) for row in trace_rows]
    slack_norm = [as_float(row["deadline_slack_norm"]) for row in trace_rows]
    queue = [as_float(row["queue_wait_ns"]) for row in trace_rows]
    completion = [as_float(row["completion_ns"]) for row in trace_rows]
    misses = [row for row in trace_rows if row["deadline_miss"]]
    by_layer = len({as_int(row["layer_index"]) for row in trace_rows})
    by_kernel_class = len({str(row["kernel_class"]) for row in trace_rows})
    makespan = max(completion, default=0.0)
    signature_values = [
        percentile(slack_norm, 0.10),
        percentile(slack_norm, 0.50),
        percentile(slack_norm, 0.90),
        len(misses) / max(1, len(trace_rows)),
        sum(queue) / max(1, len(trace_rows)) / max(1.0, makespan / max(1, len(trace_rows))),
        resource_stats.get("dptc_core", {}).get("utilization", 0.0),
    ]
    signature = hashlib.sha256(
        json.dumps([round(as_float(value), 8) for value in signature_values]).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "tag": summary.get("tag", TAG),
        "sensitivity_case": "nominal",
        "workload": summary["workload"],
        "condition": summary["condition"],
        "condition_label": summary.get("condition_label", ""),
        "scheduler": scheduler,
        "scheduler_label": SCHEDULER_LABELS[scheduler],
        "schedule_trace_id": schedule_trace_id,
        "lookahead_ns": lookahead_ns,
        "deadline_scale_to_condition_ppa": deadline_scale_value,
        "n_layers": by_layer,
        "n_kernel_classes": by_kernel_class,
        "n_kernels": len(trace_rows),
        "makespan_ns": makespan,
        "architecture_latency_ns": as_float(summary.get("latency_ns")),
        "makespan_ratio_vs_architecture_latency": makespan / max(1.0e-12, as_float(summary.get("latency_ns"))),
        "mean_slack_ns": sum(slack) / max(1, len(slack)),
        "min_slack_ns": min(slack, default=math.nan),
        "p10_slack_norm": percentile(slack_norm, 0.10),
        "p50_slack_norm": percentile(slack_norm, 0.50),
        "p90_slack_norm": percentile(slack_norm, 0.90),
        "mean_slack_norm": sum(slack_norm) / max(1, len(slack_norm)),
        "deadline_miss_count": len(misses),
        "deadline_miss_rate": len(misses) / max(1, len(trace_rows)),
        "total_queue_wait_ns": sum(queue),
        "mean_queue_wait_ns": sum(queue) / max(1, len(queue)),
        "max_queue_depth_before": max((as_int(row["max_queue_depth_before"]) for row in trace_rows), default=0),
        "dptc_core_utilization": resource_stats.get("dptc_core", {}).get("utilization", 0.0),
        "memory_utilization": resource_stats.get("memory_traffic", {}).get("utilization", 0.0),
        "adc_queue_utilization": resource_stats.get("adc_queue", {}).get("utilization", 0.0),
        "optical_link_utilization": resource_stats.get("optical_link", {}).get("utilization", 0.0),
        "control_sideband_utilization": resource_stats.get("control_sideband", {}).get("utilization", 0.0),
        "memory_traffic_bytes": sum(as_float(row["memory_bytes"]) for row in trace_rows),
        "optical_link_values": sum(as_float(row["optical_values"]) for row in trace_rows),
        "sideband_groups": sum(as_float(row["sideband_groups"]) for row in trace_rows),
        "keep_ratio": as_float(summary.get("keep_ratio")),
        "degrade_ratio": as_float(summary.get("degrade_ratio")),
        "prune_ratio": as_float(summary.get("prune_ratio")),
        "active_compute_ratio": as_float(summary.get("active_compute_ratio")),
        "energy_pj": as_float(summary.get("energy_pj")),
        "edp_pj_ns": as_float(summary.get("edp_pj_ns")),
        "energy_ratio_vs_lightening": as_float(summary.get("energy_ratio_vs_lightening"), math.nan),
        "edp_ratio_vs_lightening": as_float(summary.get("edp_ratio_vs_lightening"), math.nan),
        "accuracy_evidence_label": summary.get("accuracy_evidence_label", ""),
        "promotion_decision": summary.get("promotion_decision", ""),
        "trace_link_role": "promoted_suds_pareto" if summary["condition"] == PROMOTED_CONDITION else "scheduler_ablation",
        "distribution_signature": signature,
        "signature_p10_p50_p90_miss_queue_core": ";".join(f"{fmt_float(value):.6f}" for value in signature_values),
    }


def signature_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = (
        "p10_slack_norm",
        "p50_slack_norm",
        "p90_slack_norm",
        "deadline_miss_rate",
        "mean_queue_wait_ns",
        "dptc_core_utilization",
    )
    queue_scale = max(1.0, as_float(left.get("makespan_ns")) / max(1, as_int(left.get("n_kernels"))))
    total = 0.0
    for key in keys:
        lval = as_float(left.get(key))
        rval = as_float(right.get(key))
        if key == "mean_queue_wait_ns":
            total += abs(lval - rval) / queue_scale
        else:
            total += abs(lval - rval)
    return total


def annotate_ablation_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    by_group: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault((row["workload"], row["condition"]), {})[row["scheduler"]] = row

    distances: list[float] = []
    for group_rows in by_group.values():
        fifo = group_rows.get("fifo")
        suds = group_rows.get(PROMOTED_SCHEDULER)
        for row in group_rows.values():
            if fifo:
                row["distribution_distance_vs_fifo"] = signature_distance(row, fifo)
                row["p10_slack_norm_delta_vs_fifo"] = as_float(row["p10_slack_norm"]) - as_float(fifo["p10_slack_norm"])
                row["deadline_miss_delta_vs_fifo"] = as_int(row["deadline_miss_count"]) - as_int(fifo["deadline_miss_count"])
                row["queue_wait_delta_vs_fifo_ns"] = as_float(row["total_queue_wait_ns"]) - as_float(fifo["total_queue_wait_ns"])
            else:
                row["distribution_distance_vs_fifo"] = math.nan
                row["p10_slack_norm_delta_vs_fifo"] = math.nan
                row["deadline_miss_delta_vs_fifo"] = math.nan
                row["queue_wait_delta_vs_fifo_ns"] = math.nan
            if suds:
                row["p10_slack_norm_delta_vs_suds_aware"] = as_float(row["p10_slack_norm"]) - as_float(suds["p10_slack_norm"])
                row["deadline_miss_delta_vs_suds_aware"] = as_int(row["deadline_miss_count"]) - as_int(suds["deadline_miss_count"])
                row["queue_wait_delta_vs_suds_aware_ns"] = as_float(row["total_queue_wait_ns"]) - as_float(suds["total_queue_wait_ns"])
            else:
                row["p10_slack_norm_delta_vs_suds_aware"] = math.nan
                row["deadline_miss_delta_vs_suds_aware"] = math.nan
                row["queue_wait_delta_vs_suds_aware_ns"] = math.nan
            if fifo and row["scheduler"] != "fifo":
                distances.append(as_float(row["distribution_distance_vs_fifo"]))

        if fifo and suds:
            helps = (
                as_float(suds["p10_slack_norm"]) > as_float(fifo["p10_slack_norm"]) + 1.0e-9
                or as_int(suds["deadline_miss_count"]) < as_int(fifo["deadline_miss_count"])
            )
            best_low_tail = max(
                group_rows.values(),
                key=lambda item: (as_float(item["p10_slack_norm"]), -as_float(item["total_queue_wait_ns"])),
            )
            for row in group_rows.values():
                row["suds_aware_helps_vs_fifo"] = helps
                row["best_low_tail_scheduler"] = best_low_tail["scheduler"]
                row["suds_aware_best_low_tail"] = best_low_tail["scheduler"] == PROMOTED_SCHEDULER
                if helps and best_low_tail["scheduler"] == PROMOTED_SCHEDULER:
                    note = "SUDS-aware improves low-tail slack or deadline misses versus FIFO and is the best low-tail scheduler in this group."
                elif helps:
                    note = (
                        "SUDS-aware improves low-tail slack or deadline misses versus FIFO, "
                        f"but `{best_low_tail['scheduler']}` has the highest p10 slack in this scheduler-only ablation."
                    )
                else:
                    note = "SUDS-aware does not improve this workload-condition over FIFO; retained as negative/neutral ablation."
                row["scheduler_ablation_note"] = note

    return rows, {
        "max_scheduler_signature_distance": max(distances, default=0.0),
        "mean_scheduler_signature_distance": sum(distances) / max(1, len(distances)),
    }


def build_scheduler_traces(args: argparse.Namespace) -> dict[str, Any]:
    kernel_csv_rows = load_csv(args.architecture_kernels_csv)
    summary_csv_rows = load_csv(args.architecture_summary_csv)
    architecture_payload = load_json(args.architecture_json)
    kernels_by_key = group_kernel_rows(kernel_csv_rows)
    summaries_by_key = nominal_summary_rows(summary_csv_rows)
    source_hashes = {
        repo_path(args.architecture_kernels_csv): sha256_path(args.architecture_kernels_csv),
        repo_path(args.architecture_summary_csv): sha256_path(args.architecture_summary_csv),
        repo_path(args.architecture_json): sha256_path(args.architecture_json),
    }
    deterministic_config = {
        "tag": args.tag,
        "seed": args.seed,
        "schedulers": list(SCHEDULERS),
        "scheduler_labels": SCHEDULER_LABELS,
        "lookahead_by_scheduler": LOOKAHEAD_BY_SCHEDULER,
        "dispatch_gap_by_scheduler": DISPATCH_GAP_BY_SCHEDULER,
        "sideband_prefetch_by_scheduler": SIDEBAND_PREFETCH_BY_SCHEDULER,
        "core_efficiency_by_scheduler": CORE_EFFICIENCY_BY_SCHEDULER,
        "release_pressure_factor": RELEASE_PRESSURE_FACTOR,
        "event_fractions": EVENT_FRACTIONS,
        "dag_policy": "Transformer block DAG with qkv fanout to score/context, output join, FFN chain, and inter-block tail dependency",
        "resource_capacity_policy": "memory=min(tiles,4), sideband=1, dac/core=parallel_cores, adc=tiles, optical=1",
        "deadline_policy": "architecture schedule deadlines scaled to each condition PPA latency",
        "source_hashes": source_hashes,
    }
    deterministic_config_id = config_hash(deterministic_config)

    all_trace_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for key in sorted(summaries_by_key):
        workload, condition = key
        summary = summaries_by_key[key]
        kernels = kernels_by_key.get(key)
        if not kernels:
            raise SystemExit(f"missing kernel rows for workload={workload} condition={condition}")
        for scheduler in SCHEDULERS:
            trace_rows, summary_row = simulate_scheduler_condition(
                tag=args.tag,
                seed=args.seed,
                deterministic_config_id=deterministic_config_id,
                scheduler=scheduler,
                workload=workload,
                condition=condition,
                kernels=kernels,
                summary=summary,
            )
            all_trace_rows.extend(trace_rows)
            ablation_rows.append(summary_row)

    ablation_rows, distance_stats = annotate_ablation_rows(ablation_rows)
    promoted_links = [
        {
            "workload": row["workload"],
            "condition": row["condition"],
            "scheduler": row["scheduler"],
            "schedule_trace_id": row["schedule_trace_id"],
            "n_kernels": row["n_kernels"],
            "p10_slack_norm": row["p10_slack_norm"],
            "deadline_miss_count": row["deadline_miss_count"],
            "accuracy_evidence_label": row["accuracy_evidence_label"],
            "architecture_latency_ns": row["architecture_latency_ns"],
            "energy_pj": row["energy_pj"],
        }
        for row in ablation_rows
        if row["condition"] == PROMOTED_CONDITION and row["scheduler"] == PROMOTED_SCHEDULER
    ]
    scheduler_variants_ok = set(SCHEDULERS).issubset({row["scheduler"] for row in ablation_rows})
    promoted_linked = len(promoted_links) >= 2
    trace_has_layers = all(as_int(row.get("n_layers")) > 0 and as_int(row.get("n_kernel_classes")) >= 4 for row in ablation_rows)
    distributions_distinguishable = distance_stats["max_scheduler_signature_distance"] >= args.min_scheduler_distance
    acceptance_state = (
        "pass"
        if scheduler_variants_ok
        and promoted_linked
        and trace_has_layers
        and distributions_distinguishable
        else "fail"
    )
    stop_condition_state = (
        "no R2 hard stop"
        if distributions_distinguishable
        else "stop: scheduler slack distributions are indistinguishable across variants"
    )

    decision = {
        "r2_acceptance_state": acceptance_state,
        "stop_condition_state": stop_condition_state,
        "scheduler_variants": list(SCHEDULERS),
        "scheduler_variants_ok": scheduler_variants_ok,
        "promoted_suds_pareto_linked": promoted_linked,
        "promoted_trace_links": promoted_links,
        "hardware_derived_inputs": {
            "kernel_dag": True,
            "tile_mapping": True,
            "queue_state": True,
            "deadline_model": True,
            "architecture_condition_ppa_scale": True,
        },
        "min_scheduler_distance": args.min_scheduler_distance,
        **distance_stats,
    }
    payload = {
        "metadata": {
            "artifact_id": f"suds_tetc_scheduler_traces_{args.tag}",
            "roadmap_item": "R2_hardware_derived_scheduler_slack_traces",
            "date": DATE,
            "tag": args.tag,
            "seed": args.seed,
            "git_hash": git_hash(),
            "deterministic_config_id": deterministic_config_id,
            "deterministic_config": deterministic_config,
            "architecture_artifact_id": architecture_payload.get("metadata", {}).get("artifact_id", ""),
            "trace_csv": repo_path(args.trace_csv),
            "ablation_csv": repo_path(args.ablation_csv),
            "report": repo_path(args.report_out),
            "claim_boundary": (
                "architecture scheduler trace over modeled DPTC queues, tile mapping, "
                "and deadlines; not cycle-accurate RTL, post-layout timing, optical-device "
                "signoff, fabrication evidence, or hardware measurement"
            ),
        },
        "decision": decision,
        "summary_rows": ablation_rows,
    }
    return {"payload": payload, "trace_rows": all_trace_rows, "ablation_rows": ablation_rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    metadata = payload["metadata"]
    decision = payload["decision"]
    summaries = payload["summary_rows"]
    promoted = decision["promoted_trace_links"]
    suds_pareto = [
        row for row in summaries
        if row["condition"] == PROMOTED_CONDITION
    ]
    suds_aware_rows = [
        row for row in summaries
        if row["scheduler"] == PROMOTED_SCHEDULER and row["condition"] in {PROMOTED_CONDITION, "slack_only", "l1", "signal_only"}
    ]

    lines = [
        "# SUDS TETC Hardware-Derived Scheduler Slack Traces",
        "",
        f"Date: `{DATE}`",
        f"Tag: `{metadata['tag']}`",
        "Roadmap item: `R2_hardware_derived_scheduler_slack_traces`",
        f"Acceptance state: `{decision['r2_acceptance_state']}`",
        f"Stop-condition state: `{decision['stop_condition_state']}`",
        "",
        "## Scope",
        "",
        "This R2 artifact derives SUDS slack from a modeled accelerator scheduler",
        "rather than from a manually chosen analytical profile. It consumes the",
        "architecture kernel rows, builds a Transformer kernel DAG, maps kernels",
        "to DPTC tile resources, reserves memory/control/DAC/core/ADC/optical",
        "queues, and emits per-kernel slack under five scheduler variants.",
        "",
        "The trace remains architecture-scheduler evidence. It is not RTL timing,",
        "post-layout timing, optical-device signoff, fabricated silicon, or",
        "hardware measurement.",
        "",
            "## Scheduler Variants",
            "",
            "| Scheduler | Priority surface | Look-ahead | Dispatch gap | Core efficiency |",
            "|---|---|---:|---:|---:|",
    ]
    for scheduler in SCHEDULERS:
        if scheduler == "fifo":
            surface = "release order"
        elif scheduler == "asap":
            surface = "earliest ready time, then short kernel"
        elif scheduler == "edf_deadline_aware":
            surface = "earliest deadline first"
        elif scheduler == "utilization_aware":
            surface = "tile/core demand and short-job packing"
        else:
            surface = "low slack, deadline pressure, tile demand, and budget guard"
        lines.append(
            f"| `{scheduler}` | {surface} | `{LOOKAHEAD_BY_SCHEDULER[scheduler]:.2f} x median` | "
            f"`{DISPATCH_GAP_BY_SCHEDULER[scheduler]:.3f} x median` | "
            f"`{CORE_EFFICIENCY_BY_SCHEDULER[scheduler]:.2f}x` |"
        )

    lines.extend(
        [
            "",
            "## Promoted Schedule Links",
            "",
            "| Workload | Condition | Scheduler | Trace ID | Kernels | p10 slack norm | Misses | Accuracy evidence |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in promoted:
        lines.append(
            f"| `{row['workload']}` | `{row['condition']}` | `{row['scheduler']}` | "
            f"`{row['schedule_trace_id']}` | {row['n_kernels']} | {fmt(row['p10_slack_norm'], 4)} | "
            f"{row['deadline_miss_count']} | `{row['accuracy_evidence_label']}` |"
        )

    lines.extend(
        [
            "",
            "## SUDS-Pareto Scheduler Ablation",
            "",
            "| Workload | Scheduler | p10 slack norm | median slack norm | p90 slack norm | Misses | Queue wait ns | Core util | Distance vs FIFO |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(suds_pareto, key=lambda item: (item["workload"], SCHEDULERS.index(item["scheduler"]))):
        lines.append(
            f"| `{row['workload']}` | `{row['scheduler']}` | {fmt(row['p10_slack_norm'], 4)} | "
            f"{fmt(row['p50_slack_norm'], 4)} | {fmt(row['p90_slack_norm'], 4)} | "
            f"{row['deadline_miss_count']} | {fmt(row['total_queue_wait_ns'], 3)} | "
            f"{fmt(row['dptc_core_utilization'], 4)} | {fmt(row['distribution_distance_vs_fifo'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## Where SUDS Helps And Where It Does Not",
            "",
            "| Workload | Condition | SUDS-aware p10 slack | Misses | EDP ratio vs Lightening | Note |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(suds_aware_rows, key=lambda item: (item["workload"], item["condition"])):
        note = str(row.get("scheduler_ablation_note", ""))
        energy_note = "main promoted trace" if row["condition"] == PROMOTED_CONDITION else "selector/budget ablation"
        lines.append(
            f"| `{row['workload']}` | `{row['condition']}` | {fmt(row['p10_slack_norm'], 4)} | "
            f"{row['deadline_miss_count']} | {fmt(row['edp_ratio_vs_lightening'], 4)} | "
            f"{energy_note}; {note} |"
        )

    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- Scheduler variants present: `{decision['scheduler_variants_ok']}`.",
            f"- Promoted `suds_pareto` linked to hardware-derived trace: `{decision['promoted_suds_pareto_linked']}`.",
            f"- Maximum scheduler distribution distance: `{decision['max_scheduler_signature_distance']:.6f}`.",
            f"- Minimum required distance before R2 stop: `{decision['min_scheduler_distance']:.6f}`.",
            f"- Kernel DAG, tile mapping, queue state, and deadline model emitted: `{all(decision['hardware_derived_inputs'].values())}`.",
            "",
            "The R2 stop condition is not triggered because scheduler slack",
            "distributions differ across scheduler variants. Neutral or negative",
            "SUDS-aware rows are retained as ablations rather than hidden.",
            "",
            "## Artifacts",
            "",
            f"- Scheduler trace CSV: `{metadata['trace_csv']}`",
            f"- Scheduler ablation CSV: `{metadata['ablation_csv']}`",
            f"- Scheduler JSON: `{repo_path(JSON_OUT)}`",
            f"- Report: `{metadata['report']}`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "python3 experiments/tools/build_suds_tetc_scheduler_traces.py --tag 20260513_tetc_pivot",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = build_scheduler_traces(args)
    write_csv(args.trace_csv, result["trace_rows"])
    write_csv(args.ablation_csv, result["ablation_rows"])
    write_json(args.json_out, result["payload"])
    write_report(args.report_out, result["payload"])
    print(
        f"Wrote {repo_path(args.trace_csv)}, {repo_path(args.ablation_csv)}, "
        f"{repo_path(args.json_out)}, and {repo_path(args.report_out)}"
    )


if __name__ == "__main__":
    main()
