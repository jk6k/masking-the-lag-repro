#!/usr/bin/env python3
"""Build the R10 failure-case and uncertainty artifact for the SUDS TETC route.

R10 is a maturity gate, not a headline-number generator. It records where the
SUDS budget interface should not be promoted and checks that the current
target-regime SUDS Pareto conclusion survives bounded uncertainty over PPA
parameters, thresholds, and measured accuracy seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_suds_tetc_system_sensitivity import (
    DEFAULT_SCENARIO,
    REPO_ROOT,
    TAG,
    estimate,
)


DATE = "2026-05-14"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

ARCH_SUMMARY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_summary.csv"
SCHEDULER_TRACE_CSV = REPORT_DATA / f"suds_tetc_scheduler_traces_{TAG}.csv"
SYSTEM_SENSITIVITY_CSV = REPORT_DATA / f"suds_tetc_system_sensitivity_{TAG}.csv"
SYSTEM_SENSITIVITY_JSON = REPORT_DATA / f"suds_tetc_system_sensitivity_{TAG}.json"
WORKLOAD_EXPANSION_CSV = REPORT_DATA / f"suds_tetc_workload_expansion_{TAG}.csv"
WORKLOAD_EXPANSION_JSON = REPORT_DATA / f"suds_tetc_workload_expansion_{TAG}.json"
END_TO_END_ACCURACY_JSON = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{TAG}.json"
PARETO_DESIGN_SPACE_JSON = REPORT_DATA / f"suds_tetc_pareto_design_space_{TAG}.json"
CONSERVATIVE_PARETO_CSV = REPORT_DATA / f"suds_tetc_conservative_pareto_{TAG}.csv"
GLUE_JSON = REPORT_DATA / "suds_glue_measured_validation_20260511_p2p3_quality.json"

CSV_OUT = REPORT_DATA / f"suds_tetc_failure_suite_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_uncertainty_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_failure_uncertainty.md"

PROMOTED_CONDITION = "suds_pareto"
REFERENCE_CONDITION = "lightening_dptc"
WORKLOADS_REQUIRED = ("bert_base_glue_seq128", "mobilevit_s_transformer_blocks_256")
FAILURE_FAMILIES_REQUIRED = (
    "low_slack",
    "high_memory_pressure",
    "small_batch",
    "long_sequence",
    "activation_sensitive_layers",
    "signal_only_dominant",
)
TARGET_DRAWS = 4096
BOUNDARY_DRAWS = 2048
BOOTSTRAP_DRAWS = 4096
RNG_SEED = 20260514
ACCURACY_LOSS_BUDGET_PP = 1.0
MATERIAL_EDP_IMPROVEMENT_PCT = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--architecture-summary-csv", type=Path, default=ARCH_SUMMARY_CSV)
    parser.add_argument("--scheduler-trace-csv", type=Path, default=SCHEDULER_TRACE_CSV)
    parser.add_argument("--system-sensitivity-csv", type=Path, default=SYSTEM_SENSITIVITY_CSV)
    parser.add_argument("--system-sensitivity-json", type=Path, default=SYSTEM_SENSITIVITY_JSON)
    parser.add_argument("--workload-expansion-csv", type=Path, default=WORKLOAD_EXPANSION_CSV)
    parser.add_argument("--workload-expansion-json", type=Path, default=WORKLOAD_EXPANSION_JSON)
    parser.add_argument("--end-to-end-accuracy-json", type=Path, default=END_TO_END_ACCURACY_JSON)
    parser.add_argument("--pareto-design-space-json", type=Path, default=PARETO_DESIGN_SPACE_JSON)
    parser.add_argument("--conservative-pareto-csv", type=Path, default=CONSERVATIVE_PARETO_CSV)
    parser.add_argument("--glue-json", type=Path, default=GLUE_JSON)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    return parser.parse_args()


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing required CSV artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def percentile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if not math.isnan(value))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def nominal_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("sensitivity_case") != "nominal":
            continue
        workload = row.get("workload", "")
        condition = row.get("condition", "")
        if workload and condition:
            out[(workload, condition)] = row
    return out


def failure_row(
    *,
    case_id: str,
    family: str,
    workload: str,
    condition: str,
    reference_condition: str,
    source_artifact: str,
    source_row_id: str,
    evidence_type: str,
    metric: str,
    metric_value: float,
    energy_improvement_pct: float,
    edp_improvement_pct: float,
    edp_ratio: float,
    accuracy_delta_pp: float,
    accuracy_evidence_label: str,
    should_not_use_suds: bool,
    claim_effect: str,
    failure_mode: str,
    recommended_action: str,
    claim_boundary: str,
) -> dict[str, Any]:
    return {
        "tag": TAG,
        "date": DATE,
        "roadmap_item": "R10_failure_cases_and_uncertainty",
        "case_id": case_id,
        "case_family": family,
        "workload": workload,
        "condition": condition,
        "reference_condition": reference_condition,
        "source_artifact": source_artifact,
        "source_row_id": source_row_id,
        "evidence_type": evidence_type,
        "metric": metric,
        "metric_value": metric_value,
        "energy_improvement_vs_reference_pct": energy_improvement_pct,
        "edp_improvement_vs_reference_pct": edp_improvement_pct,
        "edp_ratio_vs_reference": edp_ratio,
        "accuracy_delta_pp": accuracy_delta_pp,
        "accuracy_evidence_label": accuracy_evidence_label,
        "should_not_use_suds": should_not_use_suds,
        "claim_effect": claim_effect,
        "failure_mode": failure_mode,
        "recommended_action": recommended_action,
        "claim_boundary": claim_boundary,
    }


def forced_keep_counterexamples(
    index: dict[tuple[str, str], dict[str, str]],
    scheduler_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    slack_by_workload: dict[str, list[float]] = defaultdict(list)
    for row in scheduler_rows:
        if row.get("condition") == PROMOTED_CONDITION and row.get("scheduler") == "suds_aware":
            slack_by_workload[row.get("workload", "")].append(as_float(row.get("deadline_slack_norm")))

    rows: list[dict[str, Any]] = []
    for workload in WORKLOADS_REQUIRED:
        reference = index[(workload, REFERENCE_CONDITION)]
        promoted = index[(workload, PROMOTED_CONDITION)]
        ref_energy = as_float(reference.get("energy_pj"))
        ref_latency = as_float(reference.get("latency_ns"))
        control = as_float(promoted.get("control_energy_pj"), 0.0)
        forced_energy = ref_energy + control
        forced_latency = ref_latency * (1.0 + min(0.002, control / max(ref_energy, 1.0)))
        edp_ratio = (forced_energy * forced_latency) / max(1e-12, ref_energy * ref_latency)
        slack_values = sorted(slack_by_workload.get(workload, []))
        p05 = percentile(slack_values, 0.05)
        rows.append(
            failure_row(
                case_id=f"r10_low_slack_forced_keep_{workload}",
                family="low_slack",
                workload=workload,
                condition="suds_forced_keep_all",
                reference_condition=REFERENCE_CONDITION,
                source_artifact=repo_path(SCHEDULER_TRACE_CSV),
                source_row_id=f"suds_aware_slack_p05={fmt(p05, 4)}",
                evidence_type="constructed_counterexample_from_scheduler_trace",
                metric="deadline_slack_norm_p05",
                metric_value=p05,
                energy_improvement_pct=(1.0 - forced_energy / ref_energy) * 100.0,
                edp_improvement_pct=(1.0 - edp_ratio) * 100.0,
                edp_ratio=edp_ratio,
                accuracy_delta_pp=0.0,
                accuracy_evidence_label=promoted.get("accuracy_evidence_label", ""),
                should_not_use_suds=True,
                claim_effect="SUDS degenerates to keep-all when slack headroom is unavailable.",
                failure_mode="low slack leaves no safe budget to degrade or prune columns",
                recommended_action="Disable SUDS or run in keep-all mode for near-deadline kernels.",
                claim_boundary=(
                    "Counterexample uses scheduler-trace slack to construct the no-headroom case; "
                    "it is a policy boundary, not a new measured accuracy run."
                ),
            )
        )
    return rows


def high_memory_counterexamples(index: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload in WORKLOADS_REQUIRED:
        reference = index[(workload, REFERENCE_CONDITION)]
        promoted = index[(workload, PROMOTED_CONDITION)]
        ref_energy = as_float(reference.get("energy_pj"))
        suds_energy = as_float(promoted.get("energy_pj"))
        ref_memory = as_float(reference.get("memory_energy_pj"))
        suds_memory = as_float(promoted.get("memory_energy_pj"))
        ref_latency = as_float(reference.get("latency_ns"))
        suds_latency = as_float(promoted.get("latency_ns"))
        memory_scale = 256.0
        ref_total = ref_energy + ref_memory * (memory_scale - 1.0)
        suds_total = suds_energy + suds_memory * (memory_scale - 1.0)
        edp_ratio = (suds_total * suds_latency) / max(1e-12, ref_total * ref_latency)
        should_not_use = edp_ratio >= 1.0 or (1.0 - edp_ratio) * 100.0 < MATERIAL_EDP_IMPROVEMENT_PCT
        rows.append(
            failure_row(
                case_id=f"r10_high_memory_pressure_{workload}",
                family="high_memory_pressure",
                workload=workload,
                condition=PROMOTED_CONDITION,
                reference_condition=REFERENCE_CONDITION,
                source_artifact=repo_path(ARCH_SUMMARY_CSV),
                source_row_id="memory_energy_scaled_256x_boundary",
                evidence_type="constructed_parametric_boundary",
                metric="memory_energy_scale",
                metric_value=memory_scale,
                energy_improvement_pct=(1.0 - suds_total / ref_total) * 100.0,
                edp_improvement_pct=(1.0 - edp_ratio) * 100.0,
                edp_ratio=edp_ratio,
                accuracy_delta_pp=as_float(promoted.get("delta_accuracy")),
                accuracy_evidence_label=promoted.get("accuracy_evidence_label", ""),
                should_not_use_suds=should_not_use,
                claim_effect="Extreme memory dominance can erase or thin the promoted EDP margin.",
                failure_mode="memory movement dominates the conversion savings",
                recommended_action="Do not promote SUDS as a memory-system optimization; report this as a boundary regime.",
                claim_boundary=(
                    "This is an intentionally severe memory-dominance counterexample from modeled PPA terms, "
                    "not a hardware measurement."
                ),
            )
        )
    return rows


def workload_boundary_rows(workload_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    small_batch_candidates = [
        row for row in workload_rows
        if row.get("condition") == PROMOTED_CONDITION
        and row.get("model") == "deit_tiny_patch16_224"
        and row.get("batch_size") == "1"
    ]
    if small_batch_candidates:
        row = small_batch_candidates[0]
        rows.append(
            failure_row(
                case_id="r10_small_batch_deit_tiny_batch1_accuracy_blocker",
                family="small_batch",
                workload=row.get("workload", ""),
                condition=PROMOTED_CONDITION,
                reference_condition=REFERENCE_CONDITION,
                source_artifact=repo_path(WORKLOAD_EXPANSION_CSV),
                source_row_id=row.get("workload", ""),
                evidence_type="simulator_only_unmeasured_accuracy_boundary",
                metric="batch_size",
                metric_value=as_float(row.get("batch_size")),
                energy_improvement_pct=as_float(row.get("energy_improvement_vs_lightening_pct")),
                edp_improvement_pct=as_float(row.get("edp_improvement_vs_lightening_pct")),
                edp_ratio=as_float(row.get("edp_ratio_vs_lightening")),
                accuracy_delta_pp=math.nan,
                accuracy_evidence_label=row.get("accuracy_evidence_label", ""),
                should_not_use_suds=True,
                claim_effect="Small-batch architecture-only rows cannot carry measured accuracy claims.",
                failure_mode=row.get("setup_blocker", "") or "governed measured accuracy is unavailable",
                recommended_action="Keep the row as simulator-only boundary evidence until an MPS accuracy run exists.",
                claim_boundary=row.get("claim_boundary", ""),
            )
        )

    long_candidates = [
        row for row in workload_rows
        if row.get("condition") == PROMOTED_CONDITION
        and row.get("model") == "bert_base"
        and row.get("sequence_length") == "512"
        and row.get("batch_size") == "1"
    ]
    if long_candidates:
        row = long_candidates[0]
        rows.append(
            failure_row(
                case_id="r10_long_sequence_bert512_accuracy_boundary",
                family="long_sequence",
                workload=row.get("workload", ""),
                condition=PROMOTED_CONDITION,
                reference_condition=REFERENCE_CONDITION,
                source_artifact=repo_path(WORKLOAD_EXPANSION_CSV),
                source_row_id=row.get("workload", ""),
                evidence_type="simulator_only_unmeasured_accuracy_boundary",
                metric="sequence_length",
                metric_value=as_float(row.get("sequence_length")),
                energy_improvement_pct=as_float(row.get("energy_improvement_vs_lightening_pct")),
                edp_improvement_pct=as_float(row.get("edp_improvement_vs_lightening_pct")),
                edp_ratio=as_float(row.get("edp_ratio_vs_lightening")),
                accuracy_delta_pp=math.nan,
                accuracy_evidence_label=row.get("accuracy_evidence_label", ""),
                should_not_use_suds=True,
                claim_effect="Long-sequence BERT remains an architecture trace without governed measured accuracy.",
                failure_mode=row.get("setup_blocker", "") or "no MPS accuracy row for this sequence length",
                recommended_action="Do not generalize measured GLUE accuracy to long-sequence rows.",
                claim_boundary=row.get("claim_boundary", ""),
            )
        )
    return rows


def activation_and_signal_rows(index: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    workload = "mobilevit_s_transformer_blocks_256"
    reference = index[(workload, REFERENCE_CONDITION)]
    promoted = index[(workload, PROMOTED_CONDITION)]
    for condition in ("suds_only", "slack_only", "l1"):
        row = index.get((workload, condition))
        if row is None:
            continue
        delta = as_float(row.get("delta_accuracy"))
        rows.append(
            failure_row(
                case_id=f"r10_activation_sensitive_{condition}_{workload}",
                family="activation_sensitive_layers",
                workload=workload,
                condition=condition,
                reference_condition=REFERENCE_CONDITION,
                source_artifact=repo_path(END_TO_END_ACCURACY_JSON),
                source_row_id=f"{condition}_delta={fmt(delta, 3)}pp",
                evidence_type="measured_mps_accuracy_boundary",
                metric="accuracy_delta_pp",
                metric_value=delta,
                energy_improvement_pct=as_float(row.get("energy_improvement_vs_lightening_pct")),
                edp_improvement_pct=as_float(row.get("edp_improvement_vs_lightening_pct")),
                edp_ratio=as_float(row.get("edp_ratio_vs_lightening")),
                accuracy_delta_pp=delta,
                accuracy_evidence_label=row.get("accuracy_evidence_label", ""),
                should_not_use_suds=delta < -ACCURACY_LOSS_BUDGET_PP,
                claim_effect="Aggressive pruning/degradation can violate the promoted accuracy budget.",
                failure_mode="activation-sensitive MobileViT-S rows lose more than 1 pp top-1",
                recommended_action="Use the conservative no-prune SUDS Pareto row or keep this as an ablation boundary.",
                claim_boundary=row.get("claim_boundary", ""),
            )
        )

    for condition in ("signal_only", "suds_signal"):
        row = index.get((workload, condition))
        if row is None:
            continue
        promoted_edp = as_float(promoted.get("edp_ratio_vs_lightening"))
        row_edp = as_float(row.get("edp_ratio_vs_lightening"))
        rows.append(
            failure_row(
                case_id=f"r10_signal_only_dominant_{condition}_{workload}",
                family="signal_only_dominant",
                workload=workload,
                condition=condition,
                reference_condition=PROMOTED_CONDITION,
                source_artifact=repo_path(ARCH_SUMMARY_CSV),
                source_row_id=f"{condition}_vs_suds_pareto",
                evidence_type="measured_accuracy_plus_modeled_ppa_boundary",
                metric="edp_delta_vs_suds_pareto_pct",
                metric_value=(promoted_edp - row_edp) * 100.0,
                energy_improvement_pct=(
                    1.0 - as_float(row.get("energy_pj")) / max(as_float(promoted.get("energy_pj")), 1e-12)
                ) * 100.0,
                edp_improvement_pct=(1.0 - row_edp / max(promoted_edp, 1e-12)) * 100.0,
                edp_ratio=row_edp / max(promoted_edp, 1e-12),
                accuracy_delta_pp=as_float(row.get("delta_accuracy")),
                accuracy_evidence_label=row.get("accuracy_evidence_label", ""),
                should_not_use_suds=True,
                claim_effect="A local signal proxy can be lower EDP under a looser accuracy boundary.",
                failure_mode="signal-only selector dominates raw PPA but not the promoted accuracy/fairness contract",
                recommended_action="Avoid universal dominance language; keep signal rows as boundary or ablation evidence.",
                claim_boundary=(
                    "Signal rows use measured MPS accuracy where available and modeled PPA; "
                    "they do not replace the same-scope promoted SUDS contract."
                ),
            )
        )
    return rows


def r6_not_beneficial_rows(system_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in system_rows:
        if row.get("benefit_preserved") != "False":
            continue
        out.append(
            failure_row(
                case_id=f"r10_conversion_boundary_{row.get('scenario_id')}_{row.get('workload')}",
                family="conversion_dominated_boundary",
                workload=row.get("workload", ""),
                condition=PROMOTED_CONDITION,
                reference_condition=REFERENCE_CONDITION,
                source_artifact=repo_path(SYSTEM_SENSITIVITY_CSV),
                source_row_id=row.get("scenario_id", ""),
                evidence_type="r6_parametric_boundary",
                metric=row.get("sweep_axis", ""),
                metric_value=as_float(row.get("sweep_value")),
                energy_improvement_pct=as_float(row.get("energy_improvement_vs_reference_pct")),
                edp_improvement_pct=as_float(row.get("edp_improvement_vs_reference_pct")),
                edp_ratio=as_float(row.get("edp_ratio_vs_reference")),
                accuracy_delta_pp=math.nan,
                accuracy_evidence_label=row.get("nominal_accuracy_evidence_label", ""),
                should_not_use_suds=True,
                claim_effect="Extreme conversion/link settings can erase MobileViT-S SUDS EDP advantage.",
                failure_mode="DAC/MZM or combined conversion stress crosses the zero-advantage boundary",
                recommended_action="Keep extreme conversion rows as boundary evidence and do not promote them as target regimes.",
                claim_boundary=row.get("claim_boundary", ""),
            )
        )
    return out


def build_failure_rows(
    *,
    arch_rows: list[dict[str, str]],
    scheduler_rows: list[dict[str, str]],
    system_rows: list[dict[str, str]],
    workload_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    index = nominal_index(arch_rows)
    rows: list[dict[str, Any]] = []
    rows.extend(forced_keep_counterexamples(index, scheduler_rows))
    rows.extend(high_memory_counterexamples(index))
    rows.extend(workload_boundary_rows(workload_rows))
    rows.extend(activation_and_signal_rows(index))
    rows.extend(r6_not_beneficial_rows(system_rows))
    rows.sort(key=lambda item: (str(item["case_family"]), str(item["workload"]), str(item["case_id"])))
    return rows


def target_scenario(rng: random.Random) -> dict[str, Any]:
    scenario = dict(DEFAULT_SCENARIO)
    scenario.update(
        {
            "memory_bandwidth_scale": rng.uniform(0.5, 2.0),
            "activation_reuse_scale": rng.uniform(0.65, 1.4),
            "adc_energy_scale": rng.uniform(0.75, 2.0),
            "dac_energy_scale": rng.uniform(0.85, 1.25),
            "laser_multiplier": rng.uniform(0.85, 1.15),
            "optical_link_loss_scale": rng.uniform(0.75, 1.25),
            "sideband_control_overhead_scale": rng.uniform(0.5, 3.0),
        }
    )
    return scenario


def boundary_scenario(rng: random.Random) -> dict[str, Any]:
    scenario = dict(DEFAULT_SCENARIO)
    scenario.update(
        {
            "memory_bandwidth_scale": rng.uniform(0.25, 1.0),
            "activation_reuse_scale": rng.uniform(0.35, 1.0),
            "adc_energy_scale": rng.uniform(1.0, 8.0),
            "dac_energy_scale": 2.0 ** rng.uniform(0.0, 7.0),
            "laser_multiplier": rng.uniform(1.0, 4.0),
            "optical_link_loss_scale": rng.uniform(1.0, 4.0),
            "sideband_control_overhead_scale": 10.0 ** rng.uniform(0.0, 2.0),
        }
    )
    return scenario


def threshold_factor(rng: random.Random) -> tuple[float, float, float, float]:
    tau_low = min(0.40, max(0.20, rng.gauss(0.30, 0.035)))
    tau_high = min(0.99, max(0.86, rng.gauss(0.95, 0.025)))
    if tau_high <= tau_low + 0.10:
        tau_high = min(0.99, tau_low + 0.10)
    width = tau_high - tau_low
    tightness = (0.65 - width) / 0.65
    energy = min(1.05, max(0.96, 1.0 + 0.025 * tightness + rng.gauss(0.0, 0.004)))
    latency = min(1.03, max(0.98, 1.0 + 0.010 * abs(tightness) + rng.gauss(0.0, 0.002)))
    return tau_low, tau_high, energy, latency


def monte_carlo(
    index: dict[tuple[str, str], dict[str, str]],
    *,
    scenario_builder: Any,
    draws: int,
    seed_offset: int,
) -> dict[str, Any]:
    rng = random.Random(RNG_SEED + seed_offset)
    values: dict[str, list[float]] = {workload: [] for workload in WORKLOADS_REQUIRED}
    energy_values: dict[str, list[float]] = {workload: [] for workload in WORKLOADS_REQUIRED}
    threshold_samples: list[dict[str, float]] = []
    for draw in range(draws):
        scenario = scenario_builder(rng)
        tau_low, tau_high, energy_factor, latency_factor = threshold_factor(rng)
        if draw < 16:
            threshold_samples.append(
                {
                    "tau_low": tau_low,
                    "tau_high": tau_high,
                    "suds_energy_factor": energy_factor,
                    "suds_latency_factor": latency_factor,
                }
            )
        for workload in WORKLOADS_REQUIRED:
            reference = index[(workload, REFERENCE_CONDITION)]
            promoted = index[(workload, PROMOTED_CONDITION)]
            ref_est = estimate(reference, scenario)
            suds_est = estimate(promoted, scenario)
            suds_energy = suds_est["estimated_energy_pj"] * energy_factor
            suds_latency = suds_est["estimated_latency_ns"] * latency_factor
            ref_energy = ref_est["estimated_energy_pj"]
            ref_latency = ref_est["estimated_latency_ns"]
            energy_improvement = (1.0 - suds_energy / ref_energy) * 100.0
            edp_improvement = (1.0 - (suds_energy * suds_latency) / (ref_energy * ref_latency)) * 100.0
            energy_values[workload].append(energy_improvement)
            values[workload].append(edp_improvement)

    by_workload: dict[str, Any] = {}
    for workload, sample in values.items():
        by_workload[workload] = {
            "draws": len(sample),
            "edp_improvement_mean_pct": sum(sample) / len(sample),
            "edp_improvement_ci95_pct": [percentile(sample, 0.025), percentile(sample, 0.975)],
            "edp_improvement_p05_pct": percentile(sample, 0.05),
            "edp_improvement_p50_pct": percentile(sample, 0.50),
            "edp_improvement_p95_pct": percentile(sample, 0.95),
            "energy_improvement_ci95_pct": [
                percentile(energy_values[workload], 0.025),
                percentile(energy_values[workload], 0.975),
            ],
            "probability_positive_edp_advantage": sum(1 for value in sample if value > 0.0) / len(sample),
            "probability_material_edp_advantage": (
                sum(1 for value in sample if value >= MATERIAL_EDP_IMPROVEMENT_PCT) / len(sample)
            ),
            "crosses_zero_advantage": percentile(sample, 0.025) <= 0.0,
        }
    return {
        "by_workload": by_workload,
        "minimum_ci95_lower_pct": min(item["edp_improvement_ci95_pct"][0] for item in by_workload.values()),
        "any_crosses_zero_advantage": any(item["crosses_zero_advantage"] for item in by_workload.values()),
        "threshold_sample_preview": threshold_samples,
    }


def bootstrap_ci(values: list[float], *, draws: int, seed_offset: int) -> dict[str, Any]:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return {
            "seed_values": [],
            "n_values": 0,
            "mean_delta_pp": math.nan,
            "ci95_delta_pp": [math.nan, math.nan],
            "worst_observed_delta_pp": math.nan,
            "stays_within_accuracy_budget": False,
        }
    rng = random.Random(RNG_SEED + seed_offset)
    means: list[float] = []
    for _ in range(draws):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        means.append(sum(sample) / len(sample))
    mean = sum(clean) / len(clean)
    return {
        "seed_values": clean,
        "n_values": len(clean),
        "mean_delta_pp": mean,
        "ci95_delta_pp": [percentile(means, 0.025), percentile(means, 0.975)],
        "worst_observed_delta_pp": min(clean),
        "stays_within_accuracy_budget": percentile(means, 0.025) >= -ACCURACY_LOSS_BUDGET_PP,
    }


def bert_seed_deltas(glue_json: dict[str, Any]) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in glue_json.get("per_seed", []):
        if row.get("condition") != "e2_l1":
            continue
        seed = row.get("seed")
        if not isinstance(seed, int):
            continue
        grouped[seed].append(as_float(row.get("delta_primary_metric")))
    return [sum(values) / len(values) for _, values in sorted(grouped.items()) if values]


def mobilevit_seed_deltas(rows: list[dict[str, str]]) -> list[float]:
    out: list[float] = []
    for row in rows:
        if row.get("row_type") == "per_seed" and row.get("condition") == "e9_suds_conservative":
            out.append(as_float(row.get("delta_top1")))
    return out


def accuracy_uncertainty(
    *,
    glue_json: dict[str, Any],
    conservative_rows: list[dict[str, str]],
) -> dict[str, Any]:
    by_workload = {
        "bert_base_glue_seq128": bootstrap_ci(bert_seed_deltas(glue_json), draws=BOOTSTRAP_DRAWS, seed_offset=101),
        "mobilevit_s_transformer_blocks_256": bootstrap_ci(
            mobilevit_seed_deltas(conservative_rows),
            draws=BOOTSTRAP_DRAWS,
            seed_offset=202,
        ),
    }
    return {
        "by_workload": by_workload,
        "all_promoted_seed_bootstrap_within_accuracy_budget": all(
            item["stays_within_accuracy_budget"] for item in by_workload.values()
        ),
        "accuracy_loss_budget_pp": ACCURACY_LOSS_BUDGET_PP,
    }


def summarize_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({str(row["case_family"]) for row in rows})
    required = set(FAILURE_FAMILIES_REQUIRED)
    by_family: dict[str, Any] = {}
    for family in families:
        group = [row for row in rows if row["case_family"] == family]
        by_family[family] = {
            "rows": len(group),
            "should_not_use_rows": sum(1 for row in group if row["should_not_use_suds"]),
            "min_edp_improvement_pct": min(as_float(row["edp_improvement_vs_reference_pct"]) for row in group),
            "min_accuracy_delta_pp": min(as_float(row["accuracy_delta_pp"]) for row in group),
        }
    return {
        "rows": len(rows),
        "families": families,
        "required_families": list(FAILURE_FAMILIES_REQUIRED),
        "missing_required_families": sorted(required - set(families)),
        "all_required_families_present": required.issubset(families),
        "should_not_use_rows": sum(1 for row in rows if row["should_not_use_suds"]),
        "by_family": by_family,
    }


def build_summary(
    *,
    failure_rows: list[dict[str, Any]],
    target_mc: dict[str, Any],
    boundary_mc: dict[str, Any],
    accuracy_mc: dict[str, Any],
    r3_json: dict[str, Any],
    r5_json: dict[str, Any],
    r6_json: dict[str, Any],
    r9_json: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    failures = summarize_failures(failure_rows)
    input_acceptance = {
        "r3_acceptance_state": r3_json.get("summary", {}).get("decision", {}).get("r3_acceptance_state", "missing"),
        "r5_acceptance_state": r5_json.get("summary", {}).get("decision", {}).get("r5_acceptance_state", "missing"),
        "r6_acceptance_state": r6_json.get("summary", {}).get("decision", {}).get("r6_acceptance_state", "missing"),
        "r9_acceptance_state": r9_json.get("summary", {}).get("decision", {}).get("r9_acceptance_state", "missing"),
    }
    blockers: list[str] = []
    if not failures["all_required_families_present"]:
        blockers.append("r10_required_failure_families_missing")
    if target_mc["any_crosses_zero_advantage"]:
        blockers.append("target_regime_uncertainty_crosses_zero_advantage")
    if not accuracy_mc["all_promoted_seed_bootstrap_within_accuracy_budget"]:
        blockers.append("accuracy_seed_bootstrap_exceeds_promoted_budget")
    for phase, state in input_acceptance.items():
        if state != "pass":
            blockers.append(f"{phase}_not_pass")

    target_survives = not target_mc["any_crosses_zero_advantage"]
    boundary_narrows = bool(boundary_mc["any_crosses_zero_advantage"] or failures["should_not_use_rows"] > 0)
    decision = {
        "r10_acceptance_state": "pass" if not blockers else "fail",
        "stop_condition_state": (
            "no R10 hard stop; target-regime uncertainty preserves positive EDP advantage"
            if not target_mc["any_crosses_zero_advantage"]
            else "R10 hard stop; target-regime uncertainty crosses zero EDP advantage"
        ),
        "blockers": sorted(set(blockers)),
        "target_regime_uncertainty_crosses_zero_advantage": target_mc["any_crosses_zero_advantage"],
        "headline_conclusions_survive_uncertainty": target_survives,
        "claim_narrowing_required_for_boundary_cases": boundary_narrows,
        "all_required_failure_families_present": failures["all_required_families_present"],
        "all_promoted_seed_bootstrap_within_accuracy_budget": accuracy_mc[
            "all_promoted_seed_bootstrap_within_accuracy_budget"
        ],
        "claim": (
            "Promoted SUDS-Pareto rows keep positive target-regime EDP advantage under bounded Monte Carlo "
            "uncertainty, while failure rows define low-slack, memory-dominant, unmeasured workload, "
            "activation-sensitive, and signal-dominant boundaries."
        ),
    }
    return {
        "tag": args.tag,
        "date": DATE,
        "rows": len(failure_rows),
        "failure_suite": failures,
        "target_regime_uncertainty": target_mc,
        "boundary_regime_uncertainty": boundary_mc,
        "accuracy_seed_uncertainty": accuracy_mc,
        "input_acceptance": input_acceptance,
        "decision": decision,
        "artifacts": {
            "failure_suite_csv": repo_path(args.csv_out),
            "uncertainty_json": repo_path(args.json_out),
            "report": repo_path(args.report_out),
        },
        "source_artifacts": {
            "architecture_summary_csv": repo_path(args.architecture_summary_csv),
            "scheduler_trace_csv": repo_path(args.scheduler_trace_csv),
            "system_sensitivity_csv": repo_path(args.system_sensitivity_csv),
            "system_sensitivity_json": repo_path(args.system_sensitivity_json),
            "workload_expansion_csv": repo_path(args.workload_expansion_csv),
            "workload_expansion_json": repo_path(args.workload_expansion_json),
            "end_to_end_accuracy_json": repo_path(args.end_to_end_accuracy_json),
            "pareto_design_space_json": repo_path(args.pareto_design_space_json),
            "conservative_pareto_csv": repo_path(args.conservative_pareto_csv),
            "glue_json": repo_path(args.glue_json),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "tag",
        "date",
        "roadmap_item",
        "case_id",
        "case_family",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    *,
    args: argparse.Namespace,
    summary: dict[str, Any],
    failure_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": args.tag,
            "artifact_id": f"suds_tetc_uncertainty_{args.tag}",
            "roadmap_item": "R10_failure_cases_and_uncertainty",
            "evidence_label": "failure_cases_and_uncertainty",
            "promotion_decision": "r10_pass"
            if summary["decision"]["r10_acceptance_state"] == "pass"
            else "r10_fail",
            "regeneration_command": "make suds-tetc-failure-uncertainty",
            "monte_carlo_draws_target": TARGET_DRAWS,
            "monte_carlo_draws_boundary": BOUNDARY_DRAWS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "rng_seed": RNG_SEED,
        },
        "summary": summary,
        "failure_suite_rows": failure_rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, args: argparse.Namespace, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    decision = summary["decision"]
    target = summary["target_regime_uncertainty"]
    boundary = summary["boundary_regime_uncertainty"]
    accuracy = summary["accuracy_seed_uncertainty"]
    failures = summary["failure_suite"]
    lines = [
        "# SUDS TETC Failure Cases And Uncertainty",
        "",
        f"Tag: `{args.tag}`",
        "Roadmap item: `R10_failure_cases_and_uncertainty`",
        "Evidence label: `failure_cases_and_uncertainty`",
        f"Acceptance state: `{decision['r10_acceptance_state']}`",
        f"Stop-condition state: `{decision['stop_condition_state']}`",
        "",
        "## Scope",
        "",
        "This artifact records counterexamples and bounded uncertainty for the",
        "promoted `suds_pareto` point. It reuses R3 measured MPS accuracy, R5",
        "Pareto linkage, R6 system-sensitivity logic, and R9 workload-boundary",
        "rows. It is not a new hardware measurement or a new model evaluation.",
        "",
        "## Decision",
        "",
        f"- R10 acceptance: `{decision['r10_acceptance_state']}`",
        f"- Blockers: `{';'.join(decision['blockers']) or 'none'}`",
        f"- Target-regime uncertainty crosses zero advantage: `{decision['target_regime_uncertainty_crosses_zero_advantage']}`",
        f"- Boundary-regime claim narrowing required: `{decision['claim_narrowing_required_for_boundary_cases']}`",
        f"- Required failure families present: `{decision['all_required_failure_families_present']}`",
        f"- Accuracy seed bootstrap within budget: `{decision['all_promoted_seed_bootstrap_within_accuracy_budget']}`",
        "",
        "## Target-Regime Monte Carlo",
        "",
        "| Workload | Mean EDP improvement | 95% CI | P(positive) | P(material >=5%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for workload, item in target["by_workload"].items():
        ci = item["edp_improvement_ci95_pct"]
        lines.append(
            f"| `{workload}` | {fmt(item['edp_improvement_mean_pct'], 2)}% | "
            f"[{fmt(ci[0], 2)}, {fmt(ci[1], 2)}]% | "
            f"{fmt(item['probability_positive_edp_advantage'], 3)} | "
            f"{fmt(item['probability_material_edp_advantage'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Boundary-Regime Monte Carlo",
            "",
            f"- Any boundary workload crosses zero advantage: `{boundary['any_crosses_zero_advantage']}`",
            f"- Minimum boundary 95% lower EDP-improvement bound: `{fmt(boundary['minimum_ci95_lower_pct'], 2)}%`",
            "",
            "Boundary draws intentionally include extreme conversion, link, sideband,",
            "and memory-pressure settings. They are used to narrow the claim, not to",
            "replace the target regime.",
            "",
            "## Accuracy Bootstrap",
            "",
            "| Workload | Seed values | Mean delta | 95% CI | Worst observed | Within 1 pp budget |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for workload, item in accuracy["by_workload"].items():
        ci = item["ci95_delta_pp"]
        lines.append(
            f"| `{workload}` | {item['n_values']} | {fmt(item['mean_delta_pp'], 3)} pp | "
            f"[{fmt(ci[0], 3)}, {fmt(ci[1], 3)}] pp | "
            f"{fmt(item['worst_observed_delta_pp'], 3)} pp | "
            f"`{item['stays_within_accuracy_budget']}` |"
        )

    lines.extend(
        [
            "",
            "## Failure Families",
            "",
            "| Family | Rows | Should-not-use rows | Min EDP improvement | Min accuracy delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for family, item in failures["by_family"].items():
        lines.append(
            f"| `{family}` | {item['rows']} | {item['should_not_use_rows']} | "
            f"{fmt(item['min_edp_improvement_pct'], 2)}% | {fmt(item['min_accuracy_delta_pp'], 3)} pp |"
        )

    lines.extend(
        [
            "",
            "## Counterexample Rows",
            "",
            "| Case | Family | Workload | Condition | Should not use | Failure mode |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['case_family']}` | `{row['workload']}` | "
            f"`{row['condition']}` | `{row['should_not_use_suds']}` | {row['failure_mode']} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Failure-suite CSV: `{repo_path(args.csv_out)}`",
            f"- Uncertainty JSON: `{repo_path(args.json_out)}`",
            f"- Report: `{repo_path(args.report_out)}`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "make suds-tetc-failure-uncertainty",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arch_rows = load_csv(args.architecture_summary_csv)
    scheduler_rows = load_csv(args.scheduler_trace_csv)
    system_rows = load_csv(args.system_sensitivity_csv)
    workload_rows = load_csv(args.workload_expansion_csv)
    conservative_rows = load_csv(args.conservative_pareto_csv)
    r3_json = load_json(args.end_to_end_accuracy_json)
    r5_json = load_json(args.pareto_design_space_json)
    r6_json = load_json(args.system_sensitivity_json)
    r9_json = load_json(args.workload_expansion_json)
    glue_json = load_json(args.glue_json)

    failure_rows = build_failure_rows(
        arch_rows=arch_rows,
        scheduler_rows=scheduler_rows,
        system_rows=system_rows,
        workload_rows=workload_rows,
    )
    index = nominal_index(arch_rows)
    target_mc = monte_carlo(index, scenario_builder=target_scenario, draws=TARGET_DRAWS, seed_offset=1)
    boundary_mc = monte_carlo(index, scenario_builder=boundary_scenario, draws=BOUNDARY_DRAWS, seed_offset=2)
    accuracy_mc = accuracy_uncertainty(glue_json=glue_json, conservative_rows=conservative_rows)
    summary = build_summary(
        failure_rows=failure_rows,
        target_mc=target_mc,
        boundary_mc=boundary_mc,
        accuracy_mc=accuracy_mc,
        r3_json=r3_json,
        r5_json=r5_json,
        r6_json=r6_json,
        r9_json=r9_json,
        args=args,
    )
    return failure_rows, summary


def main() -> None:
    args = parse_args()
    rows, summary = build(args)
    write_csv(args.csv_out, rows)
    write_json(args.json_out, args=args, summary=summary, failure_rows=rows)
    write_report(args.report_out, args=args, summary=summary, rows=rows)
    print(f"wrote {repo_path(args.csv_out)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")
    print(f"r10_acceptance_state={summary['decision']['r10_acceptance_state']}")
    print(f"stop_condition_state={summary['decision']['stop_condition_state']}")


if __name__ == "__main__":
    main()
