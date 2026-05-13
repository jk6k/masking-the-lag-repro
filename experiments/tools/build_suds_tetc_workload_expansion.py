#!/usr/bin/env python3
"""Build the R9 workload-generality expansion artifact.

R9 adds a new Transformer-family workload surface and sequence/batch sweeps
without pretending that architecture-only rows are measured accuracy results.
The generator reuses the selected TETC DPTC architecture model and records the
dataset/weights blocker for new measured DeiT-Tiny accuracy runs, then emits
simulator-only traces as explicit boundary/generalization evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from build_suds_transformer_architecture_sim import (
    CONDITION_LABELS,
    REPO_ROOT,
    TAG,
    derive_params,
    git_hash,
    load_json,
    normalize_rows,
    schedule_ops,
    simulate_condition,
    source_profile_rows,
)


DATE = "2026-05-14"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

MOBILEVIT_JSON = REPORT_DATA / "suds_mobilevit_multimodel_validation_20260511_p2p3_quality.json"
CONSERVATIVE_PARETO_JSON = REPORT_DATA / f"suds_tetc_conservative_pareto_{TAG}.json"
GLUE_JSON = REPORT_DATA / "suds_glue_measured_validation_20260511_p2p3_quality.json"
ADC_JSON = REPORT_DATA / "suds_adc_macro_sanity_20260512_j1_quality_boost.json"
RTL_JSON = REPORT_DATA / "suds_rtl_control_overhead_20260512_j2_quality_boost.json"
PHY_JSON = REPORT_DATA / "suds_phy_circuit_boundary_20260511_p2p3_quality.json"

CSV_OUT = REPORT_DATA / f"suds_tetc_workload_expansion_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_workload_expansion_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_tetc_workload_expansion.md"

CONDITIONS = (
    "lightening_dptc",
    "l1",
    "slack_only",
    "signal_only",
    "hyatten_style",
    "suds_pareto",
)
BERT_SEQUENCE_LENGTHS = (64, 128, 256, 512)
BATCH_SIZES = (1, 4, 8)
DEIT_SEQUENCE_LENGTH = 197
MPS_PYTHON = REPO_ROOT / ".venv311-mps/bin/python"
MPS_PYTHON_LABEL = ".venv311-mps/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--mobilevit-json", type=Path, default=MOBILEVIT_JSON)
    parser.add_argument("--conservative-pareto-json", type=Path, default=CONSERVATIVE_PARETO_JSON)
    parser.add_argument("--glue-json", type=Path, default=GLUE_JSON)
    parser.add_argument("--adc-json", type=Path, default=ADC_JSON)
    parser.add_argument("--rtl-json", type=Path, default=RTL_JSON)
    parser.add_argument("--phy-json", type=Path, default=PHY_JSON)
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


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def mps_probe() -> dict[str, Any]:
    if not MPS_PYTHON.is_file():
        return {
            "status": "fail",
            "python": MPS_PYTHON_LABEL,
            "mps_available": False,
            "mps_built": False,
            "torch_version": "",
            "blocker": ".venv311-mps Python is missing",
        }
    code = (
        "import json\n"
        "import torch\n"
        "print(json.dumps({"
        "'torch_version': getattr(torch, '__version__', 'unknown'), "
        "'mps_built': bool(torch.backends.mps.is_built()), "
        "'mps_available': bool(torch.backends.mps.is_available())"
        "}, sort_keys=True))\n"
    )
    try:
        completed = subprocess.run(
            [str(MPS_PYTHON), "-c", code],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "status": "fail",
            "python": MPS_PYTHON_LABEL,
            "mps_available": False,
            "mps_built": False,
            "torch_version": "",
            "blocker": f"MPS probe failed: {exc!r}",
        }
    mps_available = bool(payload.get("mps_available"))
    return {
        "status": "pass" if mps_available else "fail",
        "python": MPS_PYTHON_LABEL,
        "mps_available": mps_available,
        "mps_built": bool(payload.get("mps_built")),
        "torch_version": payload.get("torch_version", ""),
        "blocker": "" if mps_available else "MPS backend is not available",
    }


def generate_bert_ops(seq_len: int, batch_size: int) -> list[dict[str, Any]]:
    hidden = 768
    heads = 12
    head_dim = hidden // heads
    intermediate = 3072
    tokens = seq_len * batch_size
    ops: list[dict[str, Any]] = []
    for layer in range(12):
        prefix = f"bert.encoder.layer.{layer}"
        ops.extend(
            [
                {
                    "name": f"{prefix}.attention.self.qkv",
                    "type": "linear",
                    "m": tokens,
                    "d": hidden,
                    "n": 3 * hidden,
                    "kernel_class": "mha_qkv_projection",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.attention.self.qk_scores",
                    "type": "linear",
                    "m": heads * seq_len * batch_size,
                    "d": head_dim,
                    "n": seq_len,
                    "kernel_class": "mha_qk_scores",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.attention.self.av_context",
                    "type": "linear",
                    "m": heads * seq_len * batch_size,
                    "d": seq_len,
                    "n": head_dim,
                    "kernel_class": "mha_av_context",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.attention.output.dense",
                    "type": "linear",
                    "m": tokens,
                    "d": hidden,
                    "n": hidden,
                    "kernel_class": "mha_output_projection",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.intermediate.dense",
                    "type": "linear",
                    "m": tokens,
                    "d": hidden,
                    "n": intermediate,
                    "kernel_class": "ffn_expand",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.output.dense",
                    "type": "linear",
                    "m": tokens,
                    "d": intermediate,
                    "n": hidden,
                    "kernel_class": "ffn_project",
                    "layer_index": layer,
                },
            ]
        )
    return ops


def generate_vit_ops(
    *,
    model: str,
    seq_len: int,
    hidden: int,
    layers: int,
    heads: int,
    mlp_ratio: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    head_dim = hidden // heads
    intermediate = hidden * mlp_ratio
    tokens = seq_len * batch_size
    ops: list[dict[str, Any]] = []
    for layer in range(layers):
        prefix = f"{model}.blocks.{layer}"
        ops.extend(
            [
                {
                    "name": f"{prefix}.attn.qkv",
                    "type": "linear",
                    "m": tokens,
                    "d": hidden,
                    "n": 3 * hidden,
                    "kernel_class": "mha_qkv_projection",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.attn.qk_scores",
                    "type": "linear",
                    "m": heads * seq_len * batch_size,
                    "d": head_dim,
                    "n": seq_len,
                    "kernel_class": "mha_qk_scores",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.attn.av_context",
                    "type": "linear",
                    "m": heads * seq_len * batch_size,
                    "d": seq_len,
                    "n": head_dim,
                    "kernel_class": "mha_av_context",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.attn.proj",
                    "type": "linear",
                    "m": tokens,
                    "d": hidden,
                    "n": hidden,
                    "kernel_class": "mha_output_projection",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.mlp.fc1",
                    "type": "linear",
                    "m": tokens,
                    "d": hidden,
                    "n": intermediate,
                    "kernel_class": "ffn_expand",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.mlp.fc2",
                    "type": "linear",
                    "m": tokens,
                    "d": intermediate,
                    "n": hidden,
                    "kernel_class": "ffn_project",
                    "layer_index": layer,
                },
            ]
        )
    return ops


def workload_defs() -> dict[str, dict[str, Any]]:
    workloads: dict[str, dict[str, Any]] = {}
    for seq_len in BERT_SEQUENCE_LENGTHS:
        for batch_size in BATCH_SIZES:
            workload = f"bert_base_seq{seq_len}_batch{batch_size}_r9"
            workloads[workload] = {
                "workload": workload,
                "workload_family": "NLP Transformer encoder",
                "model": "bert_base",
                "dataset_or_split": (
                    f"GLUE-style architecture sweep, seq_len={seq_len}, "
                    f"batch_size={batch_size}"
                ),
                "ops": generate_bert_ops(seq_len, batch_size),
                "architecture_source": "generated canonical BERT-base encoder GEMM schedule with R9 sequence/batch sweep",
                "source_profile_workload": "bert_base_glue_seq128",
                "candidate_workload_type": "sequence_batch_sweep",
                "sequence_length": seq_len,
                "batch_size": batch_size,
                "setup_blocker": ""
                if seq_len == 128 and batch_size == 1
                else "no governed measured accuracy row for this exact sequence-length/batch setting",
            }
    for batch_size in BATCH_SIZES:
        workload = f"deit_tiny_patch16_224_batch{batch_size}_r9"
        workloads[workload] = {
            "workload": workload,
            "workload_family": "vision Transformer encoder",
            "model": "deit_tiny_patch16_224",
            "dataset_or_split": (
                "ImageNet-style architecture sweep, patch16 224, "
                f"tokens={DEIT_SEQUENCE_LENGTH}, batch_size={batch_size}"
            ),
            "ops": generate_vit_ops(
                model="deit_tiny_patch16_224",
                seq_len=DEIT_SEQUENCE_LENGTH,
                hidden=192,
                layers=12,
                heads=3,
                mlp_ratio=4,
                batch_size=batch_size,
            ),
            "architecture_source": "generated DeiT-Tiny ViT encoder GEMM schedule, patch16 224 with class token",
            "source_profile_workload": "mobilevit_s_transformer_blocks_256",
            "candidate_workload_type": "new_transformer_family",
            "sequence_length": DEIT_SEQUENCE_LENGTH,
            "batch_size": batch_size,
            "setup_blocker": (
                "no local governed DeiT-Tiny weights/dataset accuracy run found; "
                "R9 emits simulator-only traces instead of hiding the setup blocker"
            ),
        }
    return workloads


def blank_boundary_profile(anchor: dict[str, Any], *, label: str, source_condition: str) -> dict[str, Any]:
    out = dict(anchor)
    out.update(
        {
            "accuracy": math.nan,
            "delta_accuracy": math.nan,
            "accuracy_evidence_label": label,
            "promotion_decision": "boundary",
            "device": "not_run_mps_required",
            "git_hash": git_hash(),
            "n_rows": 0,
            "source_condition": source_condition,
        }
    )
    return out


def profile_for_condition(
    workload_meta: dict[str, Any],
    condition: str,
    profiles: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    source_workload = workload_meta["source_profile_workload"]
    source_profiles = profiles[source_workload]
    dense = source_profiles["e0_dense"]
    boundary_label = (
        "new_workload_unmeasured_accuracy_boundary"
        if workload_meta["candidate_workload_type"] == "new_transformer_family"
        else "sequence_batch_unmeasured_accuracy_boundary"
    )
    measured_anchor = (
        workload_meta["candidate_workload_type"] == "sequence_batch_sweep"
        and int(workload_meta["sequence_length"]) == 128
        and int(workload_meta["batch_size"]) == 1
    )

    if condition == "lightening_dptc":
        if measured_anchor:
            out = dict(dense)
            out["source_condition"] = "e0_dense"
            out["promotion_decision"] = "appendix"
            return out
        return blank_boundary_profile(dense, label=boundary_label, source_condition="e0_dense_transfer")

    if condition == "l1":
        anchor = source_profiles.get("e2_l1", dense)
        if measured_anchor:
            out = dict(anchor)
            out["source_condition"] = "e2_l1"
            out["promotion_decision"] = "appendix"
            return out
        label = boundary_label
        return blank_boundary_profile(anchor, label=label, source_condition="e2_l1_transfer")

    if condition == "slack_only":
        anchor = source_profiles.get("e3_slack", source_profiles.get("e2_l1", dense))
        if measured_anchor:
            out = dict(anchor)
            out["source_condition"] = "e3_slack"
            out["promotion_decision"] = "appendix"
            return out
        label = boundary_label
        return blank_boundary_profile(anchor, label=label, source_condition="e3_slack_transfer")

    if condition == "signal_only":
        anchor = source_profiles.get("e6_signal") or source_profiles.get("e8_overflow") or dense
        return blank_boundary_profile(anchor, label=boundary_label, source_condition="signal_transfer")

    if condition == "suds_pareto":
        if source_workload == "bert_base_glue_seq128":
            anchor = source_profiles.get("e2_l1", dense)
            source_condition = "e2_l1_schedule_guarded_transfer"
        else:
            anchor = source_profiles.get("e9_suds_conservative") or source_profiles.get("e4_suds", dense)
            source_condition = "e9_suds_conservative_transfer"
        if measured_anchor:
            out = dict(anchor)
            out.update(
                {
                    "source_condition": "e2_l1_schedule_guarded_zero_loss",
                    "promotion_decision": "appendix",
                    "accuracy_evidence_label": "measured_mps_glue",
                    "schedule_guard": "dptc_photonic_tile_schedule",
                }
            )
            return out
        return blank_boundary_profile(anchor, label=boundary_label, source_condition=source_condition)

    if condition == "hyatten_style":
        return {
            "keep_ratio": 0.15,
            "degrade_ratio": 0.85,
            "prune_ratio": 0.0,
            "adc_energy_ratio_vs_e0": 0.15 + 0.85 / 16.0,
            "accuracy_metric": dense.get("accuracy_metric", ""),
            "accuracy": math.nan,
            "delta_accuracy": math.nan,
            "accuracy_evidence_label": "literature_baseline_unmeasured_locally",
            "promotion_decision": "boundary",
            "device": "not_run_mps_required",
            "git_hash": git_hash(),
            "n_rows": 0,
            "source_condition": "hyatten_low_resolution_fraction_transfer",
        }

    raise SystemExit(f"unsupported R9 condition: {condition}")


def build_rows(args: argparse.Namespace, probe: dict[str, Any]) -> list[dict[str, Any]]:
    mobilevit = load_json(args.mobilevit_json)
    conservative = load_json(args.conservative_pareto_json)
    glue = load_json(args.glue_json)
    params = derive_params(load_json(args.adc_json), load_json(args.rtl_json), load_json(args.phy_json))
    profiles = source_profile_rows(mobilevit, glue, conservative)
    rows: list[dict[str, Any]] = []

    for workload, meta in workload_defs().items():
        schedule = schedule_ops(workload, meta, params)
        for condition in CONDITIONS:
            profile = profile_for_condition(meta, condition, profiles)
            row = simulate_condition(
                schedule,
                meta,
                workload,
                condition,
                profile,
                params,
                sensitivity_case="nominal",
                adc_sharing_mode="temporal_accum",
            )
            setup_blocker = meta["setup_blocker"]
            accuracy_status = "existing_mps_anchor" if not setup_blocker and condition in {
                "lightening_dptc",
                "l1",
                "slack_only",
                "suds_pareto",
            } else "not_run_boundary_or_blocker_recorded"
            row.update(
                {
                    "roadmap_item": "R9_workload_generality_expansion",
                    "sweep_family": meta["candidate_workload_type"],
                    "sequence_length": meta["sequence_length"],
                    "batch_size": meta["batch_size"],
                    "source_profile_workload": meta["source_profile_workload"],
                    "workload_setup_status": "pass" if not setup_blocker else "simulator_only_with_recorded_blocker",
                    "accuracy_run_status": accuracy_status,
                    "setup_blocker": setup_blocker,
                    "hidden_failure_status": "visible_blocker_or_not_failed",
                    "device_policy": "mps_required_for_accuracy_runs",
                    "mps_probe_status": probe["status"],
                    "mps_available": probe["mps_available"],
                    "mps_python": probe["python"],
                    "claim_role": (
                        "measured_anchor"
                        if accuracy_status == "existing_mps_anchor"
                        else "architecture_only_generality_boundary"
                    ),
                    "promotion_decision": "appendix_boundary",
                    "claim_boundary": (
                        "R9 workload expansion is simulator-only generality evidence unless "
                        "accuracy_evidence_label is measured_mps_*; it is not a new measured "
                        "accuracy, silicon, layout, device-solver, or bench-energy claim"
                    ),
                }
            )
            rows.append(row)
    normalize_rows(rows)
    for row in rows:
        row["supports_architecture_generalization"] = (
            row["condition"] == "suds_pareto"
            and float(row["edp_improvement_vs_lightening_pct"]) > 0.0
        )
        row["result_class"] = (
            "architecture_support_with_accuracy_boundary"
            if row["supports_architecture_generalization"]
            else "boundary_or_baseline_context"
        )
    return rows


def summarize(rows: list[dict[str, Any]], probe: dict[str, Any]) -> dict[str, Any]:
    suds_rows = [row for row in rows if row["condition"] == "suds_pareto"]
    new_workload_suds = [row for row in suds_rows if row["sweep_family"] == "new_transformer_family"]
    seq_lengths = sorted({int(row["sequence_length"]) for row in rows if row["model"] == "bert_base"})
    batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    hidden_failures = [
        row["workload"] for row in rows
        if row["accuracy_run_status"] != "existing_mps_anchor"
        and not row["setup_blocker"]
        and row["claim_role"] != "architecture_only_generality_boundary"
    ]
    setup_blockers = sorted({row["setup_blocker"] for row in rows if row["setup_blocker"]})
    min_new_edp_improvement = min(
        (float(row["edp_improvement_vs_lightening_pct"]) for row in new_workload_suds),
        default=math.nan,
    )
    min_all_suds_edp_improvement = min(
        (float(row["edp_improvement_vs_lightening_pct"]) for row in suds_rows),
        default=math.nan,
    )
    additional_workloads = sorted({row["workload"] for row in rows if row["sweep_family"] == "new_transformer_family"})
    has_new_transformer_workload = bool(additional_workloads)
    has_sequence_sweep = len(seq_lengths) >= 3
    has_batch_sweep = len(batch_sizes) >= 2
    mps_metadata_complete = probe["status"] == "pass" and all(row["device_policy"] == "mps_required_for_accuracy_runs" for row in rows)
    blockers = []
    if not has_new_transformer_workload:
        blockers.append("no_additional_transformer_workload")
    if not has_sequence_sweep:
        blockers.append("sequence_length_sweep_missing")
    if not has_batch_sweep:
        blockers.append("batch_size_sweep_missing")
    if not mps_metadata_complete:
        blockers.append("mps_metadata_incomplete")
    if hidden_failures:
        blockers.append("hidden_workload_failures_present")
    r9_acceptance_state = "pass" if not blockers else "fail"
    stop_condition_state = (
        "no R9 hard stop; DeiT-Tiny measured accuracy setup blocker is recorded "
        "and simulator-only traces are emitted"
        if r9_acceptance_state == "pass"
        else "R9 stop condition triggered; workload setup or metadata is incomplete"
    )
    return {
        "rows": len(rows),
        "suds_rows": len(suds_rows),
        "additional_transformer_workloads": additional_workloads,
        "sequence_lengths": seq_lengths,
        "batch_sizes": batch_sizes,
        "setup_blockers": setup_blockers,
        "hidden_failures": sorted(set(hidden_failures)),
        "mps_probe": probe,
        "mps_metadata_complete": mps_metadata_complete,
        "has_new_transformer_workload": has_new_transformer_workload,
        "has_sequence_sweep": has_sequence_sweep,
        "has_batch_sweep": has_batch_sweep,
        "new_workload_supports_architecture_claim": (
            bool(new_workload_suds) and min_new_edp_improvement > 0.0
        ),
        "min_new_workload_suds_edp_improvement_pct": min_new_edp_improvement,
        "min_all_suds_edp_improvement_pct": min_all_suds_edp_improvement,
        "decision": {
            "r9_acceptance_state": r9_acceptance_state,
            "stop_condition_state": stop_condition_state,
            "blockers": blockers,
            "dataset_weights_blocker_recorded": any("DeiT-Tiny" in item for item in setup_blockers),
            "no_failed_workload_hidden": not hidden_failures,
            "claim_boundary": "architecture-only workload generality; measured accuracy is not claimed for new DeiT/sequence-batch rows",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_artifacts = {
        "mobilevit_json": repo_path(args.mobilevit_json),
        "conservative_pareto_json": repo_path(args.conservative_pareto_json),
        "glue_json": repo_path(args.glue_json),
        "adc_json": repo_path(args.adc_json),
        "rtl_json": repo_path(args.rtl_json),
        "phy_json": repo_path(args.phy_json),
    }
    payload = {
        "metadata": {
            "tag": args.tag,
            "date": DATE,
            "artifact_id": f"suds_tetc_workload_expansion_{args.tag}",
            "roadmap_item": "R9_workload_generality_expansion",
            "evidence_label": "workload_generality_expansion",
            "promotion_decision": "appendix_boundary_generality_evidence",
            "git_hash": git_hash(),
            "regeneration_command": "make suds-tetc-workload-expansion",
            "source_artifacts": source_artifacts,
            "source_artifact_sha256": {
                name: sha256_path((REPO_ROOT / value).resolve())
                for name, value in source_artifacts.items()
                if (REPO_ROOT / value).is_file()
            },
        },
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    decision = summary["decision"]
    suds_rows = [row for row in rows if row["condition"] == "suds_pareto"]
    blocker_text = "; ".join(summary["setup_blockers"]) if summary["setup_blockers"] else "none"
    lines = [
        "# SUDS TETC Workload Expansion",
        "",
        f"Tag: `{args.tag}`",
        "Roadmap item: `R9_workload_generality_expansion`",
        "Evidence label: `workload_generality_expansion`",
        f"Acceptance state: `{decision['r9_acceptance_state']}`",
        f"Stop-condition state: `{decision['stop_condition_state']}`",
        "",
        "## Scope",
        "",
        "R9 adds simulator-only workload generality evidence for a new",
        "Transformer-family workload and explicit sequence/batch sweeps. It",
        "does not rerun or claim new measured accuracy. The governed MPS runtime",
        "is probed and recorded so future accuracy runs remain constrained to",
        "the project MPS policy.",
        "",
        "## Decision",
        "",
        f"- R9 acceptance: `{decision['r9_acceptance_state']}`",
        f"- Blockers: `{','.join(decision['blockers']) or 'none'}`",
        f"- MPS metadata complete: `{summary['mps_metadata_complete']}`",
        f"- New Transformer workload: `{','.join(summary['additional_transformer_workloads'])}`",
        f"- Sequence lengths: `{','.join(str(item) for item in summary['sequence_lengths'])}`",
        f"- Batch sizes: `{','.join(str(item) for item in summary['batch_sizes'])}`",
        f"- Dataset/weights blocker recorded: `{decision['dataset_weights_blocker_recorded']}`",
        "",
        "## SUDS Pareto Generality Rows",
        "",
        "| Workload | Seq | Batch | Energy improvement | EDP improvement | Accuracy status | Result class |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(suds_rows, key=lambda item: (item["model"], int(item["sequence_length"]), int(item["batch_size"]))):
        lines.append(
            "| `{workload}` | {seq} | {batch} | {energy}% | {edp}% | `{acc}` | `{klass}` |".format(
                workload=row["workload"],
                seq=row["sequence_length"],
                batch=row["batch_size"],
                energy=fmt(row["energy_improvement_vs_lightening_pct"], 2),
                edp=fmt(row["edp_improvement_vs_lightening_pct"], 2),
                acc=row["accuracy_run_status"],
                klass=row["result_class"],
            )
        )
    lines.extend(
        [
            "",
            "## Recorded Setup Boundary",
            "",
            f"- Setup blocker surface: `{blocker_text}`",
            f"- MPS probe: `{summary['mps_probe']['status']}` "
            f"with torch `{summary['mps_probe'].get('torch_version', '')}`",
            "- New DeiT-Tiny rows are architecture-only evidence until a governed",
            "  ImageNet/weights accuracy run exists on `mps`.",
            "- Long-sequence and larger-batch BERT rows are sequence/batch boundary",
            "  traces unless a matching governed accuracy artifact is produced.",
            "",
            "## Artifacts",
            "",
            f"- Workload expansion CSV: `{repo_path(args.csv_out)}`",
            f"- Workload expansion JSON: `{repo_path(args.json_out)}`",
            f"- Report: `{repo_path(args.report_out)}`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "make suds-tetc-workload-expansion",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    probe = mps_probe()
    rows = build_rows(args, probe)
    summary = summarize(rows, probe)
    write_csv(args.csv_out, rows)
    write_json(args.json_out, args, rows, summary)
    write_report(args.report_out, args, rows, summary)
    print(f"wrote {repo_path(args.csv_out)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")
    print(f"r9_acceptance_state={summary['decision']['r9_acceptance_state']}")


if __name__ == "__main__":
    main()
