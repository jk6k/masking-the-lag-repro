#!/usr/bin/env python3
"""Materialize and audit the current FULLER noise_robustness family surface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from .build_fuller_experiment_program import build_fuller_experiment_program
    from .build_fuller_phase4_intake_contract import build_fuller_phase4_intake_contract
    from .build_fuller_report_pack_contract import build_fuller_report_pack_contract
    from .materialize_fuller_experiment_execution_plan import materialize_fuller_experiment_execution_plan
    from .fuller_experiment_program_common import ROOT, _write_csv, _write_json, _write_text
except ImportError:
    from build_fuller_experiment_program import build_fuller_experiment_program  # type: ignore
    from build_fuller_phase4_intake_contract import build_fuller_phase4_intake_contract  # type: ignore
    from build_fuller_report_pack_contract import build_fuller_report_pack_contract  # type: ignore
    from materialize_fuller_experiment_execution_plan import materialize_fuller_experiment_execution_plan  # type: ignore
    from fuller_experiment_program_common import ROOT, _write_csv, _write_json, _write_text  # type: ignore


DEFAULT_PROGRAM_CONTRACT = ROOT / "configs" / "fuller_experiment_program_contract_20260422.yaml"
DEFAULT_BUNDLE = ROOT / "configs" / "fuller_implementation_execution_bundle_20260319.yaml"
DEFAULT_WRAPPER_MANIFEST = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_noise_robustness_wrapper_manifest_20260423.json"
)
DEFAULT_AUDIT_CSV = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_noise_robustness_materialization_audit_20260423.csv"
)
DEFAULT_AUDIT_JSON = (
    ROOT / "experiments" / "results" / "report_data" / "fuller_noise_robustness_materialization_audit_20260423.json"
)
DEFAULT_AUDIT_MD = (
    ROOT / "docs" / "reports" / "20260423_fuller_noise_robustness_materialization_audit.md"
)

AUDIT_FIELDS = [
    "family_id",
    "model",
    "sweep_resolution",
    "profile_count",
    "default_seed_count",
    "extra_seed_profiles",
    "extra_seed_count",
    "accuracy_run_count",
    "phase1_run_count",
    "eval_batch_size",
    "max_eval_samples",
    "current_status",
    "claim_tier",
    "readiness_gate",
    "recommended_next_step",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _noise_family_id(*, model: str, sweep_resolution: str) -> str:
    suffix_map = {
        "mobilevit_s": "S",
        "mobilevit_xs": "XS",
        "mobilevit_xxs": "XXS",
    }
    suffix = suffix_map.get(str(model).strip())
    if suffix is None:
        raise SystemExit(f"Unsupported noise family model: {model!r}")
    return f"NOISE_IMAGENET_MOBILEVIT_{suffix}_{str(sweep_resolution).upper()}"


def _seed_list(value: Any, *, default: list[int]) -> list[int]:
    if value in ("", None):
        return list(default)
    if isinstance(value, int):
        return [int(value)]
    if not isinstance(value, list):
        raise SystemExit(f"Expected integer seed list, got {value!r}")
    return [int(item) for item in value] or list(default)


def _audit_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    noise_cfg = bundle.get("noise") or {}
    seed_cfg = dict(noise_cfg.get("seed_policy") or {})
    default_seeds = _seed_list(seed_cfg.get("default_seeds"), default=[0])
    sparse_support_seeds = _seed_list(
        seed_cfg.get("sparse_support_seeds"),
        default=default_seeds,
    )
    dense_representatives = list(seed_cfg.get("dense_representative_profiles") or [])
    eval_batch_size = int(noise_cfg.get("eval_batch_size") or 0)
    max_eval_samples = noise_cfg.get("max_eval_samples")
    rows: list[dict[str, Any]] = []

    dense_cfg = noise_cfg.get("dense") or {}
    dense_alpha = list(dense_cfg.get("crosstalk_alpha") or [])
    dense_gaussian = list(dense_cfg.get("gaussian_noise_std") or [])
    dense_profile_count = len(dense_alpha) * len(dense_gaussian)
    dense_extra_profiles = 0
    dense_extra_seed_count = 0
    for profile in dense_representatives:
        alpha = float(profile["crosstalk_alpha"])
        gaussian = float(profile["gaussian_noise_std"])
        if alpha in [float(item) for item in dense_alpha] and gaussian in [
            float(item) for item in dense_gaussian
        ]:
            dense_extra_profiles += 1
            dense_extra_seed_count += len(
                [
                    seed
                    for seed in _seed_list(profile.get("extra_seeds"), default=[])
                    if seed not in set(default_seeds)
                ]
            )
    dense_accuracy_run_count = dense_profile_count * len(default_seeds) + dense_extra_seed_count
    rows.append(
        {
            "family_id": _noise_family_id(model=str(dense_cfg["model"]), sweep_resolution="dense"),
            "model": str(dense_cfg["model"]),
            "sweep_resolution": "dense",
            "profile_count": dense_profile_count,
            "default_seed_count": len(default_seeds),
            "extra_seed_profiles": dense_extra_profiles,
            "extra_seed_count": dense_extra_seed_count,
            "accuracy_run_count": dense_accuracy_run_count,
            "phase1_run_count": dense_profile_count,
            "eval_batch_size": eval_batch_size,
            "max_eval_samples": "" if max_eval_samples is None else max_eval_samples,
            "current_status": "materialized_not_started_heavy_legacy_grid",
            "claim_tier": "support_family",
            "readiness_gate": "noise_family_current_outputs_ready",
            "recommended_next_step": "bounded_noise_family_contraction_or_explicit_authorization",
        }
    )

    for item in noise_cfg.get("sparse_support") or []:
        profile_count = len(item.get("profiles") or [])
        rows.append(
            {
                "family_id": _noise_family_id(
                    model=str(item["model"]),
                    sweep_resolution="sparse",
                ),
                "model": str(item["model"]),
                "sweep_resolution": "sparse",
                "profile_count": profile_count,
                "default_seed_count": len(sparse_support_seeds),
                "extra_seed_profiles": 0,
                "extra_seed_count": 0,
                "accuracy_run_count": profile_count * len(sparse_support_seeds),
                "phase1_run_count": profile_count,
                "eval_batch_size": eval_batch_size,
                "max_eval_samples": "" if max_eval_samples is None else max_eval_samples,
                "current_status": "materialized_not_started_heavy_legacy_grid",
                "claim_tier": "support_family",
                "readiness_gate": "noise_family_current_outputs_ready",
                "recommended_next_step": "bounded_noise_family_contraction_or_explicit_authorization",
            }
        )
    return rows


def _audit_note(rows: list[dict[str, Any]], wrapper_manifest: Path) -> str:
    total_profiles = sum(int(row["profile_count"]) for row in rows)
    total_accuracy_runs = sum(int(row["accuracy_run_count"]) for row in rows)
    total_phase1_runs = sum(int(row["phase1_run_count"]) for row in rows)
    lines = [
        "# FULLER Noise Robustness Materialization Audit",
        "",
        "Date: `2026-04-23`",
        "Status: `noise_robustness_materialized_not_started`",
        "",
        "## Wrapper",
        "",
        f"- wrapper_manifest: `{wrapper_manifest}`",
        "",
        "## Scale",
        "",
        f"- total_profiles: `{total_profiles}`",
        f"- total_accuracy_runs: `{total_accuracy_runs}`",
        f"- total_phase1_runs: `{total_phase1_runs}`",
        "",
        "## Families",
        "",
    ]
    lines.extend(
        f"- `{row['family_id']}` profiles=`{row['profile_count']}` accuracy_runs=`{row['accuracy_run_count']}` status=`{row['current_status']}` next=`{row['recommended_next_step']}`"
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The current noise family is materialized and traceable through the new experiment program,",
            "but the legacy-compatible bundle remains a heavy support-family grid. It should not be launched",
            "blindly as the default next run; the governed next step is either a bounded contraction or explicit",
            "authorization to execute the full support-family sweep.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_fuller_noise_robustness_materialization_audit(
    contract_path: Path = DEFAULT_PROGRAM_CONTRACT,
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    wrapper_manifest_out: Path = DEFAULT_WRAPPER_MANIFEST,
    audit_csv: Path = DEFAULT_AUDIT_CSV,
    audit_json: Path = DEFAULT_AUDIT_JSON,
    audit_md: Path = DEFAULT_AUDIT_MD,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    build_fuller_experiment_program(contract_path, root_dir=root_dir)
    execution_payload = materialize_fuller_experiment_execution_plan(contract_path, root_dir=root_dir)
    phase4_payload = build_fuller_phase4_intake_contract(contract_path, root_dir=root_dir)
    report_payload = build_fuller_report_pack_contract(contract_path, root_dir=root_dir)
    wrapper_payload = {
        "wrapper_mode": "fuller_experiment_program",
        "legacy_entrypoint": str((root_dir / "experiments" / "tools" / "run_fuller_noise_sweeps.py").resolve()),
        "selected_family": "noise_robustness",
        "program_contract": str((contract_path if contract_path.is_absolute() else (root_dir / contract_path)).resolve()),
        "execution_plan_csv": execution_payload["execution_plan_csv"],
        "phase4_intake_contract_csv": phase4_payload["phase4_intake_contract_csv"],
        "report_contract_csv": report_payload["report_contract_csv"],
    }
    wrapper_manifest_out.parent.mkdir(parents=True, exist_ok=True)
    wrapper_manifest_out.write_text(
        json.dumps(wrapper_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    bundle = _load_yaml(bundle_path)
    rows = _audit_rows(bundle)
    total_profiles = sum(int(row["profile_count"]) for row in rows)
    total_accuracy_runs = sum(int(row["accuracy_run_count"]) for row in rows)
    total_phase1_runs = sum(int(row["phase1_run_count"]) for row in rows)

    _write_csv(audit_csv, AUDIT_FIELDS, rows)
    _write_json(
        audit_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "wrapper_payload": wrapper_payload,
            "totals": {
                "profile_count": total_profiles,
                "accuracy_run_count": total_accuracy_runs,
                "phase1_run_count": total_phase1_runs,
            },
            "rows": rows,
        },
    )
    _write_text(audit_md, _audit_note(rows, wrapper_manifest_out))
    return {
        "status": "pass",
        "row_count": len(rows),
        "profile_count": total_profiles,
        "accuracy_run_count": total_accuracy_runs,
        "phase1_run_count": total_phase1_runs,
        "audit_csv": str(audit_csv.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the FULLER noise_robustness materialization audit."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_PROGRAM_CONTRACT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--wrapper-manifest-out", type=Path, default=DEFAULT_WRAPPER_MANIFEST)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT_MD)
    args = parser.parse_args()
    payload = build_fuller_noise_robustness_materialization_audit(
        contract_path=args.contract,
        bundle_path=args.bundle,
        wrapper_manifest_out=args.wrapper_manifest_out,
        audit_csv=args.audit_csv,
        audit_json=args.audit_json,
        audit_md=args.audit_md,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
