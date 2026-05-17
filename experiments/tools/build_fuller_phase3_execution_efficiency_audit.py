#!/usr/bin/env python3
"""Build the current FULLER phase3 execution-efficiency audit surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .fuller_phase3_launch_common import (
        DEFAULT_CONTRACT,
        LANE_ORDER,
        ROOT,
        _build_mutated_command,
        _command_flag_value,
        _ensure_command_list,
        _load_json,
        _load_yaml,
        _resolve_path,
        _runtime_health_gate_from_command,
        _write_csv,
        _write_json,
        _write_text,
        _wrapped_with,
        utc_now_iso,
    )
except ImportError:
    from fuller_phase3_launch_common import (  # type: ignore
        DEFAULT_CONTRACT,
        LANE_ORDER,
        ROOT,
        _build_mutated_command,
        _command_flag_value,
        _ensure_command_list,
        _load_json,
        _load_yaml,
        _resolve_path,
        _runtime_health_gate_from_command,
        _write_csv,
        _write_json,
        _write_text,
        _wrapped_with,
        utc_now_iso,
    )

EXPERIMENTS_ROOT = ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

try:
    from ..exp_common.runtime import (
        host_memory_gb,
        host_perf_core_count,
        resolve_data_workers,
        resolve_eval_batch_size,
    )
except ImportError:
    from exp_common.runtime import (  # type: ignore
        host_memory_gb,
        host_perf_core_count,
        resolve_data_workers,
        resolve_eval_batch_size,
    )


EFFICIENCY_AUDIT_FIELDS = [
    "variant_id",
    "internal_experiment_id",
    "package_kind",
    "required_device",
    "current_workers",
    "tuned_workers",
    "current_eval_batch_size",
    "tuned_eval_batch_size",
    "dry_run_present",
    "caffeinate_present",
    "runtime_health_gate_json",
    "audit_status",
    "inefficiencies_json",
    "recommended_command_mutations_json",
    "notes",
]


def _host_tuning(contract: dict[str, Any]) -> dict[str, Any]:
    raw = contract.get("host_tuning") or {}
    tuned_workers = int(raw.get("tuned_workers") or 0)
    tuned_eval_batch_size = int(raw.get("tuned_eval_batch_size") or 0)
    required_device = str((contract.get("governance") or {}).get("required_device") or "mps").strip()
    return {
        "host_memory_gb": float(raw.get("host_memory_gb") or host_memory_gb() or 0.0),
        "perf_core_count": int(raw.get("perf_core_count") or host_perf_core_count(system="Darwin") or 0),
        "auto_workers_mps": int(
            raw.get("auto_workers_mps")
            or resolve_data_workers(None, required_device, system="Darwin")
            or 0
        ),
        "tuned_workers": tuned_workers,
        "tuned_eval_batch_size": tuned_eval_batch_size,
        "tuning_note": str(raw.get("tuning_note") or "").strip(),
        "resolved_eval_batch_size": int(
            resolve_eval_batch_size(None, required_device, "mobilevit_s") or tuned_eval_batch_size
        ),
    }


def build_phase3_execution_efficiency_audit(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    sources = contract.get("sources") or {}
    outputs = contract.get("outputs") or {}
    governance = contract.get("governance") or {}
    tuning = _host_tuning(contract)

    execution_packet_payload = _load_json(_resolve_path(root_dir, sources["phase3_execution_packet_json"]))
    execution_rows = execution_packet_payload.get("rows")
    if not isinstance(execution_rows, list):
        raise SystemExit("Phase3 execution packet JSON must expose rows")

    wrapper = [str(item) for item in governance.get("long_run_wrapper") or []]
    required_device = str(governance.get("required_device") or "mps").strip()
    tuned_workers = int(tuning["tuned_workers"])
    tuned_eval_batch_size = int(tuning["tuned_eval_batch_size"])

    audit_rows: list[dict[str, Any]] = []
    for raw_row in execution_rows:
        if not isinstance(raw_row, dict):
            raise SystemExit("Phase3 execution packet rows must be mappings")
        variant_id = str(raw_row.get("variant_id") or "").strip().upper()
        if variant_id not in LANE_ORDER:
            raise SystemExit(f"Unexpected variant in execution packet: {variant_id!r}")
        command = _ensure_command_list(raw_row.get("launch_command_json"), f"{variant_id}.launch_command_json")
        current_workers = _command_flag_value(command, "--workers")
        current_eval_batch_size = _command_flag_value(command, "--eval_batch_size")
        dry_run_present = "--dry_run" in command
        caffeinate_present = _wrapped_with(command, wrapper)
        inefficiencies: list[str] = []
        if variant_id == "HOPS" and "FLOW" in json.dumps(raw_row, ensure_ascii=False):
            inefficiencies.append("public_naming_regressed_to_FLOW")
        if current_workers == "0":
            inefficiencies.append("workers_zero_disables_mps_host_tuning")
        elif current_workers != str(tuned_workers):
            inefficiencies.append(f"workers_not_host_tuned:{current_workers or 'unset'}")
        if current_eval_batch_size != str(tuned_eval_batch_size):
            inefficiencies.append(f"eval_batch_size_not_host_tuned:{current_eval_batch_size or 'unset'}")
        if dry_run_present:
            inefficiencies.append("dry_run_still_present")
        if not caffeinate_present:
            inefficiencies.append("missing_caffeinate_wrapper")
        if _command_flag_value(command, "--device") != required_device:
            inefficiencies.append("device_not_mps")
        mutated_command, mutations = _build_mutated_command(
            command,
            wrapper=wrapper,
            required_device=required_device,
            tuned_workers=tuned_workers,
            tuned_eval_batch_size=tuned_eval_batch_size,
        )
        notes = (
            "anchor_context_repair audit" if variant_id == "ASTRA" else "runtime_smoke_launch audit"
        )
        audit_rows.append(
            {
                "variant_id": variant_id,
                "internal_experiment_id": str(raw_row.get("internal_experiment_id") or "").strip().upper(),
                "package_kind": str(raw_row.get("package_kind") or "").strip(),
                "required_device": required_device,
                "current_workers": current_workers or "",
                "tuned_workers": tuned_workers,
                "current_eval_batch_size": current_eval_batch_size or "",
                "tuned_eval_batch_size": tuned_eval_batch_size,
                "dry_run_present": dry_run_present,
                "caffeinate_present": caffeinate_present,
                "runtime_health_gate_json": _runtime_health_gate_from_command(command),
                "audit_status": "fail" if inefficiencies else "pass",
                "inefficiencies_json": inefficiencies,
                "recommended_command_mutations_json": {
                    "mutations": mutations,
                    "recommended_command": mutated_command,
                },
                "notes": notes,
            }
        )

    audit_rows.sort(key=lambda row: LANE_ORDER.index(str(row["variant_id"])))
    outputs_payload = {
        "contract_path": str(resolved_contract_path.resolve()),
        "generated_at": utc_now_iso(),
        "host_tuning": tuning,
        "rows": audit_rows,
    }
    audit_csv = _resolve_path(root_dir, outputs["efficiency_audit_csv"])
    audit_json = _resolve_path(root_dir, outputs["efficiency_audit_json"])
    audit_md = _resolve_path(root_dir, outputs["efficiency_audit_md"])
    _write_csv(audit_csv, EFFICIENCY_AUDIT_FIELDS, audit_rows)
    _write_json(audit_json, outputs_payload)

    lines = [
        "# FULLER Phase3 Execution Efficiency Audit",
        "",
        f"- generated_at: `{outputs_payload['generated_at']}`",
        f"- contract_path: `{resolved_contract_path.resolve()}`",
        f"- required_device: `{required_device}`",
        f"- host_memory_gb: `{tuning['host_memory_gb']}`",
        f"- perf_core_count: `{tuning['perf_core_count']}`",
        f"- auto_workers_mps: `{tuning['auto_workers_mps']}`",
        f"- tuned_workers: `{tuned_workers}`",
        f"- tuned_eval_batch_size: `{tuned_eval_batch_size}`",
    ]
    if tuning["tuning_note"]:
        lines.append(f"- tuning_note: `{tuning['tuning_note']}`")
    lines.extend(
        [
            "",
            "## Lane Audit",
            "",
            "| Variant | Package | Current Workers | Tuned Workers | Current Eval Batch | Tuned Eval Batch | Dry Run | Caffeinate | Audit Status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in audit_rows:
        lines.append(
            f"| `{row['variant_id']}` | `{row['package_kind']}` | `{row['current_workers'] or 'unset'}` | "
            f"`{row['tuned_workers']}` | `{row['current_eval_batch_size'] or 'unset'}` | "
            f"`{row['tuned_eval_batch_size']}` | `{row['dry_run_present']}` | "
            f"`{row['caffeinate_present']}` | `{row['audit_status']}` |"
        )
        lines.append(
            f"- `{row['variant_id']}` inefficiencies: `{row['inefficiencies_json']}`"
        )
        lines.append(
            f"- `{row['variant_id']}` recommended mutations: `{json.dumps(row['recommended_command_mutations_json'], ensure_ascii=False)}`"
        )
    _write_text(audit_md, "\n".join(lines) + "\n")

    return {
        "status": "pass",
        "variant_ids": [str(row["variant_id"]) for row in audit_rows],
        "audit_csv": str(audit_csv.resolve()),
        "audit_json": str(audit_json.resolve()),
        "audit_md": str(audit_md.resolve()),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the current FULLER phase3 execution-efficiency audit surface.",
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    payload = build_phase3_execution_efficiency_audit(Path(args.contract))
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
