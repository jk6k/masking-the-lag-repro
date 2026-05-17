#!/usr/bin/env python3
"""Validate the current FULLER phase3 execution-efficiency audit surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .build_fuller_phase3_execution_efficiency_audit import (
        EFFICIENCY_AUDIT_FIELDS,
        build_phase3_execution_efficiency_audit,
    )
    from .fuller_phase3_launch_common import (
        DEFAULT_CONTRACT,
        LANE_ORDER,
        ROOT,
        _load_csv_rows,
        _load_json,
        _load_yaml,
        _parse_json_list,
        _resolve_path,
    )
except ImportError:
    from build_fuller_phase3_execution_efficiency_audit import (  # type: ignore
        EFFICIENCY_AUDIT_FIELDS,
        build_phase3_execution_efficiency_audit,
    )
    from fuller_phase3_launch_common import (  # type: ignore
        DEFAULT_CONTRACT,
        LANE_ORDER,
        ROOT,
        _load_csv_rows,
        _load_json,
        _load_yaml,
        _parse_json_list,
        _resolve_path,
    )


def check_phase3_execution_efficiency_audit(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    outputs = contract.get("outputs") or {}
    sources = contract.get("sources") or {}
    governance = contract.get("governance") or {}
    host_tuning = contract.get("host_tuning") or {}

    audit_csv = _resolve_path(root_dir, outputs["efficiency_audit_csv"])
    audit_json = _resolve_path(root_dir, outputs["efficiency_audit_json"])
    audit_md = _resolve_path(root_dir, outputs["efficiency_audit_md"])
    execution_packet_json = _resolve_path(root_dir, sources["phase3_execution_packet_json"])
    if not execution_packet_json.exists():
        raise SystemExit("Phase3 execution packet JSON missing before efficiency audit check")
    for path in (audit_csv, audit_json, audit_md):
        if not path.exists():
            raise SystemExit(f"Missing efficiency audit output: {path}")

    audit_rows = _load_csv_rows(audit_csv)
    audit_payload = _load_json(audit_json)
    json_rows = audit_payload.get("rows")
    if not isinstance(json_rows, list):
        raise SystemExit("Efficiency audit JSON must expose rows")
    if len(audit_rows) != 7 or len(json_rows) != 7:
        raise SystemExit("Efficiency audit must contain exactly 7 rows")
    variant_ids = [str(row.get("variant_id") or "").strip().upper() for row in audit_rows]
    if variant_ids != LANE_ORDER:
        raise SystemExit(f"Efficiency audit variants drift from lane order: {variant_ids}")

    tuned_workers = str(host_tuning.get("tuned_workers") or "")
    tuned_eval_batch_size = str(host_tuning.get("tuned_eval_batch_size") or "")
    required_device = str(governance.get("required_device") or "mps").strip()
    wrapper = [str(item) for item in governance.get("long_run_wrapper") or []]
    for row in audit_rows:
        if list(row.keys()) != EFFICIENCY_AUDIT_FIELDS:
            raise SystemExit("Efficiency audit CSV fields drifted from the declared schema")
        variant_id = str(row.get("variant_id") or "").strip().upper()
        if str(row.get("required_device") or "") != required_device:
            raise SystemExit(f"{variant_id} efficiency audit drifted from required device {required_device}")
        if str(row.get("tuned_workers") or "") != tuned_workers:
            raise SystemExit(f"{variant_id} tuned_workers drifted from contract")
        if str(row.get("tuned_eval_batch_size") or "") != tuned_eval_batch_size:
            raise SystemExit(f"{variant_id} tuned_eval_batch_size drifted from contract")

    for row in audit_rows:
        variant_id = str(row.get("variant_id") or "").strip().upper()
        inefficiencies = _parse_json_list(row.get("inefficiencies_json"), f"{variant_id}.inefficiencies_json")
        if "dry_run_still_present" not in inefficiencies:
            raise SystemExit(f"{variant_id} audit must explicitly call out --dry_run when still present")
        if variant_id != "ASTRA" and "workers_zero_disables_mps_host_tuning" not in inefficiencies:
            raise SystemExit(f"{variant_id} audit must explicitly call out workers=0 as an efficiency gap")
        recommended = json.loads(str(row.get("recommended_command_mutations_json") or "{}"))
        if not isinstance(recommended, dict):
            raise SystemExit(f"{variant_id} recommended mutations must stay a mapping")
        recommended_command = recommended.get("recommended_command") or []
        if not isinstance(recommended_command, list):
            raise SystemExit(f"{variant_id} recommended command must stay a list")
        if "--dry_run" in recommended_command:
            raise SystemExit(f"{variant_id} recommended command still contains --dry_run")
        if "--resume" not in recommended_command:
            raise SystemExit(f"{variant_id} recommended command must add --resume")
        if "--workers" not in recommended_command or str(tuned_workers) not in recommended_command:
            raise SystemExit(f"{variant_id} recommended command must pin workers={tuned_workers}")
        if "--eval_batch_size" not in recommended_command or str(tuned_eval_batch_size) not in recommended_command:
            raise SystemExit(f"{variant_id} recommended command must pin eval_batch_size={tuned_eval_batch_size}")
        if "--device" not in recommended_command or required_device not in recommended_command:
            raise SystemExit(f"{variant_id} recommended command must pin device={required_device}")
        if recommended_command[: len(wrapper)] != wrapper:
            raise SystemExit(f"{variant_id} recommended command must be wrapped by {' '.join(wrapper)}")

    return {
        "status": "pass",
        "variant_ids": variant_ids,
        "audit_csv": str(audit_csv.resolve()),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the current FULLER phase3 efficiency audit surface.",
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument(
        "--rebuild_if_missing",
        action="store_true",
        help="Build the efficiency audit first if it has not yet been generated.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    contract_path = Path(args.contract)
    if args.rebuild_if_missing:
        build_phase3_execution_efficiency_audit(contract_path)
    payload = check_phase3_execution_efficiency_audit(contract_path)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
