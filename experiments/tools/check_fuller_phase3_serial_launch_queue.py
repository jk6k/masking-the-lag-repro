#!/usr/bin/env python3
"""Validate the FULLER phase3 serialized single-MPS launch queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .build_fuller_phase3_serial_launch_queue import SERIAL_QUEUE_FIELDS
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
    from build_fuller_phase3_serial_launch_queue import SERIAL_QUEUE_FIELDS  # type: ignore
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


def check_phase3_serial_launch_queue(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    outputs = contract.get("outputs") or {}
    host_tuning = contract.get("host_tuning") or {}
    governance = contract.get("governance") or {}

    queue_csv = _resolve_path(root_dir, outputs["serial_launch_queue_csv"])
    queue_json = _resolve_path(root_dir, outputs["serial_launch_queue_json"])
    plan_md = _resolve_path(root_dir, outputs["serial_launch_plan_md"])
    for path in (queue_csv, queue_json, plan_md):
        if not path.exists():
            raise SystemExit(f"Missing serial launch queue output: {path}")

    queue_rows = _load_csv_rows(queue_csv)
    queue_payload = _load_json(queue_json)
    json_rows = queue_payload.get("rows")
    if not isinstance(json_rows, list):
        raise SystemExit("Serial launch queue JSON must expose rows")
    if len(queue_rows) != 7 or len(json_rows) != 7:
        raise SystemExit("Serial launch queue must contain exactly 7 rows")
    variant_ids = [str(row.get("variant_id") or "").strip().upper() for row in queue_rows]
    if variant_ids != LANE_ORDER:
        raise SystemExit(f"Serial queue order drifted from the fixed lane order: {variant_ids}")
    if "FLOW" in variant_ids:
        raise SystemExit("HOPS naming regressed to FLOW inside the serial queue")

    tuned_workers = str(host_tuning.get("tuned_workers") or "")
    tuned_eval_batch_size = str(host_tuning.get("tuned_eval_batch_size") or "")
    wrapper = [str(item) for item in governance.get("long_run_wrapper") or []]
    required_device = str(governance.get("required_device") or "mps").strip()

    for row in queue_rows:
        if list(row.keys()) != SERIAL_QUEUE_FIELDS:
            raise SystemExit("Serial queue CSV fields drifted from the declared schema")
        variant_id = str(row.get("variant_id") or "").strip().upper()
        command = _parse_json_list(row.get("command_json"), f"{variant_id}.command_json")
        checker_command = _parse_json_list(row.get("checker_command_json"), f"{variant_id}.checker_command_json")
        required_outputs = _parse_json_list(row.get("required_outputs_json"), f"{variant_id}.required_outputs_json")
        if not command[: len(wrapper)] == wrapper:
            raise SystemExit(f"{variant_id} launch command must be wrapped by {' '.join(wrapper)}")
        if "--dry_run" in command:
            raise SystemExit(f"{variant_id} launch command still contains --dry_run")
        if "--resume" not in command:
            raise SystemExit(f"{variant_id} launch command must include --resume")
        if "--workers" not in command or tuned_workers not in command:
            raise SystemExit(f"{variant_id} launch command must pin workers={tuned_workers}")
        if "--eval_batch_size" not in command or tuned_eval_batch_size not in command:
            raise SystemExit(f"{variant_id} launch command must pin eval_batch_size={tuned_eval_batch_size}")
        if "--device" not in command or required_device not in command:
            raise SystemExit(f"{variant_id} launch command must pin device={required_device}")
        if not checker_command:
            raise SystemExit(f"{variant_id} queue row is missing checker_command_json")
        if not required_outputs:
            raise SystemExit(f"{variant_id} queue row is missing required_outputs_json")
        if str(row.get("stop_on_failure") or "").strip().lower() != "true":
            raise SystemExit(f"{variant_id} queue row must keep stop_on_failure=true")
        if variant_id == "HOPS" and "FLOW" in json.dumps(row, ensure_ascii=False):
            raise SystemExit("HOPS queue row must not expose FLOW naming")

    return {
        "status": "pass",
        "variant_ids": variant_ids,
        "queue_csv": str(queue_csv.resolve()),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the FULLER phase3 serialized single-MPS launch queue.",
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    payload = check_phase3_serial_launch_queue(Path(args.contract))
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
