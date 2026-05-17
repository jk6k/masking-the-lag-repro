#!/usr/bin/env python3
"""Build the FULLER phase3 serialized single-MPS launch queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .fuller_phase3_launch_common import (
        DEFAULT_CONTRACT,
        LANE_ORDER,
        ROOT,
        _build_mutated_command,
        _ensure_command_list,
        _lane_status_output_paths,
        _load_json,
        _load_yaml,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        utc_now_iso,
    )
except ImportError:
    from fuller_phase3_launch_common import (  # type: ignore
        DEFAULT_CONTRACT,
        LANE_ORDER,
        ROOT,
        _build_mutated_command,
        _ensure_command_list,
        _lane_status_output_paths,
        _load_json,
        _load_yaml,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        utc_now_iso,
    )


SERIAL_QUEUE_FIELDS = [
    "queue_index",
    "variant_id",
    "internal_experiment_id",
    "package_kind",
    "command_json",
    "progress_manifest_json",
    "checker_kind",
    "checker_command_json",
    "required_outputs_json",
    "stop_on_failure",
    "resume_enabled",
    "queue_status",
    "notes",
]


def build_phase3_serial_launch_queue(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    resolved_contract_path = contract_path if contract_path.is_absolute() else root_dir / contract_path
    contract = _load_yaml(resolved_contract_path)
    sources = contract.get("sources") or {}
    outputs = contract.get("outputs") or {}
    governance = contract.get("governance") or {}
    host_tuning = contract.get("host_tuning") or {}

    execution_packet_payload = _load_json(_resolve_path(root_dir, sources["phase3_execution_packet_json"]))
    execution_rows = execution_packet_payload.get("rows")
    if not isinstance(execution_rows, list):
        raise SystemExit("Phase3 execution packet JSON must expose rows")
    efficiency_audit_payload = _load_json(_resolve_path(root_dir, outputs["efficiency_audit_json"]))
    audit_rows = efficiency_audit_payload.get("rows")
    if not isinstance(audit_rows, list):
        raise SystemExit("Efficiency audit JSON must expose rows")
    execution_lookup = {
        str(row.get("variant_id") or "").strip().upper(): row
        for row in execution_rows
        if isinstance(row, dict)
    }
    audit_lookup = {
        str(row.get("variant_id") or "").strip().upper(): row
        for row in audit_rows
        if isinstance(row, dict)
    }
    wrapper = [str(item) for item in governance.get("long_run_wrapper") or []]
    required_device = str(governance.get("required_device") or "mps").strip()
    tuned_workers = int(host_tuning.get("tuned_workers") or 5)
    tuned_eval_batch_size = int(host_tuning.get("tuned_eval_batch_size") or 64)

    serial_queue_rows: list[dict[str, Any]] = []
    checker_python = str(root_dir / ".venv311-mps" / "bin" / "python")
    checker_tool = str(root_dir / "experiments" / "tools" / "check_fuller_phase3_serial_launch_status.py")
    for queue_index, variant_id in enumerate(LANE_ORDER, start=1):
        execution_row = execution_lookup.get(variant_id)
        audit_row = audit_lookup.get(variant_id)
        if execution_row is None or audit_row is None:
            raise SystemExit(f"Missing execution/audit row for {variant_id}")
        raw_command = _ensure_command_list(audit_row["recommended_command_mutations_json"]["recommended_command"], f"{variant_id}.recommended_command")
        command, _ = _build_mutated_command(
            raw_command,
            wrapper=wrapper,
            required_device=required_device,
            tuned_workers=tuned_workers,
            tuned_eval_batch_size=tuned_eval_batch_size,
        )
        lane_status_json, lane_status_md = _lane_status_output_paths(root_dir, variant_id)
        checker_command = [
            checker_python,
            checker_tool,
            "--contract",
            str(resolved_contract_path.resolve()),
            "--variant_id",
            variant_id,
            "--output_json",
            str(lane_status_json),
            "--output_md",
            str(lane_status_md),
        ]
        notes = "single-MPS serialized lane; stop queue immediately on failure"
        if variant_id == "ASTRA":
            notes = "single-MPS serialized ASTRA context-repair lane; stop queue immediately on failure"
        serial_queue_rows.append(
            {
                "queue_index": queue_index,
                "variant_id": variant_id,
                "internal_experiment_id": str(execution_row.get("internal_experiment_id") or "").strip().upper(),
                "package_kind": str(execution_row.get("package_kind") or "").strip(),
                "command_json": command,
                "progress_manifest_json": str(execution_row.get("progress_manifest_json") or ""),
                "checker_kind": "fuller_phase3_serial_launch_status",
                "checker_command_json": checker_command,
                "required_outputs_json": list(execution_row.get("required_outputs_json") or []),
                "stop_on_failure": bool(governance.get("stop_on_failure")),
                "resume_enabled": True,
                "queue_status": "ready",
                "notes": notes,
            }
        )

    queue_csv = _resolve_path(root_dir, outputs["serial_launch_queue_csv"])
    queue_json = _resolve_path(root_dir, outputs["serial_launch_queue_json"])
    plan_md = _resolve_path(root_dir, outputs["serial_launch_plan_md"])
    policy_md = _resolve_path(root_dir, outputs["launch_policy_note_md"])
    _write_csv(queue_csv, SERIAL_QUEUE_FIELDS, serial_queue_rows)
    _write_json(
        queue_json,
        {
            "contract_path": str(resolved_contract_path.resolve()),
            "generated_at": utc_now_iso(),
            "rows": serial_queue_rows,
        },
    )

    lines = [
        "# FULLER Phase3 Serial Launch Plan",
        "",
        f"- generated_at: `{utc_now_iso()}`",
        f"- required_device: `{required_device}`",
        f"- long_run_wrapper: `{wrapper}`",
        f"- tuned_workers: `{tuned_workers}`",
        f"- tuned_eval_batch_size: `{tuned_eval_batch_size}`",
        f"- fail_fast: `{governance.get('stop_on_failure')}`",
        "- resume_mode: `all lanes carry --resume; restart with --start_from <variant_id>`",
        "",
        "## Queue Order",
        "",
    ]
    for row in serial_queue_rows:
        lines.append(
            f"{row['queue_index']}. `{row['variant_id']}` "
            f"package=`{row['package_kind']}` checker=`{row['checker_kind']}`"
        )
    _write_text(plan_md, "\n".join(lines) + "\n")
    policy_lines = [
        "# FULLER Phase3 Launch Policy",
        "",
        f"- execution_tier: `runtime_smoke`",
        f"- required_device: `{required_device}`",
        f"- long_run_wrapper: `{wrapper}`",
        f"- tuned_workers: `{tuned_workers}`",
        f"- tuned_eval_batch_size: `{tuned_eval_batch_size}`",
        f"- serialized_queue: `true`",
        f"- stop_on_failure: `{governance.get('stop_on_failure')}`",
        f"- analysis_grade_enabled: `{governance.get('analysis_grade_enabled')}`",
        f"- benchmark_claim_ready: `{governance.get('benchmark_claim_ready')}`",
        "- resume_mode: `queue relaunches from the chosen lane with --start_from while each lane keeps --resume`",
        "",
        "## Queue Policy",
        "",
        "- `ASTRA` runs first as the context-repair lane.",
        "- `MESO/HOPS/DET/SPARSE/PHY/FULLER` then run in fixed serialized order on one MPS device.",
        "- Every lane keeps `--resume`, so the queue can be interrupted and restarted from any lane without rebuilding outputs.",
        "- Queue execution halts immediately on `stall_timeout`, `pathological_slow`, `command_error`, or `checker_fail`.",
    ]
    _write_text(policy_md, "\n".join(policy_lines) + "\n")

    return {
        "status": "pass",
        "variant_ids": [str(row["variant_id"]) for row in serial_queue_rows],
        "queue_csv": str(queue_csv.resolve()),
        "queue_json": str(queue_json.resolve()),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the FULLER phase3 serialized single-MPS launch queue.",
    )
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    payload = build_phase3_serial_launch_queue(Path(args.contract))
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
