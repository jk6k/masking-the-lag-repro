#!/usr/bin/env python3
"""Materialize the current FULLER experiment execution plan from the unified program."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import (
        DEFAULT_CONTRACT,
        EXECUTION_PLAN_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_execution_plan_rows,
        materialize_engineering_smoke_slice_manifests,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        DEFAULT_CONTRACT,
        EXECUTION_PLAN_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_execution_plan_rows,
        materialize_engineering_smoke_slice_manifests,
        load_program_context,
    )


def _execution_note(
    rows: list[dict[str, Any]],
    queue_status: dict[str, Any],
    *,
    slice_manifests: dict[str, str],
) -> str:
    lines = [
        "# FULLER Experiment Execution Plan",
        "",
        "Date: `2026-04-22`",
        "Status: `phase3_5_execution_plan_current`",
        "",
        "## Current Plan",
        "",
        "- `anchor_validation` now starts the active engineering-smoke queue as the ASTRA paired `512`-sample baseline-cache producer.",
        "- `lane_isolation_runtime_smoke` now materializes `MESO/HOPS/DET/SPARSE/PHY/FULLER` as quantized-only engineering-smoke rows.",
        "- analysis-grade replay is now materialized separately as a redesigned current family: ASTRA is the canonical paired full-manifest baseline, while MESO/HOPS/DET/SPARSE/FULLER are staged as quantized-only full-manifest replays behind it.",
        "- PHY is no longer queued inside the main claim-tier analysis-grade family and instead routes to `realism_calibration_support`.",
        "",
        "## Deterministic Runtime-Smoke Slices",
        "",
        f"- `512`: `{slice_manifests.get('512', '')}`",
        f"- `256`: `{slice_manifests.get('256', '')}`",
        "",
        "## Legacy Queue Replacement",
        "",
        f"- legacy_queue_state: `{queue_status.get('queue_state')}`",
        f"- legacy_current_lane: `{queue_status.get('current_lane')}`",
        f"- legacy_last_completed_lane: `{queue_status.get('last_completed_lane')}`",
        f"- legacy_halt_reason: `{queue_status.get('halt_reason')}`",
        "- active execution source of truth is now the v2 engineering-smoke execution plan below.",
        "",
        "## Materialized Steps",
        "",
    ]
    lines.extend(
        f"- `{row['step_id']}` lane=`{row['lane_id']}` pass_mode=`{row['pass_mode']}` sample_budget=`{row['sample_budget']}`"
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Replacement Rule",
            "",
            "- the legacy runtime-smoke queue is archived and no longer the active execution source of truth",
            "- active runs must start from the v2 engineering-smoke ASTRA baseline-cache producer",
        ]
    )
    return "\n".join(lines) + "\n"


def materialize_fuller_experiment_execution_plan(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    slice_manifests = materialize_engineering_smoke_slice_manifests(ctx, root_dir=root_dir)
    rows = build_execution_plan_rows(ctx)
    execution_plan_csv = _resolve_path(root_dir, outputs["execution_plan_csv"])
    execution_plan_json = _resolve_path(root_dir, outputs["execution_plan_json"])
    execution_plan_md = _resolve_path(root_dir, outputs["execution_plan_md"])
    _write_csv(execution_plan_csv, EXECUTION_PLAN_FIELDS, rows)
    _write_json(
        execution_plan_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "slice_manifests": slice_manifests,
            "rows": rows,
        },
    )
    _write_text(
        execution_plan_md,
        _execution_note(rows, ctx.phase3_queue_status, slice_manifests=slice_manifests),
    )
    return {
        "status": "pass",
        "row_count": len(rows),
        "execution_plan_csv": str(execution_plan_csv.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the FULLER experiment execution plan.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = materialize_fuller_experiment_execution_plan(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
