#!/usr/bin/env python3
"""Build the unified FULLER phase4 intake contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import (
        DEFAULT_CONTRACT,
        PHASE4_INTAKE_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_phase4_intake_rows,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        DEFAULT_CONTRACT,
        PHASE4_INTAKE_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_phase4_intake_rows,
        load_program_context,
    )


def _phase4_note(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FULLER Phase4 Intake Contract",
        "",
        "Date: `2026-04-23`",
        "Status: `phase4_intake_contract_current`",
        "",
        "## Decision",
        "",
        "Phase4 intake is now family-driven, not lane-generic. Engineering smoke families remain explicitly",
        "phase4-ineligible, while analysis-grade replay remains the only lane family that can become claim-tier evidence.",
        "Within that family, ASTRA is the canonical paired baseline and MESO/HOPS/DET/SPARSE/FULLER replay against it via full-manifest quantized-only rows.",
        "PHY is now tracked under `realism_calibration_support`, which remains support-only and outside the paper's main claim lane family.",
        "",
        "## Families",
        "",
    ]
    lines.extend(
        f"- `{row['experiment_family_id']}` intake_status=`{row['intake_status']}` phase4_eligible=`{row['phase4_eligible']}` claim_tier=`{row['claim_tier']}`"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def build_fuller_phase4_intake_contract(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    rows = build_phase4_intake_rows(ctx)
    phase4_csv = _resolve_path(root_dir, outputs["phase4_intake_contract_csv"])
    phase4_json = _resolve_path(root_dir, outputs["phase4_intake_contract_json"])
    phase4_md = _resolve_path(root_dir, outputs["phase4_intake_contract_md"])
    _write_csv(phase4_csv, PHASE4_INTAKE_FIELDS, rows)
    _write_json(
        phase4_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        },
    )
    _write_text(phase4_md, _phase4_note(rows))
    return {
        "status": "pass",
        "row_count": len(rows),
        "phase4_intake_contract_csv": str(phase4_csv.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FULLER phase4 intake contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = build_fuller_phase4_intake_contract(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
