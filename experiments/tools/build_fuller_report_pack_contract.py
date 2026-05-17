#!/usr/bin/env python3
"""Build the unified FULLER report-pack contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import (
        DEFAULT_CONTRACT,
        REPORT_CONTRACT_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_report_contract_rows,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        DEFAULT_CONTRACT,
        REPORT_CONTRACT_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_report_contract_rows,
        load_program_context,
    )


def _report_note(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# FULLER Report Pack Contract",
        "",
        "Date: `2026-04-22`",
        "Status: `report_pack_contract_current`",
        "",
        "## Deliverables",
        "",
    ]
    lines.extend(
        f"- `{row['deliverable_id']}` sources=`{row['source_family_ids_json']}` gate=`{row['readiness_gate']}`"
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Constraint",
            "",
            "No deliverable in this surface may cite an undeclared experiment family or undeclared artifact class.",
            "Engineering smoke families may appear in governance/status deliverables, but not as benchmark/proxy evidence sources.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_fuller_report_pack_contract(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    rows = build_report_contract_rows(ctx)
    report_csv = _resolve_path(root_dir, outputs["report_contract_csv"])
    report_json = _resolve_path(root_dir, outputs["report_contract_json"])
    report_md = _resolve_path(root_dir, outputs["report_contract_md"])
    _write_csv(report_csv, REPORT_CONTRACT_FIELDS, rows)
    _write_json(
        report_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        },
    )
    _write_text(report_md, _report_note(rows))
    return {
        "status": "pass",
        "row_count": len(rows),
        "report_contract_csv": str(report_csv.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FULLER report-pack contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = build_fuller_report_pack_contract(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
