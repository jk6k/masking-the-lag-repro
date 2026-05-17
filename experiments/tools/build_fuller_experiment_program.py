#!/usr/bin/env python3
"""Build the active FULLER experiment program matrix and data contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .fuller_experiment_program_common import (
        DATA_CONTRACT_FIELDS,
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        EXPERIMENT_MATRIX_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_data_contract_rows,
        build_experiment_matrix_rows,
        load_program_context,
    )
except ImportError:
    from fuller_experiment_program_common import (  # type: ignore
        DATA_CONTRACT_FIELDS,
        DEFAULT_CONTRACT,
        EXPERIMENT_FAMILY_ORDER,
        EXPERIMENT_MATRIX_FIELDS,
        ROOT,
        _resolve_path,
        _write_csv,
        _write_json,
        _write_text,
        build_data_contract_rows,
        build_experiment_matrix_rows,
        load_program_context,
    )


def _program_note(
    *,
    queue_state: str,
    current_lane: str,
    completed_lanes: list[str],
    pending_lanes: list[str],
) -> str:
    lines = [
        "# FULLER Experiment Program Refactor Note",
        "",
        "Date: `2026-04-22`",
        "Status: `phase3_5_completed`",
        "",
        "## Decision",
        "",
        "The active experiment surface is now expressed through one current `fuller_experiment_program` contract.",
        "The current `20260421/20260422` full-dataset runtime-smoke queue is frozen as a legacy current-run surface and replaced by a v2 engineering-smoke program where ASTRA produces the baseline cache and the remaining lanes run quantized-only against deterministic 512/256-sample slices.",
        "The analysis-grade family is now also redesigned around ASTRA as the canonical paired full-manifest baseline, with MESO/HOPS/DET/SPARSE/FULLER reusing that baseline through full-manifest quantized-only replays and PHY redirected into a realism/calibration support family.",
        "",
        "## Program Families",
        "",
    ]
    lines.extend(f"- `{family_id}`" for family_id in EXPERIMENT_FAMILY_ORDER)
    lines.extend(
        [
            "",
            "## Legacy Queue Bridge",
            "",
            f"- queue_state: `{queue_state}`",
            f"- current_lane: `{current_lane}`",
            f"- completed_lanes: `{completed_lanes}`",
            f"- pending_lanes: `{pending_lanes}`",
            "- replacement_policy: `archive_and_replace`",
            "",
            "## Current Meaning",
            "",
            "- `anchor_validation` is the baseline-cache producer for engineering smoke.",
            "- `lane_isolation_runtime_smoke` is now a low-cost engineering-smoke family, not a full-dataset paired replay family.",
            "- `analysis_grade_replay` now means ASTRA paired canonical baseline plus mainline quantized-only full-manifest replays for `MESO/HOPS/DET/SPARSE/FULLER`.",
            "- `realism_calibration_support` holds `PHY` outside the paper's main claim-tier replay family.",
            "- `noise/scaling/device/holdout` are explicit support or audit families rather than sidecar scripts.",
            "- `report_pack` is part of the experiment program and may not cite engineering smoke as benchmark/proxy evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def _phase35_status() -> str:
    lines = [
        "# Phase3.5 Status",
        "",
        "Date: `2026-04-22`",
        "Phase: `Experiment Program Refactor`",
        "Status: `completed`",
        "",
        "## Delivered Artifacts",
        "",
        "- `configs/fuller_experiment_program_contract_20260422.yaml`",
        "- `experiments/results/report_data/fuller_experiment_matrix_20260422.csv`",
        "- `experiments/results/report_data/fuller_experiment_data_contract_20260422.csv`",
        "- `experiments/results/report_data/fuller_experiment_program_manifest_20260422.json`",
        "- `docs/reports/20260422_fuller_experiment_program_refactor_note.md`",
        "- `experiments/results/report_data/fuller_experiment_execution_plan_20260422.csv`",
        "- `experiments/results/report_data/fuller_phase4_intake_contract_20260422.csv`",
        "- `experiments/results/report_data/fuller_report_pack_contract_20260422.csv`",
        "- `experiments/results/report_data/20260422_fuller_experiment_program_v2_slices/runtime_smoke_fast_seed0_512.csv`",
        "- `experiments/results/report_data/20260422_fuller_experiment_program_v2_slices/runtime_smoke_fast_seed0_256.csv`",
        "- `experiments/results/report_data/fuller_runtime_lane_governance_status_matrix_20260423.csv`",
        "- `experiments/results/report_data/fuller_runtime_lane_governance_status_matrix_20260423.json`",
        "- `docs/reports/20260423_fuller_runtime_lane_governance_status_matrix.md`",
        "- `experiments/results/report_data/fuller_noise_robustness_wrapper_manifest_20260423.json`",
        "- `experiments/results/report_data/fuller_noise_robustness_materialization_audit_20260423.csv`",
        "- `experiments/results/report_data/fuller_noise_robustness_materialization_audit_20260423.json`",
        "- `docs/reports/20260423_fuller_noise_robustness_materialization_audit.md`",
        "- `experiments/results/report_data/fuller_analysis_grade_replay_gate_matrix_20260423.csv`",
        "- `experiments/results/report_data/fuller_analysis_grade_replay_gate_matrix_20260423.json`",
        "- `experiments/results/report_data/fuller_analysis_grade_replay_materialization_plan_20260423.csv`",
        "- `experiments/results/report_data/fuller_analysis_grade_replay_materialization_plan_20260423.json`",
        "- `docs/reports/20260423_fuller_analysis_grade_replay_gate.md`",
        "- `docs/reports/20260423_fuller_analysis_grade_replay_materialization_plan.md`",
        "- `experiments/results/report_data/fuller_phase4_current_intake_surface_20260423.csv`",
        "- `experiments/results/report_data/fuller_phase4_current_intake_surface_20260423.json`",
        "- `docs/reports/20260423_fuller_phase4_current_intake_surface.md`",
        "- `experiments/results/report_data/legacy_runtime_smoke_bridge_20260422/legacy_runtime_smoke_bridge_manifest.json`",
        "- `docs/reports/20260422_legacy_runtime_smoke_bridge_note.md`",
        "",
        "## Exit Criteria Check",
        "",
        "- one active fuller experiment-program source of truth: `pass`",
        "- runtime-smoke now means low-cost engineering validation rather than full-dataset paired replay: `pass`",
        "- analysis-grade remains the only claim-eligible full-dataset replay family and now uses canonical-baseline reuse for mainline lanes: `pass`",
        "- PHY is removed from the main claim-tier replay family and modeled under a realism/calibration support family: `pass`",
        "- current legacy queue has been archived into an explicit readable bridge surface: `pass`",
        "- no governance rule was relaxed during the refactor: `pass`",
        "",
        "## Next Phase",
        "",
        "The v2 engineering-smoke program has now completed a clean pass across",
        "`ASTRA / MESO / HOPS / DET / SPARSE / PHY / FULLER`, and the current",
        "governance/status surface is captured in",
        "`20260423_fuller_runtime_lane_governance_status_matrix.md`.",
        "`noise_robustness` has also been materialized into a current wrapper/audit",
        "surface, but the present legacy-compatible support bundle is still heavy",
        "(`57` profiles / `81` accuracy runs) and remains intentionally unstarted.",
        "`analysis_grade_replay` is now materialized as a redesigned current",
        "family: `ASTRA` remains the canonical paired full-manifest baseline,",
        "while `MESO/HOPS/DET/SPARSE/FULLER` are retargeted to full-manifest",
        "`quantized_only` replays that reuse ASTRA baseline reference rows.",
        "`PHY` is no longer queued in the main claim-tier family and is instead",
        "redirected into `realism_calibration_support`.",
        "Phase4 now has a current intake/evidence surface in",
        "`20260423_fuller_phase4_current_intake_surface.md`. Engineering-smoke",
        "outputs remain captured for governance/status only and closed to claim-tier",
        "promotion. The active next gated step is completion of the active ASTRA",
        "canonical baseline run followed by admission of the redesigned mainline",
        "replay queue; support families, including PHY realism/calibration, remain",
        "outside the main claim-tier lane family unless explicitly scoped later.",
    ]
    return "\n".join(lines) + "\n"


def build_fuller_experiment_program(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    root_dir: Path = ROOT,
) -> dict[str, Any]:
    ctx = load_program_context(contract_path, root_dir=root_dir)
    outputs = ctx.contract.get("outputs") or {}
    coordination = ctx.contract.get("coordination") or {}

    experiment_matrix_rows = build_experiment_matrix_rows(ctx)
    data_contract_rows = build_data_contract_rows(ctx)

    experiment_matrix_csv = _resolve_path(root_dir, outputs["experiment_matrix_csv"])
    experiment_matrix_json = _resolve_path(root_dir, outputs["experiment_matrix_json"])
    data_contract_csv = _resolve_path(root_dir, outputs["data_contract_csv"])
    data_contract_json = _resolve_path(root_dir, outputs["data_contract_json"])
    program_manifest_json = _resolve_path(root_dir, outputs["program_manifest_json"])
    program_refactor_note_md = _resolve_path(root_dir, outputs["program_refactor_note_md"])
    phase35_status_md = _resolve_path(root_dir, coordination["phase35_status_md"])

    _write_csv(experiment_matrix_csv, EXPERIMENT_MATRIX_FIELDS, experiment_matrix_rows)
    _write_json(
        experiment_matrix_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": experiment_matrix_rows,
        },
    )
    _write_csv(data_contract_csv, DATA_CONTRACT_FIELDS, data_contract_rows)
    _write_json(
        data_contract_json,
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rows": data_contract_rows,
        },
    )
    queue_status = ctx.phase3_queue_status
    _write_text(
        program_refactor_note_md,
        _program_note(
            queue_state=str(queue_status.get("queue_state") or ""),
            current_lane=str(queue_status.get("current_lane") or ""),
            completed_lanes=[str(item) for item in queue_status.get("completed_lanes") or []],
            pending_lanes=[str(item) for item in queue_status.get("pending_lanes") or []],
        ),
    )
    _write_text(phase35_status_md, _phase35_status())
    _write_json(
        program_manifest_json,
        {
            "contract_path": str(ctx.contract_path.resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "phase35_status": "completed",
            "legacy_queue_state": queue_status.get("queue_state"),
            "legacy_current_lane": queue_status.get("current_lane"),
            "legacy_last_completed_lane": queue_status.get("last_completed_lane"),
            "legacy_halt_reason": queue_status.get("halt_reason"),
            "legacy_replacement_policy": "archive_and_replace",
            "legacy_bridge_manifest_json": str(
                _resolve_path(root_dir, (ctx.contract.get("outputs") or {})["legacy_runtime_smoke_bridge_manifest_json"])
            ),
            "source_artifacts": {
                key: str(_resolve_path(root_dir, value))
                for key, value in (ctx.contract.get("sources") or {}).items()
            },
            "generated_outputs": {
                "experiment_matrix_csv": str(experiment_matrix_csv.resolve()),
                "data_contract_csv": str(data_contract_csv.resolve()),
                "program_refactor_note_md": str(program_refactor_note_md.resolve()),
                "phase35_status_md": str(phase35_status_md.resolve()),
            },
        },
    )
    return {
        "status": "pass",
        "experiment_family_ids": [row["experiment_family_id"] for row in experiment_matrix_rows],
        "experiment_matrix_csv": str(experiment_matrix_csv.resolve()),
        "data_contract_csv": str(data_contract_csv.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the active FULLER experiment program.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    payload = build_fuller_experiment_program(args.contract)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
