#!/usr/bin/env python3
"""Build the R12c cross-workload policy-transfer artifact.

This generator does not launch a new accelerator-backed model run. It
consolidates governed MPS accuracy rows that already exercised the relevant
policy families across BERT/GLUE and MobileViT-S/ImageNet, then records whether
the transferred policy stays inside the 1 pp accuracy budget or becomes an
explicit transfer boundary.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260514_r12_reinforcement"
DATE = "2026-05-14"
PIVOT_TAG = "20260513_tetc_pivot"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

END_TO_END_CSV = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{PIVOT_TAG}.csv"
END_TO_END_JSON = REPORT_DATA / f"suds_tetc_end_to_end_accuracy_{PIVOT_TAG}.json"
GLUE_EXPANSION_JSON = REPORT_DATA / f"suds_tetc_glue_task_expansion_{TAG}.json"
DEIT_ACCURACY_JSON = REPORT_DATA / f"suds_tetc_deit_tiny_accuracy_{TAG}.json"

CSV_OUT = REPORT_DATA / f"suds_tetc_cross_workload_transfer_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_cross_workload_transfer_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260514_suds_tetc_r12_deep_reinforcement.md"

ACCURACY_TARGET_PP = 1.0
DIRECT_RATIO_TOLERANCE = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--end-to-end-csv", type=Path, default=END_TO_END_CSV)
    parser.add_argument("--end-to-end-json", type=Path, default=END_TO_END_JSON)
    parser.add_argument("--glue-expansion-json", type=Path, default=GLUE_EXPANSION_JSON)
    parser.add_argument("--deit-accuracy-json", type=Path, default=DEIT_ACCURACY_JSON)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--accuracy-target-pp", type=float, default=ACCURACY_TARGET_PP)
    parser.add_argument("--direct-ratio-tolerance", type=float, default=DIRECT_RATIO_TOLERANCE)
    return parser.parse_args()


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"missing required CSV artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def fmt_float(value: Any, digits: int = 4) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(val) for val in value]
    return value


def find_row(rows: list[dict[str, str]], *, workload: str, condition: str) -> dict[str, str]:
    matches = [
        row for row in rows
        if row.get("workload") == workload and row.get("condition") == condition
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one row for workload={workload!r} condition={condition!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def ratios(row: dict[str, str], prefix: str = "arch") -> dict[str, float]:
    return {
        "keep": as_float(row.get(f"{prefix}_keep_ratio")),
        "degrade": as_float(row.get(f"{prefix}_degrade_ratio")),
        "prune": as_float(row.get(f"{prefix}_prune_ratio")),
    }


def max_ratio_delta(source: dict[str, float], target: dict[str, float]) -> float:
    return max(abs(source[key] - target[key]) for key in ("keep", "degrade", "prune"))


def measured_delta(row: dict[str, str]) -> float:
    return as_float(row.get("delta_accuracy_pp"))


def row_from_evidence(
    *,
    tag: str,
    transfer_id: str,
    transfer_direction: str,
    source_row: dict[str, str],
    target_row: dict[str, str],
    transfer_mode: str,
    claim_role: str,
    accuracy_target_pp: float,
    direct_ratio_tolerance: float,
    note: str,
) -> dict[str, Any]:
    source_ratio = ratios(source_row)
    target_ratio = ratios(target_row)
    ratio_delta = max_ratio_delta(source_ratio, target_ratio)
    delta = measured_delta(target_row)
    within_target = abs(delta) <= accuracy_target_pp

    if transfer_mode == "local_control":
        evidence_strength = "local_policy_control"
        boundary_reason = ""
    elif ratio_delta <= direct_ratio_tolerance:
        evidence_strength = "direct_measured_transfer"
        boundary_reason = "" if within_target else "delta_exceeds_1pp"
    else:
        evidence_strength = "measured_policy_family_proxy"
        boundary_reason = (
            "exact_ratio_transfer_not_rerun"
            if within_target else "delta_exceeds_1pp_and_exact_ratio_transfer_not_rerun"
        )

    acceptance = "pass" if within_target and not boundary_reason else "transfer_boundary"

    return {
        "tag": tag,
        "roadmap_item": "R12c_cross_workload_policy_transfer",
        "transfer_id": transfer_id,
        "transfer_direction": transfer_direction,
        "source_workload": source_row.get("workload", ""),
        "target_workload": target_row.get("workload", ""),
        "source_model": source_row.get("model", ""),
        "target_model": target_row.get("model", ""),
        "source_policy_condition": source_row.get("condition", ""),
        "source_policy_source_condition": source_row.get("source_condition", ""),
        "target_evidence_condition": target_row.get("condition", ""),
        "target_source_condition": target_row.get("source_condition", ""),
        "transfer_mode": transfer_mode,
        "evidence_strength": evidence_strength,
        "claim_role": claim_role,
        "accuracy_metric": target_row.get("accuracy_metric", ""),
        "target_accuracy": as_float(target_row.get("accuracy")),
        "target_delta_accuracy_pp": delta,
        "accuracy_loss_target_pp": accuracy_target_pp,
        "within_1pp": within_target,
        "transfer_acceptance_state": acceptance,
        "transfer_boundary_reason": boundary_reason,
        "source_keep_ratio": source_ratio["keep"],
        "source_degrade_ratio": source_ratio["degrade"],
        "source_prune_ratio": source_ratio["prune"],
        "target_keep_ratio": target_ratio["keep"],
        "target_degrade_ratio": target_ratio["degrade"],
        "target_prune_ratio": target_ratio["prune"],
        "max_policy_ratio_delta": ratio_delta,
        "source_budget_signal": source_row.get("budget_signal", ""),
        "source_selection_signal": source_row.get("selection_signal", ""),
        "target_budget_signal": target_row.get("budget_signal", ""),
        "target_selection_signal": target_row.get("selection_signal", ""),
        "target_accuracy_evidence_label": target_row.get("accuracy_evidence_label", ""),
        "target_accuracy_source_artifact": target_row.get("accuracy_source_artifact", ""),
        "target_source_rows": target_row.get("accuracy_source_rows", ""),
        "target_source_seed_set": target_row.get("source_seed_set", ""),
        "target_source_device_set": target_row.get("source_device_set", ""),
        "source_schedule_trace_id": source_row.get("schedule_trace_id", ""),
        "target_schedule_trace_id": target_row.get("schedule_trace_id", ""),
        "source_trace_kernel_rows": source_row.get("trace_kernel_rows", ""),
        "target_trace_kernel_rows": target_row.get("trace_kernel_rows", ""),
        "source_claim_boundary": source_row.get("claim_boundary", ""),
        "target_claim_boundary": target_row.get("claim_boundary", ""),
        "notes": note,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    *,
    tag: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_tetc_cross_workload_transfer_{tag}",
            "roadmap_item": "R12c_cross_workload_policy_transfer",
            "evidence_label": "cross_workload_policy_transfer_from_governed_mps_rows",
            "regeneration_command": "make suds-tetc-cross-workload-transfer",
            "git_hash": git_hash(),
            "inputs": {
                "end_to_end_csv": repo_path(args.end_to_end_csv),
                "end_to_end_json": repo_path(args.end_to_end_json),
                "glue_expansion_json": repo_path(args.glue_expansion_json),
                "deit_accuracy_json": repo_path(args.deit_accuracy_json),
            },
            "input_sha256": {
                "end_to_end_csv": sha256_path(args.end_to_end_csv),
                "end_to_end_json": sha256_path(args.end_to_end_json),
                "glue_expansion_json": sha256_path(args.glue_expansion_json),
                "deit_accuracy_json": sha256_path(args.deit_accuracy_json),
            },
        },
        "summary": summary,
        "rows": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, *, tag: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transfer_rows = [row for row in rows if row["transfer_mode"] != "local_control"]
    boundary_rows_text = ", ".join(f"`{row}`" for row in summary["boundary_rows"]) or "`none`"

    report = f"""# SUDS TETC R12 Deep Reinforcement

Date: `{DATE}`
Tag: `{tag}`
Current focus: `R12c_cross_workload_policy_transfer`

## R12c Scope

R12c asks whether the SUDS budget/policy learned on one workload can be applied
to the other without per-workload tuning. This report reads the governed R3/R12
MPS evidence surface and records both transfer directions. It does not create a
new measured-accuracy claim beyond those source artifacts.

## Transfer Matrix

| Transfer | Evidence | Target delta | State | Boundary |
|---|---|---:|---|---|
"""
    for row in transfer_rows:
        report += (
            f"| `{row['transfer_id']}` | `{row['evidence_strength']}` | "
            f"{fmt_float(row['target_delta_accuracy_pp'])} pp | "
            f"`{row['transfer_acceptance_state']}` | "
            f"`{row['transfer_boundary_reason'] or 'none'}` |\n"
        )

    report += f"""
## Interpretation

- BERT-derived binary L1 policy transferred to MobileViT-S records
  `{summary['bert_to_mobilevit_delta_pp']:.4f}` pp top-1 delta, outside the
  `{summary['accuracy_loss_target_pp']:.1f}` pp budget, so it is a transfer
  boundary rather than a generality win.
- MobileViT-S conservative signal/overflow policy family transferred back to
  BERT records `{summary['mobilevit_to_bert_delta_pp']:.4f}` pp on the
  available measured BERT surface, but the exact MobileViT no-prune ratio was
  not rerun on BERT; this is recorded as a policy-family proxy boundary.
- The R12c answer is therefore not "one universal policy is enough." The
  evidence supports a workload-aware SUDS calibration claim, with conservative
  no-prune vision calibration preserved as the promoted MobileViT-S point.

## Acceptance

Acceptance state: `{summary['acceptance_state']}`

- Cross-workload transfer rows with measured deltas: `{summary['transfer_rows']}`
- Rows within 1 pp: `{summary['within_1pp_rows']}`
- Boundary rows: {boundary_rows_text}
- Per-workload tuning required: `{summary['requires_per_workload_tuning']}`

## Required Artifacts

- CSV: `experiments/results/report_data/suds_tetc_cross_workload_transfer_{tag}.csv`
- JSON: `experiments/results/report_data/suds_tetc_cross_workload_transfer_{tag}.json`
- Report: `docs/reports/20260514_suds_tetc_r12_deep_reinforcement.md`

## Regeneration

```bash
make suds-tetc-cross-workload-transfer
```
"""
    path.write_text(report, encoding="utf-8")


def build_rows(tag: str, e2e_rows: list[dict[str, str]], *, accuracy_target_pp: float, direct_ratio_tolerance: float) -> list[dict[str, Any]]:
    bert_promoted = find_row(e2e_rows, workload="bert_base_glue_seq128", condition="suds_pareto")
    bert_l1 = find_row(e2e_rows, workload="bert_base_glue_seq128", condition="l1")
    bert_signal = find_row(e2e_rows, workload="bert_base_glue_seq128", condition="suds_signal")
    mobilevit_promoted = find_row(e2e_rows, workload="mobilevit_s_transformer_blocks_256", condition="suds_pareto")
    mobilevit_l1 = find_row(e2e_rows, workload="mobilevit_s_transformer_blocks_256", condition="l1")

    return [
        row_from_evidence(
            tag=tag,
            transfer_id="bert_local_promoted_policy_control",
            transfer_direction="bert_policy_on_bert",
            source_row=bert_promoted,
            target_row=bert_promoted,
            transfer_mode="local_control",
            claim_role="control",
            accuracy_target_pp=accuracy_target_pp,
            direct_ratio_tolerance=direct_ratio_tolerance,
            note="BERT promoted R3 SUDS Pareto row used as the source-policy control.",
        ),
        row_from_evidence(
            tag=tag,
            transfer_id="bert_binary_l1_policy_to_mobilevit_s",
            transfer_direction="bert_to_mobilevit_s",
            source_row=bert_l1,
            target_row=mobilevit_l1,
            transfer_mode="cross_workload_transfer",
            claim_role="r12c_transfer_test",
            accuracy_target_pp=accuracy_target_pp,
            direct_ratio_tolerance=direct_ratio_tolerance,
            note=(
                "Applies the BERT fixed binary/L1 policy family to MobileViT-S using the "
                "measured MobileViT e2_l1 row already linked in R3."
            ),
        ),
        row_from_evidence(
            tag=tag,
            transfer_id="mobilevit_local_promoted_policy_control",
            transfer_direction="mobilevit_s_policy_on_mobilevit_s",
            source_row=mobilevit_promoted,
            target_row=mobilevit_promoted,
            transfer_mode="local_control",
            claim_role="control",
            accuracy_target_pp=accuracy_target_pp,
            direct_ratio_tolerance=direct_ratio_tolerance,
            note="MobileViT-S promoted conservative no-prune SUDS Pareto row used as the source-policy control.",
        ),
        row_from_evidence(
            tag=tag,
            transfer_id="mobilevit_signal_policy_family_to_bert",
            transfer_direction="mobilevit_s_to_bert",
            source_row=mobilevit_promoted,
            target_row=bert_signal,
            transfer_mode="cross_workload_transfer",
            claim_role="r12c_transfer_test",
            accuracy_target_pp=accuracy_target_pp,
            direct_ratio_tolerance=direct_ratio_tolerance,
            note=(
                "Uses the measured BERT e8 signal/overflow policy-family row as the available "
                "MobileViT-style transfer surface. Exact MobileViT tau=(0.30,0.95) no-prune "
                "ratio transfer to BERT is not rerun here and remains a recorded proxy boundary."
            ),
        ),
    ]


def build_summary(
    *,
    rows: list[dict[str, Any]],
    glue_payload: dict[str, Any],
    deit_payload: dict[str, Any],
    end_to_end_payload: dict[str, Any],
    accuracy_target_pp: float,
) -> dict[str, Any]:
    transfer_rows = [row for row in rows if row["transfer_mode"] == "cross_workload_transfer"]
    boundary_rows = [
        row["transfer_id"] for row in transfer_rows
        if row["transfer_acceptance_state"] != "pass"
    ]
    within_rows = [row for row in transfer_rows if row["within_1pp"]]
    bert_to_mobilevit = next(row for row in transfer_rows if row["transfer_direction"] == "bert_to_mobilevit_s")
    mobilevit_to_bert = next(row for row in transfer_rows if row["transfer_direction"] == "mobilevit_s_to_bert")

    exact_transfer_missing = any(
        row["evidence_strength"] == "measured_policy_family_proxy"
        for row in transfer_rows
    )
    requires_tuning = bool(boundary_rows or exact_transfer_missing)
    acceptance_state = "pass" if not requires_tuning else "boundary_recorded"

    return {
        "date": DATE,
        "tag": rows[0]["tag"] if rows else TAG,
        "transfer_rows": len(transfer_rows),
        "control_rows": len(rows) - len(transfer_rows),
        "within_1pp_rows": len(within_rows),
        "boundary_rows": boundary_rows,
        "proxy_transfer_rows": [
            row["transfer_id"] for row in transfer_rows
            if row["evidence_strength"] == "measured_policy_family_proxy"
        ],
        "accuracy_loss_target_pp": accuracy_target_pp,
        "bert_to_mobilevit_delta_pp": bert_to_mobilevit["target_delta_accuracy_pp"],
        "mobilevit_to_bert_delta_pp": mobilevit_to_bert["target_delta_accuracy_pp"],
        "requires_per_workload_tuning": requires_tuning,
        "acceptance_state": acceptance_state,
        "blockers": [],
        "source_context": {
            "r3_decision": (end_to_end_payload.get("summary") or {}).get("decision", {}),
            "r12b_acceptance_state": (glue_payload.get("summary") or {}).get("acceptance_state", ""),
            "r12b_tasks_with_non_zero_delta": (glue_payload.get("summary") or {}).get("tasks_with_non_zero_delta", []),
            "r12g_acceptance_state": (deit_payload.get("summary") or {}).get("acceptance_state", ""),
            "r12g_mean_delta_top1_pp": (deit_payload.get("summary") or {}).get("mean_delta_top1_pp", None),
        },
        "claim": (
            "R12c reports both transfer directions. BERT fixed binary/L1 transfer "
            "to MobileViT-S exceeds the 1 pp accuracy budget, while the available "
            "MobileViT-style signal/overflow transfer surface on BERT is benign but "
            "not an exact no-prune ratio rerun. The result is recorded as a "
            "cross-workload transfer boundary supporting workload-aware calibration."
        ),
    }


def main() -> int:
    args = parse_args()
    e2e_rows = load_csv(args.end_to_end_csv)
    end_to_end_payload = load_json(args.end_to_end_json)
    glue_payload = load_json(args.glue_expansion_json)
    deit_payload = load_json(args.deit_accuracy_json)

    rows = build_rows(
        args.tag,
        e2e_rows,
        accuracy_target_pp=args.accuracy_target_pp,
        direct_ratio_tolerance=args.direct_ratio_tolerance,
    )
    summary = build_summary(
        rows=rows,
        glue_payload=glue_payload,
        deit_payload=deit_payload,
        end_to_end_payload=end_to_end_payload,
        accuracy_target_pp=args.accuracy_target_pp,
    )

    write_csv(args.csv_out, rows)
    write_json(args.json_out, tag=args.tag, args=args, rows=rows, summary=summary)
    write_report(args.report_out, tag=args.tag, rows=rows, summary=summary)

    print(f"Wrote {args.csv_out} ({len(rows)} rows)")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.report_out}")
    print(f"Acceptance state: {summary['acceptance_state']}")
    print(f"BERT -> MobileViT-S delta: {summary['bert_to_mobilevit_delta_pp']:.4f} pp")
    print(f"MobileViT-S -> BERT delta: {summary['mobilevit_to_bert_delta_pp']:.4f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
