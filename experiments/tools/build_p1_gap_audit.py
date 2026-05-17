#!/usr/bin/env python3
"""Build a compact remaining P1 gap audit summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


DEFAULT_DELTA_PP_BUDGET = 1.0


def _load_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _string_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=str)
    return df[column].fillna("").astype(str).str.strip()


def _format_holdout_summary(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    df = pd.read_csv(path)
    chunks: list[str] = []
    for _, row in df.iterrows():
        chunks.append(
            f"{row['split']}: E6-E0={float(row['delta_acc_top1_mean']):.3f}\u00b1{float(row['delta_acc_top1_ci95_half']):.3f} pp"
        )
    if not chunks:
        return ""
    return f"Summary = {'; '.join(chunks)}."


def _holdout_accuracy_gate_failed(path: Path | None, delta_pp_budget: float = DEFAULT_DELTA_PP_BUDGET) -> bool | None:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    if "delta_acc_top1_mean" not in df.columns:
        return None
    deltas = pd.to_numeric(df["delta_acc_top1_mean"], errors="coerce").dropna()
    if deltas.empty:
        return None
    return bool((deltas < (-abs(float(delta_pp_budget)))).any())


def _has_formal_holdout_report(report_text: str) -> bool:
    report_norm = str(report_text or "").strip().lower()
    required_tokens = ("accuracy_csv:", "eval_manifest:", "holdout_manifest:")
    return all(token in report_norm for token in required_tokens)


def _task_context_same_task_anchor_count(path: Path) -> int:
    df = _load_csv(path)
    if df.empty:
        return 0
    source_type = _string_series(df, "source_type").str.lower()
    if source_type.empty:
        return 0
    same_task = source_type == "external_anchor"
    task_id = _string_series(df, "task_id").str.lower()
    if not task_id.empty:
        same_task = same_task & (task_id == "imagenet_cls")
    return int(same_task.sum())


def _broader_task_support_count(path: Path) -> tuple[int, str]:
    df = _load_csv(path)
    if df.empty:
        return 0, "broader task entries"
    task_group = _string_series(df, "task_group").str.lower()
    if not task_group.empty:
        values = {item for item in task_group.tolist() if item}
        return len(values), "broader task groups"
    task_id = _string_series(df, "task_id").str.lower()
    if not task_id.empty:
        values = {item for item in task_id.tolist() if item and item != "imagenet_cls"}
        return len(values), "broader task ids"
    return 0, "broader task entries"


def _context_summary(task_context_csv: Path, broader_task_csv: Path) -> tuple[bool, str]:
    same_task_anchor_count = _task_context_same_task_anchor_count(task_context_csv)
    broader_task_count, broader_label = _broader_task_support_count(broader_task_csv)
    fragments: list[str] = []
    if same_task_anchor_count:
        fragments.append(f"same-task external anchors={same_task_anchor_count}")
    if broader_task_count:
        fragments.append(f"{broader_label}={broader_task_count}")
    return same_task_anchor_count > 0 and broader_task_count > 0, "; ".join(fragments)


def _second_chain_complete(
    accuracy_csv: Path,
    runs_root: Path,
    run_prefix: str,
    model: str,
) -> bool:
    required_experiments = {"E0", "E2", "E3", "E4", "E6"}
    if not accuracy_csv.exists():
        return False
    accuracy_df = pd.read_csv(accuracy_csv)
    model_df = accuracy_df.loc[accuracy_df.get("model", pd.Series(dtype=str)) == model]
    if set(model_df.get("experiment_id", [])) != required_experiments:
        return False
    for experiment in required_experiments:
        run_dir = runs_root / f"{run_prefix}_{experiment.lower()}"
        if not (run_dir / "master_metrics.csv").is_file():
            return False
    return True


def _build_p11_row(args: argparse.Namespace) -> dict[str, str]:
    complete = _second_chain_complete(
        args.second_chain_accuracy_csv,
        args.second_chain_runs_root,
        args.second_chain_run_prefix,
        args.second_chain_model,
    )
    if complete:
        return {
            "gap_id": "P1-1",
            "status": "matched_inhouse_non_mobilevit_chain_complete",
            "acceptance_ready": "yes",
            "evidence_summary": f"Canonical in-house `{args.second_chain_model}` chain is complete across E0/E2/E3/E4/E6.",
            "recommended_next_step": "No further P1-1 action required.",
        }
    return {
        "gap_id": "P1-1",
        "status": "external_anchor_only",
        "acceptance_ready": "no",
        "evidence_summary": "Only external anchors are present; a matched in-house non-MobileViT chain is still missing.",
        "recommended_next_step": "Run the canonical non-MobileViT chain and capture measured outputs.",
    }


def _build_p14_row(args: argparse.Namespace) -> dict[str, str]:
    holdout_df = _load_csv(args.holdout_accuracy_csv)
    holdout_summary_df = _load_csv(args.holdout_summary_csv)
    experiments = sorted({str(item) for item in holdout_df.get("experiment_id", []) if str(item).strip()})
    evidence_summary = ""
    recommended = ""
    status = "blocker_audit_only"
    acceptance_ready = "no"

    summary_text = _format_holdout_summary(args.holdout_summary_csv)
    report_text = args.holdout_report_md.read_text(encoding="utf-8") if args.holdout_report_md.exists() else ""
    manifest_values = " ".join(str(item) for item in holdout_df.get("imagenet_manifest", []))
    holdout_splits = {item for item in _string_series(holdout_summary_df, "split").str.lower().tolist() if item}
    full_split_holdout = (
        set(experiments) >= {"E0", "E6"}
        and {"eval", "holdout"} <= holdout_splits
        and bool(summary_text)
        and _has_formal_holdout_report(report_text)
        and (
            "independent split-matched" in report_text.lower()
            or "complete holdout split" in report_text.lower()
        )
    )
    context_complete, context_summary = _context_summary(args.task_context_csv, args.broader_task_csv)
    second_chain_ready = _second_chain_complete(
        args.second_chain_accuracy_csv,
        args.second_chain_runs_root,
        args.second_chain_run_prefix,
        args.second_chain_model,
    )

    if full_split_holdout:
        gate_failed = _holdout_accuracy_gate_failed(args.holdout_summary_csv)
        evidence_chunks = [summary_text]
        if context_summary:
            evidence_chunks.append(f"Context = {context_summary}.")
        evidence_summary = " ".join(evidence_chunks)
        if context_complete and second_chain_ready:
            acceptance_ready = "yes"
            if gate_failed:
                status = "generalization_evidence_chain_complete_negative_result"
                recommended = (
                    "No strict P1-4 action required; keep E6 framed as a negative-result/tradeoff endpoint and avoid accuracy-safe wording."
                )
            else:
                status = "generalization_evidence_chain_complete"
                recommended = "No further P1-4 action required."
        elif gate_failed:
            status = "full_split_same_dataset_holdout_failed_accuracy_gate"
            recommended = (
                "Keep P1-4 open; add the broader task/context evidence chain or replace modeled E6 rows with dedicated measured holdout reruns."
            )
        else:
            status = "full_split_same_dataset_holdout_evidence_only"
            recommended = "Same-dataset holdout evidence does not close P1-4 on its own; add broader task/context evidence."
    elif set(experiments) >= {"E0", "E6"} and "splits_smoke" in manifest_values and summary_text:
        status = "split_specific_smoke_evidence_only"
        acceptance_ready = "no"
        evidence_summary = summary_text
        recommended = "Promote this split-aware smoke audit to a full holdout protocol with canonical manifests."
    else:
        evidence_summary = f"Current source covers experiments `{', '.join(experiments) or 'none'}` only."
        recommended = "Add independent split-matched E0/E6 rows for eval and holdout."

    return {
        "gap_id": "P1-4",
        "status": status,
        "acceptance_ready": acceptance_ready,
        "evidence_summary": evidence_summary,
        "recommended_next_step": recommended,
    }


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    p11 = rows[0]
    p14 = rows[1]
    lines = ["# Remaining P1 Gap Audit", ""]
    if p11["acceptance_ready"] == "yes" and p14["acceptance_ready"] == "yes":
        lines.append("no strict P1 evidence gaps remain")
        lines.append(f"`P1-1` is now backed by a canonical in-house `{p11['evidence_summary'].split('`')[1]}` chain")
        if p14["status"] == "generalization_evidence_chain_complete_negative_result":
            lines.append("`P1-4` is closed under the current checklist as a negative-result/tradeoff endpoint")
        else:
            lines.append("`P1-4` is closed under the current checklist")
    else:
        if p11["acceptance_ready"] == "yes":
            lines.append(f"`P1-1` is now backed by a canonical in-house `{p11['evidence_summary'].split('`')[1]}` chain")
        else:
            lines.append("`P1-1` remains open")
        if p14["status"] == "blocker_audit_only":
            lines.append("blocker audit")
            lines.append("split-routing sanity check")
        elif p14["status"] == "split_specific_smoke_evidence_only":
            lines.append("shared-source routing-only blocker is cleared")
            lines.append("smoke-scale same-dataset sanity check")
        elif p14["status"] == "full_split_same_dataset_holdout_failed_accuracy_gate":
            lines.append("full-split same-dataset holdout is present")
            lines.append("large E6 accuracy loss keeps `P1-4` open")
        elif p14["status"] == "full_split_same_dataset_holdout_evidence_only":
            lines.append("full-split same-dataset holdout is present")
            lines.append("same-dataset evidence alone does not close `P1-4`")
        elif p14["status"] == "generalization_evidence_chain_complete_negative_result":
            lines.append("full-split same-dataset holdout is present")
            lines.append("broader task/context evidence is complete, so `P1-4` closes as a negative-result/tradeoff endpoint")
        elif p14["status"] == "generalization_evidence_chain_complete":
            lines.append("full-split same-dataset holdout is present")
            lines.append("broader task/context evidence is complete, so `P1-4` is closed")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact remaining P1 gap audit.")
    parser.add_argument("--task_context_csv", type=Path, required=True)
    parser.add_argument("--broader_task_csv", type=Path, required=True)
    parser.add_argument("--holdout_accuracy_csv", type=Path, required=True)
    parser.add_argument("--holdout_report_md", type=Path, required=True)
    parser.add_argument("--holdout_summary_csv", type=Path, default=None)
    parser.add_argument("--second_chain_accuracy_csv", type=Path, required=True)
    parser.add_argument("--second_chain_runs_root", type=Path, required=True)
    parser.add_argument("--second_chain_run_prefix", required=True)
    parser.add_argument("--second_chain_model", default="resnet_50")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    rows = [_build_p11_row(args), _build_p14_row(args)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"remaining_p1_gap_audit_{args.tag}.csv"
    out_md = args.out_dir / f"remaining_p1_gap_audit_{args.tag}.md"

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["gap_id", "status", "acceptance_ready", "evidence_summary", "recommended_next_step"],
        )
        writer.writeheader()
        writer.writerows(rows)

    _write_markdown(out_md, rows)


if __name__ == "__main__":
    main()
