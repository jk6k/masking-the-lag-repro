#!/usr/bin/env python3
"""Build the R11 public mini-benchmark and external red-team artifact."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
DATE = "2026-05-14"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

CSV_OUT = REPORT_DATA / f"suds_tetc_external_red_team_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_external_red_team_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260514_suds_tetc_external_red_team.md"
SCAFFOLD_OUT = REPO_ROOT / "docs/reports/20260514_suds_tetc_major_revision_response_scaffold.md"
MANIFEST = REPO_ROOT / "configs/public_repro_manifest.json"

REQUIRED_PUBLIC_REPRO_PATHS = [
    f"experiments/results/report_data/suds_tetc_external_red_team_{TAG}.csv",
    f"experiments/results/report_data/suds_tetc_external_red_team_{TAG}.json",
    "docs/reports/20260514_suds_tetc_external_red_team.md",
    "docs/reports/20260514_suds_tetc_major_revision_response_scaffold.md",
    "experiments/tools/build_suds_tetc_external_red_team.py",
]

REQUIRED_PUBLIC_REPRO_DATA = [
    f"suds_tetc_external_red_team_{TAG}.csv",
    f"suds_tetc_external_red_team_{TAG}.json",
]

PUBLIC_REPRO_COMMANDS = [
    "make public-repro-build",
    "make public-repro-check",
    "make public-repro-render",
    "make public-repro-check",
]

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".tex",
    ".txt",
    ".log",
    ".py",
    ".v",
    ".yml",
    ".yaml",
}
TEXT_NAMES = {"Makefile", ".gitattributes", ".gitignore"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--scaffold-out", type=Path, default=SCAFFOLD_OUT)
    parser.add_argument("--public-root", type=Path, default=None)
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configured_public_root(manifest: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    destination = Path(str(manifest.get("default_destination") or "../masking-the-lag-repro"))
    if destination.is_absolute():
        return destination
    return (REPO_ROOT / destination).resolve()


def manifest_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    copy_files = {str(item) for item in manifest.get("copy_files", [])}
    required_files = {str(item) for item in manifest.get("required_files", [])}
    required_data = {str(item) for item in manifest.get("required_report_data_files", [])}
    missing_copy = [path for path in REQUIRED_PUBLIC_REPRO_PATHS if path not in copy_files]
    missing_required = [path for path in REQUIRED_PUBLIC_REPRO_PATHS if path not in required_files]
    missing_data = [path for path in REQUIRED_PUBLIC_REPRO_DATA if path not in required_data]
    blockers = (
        [f"missing_copy:{path}" for path in missing_copy]
        + [f"missing_required:{path}" for path in missing_required]
        + [f"missing_report_data:{path}" for path in missing_data]
    )
    return {
        "status": "pass" if not blockers else "fail",
        "missing_copy_files": missing_copy,
        "missing_required_files": missing_required,
        "missing_required_report_data_files": missing_data,
        "blockers": blockers,
    }


def public_repro_validation(public_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not public_root.exists():
        return {
            "status": "missing",
            "public_root": "<public_repro>",
            "validation_error_count": 1,
            "validation_errors": [f"public_root_missing:{public_root}"],
        }

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.check_public_repro_repo import validate

    report = validate(public_root)
    generated_manifest = load_json(public_root / "configs/public_repro_manifest.json")
    errors = [redact_tokens(error, manifest) for error in report.errors]
    if generated_manifest.get("tetc_evidence_tag") != TAG:
        errors.append(f"generated_manifest_tetc_evidence_tag_not_{TAG}")
    for rel_path in REQUIRED_PUBLIC_REPRO_PATHS:
        if not (public_root / rel_path).is_file():
            errors.append(f"generated_public_repro_missing:{rel_path}")
    return {
        "status": "pass" if not errors else "fail",
        "public_root": "<public_repro>",
        "validation_error_count": len(errors),
        "validation_errors": errors[:40],
    }


def built_in_private_tokens() -> set[str]:
    return {
        "/" + "Users" + "/",
        "jk" + "6k",
        "github.com/" + "jk" + "6k",
        "masking-the-lag-repro" + "." + "git",
    }


def redaction_tokens(manifest: dict[str, Any]) -> set[str]:
    return built_in_private_tokens() | {
        str(item) for item in manifest.get("banned_public_text_tokens", [])
    }


def redact_tokens(text: str, manifest: dict[str, Any]) -> str:
    for token in sorted(redaction_tokens(manifest), key=len, reverse=True):
        if token:
            text = text.replace(token, "<banned_public_text_token>")
    return text


def public_text_leak_audit(public_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not public_root.exists():
        return {"status": "missing", "matches": []}
    tokens = redaction_tokens(manifest)
    matches: list[dict[str, str]] = []
    for path in sorted(public_root.rglob("*")):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
            continue
        try:
            rel = path.relative_to(public_root).as_posix()
            if rel == "configs/public_repro_manifest.json":
                continue
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in sorted(tokens):
            if token and token in text:
                matches.append({"path": rel, "token": "<banned_public_text_token>"})
    return {
        "status": "pass" if not matches else "fail",
        "matches": matches[:40],
        "match_count": len(matches),
    }


def external_issue_rows() -> list[dict[str, Any]]:
    return [
        {
            "issue_id": "R11-EXT0",
            "lens": "external_independence",
            "severity": "medium",
            "status": "accepted_risk",
            "finding": "No independent reader could be contacted from this local Codex execution.",
            "resolution": "The public packet and request questions are ready; the manuscript and reports state that the local substitute is not equivalent to external review.",
            "claim_change": "Keep external red-team review preferred before submission, advisory in local gates, and not counted as independent validation.",
            "evidence": "paper/suds_tetc_architecture_manuscript.tex; docs/reports/20260513_suds_tetc_pre_review_major_revision.md",
        },
        {
            "issue_id": "R11-EXT1",
            "lens": "simulator_credibility",
            "severity": "high",
            "status": "fixed",
            "finding": "A reviewer will ask whether the architecture simulator is more than static accounting.",
            "resolution": "R1 event traces and R2 scheduler traces are included in the public package; R10 records failure and uncertainty boundaries.",
            "claim_change": "Use event-level modeled architecture wording, not hardware-measured or cycle-accurate silicon wording.",
            "evidence": "suds_tetc_event_sim_20260513_tetc_pivot.json; suds_tetc_scheduler_traces_20260513_tetc_pivot.json; suds_tetc_uncertainty_20260513_tetc_pivot.json",
        },
        {
            "issue_id": "R11-EXT2",
            "lens": "baseline_fairness",
            "severity": "high",
            "status": "fixed",
            "finding": "Same-simulator baselines and alternate-fabric boundaries must be visible to a skeptical reviewer.",
            "resolution": "R4 same-simulator fairness and R5 Pareto rationale are exported; TeMPO/ASTRA/HyAtten remain boundary rows where assumptions differ.",
            "claim_change": "Promote only suds_pareto and keep local-selector or alternate-fabric wins as boundary evidence.",
            "evidence": "suds_tetc_same_sim_baselines_20260513_tetc_pivot.json; suds_tetc_pareto_design_space_20260513_tetc_pivot.json",
        },
        {
            "issue_id": "R11-EXT3",
            "lens": "public_reproducibility",
            "severity": "high",
            "status": "fixed",
            "finding": "The mini benchmark must be small, checkable, and free of private paths, data, weights, and legacy-route entry points.",
            "resolution": "The manifest whitelists compact traces, configs, reports, render scripts, checksums, and reviewer instructions; public validation scans the exported text surface.",
            "claim_change": "Describe the public package as a reader-facing artifact package, not a full governed MPS rerun bundle.",
            "evidence": "configs/public_repro_manifest.json; checksums_manifest.json; scripts/check_public_repro_repo.py",
        },
        {
            "issue_id": "R11-EXT4",
            "lens": "major_revision_readiness",
            "severity": "medium",
            "status": "fixed",
            "finding": "Likely major-revision objections need pre-written response hooks tied to artifacts.",
            "resolution": "A reviewer-response scaffold maps simulator, baseline, calibration, workload, uncertainty, and reproducibility objections to exact evidence artifacts.",
            "claim_change": "Do not answer with stronger headline numbers unless a regenerated science gate supports them.",
            "evidence": "docs/reports/20260514_suds_tetc_major_revision_response_scaffold.md",
        },
    ]


def build_summary(
    manifest_status: dict[str, Any],
    validation: dict[str, Any],
    leak_audit: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    issue_policy_ok = all(row["status"] in {"fixed", "accepted_risk"} for row in rows)
    blockers = []
    if manifest_status["status"] != "pass":
        blockers.extend(manifest_status["blockers"])
    if validation["status"] != "pass":
        blockers.append(f"public_repro_validation_{validation['status']}")
    if leak_audit["status"] != "pass":
        blockers.append(f"public_text_leak_audit_{leak_audit['status']}")
    if not issue_policy_ok:
        blockers.append("external_red_team_issue_policy_not_closed")
    stop_triggered = leak_audit["status"] == "fail"
    return {
        "date": DATE,
        "tag": TAG,
        "r11_acceptance_state": "pass" if not blockers else "fail",
        "stop_condition_state": "no R11 hard stop" if not stop_triggered else "R11 hard stop: public package leak detected",
        "manifest_audit": manifest_status,
        "public_repro_validation": validation,
        "public_text_leak_audit": leak_audit,
        "external_reader_status": "packet_ready_not_sent_from_codex",
        "external_issue_policy": "fixed_or_accepted_risk" if issue_policy_ok else "open_issue_present",
        "accepted_risk_count": sum(1 for row in rows if row["status"] == "accepted_risk"),
        "fixed_issue_count": sum(1 for row in rows if row["status"] == "fixed"),
        "public_repro_commands": PUBLIC_REPRO_COMMANDS,
        "blockers": blockers,
    }


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SUDS TETC External Red-Team And Public Mini Benchmark",
        "",
        f"Date: `{DATE}`",
        f"Tag: `{TAG}`",
        "Evidence label: `r11_public_mini_benchmark_external_red_team`",
        f"Status: `{summary['r11_acceptance_state']}`",
        f"Stop-condition state: `{summary['stop_condition_state']}`",
        "",
        "## Public Mini Benchmark",
        "",
        f"- Manifest audit: `{summary['manifest_audit']['status']}`",
        f"- Generated package validation: `{summary['public_repro_validation']['status']}`",
        f"- Public text leak audit: `{summary['public_text_leak_audit']['status']}`",
        f"- Validation error count: `{summary['public_repro_validation']['validation_error_count']}`",
        f"- Text leak match count: `{summary['public_text_leak_audit'].get('match_count', 0)}`",
        "",
        "Required live commands:",
        "",
    ]
    lines.extend(f"- `{command}`" for command in PUBLIC_REPRO_COMMANDS)
    lines.extend(
        [
            "",
            "## External Red-Team Record",
            "",
            f"- External reader status: `{summary['external_reader_status']}`",
            f"- Issue policy: `{summary['external_issue_policy']}`",
            f"- Fixed issues: `{summary['fixed_issue_count']}`",
            f"- Accepted risks: `{summary['accepted_risk_count']}`",
            "",
            "No independent reader could be contacted inside this local execution",
            "environment. The external-review gap is therefore recorded as an",
            "accepted risk, not as independent validation. The manuscript and",
            "pre-review report preserve that boundary.",
            "",
            "## Issue Table",
            "",
            "| ID | Lens | Severity | Status | Finding | Resolution | Claim change |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['issue_id']}` | `{row['lens']}` | `{row['severity']}` | "
            f"`{row['status']}` | {row['finding']} | {row['resolution']} | {row['claim_change']} |"
        )
    lines.extend(
        [
            "",
            "## Reader Request Packet",
            "",
            "Ask one or two external readers to review these four questions:",
            "",
            "1. Does the event-level simulator evidence make the PPA argument credible enough for an architecture paper?",
            "2. Are the same-simulator baselines and boundary fabrics separated clearly enough?",
            "3. Are the claim boundaries around ADC, RTL, PHY, and uncertainty conservative enough?",
            "4. Can the public mini benchmark be checked without private data, weights, literature mirrors, or personal paths?",
            "",
            "External feedback should be resolved by either a code/data fix, a",
            "manuscript claim change, or an explicit accepted-risk entry before upload.",
        ]
    )
    if summary["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in summary["blockers"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_scaffold(path: Path) -> None:
    lines = [
        "# SUDS TETC Major-Revision Response Scaffold",
        "",
        f"Date: `{DATE}`",
        f"Tag: `{TAG}`",
        "Purpose: `pre-write response hooks for likely TETC reviewer objections`",
        "",
        "## Response Table",
        "",
        "| Reviewer concern | Response stance | Evidence artifacts | Claim boundary |",
        "|---|---|---|---|",
        "| The simulator is static accounting, not architecture evaluation. | Point to R1 event traces, R2 scheduler traces, and R6/R10 sensitivity; offer to expand event traces in revision. | `suds_tetc_event_sim_20260513_tetc_pivot.json`; `suds_tetc_scheduler_traces_20260513_tetc_pivot.json`; `suds_tetc_uncertainty_20260513_tetc_pivot.json` | Modeled architecture/event-level evidence only. |",
        "| SUDS may be dominated by simpler selectors. | Show R4 same-scope equal-accuracy matrix and R5 selected Pareto rationale; keep L1/signal wins visible as boundary or ablation evidence. | `suds_tetc_same_sim_baselines_20260513_tetc_pivot.json`; `suds_tetc_pareto_design_space_20260513_tetc_pivot.json` | Promote only `suds_pareto`. |",
        "| The calibration story overclaims circuit closure. | Reiterate ADC macro, RTL, and PHY artifacts as calibration/proxy/boundary inputs. | `suds_tetc_rtl_control_plane_20260513_tetc_pivot.json`; `suds_tetc_calibration_ranges_20260513_tetc_pivot.json` | No foundry, layout, silicon, bench-energy, or device-solver claim. |",
        "| Workload generality is narrow. | Present R9 as architecture-only generality expansion and state measured accuracy is limited to governed MPS BERT/GLUE and MobileViT-S rows. | `suds_tetc_workload_expansion_20260513_tetc_pivot.json`; `suds_tetc_end_to_end_accuracy_20260513_tetc_pivot.json` | New DeiT-Tiny and sequence/batch rows are not measured-accuracy promotion rows. |",
        "| Failure cases are hidden. | Point to R10 failure families and boundary-regime uncertainty crossing zero; narrow target-regime claims accordingly. | `suds_tetc_failure_suite_20260513_tetc_pivot.csv`; `suds_tetc_uncertainty_20260513_tetc_pivot.json` | SUDS should not be used in named low-slack or boundary regimes without fallback. |",
        "| The public artifact may leak private data or legacy route language. | Point to the whitelist manifest, checksum manifest, public-repro validator, and R11 text-surface scan. | `configs/public_repro_manifest.json`; `checksums_manifest.json`; `suds_tetc_external_red_team_20260513_tetc_pivot.json` | Public package is a compact reader benchmark, not a full private rerun workspace. |",
        "",
        "## Revision Discipline",
        "",
        "Do not raise headline claims in the response unless the relevant R-phase",
        "artifact and `make suds-tetc-science-gate` are regenerated and pass. If a",
        "reviewer asks for unavailable silicon, foundry, P&R, or bench evidence,",
        "answer by narrowing the claim rather than inventing closure.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_json(MANIFEST)
    public_root = configured_public_root(manifest, args.public_root)
    rows = external_issue_rows()
    manifest_status = manifest_audit(manifest)
    validation = public_repro_validation(public_root, manifest)
    leak_audit = public_text_leak_audit(public_root, manifest)
    summary = build_summary(manifest_status, validation, leak_audit, rows)
    payload = {
        "metadata": {
            "tag": TAG,
            "artifact_id": f"suds_tetc_external_red_team_{TAG}",
            "evidence_label": "r11_public_mini_benchmark_external_red_team",
            "regeneration_command": "make suds-tetc-external-red-team",
        },
        "summary": summary,
        "rows": rows,
        "artifacts": {
            "csv": repo_path(args.csv_out),
            "json": repo_path(args.json_out),
            "report": repo_path(args.report_out),
            "major_revision_scaffold": repo_path(args.scaffold_out),
        },
    }
    write_csv(args.csv_out, rows)
    write_json(args.json_out, payload)
    write_report(args.report_out, summary, rows)
    write_scaffold(args.scaffold_out)
    print(f"wrote {repo_path(args.csv_out)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")
    print(f"wrote {repo_path(args.scaffold_out)}")
    print(f"r11_acceptance_state={summary['r11_acceptance_state']}")


if __name__ == "__main__":
    main()
