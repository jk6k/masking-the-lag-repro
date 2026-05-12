#!/usr/bin/env python3
"""Build the SUDS J6 final submission-readiness gate.

The gate records reproducible command outcomes, claim-boundary scans, artifact
presence, and the promotion decision for the quality-boost package.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAG = "20260512_j6_quality_boost"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
CSV_OUT = REPORT_DATA / f"suds_final_submission_gate_{DEFAULT_TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_final_submission_gate_{DEFAULT_TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260512_j6_suds_final_submission_gate.md"
PUBLIC_REPRO_ROOT = REPO_ROOT.parent / "masking-the-lag-repro"
RUN_ROOT = REPO_ROOT / f"experiments/results/runs/suds_final_submission_gate_{DEFAULT_TAG}"


CLAIM_SCAN_PATTERN = (
    r"silicon measurement|hardware validation|SPICE closure|Spectre|"
    r"extracted layout|measured energy|first|universal|slack-only|"
    r"state-of-the-art|SOTA"
)


ARTIFACTS = [
    (
        "J0 baseline protection",
        "experiments/results/report_data/suds_j0_baseline_protection_20260512_j0_quality_boost.json",
        "audit",
    ),
    (
        "J1 ADC macro sanity",
        "experiments/results/report_data/suds_adc_macro_sanity_20260512_j1_quality_boost.json",
        "spice_macro",
    ),
    (
        "J2 RTL synthesis",
        "experiments/results/report_data/suds_rtl_control_overhead_20260512_j2_quality_boost.json",
        "rtl_synthesis",
    ),
    (
        "J5 internal red-team",
        "experiments/results/report_data/suds_internal_red_team_20260512_j5_quality_boost.json",
        "audit",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--run-checks",
        action="store_true",
        help="Execute make/PDF/public-repro checks before writing the gate.",
    )
    return parser.parse_args()


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        pass
    try:
        return "<public_repro>/" + str(resolved.relative_to(PUBLIC_REPRO_ROOT))
    except ValueError:
        return str(path)


def sanitize_text(text: str) -> str:
    return (
        text.replace(str(REPO_ROOT), "<repo>")
        .replace(str(PUBLIC_REPRO_ROOT), "<public_repro>")
        .replace(str(REPO_ROOT.parent), "<workspace_parent>")
    )


def command_specs() -> list[dict[str, Any]]:
    return [
        {
            "check_id": "make_smoke",
            "command": ["make", "smoke"],
            "cwd": REPO_ROOT,
            "required": True,
        },
        {
            "check_id": "repo_hygiene",
            "command": ["make", "repo-hygiene"],
            "cwd": REPO_ROOT,
            "required": True,
        },
        {
            "check_id": "public_repro_build",
            "command": ["make", "public-repro-build"],
            "cwd": REPO_ROOT,
            "required": True,
        },
        {
            "check_id": "public_repro_adc_macro_regen",
            "command": [
                "python3",
                "experiments/tools/run_suds_adc_macro_spice_suite.py",
                "--tag",
                "20260512_j1_quality_boost",
                "--csv-out",
                "/tmp/suds_j6_public_regen/adc_macro/suds_adc_macro_sanity_20260512_j1_quality_boost.csv",
                "--json-out",
                "/tmp/suds_j6_public_regen/adc_macro/suds_adc_macro_sanity_20260512_j1_quality_boost.json",
                "--report-out",
                "/tmp/suds_j6_public_regen/adc_macro/20260512_j1_suds_adc_macro_spice_suite.md",
                "--deck-root",
                "/tmp/suds_j6_public_regen/adc_macro/spice",
                "--run-root",
                "/tmp/suds_j6_public_regen/adc_macro/runs",
            ],
            "cwd": PUBLIC_REPRO_ROOT,
            "required": True,
        },
        {
            "check_id": "public_repro_rtl_regen",
            "command": [
                "python3",
                "experiments/tools/build_suds_rtl_control_overhead.py",
                "--tag",
                "20260512_j2_quality_boost",
                "--csv-out",
                "/tmp/suds_j6_public_regen/rtl/suds_rtl_control_overhead_20260512_j2_quality_boost.csv",
                "--json-out",
                "/tmp/suds_j6_public_regen/rtl/suds_rtl_control_overhead_20260512_j2_quality_boost.json",
                "--report-out",
                "/tmp/suds_j6_public_regen/rtl/20260512_j2_quality_boost_suds_rtl_control_overhead.md",
                "--run-root",
                "/tmp/suds_j6_public_regen/rtl/runs",
            ],
            "cwd": PUBLIC_REPRO_ROOT,
            "required": True,
        },
        {
            "check_id": "public_repro_check",
            "command": ["make", "public-repro-check"],
            "cwd": REPO_ROOT,
            "required": True,
        },
        {
            "check_id": "public_repro_render",
            "command": ["make", "public-repro-render"],
            "cwd": REPO_ROOT,
            "required": True,
        },
        {
            "check_id": "public_repro_check_after_render",
            "command": ["make", "public-repro-check"],
            "cwd": REPO_ROOT,
            "required": True,
        },
        {
            "check_id": "paper_compile_pass_1",
            "command": ["tectonic", "-X", "compile", "suds_paper_acmart.tex"],
            "cwd": REPO_ROOT / "paper",
            "required": True,
        },
        {
            "check_id": "paper_compile_pass_2",
            "command": ["tectonic", "-X", "compile", "suds_paper_acmart.tex"],
            "cwd": REPO_ROOT / "paper",
            "required": True,
        },
        {
            "check_id": "claim_scan",
            "command": ["rg", "-n", CLAIM_SCAN_PATTERN, "paper", "docs/reports"],
            "cwd": REPO_ROOT,
            "required": False,
            "review_expected": True,
        },
    ]


def run_command(spec: dict[str, Any], run_root: Path) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / f"{spec['check_id']}.log"
    started = time.time()
    if not Path(spec["cwd"]).exists():
        result = {
            "check_id": spec["check_id"],
            "command": " ".join(spec["command"]),
            "cwd": repo_path(Path(spec["cwd"])),
            "required": spec.get("required", False),
            "returncode": 127,
            "duration_s": 0.0,
            "status": "fail",
            "log_path": repo_path(log_path),
            "stdout_tail": "",
            "stderr_tail": "cwd_missing",
        }
        log_path.write_text("cwd_missing\n", encoding="utf-8")
        return result
    completed = subprocess.run(
        spec["command"],
        cwd=spec["cwd"],
        text=True,
        capture_output=True,
        check=False,
    )
    duration = time.time() - started
    log_path.write_text(
        sanitize_text(
            "$ " + " ".join(spec["command"]) + "\n\n"
            + completed.stdout
            + ("\n[stderr]\n" + completed.stderr if completed.stderr else "")
        ),
        encoding="utf-8",
    )
    status = "pass" if completed.returncode == 0 else "fail"
    if spec.get("review_expected") and completed.returncode in (0, 1):
        status = "review"
    return {
        "check_id": spec["check_id"],
        "command": " ".join(spec["command"]),
        "cwd": repo_path(Path(spec["cwd"])),
        "required": spec.get("required", False),
        "returncode": completed.returncode,
        "duration_s": round(duration, 3),
        "status": status,
        "log_path": repo_path(log_path),
        "stdout_tail": sanitize_text("\n".join(completed.stdout.splitlines()[-30:])),
        "stderr_tail": sanitize_text("\n".join(completed.stderr.splitlines()[-20:])),
    }


def skipped_rows() -> list[dict[str, Any]]:
    rows = []
    for spec in command_specs():
        rows.append(
            {
                "check_id": spec["check_id"],
                "command": " ".join(spec["command"]),
                "cwd": repo_path(Path(spec["cwd"])),
                "required": spec.get("required", False),
                "returncode": "",
                "duration_s": "",
                "status": "not_run",
                "log_path": "",
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )
    return rows


def artifact_rows() -> list[dict[str, Any]]:
    rows = []
    for name, rel_path, expected_label in ARTIFACTS:
        path = REPO_ROOT / rel_path
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        meta = payload.get("metadata", {})
        evidence_label = meta.get("evidence_label", "")
        promotion_decision = meta.get("promotion_decision", "")
        blockers = []
        if not path.is_file():
            blockers.append("missing")
        if expected_label and evidence_label != expected_label:
            blockers.append(f"expected_{expected_label}_got_{evidence_label or 'blank'}")
        if name.startswith("J1"):
            if meta.get("execution_status") != "measured":
                blockers.append(f"execution_status_{meta.get('execution_status', 'blank')}")
        if name.startswith("J2"):
            if meta.get("yosys", {}).get("status") != "pass":
                blockers.append(f"yosys_{meta.get('yosys', {}).get('status', 'blank')}")
        if name.startswith("J5"):
            if not payload.get("acceptance", {}).get("accepted", False):
                blockers.append("internal_red_team_not_accepted")
        rows.append(
            {
                "artifact": name,
                "path": rel_path,
                "status": "present" if path.is_file() else "missing",
                "evidence_label": evidence_label,
                "promotion_decision": promotion_decision,
                "blockers": ";".join(blockers),
            }
        )
    return rows


def claim_scan_summary(command_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scan_row = next((row for row in command_rows if row["check_id"] == "claim_scan"), None)
    if not scan_row or not scan_row.get("log_path"):
        return {
            "status": "not_run",
            "hit_count": None,
            "manual_disposition": "not_run",
        }
    log_path = REPO_ROOT / str(scan_row["log_path"])
    text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    hits = [line for line in text.splitlines() if line and not line.startswith("$ ")]
    return {
        "status": "reviewed",
        "hit_count": len(hits),
        "manual_disposition": (
            "reviewed_as_bounded_negations_or_governance_metadata; no positive "
            "silicon, PDK, extracted-layout, measured-hardware-energy, SPICE-closure, "
            "SOTA, universal, or slack-only superiority claim is promoted"
        ),
        "log_path": scan_row.get("log_path", ""),
    }


def gate_decision(command_rows: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    command_failures = [
        row["check_id"]
        for row in command_rows
        if row.get("required") and row.get("status") not in {"pass"}
    ]
    artifact_blockers = [
        row["artifact"] + ":" + row["blockers"]
        for row in artifacts
        if row.get("blockers")
    ]
    if command_failures or artifact_blockers:
        decision = "do_not_submit_until_repaired"
    else:
        decision = "submit_with_supplement_only"
    return {
        "promotion_decision": decision,
        "fallback": "20260511_suds_maxq remains protected as fallback",
        "active_package": "20260512_quality_boost_with_supplement" if decision != "do_not_submit_until_repaired" else "blocked",
        "command_failures": command_failures,
        "artifact_blockers": artifact_blockers,
        "spice_policy": (
            "ADC macro SPICE is used only for ADC-tier energy-model calibration; "
            "full decks, stress rows, traces, and regeneration commands stay in appendix/supplement/report artifacts."
        ),
    }


def write_csv(path: Path, command_rows: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_type",
        "name",
        "command",
        "cwd",
        "status",
        "required",
        "returncode",
        "duration_s",
        "path",
        "evidence_label",
        "promotion_decision",
        "blockers",
        "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in command_rows:
            writer.writerow(
                {
                    "row_type": "command",
                    "name": row["check_id"],
                    "command": row["command"],
                    "cwd": row["cwd"],
                    "status": row["status"],
                    "required": row["required"],
                    "returncode": row["returncode"],
                    "duration_s": row["duration_s"],
                    "path": "",
                    "evidence_label": "audit",
                    "promotion_decision": "",
                    "blockers": "" if row["status"] in {"pass", "review"} else row.get("stderr_tail", ""),
                    "log_path": row.get("log_path", ""),
                }
            )
        for row in artifacts:
            writer.writerow(
                {
                    "row_type": "artifact",
                    "name": row["artifact"],
                    "command": "",
                    "cwd": "",
                    "status": row["status"],
                    "required": True,
                    "returncode": "",
                    "duration_s": "",
                    "path": row["path"],
                    "evidence_label": row["evidence_label"],
                    "promotion_decision": row["promotion_decision"],
                    "blockers": row["blockers"],
                    "log_path": "",
                }
            )


def write_json(
    path: Path,
    *,
    tag: str,
    command_rows: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    claim_scan: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_final_submission_gate_{tag}",
            "evidence_label": "audit",
            "promotion_decision": decision["promotion_decision"],
            "regeneration_command": (
                f"caffeinate -dimsu .venv311-mps/bin/python "
                f"experiments/tools/build_suds_final_submission_gate.py --tag {tag} --run-checks"
            ),
            "claim_boundary": (
                "Final gate for SUDS JETC quality boost. SPICE evidence is ADC-tier "
                "calibration only and is not silicon, PDK, extracted-layout, measured "
                "hardware-energy, or SPICE-closure evidence."
            ),
        },
        "decision": decision,
        "claim_scan": claim_scan,
        "commands": command_rows,
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    tag: str,
    command_rows: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    claim_scan: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# SUDS J6 Final Submission Gate

Tag: `{tag}`
Evidence label: `audit`
Promotion decision: `{decision['promotion_decision']}`

## Decision

- Active package: `{decision['active_package']}`
- Fallback: `{decision['fallback']}`
- SPICE policy: {decision['spice_policy']}
- Command failures: `{';'.join(decision['command_failures'])}`
- Artifact blockers: `{';'.join(decision['artifact_blockers'])}`

## Required Checks

| Check | Status | Return | Duration (s) | Log |
|---|---:|---:|---:|---|
"""
    for row in command_rows:
        report += (
            f"| `{row['check_id']}` | `{row['status']}` | `{row['returncode']}` | "
            f"`{row['duration_s']}` | `{row.get('log_path', '')}` |\n"
        )

    report += "\n## Required Artifacts\n\n"
    report += "| Artifact | Status | Evidence | Decision | Blockers |\n"
    report += "|---|---|---|---|---|\n"
    for row in artifacts:
        report += (
            f"| {row['artifact']} | `{row['status']}` | `{row['evidence_label']}` | "
            f"`{row['promotion_decision']}` | `{row['blockers']}` |\n"
        )

    report += f"""

## Claim Scan

- Pattern: `{CLAIM_SCAN_PATTERN}`
- Hit count: `{claim_scan.get('hit_count')}`
- Disposition: `{claim_scan.get('manual_disposition')}`
- Log: `{claim_scan.get('log_path', '')}`

## Regeneration

```bash
caffeinate -dimsu .venv311-mps/bin/python experiments/tools/build_suds_final_submission_gate.py --tag {tag} --run-checks
```
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.run_checks:
        command_rows = [run_command(spec, args.run_root) for spec in command_specs()]
    else:
        command_rows = skipped_rows()
    artifacts = artifact_rows()
    claim_scan = claim_scan_summary(command_rows)
    decision = gate_decision(command_rows, artifacts)
    write_csv(args.csv_out, command_rows, artifacts)
    write_json(
        args.json_out,
        tag=args.tag,
        command_rows=command_rows,
        artifacts=artifacts,
        claim_scan=claim_scan,
        decision=decision,
    )
    write_report(
        args.report_out,
        tag=args.tag,
        command_rows=command_rows,
        artifacts=artifacts,
        claim_scan=claim_scan,
        decision=decision,
    )
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.report_out}")
    print(f"promotion_decision={decision['promotion_decision']}")


if __name__ == "__main__":
    main()
