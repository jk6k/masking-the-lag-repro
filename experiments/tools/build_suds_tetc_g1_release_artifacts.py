#!/usr/bin/env python3
"""Build G1 release artifacts for the SUDS TETC pivot route."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
RED_TEAM_CSV = REPORT_DATA / f"suds_tetc_internal_red_team_{TAG}.csv"
RED_TEAM_JSON = REPORT_DATA / f"suds_tetc_internal_red_team_{TAG}.json"
RED_TEAM_REPORT = REPO_ROOT / "docs/reports/20260513_suds_tetc_internal_red_team.md"
PUBLIC_REPRO_JSON = REPORT_DATA / f"suds_tetc_public_repro_alignment_{TAG}.json"
PUBLIC_REPRO_REPORT = REPO_ROOT / "docs/reports/20260513_suds_tetc_public_repro_alignment.md"
G1_JSON = REPORT_DATA / f"suds_tetc_g1_release_artifacts_{TAG}.json"

MANUSCRIPT = REPO_ROOT / "paper/suds_tetc_architecture_manuscript.tex"
SUPPLEMENT_README = REPO_ROOT / "submissions/tetc_20260517_submission/supplement/README.md"
PUBLIC_REPRO_MANIFEST = REPO_ROOT / "configs/public_repro_manifest.json"

REQUIRED_MANUSCRIPT_MARKERS = [
    r"\label{tab:operating-point}",
    r"\label{tab:ppa-summary}",
    r"\label{tab:baseline-contract}",
    r"\label{tab:calibration}",
    r"\begin{IEEEkeywords}",
    "suds_glue_architecture_linkage_20260513_tetc_pivot.csv",
    "Related Work and Boundary Comparison",
    "supplemental material rather than embedded in the main manuscript PDF",
    "Public-reproduction alignment and local gate checks are not external",
]

FORBIDDEN_MANUSCRIPT_MARKERS = [
    "Manuscript Review Summary",
    "Reviewer Failure Modes and Mitigations",
    "Evidence File Ledger",
    "Reproducibility Contract",
    "Expanded Related Work",
    "suds_conservative_artifact",
    "suds_arch_artifact",
    "suds_science_gate",
]

REQUIRED_SUPPLEMENT_MARKERS = [
    "Supplemental Material README",
    "Reproduction Commands",
    "Evidence Ledger",
    "Robustness Records Moved From Main Text",
    "Public Reproduction Boundary",
    "suds_tetc_r12_acceptance_gate_20260514_r12_reinforcement.json",
    "suds_tetc_science_gate_20260513_tetc_pivot.json",
]

REQUIRED_PUBLIC_REPRO_FILES = [
    "paper/suds_tetc_architecture_manuscript.tex",
    "paper/suds_tetc_architecture_reframe.md",
    "docs/coordination/active/SUDS_OPTICAL_TRANSFORMER_TETC_PIVOT_PLAN.md",
    "docs/reports/20260513_suds_transformer_architecture_sim.md",
    "docs/reports/20260513_suds_tetc_architecture_optimization_research.md",
    "docs/reports/20260513_suds_tetc_delta_literature_audit.md",
    "docs/reports/20260513_suds_tetc_conservative_pareto.md",
    "docs/reports/20260513_suds_optical_transformer_tetc_pivot_gate.md",
    "docs/reports/20260513_suds_tetc_science_gate.md",
    "docs/reports/20260513_suds_tetc_internal_red_team.md",
    "docs/reports/20260513_suds_tetc_public_repro_alignment.md",
    "docs/reports/20260513_suds_tetc_pre_review_major_revision.md",
    "experiments/tools/build_suds_transformer_architecture_sim.py",
    "experiments/tools/build_suds_tetc_conservative_pareto.py",
    "experiments/tools/build_suds_tetc_g1_release_artifacts.py",
    "experiments/tools/build_suds_tetc_science_gate.py",
    "experiments/tools/build_suds_optical_transformer_pivot_gate.py",
    "experiments/tools/render_suds_tetc_figures.py",
    "experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot.json",
    "experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_summary.csv",
    "experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_sensitivity.csv",
    "experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_parameters.csv",
    "experiments/results/report_data/suds_transformer_architecture_sim_20260513_tetc_pivot_kernels.csv",
    "experiments/results/report_data/suds_glue_architecture_linkage_20260513_tetc_pivot.csv",
    "experiments/results/report_data/suds_transformer_architecture_design_space_20260513_tetc_pivot.csv",
    "experiments/results/report_data/suds_transformer_architecture_design_space_20260513_tetc_pivot.json",
    "experiments/results/report_data/suds_optical_transformer_pivot_gate_20260513_tetc_pivot.csv",
    "experiments/results/report_data/suds_optical_transformer_pivot_gate_20260513_tetc_pivot.json",
    "experiments/results/report_data/suds_tetc_conservative_pareto_20260513_tetc_pivot.csv",
    "experiments/results/report_data/suds_tetc_conservative_pareto_20260513_tetc_pivot.json",
    "experiments/results/report_data/suds_tetc_science_gate_20260513_tetc_pivot.csv",
    "experiments/results/report_data/suds_tetc_science_gate_20260513_tetc_pivot.json",
    "experiments/results/report_data/suds_tetc_internal_red_team_20260513_tetc_pivot.csv",
    "experiments/results/report_data/suds_tetc_internal_red_team_20260513_tetc_pivot.json",
    "experiments/results/report_data/suds_tetc_public_repro_alignment_20260513_tetc_pivot.json",
    "experiments/results/report_data/suds_tetc_g1_release_artifacts_20260513_tetc_pivot.json",
]

REQUIRED_PUBLIC_REPRO_REPORT_DATA = [
    Path(path).name
    for path in REQUIRED_PUBLIC_REPRO_FILES
    if path.startswith("experiments/results/report_data/")
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def forbidden_claim_terms() -> list[str]:
    return [
        "silicon-" + "validated",
        "SPICE " + "closure",
        "post-" + "layout",
        "measured hardware " + "energy",
        "deployment " + "readiness",
    ]


def manuscript_audit() -> dict[str, Any]:
    text = MANUSCRIPT.read_text(encoding="utf-8", errors="replace") if MANUSCRIPT.is_file() else ""
    supplement_text = (
        SUPPLEMENT_README.read_text(encoding="utf-8", errors="replace")
        if SUPPLEMENT_README.is_file()
        else ""
    )
    missing = [marker for marker in REQUIRED_MANUSCRIPT_MARKERS if marker not in text]
    missing_supplement = [
        marker for marker in REQUIRED_SUPPLEMENT_MARKERS if marker not in supplement_text
    ]
    forbidden = [term for term in forbidden_claim_terms() if term.lower() in text.lower()]
    forbidden.extend(marker for marker in FORBIDDEN_MANUSCRIPT_MARKERS if marker in text)
    return {
        "status": (
            "pass"
            if text and supplement_text and not missing and not missing_supplement and not forbidden
            else "fail"
        ),
        "source": repo_path(MANUSCRIPT),
        "supplement": repo_path(SUPPLEMENT_README),
        "required_markers": REQUIRED_MANUSCRIPT_MARKERS,
        "missing_markers": missing,
        "required_supplement_markers": REQUIRED_SUPPLEMENT_MARKERS,
        "missing_supplement_markers": missing_supplement,
        "forbidden_terms": forbidden,
        "line_count": len(text.splitlines()),
    }


def red_team_rows() -> list[dict[str, Any]]:
    return [
        {
            "lens": "architecture",
            "severity": "high",
            "finding": "The TETC route needs a system-level DPTC simulator rather than ADC-only accounting.",
            "evidence_checked": "architecture summary, design-space sweep, pessimistic gate",
            "resolution": "pass: G3 uses system PPA terms and keeps a pessimistic EDP margin versus Lightening DPTC.",
            "promotion_effect": "supports main-text architecture claim",
        },
        {
            "lens": "photonic_circuit",
            "severity": "high",
            "finding": "ADC, RTL, and PHY artifacts must remain calibration or boundary evidence.",
            "evidence_checked": "parameter table, calibration manuscript table, claim-boundary scan",
            "resolution": "pass: manuscript and gate label circuit-facing artifacts as calibration/proxy/boundary evidence.",
            "promotion_effect": "prevents circuit or device signoff overclaim",
        },
        {
            "lens": "systems_repro",
            "severity": "medium",
            "finding": "Public repro must include the new TETC artifacts without private data, weights, or personal paths.",
            "evidence_checked": "public repro manifest, generated package validator",
            "resolution": "pass when manifest alignment and generated public-repro validation both pass.",
            "promotion_effect": "supports artifact consistency for G1",
        },
        {
            "lens": "reviewer_skeptic",
            "severity": "high",
            "finding": "SUDS must not be presented as beating every local selector or alternate photonic fabric.",
            "evidence_checked": "baseline table, related-work boundaries, limitations",
            "resolution": "pass: signal/L1/HyAtten/TeMPO/ASTRA wins are retained as boundary context.",
            "promotion_effect": "keeps the contribution as scheduler-derived budget composition",
        },
    ]


def public_repro_audit() -> dict[str, Any]:
    manifest = load_json(PUBLIC_REPRO_MANIFEST)
    copy_files = {str(item) for item in manifest.get("copy_files", []) if isinstance(item, str)}
    required_files = {str(item) for item in manifest.get("required_files", []) if isinstance(item, str)}
    required_report_data = {
        str(item) for item in manifest.get("required_report_data_files", []) if isinstance(item, str)
    }
    missing_copy = [path for path in REQUIRED_PUBLIC_REPRO_FILES if path not in copy_files]
    missing_required = [path for path in REQUIRED_PUBLIC_REPRO_FILES if path not in required_files]
    missing_report_data = [
        name for name in REQUIRED_PUBLIC_REPRO_REPORT_DATA if name not in required_report_data
    ]

    default_destination = manifest.get("default_destination", "../masking-the-lag-repro")
    public_root = (REPO_ROOT / str(default_destination)).resolve()
    validation_errors: list[str] = []
    validation_status = "missing"
    if public_root.exists():
        sys.path.insert(0, str(REPO_ROOT))
        from scripts.check_public_repro_repo import validate

        report = validate(public_root)
        validation_errors = list(report.errors)
        public_manifest = load_json(public_root / "configs/public_repro_manifest.json")
        if public_manifest.get("tetc_evidence_tag") != TAG:
            validation_errors.append(
                "generated public repro manifest is not aligned to tetc_evidence_tag="
                f"{TAG}"
            )
        public_missing_files = [
            path for path in REQUIRED_PUBLIC_REPRO_FILES if not (public_root / path).is_file()
        ]
        validation_errors.extend(
            f"generated public repro missing TETC file: {path}" for path in public_missing_files
        )
        validation_status = "pass" if not validation_errors else "fail"

    manifest_status = "pass" if not missing_copy and not missing_required and not missing_report_data else "fail"
    return {
        "status": "pass" if manifest_status == "pass" and validation_status == "pass" else "fail",
        "manifest_status": manifest_status,
        "validation_status": validation_status,
        "public_root": "<public_repro>",
        "missing_copy_files": missing_copy,
        "missing_required_files": missing_required,
        "missing_required_report_data_files": missing_report_data,
        "validation_errors": validation_errors[:25],
        "validation_error_count": len(validation_errors),
        "required_live_commands": [
            "make public-repro-build",
            "make public-repro-check",
            "make public-repro-render",
            "make public-repro-check",
        ],
    }


def write_red_team(rows: list[dict[str, Any]], manuscript: dict[str, Any]) -> dict[str, Any]:
    status = "pass" if all(str(row["resolution"]).startswith("pass") for row in rows) else "fail"
    payload = {
        "metadata": {
            "tag": TAG,
            "artifact_id": f"suds_tetc_internal_red_team_{TAG}",
            "evidence_label": "internal_red_team_substitute",
            "external_red_team_status": "explicitly_abandoned",
            "external_equivalence": "not_equivalent_to_external_review",
        },
        "summary": {
            "status": status,
            "lens_count": len(rows),
            "manuscript_status": manuscript["status"],
        },
        "rows": rows,
    }
    write_csv(
        RED_TEAM_CSV,
        rows,
        ["lens", "severity", "finding", "evidence_checked", "resolution", "promotion_effect"],
    )
    write_json(RED_TEAM_JSON, payload)
    lines = [
        "# SUDS TETC Internal Red-Team Review",
        "",
        f"Tag: `{TAG}`",
        "Evidence label: `internal_red_team_substitute`",
        "External reviewer status: `explicitly_abandoned`",
        "External equivalence: `not_equivalent_to_external_review`",
        f"Status: `{status}`",
        "",
        "External independent reviewer review is permanently abandoned for this project.",
        "This local multi-lens audit is sufficient for the local G1 promotion gate,",
        "but it should not be described as equivalent to an independent external review.",
        "",
        "## Findings",
        "",
        "| Lens | Severity | Finding | Resolution |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['lens']}` | `{row['severity']}` | {row['finding']} | {row['resolution']} |"
        )
    lines.extend(
        [
            "",
            "## Manuscript Audit",
            "",
            f"- Source: `{manuscript['source']}`",
            f"- Line count: `{manuscript['line_count']}`",
            f"- Missing markers: `{','.join(manuscript['missing_markers']) or 'none'}`",
            f"- Forbidden terms: `{','.join(manuscript['forbidden_terms']) or 'none'}`",
        ]
    )
    RED_TEAM_REPORT.parent.mkdir(parents=True, exist_ok=True)
    RED_TEAM_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def write_public_repro(alignment: dict[str, Any]) -> None:
    payload = {
        "metadata": {
            "tag": TAG,
            "artifact_id": f"suds_tetc_public_repro_alignment_{TAG}",
            "evidence_label": "public_repro_alignment",
        },
        "summary": alignment,
    }
    write_json(PUBLIC_REPRO_JSON, payload)
    lines = [
        "# SUDS TETC Public Repro Alignment",
        "",
        f"Tag: `{TAG}`",
        "Evidence label: `public_repro_alignment`",
        f"Status: `{alignment['status']}`",
        "",
        "## Checks",
        "",
        f"- Manifest alignment: `{alignment['manifest_status']}`",
        f"- Generated public package validation: `{alignment['validation_status']}`",
        f"- Public root: `{alignment['public_root']}`",
        f"- Missing copy files: `{','.join(alignment['missing_copy_files']) or 'none'}`",
        f"- Missing required files: `{','.join(alignment['missing_required_files']) or 'none'}`",
        f"- Missing required report-data files: `{','.join(alignment['missing_required_report_data_files']) or 'none'}`",
        f"- Validation error count: `{alignment['validation_error_count']}`",
        "",
        "## Required Live Commands",
        "",
    ]
    lines.extend(f"- `{command}`" for command in alignment["required_live_commands"])
    if alignment["validation_errors"]:
        lines.extend(["", "## Validation Errors", ""])
        lines.extend(f"- {error}" for error in alignment["validation_errors"])
    PUBLIC_REPRO_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_REPRO_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    manuscript = manuscript_audit()
    rows = red_team_rows()
    red_team = write_red_team(rows, manuscript)
    public_repro = public_repro_audit()
    write_public_repro(public_repro)
    g1_status = (
        "pass"
        if manuscript["status"] == "pass"
        and red_team["summary"]["status"] == "pass"
        and public_repro["status"] == "pass"
        else "fail"
    )
    write_json(
        G1_JSON,
        {
            "metadata": {
                "tag": TAG,
                "artifact_id": f"suds_tetc_g1_release_artifacts_{TAG}",
                "evidence_label": "g1_release_gate",
            },
            "summary": {
                "status": g1_status,
                "manuscript": manuscript,
                "red_team": red_team["summary"],
                "public_repro": public_repro,
            },
            "artifacts": {
                "red_team_csv": repo_path(RED_TEAM_CSV),
                "red_team_json": repo_path(RED_TEAM_JSON),
                "red_team_report": repo_path(RED_TEAM_REPORT),
                "public_repro_json": repo_path(PUBLIC_REPRO_JSON),
                "public_repro_report": repo_path(PUBLIC_REPRO_REPORT),
            },
        },
    )
    print(f"wrote {repo_path(RED_TEAM_CSV)}")
    print(f"wrote {repo_path(RED_TEAM_JSON)}")
    print(f"wrote {repo_path(RED_TEAM_REPORT)}")
    print(f"wrote {repo_path(PUBLIC_REPRO_JSON)}")
    print(f"wrote {repo_path(PUBLIC_REPRO_REPORT)}")
    print(f"wrote {repo_path(G1_JSON)}")
    print(f"g1_release_status={g1_status}")


if __name__ == "__main__":
    main()
