"""Path-policy helpers for keeping main-project artifacts out of archived subprojects."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVED_SUBPROJECT_ROOTS = {"AICAS"}
MAIN_PROJECT_REPORT_DATA_DIR = ROOT / "experiments" / "results" / "report_data"
MAIN_PROJECT_REPORT_FIG_DIR = ROOT / "experiments" / "results" / "report_figures"
MAIN_PROJECT_REPORT_TABLE_DIR = ROOT / "experiments" / "results" / "report_tables"


def resolve_repo_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def assert_main_project_path(
    path: Path | str,
    *,
    arg_name: str,
    allow_archived_subproject: bool = False,
) -> Path:
    resolved = resolve_repo_path(path)
    try:
        rel = resolved.relative_to(ROOT)
    except ValueError:
        return resolved
    if not rel.parts:
        return resolved
    if rel.parts[0] in ARCHIVED_SUBPROJECT_ROOTS and not allow_archived_subproject:
        raise SystemExit(
            f"{arg_name} points into archived subproject '{rel.parts[0]}': {resolved}. "
            "Main-project report builders must read/write under experiments/results/* unless "
            "--allow_archived_subproject_path is set explicitly for a packaging-only action."
        )
    return resolved
