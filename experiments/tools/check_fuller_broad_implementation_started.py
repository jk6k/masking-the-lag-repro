#!/usr/bin/env python3
"""Validate the governed broader fuller-implementation started contract."""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path
from typing import Any

import yaml

try:
    from . import phase1_matrix_runner
    from .check_fuller_broader_implementation_extension_readiness import (
        _load_yaml as _load_extension_yaml,
        evaluate_contract as evaluate_extension_contract,
    )
    from .path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path
except ImportError:
    import phase1_matrix_runner  # type: ignore
    from check_fuller_broader_implementation_extension_readiness import (  # type: ignore
        _load_yaml as _load_extension_yaml,
        evaluate_contract as evaluate_extension_contract,
    )
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_broad_implementation_started_contract_20260319.yaml"
DEFAULT_OUT_DIR = MAIN_PROJECT_REPORT_DATA_DIR


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON mapping in {path}")
    return data


def _require_mapping(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Missing required mapping: {key}")
    return value


def _require_list(cfg: dict[str, Any], key: str) -> list[str]:
    value = cfg.get(key)
    if not isinstance(value, list) or not value:
        raise SystemExit(f"Missing required list: {key}")
    return [str(item) for item in value]


def _record(rows: list[dict[str, str]], check_id: str, status: str, detail: str) -> None:
    rows.append({"check_id": check_id, "status": status, "detail": detail})


def evaluate_contract(contract: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checks: list[dict[str, str]] = []

    meta = _require_mapping(contract, "meta")
    upstream = _require_mapping(contract, "upstream")
    execution_policy = _require_mapping(contract, "execution_policy")
    smoke_run = _require_mapping(contract, "smoke_run")
    state = _require_mapping(contract, "state")

    extension_contract_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("extension_readiness_contract_yaml") or "")),
        arg_name="upstream.extension_readiness_contract_yaml",
    )
    extension_report_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("extension_readiness_report_md") or "")),
        arg_name="upstream.extension_readiness_report_md",
    )
    extension_note_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("extension_readiness_note_md") or "")),
        arg_name="upstream.extension_readiness_note_md",
    )
    for label, path in (
        ("upstream.extension_readiness_contract_yaml", extension_contract_path),
        ("upstream.extension_readiness_report_md", extension_report_path),
        ("upstream.extension_readiness_note_md", extension_note_path),
    ):
        _record(checks, label, "pass" if path.exists() else "fail", str(path))

    extension_contract = _load_extension_yaml(extension_contract_path)
    _, extension_summary = evaluate_extension_contract(extension_contract)
    extension_ok = bool(extension_summary.get("overall_ok"))
    extension_boundary_ok = (
        bool(extension_summary.get("broader_extension_lane_open"))
        and not bool(extension_summary.get("broad_matrix_allowed"))
        and not bool(extension_summary.get("cross_mechanism_wave_allowed"))
        and not bool(extension_summary.get("det_sparse_reopen_allowed"))
    )
    _record(
        checks,
        "upstream.extension_contract_status",
        "pass" if extension_ok else "fail",
        f"overall_ok={extension_summary.get('overall_ok')!r}",
    )
    _record(
        checks,
        "upstream.extension_boundary_status",
        "pass" if extension_boundary_ok else "fail",
        (
            f"broader_extension_lane_open={extension_summary.get('broader_extension_lane_open')!r} "
            f"broad_matrix_allowed={extension_summary.get('broad_matrix_allowed')!r} "
            f"cross_mechanism_wave_allowed={extension_summary.get('cross_mechanism_wave_allowed')!r} "
            f"det_sparse_reopen_allowed={extension_summary.get('det_sparse_reopen_allowed')!r}"
        ),
    )

    _record(
        checks,
        "execution_policy.device",
        "pass" if str(execution_policy.get("device")) == "mps" else "fail",
        f"device={execution_policy.get('device')!r}",
    )
    _record(
        checks,
        "execution_policy.execution_surface",
        "pass"
        if str(execution_policy.get("execution_surface")) == "host_unsandboxed_caffeinate_required"
        else "fail",
        f"execution_surface={execution_policy.get('execution_surface')!r}",
    )
    required_launch_prefix = _require_list(execution_policy, "required_launch_prefix")
    _record(
        checks,
        "execution_policy.required_launch_prefix",
        "pass" if required_launch_prefix == ["caffeinate", "-dimsu"] else "fail",
        repr(required_launch_prefix),
    )

    python_bin = str(smoke_run.get("python_bin") or "").strip()
    top_level_runner = assert_main_project_path(
        resolve_repo_path(str(smoke_run.get("top_level_runner_py") or "")),
        arg_name="smoke_run.top_level_runner_py",
    )
    template_yaml = assert_main_project_path(
        resolve_repo_path(str(smoke_run.get("template_yaml") or "")),
        arg_name="smoke_run.template_yaml",
    )
    generated_config = assert_main_project_path(
        resolve_repo_path(str(smoke_run.get("generated_config_yaml") or "")),
        arg_name="smoke_run.generated_config_yaml",
    )
    expected_run_dir = assert_main_project_path(
        resolve_repo_path(str(smoke_run.get("expected_run_dir") or "")),
        arg_name="smoke_run.expected_run_dir",
    )
    for label, path in (
        ("smoke_run.top_level_runner_py", top_level_runner),
        ("smoke_run.template_yaml", template_yaml),
        ("smoke_run.generated_config_yaml", generated_config),
        ("smoke_run.expected_run_dir", expected_run_dir),
    ):
        _record(checks, label, "pass" if path.exists() else "fail", str(path))

    top_level_command = [
        python_bin,
        str(top_level_runner),
        "--template",
        str(template_yaml),
        "--experiments",
        str(smoke_run.get("experiments_arg") or ""),
        "--run_prefix",
        str(smoke_run.get("run_prefix") or ""),
    ]
    _record(
        checks,
        "smoke_run.top_level_command_shape",
        "pass"
        if python_bin and str(smoke_run.get("experiments_arg") or "") == "all"
        else "fail",
        shlex.join(top_level_command),
    )

    generated_payload = _load_yaml(generated_config)
    generated_run = _require_mapping(generated_payload, "run")
    generated_switches = _require_mapping(generated_payload, "switches")
    _record(
        checks,
        "smoke_run.generated_config_state",
        "pass"
        if (
            str(generated_run.get("experiment_id") or "").upper() == str(smoke_run.get("expected_experiment_id") or "").upper()
            and str(generated_run.get("device") or "").lower() == "mps"
            and str(generated_run.get("execution_surface") or "") == "host_unsandboxed_caffeinate_required"
            and generated_switches == {"meso": True, "flow": True, "det": False, "sparse": False, "phy": True}
        )
        else "fail",
        (
            f"run={generated_run!r} "
            f"switches={generated_switches!r}"
        ),
    )

    child_command = phase1_matrix_runner._build_phase1_command(
        python_bin=python_bin,
        cfg_path=generated_config,
        cfg=generated_payload,
    )
    _record(
        checks,
        "smoke_run.child_command_runtime_policy",
        "pass" if child_command[:2] == required_launch_prefix else "fail",
        shlex.join(child_command),
    )

    required_run_files = _require_list(smoke_run, "required_run_files")
    for relative_name in required_run_files:
        path = expected_run_dir / relative_name
        _record(
            checks,
            f"smoke_run.required_run_file:{relative_name}",
            "pass" if path.exists() else "fail",
            str(path),
        )

    run_metadata = _load_json(expected_run_dir / "run_metadata.json")
    metadata_run = _require_mapping(run_metadata, "run")
    metadata_switches = _require_mapping(run_metadata, "switches")
    _record(
        checks,
        "smoke_run.run_metadata_state",
        "pass"
        if (
            str(metadata_run.get("experiment_id") or "").upper() == str(smoke_run.get("expected_experiment_id") or "").upper()
            and str(metadata_run.get("device") or "").lower() == "mps"
            and metadata_switches == {"meso": True, "flow": True, "det": False, "sparse": False, "phy": True}
        )
        else "fail",
        f"run={metadata_run!r} switches={metadata_switches!r}",
    )

    summary_csv = expected_run_dir / "phase1_summary.csv"
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary_row = summary_rows[0] if summary_rows else {}
    summary_ok = (
        len(summary_rows) >= 1
        and str(summary_row.get("experiment_id") or "").upper() == str(smoke_run.get("expected_experiment_id") or "").upper()
        and str(summary_row.get("meso_cost_model_mode") or "") == "explicit_topology_v1"
        and str(summary_row.get("integrated_system_cost_mode") or "") == "integrated_minimal_v1"
        and str(summary_row.get("phy_penalty_table_version") or "") == "parametric-v1"
    )
    _record(
        checks,
        "smoke_run.phase1_summary_state",
        "pass" if summary_ok else "fail",
        f"rows={len(summary_rows)} first_row={summary_row!r}",
    )

    extension_readiness_required = bool(state.get("extension_readiness_required"))
    orchestrator_runtime_guard_required = bool(state.get("orchestrator_runtime_guard_required"))
    host_unsandboxed_mps_smoke_required = bool(state.get("host_unsandboxed_mps_smoke_required"))
    broader_fuller_implementation_started = bool(state.get("broader_fuller_implementation_started"))
    broad_matrix_allowed = bool(state.get("broad_matrix_allowed"))
    cross_mechanism_wave_allowed = bool(state.get("cross_mechanism_wave_allowed"))
    det_sparse_reopen_allowed = bool(state.get("det_sparse_reopen_allowed"))
    coherence = (
        extension_readiness_required
        and extension_ok
        and extension_boundary_ok
        and orchestrator_runtime_guard_required
        and child_command[:2] == required_launch_prefix
        and host_unsandboxed_mps_smoke_required
        and summary_ok
        and broader_fuller_implementation_started
        and not broad_matrix_allowed
        and not cross_mechanism_wave_allowed
        and not det_sparse_reopen_allowed
    )
    _record(
        checks,
        "state.coherence",
        "pass" if coherence else "fail",
        (
            f"extension_readiness_required={extension_readiness_required} "
            f"extension_ok={extension_ok} "
            f"extension_boundary_ok={extension_boundary_ok} "
            f"orchestrator_runtime_guard_required={orchestrator_runtime_guard_required} "
            f"host_unsandboxed_mps_smoke_required={host_unsandboxed_mps_smoke_required} "
            f"broader_fuller_implementation_started={broader_fuller_implementation_started} "
            f"broad_matrix_allowed={broad_matrix_allowed} "
            f"cross_mechanism_wave_allowed={cross_mechanism_wave_allowed} "
            f"det_sparse_reopen_allowed={det_sparse_reopen_allowed}"
        ),
    )

    summary = {
        "tag": str(meta.get("tag") or "unknown"),
        "started": broader_fuller_implementation_started,
        "broad_matrix_allowed": broad_matrix_allowed,
        "cross_mechanism_wave_allowed": cross_mechanism_wave_allowed,
        "overall_ok": all(row["status"] == "pass" for row in checks),
    }
    return checks, summary


def _write_report(path: Path, checks: list[dict[str, str]], summary: dict[str, Any], contract_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fuller Broad-Implementation Started Report",
        "",
        "Scope",
        f"- Contract: `{contract_path}`",
        f"- Tag: `{summary['tag']}`",
        "",
        "State",
        f"- Broader fuller implementation started: `{summary['started']}`",
        f"- Broad matrix allowed: `{summary['broad_matrix_allowed']}`",
        f"- Cross-mechanism wave allowed: `{summary['cross_mechanism_wave_allowed']}`",
        "",
        "Checks",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the governed broader fuller-implementation started contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    contract_path = assert_main_project_path(resolve_repo_path(args.contract), arg_name="--contract")
    out_dir = assert_main_project_path(resolve_repo_path(args.out_dir), arg_name="--out_dir")
    contract = _load_yaml(contract_path)
    checks, summary = evaluate_contract(contract)
    report_path = out_dir / f"fuller_broad_implementation_started_{summary['tag']}.md"
    _write_report(report_path, checks, summary, contract_path)
    overall = "OK" if summary["overall_ok"] else "FAIL"
    print(f"[fuller-broad-implementation-started] overall={overall} report={report_path}")
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
