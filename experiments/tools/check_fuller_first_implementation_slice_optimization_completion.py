#!/usr/bin/env python3
"""Validate the bounded first-slice implementation-optimization completion contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

try:
    from .check_fuller_first_implementation_slice import (
        _load_yaml as _load_first_slice_yaml,
        evaluate_contract as evaluate_first_slice_contract,
    )
    from .check_fuller_first_slice_smoke import (
        _load_yaml as _load_smoke_yaml,
        evaluate_template as evaluate_first_slice_template,
    )
    from .path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path
except ImportError:
    from check_fuller_first_implementation_slice import (  # type: ignore
        _load_yaml as _load_first_slice_yaml,
        evaluate_contract as evaluate_first_slice_contract,
    )
    from check_fuller_first_slice_smoke import (  # type: ignore
        _load_yaml as _load_smoke_yaml,
        evaluate_template as evaluate_first_slice_template,
    )
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_first_implementation_slice_optimization_completion_contract_20260319.yaml"
DEFAULT_OUT_DIR = MAIN_PROJECT_REPORT_DATA_DIR


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
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


def _check_paths(rows: list[dict[str, str]], check_id_prefix: str, values: list[str]) -> None:
    for raw in values:
        path = assert_main_project_path(resolve_repo_path(raw), arg_name=check_id_prefix)
        _record(rows, f"{check_id_prefix}:{raw}", "pass" if path.exists() else "fail", str(path))


def evaluate_contract(contract: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    checks: list[dict[str, str]] = []

    meta = _require_mapping(contract, "meta")
    upstream = _require_mapping(contract, "upstream")
    paper_route = _require_mapping(contract, "paper_route")
    execution_policy = _require_mapping(contract, "execution_policy")
    closure_scope = _require_mapping(contract, "closure_scope")
    state = _require_mapping(contract, "state")

    upstream_preimpl_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("preimplementation_contract_yaml") or "")),
        arg_name="upstream.preimplementation_contract_yaml",
    )
    upstream_first_slice_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("first_slice_contract_yaml") or "")),
        arg_name="upstream.first_slice_contract_yaml",
    )
    upstream_readiness_report_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("first_slice_readiness_report_md") or "")),
        arg_name="upstream.first_slice_readiness_report_md",
    )
    upstream_smoke_report_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("first_slice_smoke_report_md") or "")),
        arg_name="upstream.first_slice_smoke_report_md",
    )
    upstream_start_note_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("entry_lane_start_note_md") or "")),
        arg_name="upstream.entry_lane_start_note_md",
    )
    upstream_kickoff_pack_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("kickoff_pack_md") or "")),
        arg_name="upstream.kickoff_pack_md",
    )
    upstream_preflight_board_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("launch_preflight_board_md") or "")),
        arg_name="upstream.launch_preflight_board_md",
    )
    for label, path in (
        ("upstream.preimplementation_contract_yaml", upstream_preimpl_path),
        ("upstream.first_slice_contract_yaml", upstream_first_slice_path),
        ("upstream.first_slice_readiness_report_md", upstream_readiness_report_path),
        ("upstream.first_slice_smoke_report_md", upstream_smoke_report_path),
        ("upstream.entry_lane_start_note_md", upstream_start_note_path),
        ("upstream.kickoff_pack_md", upstream_kickoff_pack_path),
        ("upstream.launch_preflight_board_md", upstream_preflight_board_path),
    ):
        _record(checks, label, "pass" if path.exists() else "fail", str(path))

    upstream_preimpl = _load_yaml(upstream_preimpl_path)
    upstream_launch_state = _require_mapping(upstream_preimpl, "launch_state")
    upstream_bounded_collection_ready = bool(
        upstream_launch_state.get("bounded_collection_ready", upstream_launch_state.get("large_scale_launch_ready"))
    )
    _record(
        checks,
        "upstream.bounded_collection_ready",
        "pass" if upstream_bounded_collection_ready else "fail",
        (
            f"bounded_collection_ready={upstream_launch_state.get('bounded_collection_ready')!r} "
            f"large_scale_launch_ready={upstream_launch_state.get('large_scale_launch_ready')!r}"
        ),
    )

    first_slice_contract = _load_first_slice_yaml(upstream_first_slice_path)
    _, first_slice_summary = evaluate_first_slice_contract(first_slice_contract)
    first_slice_ok = bool(first_slice_summary.get("overall_ok"))
    first_slice_started = (
        bool(first_slice_summary.get("implementation_started"))
        and not bool(first_slice_summary.get("broad_matrix_allowed"))
        and str(first_slice_summary.get("first_slice_state")) == "dedicated_entry_lane_smoke_started"
        and bool(first_slice_summary.get("dedicated_entry_lane_landed"))
        and bool(first_slice_summary.get("integrated_smoke_test_landed"))
    )
    _record(
        checks,
        "upstream.first_slice_contract_status",
        "pass" if first_slice_ok else "fail",
        (
            f"overall_ok={first_slice_summary.get('overall_ok')!r} "
            f"first_slice_state={first_slice_summary.get('first_slice_state')!r}"
        ),
    )
    _record(
        checks,
        "upstream.first_slice_started_state",
        "pass" if first_slice_started else "fail",
        (
            f"implementation_started={first_slice_summary.get('implementation_started')!r} "
            f"broad_matrix_allowed={first_slice_summary.get('broad_matrix_allowed')!r} "
            f"dedicated_entry_lane_landed={first_slice_summary.get('dedicated_entry_lane_landed')!r} "
            f"integrated_smoke_test_landed={first_slice_summary.get('integrated_smoke_test_landed')!r}"
        ),
    )

    first_slice = _require_mapping(first_slice_contract, "first_slice")
    template_path = assert_main_project_path(
        resolve_repo_path(str(first_slice.get("dedicated_entry_template_yaml") or "")),
        arg_name="first_slice.dedicated_entry_template_yaml",
    )
    template = _load_smoke_yaml(template_path)
    _, smoke_summary = evaluate_first_slice_template(template)
    smoke_ok = bool(smoke_summary.get("overall_ok"))
    smoke_switches = smoke_summary.get("resolved_switches")
    _record(
        checks,
        "upstream.first_slice_template_status",
        "pass" if smoke_ok else "fail",
        (
            f"overall_ok={smoke_summary.get('overall_ok')!r} "
            f"resolved_switches={smoke_switches!r}"
        ),
    )

    expected_route = "bounded local baseline + narrow HOPS paper"
    _record(
        checks,
        "paper_route.active_safe_route",
        "pass" if str(paper_route.get("active_safe_route")) == expected_route else "fail",
        f"active_safe_route={paper_route.get('active_safe_route')!r}",
    )
    _record(
        checks,
        "execution_policy.device",
        "pass" if str(execution_policy.get("device")) == "mps" else "fail",
        f"device={execution_policy.get('device')!r}",
    )
    _record(
        checks,
        "execution_policy.cpu_fallback_allowed",
        "pass" if execution_policy.get("cpu_fallback_allowed") is False else "fail",
        f"cpu_fallback_allowed={execution_policy.get('cpu_fallback_allowed')!r}",
    )
    _record(
        checks,
        "execution_policy.long_running_mps_surface",
        "pass"
        if str(execution_policy.get("long_running_mps_surface")) == "host_unsandboxed_caffeinate_required"
        else "fail",
        f"long_running_mps_surface={execution_policy.get('long_running_mps_surface')!r}",
    )
    _record(
        checks,
        "closure_scope.lane_roles",
        "pass"
        if (
            str(closure_scope.get("primary_lane")) == "HOPS"
            and str(closure_scope.get("support_lane")) == "MESO"
            and str(closure_scope.get("bounded_support")) == "PHY"
            and str(closure_scope.get("astra_substrate_role")) == "baseline_execution_substrate"
        )
        else "fail",
        (
            f"primary_lane={closure_scope.get('primary_lane')!r} "
            f"support_lane={closure_scope.get('support_lane')!r} "
            f"bounded_support={closure_scope.get('bounded_support')!r} "
            f"astra_substrate_role={closure_scope.get('astra_substrate_role')!r}"
        ),
    )
    _record(
        checks,
        "closure_scope.forbidden_reopens",
        "pass" if _require_list(closure_scope, "forbidden_reopens") == ["DET", "SPARSE"] else "fail",
        f"forbidden_reopens={closure_scope.get('forbidden_reopens')!r}",
    )

    disallowed_expansions = set(_require_list(closure_scope, "disallowed_expansions"))
    _record(
        checks,
        "closure_scope.disallowed_expansions",
        "pass"
        if disallowed_expansions
        >= {"broad_rerun_matrix", "broad_implementation_wave", "freeze_replacement", "paper_claim_broadening"}
        else "fail",
        f"disallowed_expansions={sorted(disallowed_expansions)!r}",
    )
    _check_paths(
        checks,
        "closure_scope.required_existing_artifacts",
        _require_list(closure_scope, "required_existing_artifacts"),
    )
    _check_paths(checks, "closure_scope.required_docs", _require_list(closure_scope, "required_docs"))

    upstream_bounded_collection_ready_required = bool(state.get("upstream_bounded_collection_ready_required"))
    first_slice_started_required = bool(state.get("first_slice_started_required"))
    smoke_gate_required = bool(state.get("smoke_gate_required"))
    bounded_optimization_complete = bool(state.get("bounded_optimization_complete"))
    broad_matrix_allowed = bool(state.get("broad_matrix_allowed"))
    further_broad_implementation_open = bool(state.get("further_broad_implementation_open"))
    det_sparse_reopen_allowed = bool(state.get("det_sparse_reopen_allowed"))
    coherence = (
        upstream_bounded_collection_ready_required
        and upstream_bounded_collection_ready
        and first_slice_started_required
        and first_slice_ok
        and first_slice_started
        and smoke_gate_required
        and smoke_ok
        and bounded_optimization_complete
        and not broad_matrix_allowed
        and not further_broad_implementation_open
        and not det_sparse_reopen_allowed
    )
    _record(
        checks,
        "state.coherence",
        "pass" if coherence else "fail",
        (
            f"upstream_bounded_collection_ready_required={upstream_bounded_collection_ready_required} "
            f"upstream_bounded_collection_ready={upstream_bounded_collection_ready} "
            f"first_slice_started_required={first_slice_started_required} "
            f"first_slice_ok={first_slice_ok} "
            f"first_slice_started={first_slice_started} "
            f"smoke_gate_required={smoke_gate_required} "
            f"smoke_ok={smoke_ok} "
            f"bounded_optimization_complete={bounded_optimization_complete} "
            f"broad_matrix_allowed={broad_matrix_allowed} "
            f"further_broad_implementation_open={further_broad_implementation_open} "
            f"det_sparse_reopen_allowed={det_sparse_reopen_allowed}"
        ),
    )

    summary = {
        "tag": str(meta.get("tag") or "unknown"),
        "paper_route": paper_route,
        "slice_name": str(closure_scope.get("slice_name") or ""),
        "upstream_bounded_collection_ready": upstream_bounded_collection_ready,
        "first_slice_ok": first_slice_ok,
        "first_slice_started": first_slice_started,
        "smoke_ok": smoke_ok,
        "bounded_optimization_complete": bounded_optimization_complete,
        "broad_matrix_allowed": broad_matrix_allowed,
        "further_broad_implementation_open": further_broad_implementation_open,
        "det_sparse_reopen_allowed": det_sparse_reopen_allowed,
        "overall_ok": all(row["status"] == "pass" for row in checks),
    }
    return checks, summary


def _write_report(path: Path, checks: list[dict[str, str]], summary: dict[str, Any], contract_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fuller First-Implementation Slice Optimization Completion Report",
        "",
        "Scope",
        f"- Contract: `{contract_path}`",
        f"- Tag: `{summary['tag']}`",
        f"- Active route: `{summary['paper_route']['active_safe_route']}`",
        f"- Slice: `{summary['slice_name']}`",
        "",
        "State",
        f"- Upstream bounded collection ready: `{summary['upstream_bounded_collection_ready']}`",
        f"- First-slice checker status: `{summary['first_slice_ok']}`",
        f"- First-slice started state: `{summary['first_slice_started']}`",
        f"- Smoke gate status: `{summary['smoke_ok']}`",
        f"- Bounded optimization complete: `{summary['bounded_optimization_complete']}`",
        f"- Broad matrix allowed: `{summary['broad_matrix_allowed']}`",
        f"- Further broad implementation open: `{summary['further_broad_implementation_open']}`",
        f"- DET/SPARSE reopen allowed: `{summary['det_sparse_reopen_allowed']}`",
        "",
        "Checks",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the bounded first-slice implementation-optimization completion contract."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    contract_path = assert_main_project_path(resolve_repo_path(args.contract), arg_name="--contract")
    out_dir = assert_main_project_path(resolve_repo_path(args.out_dir), arg_name="--out_dir")
    contract = _load_yaml(contract_path)
    checks, summary = evaluate_contract(contract)
    report_path = out_dir / f"fuller_first_implementation_slice_optimization_completion_{summary['tag']}.md"
    _write_report(report_path, checks, summary, contract_path)
    overall = "OK" if summary["overall_ok"] else "FAIL"
    print(f"[fuller-first-slice-optimization-completion] overall={overall} report={report_path}")
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
