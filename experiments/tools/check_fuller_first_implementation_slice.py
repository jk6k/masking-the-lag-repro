#!/usr/bin/env python3
"""Validate the governed first fuller-implementation slice kickoff contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

try:
    from .path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path
except ImportError:
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_first_implementation_slice_contract_20260319.yaml"
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
    first_slice = _require_mapping(contract, "first_slice")
    stop_rules = _require_mapping(contract, "stop_rules")
    state = _require_mapping(contract, "state")

    upstream_contract_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("preimplementation_contract_yaml") or "")),
        arg_name="upstream.preimplementation_contract_yaml",
    )
    upstream_report_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("preimplementation_readiness_report_md") or "")),
        arg_name="upstream.preimplementation_readiness_report_md",
    )
    preflight_board_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("launch_preflight_board_md") or "")),
        arg_name="upstream.launch_preflight_board_md",
    )
    for label, path in (
        ("upstream.preimplementation_contract_yaml", upstream_contract_path),
        ("upstream.preimplementation_readiness_report_md", upstream_report_path),
        ("upstream.launch_preflight_board_md", preflight_board_path),
    ):
        _record(checks, label, "pass" if path.exists() else "fail", str(path))

    upstream_contract = _load_yaml(upstream_contract_path)
    upstream_launch_state = _require_mapping(upstream_contract, "launch_state")
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
        "first_slice.current_state",
        "pass"
        if str(first_slice.get("current_state")) in {"kickoff_ready_not_started", "dedicated_entry_lane_smoke_started"}
        else "fail",
        f"current_state={first_slice.get('current_state')!r}",
    )
    _record(
        checks,
        "first_slice.lane_roles",
        "pass"
        if (
            str(first_slice.get("primary_lane")) == "HOPS"
            and str(first_slice.get("support_lane")) == "MESO"
            and str(first_slice.get("bounded_support")) == "PHY"
            and str(first_slice.get("astra_substrate_role")) == "baseline_execution_substrate"
        )
        else "fail",
        (
            f"primary_lane={first_slice.get('primary_lane')!r} "
            f"support_lane={first_slice.get('support_lane')!r} "
            f"bounded_support={first_slice.get('bounded_support')!r} "
            f"astra_substrate_role={first_slice.get('astra_substrate_role')!r}"
        ),
    )
    _record(
        checks,
        "first_slice.forbidden_reopens",
        "pass" if _require_list(first_slice, "forbidden_reopens") == ["DET", "SPARSE"] else "fail",
        f"forbidden_reopens={first_slice.get('forbidden_reopens')!r}",
    )

    _check_paths(checks, "first_slice.owned_runner_files", _require_list(first_slice, "owned_runner_files"))
    _check_paths(checks, "first_slice.owned_support_files", _require_list(first_slice, "owned_support_files"))
    _check_paths(checks, "first_slice.transitional_config_anchors", _require_list(first_slice, "transitional_config_anchors"))
    _check_paths(checks, "first_slice.dedicated_entry_template_yaml", [str(first_slice.get("dedicated_entry_template_yaml") or "")])
    _check_paths(
        checks,
        "first_slice.archived_mechanism_audit_note_md",
        [str(first_slice.get("archived_mechanism_audit_note_md") or "")],
    )
    _check_paths(checks, "first_slice.smoke_checker_py", [str(first_slice.get("smoke_checker_py") or "")])
    _check_paths(checks, "first_slice.smoke_report_md", [str(first_slice.get("smoke_report_md") or "")])
    _check_paths(checks, "first_slice.smoke_test_py", [str(first_slice.get("smoke_test_py") or "")])
    _check_paths(checks, "first_slice.required_docs", _require_list(first_slice, "required_docs"))
    _check_paths(checks, "first_slice.required_tests", _require_list(first_slice, "required_tests"))

    required_deliverables = set(_require_list(first_slice, "next_thread_required_deliverables"))
    _record(
        checks,
        "first_slice.next_thread_required_deliverables",
        "pass"
        if required_deliverables
        >= {
            "dedicated_implementation_facing_config_template",
            "one_bounded_integrated_smoke_test",
            "no_broad_rerun_matrix_before_smoke",
        }
        else "fail",
        f"next_thread_required_deliverables={sorted(required_deliverables)!r}",
    )
    start_criteria = set(_require_list(first_slice, "start_criteria"))
    _record(
        checks,
        "first_slice.start_criteria",
        "pass"
        if start_criteria >= {"dedicated_entry_lane_landed", "integrated_smoke_test_landed", "stop_rules_preserved"}
        else "fail",
        f"start_criteria={sorted(start_criteria)!r}",
    )

    forbidden_actions = set(_require_list(stop_rules, "forbidden_actions"))
    _record(
        checks,
        "stop_rules.forbidden_actions",
        "pass"
        if forbidden_actions
        >= {
            "broad_rerun_matrix_before_integrated_smoke",
            "det_or_sparse_reopen",
            "paper_claim_broadening_during_kickoff",
            "meso_as_independent_retained_lane",
            "fuller_evidence_closed_wording",
        }
        else "fail",
        f"forbidden_actions={sorted(forbidden_actions)!r}",
    )

    kickoff_pack_complete = bool(state.get("kickoff_pack_complete"))
    kickoff_ready = bool(state.get("kickoff_ready"))
    dedicated_entry_lane_landed = bool(state.get("dedicated_entry_lane_landed"))
    integrated_smoke_test_landed = bool(state.get("integrated_smoke_test_landed"))
    broad_matrix_allowed = bool(state.get("broad_matrix_allowed"))
    implementation_started = bool(state.get("implementation_started"))
    upstream_bounded_collection_ready_required = bool(state.get("upstream_bounded_collection_ready_required"))
    current_state = str(first_slice.get("current_state") or "")
    kickoff_state_ok = (
        current_state == "kickoff_ready_not_started"
        and not dedicated_entry_lane_landed
        and not integrated_smoke_test_landed
        and not implementation_started
        and not broad_matrix_allowed
    )
    started_state_ok = (
        current_state == "dedicated_entry_lane_smoke_started"
        and dedicated_entry_lane_landed
        and integrated_smoke_test_landed
        and implementation_started
        and not broad_matrix_allowed
    )
    coherence = (
        upstream_bounded_collection_ready_required
        and upstream_bounded_collection_ready
        and kickoff_pack_complete
        and kickoff_ready
        and (kickoff_state_ok or started_state_ok)
    )
    _record(
        checks,
        "state.coherence",
        "pass" if coherence else "fail",
        (
            f"upstream_bounded_collection_ready_required={upstream_bounded_collection_ready_required} "
            f"upstream_bounded_collection_ready={upstream_bounded_collection_ready} "
            f"kickoff_pack_complete={kickoff_pack_complete} "
            f"kickoff_ready={kickoff_ready} "
            f"current_state={current_state!r} "
            f"dedicated_entry_lane_landed={dedicated_entry_lane_landed} "
            f"integrated_smoke_test_landed={integrated_smoke_test_landed} "
            f"broad_matrix_allowed={broad_matrix_allowed} "
            f"implementation_started={implementation_started}"
        ),
    )

    summary = {
        "tag": str(meta.get("tag") or "unknown"),
        "paper_route": paper_route,
        "first_slice_name": str(first_slice.get("name") or ""),
        "first_slice_state": str(first_slice.get("current_state") or ""),
        "upstream_bounded_collection_ready": upstream_bounded_collection_ready,
        "kickoff_pack_complete": kickoff_pack_complete,
        "kickoff_ready": kickoff_ready,
        "dedicated_entry_lane_landed": dedicated_entry_lane_landed,
        "integrated_smoke_test_landed": integrated_smoke_test_landed,
        "broad_matrix_allowed": broad_matrix_allowed,
        "implementation_started": implementation_started,
        "overall_ok": all(row["status"] == "pass" for row in checks),
    }
    return checks, summary


def _write_report(path: Path, checks: list[dict[str, str]], summary: dict[str, Any], contract_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fuller First-Implementation Slice Readiness Report",
        "",
        "Scope",
        f"- Contract: `{contract_path}`",
        f"- Tag: `{summary['tag']}`",
        f"- Active route: `{summary['paper_route']['active_safe_route']}`",
        f"- First slice: `{summary['first_slice_name']}`",
        "",
        "State",
        f"- Upstream bounded collection ready: `{summary['upstream_bounded_collection_ready']}`",
        f"- Kickoff pack complete: `{summary['kickoff_pack_complete']}`",
        f"- Kickoff ready: `{summary['kickoff_ready']}`",
        f"- First slice state: `{summary['first_slice_state']}`",
        f"- Dedicated entry lane landed: `{summary['dedicated_entry_lane_landed']}`",
        f"- Integrated smoke test landed: `{summary['integrated_smoke_test_landed']}`",
        f"- Broad matrix allowed: `{summary['broad_matrix_allowed']}`",
        f"- Implementation started: `{summary['implementation_started']}`",
        "",
        "Checks",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the first fuller-implementation slice kickoff contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    contract_path = assert_main_project_path(resolve_repo_path(args.contract), arg_name="--contract")
    out_dir = assert_main_project_path(resolve_repo_path(args.out_dir), arg_name="--out_dir")
    contract = _load_yaml(contract_path)
    checks, summary = evaluate_contract(contract)
    report_path = out_dir / f"fuller_first_implementation_slice_readiness_{summary['tag']}.md"
    _write_report(report_path, checks, summary, contract_path)
    overall = "OK" if summary["overall_ok"] else "FAIL"
    print(f"[fuller-first-slice] overall={overall} report={report_path}")
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
