#!/usr/bin/env python3
"""Validate the governed broader fuller-implementation extension-readiness contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

try:
    from .check_fuller_first_implementation_slice_optimization_completion import (
        _load_yaml as _load_completion_yaml,
        evaluate_contract as evaluate_completion_contract,
    )
    from .check_fuller_preimplementation_readiness import (
        _load_yaml as _load_preimplementation_yaml,
        evaluate_contract as evaluate_preimplementation_contract,
    )
    from .path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path
except ImportError:
    from check_fuller_first_implementation_slice_optimization_completion import (  # type: ignore
        _load_yaml as _load_completion_yaml,
        evaluate_contract as evaluate_completion_contract,
    )
    from check_fuller_preimplementation_readiness import (  # type: ignore
        _load_yaml as _load_preimplementation_yaml,
        evaluate_contract as evaluate_preimplementation_contract,
    )
    from path_policy import MAIN_PROJECT_REPORT_DATA_DIR, assert_main_project_path, resolve_repo_path  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs" / "fuller_broader_implementation_extension_readiness_contract_20260319.yaml"
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
    governed_route = _require_mapping(contract, "governed_route")
    execution_policy = _require_mapping(contract, "execution_policy")
    extension_scope = _require_mapping(contract, "extension_scope")
    state = _require_mapping(contract, "state")

    completion_contract_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("completion_contract_yaml") or "")),
        arg_name="upstream.completion_contract_yaml",
    )
    completion_report_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("completion_report_md") or "")),
        arg_name="upstream.completion_report_md",
    )
    meso_readiness_note_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("meso_readiness_note_md") or "")),
        arg_name="upstream.meso_readiness_note_md",
    )
    preimplementation_contract_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("preimplementation_contract_yaml") or "")),
        arg_name="upstream.preimplementation_contract_yaml",
    )
    preimplementation_report_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("preimplementation_report_md") or "")),
        arg_name="upstream.preimplementation_report_md",
    )
    project_final_closure_board_path = assert_main_project_path(
        resolve_repo_path(str(upstream.get("project_final_closure_board_md") or "")),
        arg_name="upstream.project_final_closure_board_md",
    )
    for label, path in (
        ("upstream.completion_contract_yaml", completion_contract_path),
        ("upstream.completion_report_md", completion_report_path),
        ("upstream.meso_readiness_note_md", meso_readiness_note_path),
        ("upstream.preimplementation_contract_yaml", preimplementation_contract_path),
        ("upstream.preimplementation_report_md", preimplementation_report_path),
        ("upstream.project_final_closure_board_md", project_final_closure_board_path),
    ):
        _record(checks, label, "pass" if path.exists() else "fail", str(path))

    completion_contract = _load_completion_yaml(completion_contract_path)
    _, completion_summary = evaluate_completion_contract(completion_contract)
    completion_ok = bool(completion_summary.get("overall_ok"))
    completion_boundary_ok = (
        bool(completion_summary.get("bounded_optimization_complete"))
        and not bool(completion_summary.get("broad_matrix_allowed"))
        and not bool(completion_summary.get("further_broad_implementation_open"))
        and not bool(completion_summary.get("det_sparse_reopen_allowed"))
    )
    _record(
        checks,
        "upstream.completion_contract_status",
        "pass" if completion_ok else "fail",
        f"overall_ok={completion_summary.get('overall_ok')!r}",
    )
    _record(
        checks,
        "upstream.completion_boundary_status",
        "pass" if completion_boundary_ok else "fail",
        (
            f"bounded_optimization_complete={completion_summary.get('bounded_optimization_complete')!r} "
            f"broad_matrix_allowed={completion_summary.get('broad_matrix_allowed')!r} "
            f"further_broad_implementation_open={completion_summary.get('further_broad_implementation_open')!r} "
            f"det_sparse_reopen_allowed={completion_summary.get('det_sparse_reopen_allowed')!r}"
        ),
    )

    preimplementation_contract = _load_preimplementation_yaml(preimplementation_contract_path)
    _, preimplementation_summary = evaluate_preimplementation_contract(preimplementation_contract)
    preimplementation_ok = bool(preimplementation_summary.get("overall_ok"))
    fuller_target = _require_mapping(preimplementation_contract, "fuller_target")
    architecture_traceability = _require_mapping(preimplementation_contract, "architecture_traceability")
    preimplementation_launch_state = _require_mapping(preimplementation_contract, "launch_state")
    architecture_candidate_started = (
        str(fuller_target.get("name")) == "HOPS-centered cross-layer accelerator"
        and bool(
            preimplementation_launch_state.get(
                "bounded_collection_ready",
                preimplementation_launch_state.get("large_scale_launch_ready"),
            )
        )
        and bool(architecture_traceability.get("claim_traceability_ready"))
    )
    _record(
        checks,
        "upstream.preimplementation_contract_status",
        "pass" if preimplementation_ok else "fail",
        f"overall_ok={preimplementation_summary.get('overall_ok')!r}",
    )
    _record(
        checks,
        "upstream.architecture_candidate_status",
        "pass" if architecture_candidate_started else "fail",
        (
            f"design_name={fuller_target.get('name')!r} "
            f"bounded_collection_ready={preimplementation_launch_state.get('bounded_collection_ready')!r} "
            f"large_scale_launch_ready={preimplementation_launch_state.get('large_scale_launch_ready')!r} "
            f"claim_traceability_ready={architecture_traceability.get('claim_traceability_ready')!r}"
        ),
    )

    meso_readiness_text = meso_readiness_note_path.read_text(encoding="utf-8") if meso_readiness_note_path.exists() else ""
    meso_ready = (
        "bounded formal redesign / implementation lane now" in meso_readiness_text
        and "broad cross-mechanism redesign wave" in meso_readiness_text
    )
    _record(
        checks,
        "upstream.meso_readiness_semantics",
        "pass" if meso_ready else "fail",
        "requires bounded MESO lane readiness and explicit non-broad-wave wording",
    )

    expected_route = "bounded local baseline + narrow HOPS paper"
    _record(
        checks,
        "governed_route.active_safe_route",
        "pass" if str(governed_route.get("active_safe_route")) == expected_route else "fail",
        f"active_safe_route={governed_route.get('active_safe_route')!r}",
    )
    _record(
        checks,
        "governed_route.retained_route_preserved",
        "pass"
        if governed_route.get("retained_route_preserved") is True and governed_route.get("claim_promotion_allowed") is False
        else "fail",
        repr(governed_route),
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
        "extension_scope.lane_roles",
        "pass"
        if (
            str(extension_scope.get("design_name")) == "HOPS-centered cross-layer accelerator"
            and str(extension_scope.get("astra_substrate_role")) == "baseline_execution_substrate"
            and str(extension_scope.get("primary_lane")) == "HOPS"
            and str(extension_scope.get("extension_lane")) == "MESO"
            and str(extension_scope.get("support_lane")) == "PHY"
        )
        else "fail",
        (
            f"design_name={extension_scope.get('design_name')!r} "
            f"astra_substrate_role={extension_scope.get('astra_substrate_role')!r} "
            f"primary_lane={extension_scope.get('primary_lane')!r} "
            f"extension_lane={extension_scope.get('extension_lane')!r} "
            f"support_lane={extension_scope.get('support_lane')!r}"
        ),
    )
    _record(
        checks,
        "extension_scope.deferred_hooks",
        "pass" if _require_list(extension_scope, "deferred_hooks") == ["DET", "SPARSE"] else "fail",
        f"deferred_hooks={extension_scope.get('deferred_hooks')!r}",
    )
    _record(
        checks,
        "extension_scope.retired_hooks",
        "pass" if _require_list(extension_scope, "retired_hooks") == ["DET", "SPARSE"] else "fail",
        f"retired_hooks={extension_scope.get('retired_hooks')!r}",
    )
    retired_hooks_note_path = assert_main_project_path(
        resolve_repo_path(str(extension_scope.get("retired_hooks_note_md") or "")),
        arg_name="extension_scope.retired_hooks_note_md",
    )
    _record(
        checks,
        "extension_scope.retired_hooks_note_md",
        "pass" if retired_hooks_note_path.exists() else "fail",
        str(retired_hooks_note_path),
    )
    allowed_next_scope = set(_require_list(extension_scope, "allowed_next_scope"))
    _record(
        checks,
        "extension_scope.allowed_next_scope",
        "pass"
        if allowed_next_scope
        >= {
            "flow_primary_integration_extension",
            "meso_explicit_cost_topology_redesign",
            "phy_support_only_cleanup",
        }
        else "fail",
        f"allowed_next_scope={sorted(allowed_next_scope)!r}",
    )
    disallowed_expansions = set(_require_list(extension_scope, "disallowed_expansions"))
    _record(
        checks,
        "extension_scope.disallowed_expansions",
        "pass"
        if disallowed_expansions
        >= {
            "broad_rerun_matrix",
            "cross_mechanism_wave",
            "det_reopen",
            "sparse_reopen",
            "freeze_replacement",
            "paper_claim_promotion",
        }
        else "fail",
        f"disallowed_expansions={sorted(disallowed_expansions)!r}",
    )
    _check_paths(checks, "extension_scope.required_docs", _require_list(extension_scope, "required_docs"))
    _check_paths(
        checks,
        "extension_scope.required_existing_artifacts",
        _require_list(extension_scope, "required_existing_artifacts"),
    )

    bounded_first_slice_optimization_complete_required = bool(state.get("bounded_first_slice_optimization_complete_required"))
    architecture_candidate_lane_required = bool(state.get("architecture_candidate_lane_required"))
    meso_redesign_ready_required = bool(state.get("meso_redesign_ready_required"))
    broader_extension_lane_open = bool(state.get("broader_extension_lane_open"))
    broad_matrix_allowed = bool(state.get("broad_matrix_allowed"))
    cross_mechanism_wave_allowed = bool(state.get("cross_mechanism_wave_allowed"))
    det_sparse_reopen_allowed = bool(state.get("det_sparse_reopen_allowed"))
    coherence = (
        bounded_first_slice_optimization_complete_required
        and completion_ok
        and completion_boundary_ok
        and architecture_candidate_lane_required
        and preimplementation_ok
        and architecture_candidate_started
        and meso_redesign_ready_required
        and meso_ready
        and broader_extension_lane_open
        and not broad_matrix_allowed
        and not cross_mechanism_wave_allowed
        and not det_sparse_reopen_allowed
    )
    _record(
        checks,
        "state.coherence",
        "pass" if coherence else "fail",
        (
            f"bounded_first_slice_optimization_complete_required={bounded_first_slice_optimization_complete_required} "
            f"completion_ok={completion_ok} "
            f"completion_boundary_ok={completion_boundary_ok} "
            f"architecture_candidate_lane_required={architecture_candidate_lane_required} "
            f"preimplementation_ok={preimplementation_ok} "
            f"architecture_candidate_started={architecture_candidate_started} "
            f"meso_redesign_ready_required={meso_redesign_ready_required} "
            f"meso_ready={meso_ready} "
            f"broader_extension_lane_open={broader_extension_lane_open} "
            f"broad_matrix_allowed={broad_matrix_allowed} "
            f"cross_mechanism_wave_allowed={cross_mechanism_wave_allowed} "
            f"det_sparse_reopen_allowed={det_sparse_reopen_allowed}"
        ),
    )

    summary = {
        "tag": str(meta.get("tag") or "unknown"),
        "governed_route": governed_route,
        "design_name": str(extension_scope.get("design_name") or ""),
        "completion_ok": completion_ok,
        "architecture_candidate_started": architecture_candidate_started,
        "meso_ready": meso_ready,
        "broader_extension_lane_open": broader_extension_lane_open,
        "broad_matrix_allowed": broad_matrix_allowed,
        "cross_mechanism_wave_allowed": cross_mechanism_wave_allowed,
        "det_sparse_reopen_allowed": det_sparse_reopen_allowed,
        "overall_ok": all(row["status"] == "pass" for row in checks),
    }
    return checks, summary


def _write_report(path: Path, checks: list[dict[str, str]], summary: dict[str, Any], contract_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fuller Broader-Implementation Extension Readiness Report",
        "",
        "Scope",
        f"- Contract: `{contract_path}`",
        f"- Tag: `{summary['tag']}`",
        f"- Active route: `{summary['governed_route']['active_safe_route']}`",
        f"- Design name: `{summary['design_name']}`",
        "",
        "State",
        f"- Completion checker status: `{summary['completion_ok']}`",
        f"- Architecture candidate started: `{summary['architecture_candidate_started']}`",
        f"- MESO redesign ready: `{summary['meso_ready']}`",
        f"- Broader extension lane open: `{summary['broader_extension_lane_open']}`",
        f"- Broad matrix allowed: `{summary['broad_matrix_allowed']}`",
        f"- Cross-mechanism wave allowed: `{summary['cross_mechanism_wave_allowed']}`",
        f"- DET/SPARSE reopen allowed: `{summary['det_sparse_reopen_allowed']}`",
        "",
        "Checks",
    ]
    for row in checks:
        lines.append(f"- `{row['status']}` `{row['check_id']}`: {row['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the governed broader fuller-implementation extension-readiness contract."
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    contract_path = assert_main_project_path(resolve_repo_path(args.contract), arg_name="--contract")
    out_dir = assert_main_project_path(resolve_repo_path(args.out_dir), arg_name="--out_dir")
    contract = _load_yaml(contract_path)
    checks, summary = evaluate_contract(contract)
    report_path = out_dir / f"fuller_broader_implementation_extension_readiness_{summary['tag']}.md"
    _write_report(report_path, checks, summary, contract_path)
    overall = "OK" if summary["overall_ok"] else "FAIL"
    print(f"[fuller-broader-extension-readiness] overall={overall} report={report_path}")
    if not summary["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
