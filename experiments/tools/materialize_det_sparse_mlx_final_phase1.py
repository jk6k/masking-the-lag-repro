#!/usr/bin/env python3
"""Materialize and merge the paper-facing MLX final-freeze phase-1 config set."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp_common.realism_proxy_calibration import (  # noqa: E402
    build_realistic_proxy_profile,
    write_realistic_proxy_profile,
)


RUNS_ROOT = ROOT / "experiments" / "results" / "runs"
GENERATED_ROOT = ROOT / "experiments" / "results" / "generated_configs"
OVERLAY_ROOT = ROOT / "experiments" / "results" / "dev"
SPLIT_MANIFEST_ROOT = ROOT / "experiments" / "results" / "accuracy" / "splits_quickpack_20260212"
SOURCE_PREFIX = "20260228_opt_sync_"

GROUP_SOURCE_RUNS: dict[str, list[str]] = {
    "core": [
        "20260228_opt_sync_core_e0",
        "20260228_opt_sync_core_e1",
        "20260228_opt_sync_core_e2",
        "20260228_opt_sync_core_e3",
        "20260228_opt_sync_core_e4",
        "20260228_opt_sync_core_e5",
        "20260228_opt_sync_core_e6",
    ],
    "extras": [
        "20260228_opt_sync_core_e4_s1",
        "20260228_opt_sync_core_e4_s2",
        "20260228_opt_sync_core_e6_s1",
        "20260228_opt_sync_core_e6_s2",
    ],
    "scaling": [
        "20260228_opt_sync_scan_e0_batch2",
        "20260228_opt_sync_scan_e0_batch4",
        "20260228_opt_sync_scan_e0_seq128",
        "20260228_opt_sync_scan_e0_seq256",
    ],
    "det_sweep": [
        "20260228_opt_sync_scan_e3_k4",
        "20260228_opt_sync_scan_e3_k8",
        "20260228_opt_sync_scan_e3_k16",
        "20260228_opt_sync_scan_e3_k24",
        "20260228_opt_sync_scan_e3_k32",
        "20260228_opt_sync_scan_e3_k48",
        "20260228_opt_sync_scan_e3_k64",
        "20260228_opt_sync_scan_e3_k80",
        "20260228_opt_sync_scan_e3_k96",
        "20260228_opt_sync_scan_e3_k112",
        "20260228_opt_sync_scan_e3_k129",
    ],
    "sparse_sweep": [
        "20260228_opt_sync_scan_e4_t00",
        "20260228_opt_sync_scan_e4_t05",
        "20260228_opt_sync_scan_e4_t10",
        "20260228_opt_sync_scan_e4_t15",
        "20260228_opt_sync_scan_e4_t20",
        "20260228_opt_sync_scan_e4_t25",
        "20260228_opt_sync_scan_e4_t30",
        "20260228_opt_sync_scan_e4_t40",
        "20260228_opt_sync_scan_e4_t50",
    ],
    "fanout_sweep": [
        "20260228_opt_sync_scan_e1_f2",
        "20260228_opt_sync_scan_e1_f4",
        "20260228_opt_sync_scan_e1_f6",
        "20260228_opt_sync_scan_e1_f8",
        "20260228_opt_sync_scan_e1_f12",
        "20260228_opt_sync_scan_e1_f16",
        "20260228_opt_sync_scan_e1_f24",
        "20260228_opt_sync_scan_e1_f32",
    ],
    "phy_sweep": [
        "20260228_opt_sync_scan_e5_n4",
        "20260228_opt_sync_scan_e5_n8",
        "20260228_opt_sync_scan_e5_n12",
        "20260228_opt_sync_scan_e5_n16",
        "20260228_opt_sync_scan_e5_n20",
        "20260228_opt_sync_scan_e5_n24",
        "20260228_opt_sync_scan_e5_n32",
        "20260228_opt_sync_scan_e5_n48",
        "20260228_opt_sync_scan_e5_n64",
    ],
}

OVERLAY_EXPERIMENTS = {
    "E0": ["core_e0"],
    "E2": ["core_e2"],
    "E3": ["core_e3"],
    "E4": ["core_e4", "core_e4_s1", "core_e4_s2"],
    "E6": ["core_e6", "core_e6_s1", "core_e6_s2"],
}

SUPPORT_FANOUT_FIELDS = [
    "fanout_cfg",
    "run_id",
    "n_models",
    "latency_ms",
    "energy_j",
    "tops_w",
    "speedup_vs_E0",
    "fanout",
    "serializers_saved",
    "broadcast_driver_energy_j",
    "net_energy_gain_j",
    "acc_drop_pp_nonempty",
]

SUPPORT_PHY_FIELDS = [
    "N_wdm",
    "run_id",
    "n_models",
    "P_laser_dbm",
    "P_laser_mw",
    "PP_crosstalk_db",
    "Loss_path_db",
    "energy_j",
    "tops_w",
    "latency_ms",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _target_run_id(program_tag: str, source_run_id: str) -> str:
    if not source_run_id.startswith(SOURCE_PREFIX):
        raise SystemExit(f"Unsupported source run id prefix: {source_run_id}")
    return f"{program_tag}_{source_run_id[len(SOURCE_PREFIX):]}"


def _support_root(out_root: Path, program_tag: str) -> Path:
    return out_root / program_tag / "support"


def _support_relpath(program_tag: str, filename: str) -> str:
    return str(Path("experiments") / "results" / "generated_configs" / program_tag / "support" / filename)


def _freeze_local_support_paths(program_tag: str) -> dict[str, str]:
    return {
        "realism_profile_yaml": _support_relpath(
            program_tag,
            f"fuller_realistic_proxy_calibration_profile_{program_tag}.yaml",
        ),
        "realism_profile_report_md": _support_relpath(
            program_tag,
            f"fuller_realistic_proxy_calibration_report_{program_tag}.md",
        ),
        "support_manifest_json": _support_relpath(
            program_tag,
            f"fuller_realistic_proxy_support_manifest_{program_tag}.json",
        ),
        "fanout_support_csv": _support_relpath(
            program_tag,
            f"quickscan_e1_fanout_sweep_{program_tag}.csv",
        ),
        "phy_support_csv": _support_relpath(
            program_tag,
            f"quickscan_e5_phy_n_sweep_{program_tag}.csv",
        ),
        "scope_report_md": _support_relpath(
            program_tag,
            f"phase6_scope_report_{program_tag}.md",
        ),
        "flow_calibration_source": (
            f"experiments/results/runs/{program_tag}_core_e0/master_metrics.csv;"
            f"experiments/results/runs/{program_tag}_core_e2/master_metrics.csv"
        ),
    }


def _freeze_local_split_manifest(split: str) -> str:
    if split not in {"eval", "calib"}:
        raise SystemExit(f"Unsupported split manifest kind: {split}")
    return str(Path("experiments") / "results" / "accuracy" / "splits_quickpack_20260212" / f"imagenet_val_{split}.csv")


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _read_master_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise SystemExit(f"Missing header in {path}")
        return list(reader.fieldnames), rows


def _write_master_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float | None, *, integer: bool = False) -> str:
    if value is None:
        return ""
    if integer:
        return str(int(round(value)))
    return f"{value:.12g}"


def _mean_field(rows: list[dict[str, str]], field: str) -> float | None:
    values = [
        value
        for row in rows
        for value in [_to_float(row.get(field))]
        if value is not None
    ]
    if not values:
        return None
    return sum(values) / float(len(values))


def _run_rows(runs_root: Path, run_id: str) -> list[dict[str, str]]:
    _, rows = _read_master_rows(runs_root / run_id / "master_metrics.csv")
    if not rows:
        raise SystemExit(f"Missing master_metrics rows for {run_id}")
    return rows


def _aggregate_fanout_row(run_id: str, rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        "fanout_cfg": _format_number(_mean_field(rows, "fanout"), integer=True),
        "run_id": run_id,
        "n_models": str(len(rows)),
        "latency_ms": _format_number(_mean_field(rows, "latency_ms")),
        "energy_j": _format_number(_mean_field(rows, "energy_j")),
        "tops_w": _format_number(_mean_field(rows, "tops_w")),
        "speedup_vs_E0": _format_number(_mean_field(rows, "speedup_vs_E0")),
        "fanout": _format_number(_mean_field(rows, "fanout")),
        "serializers_saved": _format_number(_mean_field(rows, "serializers_saved")),
        "broadcast_driver_energy_j": _format_number(_mean_field(rows, "broadcast_driver_energy_j")),
        "net_energy_gain_j": _format_number(_mean_field(rows, "net_energy_gain_j")),
        "acc_drop_pp_nonempty": str(sum(1 for row in rows if str(row.get("acc_drop_pp") or "").strip())),
    }


def _aggregate_phy_row(run_id: str, rows: list[dict[str, str]]) -> dict[str, str]:
    return {
        "N_wdm": _format_number(_mean_field(rows, "N_wdm"), integer=True),
        "run_id": run_id,
        "n_models": str(len(rows)),
        "P_laser_dbm": _format_number(_mean_field(rows, "P_laser_dbm")),
        "P_laser_mw": _format_number(_mean_field(rows, "P_laser_mw")),
        "PP_crosstalk_db": _format_number(_mean_field(rows, "PP_crosstalk_db")),
        "Loss_path_db": _format_number(_mean_field(rows, "Loss_path_db")),
        "energy_j": _format_number(_mean_field(rows, "energy_j")),
        "tops_w": _format_number(_mean_field(rows, "tops_w")),
        "latency_ms": _format_number(_mean_field(rows, "latency_ms")),
    }


def _write_support_scope_report(path: Path, *, program_tag: str) -> None:
    lines = [
        "# MLX Final Freeze PHY Scope Support",
        "",
        f"Program tag: `{program_tag}`",
        "",
        "Purpose",
        "- Freeze-local support note for the active DET/SPARSE MLX final freeze.",
        "- Records that PHY support remains a bounded modeling envelope for the active local MLX chain.",
        "- Avoids inherited historical quick-report or calibration-audit path references in active provenance fields.",
        "",
        "Boundary",
        "- Support-only scope for active PHY sweep interpretation.",
        "- Not benchmark-equivalent hardware evidence.",
        "- Active provenance is restricted to current-program support CSVs and run outputs.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_support_artifacts(*, program_tag: str, runs_root: Path, out_root: Path) -> dict[str, str]:
    support_root = _support_root(out_root, program_tag)
    support_paths = _freeze_local_support_paths(program_tag)

    fanout_run_ids = [
        _target_run_id(program_tag, source_run_id)
        for source_run_id in GROUP_SOURCE_RUNS["fanout_sweep"]
    ]
    phy_run_ids = [
        _target_run_id(program_tag, source_run_id)
        for source_run_id in GROUP_SOURCE_RUNS["phy_sweep"]
    ]
    fanout_rows = [_aggregate_fanout_row(run_id, _run_rows(runs_root, run_id)) for run_id in fanout_run_ids]
    fanout_rows.sort(key=lambda row: int(row["fanout_cfg"]))
    phy_rows = [_aggregate_phy_row(run_id, _run_rows(runs_root, run_id)) for run_id in phy_run_ids]
    phy_rows.sort(key=lambda row: int(row["N_wdm"]))

    fanout_csv = support_root / Path(support_paths["fanout_support_csv"]).name
    phy_csv = support_root / Path(support_paths["phy_support_csv"]).name
    scope_md = support_root / Path(support_paths["scope_report_md"]).name
    profile_yaml = support_root / Path(support_paths["realism_profile_yaml"]).name
    profile_report_md = support_root / Path(support_paths["realism_profile_report_md"]).name
    support_manifest_json = support_root / Path(support_paths["support_manifest_json"]).name

    _write_master_rows(fanout_csv, SUPPORT_FANOUT_FIELDS, fanout_rows)
    _write_master_rows(phy_csv, SUPPORT_PHY_FIELDS, phy_rows)
    _write_support_scope_report(scope_md, program_tag=program_tag)

    profile = build_realistic_proxy_profile(
        model="mobilevit_s",
        buffer_depth=2,
        baseline_flow_run_csv=runs_root / f"{program_tag}_core_e0" / "master_metrics.csv",
        flow_run_csv=runs_root / f"{program_tag}_core_e2" / "master_metrics.csv",
        meso_fanout_sweep_csv=fanout_csv,
        phy_n_sweep_csv=phy_csv,
        astra_summary_csv=runs_root / f"{program_tag}_astra" / "master_metrics.csv",
        fuller_summary_csv=runs_root / f"{program_tag}_fuller" / "master_metrics.csv",
        phase6_scope_report_md=scope_md,
    )
    profile_meta = dict(profile.get("meta") or {})
    profile_meta["profile_id"] = f"{program_tag}_realistic_proxy_v1"
    profile_meta["goal"] = "freeze-local realistic proxy calibration profile for the DET/SPARSE MLX final freeze"
    profile["meta"] = profile_meta
    write_realistic_proxy_profile(
        profile=profile,
        out_yaml=profile_yaml,
        out_report_md=profile_report_md,
    )

    support_manifest = {
        "program_tag": program_tag,
        "fanout_support_csv": str(fanout_csv),
        "phy_support_csv": str(phy_csv),
        "scope_report_md": str(scope_md),
        "realism_profile_yaml": str(profile_yaml),
        "realism_profile_report_md": str(profile_report_md),
        "flow_calibration_source": support_paths["flow_calibration_source"],
        "phase1_anchors": {
            "core_e0_master_csv": str(runs_root / f"{program_tag}_core_e0" / "master_metrics.csv"),
            "core_e2_master_csv": str(runs_root / f"{program_tag}_core_e2" / "master_metrics.csv"),
            "astra_master_csv": str(runs_root / f"{program_tag}_astra" / "master_metrics.csv"),
            "fuller_master_csv": str(runs_root / f"{program_tag}_fuller" / "master_metrics.csv"),
        },
    }
    _write_json(support_manifest_json, support_manifest)
    return support_paths


def _materialize_config(
    *,
    source_run_id: str,
    target_run_id: str,
    out_path: Path,
    device: str,
    accuracy_source_csv: str,
    e0_latency_csv: str,
    realism_profile_yaml: str,
    support_paths: dict[str, str],
    calib_manifest_csv: str,
    eval_manifest_csv: str,
) -> None:
    cfg = copy.deepcopy(_load_yaml(RUNS_ROOT / source_run_id / "config_snapshot.yaml"))
    run_cfg = dict(cfg.get("run") or {})
    run_cfg["run_id"] = target_run_id
    run_cfg["device"] = device
    run_cfg["execution_surface"] = "host_unsandboxed_caffeinate_required"
    run_cfg["long_run_launch_prefix"] = ["caffeinate", "-dimsu"]
    notes = str(run_cfg.get("notes") or "").strip()
    run_cfg["notes"] = " ".join(item for item in [notes, f"mlx_final_freeze_rebuild:{target_run_id}"] if item)
    run_cfg.pop("timestamp_utc", None)
    run_cfg.pop("date", None)
    cfg["run"] = run_cfg

    realism_cfg = dict(cfg.get("realism") or {})
    realism_cfg["calibration_profile_yaml"] = realism_profile_yaml
    cfg["realism"] = realism_cfg

    data_cfg = dict(cfg.get("data") or {})
    data_cfg["calib_manifest_csv"] = calib_manifest_csv
    data_cfg["eval_manifest_csv"] = eval_manifest_csv
    cfg["data"] = data_cfg

    accuracy_cfg = dict(cfg.get("accuracy") or {})
    accuracy_cfg["source_csv"] = accuracy_source_csv
    accuracy_cfg["context_run_id"] = ""
    accuracy_cfg["require_context_match"] = False
    cfg["accuracy"] = accuracy_cfg

    flow_cfg = dict(cfg.get("flow") or {})
    flow_cfg["calibration_source"] = support_paths["flow_calibration_source"]
    cfg["flow"] = flow_cfg

    meso_cfg = dict(cfg.get("meso") or {})
    meso_cfg["calibration_source"] = support_paths["fanout_support_csv"]
    cfg["meso"] = meso_cfg

    phy_cfg = dict(cfg.get("phy") or {})
    phy_cfg["calibration_source"] = f"{support_paths['phy_support_csv']};{support_paths['scope_report_md']}"
    cfg["phy"] = phy_cfg

    integrated_cfg = dict(cfg.get("integrated_system_costs") or {})
    integrated_cfg["calibration_source"] = support_paths["support_manifest_json"]
    cfg["integrated_system_costs"] = integrated_cfg

    baseline_ref = dict(cfg.get("baseline_ref") or {})
    baseline_ref["e0_latency_csv"] = e0_latency_csv
    cfg["baseline_ref"] = baseline_ref

    outputs = dict(cfg.get("outputs") or {})
    outputs["out_dir"] = "results/runs"
    outputs["append_master"] = False
    outputs["save_config_snapshot"] = True
    cfg["outputs"] = outputs

    _write_yaml(out_path, cfg)


def _selected_source_runs(groups: list[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for run_id in GROUP_SOURCE_RUNS[group]:
            if run_id in seen:
                continue
            seen.add(run_id)
            selected.append(run_id)
    return selected


def _write_materialize_outputs(
    *,
    program_tag: str,
    groups: list[str],
    out_root: Path,
    device: str,
    accuracy_source_csv: str,
    e0_latency_csv: str,
) -> Path:
    phase1_root = out_root / program_tag / "phase1_configs"
    support_paths = _freeze_local_support_paths(program_tag)
    calib_manifest_csv = _freeze_local_split_manifest("calib")
    eval_manifest_csv = _freeze_local_split_manifest("eval")
    target_run_ids_by_group = {
        group: [_target_run_id(program_tag, source_run_id) for source_run_id in GROUP_SOURCE_RUNS[group]]
        for group in groups
    }

    seen_target_run_ids: set[str] = set()
    for source_run_id in _selected_source_runs(groups):
        target_run_id = _target_run_id(program_tag, source_run_id)
        if target_run_id in seen_target_run_ids:
            continue
        seen_target_run_ids.add(target_run_id)
        _materialize_config(
            source_run_id=source_run_id,
            target_run_id=target_run_id,
            out_path=phase1_root / f"{target_run_id}.yaml",
            device=device,
            accuracy_source_csv=accuracy_source_csv,
            e0_latency_csv=("" if target_run_id.endswith("_core_e0") else e0_latency_csv),
            realism_profile_yaml=support_paths["realism_profile_yaml"],
            support_paths=support_paths,
            calib_manifest_csv=calib_manifest_csv,
            eval_manifest_csv=eval_manifest_csv,
        )

    manifest = {
        "program_tag": program_tag,
        "device": device,
        "accuracy_source_csv": accuracy_source_csv,
        "e0_latency_csv": e0_latency_csv,
        "groups": groups,
        "materialization_basis": "config_snapshot_clone_with_freeze_local_rewrite",
        "realism_calibration_profile_yaml": support_paths["realism_profile_yaml"],
        "support_manifest_json": support_paths["support_manifest_json"],
        "split_manifests": {
            "calib": calib_manifest_csv,
            "eval": eval_manifest_csv,
        },
        "target_run_ids_by_group": target_run_ids_by_group,
        "generated_config_root": str(phase1_root),
    }
    manifest_path = out_root / program_tag / "phase1_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _build_overlay(program_tag: str, runs_root: Path, overlay_root: Path) -> Path:
    target_root = overlay_root / f"headline_runs_overlay_{program_tag}"
    for experiment_id, suffixes in OVERLAY_EXPERIMENTS.items():
        source_rows: list[dict[str, str]] = []
        fieldnames: list[str] | None = None
        target_run_id = f"{program_tag}_{suffixes[0]}"
        for suffix in suffixes:
            source_run_id = f"{program_tag}_{suffix}"
            fieldnames_i, rows = _read_master_rows(runs_root / source_run_id / "master_metrics.csv")
            if fieldnames is None:
                fieldnames = fieldnames_i
            elif fieldnames != fieldnames_i:
                raise SystemExit(f"Inconsistent master_metrics headers for overlay {experiment_id}")
            source_rows.extend(rows)
        if fieldnames is None:
            raise SystemExit(f"No source rows collected for overlay {experiment_id}")
        source_rows.sort(key=lambda row: (row.get("model", ""), int(float(row.get("seed") or 0.0))))
        _write_master_rows(target_root / target_run_id / "master_metrics.csv", fieldnames, source_rows)
    return target_root


def _parse_groups(raw: str) -> list[str]:
    groups = [item.strip() for item in raw.split(",") if item.strip()]
    if not groups:
        raise SystemExit("--groups must select at least one group")
    unknown = [group for group in groups if group not in GROUP_SOURCE_RUNS]
    if unknown:
        raise SystemExit(f"Unknown groups: {unknown}")
    return groups


def _cmd_materialize(args: argparse.Namespace) -> None:
    manifest_path = _write_materialize_outputs(
        program_tag=args.program_tag,
        groups=_parse_groups(args.groups),
        out_root=args.generated_root,
        device=args.device,
        accuracy_source_csv=args.accuracy_source_csv,
        e0_latency_csv=args.e0_latency_csv,
    )
    print(f"[mlx-final-phase1] wrote {manifest_path}")


def _cmd_build_overlay(args: argparse.Namespace) -> None:
    overlay_root = _build_overlay(args.program_tag, args.runs_root, args.overlay_root)
    print(f"[mlx-final-phase1] wrote {overlay_root}")


def _cmd_build_support(args: argparse.Namespace) -> None:
    support_paths = _write_support_artifacts(
        program_tag=args.program_tag,
        runs_root=args.runs_root,
        out_root=args.generated_root,
    )
    for key, value in support_paths.items():
        if value.startswith("experiments/"):
            print(f"[mlx-final-phase1] {key}={_resolve_repo_path(value)}")
        else:
            print(f"[mlx-final-phase1] {key}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize MLX final-freeze phase-1 configs and headline overlays.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    materialize = sub.add_parser("materialize", help="Create phase-1 config files for the selected run groups.")
    materialize.add_argument("--program_tag", required=True)
    materialize.add_argument("--groups", default="core")
    materialize.add_argument("--generated_root", type=Path, default=GENERATED_ROOT)
    materialize.add_argument("--device", default="mps")
    materialize.add_argument("--accuracy_source_csv", default="")
    materialize.add_argument("--e0_latency_csv", default="")
    materialize.set_defaults(func=_cmd_materialize)

    overlay = sub.add_parser("build-overlay", help="Merge master_metrics into a headline overlay root.")
    overlay.add_argument("--program_tag", required=True)
    overlay.add_argument("--runs_root", type=Path, default=RUNS_ROOT)
    overlay.add_argument("--overlay_root", type=Path, default=OVERLAY_ROOT)
    overlay.set_defaults(func=_cmd_build_overlay)

    support = sub.add_parser("build-support", help="Build freeze-local support CSVs and realism profile from new-tag run outputs.")
    support.add_argument("--program_tag", required=True)
    support.add_argument("--runs_root", type=Path, default=RUNS_ROOT)
    support.add_argument("--generated_root", type=Path, default=GENERATED_ROOT)
    support.set_defaults(func=_cmd_build_support)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
