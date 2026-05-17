#!/usr/bin/env python3
"""Materialize and optionally run the optimized fuller noise sweeps."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from accuracy.eval_cvnets_imagenet_noise import RESULT_FIELDNAMES
try:
    from .build_fuller_experiment_program import build_fuller_experiment_program
    from .build_fuller_phase4_intake_contract import build_fuller_phase4_intake_contract
    from .build_fuller_report_pack_contract import build_fuller_report_pack_contract
    from .materialize_fuller_experiment_execution_plan import materialize_fuller_experiment_execution_plan
except ImportError:
    from build_fuller_experiment_program import build_fuller_experiment_program  # type: ignore
    from build_fuller_phase4_intake_contract import build_fuller_phase4_intake_contract  # type: ignore
    from build_fuller_report_pack_contract import build_fuller_report_pack_contract  # type: ignore
    from materialize_fuller_experiment_execution_plan import materialize_fuller_experiment_execution_plan  # type: ignore

DEFAULT_BUNDLE = ROOT / "configs" / "fuller_implementation_execution_bundle_20260319.yaml"
DEFAULT_EVAL_SCRIPT = ROOT / "experiments" / "accuracy" / "eval_cvnets_imagenet_noise.py"
DEFAULT_MLX_EVAL_SCRIPT = ROOT / "experiments" / "accuracy" / "eval_mlx_imagenet_noise.py"
DEFAULT_PHASE1_RUNNER = ROOT / "experiments" / "tools" / "phase1_runner.py"
DEFAULT_PROGRAM_CONTRACT = ROOT / "configs" / "fuller_experiment_program_contract_20260422.yaml"
MANIFEST_FIELDS = [
    "family_id",
    "accuracy_backend",
    "engine",
    "parity_status",
    "parity_report_ref",
    "model",
    "profile",
    "sweep_resolution",
    "crosstalk_alpha",
    "gaussian_noise_std",
    "accuracy_run_id",
    "accuracy_source_run_id",
    "accuracy_seed_list",
    "representative_profile",
    "source_run_tag",
    "source_status",
    "evidence_basis",
    "robustness_claim_status",
    "weights_npz",
    "phase1_run_id",
    "accuracy_results_csv",
    "phase1_config_yaml",
    "phase1_run_dir",
    "accuracy_launch_policy",
    "phase1_launch_policy",
    "accuracy_command",
    "phase1_command",
]
SUMMARY_FIELDS = [
    "model",
    "profile",
    "sweep_resolution",
    "crosstalk_alpha",
    "gaussian_noise_std",
    "accuracy_backend",
    "engine",
    "parity_status",
    "parity_report_ref",
    "acc_top1",
    "acc_drop_pp",
    "latency_ms",
    "energy_j",
    "accuracy_results_csv",
    "accuracy_source_run_ids",
    "accuracy_seeds",
    "seed_count",
    "complete",
    "representative_profile",
    "source_run_tag",
    "source_status",
    "evidence_basis",
    "robustness_claim_status",
    "phase1_run_id",
    "phase1_run_dir",
    "accuracy_launch_policy",
    "phase1_launch_policy",
]

NOISE_MODEL_SUFFIXES = {
    "mobilevit_s": "s",
    "mobilevit_xs": "xs",
    "mobilevit_xxs": "xxs",
}


def _noise_model_suffix(model: str) -> str:
    suffix = NOISE_MODEL_SUFFIXES.get(str(model).strip())
    if suffix is None:
        raise SystemExit(f"Unsupported governed noise model: {model!r}")
    return suffix


def _noise_family_id(*, model: str, sweep_resolution: str) -> str:
    return f"NOISE_IMAGENET_MOBILEVIT_{_noise_model_suffix(model).upper()}_{str(sweep_resolution).upper()}"


def _noise_artifact_id(*, model: str, sweep_resolution: str) -> str:
    return f"noise_accuracy_summary_{_noise_model_suffix(model)}_{str(sweep_resolution).lower()}"


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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def _int_list(value: Any, *, default: list[int]) -> list[int]:
    if value in ("", None):
        return list(default)
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        raise SystemExit(f"Expected integer list, got {value!r}")
    resolved: list[int] = []
    seen: set[int] = set()
    for item in value:
        integer = int(item)
        if integer in seen:
            continue
        seen.add(integer)
        resolved.append(integer)
    return resolved or list(default)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _load_weights_npz_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    exports = payload.get("exports")
    if isinstance(exports, list):
        mapping: dict[str, str] = {}
        for item in exports:
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or "").strip()
            weights_npz = str(item.get("weights_npz") or item.get("output_path") or "").strip()
            if model and weights_npz:
                mapping[model] = weights_npz
        return mapping
    model = str(payload.get("model") or "").strip()
    weights_npz = str(payload.get("weights_npz") or payload.get("output_path") or "").strip()
    if model and weights_npz:
        return {model: weights_npz}
    raise SystemExit(f"Unable to parse MLX weights manifest: {path}")


def _artifact_paths(bundle: dict[str, Any]) -> dict[str, Path]:
    design_contract = _load_yaml(ROOT / str(bundle["inputs"]["design_contract_yaml"]))
    data_contract_csv = ROOT / str((design_contract.get("artifacts") or {})["data_contract_csv"])
    rows = _read_csv(data_contract_csv)
    summary_out_root_value = (bundle.get("paths") or {}).get("summary_out_root")
    summary_out_root = ROOT / str(summary_out_root_value) if summary_out_root_value else None
    paths: dict[str, Path] = {}
    for row in rows:
        target = ROOT / str(row["future_output_path"])
        if summary_out_root is not None:
            target = summary_out_root / target.name
        paths[str(row["artifact_id"])] = target
    return paths


def _runtime_launch_prefix(run_cfg: dict[str, Any]) -> list[str]:
    execution_surface = str(run_cfg.get("execution_surface") or "").strip()
    if execution_surface == "host_unsandboxed_caffeinate_required":
        return ["caffeinate", "-dimsu"]
    return []


def _accuracy_launch_prefix(
    accuracy_backend: str,
    run_cfg: dict[str, Any],
) -> list[str]:
    if accuracy_backend != "mlx":
        return []
    return _runtime_launch_prefix(run_cfg)


def _launch_policy_label(prefix: list[str]) -> str:
    if not prefix:
        return "direct"
    return " ".join(prefix)


def _default_mlx_parity_report_ref() -> str:
    completion_note = ROOT / "docs" / "reports" / "20260331_mlx_accuracy_backend_bounded_closure_completion_note.md"
    if completion_note.is_file():
        return str(completion_note)
    tmp_report = Path("/tmp/mlx_mobilevit_parity_report.json")
    if tmp_report.is_file():
        return str(tmp_report)
    return ""


def _resolve_parity_report_ref(
    accuracy_backend: str,
    parity_report_ref: str | None,
) -> str:
    if accuracy_backend != "mlx":
        return ""
    if parity_report_ref:
        return str(parity_report_ref)
    return _default_mlx_parity_report_ref()


def _parity_status_for_backend(accuracy_backend: str) -> str:
    if accuracy_backend == "mlx":
        return "mlx_parity_optimization_deferred"
    return "not_applicable_torch_backend"


def _phase1_command(python_bin: str, config_path: Path, run_cfg: dict[str, Any]) -> list[str]:
    return _runtime_launch_prefix(run_cfg) + [python_bin, str(DEFAULT_PHASE1_RUNNER), "--config", str(config_path)]


def _sanitize_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _profile_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    noise_cfg = bundle["noise"]
    rows: list[dict[str, Any]] = []
    dense_variants = [dict(noise_cfg["dense"])] + [dict(item) for item in (noise_cfg.get("dense_support") or [])]
    for dense in dense_variants:
        model = str(dense.get("model") or "")
        if not model:
            raise SystemExit("noise.dense and noise.dense_support rows require model")
        for gaussian in dense["gaussian_noise_std"]:
            for alpha in dense["crosstalk_alpha"]:
                rows.append(
                    {
                        "family_id": _noise_family_id(model=model, sweep_resolution="dense"),
                        "model": model,
                        "profile": f"dense_a{_sanitize_float(float(alpha))}_g{_sanitize_float(float(gaussian))}",
                        "sweep_resolution": "dense",
                        "crosstalk_alpha": float(alpha),
                        "gaussian_noise_std": float(gaussian),
                    }
                )
    sparse_rows = noise_cfg.get("sparse_support") or []
    for item in sparse_rows:
        model = str(item["model"])
        family_id = _noise_family_id(model=model, sweep_resolution="sparse")
        for profile in item.get("profiles") or []:
            rows.append(
                {
                    "family_id": family_id,
                    "model": model,
                    "profile": str(profile["profile"]),
                    "sweep_resolution": "sparse",
                    "crosstalk_alpha": float(profile["crosstalk_alpha"]),
                    "gaussian_noise_std": float(profile["gaussian_noise_std"]),
                }
            )
    return rows


def _phase1_cfg_for_profile(
    template: dict[str, Any],
    *,
    row: dict[str, Any],
    run_id: str,
    accuracy_results_csv: Path | None = None,
    accuracy_source_run_id: str | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(template)
    cfg["run"]["run_id"] = run_id
    cfg["run"]["experiment_id"] = f"{row['family_id']}_MODEL"
    cfg["run"]["notes"] = f"fuller_noise_profile:{row['profile']}"
    cfg["models"]["keys"] = [row["model"]]
    cfg["switches"] = {"meso": True, "flow": True, "det": False, "sparse": False, "phy": True}
    cfg["noise_injection"]["enabled"] = True
    cfg["noise_injection"]["crosstalk_alpha"] = float(row["crosstalk_alpha"])
    cfg["noise_injection"]["gaussian_noise_std"] = float(row["gaussian_noise_std"])
    cfg["p1_align"]["gaussian_noise_std_ref"] = float(row["gaussian_noise_std"])
    cfg["p1_align"]["crosstalk_alpha_ref"] = float(row["crosstalk_alpha"])
    if accuracy_results_csv is not None:
        accuracy_cfg = cfg.setdefault("accuracy", {})
        accuracy_cfg["source_csv"] = str(accuracy_results_csv)
        accuracy_cfg["context_run_id"] = str(accuracy_source_run_id or "")
        accuracy_cfg["require_context_match"] = False
    cfg["outputs"]["append_master"] = False
    return cfg


def _accuracy_command(
    *,
    accuracy_backend: str,
    launch_prefix: list[str] | None = None,
    python_bin: str,
    eval_script: Path,
    imagenet_val: str,
    weights_dir: str | None,
    weights_npz: str | None,
    results_csv: Path,
    row: dict[str, Any],
    eval_batch_size: int,
    max_eval_samples: int | None,
    seed: int,
    append: bool,
) -> list[str]:
    command = list(launch_prefix or []) + [
        python_bin,
        str(eval_script),
        "--imagenet_val",
        imagenet_val,
        "--opencv_pipeline",
        "--models",
        str(row["model"]),
        "--device",
        "mps",
        "--eval_batch_size",
        str(eval_batch_size),
        "--results_csv",
        str(results_csv),
        "--run_id",
        str(row["accuracy_run_id"]),
        "--source_run_id",
        str(row["accuracy_run_id"]),
        "--seed",
        str(seed),
        "--profile",
        str(row["profile"]),
        "--sweep_resolution",
        str(row["sweep_resolution"]),
        "--workload",
        "W0_mobilevit_imagenet",
        "--crosstalk_alpha",
        str(row["crosstalk_alpha"]),
        "--quant_bits",
        "8",
    ]
    if accuracy_backend == "torch":
        command.extend(
            [
                "--workers",
                "0",
                "--gaussian_noise_std",
                str(row["gaussian_noise_std"]),
                "--enable_attention",
            ]
        )
    elif accuracy_backend == "mlx":
        command.extend(
            [
                "--noise_sigma_lsb",
                str(row["gaussian_noise_std"]),
            ]
        )
    else:
        raise SystemExit(f"Unsupported accuracy backend: {accuracy_backend}")
    if append:
        command.append("--append")
    if max_eval_samples is not None:
        command.extend(["--max_eval_samples", str(max_eval_samples)])
    if weights_npz:
        command.extend(["--weights_npz", weights_npz])
    elif weights_dir:
        command.extend(["--weights_dir", weights_dir])
    return command


def _materialize_manifest(
    *,
    bundle: dict[str, Any],
    python_bin: str,
    imagenet_val: str,
    weights_dir: str | None,
    weights_npz_manifest: Path | None,
    accuracy_backend: str,
    parity_report_ref: str | None = None,
) -> list[dict[str, Any]]:
    template = _load_yaml(ROOT / str(bundle["inputs"]["phase1_template_yaml"]))
    generated_root = ROOT / str(bundle["paths"]["generated_config_dir"]) / "noise"
    raw_accuracy_dir = ROOT / str(bundle["paths"]["noise_raw_accuracy_dir"])
    eval_batch_size = int(bundle["noise"]["eval_batch_size"])
    max_eval_samples = _optional_int(bundle["noise"].get("max_eval_samples"))
    seed_cfg = dict((bundle.get("noise") or {}).get("seed_policy") or {})
    default_seeds = _int_list(seed_cfg.get("default_seeds"), default=[0])
    sparse_support_seeds = _int_list(seed_cfg.get("sparse_support_seeds"), default=default_seeds)
    dense_representative_profiles = list(seed_cfg.get("dense_representative_profiles") or [])
    manifest_rows: list[dict[str, Any]] = []
    eval_script = (
        DEFAULT_MLX_EVAL_SCRIPT if accuracy_backend == "mlx" else DEFAULT_EVAL_SCRIPT
    )
    weights_npz_by_model = _load_weights_npz_manifest(weights_npz_manifest)
    for row in _profile_rows(bundle):
        slug = f"{row['model']}_{row['profile']}"
        phase1_run_id = f"{bundle['meta']['tag']}_{slug}_phase1"
        accuracy_run_id = f"{bundle['meta']['tag']}_{slug}_acc"
        accuracy_source_run_id = accuracy_run_id
        accuracy_results_csv = raw_accuracy_dir / f"{slug}.csv"
        phase1_cfg = _phase1_cfg_for_profile(
            template,
            row=row,
            run_id=phase1_run_id,
            accuracy_results_csv=accuracy_results_csv,
            accuracy_source_run_id=accuracy_source_run_id,
        )
        phase1_cfg_path = generated_root / f"{slug}.yaml"
        _write_yaml(phase1_cfg_path, phase1_cfg)
        phase1_run_dir = ROOT / "experiments" / "results" / "runs" / phase1_run_id
        resolved_parity_report_ref = _resolve_parity_report_ref(
            accuracy_backend,
            parity_report_ref,
        )
        accuracy_launch_prefix = _accuracy_launch_prefix(
            accuracy_backend,
            phase1_cfg["run"],
        )
        phase1_launch_prefix = _runtime_launch_prefix(phase1_cfg["run"])
        manifest_row = {
            **row,
            "accuracy_backend": accuracy_backend,
            "engine": accuracy_backend,
            "parity_status": _parity_status_for_backend(accuracy_backend),
            "parity_report_ref": resolved_parity_report_ref,
            "accuracy_run_id": accuracy_run_id,
            "accuracy_source_run_id": accuracy_source_run_id,
            "representative_profile": "",
            "source_run_tag": str(bundle["meta"]["tag"]),
            "source_status": "planned_current_basis",
            "evidence_basis": "current_basis_noise_preflight",
            "robustness_claim_status": "bounded_sensitivity",
            "phase1_run_id": phase1_run_id,
            "accuracy_results_csv": str(accuracy_results_csv),
            "phase1_config_yaml": str(phase1_cfg_path),
            "phase1_run_dir": str(phase1_run_dir),
            "accuracy_launch_policy": _launch_policy_label(accuracy_launch_prefix),
            "phase1_launch_policy": _launch_policy_label(phase1_launch_prefix),
        }
        resolved_weights_npz = (
            weights_npz_by_model.get(str(row["model"]))
            if accuracy_backend == "mlx"
            else None
        )
        if resolved_weights_npz:
            manifest_row["weights_npz"] = resolved_weights_npz
        seed_list: list[int]
        if str(row["sweep_resolution"]) == "dense":
            seed_list = default_seeds
            for rep in dense_representative_profiles:
                if (
                    float(rep["crosstalk_alpha"]) == float(row["crosstalk_alpha"])
                    and float(rep["gaussian_noise_std"]) == float(row["gaussian_noise_std"])
                ):
                    manifest_row["representative_profile"] = str(rep.get("profile") or "")
                    seed_list = default_seeds + [
                        seed for seed in _int_list(rep.get("extra_seeds"), default=[]) if seed not in set(default_seeds)
                    ]
                    break
        else:
            seed_list = sparse_support_seeds
        manifest_row["accuracy_seed_list"] = ";".join(str(seed) for seed in seed_list)
        accuracy_commands = [
            _accuracy_command(
                accuracy_backend=accuracy_backend,
                launch_prefix=accuracy_launch_prefix,
                python_bin=python_bin,
                eval_script=eval_script,
                imagenet_val=imagenet_val,
                weights_dir=weights_dir,
                weights_npz=resolved_weights_npz,
                results_csv=accuracy_results_csv,
                row=manifest_row,
                eval_batch_size=eval_batch_size,
                max_eval_samples=max_eval_samples,
                seed=seed,
                append=index > 0,
            )
            for index, seed in enumerate(seed_list)
        ]
        phase1_cmd = _phase1_command(python_bin, phase1_cfg_path, phase1_cfg["run"])
        manifest_row["accuracy_command"] = " && ".join(
            " ".join(shlex.quote(part) for part in command) for command in accuracy_commands
        )
        manifest_row["phase1_command"] = " ".join(shlex.quote(part) for part in phase1_cmd)
        manifest_rows.append(manifest_row)
    return manifest_rows


def _run_manifest(manifest_rows: list[dict[str, Any]]) -> None:
    for row in manifest_rows:
        accuracy_commands = [item.strip() for item in str(row["accuracy_command"]).split("&&") if item.strip()]
        for command_text in accuracy_commands:
            subprocess.run(shlex.split(command_text), cwd=str(ROOT), check=True)
        subprocess.run(shlex.split(str(row["phase1_command"])), cwd=str(ROOT), check=True)


def _matching_accuracy_rows(path: Path, *, model: str, alpha: float, gaussian: float) -> list[dict[str, str]]:
    rows = _read_csv(path)
    matched: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("baseline")) == "True":
            continue
        if str(row.get("model")) != model:
            continue
        try:
            alpha_value = float(row.get("crosstalk_alpha") or 0.0)
            gaussian_value = float(row.get("gaussian_noise_std") or 0.0)
        except ValueError:
            continue
        if alpha_value == alpha and gaussian_value == gaussian:
            matched.append(row)
    if not matched:
        raise SystemExit(f"Missing accuracy row for model={model} alpha={alpha} gaussian={gaussian} in {path}")
    return matched


def _single_phase1_row(run_dir: Path) -> dict[str, str]:
    rows = _read_csv(run_dir / "phase1_summary.csv")
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one phase1_summary row in {run_dir}")
    return rows[0]


def _unique_strings(values: list[str]) -> str:
    resolved: list[str] = []
    seen: set[str] = set()
    for value in values:
        entry = str(value or "").strip()
        if not entry or entry in seen:
            continue
        seen.add(entry)
        resolved.append(entry)
    return ";".join(resolved)


def _build_summary_rows(manifest_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        accuracy_rows = _matching_accuracy_rows(
            Path(str(row["accuracy_results_csv"])),
            model=str(row["model"]),
            alpha=float(row["crosstalk_alpha"]),
            gaussian=float(row["gaussian_noise_std"]),
        )
        phase1_row = _single_phase1_row(Path(str(row["phase1_run_dir"])))
        top1 = _mean([float(accuracy_row["top1"]) for accuracy_row in accuracy_rows])
        drop_values: list[float] = []
        for accuracy_row in accuracy_rows:
            top1_delta = accuracy_row.get("top1_delta")
            drop_values.append(0.0 if top1_delta in ("", None) else max(0.0, -float(top1_delta)))
        acc_drop_pp = _mean(drop_values)
        summary_row = {
            "model": row["model"],
            "profile": row["profile"],
            "sweep_resolution": row["sweep_resolution"],
            "crosstalk_alpha": row["crosstalk_alpha"],
            "gaussian_noise_std": row["gaussian_noise_std"],
            "accuracy_backend": row["accuracy_backend"],
            "engine": row["engine"],
            "parity_status": row["parity_status"],
            "parity_report_ref": row["parity_report_ref"],
            "acc_top1": top1,
            "acc_drop_pp": acc_drop_pp,
            "latency_ms": float(phase1_row["latency_ms"]),
            "energy_j": float(phase1_row["energy_j"]),
            "accuracy_results_csv": row["accuracy_results_csv"],
            "accuracy_source_run_ids": _unique_strings(
                [str(accuracy_row.get("source_run_id") or "") for accuracy_row in accuracy_rows]
            ),
            "accuracy_seeds": _unique_strings(
                [str(accuracy_row.get("seed") or "") for accuracy_row in accuracy_rows]
            ),
            "seed_count": len({str(accuracy_row.get("seed") or "") for accuracy_row in accuracy_rows if str(accuracy_row.get("seed") or "").strip()}),
            "complete": "true" if len({str(accuracy_row.get("seed") or "") for accuracy_row in accuracy_rows if str(accuracy_row.get("seed") or "").strip()}) >= 3 else "false",
            "representative_profile": row.get("representative_profile", ""),
            "source_run_tag": row.get("source_run_tag", ""),
            "source_status": "regenerated_current",
            "evidence_basis": row.get("evidence_basis", "current_basis_noise"),
            "robustness_claim_status": row.get("robustness_claim_status", "bounded_sensitivity"),
            "phase1_run_id": row["phase1_run_id"],
            "phase1_run_dir": row["phase1_run_dir"],
            "accuracy_launch_policy": row["accuracy_launch_policy"],
            "phase1_launch_policy": row["phase1_launch_policy"],
        }
        artifact_id = _noise_artifact_id(
            model=str(row["model"]),
            sweep_resolution=str(row["sweep_resolution"]),
        )
        grouped.setdefault(artifact_id, []).append(summary_row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the optimized fuller noise sweeps.")
    parser.add_argument("--program_contract", type=Path, default=DEFAULT_PROGRAM_CONTRACT)
    parser.add_argument("--wrapper_manifest_out", type=Path, default=None)
    parser.add_argument("--legacy_execute", action="store_true")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--imagenet_val", default=None)
    parser.add_argument("--weights_dir", default=None)
    parser.add_argument("--weights_npz_manifest", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--accuracy_backend",
        choices=("torch", "mlx"),
        default="mlx",
        help="Accuracy evaluation backend. Active mainline uses MLX.",
    )
    parser.add_argument(
        "--parity_report_ref",
        default=None,
        help="Optional provenance reference recorded on MLX-backed manifest/summary rows.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    args = parser.parse_args()

    if not args.legacy_execute:
        build_fuller_experiment_program(args.program_contract, root_dir=ROOT)
        execution_payload = materialize_fuller_experiment_execution_plan(args.program_contract, root_dir=ROOT)
        phase4_payload = build_fuller_phase4_intake_contract(args.program_contract, root_dir=ROOT)
        report_payload = build_fuller_report_pack_contract(args.program_contract, root_dir=ROOT)
        payload = {
            "wrapper_mode": "fuller_experiment_program",
            "legacy_entrypoint": str(Path(__file__).resolve()),
            "selected_family": "noise_robustness",
            "program_contract": str((args.program_contract if args.program_contract.is_absolute() else (ROOT / args.program_contract)).resolve()),
            "execution_plan_csv": execution_payload["execution_plan_csv"],
            "phase4_intake_contract_csv": phase4_payload["phase4_intake_contract_csv"],
            "report_contract_csv": report_payload["report_contract_csv"],
        }
        if args.wrapper_manifest_out:
            args.wrapper_manifest_out.parent.mkdir(parents=True, exist_ok=True)
            args.wrapper_manifest_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return

    if not args.imagenet_val:
        raise SystemExit("--imagenet_val is required when --legacy_execute is set")

    bundle = _load_yaml(args.bundle)
    artifact_paths = _artifact_paths(bundle)
    manifest_rows = _materialize_manifest(
        bundle=bundle,
        python_bin=args.python,
        imagenet_val=args.imagenet_val,
        weights_dir=args.weights_dir,
        weights_npz_manifest=args.weights_npz_manifest,
        accuracy_backend=args.accuracy_backend,
        parity_report_ref=args.parity_report_ref,
    )
    manifest_path = ROOT / str(bundle["paths"]["noise_manifest_csv"])
    _write_csv(manifest_path, MANIFEST_FIELDS, manifest_rows)
    if args.dry_run:
        for row in manifest_rows:
            print(row["phase1_command"])
            print(row["accuracy_command"])
        return
    if args.execute:
        _run_manifest(manifest_rows)
    if not args.skip_build:
        grouped_rows = _build_summary_rows(manifest_rows)
        for artifact_id, rows in grouped_rows.items():
            _write_csv(artifact_paths[artifact_id], SUMMARY_FIELDS, rows)
            print(f"[fuller-noise] wrote {artifact_paths[artifact_id]}")


if __name__ == "__main__":
    main()
