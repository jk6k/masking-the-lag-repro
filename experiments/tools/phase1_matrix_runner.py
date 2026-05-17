#!/usr/bin/env python3
"""Materialize bounded phase-1 configs for legacy or non-legacy experiment lanes."""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

try:
    from . import phase1_runner
except ImportError:
    import phase1_runner  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENERATED_ROOT = ROOT / "experiments" / "results" / "generated_configs"
DEFAULT_PYTHON_BIN = ".venv/bin/python"


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


def _slugify_experiment_id(experiment_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(experiment_id).strip()).strip("_").lower()
    return slug or "phase1"


def _legacy_experiment_ids() -> list[str]:
    return list(phase1_runner.EXPERIMENT_SWITCH_MATRIX.keys())


def _resolve_requested_experiment_ids(template: dict[str, Any], experiments_arg: str) -> list[str]:
    template_run = template.get("run") or {}
    template_experiment_id = str(template_run.get("experiment_id") or "E0").strip().upper()
    legacy_ids = _legacy_experiment_ids()

    raw_tokens = [token for token in re.split(r"[\s,]+", experiments_arg.strip()) if token]
    if not raw_tokens:
        raw_tokens = ["all"]

    resolved: list[str] = []
    for token in raw_tokens:
        if token.lower() == "all":
            if template_experiment_id in phase1_runner.EXPERIMENT_SWITCH_MATRIX:
                resolved.extend(legacy_ids)
            else:
                resolved.append(template_experiment_id)
            continue
        resolved.append(token.upper())

    deduped: list[str] = []
    seen: set[str] = set()
    for experiment_id in resolved:
        if experiment_id in seen:
            continue
        seen.add(experiment_id)
        deduped.append(experiment_id)
    return deduped


def _resolve_switches(template: dict[str, Any], experiment_id: str) -> dict[str, bool]:
    if experiment_id in phase1_runner.EXPERIMENT_SWITCH_MATRIX:
        return dict(phase1_runner.EXPERIMENT_SWITCH_MATRIX[experiment_id])
    return phase1_runner._resolve_switches(copy.deepcopy(template), experiment_id)  # type: ignore[attr-defined]


def _materialize_config(
    template: dict[str, Any],
    *,
    experiment_id: str,
    run_prefix: str,
) -> tuple[str, dict[str, Any]]:
    cfg = copy.deepcopy(template)
    run_cfg = cfg.get("run") or {}
    run_cfg["experiment_id"] = experiment_id
    slug = _slugify_experiment_id(experiment_id)
    if run_prefix:
        run_cfg["run_id"] = f"{run_prefix}_{slug}"
        notes = str(run_cfg.get("notes") or "").strip()
        run_cfg["notes"] = f"{notes} {run_prefix}".strip()
    cfg["run"] = run_cfg

    switches = _resolve_switches(cfg, experiment_id)
    cfg["switches"] = dict(switches)
    phase1_runner._sync_section_enabled(cfg, switches)  # type: ignore[attr-defined]
    return slug, cfg


def materialize_configs(
    *,
    template_path: Path,
    experiments_arg: str,
    run_prefix: str,
    generated_root: Path,
) -> list[Path]:
    template = _load_yaml(template_path)
    experiment_ids = _resolve_requested_experiment_ids(template, experiments_arg)
    out_dir = generated_root / run_prefix

    generated_paths: list[Path] = []
    for experiment_id in experiment_ids:
        slug, cfg = _materialize_config(template, experiment_id=experiment_id, run_prefix=run_prefix)
        cfg_path = out_dir / f"{slug}.yaml"
        _write_yaml(cfg_path, cfg)
        generated_paths.append(cfg_path)
    return generated_paths


def _build_phase1_command(*, python_bin: str, cfg_path: Path, cfg: dict[str, Any]) -> list[str]:
    run_cfg = cfg.get("run") or {}
    launch_prefix = run_cfg.get("long_run_launch_prefix") or []
    if not isinstance(launch_prefix, list):
        raise SystemExit(f"run.long_run_launch_prefix must be a list in {cfg_path}")
    runner_path = Path(phase1_runner.__file__).resolve()
    return [*map(str, launch_prefix), python_bin, str(runner_path), "--config", str(cfg_path)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize phase-1 configs for bounded legacy or non-legacy experiment lanes."
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--experiments", default="all")
    parser.add_argument("--run_prefix", required=True)
    parser.add_argument("--generated_root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--python_bin", default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    template_path = args.template if args.template.is_absolute() else ROOT / args.template
    generated_root = args.generated_root if args.generated_root.is_absolute() else ROOT / args.generated_root

    generated_paths = materialize_configs(
        template_path=template_path,
        experiments_arg=args.experiments,
        run_prefix=args.run_prefix,
        generated_root=generated_root,
    )
    for cfg_path in generated_paths:
        cfg = _load_yaml(cfg_path)
        command = _build_phase1_command(python_bin=args.python_bin, cfg_path=cfg_path, cfg=cfg)
        print(f"[phase1-matrix-runner] generated={cfg_path}")
        print(" ".join(command))
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
