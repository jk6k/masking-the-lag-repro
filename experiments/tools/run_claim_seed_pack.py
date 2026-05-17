"""Run seed pack for claim gate and merge seed rows into base runs.

Typical use:
1) reuse base run config snapshots as templates,
2) launch supplemental seed runs (e.g. seed 1/2),
3) merge seed rows back into base run `master_metrics.csv` so gate can count
   >=3 distinct seeds on canonical run IDs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "results" / "runs"
RUNNER = ROOT / "experiments" / "tools" / "phase1_runner.py"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write_csv_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for item in _parse_csv_list(raw):
        out.append(int(item))
    return out


def _seed_sort_key(seed_text: str) -> tuple[int, str]:
    s = str(seed_text or "").strip()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return (0, f"{int(s):08d}")
    return (1, s)


def _prepare_configs(
    *,
    base_runs: list[str],
    seeds: list[int],
    device: str,
    out_dir: Path,
    calib_manifest: str,
    eval_manifest: str,
    data_split: str,
    holdout_manifest: str,
    enable_stochastic_accuracy: bool,
    stochastic_std_pp: float,
    stochastic_noise_scale: float,
    stochastic_min_std_pp: float,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for base_run in base_runs:
        cfg_path = RUNS / base_run / "config_snapshot.yaml"
        if not cfg_path.exists():
            raise SystemExit(f"Missing config snapshot for base run: {base_run}")
        cfg = _load_yaml(cfg_path)

        for seed in seeds:
            run_id = f"{base_run}_s{seed}"
            cfg_seed = copy.deepcopy(cfg)
            run_cfg = cfg_seed.get("run") or {}
            data_cfg = cfg_seed.get("data") or {}
            accuracy_cfg = cfg_seed.get("accuracy") or {}

            run_cfg["run_id"] = run_id
            run_cfg["seed"] = int(seed)
            run_cfg["device"] = device
            cfg_seed["run"] = run_cfg

            if calib_manifest and not str(data_cfg.get("calib_manifest_csv") or "").strip():
                data_cfg["calib_manifest_csv"] = calib_manifest
            if eval_manifest and not str(data_cfg.get("eval_manifest_csv") or "").strip():
                data_cfg["eval_manifest_csv"] = eval_manifest
            if holdout_manifest and not str(data_cfg.get("holdout_manifest_csv") or "").strip():
                data_cfg["holdout_manifest_csv"] = holdout_manifest
            if data_split:
                data_cfg["split"] = data_split
            cfg_seed["data"] = data_cfg

            if enable_stochastic_accuracy:
                stoch_cfg = accuracy_cfg.get("stochastic_uncertainty") or {}
                stoch_cfg["enabled"] = True
                stoch_cfg["std_pp"] = float(stochastic_std_pp)
                stoch_cfg["noise_scale"] = float(stochastic_noise_scale)
                stoch_cfg["min_std_pp"] = float(stochastic_min_std_pp)
                accuracy_cfg["stochastic_uncertainty"] = stoch_cfg
            cfg_seed["accuracy"] = accuracy_cfg

            cfg_out = out_dir / f"{run_id}.yaml"
            _write_yaml(cfg_out, cfg_seed)
            jobs.append(
                {
                    "base_run": base_run,
                    "seed": seed,
                    "run_id": run_id,
                    "config": cfg_out,
                }
            )
    return jobs


def _run_jobs(jobs: list[dict[str, Any]], python_bin: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs, 1):
        cmd = [python_bin, str(RUNNER), "--config", str(job["config"])]
        print(f"[claim-seed] ({idx}/{len(jobs)}) run {job['run_id']} seed={job['seed']}")
        print(f"[claim-seed] cmd: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=str(ROOT))
        if proc.returncode != 0:
            failures.append({**job, "returncode": proc.returncode})
    return failures


def _merge_seed_rows(base_run: str, seeds: list[int]) -> tuple[int, int]:
    base_master = RUNS / base_run / "master_metrics.csv"
    base_fields, base_rows = _read_csv_rows(base_master)
    if not base_rows:
        raise SystemExit(f"Missing base master metrics rows: {base_master}")

    all_fields = list(base_fields)
    merged_map: dict[tuple[str, str], dict[str, Any]] = {}

    for row in base_rows:
        model = str(row.get("model", "")).strip()
        seed = str(row.get("seed", "")).strip()
        merged_map[(model, seed)] = dict(row)

    added = 0
    for seed in seeds:
        seed_run = f"{base_run}_s{seed}"
        seed_master = RUNS / seed_run / "master_metrics.csv"
        fields, rows = _read_csv_rows(seed_master)
        if not rows:
            raise SystemExit(f"Missing supplemental master metrics rows: {seed_master}")
        for f in fields:
            if f not in all_fields:
                all_fields.append(f)
        for row in rows:
            model = str(row.get("model", "")).strip()
            if not model:
                continue
            row_copy = dict(row)
            row_copy["run_id"] = base_run
            row_copy["seed"] = str(seed)
            key = (model, str(seed))
            if key not in merged_map:
                added += 1
            merged_map[key] = row_copy

    out_rows = sorted(
        merged_map.values(),
        key=lambda r: (
            str(r.get("model", "")),
            _seed_sort_key(str(r.get("seed", ""))),
        ),
    )
    backup = base_master.with_suffix(".csv.bak_seedmerge")
    if not backup.exists():
        backup.write_text(base_master.read_text(encoding="utf-8"), encoding="utf-8")
    _write_csv_rows(base_master, all_fields, out_rows)
    return len(out_rows), added


def main() -> None:
    parser = argparse.ArgumentParser(description="Run claim seed pack and merge seed rows to base runs.")
    parser.add_argument(
        "--base_runs",
        required=True,
        help="Comma list of canonical runs, e.g. A_e0,A_e6,A_e3_k64,A_e4_t20",
    )
    parser.add_argument(
        "--seeds",
        default="1,2",
        help="Comma list of supplemental seeds (default: 1,2).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="run.device used in generated configs (default: cuda).",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python interpreter for phase1_runner.py",
    )
    parser.add_argument(
        "--generated_dir",
        default="experiments/results/generated_configs/claim_seed_pack",
        help="Directory to store generated configs.",
    )
    parser.add_argument(
        "--calib_manifest",
        default="",
        help="Optional calib manifest path to inject when empty.",
    )
    parser.add_argument(
        "--eval_manifest",
        default="",
        help="Optional eval manifest path to inject when empty.",
    )
    parser.add_argument(
        "--holdout_manifest",
        default="",
        help="Optional holdout manifest path to inject when empty.",
    )
    parser.add_argument(
        "--data_split",
        default="",
        help="Optional data.split override for generated configs (e.g. eval,holdout).",
    )
    parser.add_argument(
        "--enable_stochastic_accuracy",
        action="store_true",
        help="Enable seed-driven stochastic accuracy uncertainty in generated configs.",
    )
    parser.add_argument(
        "--stochastic_std_pp",
        type=float,
        default=0.15,
        help="Base std (pp) for stochastic uncertainty path when enabled.",
    )
    parser.add_argument(
        "--stochastic_noise_scale",
        type=float,
        default=0.0,
        help="Extra std scaling coefficient multiplied by noise severity.",
    )
    parser.add_argument(
        "--stochastic_min_std_pp",
        type=float,
        default=0.0,
        help="Lower bound for stochastic std (pp).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute generated configs with phase1_runner.py.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge supplemental seed rows into base run master_metrics.csv.",
    )
    args = parser.parse_args()

    base_runs = _parse_csv_list(args.base_runs)
    seeds = _parse_seeds(args.seeds)
    generated_dir = ROOT / args.generated_dir

    if not base_runs:
        raise SystemExit("No base runs provided.")
    if not seeds:
        raise SystemExit("No supplemental seeds provided.")
    data_split = str(args.data_split).strip().lower()
    if data_split and data_split not in {"eval", "calib", "holdout"}:
        raise SystemExit("--data_split must be one of eval, calib, holdout.")

    jobs = _prepare_configs(
        base_runs=base_runs,
        seeds=seeds,
        device=args.device,
        out_dir=generated_dir,
        calib_manifest=args.calib_manifest,
        eval_manifest=args.eval_manifest,
        data_split=data_split,
        holdout_manifest=args.holdout_manifest,
        enable_stochastic_accuracy=bool(args.enable_stochastic_accuracy),
        stochastic_std_pp=float(args.stochastic_std_pp),
        stochastic_noise_scale=float(args.stochastic_noise_scale),
        stochastic_min_std_pp=float(args.stochastic_min_std_pp),
    )
    print(f"[claim-seed] generated configs: {generated_dir}")
    print(f"[claim-seed] jobs={len(jobs)}")

    if args.run:
        failures = _run_jobs(jobs, args.python_bin)
        if failures:
            print(f"[claim-seed] failures={len(failures)}")
            for item in failures:
                print(
                    f"[claim-seed] failed run_id={item['run_id']} "
                    f"seed={item['seed']} rc={item['returncode']}"
                )
            raise SystemExit(2)

    if args.merge:
        for base_run in base_runs:
            rows, added = _merge_seed_rows(base_run, seeds)
            print(f"[claim-seed] merged base={base_run} rows={rows} added_seed_rows={added}")


if __name__ == "__main__":
    main()
