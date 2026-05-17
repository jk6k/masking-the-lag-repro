"""Backfill reproducibility metadata for existing run config snapshots.

This helper updates `config_snapshot.yaml` under run directories to satisfy
strict metadata checks used by quick-report compliance gate:
1) ensure `run.seed` is present,
2) ensure split manifest paths are present when provided.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "results" / "runs"


def _read_master_seed(master_path: Path) -> str | None:
    if not master_path.exists():
        return None
    with master_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        seed = str(row.get("seed", "")).strip()
        if seed:
            return seed
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _resolve_run_dirs(prefixes: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for prefix in prefixes:
        pfx = prefix.strip()
        if not pfx:
            continue
        for run_dir in RUNS.glob(f"{pfx}*"):
            if not run_dir.is_dir():
                continue
            if run_dir.name in seen:
                continue
            if not (run_dir / "config_snapshot.yaml").exists():
                continue
            if not (run_dir / "master_metrics.csv").exists():
                continue
            seen.add(run_dir.name)
            out.append(run_dir)
    return sorted(out, key=lambda p: p.name)


def _normalize_manifest_path(raw: str | None) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    p = Path(text)
    if p.is_absolute():
        try:
            return str(p.relative_to(ROOT)).replace("\\", "/")
        except Exception:
            return text.replace("\\", "/")
    return text.replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill run metadata for strict gate.")
    parser.add_argument(
        "--run_prefixes",
        required=True,
        help="Comma-separated run_id prefixes, e.g. 20260228_opt_sync_core,20260228_opt_sync_scan",
    )
    parser.add_argument(
        "--calib_manifest",
        default="",
        help="Path to calib split manifest (relative to repo root preferred).",
    )
    parser.add_argument(
        "--eval_manifest",
        default="",
        help="Path to eval split manifest (relative to repo root preferred).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report planned changes.",
    )
    args = parser.parse_args()

    calib_manifest = _normalize_manifest_path(args.calib_manifest)
    eval_manifest = _normalize_manifest_path(args.eval_manifest)
    prefixes = [p.strip() for p in args.run_prefixes.split(",") if p.strip()]
    run_dirs = _resolve_run_dirs(prefixes)

    if not run_dirs:
        raise SystemExit("No matching runs found for --run_prefixes.")

    changed_runs = 0
    seed_fixes = 0
    calib_fixes = 0
    eval_fixes = 0

    for run_dir in run_dirs:
        cfg_path = run_dir / "config_snapshot.yaml"
        master_path = run_dir / "master_metrics.csv"
        cfg = _load_yaml(cfg_path)
        if not cfg:
            print(f"[skip] unreadable config: {cfg_path}")
            continue

        run_cfg = cfg.get("run") or {}
        data_cfg = cfg.get("data") or {}
        changed = False

        run_seed = run_cfg.get("seed")
        if run_seed is None or str(run_seed).strip() == "":
            inferred = _read_master_seed(master_path)
            if inferred is not None:
                run_cfg["seed"] = int(inferred) if inferred.isdigit() else inferred
                seed_fixes += 1
                changed = True

        if calib_manifest and not str(data_cfg.get("calib_manifest_csv") or "").strip():
            data_cfg["calib_manifest_csv"] = calib_manifest
            calib_fixes += 1
            changed = True
        if eval_manifest and not str(data_cfg.get("eval_manifest_csv") or "").strip():
            data_cfg["eval_manifest_csv"] = eval_manifest
            eval_fixes += 1
            changed = True

        if not changed:
            continue

        cfg["run"] = run_cfg
        cfg["data"] = data_cfg
        changed_runs += 1
        print(f"[update] {run_dir.name}")
        if not args.dry_run:
            cfg_path.write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )

    print(
        "[summary] "
        f"runs_scanned={len(run_dirs)} changed_runs={changed_runs} "
        f"seed_fixes={seed_fixes} calib_fixes={calib_fixes} eval_fixes={eval_fixes} "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
