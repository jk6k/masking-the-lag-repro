"""Unified sweep config generator for Phase-1 experiments.

Generates YAML configs for parameter sweeps (k, tau, fanout, PHY N_wdm)
and optionally runs them via phase1_runner.py.

Usage:
    python tools/quickscan_sweep_gen.py \
        --template ../../configs/phase1_template.yaml \
        --prefix quickscan_dense_20260217 \
        --sweeps k,tau,fanout,phy_n \
        --e0-baseline results/runs/quickpack_accfix_20260212_e0/phase1_summary.csv \
        [--accuracy-csv path/to/accuracy.csv] \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT_DIR.parent
PHASE1_RUNNER = ROOT_DIR / "tools" / "phase1_runner.py"

# ---------------------------------------------------------------------------
# Default sweep grids (densified)
# ---------------------------------------------------------------------------
DEFAULT_K_GRID = [4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 129]
DEFAULT_TAU_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
DEFAULT_FANOUT_GRID = [2, 4, 6, 8, 12, 16, 24, 32]
DEFAULT_PHY_N_GRID = [4, 8, 12, 16, 20, 24, 32, 48, 64]

EXPERIMENT_SWITCH_MATRIX: dict[str, dict[str, bool]] = {
    "E0": {"meso": False, "flow": False, "det": False, "sparse": False, "phy": False},
    "E1": {"meso": True,  "flow": False, "det": False, "sparse": False, "phy": False},
    "E2": {"meso": False, "flow": True,  "det": False, "sparse": False, "phy": False},
    "E3": {"meso": False, "flow": False, "det": True,  "sparse": False, "phy": False},
    "E4": {"meso": False, "flow": False, "det": False, "sparse": True,  "phy": False},
    "E5": {"meso": False, "flow": False, "det": False, "sparse": False, "phy": True},
    "E6": {"meso": True,  "flow": True,  "det": True,  "sparse": True,  "phy": True},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def _estimate_sparse_active_fraction_from_tau(sparse_cfg: dict[str, Any]) -> float | None:
    tau = sparse_cfg.get("tau_global")
    if tau is None:
        return None
    try:
        tau_value = float(tau)
    except (TypeError, ValueError):
        return None
    if tau_value <= 0.0:
        return 1.0
    min_active_fraction = sparse_cfg.get("min_active_fraction")
    try:
        min_active_value = float(min_active_fraction) if min_active_fraction is not None else 0.0
    except (TypeError, ValueError):
        min_active_value = 0.0
    min_active_value = _clamp_unit_interval(min_active_value)
    return _clamp_unit_interval(max(min_active_value, 1.0 - tau_value))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _apply_switches(cfg: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    """Apply experiment switch matrix to config (same logic as phase1_matrix_runner)."""
    switches = EXPERIMENT_SWITCH_MATRIX[experiment_id]
    run_cfg = cfg.get("run") or {}
    run_cfg["experiment_id"] = experiment_id
    cfg["run"] = run_cfg
    cfg["switches"] = switches

    for key in ("meso", "flow", "sparse", "phy"):
        section = cfg.get(key) or {}
        section["enabled"] = switches[key]
        cfg[key] = section

    sc_det = cfg.get("sc_det") or {}
    early_stop = sc_det.get("early_stop") or {}
    early_stop["enabled"] = switches["det"]
    sc_det["early_stop"] = early_stop
    cfg["sc_det"] = sc_det

    # Set defaults for enabled sections
    if switches["meso"]:
        meso = cfg.get("meso") or {}
        if meso.get("fanout") is None or int(meso.get("fanout") or 0) == 0:
            meso["fanout"] = 4
        if meso.get("load_scale") is None:
            meso["load_scale"] = 0.85
        if meso.get("broadcast_overhead_mj") is None:
            meso["broadcast_overhead_mj"] = 0.1
        cfg["meso"] = meso

    if switches["flow"]:
        flow = cfg.get("flow") or {}
        if flow.get("latency_scale") is None:
            flow["latency_scale"] = 0.85
        cfg["flow"] = flow

    if switches["sparse"]:
        sparse = cfg.get("sparse") or {}
        if sparse.get("tau_global") is None:
            sparse["tau_global"] = 0.25
        sparse["use_tau_for_gating"] = True
        compat_active_fraction = _estimate_sparse_active_fraction_from_tau(sparse)
        if compat_active_fraction is not None:
            sparse["active_fraction"] = compat_active_fraction
        elif sparse.get("active_fraction") is None:
            sparse["active_fraction"] = 1.0
        cfg["sparse"] = sparse

    return cfg


# ---------------------------------------------------------------------------
# Sweep generators
# ---------------------------------------------------------------------------

def _gen_k_sweep(
    template: dict[str, Any],
    prefix: str,
    k_grid: list[int],
    baseline_csv: str,
    accuracy_csv: str,
    out_dir: Path,
) -> list[Path]:
    """Generate E3 (DET) configs sweeping k_global."""
    configs = []
    for k in k_grid:
        cfg = yaml.safe_load(yaml.safe_dump(template))  # deep copy
        cfg = _apply_switches(cfg, "E3")

        if accuracy_csv:
            cfg.setdefault("accuracy", {})["source_csv"] = accuracy_csv

        sc_det = cfg.get("sc_det") or {}
        early_stop = sc_det.get("early_stop") or {}
        early_stop["k_global"] = k
        early_stop["enabled"] = True
        sc_det["early_stop"] = early_stop
        cfg["sc_det"] = sc_det

        run_id = f"{prefix}_e3_k{k}"
        cfg["run"]["run_id"] = run_id
        if baseline_csv:
            cfg.setdefault("baseline_ref", {})["e0_latency_csv"] = baseline_csv

        cfg_path = out_dir / f"e3_k{k}.yaml"
        _dump_yaml(cfg_path, cfg)
        configs.append(cfg_path)
    return configs


def _gen_tau_sweep(
    template: dict[str, Any],
    prefix: str,
    tau_grid: list[float],
    baseline_csv: str,
    accuracy_csv: str,
    out_dir: Path,
) -> list[Path]:
    """Generate E4 SPARSE configs with tau-first control semantics."""
    configs = []
    for tau in tau_grid:
        cfg = yaml.safe_load(yaml.safe_dump(template))
        cfg = _apply_switches(cfg, "E4")

        if accuracy_csv:
            cfg.setdefault("accuracy", {})["source_csv"] = accuracy_csv

        sparse = cfg.get("sparse") or {}
        sparse["tau_global"] = tau
        sparse["use_tau_for_gating"] = True
        sparse["active_fraction"] = _estimate_sparse_active_fraction_from_tau(sparse) or 1.0
        sparse["enabled"] = True
        cfg["sparse"] = sparse

        # Label: t05 for tau=0.05, t50 for tau=0.50
        t_label = f"t{int(tau * 100):02d}"
        run_id = f"{prefix}_e4_{t_label}"
        cfg["run"]["run_id"] = run_id
        if baseline_csv:
            cfg.setdefault("baseline_ref", {})["e0_latency_csv"] = baseline_csv

        cfg_path = out_dir / f"e4_{t_label}.yaml"
        _dump_yaml(cfg_path, cfg)
        configs.append(cfg_path)
    return configs


def _gen_fanout_sweep(
    template: dict[str, Any],
    prefix: str,
    fanout_grid: list[int],
    baseline_csv: str,
    accuracy_csv: str,
    out_dir: Path,
) -> list[Path]:
    """Generate E1 (MESO) configs sweeping fanout."""
    configs = []
    for f in fanout_grid:
        cfg = yaml.safe_load(yaml.safe_dump(template))
        cfg = _apply_switches(cfg, "E1")

        if accuracy_csv:
            cfg.setdefault("accuracy", {})["source_csv"] = accuracy_csv

        meso = cfg.get("meso") or {}
        meso["fanout"] = f
        meso["enabled"] = True
        cfg["meso"] = meso

        run_id = f"{prefix}_e1_f{f}"
        cfg["run"]["run_id"] = run_id
        if baseline_csv:
            cfg.setdefault("baseline_ref", {})["e0_latency_csv"] = baseline_csv

        cfg_path = out_dir / f"e1_f{f}.yaml"
        _dump_yaml(cfg_path, cfg)
        configs.append(cfg_path)
    return configs


def _gen_phy_n_sweep(
    template: dict[str, Any],
    prefix: str,
    n_grid: list[int],
    baseline_csv: str,
    accuracy_csv: str,
    out_dir: Path,
) -> list[Path]:
    """Generate E5 (PHY) configs sweeping wdm_channels_n."""
    configs = []
    for n in n_grid:
        cfg = yaml.safe_load(yaml.safe_dump(template))
        cfg = _apply_switches(cfg, "E5")

        if accuracy_csv:
            cfg.setdefault("accuracy", {})["source_csv"] = accuracy_csv

        phy = cfg.get("phy") or {}
        phy["wdm_channels_n"] = n
        phy["enabled"] = True
        cfg["phy"] = phy

        run_id = f"{prefix}_e5_n{n}"
        cfg["run"]["run_id"] = run_id
        if baseline_csv:
            cfg.setdefault("baseline_ref", {})["e0_latency_csv"] = baseline_csv

        cfg_path = out_dir / f"e5_n{n}.yaml"
        _dump_yaml(cfg_path, cfg)
        configs.append(cfg_path)
    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sweep configs and optionally run them."
    )
    parser.add_argument(
        "--template",
        default=str(REPO_ROOT / "configs" / "phase1_template.yaml"),
        help="Template YAML path.",
    )
    parser.add_argument(
        "--prefix",
        default=f"quickscan_dense_{time.strftime('%Y%m%d')}",
        help="Run-id prefix for all sweep runs.",
    )
    parser.add_argument(
        "--sweeps",
        default="k,tau,fanout,phy_n",
        help="Comma-separated sweep types: k, tau, fanout, phy_n",
    )
    parser.add_argument(
        "--e0-baseline",
        default="",
        help="Path to E0 phase1_summary.csv for speedup computation.",
    )
    parser.add_argument(
        "--accuracy-csv",
        default="",
        help="Path to accuracy CSV.",
    )
    parser.add_argument(
        "--k-grid",
        default=",".join(str(x) for x in DEFAULT_K_GRID),
        help="Comma-separated k values.",
    )
    parser.add_argument(
        "--tau-grid",
        default=",".join(f"{x:.2f}" for x in DEFAULT_TAU_GRID),
        help="Comma-separated tau values.",
    )
    parser.add_argument(
        "--fanout-grid",
        default=",".join(str(x) for x in DEFAULT_FANOUT_GRID),
        help="Comma-separated fanout values.",
    )
    parser.add_argument(
        "--phy-n-grid",
        default=",".join(str(x) for x in DEFAULT_PHY_N_GRID),
        help="Comma-separated N_wdm values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate YAML configs, do not run experiments.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for phase1_runner.py.",
    )
    args = parser.parse_args()

    template_path = Path(args.template).resolve()
    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")

    template = _load_yaml(template_path)
    prefix = args.prefix.strip()
    sweeps = [s.strip().lower() for s in args.sweeps.split(",") if s.strip()]
    baseline_csv = args.e0_baseline.strip()
    accuracy_csv = args.accuracy_csv.strip()

    gen_dir = ROOT_DIR / "results" / "generated_configs" / prefix
    gen_dir.mkdir(parents=True, exist_ok=True)

    all_configs: list[tuple[str, Path]] = []

    if "k" in sweeps:
        k_grid = [int(x.strip()) for x in args.k_grid.split(",") if x.strip()]
        cfgs = _gen_k_sweep(template, prefix, k_grid, baseline_csv, accuracy_csv, gen_dir)
        for c in cfgs:
            all_configs.append(("k-sweep", c))
        print(f"[sweep-gen] Generated {len(cfgs)} k-sweep configs")

    if "tau" in sweeps:
        tau_grid = [float(x.strip()) for x in args.tau_grid.split(",") if x.strip()]
        cfgs = _gen_tau_sweep(template, prefix, tau_grid, baseline_csv, accuracy_csv, gen_dir)
        for c in cfgs:
            all_configs.append(("tau-sweep", c))
        print(f"[sweep-gen] Generated {len(cfgs)} tau-sweep configs")

    if "fanout" in sweeps:
        fanout_grid = [int(x.strip()) for x in args.fanout_grid.split(",") if x.strip()]
        cfgs = _gen_fanout_sweep(template, prefix, fanout_grid, baseline_csv, accuracy_csv, gen_dir)
        for c in cfgs:
            all_configs.append(("fanout-sweep", c))
        print(f"[sweep-gen] Generated {len(cfgs)} fanout-sweep configs")

    if "phy_n" in sweeps:
        phy_n_grid = [int(x.strip()) for x in args.phy_n_grid.split(",") if x.strip()]
        cfgs = _gen_phy_n_sweep(template, prefix, phy_n_grid, baseline_csv, accuracy_csv, gen_dir)
        for c in cfgs:
            all_configs.append(("phy_n-sweep", c))
        print(f"[sweep-gen] Generated {len(cfgs)} phy_n-sweep configs")

    print(f"\n[sweep-gen] Total: {len(all_configs)} configs in {gen_dir}")

    if args.dry_run:
        print("[sweep-gen] --dry-run: skipping execution.")
        for label, cfg_path in all_configs:
            print(f"  [{label}] {cfg_path.name}")
        return

    print(f"\n[sweep-gen] Running {len(all_configs)} experiments...\n")
    for i, (label, cfg_path) in enumerate(all_configs, 1):
        cmd = [args.python, str(PHASE1_RUNNER), "--config", str(cfg_path)]
        print(f"[{i}/{len(all_configs)}] [{label}] {cfg_path.name}", flush=True)
        try:
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        except subprocess.CalledProcessError as exc:
            print(f"  ⚠ FAILED (exit {exc.returncode}), continuing...")

    print(f"\n[sweep-gen] Done. {len(all_configs)} runs completed.")


if __name__ == "__main__":
    main()
