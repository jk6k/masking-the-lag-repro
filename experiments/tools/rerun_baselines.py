"""Rerun E5 and E6 baselines with updated template parameters.

This ensures the Overview data (Fig 10, 11) reflects the fixes to PHY/MESO power.
"""
import sys
import subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
TEMPLATE = REPO_ROOT / "configs" / "phase1_template.yaml"
RUNNER = ROOT / "tools" / "phase1_runner.py"

EXPERIMENT_SWITCH_MATRIX = {
    "E0": {"meso": False, "flow": False, "det": False, "sparse": False, "phy": False},
    "E1": {"meso": True,  "flow": False, "det": False, "sparse": False, "phy": False},
    "E2": {"meso": False, "flow": True,  "det": False, "sparse": False, "phy": False},
    "E3": {"meso": False, "flow": False, "det": True,  "sparse": False, "phy": False},
    "E4": {"meso": False, "flow": False, "det": False, "sparse": True,  "phy": False},
    "E5": {"meso": False, "flow": False, "det": False, "sparse": False, "phy": True},
    "E6": {"meso": True,  "flow": True,  "det": True,  "sparse": True,  "phy": True},
}

def _apply_switches(cfg, experiment_id):
    switches = EXPERIMENT_SWITCH_MATRIX[experiment_id]
    run_cfg = cfg.get("run") or {}
    run_cfg["experiment_id"] = experiment_id
    cfg["run"] = run_cfg
    cfg["switches"] = switches

    for key in ("meso", "flow", "sparse", "phy"):
        section = cfg.get(key) or {}
        section["enabled"] = switches[key]
        cfg[key] = section
    
    # Enable DET if needed (E6)
    sc_det = cfg.get("sc_det") or {}
    early_stop = sc_det.get("early_stop") or {}
    early_stop["enabled"] = switches["det"]
    sc_det["early_stop"] = early_stop
    cfg["sc_det"] = sc_det
    
    # Defaults handled by runner or template, but let's ensure enabled flags
    return cfg

def run_baseline(e_id, run_id):
    print(f"Generating config for {e_id} -> {run_id}...")
    with TEMPLATE.open("r") as f:
        cfg = yaml.safe_load(f)
    
    cfg = _apply_switches(cfg, e_id)
    cfg["run"]["run_id"] = run_id
    
    # Inject accuracy CSV if needed? E6 uses DET/Sparse.
    # Yes, E6 needs accuracy CSV to skip verification or run correctly?
    # Actually, E6 uses DET early stop which needs accuracy if k-search enabled?
    # But E6 baseline usually has fixed parameters or default?
    # Template has `early_stop.k_global: 64`.
    # And `accuracy.source_csv: ""`.
    # I should inject accuracy CSV here too!
    # Path: experiments/results/accuracy/accuracy_noise_quickpack_acc_20260212.csv
    acc_csv = "experiments/results/accuracy/accuracy_noise_quickpack_acc_20260212.csv"
    cfg.setdefault("accuracy", {})["source_csv"] = acc_csv
    
    # Also E0 baseline ref?
    # E6 speedup calc needs E0 baseline.
    # Path: experiments/results/runs/quickpack_accfix_20260212_e0/phase1_summary.csv
    base_csv = "experiments/results/runs/quickpack_accfix_20260212_e0/phase1_summary.csv"
    if e_id != "E0":
        cfg.setdefault("baseline_ref", {})["e0_latency_csv"] = base_csv

    out_cfg = ROOT / "results" / "generated_configs" / "rerun_baseline" / f"{run_id}.yaml"
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    with out_cfg.open("w") as f:
        yaml.safe_dump(cfg, f)
    
    print(f"Running {out_cfg.name}...")
    cmd = [sys.executable, str(RUNNER), "--config", str(out_cfg)]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

def main():
    # E0 must run first to provide baseline latency for others
    run_baseline("E0", "quickpack_accfix_20260212_e0")
    
    experiments = ["E1", "E2", "E3", "E4", "E5", "E6"]
    for e_id in experiments:
        run_id = f"quickpack_accfix_20260212_{e_id.lower()}"
        run_baseline(e_id, run_id)

if __name__ == "__main__":
    main()
