import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def run(cmd):
    print(f"\n>>> Running: {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=str(ROOT))

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # 1. Re-run Baselines E0-E6
    # This prepares master_metrics for E0-E6 with consistent environment/params
    # print("\n=== STEP 1: Re-running Baselines E0-E6 ===")
    # run(f"{sys.executable} experiments/tools/rerun_baselines.py")
    
    # 2. Re-run Sweeps
    # Uses quickscan_sweep_gen.py to regenerate configs AND run them (--run-experiments)
    # Injecting accuracy CSV is critical for Phase 1 runner to pick up baseline accuracy
    # We use prefix 'quickscan_final_20260217' to distinguish from previous runs
    # print("\n=== STEP 2: Re-running Sweeps ===")
    acc_csv = "experiments/results/accuracy/accuracy_noise_quickpack_acc_20260212.csv"
    prefix = "quickscan_final_20260217"
    # quickscan_sweep_gen runs by default unless --dry-run is passed. Removed --run-experiments
    # Added --e0-baseline to inject baseline reference for speedup calculation
    e0_baseline = "experiments/results/runs/quickpack_accfix_20260212_e0/phase1_summary.csv"
    # run(f"{sys.executable} experiments/tools/quickscan_sweep_gen.py --accuracy-csv {acc_csv} --prefix {prefix} --e0-baseline {e0_baseline}")
    
    # 3. Build Reports
    # Combine E0-E6 baselines (prefix `quickpack_accfix_20260212`) 
    # with Sweeps (prefix `quickscan_final_20260217`)
    # 3. Build Reports
    # Combine E0-E6 baselines (prefix `quickpack_accfix_20260212`) 
    # with Sweeps (prefix `quickscan_final_20260217`)
    # print("\n=== STEP 3: Building Reports ===")
    out_dir = "experiments/results/quick_reports/final_paper_v2"
    # run(f"{sys.executable} experiments/tools/build_quick_reports.py "
    #     f"--main_prefix quickpack_accfix_20260212 "
    #     f"--quickscan_prefix {prefix} "
    #     f"--tau_prefix {prefix} "
    #     f"--phy_n_prefix {prefix} "
    #     f"--accuracy_csv {acc_csv} "
    #     f"--out_dir {out_dir} "
    #     f"--sync_master_csv")
        
    # 4. Render Figures
    print("\n=== STEP 4: Rendering Figures ===")
    run(f"{sys.executable} experiments/tools/render_paper_figures.py --quick_dir {out_dir} --out_dir figures/paper_figures_v2")

if __name__ == "__main__":
    main()
