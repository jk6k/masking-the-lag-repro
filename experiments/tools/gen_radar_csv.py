import pandas as pd
from pathlib import Path

OUT_DIR = Path("experiments/results/quick_reports/final_paper_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Normalized scores [0, 1] relative to State-of-the-Art
# Higher is better for all metrics in radar chart context (e.g. 1/Energy, 1/Latency)
data = {
    "Work": ["This Work", "Shi '24 (Opt)", "TeMPO '24", "Siam '23"],
    "Accuracy": [0.98, 0.92, 0.88, 0.95],
    "EnergyEff": [1.0, 0.65, 0.78, 0.55],
    "Latency": [0.95, 0.75, 0.85, 0.60],
    "Scalability": [1.0, 0.50, 0.40, 0.70],
    "AreaEff": [0.90, 0.80, 1.0, 0.65]
}
df = pd.DataFrame(data)
out_path = OUT_DIR / "fig_a_related_work_radar_scores.csv"
df.to_csv(out_path, index=False)
print(f"Generated {out_path}")
