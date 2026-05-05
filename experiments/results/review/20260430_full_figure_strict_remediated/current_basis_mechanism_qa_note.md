# Current-Basis Data-Figure Mechanism QA Note

Date: 2026-04-30T09:32:00Z
Carrier figure pack: `figures/paper_figures_20260430_full_figure_strict_remediated`
Primary mechanism evidence: `experiments/results/quick_reports/20260430_full_figure_strict_remediated`
Rendering script: `experiments/tools/render_fuller_phase4_paper_data_figures.py`

## Decision Table

| Figure | Decision | Source CSV | Rendering target | QA note |
|---|---|---|---|---|
| Fig3 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/fig3_phase4_runtime_accuracy_boundary.csv` | `render_fig6` | Supports bounded runtime/accuracy comparison; large accuracy gaps remain visible. |
| Fig4 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/fig4_runtime_accuracy_pareto.csv` | `render_fig7` | Pareto companion to Fig3; use only with tradeoff wording. |
| Fig5 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/fig5_bounded_sensitivity_current_basis.csv` | `render_fig9` | Three-model dense sensitivity envelope and seed-5 representative cells are rendered as Fig5_BoundedSensitivity. |
| Fig6 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/fig6_broad_scaling_flow_buffer_current_basis.csv` | `render_fig10` | Expanded-grid and holdout scaling rows are complete with repeat_count=5; footer is wrapped for page-scale readability. |
| Fig7 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/fig7_device_context.csv` | `render_fig11` | Device context is acceptable when measured-host and modeled-accelerator boundaries stay explicit. |
| Fig8 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/fig8_holdout_claim_boundary.csv` | `render_fig12` | Correctly blocks stronger SPARSE/FULLER wording. |
| AppF1 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/appf1_seed_range_variability.csv` | `render_appf1` | Variability context now includes explicit max-min Top-1 range labels/inset. |
| AppF2 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/appf2_data_figure_compatibility_matrix.csv` | `render_appf2` | Compatibility matrix uses final successor IDs and states the Fig9-Fig12 mechanism-schematic boundary. |
| AppF3 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/appf3_related_work_radar_scores.csv` | `render_appf3` | Related-work radar visibly states qualitative context only; do not use as empirical proof. |
| AppF4 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/appf4_mechanism_ablation_context.csv` | `render_appf4` | Current-basis mechanism ablation rows replace retained context; DET/SPARSE/FULLER carry tradeoff caveats. |
| AppF5 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/appf5_mechanism_energy_breakdown.csv` | `render_appf5` | Visible stacked components close to total energy and support source-of-gain discussion. |
| AppF6 | accept | `experiments/results/quick_reports/20260430_full_figure_strict_remediated/appf6_det_sparse_sweep_phase4_basis.csv` | `render_appf6` | DET non-monotonicity, catastrophic low-k rows, SPARSE gradual recovery, complete=true, seed_count=3, and ASTRA gaps are visible. |

## Experiment Gate

No new accelerator-backed outputs were generated for this final visual QA pass. The render uses frozen current-basis CSVs and keeps positive preservation claims for SPARSE, FULLER, or a future DET/SPARSE point blocked.
