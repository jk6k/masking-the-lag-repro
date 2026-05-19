# Fuller Implementation Experiment Design Status

- tag: `20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense`
- state: `raw_collection_complete`
- active_freeze_retained: `20260402_det_sparse_mlx_final_cpu_context_repair_freeze`
- route_summary: `successor dense-support widening for MobileViT-XS/XXS has completed raw MPS collection, noise-model phase1 pairing, and governed dense noise summaries; active paper-facing freeze remains unchanged`
- built_artifacts:
  - `experiments/results/report_data/fuller_noise_paired_metrics_20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense.csv`
  - `experiments/results/report_data/fuller_noise_accuracy_summary_s_dense_20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense.csv`
  - `experiments/results/report_data/fuller_noise_accuracy_summary_xs_dense_20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense.csv`
  - `experiments/results/report_data/fuller_noise_accuracy_summary_xxs_dense_20260403_det_sparse_mlx_final_cpu_context_repair_xs_xxs_dense.csv`
- coverage_note: `all three governed dense summaries now cover the same 35 unique gaussian_noise_std x crosstalk_alpha coordinates; each CSV has 39 rows because the four representative extra-seed anchors are preserved as duplicate-coordinate profile rows and must be coordinate-aggregated before paper-facing surface rendering`
- next_gate: `rebuild a successor quick-report / figure-pack namespace from the new dense artifacts, then perform a focused Fig13/AppF7 evidence audit before any manuscript use`
