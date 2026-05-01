# Data Pack Worker Figure Brief

Date: 2026-04-30

Run tag: 20260430_full_figure_strict_remediated

Figure IDs: Fig3-Fig8, AppF1-AppF6

Figure type: release-facing Matplotlib data-figure remediation pack

## Governed Sources

- render script: experiments/tools/render_fuller_phase4_paper_data_figures.py
- quick-report directory: experiments/results/quick_reports/20260430_full_figure_strict_remediated
- mechanism quick-report directory: experiments/results/quick_reports/20260430_full_figure_strict_remediated
- output pack: figures/paper_figures_20260430_full_figure_strict_remediated
- review directory: experiments/results/review/20260430_full_figure_strict_remediated

## Target Venue And Literature Style Anchors

- target venue or paper family: IEEE-style compact data figures for photonic and accelerator papers
- exemplar 1: CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md / Fig6-Fig8 / borrow compact comparison posture / avoid claim framing transfer
- exemplar 2: Lightening_Transformer_HPCA2024.md / Fig11-Fig15 / borrow restrained breakdown, scaling, and sensitivity-sweep density / avoid dataset semantics transfer
- exemplar 3: 2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md / Fig6-Fig8 / borrow speedup-energy rhythm and scalability split / avoid benchmark-equivalence wording
- exemplar 4: Noisy_Machines_Understanding_Noisy_Neural_Networks_and_Enhancing_Robustness_to_Analog_Hardware_Errors_Using_Distillation_arXiv2001.04974.md / Fig3-Fig4 / borrow errorbar/sensitivity posture / avoid method-claim transfer
- composition-only references logged in traceability handoff: yes

## Field Mapping Declaration

- Fig3: y=lane; left x=speedup_vs_astra; right x=top1_mean with top1_min/top1_max range; hatch=claim_boundary.
- Fig4: x=speedup_vs_astra; y=top1_mean; marker/face=claim_boundary; label=lane.
- Fig5: panels=model_variant; heatmap x=noise_sigma_lsb, y=crosstalk_alpha, color=acc_drop_pp; bounded sensitivity only.
- Fig6: heatmap rows=method, columns=model_variant; colors summarize mean throughput_images_s and mean latency_ms across the declared grid and holdout points.
- Fig7: platform panels for latency_ms, energy_j, avg_power_w, throughput_images_s; hatch=modeled FULLER endpoint.
- Fig8: y=lane; x=top1_mean; labels include speedup_vs_astra and holdout_gate.
- AppF1: x=lane; y=top1_mean with top1_min/top1_max range, plus explicit range labels/inset.
- AppF2: rows=legacy_figure_id; action=compatibility_action; successor=successor_figure_id after final numbering.
- AppF3: polar axes are qualitative related-work score dimensions copied from retained context rows.
- AppF4: x=experiment_id/mechanism_label; panels=speedup_vs_E0, energy_ratio_vs_E0, acc_delta_vs_E0_pp.
- AppF5: x=experiment_id/mechanism_label; stacked bars=memory_move_mj, conversion_control_mj, optical_static_mj.
- AppF6: left x=det_k_global and right x=sparse_tau_global; y=top1_mean with ASTRA paired baseline; complete=true and seed_count=3 rows.

## Units And Labels

- speedup: x relative to ASTRA or E0 as declared by each CSV
- accuracy: Top-1 accuracy (%) or accuracy drop in percentage points
- runtime/device context: milliseconds, joules, watts, and images/s as provided by frozen CSV inputs
- claim boundaries: Fig5 is bounded sensitivity only; Fig6 is declared-grid timing evidence only; no broad noise tolerance, universal scaling, silicon measurement, device superiority, or DET/SPARSE/FULLER accuracy-preservation claim

## Output Plan

- expected exports: SVG, PDF, PNG for remediated data figures
- canonical Fig5 asset stem: Fig5_BoundedSensitivity
- grayscale previews: generated in the remediated review directory
- traceability handoff: data-pack worker review artifacts only; pack registry/LaTeX reconciliation remains in sibling-agent scopes

## Rerun Decision

- decision: redraw only
- reason: the requested fixes are CSV successor IDs, visible boundary/range wording, title/stem naming, and layout wrapping
- accelerator/model run status: no CUDA, MPS, or local model evaluation is needed
