# SUDS TETC Hardware-Derived Scheduler Slack Traces

Date: `2026-05-13`
Tag: `20260513_tetc_pivot`
Roadmap item: `R2_hardware_derived_scheduler_slack_traces`
Acceptance state: `pass`
Stop-condition state: `no R2 hard stop`

## Scope

This R2 artifact derives SUDS slack from a modeled accelerator scheduler
rather than from a manually chosen analytical profile. It consumes the
architecture kernel rows, builds a Transformer kernel DAG, maps kernels
to DPTC tile resources, reserves memory/control/DAC/core/ADC/optical
queues, and emits per-kernel slack under five scheduler variants.

The trace remains architecture-scheduler evidence. It is not RTL timing,
post-layout timing, optical-device signoff, fabricated silicon, or
hardware measurement.

## Scheduler Variants

| Scheduler | Priority surface | Look-ahead | Dispatch gap | Core efficiency |
|---|---|---:|---:|---:|
| `fifo` | release order | `0.00 x median` | `0.035 x median` | `0.96x` |
| `asap` | earliest ready time, then short kernel | `0.05 x median` | `0.018 x median` | `1.00x` |
| `edf_deadline_aware` | earliest deadline first | `0.35 x median` | `0.012 x median` | `1.00x` |
| `utilization_aware` | tile/core demand and short-job packing | `0.55 x median` | `0.010 x median` | `1.06x` |
| `suds_aware` | low slack, deadline pressure, tile demand, and budget guard | `0.70 x median` | `0.006 x median` | `1.03x` |

## Promoted Schedule Links

| Workload | Condition | Scheduler | Trace ID | Kernels | p10 slack norm | Misses | Accuracy evidence |
|---|---|---|---|---:|---:|---:|---|
| `bert_base_glue_seq128` | `suds_pareto` | `suds_aware` | `r2_7b7427345a30be24` | 72 | 0.6738 | 0 | `measured_mps_glue` |
| `mobilevit_s_transformer_blocks_256` | `suds_pareto` | `suds_aware` | `r2_e1d13a0a1855de8b` | 54 | 0.6904 | 0 | `measured_mps_imagenet` |

## SUDS-Pareto Scheduler Ablation

| Workload | Scheduler | p10 slack norm | median slack norm | p90 slack norm | Misses | Queue wait ns | Core util | Distance vs FIFO |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bert_base_glue_seq128` | `fifo` | 0.5706 | 0.6037 | 0.7682 | 0 | 72.105 | 0.4873 | 0.0000 |
| `bert_base_glue_seq128` | `asap` | 0.6400 | 0.7581 | 0.9543 | 0 | 65.914 | 0.4888 | 0.4124 |
| `bert_base_glue_seq128` | `edf_deadline_aware` | 0.6475 | 0.7711 | 0.9618 | 0 | 65.130 | 0.4920 | 0.4436 |
| `bert_base_glue_seq128` | `utilization_aware` | 0.6794 | 0.8135 | 0.9792 | 0 | 0.000 | 0.4761 | 0.5507 |
| `bert_base_glue_seq128` | `suds_aware` | 0.6738 | 0.8090 | 0.9737 | 0 | 40.168 | 0.4890 | 0.5202 |
| `mobilevit_s_transformer_blocks_256` | `fifo` | 0.6523 | 0.8653 | 0.9724 | 0 | 27.166 | 0.6833 | 0.0000 |
| `mobilevit_s_transformer_blocks_256` | `asap` | 0.6754 | 0.8978 | 0.9801 | 0 | 5.777 | 0.6693 | 0.0967 |
| `mobilevit_s_transformer_blocks_256` | `edf_deadline_aware` | 0.6765 | 0.9020 | 0.9810 | 0 | 4.078 | 0.6702 | 0.1035 |
| `mobilevit_s_transformer_blocks_256` | `utilization_aware` | 0.6958 | 0.9102 | 0.9853 | 0 | 0.000 | 0.6340 | 0.1753 |
| `mobilevit_s_transformer_blocks_256` | `suds_aware` | 0.6904 | 0.9089 | 0.9854 | 0 | 0.039 | 0.6524 | 0.1503 |

## Where SUDS Helps And Where It Does Not

| Workload | Condition | SUDS-aware p10 slack | Misses | EDP ratio vs Lightening | Note |
|---|---|---:|---:|---:|---|
| `bert_base_glue_seq128` | `l1` | 0.6733 | 0 | 0.6652 | selector/budget ablation; SUDS-aware improves low-tail slack or deadline misses versus FIFO, but `utilization_aware` has the highest p10 slack in this scheduler-only ablation. |
| `bert_base_glue_seq128` | `signal_only` | 0.6733 | 0 | 0.7536 | selector/budget ablation; SUDS-aware improves low-tail slack or deadline misses versus FIFO, but `utilization_aware` has the highest p10 slack in this scheduler-only ablation. |
| `bert_base_glue_seq128` | `slack_only` | 0.6738 | 0 | 0.6656 | selector/budget ablation; SUDS-aware improves low-tail slack or deadline misses versus FIFO, but `utilization_aware` has the highest p10 slack in this scheduler-only ablation. |
| `bert_base_glue_seq128` | `suds_pareto` | 0.6738 | 0 | 0.6656 | main promoted trace; SUDS-aware improves low-tail slack or deadline misses versus FIFO, but `utilization_aware` has the highest p10 slack in this scheduler-only ablation. |
| `mobilevit_s_transformer_blocks_256` | `l1` | 0.6892 | 0 | 0.6663 | selector/budget ablation; SUDS-aware improves low-tail slack or deadline misses versus FIFO, but `utilization_aware` has the highest p10 slack in this scheduler-only ablation. |
| `mobilevit_s_transformer_blocks_256` | `signal_only` | 0.6892 | 0 | 0.7440 | selector/budget ablation; SUDS-aware improves low-tail slack or deadline misses versus FIFO, but `utilization_aware` has the highest p10 slack in this scheduler-only ablation. |
| `mobilevit_s_transformer_blocks_256` | `slack_only` | 0.6905 | 0 | 0.6708 | selector/budget ablation; SUDS-aware improves low-tail slack or deadline misses versus FIFO, but `utilization_aware` has the highest p10 slack in this scheduler-only ablation. |
| `mobilevit_s_transformer_blocks_256` | `suds_pareto` | 0.6904 | 0 | 0.9190 | main promoted trace; SUDS-aware improves low-tail slack or deadline misses versus FIFO, but `utilization_aware` has the highest p10 slack in this scheduler-only ablation. |

## Acceptance

- Scheduler variants present: `True`.
- Promoted `suds_pareto` linked to hardware-derived trace: `True`.
- Maximum scheduler distribution distance: `0.566732`.
- Minimum required distance before R2 stop: `0.010000`.
- Kernel DAG, tile mapping, queue state, and deadline model emitted: `True`.

The R2 stop condition is not triggered because scheduler slack
distributions differ across scheduler variants. Neutral or negative
SUDS-aware rows are retained as ablations rather than hidden.

## Artifacts

- Scheduler trace CSV: `experiments/results/report_data/suds_tetc_scheduler_traces_20260513_tetc_pivot.csv`
- Scheduler ablation CSV: `experiments/results/report_data/suds_tetc_scheduler_ablation_20260513_tetc_pivot.csv`
- Scheduler JSON: `experiments/results/report_data/suds_tetc_scheduler_traces_20260513_tetc_pivot.json`
- Report: `docs/reports/20260513_suds_tetc_scheduler_slack_traces.md`

## Regeneration

```bash
python3 experiments/tools/build_suds_tetc_scheduler_traces.py --tag 20260513_tetc_pivot
```
