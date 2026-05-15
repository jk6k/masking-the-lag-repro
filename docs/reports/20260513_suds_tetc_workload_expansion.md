# SUDS TETC Workload Expansion

Tag: `20260513_tetc_pivot`
Roadmap item: `R9_workload_generality_expansion`
Evidence label: `workload_generality_expansion`
Acceptance state: `pass`
Stop-condition state: `no R9 hard stop; DeiT-Tiny measured accuracy setup blocker is recorded and simulator-only traces are emitted`

## Scope

R9 adds simulator-only workload generality evidence for a new
Transformer-family workload and explicit sequence/batch sweeps. It
does not rerun or claim new measured accuracy. The governed MPS runtime
is probed and recorded so future accuracy runs remain constrained to
the project MPS policy.

## Decision

- R9 acceptance: `pass`
- Blockers: `none`
- MPS metadata complete: `True`
- New Transformer workload: `deit_tiny_patch16_224_batch1_r9,deit_tiny_patch16_224_batch4_r9,deit_tiny_patch16_224_batch8_r9`
- Sequence lengths: `64,128,256,512`
- Batch sizes: `1,4,8`
- Dataset/weights blocker recorded: `True`

## SUDS Pareto Generality Rows

| Workload | Seq | Batch | Energy improvement | EDP improvement | Accuracy status | Result class |
|---|---:|---:|---:|---:|---|---|
| `bert_base_seq64_batch1_r9` | 64 | 1 | 29.26% | 33.47% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq64_batch4_r9` | 64 | 4 | 29.19% | 33.41% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq64_batch8_r9` | 64 | 8 | 29.13% | 33.35% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq128_batch1_r9` | 128 | 1 | 29.23% | 33.44% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq128_batch4_r9` | 128 | 4 | 29.13% | 33.34% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq128_batch8_r9` | 128 | 8 | 29.07% | 33.28% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq256_batch1_r9` | 256 | 1 | 29.18% | 33.38% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq256_batch4_r9` | 256 | 4 | 29.05% | 33.27% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq256_batch8_r9` | 256 | 8 | 29.01% | 33.22% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq512_batch1_r9` | 512 | 1 | 29.10% | 33.30% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq512_batch4_r9` | 512 | 4 | 29.00% | 33.21% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `bert_base_seq512_batch8_r9` | 512 | 8 | 28.97% | 33.18% | `existing_mps_anchor` | `architecture_support_with_accuracy_boundary` |
| `deit_tiny_patch16_224_batch1_r9` | 197 | 1 | 7.07% | 6.90% | `not_run_boundary_or_blocker_recorded` | `architecture_support_with_accuracy_boundary` |
| `deit_tiny_patch16_224_batch4_r9` | 197 | 4 | 9.94% | 9.77% | `not_run_boundary_or_blocker_recorded` | `architecture_support_with_accuracy_boundary` |
| `deit_tiny_patch16_224_batch8_r9` | 197 | 8 | 10.77% | 10.61% | `not_run_boundary_or_blocker_recorded` | `architecture_support_with_accuracy_boundary` |

## Recorded Setup Boundary

- Setup blocker surface: `no governed measured accuracy row for this exact sequence-length/batch setting; no local governed DeiT-Tiny weights/dataset accuracy run found; R9 emits simulator-only traces instead of hiding the setup blocker`
- MPS probe: `pass` with torch `2.11.0`
- New DeiT-Tiny rows are architecture-only evidence until a governed
  ImageNet/weights accuracy run exists on `mps`.
- Long-sequence and larger-batch BERT rows are sequence/batch boundary
  traces unless a matching governed accuracy artifact is produced.

## Artifacts

- Workload expansion CSV: `experiments/results/report_data/suds_tetc_workload_expansion_20260513_tetc_pivot.csv`
- Workload expansion JSON: `experiments/results/report_data/suds_tetc_workload_expansion_20260513_tetc_pivot.json`
- Report: `docs/reports/20260513_suds_tetc_workload_expansion.md`

## Regeneration

```bash
make suds-tetc-workload-expansion
```
