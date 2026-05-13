# SUDS TETC Delta Literature Audit

Date: `2026-05-13`
Scope: narrow post-Wave-B+ check for 2025-2026 architecture deltas that could
force a plan change before G1 manuscript promotion.

## Decision

No broad architecture re-research is needed. The current Wave B+ design-space
closure is sufficient for the next gate: freeze the selected DPTC operating
point, then move to manuscript integration and red-team.

The plan should change from "continue architecture optimization" to:

1. delta-audit the latest adjacent photonic Transformer work;
2. keep Lightening/HyAtten/TeMPO/ASTRA/ENLighten as the core architecture set;
3. treat new 2026 items as related-work or boundary-pressure, not as required
   same-fabric baselines;
4. promote G1 only after manuscript/table/figure integration and red-team.

## Delta Findings

| Item | Source | Relevance to SUDS | Plan impact |
|---|---|---|---|
| ASTRA 2026 arXiv refresh | [arXiv:2604.09759](https://arxiv.org/abs/2604.09759) | Confirms the stochastic photonic Transformer path is active and current. It uses optical stochastic multipliers and unary/analog homodyne accumulation for dynamic tensor computation. | Keep `astra_boundary`; update related-work wording to cite the 2026 arXiv version as the current public surface. |
| Light-Bound Transformers | [arXiv:2604.04330](https://arxiv.org/abs/2604.04330) | Adds a 2026 robustness/noise-training lane for silicon-photonic ViTs, with bank-level MR noise proxies and chance-constrained training. | Add as robustness boundary/future-work pressure. Do not add as a same-scope accelerator baseline. |
| Integrated electro-optic attention nonlinearities | [arXiv:2604.09512](https://arxiv.org/abs/2604.09512) | Targets attention nonlinearities such as Softmax/Sigmoid with TFLN MZMs, including 4-bit analog-unit quantization and noise characterization. | Add as non-GEMM attention-function boundary. Current simulator is GEMM/PPA focused, so no immediate model change. |
| PRISM long-context block selection | [arXiv:2603.21576](https://arxiv.org/abs/2603.21576) | Uses photonic ranking for KV block selection in long-context LLM inference, shifting the bottleneck from dense attention compute to memory-bound block retrieval. | Add as scope boundary. SUDS current workloads are short-sequence BERT/MobileViT blocks, not long-context KV-cache selection. |
| MXFormer CIM Transformer accelerator | [arXiv:2602.12480](https://arxiv.org/abs/2602.12480) | Non-photonic CIM architecture with weight-stationary Transformer blocks and strong claimed density/efficiency versus mixed accelerator classes. | Reviewer-pressure item only. Avoid broad "best accelerator" claims; compare only within matched photonic/DPTC scope. |

## Plan Adjustment

The active plan should treat Wave B+ as closed:

- `G2/G3/G4/G5` already pass.
- The selected operating point remains `tile_dim=32`, `tiles=4`,
  `cores_per_tile=2`, `sideband_group_cols=32`, `adc_sharing=temporal_accum`.
- `suds_pareto` is the only promoted main SUDS row after the science gate.
- `SUDS+L1`, `SUDS+signal`, and `suds_only` are retained as ablations or
  boundary context rather than promoted headline rows.
- TeMPO and ASTRA remain boundary fabrics.
- Signal-only, L1-only, HyAtten-style, and ENLighten-like selector wins should
  be written as local-selector boundary evidence.

## G1 Promotion Checklist

Before promoting G1, update or red-team:

1. Abstract and introduction: scheduler-derived budget interface, not direct
   slack selector.
2. Main comparison table: `suds_pareto` primary; `SUDS+L1`, `SUDS+signal`,
   and `suds_only` ablations.
3. Architecture design-space table: selected operating point and reason.
4. Related-work boundary table: Lightening, HyAtten, TeMPO, ASTRA, ENLighten,
   plus the 2026 delta items above.
5. Claim-boundary pass: no hardware-energy, physical-design, foundry, or
   deployment claims beyond the evidence.
6. Reproduction alignment: report-data artifacts and manuscript table labels
   must agree.
