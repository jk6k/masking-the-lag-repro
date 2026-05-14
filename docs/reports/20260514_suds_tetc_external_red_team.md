# SUDS TETC External Red-Team And Public Mini Benchmark

Date: `2026-05-14`
Tag: `20260513_tetc_pivot`
Evidence label: `r11_public_mini_benchmark_external_red_team`
Status: `pass`
Stop-condition state: `no R11 hard stop`

## Public Mini Benchmark

- Manifest audit: `pass`
- Generated package validation: `pass`
- Public text leak audit: `pass`
- Validation error count: `0`
- Text leak match count: `0`

Required live commands:

- `make public-repro-build`
- `make public-repro-check`
- `make public-repro-render`
- `make public-repro-check`

## External Red-Team Record

- External reader status: `packet_ready_not_sent_from_codex`
- Issue policy: `fixed_or_accepted_risk`
- Fixed issues: `4`
- Accepted risks: `1`

No independent reader could be contacted inside this local execution
environment. The external-review gap is therefore recorded as an
accepted risk, not as independent validation. The manuscript and
pre-review report preserve that boundary.

## Issue Table

| ID | Lens | Severity | Status | Finding | Resolution | Claim change |
|---|---|---|---|---|---|---|
| `R11-EXT0` | `external_independence` | `medium` | `accepted_risk` | No independent reader could be contacted from this local Codex execution. | The public packet and request questions are ready; the manuscript and reports state that the local substitute is not equivalent to external review. | Keep external red-team review preferred before submission, advisory in local gates, and not counted as independent validation. |
| `R11-EXT1` | `simulator_credibility` | `high` | `fixed` | A reviewer will ask whether the architecture simulator is more than static accounting. | R1 event traces and R2 scheduler traces are included in the public package; R10 records failure and uncertainty boundaries. | Use event-level modeled architecture wording, not hardware-measured or cycle-accurate silicon wording. |
| `R11-EXT2` | `baseline_fairness` | `high` | `fixed` | Same-simulator baselines and alternate-fabric boundaries must be visible to a skeptical reviewer. | R4 same-simulator fairness and R5 Pareto rationale are exported; TeMPO/ASTRA/HyAtten remain boundary rows where assumptions differ. | Promote only suds_pareto and keep local-selector or alternate-fabric wins as boundary evidence. |
| `R11-EXT3` | `public_reproducibility` | `high` | `fixed` | The mini benchmark must be small, checkable, and free of private paths, data, weights, and legacy-route entry points. | The manifest whitelists compact traces, configs, reports, render scripts, checksums, and reviewer instructions; public validation scans the exported text surface. | Describe the public package as a reader-facing artifact package, not a full governed MPS rerun bundle. |
| `R11-EXT4` | `major_revision_readiness` | `medium` | `fixed` | Likely major-revision objections need pre-written response hooks tied to artifacts. | A reviewer-response scaffold maps simulator, baseline, calibration, workload, uncertainty, and reproducibility objections to exact evidence artifacts. | Do not answer with stronger headline numbers unless a regenerated science gate supports them. |

## Reader Request Packet

Ask one or two external readers to review these four questions:

1. Does the event-level simulator evidence make the PPA argument credible enough for an architecture paper?
2. Are the same-simulator baselines and boundary fabrics separated clearly enough?
3. Are the claim boundaries around ADC, RTL, PHY, and uncertainty conservative enough?
4. Can the public mini benchmark be checked without private data, weights, literature mirrors, or personal paths?

External feedback should be resolved by either a code/data fix, a
manuscript claim change, or an explicit accepted-risk entry before upload.
