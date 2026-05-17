# SUDS TETC R13 Final Freeze

Date: `2026-05-17`
Plan item: `R13-4`
Acceptance state: `pass`
Release commit: `not_requested`

## Scope

R13-4 freezes the active TETC submission-candidate artifacts after the R13-1
R12 evidence integration, R13-2 baseline parameter crosswalk, and R13-3
DeiT-Tiny conservative policy search. R13-3 passes only as a bounded secondary
DeiT-Tiny support point; the original R12g `e2_l1` DeiT-Tiny row remains a
vision-generality boundary and the two headline selected workloads remain
unchanged.

## R13-3 Result Added Before Freeze

| Item | Result |
|---|---|
| Command | `caffeinate -dimsu .venv311-mps/bin/python experiments/tools/build_suds_tetc_deit_tiny_conservative_policy_search.py --tag 20260517_r13 --device mps` |
| MPS guard | `device=mps`; `PYTORCH_ENABLE_MPS_FALLBACK=0`; MPS available |
| Screening | `screening_only`, 2,048 ImageNet validation images, seed 0 |
| Full candidates | `degrade_only_signal_safe`, `light_degrade_keep90` |
| Full validation | 3 seeds x 50,000 ImageNet validation images per selected candidate |
| Selected secondary point | `degrade_only_signal_safe` |
| Accuracy | mean top-1 delta `-0.0360` pp; worst-seed delta `-0.0800` pp |
| Architecture join | minimum modeled EDP improvement `6.901%` vs same-scope Lightening-style reference |
| Artifact | `experiments/results/report_data/suds_tetc_deit_tiny_conservative_policy_search_20260517_r13.json` |

## Gate Results

| Command | Result | Key output |
|---|---|---|
| `make repo-hygiene` | pass | `0 error(s), 0 warning(s)` |
| `caffeinate -dimsu make suds-tetc-final-gate` | pass | pivot gate `tetc_submission_ready`; science gate `science_gate_pass_local_submission_candidate` |
| `caffeinate -dimsu make suds-tetc-submission-figure-pack` | pass | regenerated Fig4-Fig7 in the active submission figure pack |
| `python3 experiments/tools/check_figure_numbering_registry.py --pack_dir figures/suds_tetc_20260516_submission_figure_pack` | pass | `active=7`; `main_active=Fig1..Fig7` |
| `cd paper && tectonic -X compile suds_tetc_architecture_manuscript.tex` | pass | wrote `paper/suds_tetc_architecture_manuscript.pdf`; underfull box warnings only |
| `caffeinate -dimsu make public-repro-build` | pass | wrote `<public_repro>` |
| `make public-repro-check` before render | pass | `0 error(s)` |
| `caffeinate -dimsu make public-repro-render` | pass | regenerated public Fig4-Fig7 under `build/rendered_figures` |
| `make public-repro-check` after render | pass | `0 error(s)` |
| `make dirty-audit` | pass | `51 dirty path(s)`, `51 managed`, `0 unmanaged` |

## Public Reproduction Refresh

The public reproduction manifest now exports the R13 crosswalk and R13-3
policy-search artifacts:

- `experiments/results/report_data/suds_tetc_baseline_parameter_crosswalk_20260517_r13.csv`
- `experiments/results/report_data/suds_tetc_baseline_parameter_crosswalk_20260517_r13.json`
- `experiments/results/report_data/suds_tetc_deit_tiny_conservative_policy_search_20260517_r13.csv`
- `experiments/results/report_data/suds_tetc_deit_tiny_conservative_policy_search_20260517_r13.json`
- `docs/reports/20260517_suds_tetc_baseline_parameter_crosswalk.md`
- `docs/reports/20260517_suds_tetc_deit_tiny_conservative_policy_search.md`

The generated package at `<public_repro>` includes
the R13-3 CSV, JSON, report, and generator script, and validates with zero
errors before and after public figure rendering.

## Claim Discipline

The manuscript and supplement describe R13-3 as a bounded secondary DeiT-Tiny
support point only. They keep the R12g `e2_l1` row as a boundary and do not
claim silicon, layout, bench-energy measurement, universal workload transfer,
universal vision-model applicability, or slack-only semantic importance.

## Dirty State

No release commit was created because the user did not request one. The
post-refresh dirty audit reported only managed roots:

- managed experiments: `20`
- managed figures: `16`
- managed docs: `10`
- managed paper: `2`
- managed Makefile: `1`
- managed configs: `1`
- managed submissions: `1`
- unmanaged paths: `0`

The dirty set is attributable to the managed R13/final-freeze bundle,
regenerated gates, regenerated figures, manuscript PDF/source, supplement, and
public reproduction manifest updates. Upload should use the active manuscript,
active seven-figure pack, supplement, and public reproduction package generated
by this freeze.

## Final Decision

R13-4 passes after R13-3 integration. The TETC package is locally
submission-ready subject to the remaining venue-specific IEEE TETC formatting,
bibliography, submission metadata, and institutional target-journal partition
checks.
