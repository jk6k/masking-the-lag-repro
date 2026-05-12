# SUDS J0 Baseline Protection

Tag: `20260512_j0_quality_boost`
Evidence label: `audit`
Promotion decision: `do_not_replace_main_freeze`

## Decision

The `20260511_suds_maxq` package remains the protected fallback before any J1
SPICE macro evidence is promoted. Existing worktree changes were recorded and
not reverted.

## J0 Gate Results

| Gate | Command | Status | Decision |
|---|---|---|---|
| Worktree snapshot | `git status --short --branch` | `recorded_dirty_existing_changes` | `do_not_replace_main_freeze` |
| Public repro build | `caffeinate -dimsu make public-repro-build` | `pass` | `do_not_replace_main_freeze` |
| Public repro check | `caffeinate -dimsu make public-repro-check` | `pass`, 0 errors | `do_not_replace_main_freeze` |
| Public repro render | `caffeinate -dimsu make public-repro-render` | `pass` | `do_not_replace_main_freeze` |
| ACM paper compile | `caffeinate -dimsu tectonic -X compile suds_paper_acmart.tex` | `pass`, TeX warnings only | `do_not_replace_main_freeze` |
| SPICE tool check | `command -v ngspice; command -v xyce; command -v Xyce` | `blocked_tool_missing` | `boundary` |

## Recorded Pre-J1 Worktree

- `M configs/public_repro_manifest.json`
- `M docs/coordination/active/README.md`
- `M docs/reports/20260511_p2p3_quality_suds_external_red_team_rounds.md`
- `M experiments/tools/render_suds_figures.py`
- `M paper/suds_paper_acmart.pdf`
- `M paper/suds_paper_acmart.tex`
- `?? docs/coordination/active/SUDS_JETC_FINAL_QUALITY_BOOST_PLAN.md`

## Artifacts

- CSV: `experiments/results/report_data/suds_j0_baseline_protection_20260512_j0_quality_boost.csv`
- JSON: `experiments/results/report_data/suds_j0_baseline_protection_20260512_j0_quality_boost.json`
- Report: `docs/reports/20260512_j0_suds_baseline_protection.md`
- Current compiled PDF SHA-256: `1e3b008520a1ac993f80bc0ad71fa22487260266ad40a9f4522d8e68c515b2d7`

## Regeneration

```bash
git status --short --branch
caffeinate -dimsu make public-repro-build
caffeinate -dimsu make public-repro-check
caffeinate -dimsu make public-repro-render
cd paper
caffeinate -dimsu tectonic -X compile suds_paper_acmart.tex
```

## Boundary

This report is audit evidence only. It does not promote J1 SPICE results and
does not make silicon, PDK, extracted-layout, measured hardware-energy, or
SPICE-closure claims.
