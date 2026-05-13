# SUDS TETC Architecture Optimization Research

Date: `2026-05-13`
Scope: Wave B+ narrow architecture research for photonic Transformer accelerator design closure.

## Retrieval

Local literature entry point was `original_papers/AI_KNOWLEDGE_BASE.md`. The
governed vector query was run with MPS/MLX:

```bash
caffeinate -dimsu .venv311-mps/bin/python \
  experiments/tools/kb_rag_query.py \
  --query "photonic Transformer accelerator dataflow conversion fabric control granularity Lightening HyAtten TeMPO ASTRA ENLighten" \
  --device mps \
  --query-backend mlx
```

The top local hits were ENLighten, Lightening, HyAtten, and ASTRA. TeMPO was
then read directly from the same theme folder because it is the time-multiplexed
dynamic PTC boundary needed for the design-space comparison.

## Architecture Findings

| Work | Architecture point | Mechanism SUDS can borrow | Boundary that SUDS must not claim |
|---|---|---|---|
| Lightening | Dynamic full-range DPTC with output-stationary GEMM mapping, inter-core operand broadcast, 5 GHz tile timing, 2 MB global SRAM, 32 KB subarrays, and LT-B four-tile/two-core configuration. | Keep `lightening_dptc` as the reference fabric; select 32 x 32 tiles, 4 tiles, 2 cores/tile, and temporal accumulation as the default operating point. | Do not claim a new photonic core; SUDS is a control/budget interface on a DPTC-style fabric. |
| HyAtten | Conversion cost is the dominant bottleneck; 32 x 32 arrays emit 1024 analog outputs/cycle; more than 85 percent of signals can use low-resolution conversion while high-range signals are handled digitally. | Keep `hyatten_style` as a strong conversion-fabric baseline and interpret signal-only wins as local-selector boundary evidence. | Do not claim a local HyAtten implementation or its area-normalized results; only model a matched low-resolution/digital-fallback row. |
| TeMPO | Time-multiplexed dynamic PTC with slow-light modulators, multi-tile/multi-core sharing, and hierarchical photocurrent/temporal/digital accumulation. | Add `tempo_time_multiplexed` as an alternate ADC/readout sharing boundary in the sweep. | Do not merge TeMPO device/co-packaging assumptions into the selected Lightening-style DPTC point. |
| ASTRA | Stochastic optical Transformer fabric with homodyne/single-wavelength VDPEs, no DAC-heavy analog multi-level encoding, OS dataflow, pipeline scheduling, and temporal analog accumulation. | Add `astra_boundary` as a stochastic readout/encoding boundary and use it to stress whether DPTC-style conversion is the right comparison surface. | Do not treat stochastic signed multipliers, serializers, OAGs, or comb-laser assumptions as SUDS evidence. |
| ENLighten | PTC-aware low-rank plus structured column sparsity, L1-style column retention, densification, reconfigurable PTC granularity, power gating, and cross-PTC ADC/TIA sharing. | Use it to justify why L1/signal/HyAtten can be stronger local selectors; keep `suds_only` as ablation and promote `SUDS+L1` / `SUDS+signal` as main rows. | Do not claim ENLighten compression or reconfigurable sparse-engine benefits without a matching compression flow. |

## Design Decision

The architecture design is adequate only after adding a design-space artifact.
The previous gate-passing version had the right route, but it was too single
point: it could say "Lightening-style DPTC plus SUDS control" but could not
show why the chosen tile/control/conversion point was selected over nearby
photonic Transformer fabrics.

Wave B+ therefore selects:

| Knob | Selected value | Reason |
|---|---:|---|
| `tile_dim` | 32 | Matches Lightening/HyAtten DPTC comparison surface. |
| `tiles` | 4 | Matches LT-B-style reference. |
| `cores_per_tile` | 2 | Keeps per-tile accumulation comparable to the literature anchor. |
| `sideband_group_cols` | 32 | Uses the local RTL sideband calibration anchor. |
| `adc_sharing` | `temporal_accum` | Preserves output-stationary DPTC temporal accumulation. |

The sweep now covers `tile_dim={16,32,64}`, `tiles={2,4,8}`,
`cores_per_tile={1,2,4}`, `sideband_group_cols={16,32,64,128}`, and
`adc_sharing={per_array,per_tile,temporal_accum}`. The CSV/JSON outputs record
energy, latency, EDP, area, memory movement, optical-link energy, and control
overhead, with Pareto flags computed within each workload/condition group.

## SUDS Claim Optimization

The stronger architecture statement is:

> SUDS is a scheduler-derived budget interface for DPTC-style photonic
> Transformer accelerators.

That statement is safer and more useful than "slack selector." It separates
three responsibilities:

1. The scheduler exports timing slack and deadlines.
2. SUDS converts that schedule context into KEEP/DEGRADE/PRUNE budgets.
3. Local selectors, especially L1 and signal/overflow selectors, choose exact
columns within the budget.

Main comparisons should be `SUDS+L1` and `SUDS+signal`. `suds_only` remains an
ablation for the budget interface without a local selector. If L1, signal-only,
or HyAtten-style rows beat a SUDS composition, the paper should keep the result
as boundary evidence: local selection can beat budgeted composition under that
workload and fabric, rather than proving that the budget interface is invalid.

## Implementation Actions

- Extend simulator rows with `tempo_time_multiplexed` and `astra_boundary`.
- Emit `suds_transformer_architecture_design_space_20260513_tetc_pivot.csv`
  and matching JSON with selected-point and Pareto rows.
- Update the TETC manuscript source with "Architecture Design Space and
  Selected Operating Point" and a selected operating-point table.
- Keep the protected fallback manuscript unchanged.

