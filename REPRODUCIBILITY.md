# Reproducibility Guide

The active public freeze is `20260430_full_figure_strict_remediated`.

Run the full public validation gate:

```bash
make repro-check
```

This checks the freeze pointer, figure numbering registry, data-figure evidence
pack, claim boundaries, and public repository surface.

Render the Matplotlib data figures into `build/rendered_figures/`:

```bash
make render-paper-figures
```

The reproduction package does not include ImageNet, model checkpoints, local
virtual environments, private paper mirrors, or internal project coordination
assets. Full accelerator-backed reruns on the project Mac use Apple Silicon
`mps`; this public package focuses on inspection and regeneration of the
promoted lightweight evidence and figures.
