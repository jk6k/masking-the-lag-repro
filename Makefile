PYTHON ?= python3
FREEZE_TAG := 20260430_full_figure_strict_remediated
QUICK_DIR := experiments/results/quick_reports/20260430_full_figure_strict_remediated
PACK_DIR := figures/paper_figures_20260430_full_figure_strict_remediated
REVIEW_DIR := experiments/results/review/20260430_full_figure_strict_remediated
BUILD_FIG_DIR := build/rendered_figures
BUILD_REVIEW_DIR := build/rendered_review
IMAGENET_VAL ?=
WEIGHTS_DIR ?=
MLX_WEIGHTS_DIR ?= build/mlx_weights
WEIGHTS_NPZ_MANIFEST ?= $(MLX_WEIGHTS_DIR)/manifest.json
SPLIT_DIR ?= build/imagenet_splits
FULL_RERUN_PYTHON ?= $(PYTHON)

.PHONY: repro-check render-paper-figures test smoke status full-rerun-preflight imagenet-splits export-mlx-weights full-rerun-noise-dry-run full-rerun-noise-execute clean

repro-check:
	$(PYTHON) scripts/check_public_repro_repo.py --root .

render-paper-figures:
	rm -rf $(BUILD_FIG_DIR) $(BUILD_REVIEW_DIR)
	$(PYTHON) experiments/tools/render_fuller_phase4_paper_data_figures.py \
		--quick_dir $(QUICK_DIR) \
		--mechanism_quick_dir $(QUICK_DIR) \
		--out_dir $(BUILD_FIG_DIR) \
		--review_dir $(BUILD_REVIEW_DIR)
	$(PYTHON) scripts/render_fig2_pipeline_matplotlib.py \
		--input-csv experiments/results/quick_reports/20260424_fig2_hops_current_timeline/fig2_hops_current_timeline_input_20260424.csv \
		--run-tag $(FREEZE_TAG) \
		--out-dir $(BUILD_FIG_DIR) \
		--review-dir $(BUILD_REVIEW_DIR)

test:
	$(PYTHON) -m pytest -q experiments/tests/test_public_repro_surface.py

smoke: repro-check

full-rerun-preflight:
	$(PYTHON) scripts/validate_full_rerun_setup.py \
		--imagenet-val "$(IMAGENET_VAL)" \
		--weights-dir "$(WEIGHTS_DIR)" \
		--weights-npz-manifest "$(WEIGHTS_NPZ_MANIFEST)" \
		--require-mps \
		--require-mlx \
		--strict-imagenet

imagenet-splits:
	$(PYTHON) experiments/accuracy/make_imagenet_split_manifest.py \
		--imagenet_val "$(IMAGENET_VAL)" \
		--out_dir "$(SPLIT_DIR)" \
		--calib_fraction 0.1 \
		--seed 0

export-mlx-weights:
	$(PYTHON) experiments/tools/export_mobilevit_weights_npz.py \
		--models mobilevit_xxs,mobilevit_xs,mobilevit_s \
		--weights_dir "$(WEIGHTS_DIR)" \
		--out_dir "$(MLX_WEIGHTS_DIR)" \
		--manifest_json "$(WEIGHTS_NPZ_MANIFEST)" \
		--tag "$(FREEZE_TAG)"

full-rerun-noise-dry-run:
	$(PYTHON) experiments/tools/run_fuller_noise_sweeps.py \
		--legacy_execute \
		--dry_run \
		--bundle configs/fuller_final_unreserved_noise_20260427.yaml \
		--imagenet_val "$(IMAGENET_VAL)" \
		--weights_npz_manifest "$(WEIGHTS_NPZ_MANIFEST)" \
		--python "$(FULL_RERUN_PYTHON)" \
		--accuracy_backend mlx

full-rerun-noise-execute:
	caffeinate -dimsu $(PYTHON) experiments/tools/run_fuller_noise_sweeps.py \
		--legacy_execute \
		--execute \
		--bundle configs/fuller_final_unreserved_noise_20260427.yaml \
		--imagenet_val "$(IMAGENET_VAL)" \
		--weights_npz_manifest "$(WEIGHTS_NPZ_MANIFEST)" \
		--python "$(FULL_RERUN_PYTHON)" \
		--accuracy_backend mlx

status:
	$(PYTHON) -c 'import json; from pathlib import Path; payload = json.loads(Path("experiments/results/paper_sync/current_freeze.json").read_text()); [print(key + "=" + str(payload.get(key))) for key in ("freeze_tag", "quick_reports_dir", "paper_figures_dir", "review_dir")]'

clean:
	rm -rf build .pytest_cache
