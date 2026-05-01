PYTHON ?= python3
FREEZE_TAG := 20260430_full_figure_strict_remediated
QUICK_DIR := experiments/results/quick_reports/$(FREEZE_TAG)
PACK_DIR := figures/paper_figures_$(FREEZE_TAG)
REVIEW_DIR := experiments/results/review/$(FREEZE_TAG)
BUILD_FIG_DIR := build/rendered_figures
BUILD_REVIEW_DIR := build/rendered_review

.PHONY: repro-check render-paper-figures smoke clean

repro-check:
	$(PYTHON) scripts/check_public_repro_repo.py --root .

render-paper-figures:
	rm -rf $(BUILD_FIG_DIR) $(BUILD_REVIEW_DIR)
	$(PYTHON) experiments/tools/render_fuller_phase4_paper_data_figures.py \
		--quick_dir $(QUICK_DIR) \
		--mechanism_quick_dir $(QUICK_DIR) \
		--out_dir $(BUILD_FIG_DIR) \
		--review_dir $(BUILD_REVIEW_DIR)

smoke: repro-check

clean:
	rm -rf build .pytest_cache
