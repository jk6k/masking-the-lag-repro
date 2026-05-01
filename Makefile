PYTHON ?= python3
FREEZE_TAG := 20260430_full_figure_strict_remediated
QUICK_DIR := experiments/results/quick_reports/20260430_full_figure_strict_remediated
PACK_DIR := figures/paper_figures_20260430_full_figure_strict_remediated
REVIEW_DIR := experiments/results/review/20260430_full_figure_strict_remediated
BUILD_FIG_DIR := build/rendered_figures
BUILD_REVIEW_DIR := build/rendered_review

.PHONY: repro-check render-paper-figures test smoke status clean

repro-check:
	$(PYTHON) scripts/check_public_repro_repo.py --root .

render-paper-figures:
	rm -rf $(BUILD_FIG_DIR) $(BUILD_REVIEW_DIR)
	$(PYTHON) experiments/tools/render_fuller_phase4_paper_data_figures.py \
		--quick_dir $(QUICK_DIR) \
		--mechanism_quick_dir $(QUICK_DIR) \
		--out_dir $(BUILD_FIG_DIR) \
		--review_dir $(BUILD_REVIEW_DIR)

test:
	$(PYTHON) -m pytest -q experiments/tests/test_public_repro_surface.py

smoke: repro-check

status:
	$(PYTHON) -c 'import json; from pathlib import Path; payload = json.loads(Path("experiments/results/paper_sync/current_freeze.json").read_text()); [print(key + "=" + str(payload.get(key))) for key in ("freeze_tag", "quick_reports_dir", "paper_figures_dir", "review_dir")]'

clean:
	rm -rf build .pytest_cache
