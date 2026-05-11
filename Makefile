PYTHON ?= python3
FREEZE_TAG := 20260511_suds_maxq
QUICK_DIR := experiments/results/quick_reports/20260511_suds_maxq
PHASE_DIR := experiments/results/runs
REPORT_DATA_DIR := experiments/results/report_data
PACK_DIR := figures/paper_figures_20260511_suds_maxq
REVIEW_DIR := experiments/results/review/20260511_suds_maxq_public
BUILD_FIG_DIR := build/rendered_figures

.PHONY: repro-check render-paper-figures test smoke status clean

repro-check:
	$(PYTHON) scripts/check_public_repro_repo.py --root .

render-paper-figures:
	rm -rf $(BUILD_FIG_DIR)
	@for fig in fig1 fig2 fig3 fig4 fig5 appf1 appf2 appf3 appf4; do \
		$(PYTHON) experiments/tools/render_suds_figures.py \
			--phase-dir $(PHASE_DIR) \
			--output-dir $(BUILD_FIG_DIR) \
			--figure $$fig; \
	done
	@if [ -f "$(REPORT_DATA_DIR)/suds_ablation_matrix_20260511_maxq.csv" ]; then \
		$(PYTHON) experiments/tools/render_suds_maxq_fig6_ablation.py \
			--input-csv $(REPORT_DATA_DIR)/suds_ablation_matrix_20260511_maxq.csv \
			--output-dir $(BUILD_FIG_DIR); \
	fi

test:
	$(PYTHON) -m pytest -q experiments/tests/test_public_repro_surface.py

smoke: repro-check

status:
	$(PYTHON) -c 'import json; from pathlib import Path; payload = json.loads(Path("experiments/results/paper_sync/current_freeze.json").read_text()); [print(key + "=" + str(payload.get(key))) for key in ("freeze_tag", "phase_dir", "report_data_dir", "paper_figures_dir", "review_dir")]'

clean:
	rm -rf build .pytest_cache
