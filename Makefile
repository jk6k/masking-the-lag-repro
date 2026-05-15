PYTHON ?= python3
FREEZE_TAG := 20260513_tetc_pivot
QUICK_DIR := experiments/results/quick_reports/20260513_tetc_pivot
PHASE_DIR := experiments/results/runs
REPORT_DATA_DIR := experiments/results/report_data
PACK_DIR := figures/suds_tetc_20260516_submission_figure_pack
REVIEW_DIR := experiments/results/review/20260513_tetc_pivot_public
BUILD_FIG_DIR := build/rendered_figures

.PHONY: repro-check render-paper-figures test smoke status clean

repro-check:
	$(PYTHON) scripts/check_public_repro_repo.py --root .

render-paper-figures:
	rm -rf $(BUILD_FIG_DIR)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) experiments/tools/render_suds_tetc_submission_data_figures.py \
		--summary-csv $(REPORT_DATA_DIR)/suds_transformer_architecture_sim_20260513_tetc_pivot_summary.csv \
		--conservative-json $(REPORT_DATA_DIR)/suds_tetc_conservative_pareto_20260513_tetc_pivot.json \
		--scheduler-ablation-csv $(REPORT_DATA_DIR)/suds_tetc_scheduler_ablation_20260513_tetc_pivot.csv \
		--output-dir $(BUILD_FIG_DIR)

test:
	$(PYTHON) -m pytest -q experiments/tests/test_public_repro_surface.py

smoke: repro-check

status:
	$(PYTHON) -c 'import json; from pathlib import Path; payload = json.loads(Path("experiments/results/paper_sync/current_freeze.json").read_text()); [print(key + "=" + str(payload.get(key))) for key in ("freeze_tag", "phase_dir", "report_data_dir", "paper_figures_dir", "review_dir")]'

clean:
	rm -rf build .pytest_cache
