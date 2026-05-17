"""Render optimized publication figures (Fig7-Fig20 + App-F1..App-F8).

This script consumes quick-report CSVs for a single run tag and exports
both SVG and PDF figures with consistent style, naming, and traceability.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_QUICK_DIR = RESULTS / "quick_reports" / "final_paper_v2"
STYLE_PATH = Path(__file__).parent / "paper_style.mplstyle"

if STYLE_PATH.exists():
    plt.style.use(str(STYLE_PATH))
else:
    plt.style.use("seaborn-v0_8-paper")

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

# Morandi Aesthetic Palette per IEEE Chief Visual Chair constraints
PALETTE = [
    "#6B7F8C",  # 0: Muted Navy/Blue
    "#8B9DAA",  # 1: Slate Blue
    "#B7A99E",  # 2: Sand/Orange
    "#7C8E77",  # 3: Sage Green
    "#988BA5",  # 4: Muted Purple
    "#888582",  # 5: Slate Grey
    "#C96A6A",  # 6: Muted Red
    "#D9E1E6",  # 7: Light Slate
]

EXP_COLOR = {
    "E0": "#5A6470",  # Slate Baseline
    "E1": "#8B9DAA",  
    "E2": "#7C8E77",  # Morandi Green
    "E3": "#B7A99E",  
    "E4": "#C96A6A",  
    "E5": "#988BA5",  
    "E6": "#888582",  
}
MODULE_ORDER = ["conv", "attention", "ffn", "other"]
MODULE_COLOR = {
    "conv": "#6B7F8C",
    "attention": "#B7A99E",
    "ffn": "#7C8E77",
    "other": "#988BA5",
}
ACCELERATOR_DEVICE_NAME = "MTL-FULLER accelerator"
ACCELERATOR_SHORT_NAME = "MTL-FULLER"

MAIN_FIGS = {"Fig8", "Fig10", "Fig12", "Fig13", "Fig15", "Fig17", "Fig19", "Fig20"}
APPENDIX_FIGS = {
    "Fig7",
    "Fig9",
    "Fig14",
    "Fig16",
    "Fig18",
    "AppF1",
    "AppF2",
    "AppF3",
    "AppF4",
    "AppF5",
    "AppF6",
    "AppF7",
    "AppF8",
}

TRACE_ORDER = [
    "Fig7",
    "Fig8",
    "Fig9",
    "Fig10",
    "Fig11",
    "Fig12",
    "Fig13",
    "Fig14",
    "Fig15",
    "Fig16",
    "Fig17",
    "Fig18",
    "Fig19",
    "Fig20",
    "AppF1",
    "AppF2",
    "AppF3",
    "AppF4",
    "AppF5",
    "AppF6",
    "AppF7",
    "AppF8",
]

LEGACY_FIG_ID_ALIAS = {
    "AppF9": "Fig20",
}

FIGURE_TITLES = {
    "Fig7": "Related Work",
    "Fig8": "DET Accuracy vs BSL",
    "Fig9": "Prefix Error",
    "Fig10": "Energy Breakdown",
    "Fig11": "Timeline Proxy",
    "Fig12": "PHY N Sweep",
    "Fig13": "Accuracy Heatmap",
    "Fig14": "MESO Fanout",
    "Fig15": "Sparse Pareto",
    "Fig16": "Alignment",
    "Fig17": "Overall Pareto",
    "Fig18": "DET Waterfall",
    "Fig19": "Ablation",
    "Fig20": "Device Comparison",
    "AppF1": "Workload Generalization",
    "AppF2": "Module Breakdown",
    "AppF3": "Batch Scaling",
    "AppF4": "Sequence Scaling",
    "AppF5": "WDM MRR Quant",
    "AppF6": "Broadcast vs One-to-One",
    "AppF7": "Noise Robustness Surface",
    "AppF8": "HPAT vs Cross Platform",
}

FIGURE_STYLE_META: dict[str, dict[str, str]] = {
    "Fig8": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig8 | "
            "original_papers/markdown/06_robustness_and_noise_resilience/"
            "Noisy_Machines_Understanding_Noisy_Neural_Networks_and_Enhancing_Robustness_to_Analog_Hardware_Errors_Using_Distillation_arXiv2001.04974.md::Fig3"
        ),
        "scope": (
            "composition_only; family=single_parameter_sweep; "
            "borrow=legend_posture+marker_rhythm+whitespace; "
            "avoid=plot_internal_title+claim_framing"
        ),
    },
    "Fig10": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig7 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig11 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig12"
        ),
        "scope": (
            "composition_only; family=stacked_breakdown; "
            "borrow=component_order+compact_legend+grayscale_hierarchy; "
            "avoid=caption_semantics+claim_framing"
        ),
    },
    "Fig12": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig8 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig9"
        ),
        "scope": (
            "composition_only; family=dual_axis_scaling_sweep; "
            "borrow=single_column_geometry+clean_axis_hierarchy+legend_restraint; "
            "avoid=plot_internal_title+caption_like_text"
        ),
    },
    "Fig13": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig14 | "
            "original_papers/markdown/06_robustness_and_noise_resilience/"
            "Noisy_Machines_Understanding_Noisy_Neural_Networks_and_Enhancing_Robustness_to_Analog_Hardware_Errors_Using_Distillation_arXiv2001.04974.md::Fig2"
        ),
        "scope": (
            "composition_only; family=robustness_heatmap; "
            "borrow=boundary_emphasis+threshold_reading+compact_colorbar; "
            "avoid=tiny_annotations+crowded_axis_text"
        ),
    },
    "Fig15": {
        "anchors": (
            "original_papers/markdown/02_photonic_nn_accelerators/"
            "CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md::Fig6 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig13 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2402.03247_HEANA_High-throughput_Energy-efficient_Approximate_Nonlinear_Transformer_Accelerator.md::Fig15"
        ),
        "scope": (
            "composition_only; family=pareto_path; "
            "borrow=sparse_point_labeling+frontier_posture+single_column_spacing; "
            "avoid=label_every_point+plot_internal_title"
        ),
    },
    "Fig17": {
        "anchors": (
            "original_papers/markdown/02_photonic_nn_accelerators/"
            "CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md::Fig6 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig13 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig6"
        ),
        "scope": (
            "composition_only; family=overall_comparison_scatter; "
            "borrow=frontier_highlighting+marker_hierarchy+annotation_restraint; "
            "avoid=callout_stack+claim_framing"
        ),
    },
    "Fig19": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig6 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2402.03247_HEANA_High-throughput_Energy-efficient_Approximate_Nonlinear_Transformer_Accelerator.md::Fig15 | "
            "original_papers/markdown/02_photonic_nn_accelerators/"
            "CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md::Fig7"
        ),
        "scope": (
            "composition_only; family=ablation_comparison; "
            "borrow=variant_hierarchy+compact_legend+grayscale_safe_dual_encoding; "
            "avoid=variant_omission+ambiguous_metric_sourcing"
        ),
    },
    "Fig9": {
        "anchors": (
            "original_papers/markdown/06_robustness_and_noise_resilience/"
            "Noisy_Machines_Understanding_Noisy_Neural_Networks_and_Enhancing_Robustness_to_Analog_Hardware_Errors_Using_Distillation_arXiv2001.04974.md::Fig2 | "
            "original_papers/markdown/06_robustness_and_noise_resilience/"
            "Noisy_Machines_Understanding_Noisy_Neural_Networks_and_Enhancing_Robustness_to_Analog_Hardware_Errors_Using_Distillation_arXiv2001.04974.md::Fig3"
        ),
        "scope": (
            "composition_only; family=error_sweep; "
            "borrow=caption_first_posture+log_axis_clarity+restrained_legend; "
            "avoid=plot_internal_title"
        ),
    },
    "Fig11": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig8 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig9"
        ),
        "scope": (
            "composition_only; family=paired_proxy_comparison; "
            "borrow=single_column_geometry+paired_metric_restraint; "
            "avoid=plot_internal_title"
        ),
    },
    "Fig14": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig8 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig13"
        ),
        "scope": (
            "composition_only; family=break_even_sweep; "
            "borrow=legend_restraint+clean_axis_hierarchy; "
            "avoid=plot_internal_title"
        ),
    },
    "Fig16": {
        "anchors": (
            "original_papers/markdown/06_robustness_and_noise_resilience/"
            "Noisy_Machines_Understanding_Noisy_Neural_Networks_and_Enhancing_Robustness_to_Analog_Hardware_Errors_Using_Distillation_arXiv2001.04974.md::Fig2 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig14"
        ),
        "scope": (
            "composition_only; family=alignment_sweep; "
            "borrow=noise_axis_clarity+caption_first_posture; "
            "avoid=plot_internal_title"
        ),
    },
    "Fig18": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig12 | "
            "original_papers/markdown/02_photonic_nn_accelerators/"
            "CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md::Fig7"
        ),
        "scope": (
            "composition_only; family=waterfall_or_component_delta; "
            "borrow=compact_category_labels+single_column_balance; "
            "avoid=plot_internal_title"
        ),
    },
    "AppF2": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig7 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig11"
        ),
        "scope": (
            "composition_only; family=stacked_breakdown; "
            "borrow=component_hierarchy+compact_top_legend; "
            "avoid=plot_internal_title"
        ),
    },
    "AppF3": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig8 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig9"
        ),
        "scope": (
            "composition_only; family=batch_scaling; "
            "borrow=single_column_scaling_layout+restrained_legend; "
            "avoid=plot_internal_title"
        ),
    },
    "AppF4": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig8 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig9"
        ),
        "scope": (
            "composition_only; family=sequence_scaling; "
            "borrow=paired_axis_scaling_posture+clean_dual_axis_labels; "
            "avoid=plot_internal_title"
        ),
    },
    "AppF5": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig8 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig9"
        ),
        "scope": (
            "composition_only; family=support_wdm_sweep; "
            "borrow=single_axis_restraint+support_note_posture; "
            "avoid=nonexistent_multi_axis_claims+large_suptitle"
        ),
    },
    "AppF7": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig14 | "
            "original_papers/markdown/06_robustness_and_noise_resilience/"
            "Noisy_Machines_Understanding_Noisy_Neural_Networks_and_Enhancing_Robustness_to_Analog_Hardware_Errors_Using_Distillation_arXiv2001.04974.md::Fig2"
        ),
        "scope": (
            "composition_only; family=robustness_heatmap; "
            "borrow=boundary_emphasis+compact_colorbar; "
            "avoid=plot_internal_title"
        ),
    },
    "Fig7": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig6 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig13 | "
            "original_papers/markdown/02_photonic_nn_accelerators/"
            "CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md::Fig6"
        ),
        "scope": (
            "composition_only; family=qualitative_rubric_matrix; "
            "anchor_mode=nearest_comparison_family; "
            "borrow=matrix_restraint+bounded_annotation_density+benchmark_caveat_posture; "
            "avoid=benchmark_claim_framing+self_emphasis_fill"
        ),
    },
    "AppF1": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig6 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig13 | "
            "original_papers/markdown/02_photonic_nn_accelerators/"
            "CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md::Fig6"
        ),
        "scope": (
            "composition_only; family=internal_model_scale_context; "
            "borrow=single_workload_tradeoff_context+light_connectors+footer_restraint; "
            "avoid=workload_generalization_framing+dummy_series_legend"
        ),
    },
    "AppF6": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig8 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig11 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig12"
        ),
        "scope": (
            "composition_only; family=paired_sweep_comparison; "
            "borrow=shared_legend+panel_balance+caption_first_posture; "
            "avoid=redundant_subplot_titles+large_suptitle"
        ),
    },
    "AppF8": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig6 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig13 | "
            "original_papers/markdown/02_photonic_nn_accelerators/"
            "CrossLight_A_Cross-Layer_Optimized_Silicon_Photonic_Neural_Network_Accelerator_arXiv2102.06960v1.md::Fig6"
        ),
        "scope": (
            "composition_only; family=tier_separated_platform_context; "
            "borrow=panel_separation+marker_restraint+comparison_whitespace; "
            "avoid=pooled_benchmark_axis+unexplained_encoding"
        ),
    },
    "Fig20": {
        "anchors": (
            "original_papers/markdown/01_transformer_attention_photonic/"
            "2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md::Fig7 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig11 | "
            "original_papers/markdown/01_transformer_attention_photonic/"
            "Lightening_Transformer_HPCA2024.md::Fig12"
        ),
        "scope": (
            "composition_only; family=grouped_device_comparison; "
            "borrow=direct_bar_comparison+top_legend+metric_parallelism; "
            "avoid=tier_split_layout+generic_device_labels+pooled_benchmark_bar_family"
        ),
    },
}

WARN_MISSING_CSV = False
MISSING_CSVS: set[str] = set()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        MISSING_CSVS.add(str(path))
        if WARN_MISSING_CSV:
            print(f"[render] missing csv: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[render] failed to read {path}: {exc}")
        return pd.DataFrame()


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _fig_tier(fig_id: str) -> str:
    if fig_id in MAIN_FIGS:
        return "main"
    return "appendix"


def _format_log_tick_plain(value: float, _pos: float) -> str:
    if value <= 0:
        return ""
    exponent = int(round(math.log10(value)))
    if not math.isclose(value, 10**exponent, rel_tol=1e-9, abs_tol=1e-12):
        return ""
    return f"1e{exponent}"


def _clean_platform_context_rows(
    df: pd.DataFrame,
    *,
    note_parts: list[str],
    gate_name: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    start_rows = len(df)
    df = df.drop_duplicates().copy()
    removed_exact = start_rows - len(df)
    if removed_exact:
        note_parts.append(f"{gate_name} removed_exact_duplicates={removed_exact}")

    src_series = df.get("source_type", pd.Series("", index=df.index)).astype(str).str.lower()
    platform_series = df.get("platform_class", pd.Series("", index=df.index)).astype(str).str.upper()
    latency_series = _to_num(df.get("latency_ms", pd.Series(np.nan, index=df.index)))
    energy_series = _to_num(df.get("energy_j", pd.Series(np.nan, index=df.index)))
    power_series = _to_num(df.get("avg_power_w", pd.Series(np.nan, index=df.index)))

    incomplete_mask = (
        src_series.eq("measured")
        & latency_series.notna()
        & energy_series.notna()
        & power_series.notna()
        & (energy_series <= 0.0)
        & (power_series <= 0.0)
    )

    removable_idx: list[int] = []
    for idx, row in df[incomplete_mask].iterrows():
        model = str(row.get("model", ""))
        platform = str(row.get("platform_class", "")).upper()
        src = str(row.get("source_type", "")).lower()
        latency = float(row.get("latency_ms", float("nan")))
        if not math.isfinite(latency):
            continue

        lat_close = pd.Series(
            np.isclose(latency_series.to_numpy(dtype=float), latency, rtol=1e-9, atol=1e-9),
            index=df.index,
        )
        candidate_mask = (
            (df.index != idx)
            & df.get("model", pd.Series("", index=df.index)).astype(str).eq(model)
            & platform_series.eq(platform)
            & src_series.eq(src)
            & latency_series.notna()
            & lat_close
            & (energy_series > 0.0)
            & (power_series > 0.0)
        )
        if candidate_mask.any():
            removable_idx.append(idx)

    if removable_idx:
        df = df.drop(index=removable_idx).copy()
        note_parts.append(f"{gate_name} removed_incomplete_measured_rows={len(removable_idx)}")

    unresolved_mask = (
        src_series.loc[df.index].eq("measured")
        & latency_series.loc[df.index].notna()
        & energy_series.loc[df.index].notna()
        & power_series.loc[df.index].notna()
        & ((energy_series.loc[df.index] <= 0.0) | (power_series.loc[df.index] <= 0.0))
    )
    unresolved = int(unresolved_mask.sum())
    if unresolved:
        note_parts.append(f"{gate_name} unresolved_nonpositive_rows={unresolved}")
        print(f"[render][warn] {gate_name} unresolved_nonpositive_rows={unresolved}")

    return df


def _device_display_name(row: pd.Series) -> str:
    for key in ["device_display_name", "device_model", "device_name"]:
        val = str(row.get(key, "") or "").strip()
        if val:
            return val
    platform = str(row.get("platform_class", "")).upper()
    if platform == "CPU":
        return "CPU"
    if platform == "GPU":
        return "GPU (MPS)"
    if platform == "HPAT":
        return ACCELERATOR_DEVICE_NAME
    return platform or "unknown"


def _platform_display_name(row: pd.Series, fallback: str) -> str:
    value = str(row.get("platform_display_name", "") or "").strip()
    return value or fallback


def _host_and_device_note(df: pd.DataFrame) -> str:
    measured = df[df.get("source_type", pd.Series("", index=df.index)).astype(str).str.lower() == "measured"].copy()
    if measured.empty:
        return ""
    host = ""
    if "host_name" in measured.columns:
        hosts = measured["host_name"].dropna().astype(str)
        hosts = hosts[hosts.str.strip().ne("")]
        if not hosts.empty:
            host = str(hosts.mode().iloc[0])
    if host.endswith(".local"):
        host = host[: -len(".local")]
    cpu_name = ""
    gpu_name = ""
    cpu_rows = measured[measured["platform_class"].astype(str).str.upper() == "CPU"]
    gpu_rows = measured[measured["platform_class"].astype(str).str.upper() == "GPU"]
    if not cpu_rows.empty:
        cpu_name = _device_display_name(cpu_rows.iloc[0])
    if not gpu_rows.empty:
        gpu_name = _device_display_name(gpu_rows.iloc[0])
    parts = []
    if host:
        parts.append(f"Host: {host}")
    if cpu_name:
        parts.append(f"CPU: {cpu_name}")
    if gpu_name:
        parts.append(f"GPU: {gpu_name}")
    return " | ".join(parts)


def _host_and_device_note_lines(note: str) -> list[str]:
    parts = [part.strip() for part in note.split("|") if part.strip()]
    if not parts:
        return []
    if len(parts) <= 2:
        return [" | ".join(parts)]
    if len(parts) == 3:
        return parts
    return [parts[0], parts[1], " | ".join(parts[2:])]


def _host_and_device_footer_line(df: pd.DataFrame) -> str:
    measured = df[df.get("source_type", pd.Series("", index=df.index)).astype(str).str.lower() == "measured"].copy()
    if measured.empty:
        return ""
    cpu_name = ""
    gpu_name = ""
    cpu_rows = measured[measured["platform_class"].astype(str).str.upper() == "CPU"]
    gpu_rows = measured[measured["platform_class"].astype(str).str.upper() == "GPU"]
    if not cpu_rows.empty:
        cpu_name = _device_display_name(cpu_rows.iloc[0])
    if not gpu_rows.empty:
        gpu_name = _device_display_name(gpu_rows.iloc[0])
    if cpu_name and gpu_name:
        return f"Measured on {cpu_name} and {gpu_name}."
    if cpu_name:
        return f"Measured on {cpu_name}."
    if gpu_name:
        return f"Measured on {gpu_name}."
    return ""


def _run_tag_from_out_dir(out_dir: Path) -> str:
    name = out_dir.name
    prefix = "paper_figures_"
    return name[len(prefix):] if name.startswith(prefix) else name


def _review_dir(run_tag: str) -> Path:
    return ROOT / "results" / "review" / run_tag


def _repo_rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.parent.resolve()))
    except Exception:
        return str(path)


def _write_grayscale(png_path: Path, fig_name: str, review_dir: Path) -> None:
    if not HAS_PIL:
        return
    review_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(png_path).convert("L")
    img.save(review_dir / f"{fig_name}_grayscale.png")


def _heatmap(
    data: pd.DataFrame,
    *,
    ax: plt.Axes,
    annot: bool = False,
    fmt: str = ".0f",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    linewidths: float = 0.0,
    linecolor: str = "white",
    cbar: bool = True,
    cbar_kws: dict[str, Any] | None = None,
):
    if HAS_SEABORN:
        return sns.heatmap(
            data,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=linewidths,
            linecolor=linecolor,
            cbar=cbar,
            cbar_kws=cbar_kws,
            ax=ax,
        )

    arr = data.to_numpy(dtype=float)
    mesh = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(list(data.columns))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(list(data.index))

    if linewidths > 0:
        for x in range(data.shape[1] + 1):
            ax.axvline(x - 0.5, color=linecolor, linewidth=linewidths)
        for y in range(data.shape[0] + 1):
            ax.axhline(y - 0.5, color=linecolor, linewidth=linewidths)

    if annot:
        for yi in range(data.shape[0]):
            for xi in range(data.shape[1]):
                val = float(arr[yi, xi])
                ax.text(xi, yi, format(val, fmt), ha="center", va="center")

    cbar_obj = None
    if cbar:
        cbar_kws = dict(cbar_kws or {})
        label = cbar_kws.pop("label", None)
        cbar_obj = ax.figure.colorbar(mesh, ax=ax, **cbar_kws)
        if label:
            cbar_obj.set_label(label)
    mesh.colorbar = cbar_obj
    return SimpleNamespace(collections=[mesh])


def _save_fig(
    fig: plt.Figure,
    fig_name: str,
    out_dir: Path,
    *,
    use_tight_layout: bool = True,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{fig_name}.svg"
    pdf_path = out_dir / f"{fig_name}.pdf"
    png_path = out_dir / f"{fig_name}.png"
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"[render] wrote {svg_path.name} / {pdf_path.name} / {png_path.name}")

    _write_grayscale(png_path, fig_name, _review_dir(_run_tag_from_out_dir(out_dir)))

    return svg_path, pdf_path, png_path


def _record_trace(
    trace_rows: list[dict[str, Any]],
    *,
    fig_id: str,
    figure_file: Path,
    input_csvs: list[Path],
    run_tag: str,
    command: str,
    params_summary: str = "",
    literature_style_anchors: str = "",
    literature_anchor_scope: str = "",
    notes: str = "",
) -> None:
    style_meta = FIGURE_STYLE_META.get(fig_id, {})
    if not literature_style_anchors:
        literature_style_anchors = style_meta.get("anchors", "")
    if not literature_anchor_scope:
        literature_anchor_scope = style_meta.get("scope", "")
    trace_rows.append(
        {
            "figure_id": fig_id,
            "manuscript_tier": _fig_tier(fig_id),
            "figure_file": str(figure_file),
            "input_csvs": ";".join(str(p) for p in input_csvs),
            "script_entry": "experiments/tools/render_paper_figures.py",
            "command": command,
            "run_tag": run_tag,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "params_summary": params_summary,
            "literature_style_anchors": literature_style_anchors,
            "literature_anchor_scope": literature_anchor_scope,
            "notes": notes,
        }
    )


def _write_manifests(
    trace_rows: list[dict[str, Any]],
    out_dir: Path,
    run_tag: str,
    selected_fig_ids: set[str] | None = None,
) -> list[Path]:
    written: list[Path] = []
    review_dir = _review_dir(run_tag)
    review_trace = review_dir / "figure_traceability.csv"
    for row in trace_rows:
        fig_id = str(row["figure_id"])
        if selected_fig_ids and fig_id not in selected_fig_ids:
            continue
        fig_svg = Path(str(row["figure_file"]))
        fig_stem = fig_svg.stem
        manifest_path = out_dir / f"{fig_stem}_manifest.json"
        title = FIGURE_TITLES.get(fig_id, fig_stem.replace("_", " "))
        notes = [str(row["notes"])] if str(row.get("notes", "")).strip() else []
        sources = [
            item
            for item in str(row.get("input_csvs", "")).split(";")
            if item.strip()
        ]
        payload = {
            "figure_id": fig_id,
            "title": title,
            "status": "frozen_pack_rendered",
            "run_tag": row["run_tag"],
            "generated_at": row["generated_at"],
            "frozen_outputs": {
                "svg": _repo_rel(out_dir / f"{fig_stem}.svg"),
                "pdf": _repo_rel(out_dir / f"{fig_stem}.pdf"),
                "png": _repo_rel(out_dir / f"{fig_stem}.png"),
            },
            "review_artifacts": {
                "grayscale_png": _repo_rel(review_dir / f"{fig_stem}_grayscale.png"),
                "traceability_csv": _repo_rel(review_trace),
            },
            "workflow": "deterministic matplotlib redraw from frozen quick-report csvs",
            "script_entry": row["script_entry"],
            "sources": sources,
            "params_summary": row.get("params_summary", ""),
            "literature_style_anchors": [
                item.strip()
                for item in str(row.get("literature_style_anchors", "")).split("|")
                if item.strip()
            ],
            "literature_anchor_scope": row.get("literature_anchor_scope", ""),
            "notes": notes,
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(manifest_path)
    return written


def _trace_sort_key(fig_id: str) -> int:
    try:
        return TRACE_ORDER.index(fig_id)
    except ValueError:
        return len(TRACE_ORDER)


def _resolve_radar_csv(quick_dir: Path, radar_csv: Path | None) -> tuple[Path | None, str]:
    if radar_csv is not None:
        return (radar_csv if radar_csv.exists() else None), "explicit"
    local = quick_dir / "fig_a_related_work_radar_scores.csv"
    if local.exists():
        return local, "quick_dir"
    fallback = [
        RESULTS / "quick_reports" / "20260222_cuda_v31" / "fig_a_related_work_radar_scores.csv",
        RESULTS / "quick_reports" / "rerun_noncuda_20260222" / "fig_a_related_work_radar_scores.csv",
    ]
    for path in fallback:
        if path.exists():
            return path, "fallback"
    return None, "missing"


def plot_fig7_radar(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
    radar_csv: Path | None,
) -> None:
    source_path, source_mode = _resolve_radar_csv(quick_dir, radar_csv)
    if source_path is None:
        print("[render] skip Fig7: radar source csv not found")
        return
    df = _load_csv(source_path)
    if df.empty:
        print("[render] skip Fig7: empty radar csv")
        return

    value_cols = [c for c in df.columns if c.lower() not in {"work", "method"}]
    if not value_cols:
        print("[render] skip Fig7: no score columns")
        return

    score = df.copy()
    work_col = "Work" if "Work" in score.columns else "work"
    score[value_cols] = score[value_cols].apply(_to_num)
    score = score.dropna(subset=[work_col])
    score = score.set_index(work_col)[value_cols].astype(float)
    score = score.clip(lower=1.0, upper=5.0)
    labels = [c.replace("_", "\n") for c in value_cols]

    fig, ax = plt.subplots(figsize=(3.62, 2.92))
    hm = _heatmap(
        score,
        annot=True,
        fmt=".0f",
        cmap="Greys",
        vmin=1.0,
        vmax=5.0,
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        cbar_kws={"label": "Qualitative Rubric Score", "ticks": [1, 2, 3, 4, 5], "pad": 0.02, "fraction": 0.08},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(labels, rotation=0, fontsize=7.1)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7.2)
    cbar = hm.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=7.2, pad=1)
        cbar.set_label("Qualitative Rubric Score", fontsize=7.4, labelpad=4)

    if "Masking-the-Lag" in score.index:
        row_idx = score.index.tolist().index("Masking-the-Lag")
        for xi in range(len(value_cols)):
            ax.add_patch(plt.Rectangle((xi, row_idx), 1, 1, fill=False, edgecolor="#d62728", linewidth=1.3))

    fig.text(
        0.5,
        0.015,
        "Qualitative rubric matrix only; bounded design-position aid, not an apples-to-apples benchmark.",
        ha="center",
        fontsize=7.2,
        color="#666666",
    )
    svg_path, _, _ = _save_fig(fig, "Fig7_RelatedWork", out_dir)

    note_parts: list[str] = []
    if source_mode == "fallback":
        note_parts.append("radar csv from static fallback (non-run-dependent)")
    basis_note = quick_dir / "audits" / "fig7_related_work_scoring_basis_20260328.md"
    if basis_note.exists():
        note_parts.append(f"qualitative scoring basis: {basis_note}")
    _record_trace(
        trace_rows,
        fig_id="Fig7",
        figure_file=svg_path,
        input_csvs=[source_path],
        run_tag=run_tag,
        command=cmd,
        params_summary=f"qualitative_rubric_matrix=1to5; axes=6; score_source={source_mode}",
        notes="; ".join(note_parts),
    )


def plot_fig8_det_acc(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
    det_summary_csv: Path | None = None,
) -> None:
    if det_summary_csv is None:
        auto_summary = quick_dir / "fig8_det_k_summary.csv"
        if auto_summary.exists():
            det_summary_csv = auto_summary
    if det_summary_csv is not None and det_summary_csv.exists():
        df = _load_csv(det_summary_csv)
        required = {
            "det_k_global",
            "paired_model_mean_delta_vs_e0_quant_pp",
            "speedup_vs_E0",
        }
        if required.issubset(df.columns):
            df["det_k_global"] = _to_num(df["det_k_global"])
            df["paired_model_mean_delta_vs_e0_quant_pp"] = _to_num(df["paired_model_mean_delta_vs_e0_quant_pp"])
            df["speedup_vs_E0"] = _to_num(df["speedup_vs_E0"])
            if "prefix_error_mean" in df.columns:
                df["prefix_error_mean"] = _to_num(df["prefix_error_mean"])
            
            has_error_bars = False
            if "paired_model_std_delta_vs_e0_quant_pp" in df.columns:
                df["paired_model_std_delta_vs_e0_quant_pp"] = _to_num(df["paired_model_std_delta_vs_e0_quant_pp"])
                has_error_bars = True
                
            df = df.dropna(subset=["det_k_global", "paired_model_mean_delta_vs_e0_quant_pp", "speedup_vs_E0"])
            df = df.sort_values("det_k_global")
            if not df.empty:
                fig, ax = plt.subplots(figsize=(3.60, 2.70))
                
                if has_error_bars:
                    std = df["paired_model_std_delta_vs_e0_quant_pp"].fillna(0)
                    ax.fill_between(
                        df["det_k_global"],
                        df["paired_model_mean_delta_vs_e0_quant_pp"] - std,
                        df["paired_model_mean_delta_vs_e0_quant_pp"] + std,
                        color=PALETTE[0],
                        alpha=0.2,
                        zorder=2,
                        label="± 1 std dev"
                    )

                ax.plot(
                    df["det_k_global"],
                    df["paired_model_mean_delta_vs_e0_quant_pp"],
                    marker="o",
                    markersize=5.0,
                    markerfacecolor="white",
                    markeredgecolor=PALETTE[0],
                    markeredgewidth=1.2,
                    color=PALETTE[0],
                    linewidth=1.5,
                    zorder=3,
                    label="delta vs E0 Baseline",
                )
                ax.axhline(0.0, color="#444444", linewidth=1.2, linestyle="--", zorder=1, label="E0 Baseline (0.0)")
                ax.set_xlabel("DET truncation k")
                ax.set_ylabel("Top-1 Delta vs E0 (pp)")
                ax.margins(x=0.05, y=0.35)

                ax2 = ax.twinx()
                ax2.plot(
                    df["det_k_global"],
                    df["speedup_vs_E0"],
                    marker="s",
                    markersize=4.2,
                    markerfacecolor=PALETTE[2],
                    markeredgecolor=PALETTE[2],
                    color=PALETTE[2],
                    linewidth=1.3,
                    linestyle="--",
                    zorder=2,
                    label="speedup (x)",
                )
                ax2.set_ylabel("Speedup vs E0", color=PALETTE[2])
                ax2.tick_params(axis="y", labelcolor=PALETTE[2])
                ax2.margins(x=0.05, y=0.35)

                k64 = df[df["det_k_global"] == 64]
                if not k64.empty:
                    row = k64.iloc[0]
                    ax.scatter(
                        [row["det_k_global"]],
                        [row["paired_model_mean_delta_vs_e0_quant_pp"]],
                        marker="*",
                        s=55,
                        color=PALETTE[6],
                        zorder=4,
                        label="Promoted point (k=64)"
                    )

                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(handles1 + handles2, labels1 + labels2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.2)
                fig.text(
                    0.5,
                    -0.05,
                    "Displayed DET-k points are measured under the active MLX support sweep; "
                    "k=64 is the promoted operating point and k=129 is the full-BSL reference.",
                    ha="center",
                    fontsize=7.0,
                    color="#666666",
                )
                svg_path, _, _ = _save_fig(fig, "Fig8_DET_AccVsBSL", out_dir)

                _record_trace(
                    trace_rows,
                    fig_id="Fig8",
                    figure_file=svg_path,
                    input_csvs=[det_summary_csv],
                    run_tag=run_tag,
                    command=cmd,
                    params_summary="mode=mlx_final_det_k_summary; delta_ref=E0_quantized; speedup_ref=E0",
                    notes="Displayed DET-k points are measured under the active MLX support sweep.",
                )
                return

    path = quick_dir / "quickscan_e3_k_sweep.csv"
    df = _load_csv(path)
    if df.empty:
        return

    meas_col = _pick_col(df, ["measured_acc_drop_pp_mean", "acc_drop_pp_mean"])
    if meas_col is None:
        return
    proj_col = _pick_col(df, ["modeled_acc_drop_pp_mean"])
    df["avg_effective_bsl"] = _to_num(df["avg_effective_bsl"])
    df[meas_col] = _to_num(df[meas_col])
    if proj_col is not None:
        df[proj_col] = _to_num(df[proj_col])
    keep_cols = ["avg_effective_bsl", meas_col] + ([proj_col] if proj_col else [])
    df = df.dropna(subset=keep_cols).sort_values("avg_effective_bsl")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(3.60, 2.70))
    
    if "measured_acc_drop_pp_std" in df.columns:
        std_m = _to_num(df["measured_acc_drop_pp_std"]).fillna(0)
        ax.fill_between(
            df["avg_effective_bsl"],
            df[meas_col] - std_m,
            df[meas_col] + std_m,
            color=PALETTE[0], alpha=0.2, zorder=2, label="± measured std"
        )
        
    ax.plot(
        df["avg_effective_bsl"],
        df[meas_col],
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=PALETTE[0],
        markeredgewidth=1.3,
        color=PALETTE[0],
        linewidth=1.4,
        zorder=3,
        label="measured",
    )
    has_projected = False
    if proj_col is not None:
        delta = (df[proj_col] - df[meas_col]).abs().max()
        if pd.notna(delta) and float(delta) > 1e-6:
            has_projected = True
            if "modeled_acc_drop_pp_std" in df.columns:
                std_p = _to_num(df["modeled_acc_drop_pp_std"]).fillna(0)
                ax.fill_between(
                    df["avg_effective_bsl"],
                    df[proj_col] - std_p,
                    df[proj_col] + std_p,
                    color=PALETTE[3], alpha=0.2, zorder=2, label="± projected std"
                )
            ax.plot(
                df["avg_effective_bsl"],
                df[proj_col],
                marker="^",
                markersize=4.6,
                markerfacecolor=PALETTE[3],
                markeredgecolor=PALETTE[3],
                color=PALETTE[3],
                linewidth=1.2,
                linestyle="--",
                zorder=4,
                label="projected",
            )
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.3, label="1.0 pp budget Baseline", zorder=2)
    ax.set_xlabel("Effective Bit-Serial Length (BSL)")
    ax.set_ylabel("Accuracy Drop (pp)")
    ax.margins(x=0.05, y=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=7.2, handlelength=1.7)
    svg_path, _, _ = _save_fig(fig, "Fig8_DET_AccVsBSL", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig8",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary=f"measured_col={meas_col}; projected_col={proj_col or 'none'}; projected_curve={has_projected}",
    )


def plot_fig9_prefix_error(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "fig_d_prefix_error_vs_k.csv"
    df = _load_csv(path)
    if df.empty:
        return

    df["k"] = _to_num(df["k"])
    df["prefix_error_mean"] = _to_num(df["prefix_error_mean"])
    df["prefix_error_p95"] = _to_num(df["prefix_error_p95"])
    df = df.dropna(subset=["k", "prefix_error_mean", "prefix_error_p95"]).sort_values("k")
    if df.empty:
        return

    eps = 1e-6
    m = df["prefix_error_mean"].clip(lower=eps)
    p95 = df["prefix_error_p95"].clip(lower=eps)

    fig, ax = plt.subplots(figsize=(3.60, 2.70))
    
    if "prefix_error_std" in df.columns:
        std = _to_num(df["prefix_error_std"]).fillna(0)
        ax.fill_between(df["k"], (m - std).clip(lower=eps), (m + std), color=PALETTE[0], alpha=0.2, zorder=1, label="± std")
        
    ax.plot(df["k"], m, marker="o", linestyle="-", color=PALETTE[0], label="mean", linewidth=1.4, markersize=4.6, markerfacecolor="white", markeredgewidth=1.0)
    ax.plot(df["k"], p95, marker="s", linestyle="--", color=PALETTE[3], label="p95", linewidth=1.3, markersize=4.2, markerfacecolor="white", markeredgewidth=1.0)
    
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_format_log_tick_plain))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("Prefix Length (k)")
    ax.set_ylabel("Prefix Error Probability")
    ax.margins(x=0.05, y=0.35)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7.4, loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2)
    svg_path, _, _ = _save_fig(fig, "Fig9_PrefixError", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig9",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="x=k; y=prefix_error_mean|prefix_error_p95; logy",
    )


def plot_fig10_energy_breakdown(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "energy_breakdown_summary.csv"
    df = _load_csv(path)
    if df.empty:
        return

    exp_order = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    df = df[df["experiment_id"].isin(exp_order)].copy()
    if df.empty:
        return
    df["experiment_id"] = pd.Categorical(df["experiment_id"], categories=exp_order, ordered=True)
    df = df.sort_values("experiment_id")

    raw_cols = [
        "energy_breakdown_conversion_control_j",
        "energy_breakdown_memory_move_j",
        "energy_breakdown_oe_j",
        "energy_breakdown_adc_pca_j",
        "energy_breakdown_laser_optical_j",
        "energy_breakdown_other_static_j",
    ]
    for c in raw_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = _to_num(df[c]).fillna(0.0)

    grouped = {
        "memory/move": df["energy_breakdown_memory_move_j"].to_numpy(dtype=float) * 1e3,
        "conv/control": df["energy_breakdown_conversion_control_j"].to_numpy(dtype=float) * 1e3,
        "I/O+static": (
            df["energy_breakdown_oe_j"]
            + df["energy_breakdown_adc_pca_j"]
            + df["energy_breakdown_laser_optical_j"]
            + df["energy_breakdown_other_static_j"]
        ).to_numpy(dtype=float)
        * 1e3,
    }
    fig, ax = plt.subplots(figsize=(3.62, 2.52))
    colors = {
        "memory/move": "#6b6b6b",
        "conv/control": "#b0b0b0",
        "I/O+static": "#e2e2e2",
    }
    hatches = {
        "memory/move": "",
        "conv/control": "///",
        "I/O+static": "..",
    }

    bottom = np.zeros(len(df), dtype=float)
    x = np.arange(len(df))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.grid(axis="x", visible=False)
    for label in ["memory/move", "conv/control", "I/O+static"]:
        vals = grouped[label]
        bars = ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.68,
            color=colors[label],
            edgecolor="#333333",
            linewidth=0.5,
            label=label,
        )
        for bar in bars:
            bar.set_hatch(hatches[label])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df["experiment_id"].astype(str))
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Energy per Inference (mJ)")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.margins(x=0.05, y=0.35)
    ax.set_ylim(0.0, float(bottom.max()) * 1.14)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=3,
        frameon=False,
        fontsize=7.4,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.25,
        columnspacing=0.8,
        borderpad=0.2,
    )
    svg_path, _, _ = _save_fig(fig, "Fig10_EnergyBreakdown", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig10",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="units=mJ; grouped=memory_move|conv_control|io_static; grayscale_stack+hatch",
    )


def plot_fig11_timeline(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "quickpack_e0_e6_overview.csv"
    df = _load_csv(path)
    if df.empty:
        return

    df = df[df["experiment_id"].isin(["E0", "E2"])].copy()
    if df.empty:
        return
    df["bubble_cycles"] = _to_num(df["bubble_cycles"])
    df["utilization_avg"] = _to_num(df["utilization_avg"])
    df = df.dropna(subset=["bubble_cycles", "utilization_avg"])
    if df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(3.60, 2.48))
    x = np.arange(len(df))
    w = 0.36
    b1 = ax1.bar(x - w / 2, df["bubble_cycles"], width=w, color=PALETTE[3], label="bubble cycles")
    ax1.set_ylabel("Bubble Cycles", color=PALETTE[3])
    ax1.tick_params(axis="y", labelcolor=PALETTE[3])
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["experiment_id"])
    ax1.set_xlabel("Experiment")

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w / 2, df["utilization_avg"], width=w, color=PALETTE[2], label="utilization")
    ax2.set_ylabel("Utilization (fraction)", color=PALETTE[2])
    ax2.tick_params(axis="y", labelcolor=PALETTE[2])
    ax2.set_ylim(0.0, max(1.0, float(df["utilization_avg"].max()) * 1.15))

    ax1.legend([b1, b2], ["bubble cycles", "utilization"], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=7.4)
    svg_path, _, _ = _save_fig(fig, "Fig11_TimelineProxy", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig11",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="x=experiment_id(E0|E2); y=bubble_cycles|utilization_avg",
    )


def plot_fig12_phy_sweep(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "quickscan_e5_phy_n_sweep.csv"
    df = _load_csv(path)
    if df.empty:
        return

    df["N_wdm"] = _to_num(df["N_wdm"])
    df["P_laser_dbm"] = _to_num(df["P_laser_dbm"])
    if "P_laser_mw" in df.columns:
        df["P_laser_mw"] = _to_num(df["P_laser_mw"])
    df["PP_crosstalk_db"] = _to_num(df["PP_crosstalk_db"])
    df = df.dropna(subset=["N_wdm", "P_laser_dbm", "PP_crosstalk_db"]).sort_values("N_wdm")
    if df.empty:
        return

    laser_col = "P_laser_mw" if "P_laser_mw" in df.columns and df["P_laser_mw"].notna().any() else "P_laser_dbm"
    laser_label = "Laser Power (mW)" if laser_col == "P_laser_mw" else "Laser Power (dBm)"
    fig, ax1 = plt.subplots(figsize=(3.60, 2.48))
    l1 = ax1.plot(
        df["N_wdm"],
        df[laser_col],
        marker="o",
        color=PALETTE[0],
        linewidth=1.4,
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="Laser power",
    )[0]
    ax1.set_xlabel("WDM Channels ($N_{wdm}$)")
    ax1.set_ylabel(laser_label, color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    ax1.grid(axis="y", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)
    ax1.grid(axis="x", visible=False)
    if (df["N_wdm"] == 16).any():
        ax1.axvline(16, color="#666666", linestyle=":", linewidth=0.9)
        row16 = df[df["N_wdm"] == 16].iloc[0]
        ax1.annotate("chosen FULLER setting", (16, float(row16[laser_col])), xytext=(6, 8), textcoords="offset points", fontsize=7.0, color="#555555")

    ax2 = ax1.twinx()
    l2 = ax2.plot(
        df["N_wdm"],
        df["PP_crosstalk_db"],
        marker="s",
        linestyle="--",
        color=PALETTE[3],
        linewidth=1.3,
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="Crosstalk",
    )[0]
    ax2.set_ylabel("Crosstalk (dB)", color=PALETTE[3])
    ax2.tick_params(axis="y", labelcolor=PALETTE[3])

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left", frameon=False, fontsize=7.4)
    fig.text(
        0.5,
        0.015,
        "Support-only PHY envelope: the vertical guide marks the retained FULLER operating point; this figure constrains realism, not system-level performance.",
        ha="center",
        fontsize=7.0,
        color="#666666",
    )
    svg_path, _, _ = _save_fig(fig, "Fig12_PHY_N_Sweep", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig12",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary=f"x=N_wdm; left={laser_col}; right=PP_crosstalk_db; operating_point=16; single_column_dual_axis",
        notes="Support-only PHY realism envelope with the retained FULLER operating point marked explicitly.",
    )


def plot_fig13_heatmap(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    points_path = quick_dir / "fig_h_accuracy_heatmap_points.csv"
    agg_path = quick_dir / "fig13_accuracy_heatmap_aggregated.csv"
    points = _load_csv(points_path)
    agg = _load_csv(agg_path) if agg_path.exists() else pd.DataFrame()
    if points.empty and agg.empty:
        return

    if not points.empty:
        for col in ["sigma_lsb", "crosstalk_alpha", "acc_drop_pp"]:
            points[col] = _to_num(points[col])
        points = points.dropna(subset=["model", "sigma_lsb", "crosstalk_alpha", "acc_drop_pp"])
    if not agg.empty:
        for col in ["sigma_lsb", "crosstalk_alpha", "acc_drop_pp_cell_mean"]:
            agg[col] = _to_num(agg[col])
        agg = agg.dropna(subset=["sigma_lsb", "crosstalk_alpha", "acc_drop_pp_cell_mean"])

    if points.empty:
        return

    per_model = (
        points.groupby(["model", "sigma_lsb", "crosstalk_alpha"], as_index=False)
        .agg(
            acc_drop_pp_model_mean=("acc_drop_pp", "mean"),
            replicate_count=("acc_drop_pp", "size"),
        )
    )
    if agg.empty:
        agg = (
            per_model.groupby(["sigma_lsb", "crosstalk_alpha"], as_index=False)
            .agg(
                acc_drop_pp_cell_mean=("acc_drop_pp_model_mean", "mean"),
                model_count=("model", "nunique"),
                replicate_count_total=("replicate_count", "sum"),
                replicate_count_per_model_min=("replicate_count", "min"),
                replicate_count_per_model_max=("replicate_count", "max"),
            )
        )

    model_labels = {
        "mobilevit_s": "MobileViT-S",
        "mobilevit_xs": "MobileViT-XS",
        "mobilevit_xxs": "MobileViT-XXS",
    }
    model_order = [m for m in ["mobilevit_s", "mobilevit_xs", "mobilevit_xxs"] if m in set(per_model["model"].astype(str))]
    if not model_order:
        return

    panel_data: list[tuple[str, pd.DataFrame]] = []
    for model in model_order:
        panel_df = per_model[per_model["model"].astype(str) == model]
        pivot = panel_df.pivot(index="crosstalk_alpha", columns="sigma_lsb", values="acc_drop_pp_model_mean").sort_index(ascending=False)
        if not pivot.empty:
            panel_data.append((model_labels.get(model, model), pivot))

    mean_pivot = agg.pivot(index="crosstalk_alpha", columns="sigma_lsb", values="acc_drop_pp_cell_mean").sort_index(ascending=False)
    mean_count_pivot = (
        agg.pivot(index="crosstalk_alpha", columns="sigma_lsb", values="model_count").sort_index(ascending=False)
        if "model_count" in agg.columns
        else pd.DataFrame()
    )
    if not mean_pivot.empty:
        panel_data.append(("Coverage-aware Mean", mean_pivot))
    if len(panel_data) < 2:
        return

    values = np.concatenate([pivot.to_numpy(dtype=float).ravel() for _, pivot in panel_data])
    values = values[~np.isnan(values)]
    if values.size == 0:
        return
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    fig = plt.figure(figsize=(7.18, 5.46))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.07],
        height_ratios=[1.0, 1.0],
        left=0.09,
        right=0.93,
        top=0.90,
        bottom=0.17,
        wspace=0.24,
        hspace=0.30,
    )
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )
    cax = fig.add_subplot(gs[:, 2])
    axes_flat = list(axes.flat)
    cbar_mesh = None
    budget = 1.0
    note_parts: list[str] = []

    for idx, ((title, pivot), ax) in enumerate(zip(panel_data, axes_flat, strict=False)):
        hm = _heatmap(
            pivot,
            annot=False,
            cmap="cividis",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            linewidths=0.45,
            linecolor="white",
            cbar=False,
        )
        mesh = hm.collections[0]
        if cbar_mesh is None:
            cbar_mesh = mesh

        ax.set_title(title, fontsize=8.4, fontweight="bold", pad=6)
        ax.tick_params(axis="x", rotation=0, pad=1, labelsize=7.4)
        ax.tick_params(axis="y", rotation=0, pad=1, labelsize=7.4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if idx % 2 == 1:
            ax.tick_params(labelleft=False)
        if idx < 2:
            ax.tick_params(labelbottom=False)

        cmap = mesh.cmap
        norm = mesh.norm
        for yi, alpha in enumerate(pivot.index):
            for xi, sigma in enumerate(pivot.columns):
                val = float(pivot.loc[alpha, sigma])
                r, g, b, _ = cmap(norm(val))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                text_color = "white" if luminance < 0.45 else "#111111"
                x_center = xi + 0.5 if HAS_SEABORN else xi
                y_center = yi + 0.5 if HAS_SEABORN else yi
                ax.text(
                    x_center,
                    y_center,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    fontweight="bold",
                    color=text_color,
                )
                if val <= budget:
                    rect_x = xi if HAS_SEABORN else xi - 0.5
                    rect_y = yi if HAS_SEABORN else yi - 0.5
                    ax.add_patch(
                        plt.Rectangle(
                            (rect_x, rect_y),
                            1,
                            1,
                            fill=False,
                            edgecolor="#111111",
                            linewidth=1.1,
                        )
                    )

    for ax in axes_flat[len(panel_data):]:
        ax.axis("off")

    if cbar_mesh is not None:
        cbar = fig.colorbar(cbar_mesh, cax=cax, orientation="vertical")
        cbar.outline.set_linewidth(0.6)
        cbar.ax.tick_params(labelsize=7.4, pad=1)
        cbar.set_label("Accuracy Drop (pp)", fontsize=7.8, labelpad=4)

    rep_min = int(_to_num(per_model["replicate_count"]).min())
    rep_max = int(_to_num(per_model["replicate_count"]).max())
    model_count = len(model_order)
    partial_coverage_cells = 0
    full_coverage_cells = 0
    full_coverage_coords: list[tuple[float, float]] = []
    sparse_support_only_cells = 0
    duplicate_profile_cells = 0
    if not mean_count_pivot.empty:
        partial_coverage_cells = int((_to_num(mean_count_pivot.stack()).fillna(0) < model_count).sum())
        full_coverage_cells = int((_to_num(mean_count_pivot.stack()).fillna(0) == model_count).sum())
        for alpha in mean_count_pivot.index:
            for sigma in mean_count_pivot.columns:
                count_value = mean_count_pivot.loc[alpha, sigma]
                if pd.isna(count_value):
                    continue
                if int(count_value) == model_count:
                    full_coverage_coords.append((float(sigma), float(alpha)))
                elif int(count_value) == 1:
                    sparse_support_only_cells += 1
    if "replicate_count_per_model_max" in agg.columns:
        duplicate_profile_cells = int((_to_num(agg["replicate_count_per_model_max"]).fillna(0) > 1).sum())
    fig.supxlabel("Gaussian Noise $\\sigma$ (LSB)", fontsize=8.0, y=0.085)
    fig.supylabel("Crosstalk $\\alpha$", fontsize=8.0, x=0.035)
    if partial_coverage_cells == 0 and full_coverage_cells > 0:
        coverage_note = (
            "Coverage-aware mean averages per-model coordinate means, then averages across models.\n"
            f"All {full_coverage_cells} populated cells have full {model_count}-model coverage; repeated representative anchors are collapsed before averaging."
        )
    elif full_coverage_coords:
        coverage_coord_text = ", ".join(
            f"({sigma:g}, {alpha:g})" for sigma, alpha in sorted(full_coverage_coords)
        )
        coverage_note = (
            "Coverage-aware mean averages available models per cell.\n"
            f"Full {model_count}-model coverage appears only at {coverage_coord_text}; "
            "all other populated cells are MobileViT-S-only by design."
        )
    else:
        coverage_note = (
            "Coverage-aware mean averages available models per cell.\n"
            "Partial-coverage cells are expected in this bounded support surface."
        )
    fig.text(0.5, 0.112, coverage_note, ha="center", fontsize=6.6, color="#666666")
    note_parts.append(
        "panel_layout=2x2(model-specific+mean); "
        f"model_order={'|'.join(model_order)}; "
        f"aggregation_rule=per_model_mean_then_cross_model_mean; model_count={model_count}; replicate_range={rep_min}-{rep_max}; partial_coverage_cells={partial_coverage_cells}; full_coverage_cells={full_coverage_cells}; duplicate_profile_cells={duplicate_profile_cells}"
    )
    if full_coverage_coords:
        note_parts.append(
            "full_coverage_coords="
            + "|".join(f"({sigma:g},{alpha:g})" for sigma, alpha in sorted(full_coverage_coords))
            + f"; sparse_support_only_cells={sparse_support_only_cells}"
        )

    svg_path, _, _ = _save_fig(fig, "Fig13_AccHeatmap", out_dir, use_tight_layout=False)
    _record_trace(
        trace_rows,
        fig_id="Fig13",
        figure_file=svg_path,
        input_csvs=[points_path] + ([agg_path] if agg_path.exists() else []),
        run_tag=run_tag,
        command=cmd,
        params_summary="2x2 heatmap panels (MobileViT-S|XS|XXS|mean); shared cividis scale; contrast-aware labels; shared vertical_colorbar; budget_outline=1.0pp",
        notes="; ".join(note_parts),
    )


def plot_fig14_fanout(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "quickscan_e1_fanout_sweep.csv"
    df = _load_csv(path)
    if df.empty:
        return

    df["fanout"] = _to_num(df.get("fanout", df.get("fanout_cfg")))
    df["broadcast_driver_energy_j"] = _to_num(df["broadcast_driver_energy_j"])
    df["net_energy_gain_j"] = _to_num(df["net_energy_gain_j"])
    df = df.dropna(subset=["fanout", "broadcast_driver_energy_j", "net_energy_gain_j"]).sort_values("fanout")
    if df.empty:
        return

    cost = df["broadcast_driver_energy_j"]
    benefit = df["net_energy_gain_j"] + df["broadcast_driver_energy_j"]

    fig, ax = plt.subplots(figsize=(3.60, 2.48))
    fig.subplots_adjust(bottom=0.22)
    ax.plot(df["fanout"], benefit, marker="o", color=PALETTE[2], label="benefit", linewidth=1.4, markersize=4.6, markerfacecolor="white", markeredgewidth=1.0)
    ax.plot(df["fanout"], cost, marker="s", linestyle="--", color=PALETTE[3], label="cost", linewidth=1.3, markersize=4.2, markerfacecolor="white", markeredgewidth=1.0)
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xlabel("Broadcast Fanout")
    ax.set_ylabel("Energy (J)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    fig.text(
        0.5,
        0.015,
        "Support-only MESO cost envelope: fanout changes explicit benefit and driver-cost terms, not the full end-to-end performance stack.",
        ha="center",
        fontsize=7.0,
        color="#666666",
    )
    svg_path, _, _ = _save_fig(fig, "Fig14_MESO_Fanout", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig14",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="x=fanout; y=benefit|cost",
        notes="Support-only MESO break-even figure; the retained sweep is an explicit cost model, not a measured system-performance sweep.",
    )


def plot_fig15_pareto(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "fig_j_sparse_tau_pareto.csv"
    df = _load_csv(path)
    if df.empty:
        return

    main_col = _pick_col(df, ["measured_acc_drop_pp_vs_E0_mean", "acc_drop_pp_mean"])
    if main_col is None:
        return
    support_col = _pick_col(df, ["support_acc_drop_pp_vs_E0_mean", "measured_acc_drop_pp_vs_E0_mean"])
    df["tau"] = _to_num(df["tau"])
    df[main_col] = _to_num(df[main_col])
    if support_col is not None:
        df[support_col] = _to_num(df[support_col])
    df["energy_saved_pct"] = _to_num(df["energy_saved_pct"])
    df = df.dropna(subset=["tau", main_col, "energy_saved_pct"]).sort_values("tau")
    if df.empty:
        return
    evidence = df.get("accuracy_evidence", pd.Series("", index=df.index)).astype(str).str.lower()
    support_df = df.dropna(subset=[support_col]).copy() if support_col is not None else pd.DataFrame()
    anchor_df = df[evidence.str.contains("full_eval_anchor")].copy()

    fig, ax = plt.subplots(figsize=(3.62, 2.45))
    if not support_df.empty:
        ax.plot(
            support_df[support_col],
            support_df["energy_saved_pct"],
            linestyle="--",
            color="#8a8a8a",
            linewidth=1.0,
            zorder=1,
            label="support sweep (context only)",
        )
        ax.scatter(
            support_df[support_col],
            support_df["energy_saved_pct"],
            s=30,
            marker="o",
            facecolors="white",
            edgecolors="#8a8a8a",
            linewidths=0.9,
            zorder=2,
        )
    if not anchor_df.empty:
        ax.scatter(
            anchor_df[main_col],
            anchor_df["energy_saved_pct"],
            s=78,
            marker="*",
            facecolors="#d62728",
            edgecolors="#8c1d18",
            linewidths=0.8,
            zorder=4,
            label="governed E4 full-eval anchor",
        )
    label_offsets = {0.00: (5, 4), 0.25: (5, -10), 0.50: (-30, 4)}
    for tau in [0.00, 0.25, 0.50]:
        sub = df[np.isclose(df["tau"], tau)]
        if sub.empty:
            continue
        row = sub.iloc[0]
        row_evidence = str(row.get("accuracy_evidence", "")).lower()
        use_anchor_x = "full_eval_anchor" in row_evidence
        x_value = (
            float(row[main_col])
            if use_anchor_x
            else float(row[support_col]) if support_col is not None and pd.notna(row[support_col]) else float(row[main_col])
        )
        ax.annotate(
            f"tau={row['tau']:.2f}",
            (x_value, row["energy_saved_pct"]),
            xytext=label_offsets.get(float(tau), (4, 4)),
            textcoords="offset points",
            fontsize=7.4,
        )
    if len(df) >= 2 and support_col is not None and not support_df.empty:
        start = support_df.iloc[min(1, len(support_df) - 1)]
        end = support_df.iloc[-1]
        ax.annotate(
            "higher tau",
            (end[support_col], end["energy_saved_pct"]),
            xytext=(start[support_col], start["energy_saved_pct"] + 7.0),
            arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8},
            fontsize=7.2,
            color="#555555",
        )

    ax.set_xlabel("Top-1 Drop vs E0 (pp)")
    ax.set_ylabel("Energy Saved (%)")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    fig.text(
        0.5,
        0.015,
        "Gray sweep is support-only context. The red star is the promoted E4 full-eval point at duty=0.75.",
        ha="center",
        fontsize=7.0,
        color="#666666",
    )
    svg_path, _, _ = _save_fig(fig, "Fig15_SparsePareto", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig15",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary=f"support_col={support_col or 'none'}; anchor_col={main_col}; x_ref=E0_quantized; governed_anchor_overlay=true; support_context_only=true",
        notes="Gray sweep is support-only context; the highlighted star is the governed E4 full-eval anchor at duty=0.75.",
    )


def plot_fig16_alignment(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "p0_p1_alignment_summary.csv"
    df = _load_csv(path)
    if df.empty:
        return

    for col in ["delta_p_db", "sigma_lsb_pred", "crosstalk_alpha_pred", "p_laser_dbm_eff"]:
        df[col] = _to_num(df[col])
    df = df.dropna(subset=["delta_p_db", "sigma_lsb_pred", "p_laser_dbm_eff"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(3.60, 2.48))
    e5 = df[df["experiment_id"] == "E5"].sort_values("delta_p_db")
    e6 = df[df["experiment_id"] == "E6"].sort_values("delta_p_db")
    sigma_identical = (
        not e5.empty
        and not e6.empty
        and len(e5) == len(e6)
        and np.allclose(e5["sigma_lsb_pred"].to_numpy(dtype=float), e6["sigma_lsb_pred"].to_numpy(dtype=float))
    )
    if sigma_identical:
        ax.plot(
            e5["delta_p_db"],
            e5["sigma_lsb_pred"],
            marker="o",
            linestyle="-",
            color="#555555",
            linewidth=1.3,
            label="shared sigma envelope",
        )
        ax.set_ylabel("Predicted Noise $\\sigma$ (LSB)", color="#555555")
        ax.tick_params(axis="y", labelcolor="#555555")
        ax2 = ax.twinx()
        for exp, sub, marker, ls, col in [
            ("E5", e5, "o", "-", PALETTE[0]),
            ("E6", e6, "s", "--", PALETTE[3]),
        ]:
            ax2.plot(
                sub["delta_p_db"],
                sub["p_laser_dbm_eff"],
                marker=marker,
                linestyle=ls,
                color=col,
                linewidth=1.3,
                label=f"{exp} effective laser power",
            )
        ax2.set_ylabel("Effective Laser Power (dBm)")
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=7.2, loc="upper left")
    else:
        for exp, marker, ls, col in [("E5", "o", "-", PALETTE[0]), ("E6", "s", "--", PALETTE[3])]:
            sub = df[df["experiment_id"] == exp].sort_values("delta_p_db")
            if sub.empty:
                continue
            ax.plot(
                sub["delta_p_db"],
                sub["sigma_lsb_pred"],
                marker=marker,
                linestyle=ls,
                color=col,
                label=f"{exp} Gaussian noise sigma",
            )
        ax.set_ylabel("Predicted Noise $\\sigma$ (LSB)")
        ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax.invert_xaxis()
    ax.set_xlabel("Power Delta (dB)")
    svg_path, _, _ = _save_fig(fig, "Fig16_Alignment", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig16",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="x=delta_p_db; shared_sigma_if_identical=true; secondary_y=p_laser_dbm_eff_when_needed",
        notes="When E5/E6 sigma envelopes are identical, the figure collapses sigma to one shared curve and uses the secondary axis for effective laser power.",
    )


def plot_fig17_overall_pareto(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "quickpack_e0_e6_overview.csv"
    df = _load_csv(path)
    if df.empty:
        return

    keep = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    df = df[df["experiment_id"].isin(keep)].copy()
    if df.empty:
        return

    required_cols = ["energy_j", "speedup_vs_E0", "throughput_images_s"]
    if any(col not in df.columns for col in required_cols):
        return
    num_cols = ["energy_j", "speedup_vs_E0", "throughput_images_s"]
    for col in num_cols:
        df[col] = _to_num(df[col])
    df = df.dropna(subset=["energy_j", "speedup_vs_E0"])
    if df.empty:
        return
    df["accuracy_drop_pp_measured"] = _to_num(
        df.get("measured_acc_drop_pp_vs_E0_mean", pd.Series(np.nan, index=df.index))
    )

    def _interp_drop_from_surface(agg: pd.DataFrame, sigma: float, alpha: float) -> float:
        xs = sorted(float(v) for v in agg["sigma_lsb"].dropna().unique())
        ys = sorted(float(v) for v in agg["crosstalk_alpha"].dropna().unique())
        surface = {
            (float(row["sigma_lsb"]), float(row["crosstalk_alpha"])): float(row["acc_drop_pp_cell_mean"])
            for _, row in agg.iterrows()
        }

        def _bounds(values: list[float], target: float) -> tuple[float, float]:
            if target <= values[0]:
                return values[0], values[0]
            if target >= values[-1]:
                return values[-1], values[-1]
            for idx in range(1, len(values)):
                if target <= values[idx]:
                    return values[idx - 1], values[idx]
            return values[-1], values[-1]

        x0, x1 = _bounds(xs, sigma)
        y0, y1 = _bounds(ys, alpha)

        def _value(x: float, y: float) -> float:
            return surface[(x, y)]

        if x0 == x1 and y0 == y1:
            return _value(x0, y0)
        if x0 == x1:
            v0 = _value(x0, y0)
            v1 = _value(x0, y1)
            if y0 == y1:
                return v0
            return v0 + (v1 - v0) * ((alpha - y0) / (y1 - y0))
        if y0 == y1:
            v0 = _value(x0, y0)
            v1 = _value(x1, y0)
            return v0 + (v1 - v0) * ((sigma - x0) / (x1 - x0))
        q11 = _value(x0, y0)
        q21 = _value(x1, y0)
        q12 = _value(x0, y1)
        q22 = _value(x1, y1)
        tx = (sigma - x0) / (x1 - x0)
        ty = (alpha - y0) / (y1 - y0)
        return (
            q11 * (1.0 - tx) * (1.0 - ty)
            + q21 * tx * (1.0 - ty)
            + q12 * (1.0 - tx) * ty
            + q22 * tx * ty
        )

    df["accuracy_drop_pp_total"] = df["accuracy_drop_pp_measured"]

    x_col = "energy_j"
    y_col = "speedup_vs_E0"
    x_label = "Energy per Inference (J)"
    y_label = "Speedup vs. E0 (x)"

    fig, ax = plt.subplots(figsize=(3.60, 2.48))
    markers = {"E0": "o", "E1": "s", "E2": "D", "E3": "^", "E4": "v", "E5": "P", "E6": "X"}
    for exp in keep:
        sub = df[df["experiment_id"] == exp]
        if sub.empty:
            continue
        color = EXP_COLOR.get(exp, PALETTE[0])
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=92,
            marker=markers.get(exp, "o"),
            facecolors=color,
            edgecolors=color,
            linewidths=0.8,
            zorder=4,
        )
    label_offset = {
        "E2": (12, 16),
        "E3": (14, -4),
        "E4": (10, 10),
        "E6": (12, -6),
        "E5": (8, 12),
    }
    for exp in ["E2", "E3", "E4", "E5", "E6"]:
        sub = df[df["experiment_id"] == exp]
        if sub.empty:
            continue
        row = sub.iloc[0]
        acc_drop = float(row.get("accuracy_drop_pp_measured", math.nan))
        label = exp if not math.isfinite(acc_drop) else f"{exp}\n{acc_drop:.1f} pp"
        ax.annotate(
            label,
            (row[x_col], row[y_col]),
            xytext=label_offset.get(exp, (6, 6)),
            textcoords="offset points",
            fontsize=7.2,
            fontweight="bold",
            color="#111111",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )

    cluster = df[df["experiment_id"].isin(["E0", "E1"])]
    if not cluster.empty:
        cluster_x = float(cluster[x_col].mean())
        cluster_y = float(cluster[y_col].mean())
        cluster_acc = float(cluster["accuracy_drop_pp_measured"].mean()) if cluster["accuracy_drop_pp_measured"].notna().any() else math.nan
        cluster_label = "E0/E1"
        if math.isfinite(cluster_acc):
            cluster_label = f"{cluster_label}\n{cluster_acc:.1f} pp"
        ax.annotate(
            cluster_label,
            (cluster_x, cluster_y),
            xytext=(-48, -14),
            textcoords="offset points",
            fontsize=7.2,
            color="#555555",
            ha="right",
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#555555",
                "linewidth": 0.7,
                "shrinkA": 2,
                "shrinkB": 5,
            },
        )

    ax.ticklabel_format(style="plain", axis="x", useOffset=False)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    x_min = float(df[x_col].min())
    x_max = float(df[x_col].max())
    y_min = float(df[y_col].min())
    y_max = float(df[y_col].max())
    x_pad = max(0.0022, 0.28 * (x_max - x_min))
    y_pad = max(0.05, 0.12 * (y_max - y_min))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(max(0.90, y_min - y_pad), y_max + y_pad)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    measured_handle = plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, markerfacecolor=PALETTE[0], markeredgecolor=PALETTE[0], label="measured / shared-ref")
    legend_handles = [measured_handle]
    legend_labels = ["measured / shared-ref"]
    footer_bits = [
        "Labels show mechanism-only accuracy drop vs E0",
        "E1 shares the E0 full-eval reference",
    ]
    ax.legend(handles=legend_handles, labels=legend_labels, frameon=False, fontsize=7.1, loc="upper right")
    fig.text(0.5, 0.015, "; ".join(footer_bits) + ".", ha="center", fontsize=5.9, color="#666666")
    svg_path, _, _ = _save_fig(fig, "Fig17_OverallPareto", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig17",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary=(
            f"x_col={x_col}; y_col={y_col}; inline_accuracy_drop=mechanism_only"
        ),
        notes=(
            "Inline labels report mechanism-only accuracy drop vs E0; "
            "filled markers denote measured/shared-reference accuracy; "
            "E1 shares the E0 full-eval reference in the active freeze."
        ),
    )


def plot_fig18_waterfall(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    path = quick_dir / "det_net_gain_waterfall.csv"
    df = _load_csv(path)
    if df.empty:
        return

    row = df[df["experiment_id"] == "E6"]
    if row.empty:
        row = df.iloc[[0]]
    row = row.iloc[0]

    det_saved = float(_to_num(pd.Series([row.get("det_saved_j")])).iloc[0] or 0.0)
    det_overhead = float(_to_num(pd.Series([row.get("det_overhead_j")])).iloc[0] or 0.0)
    det_net = float(_to_num(pd.Series([row.get("det_net_gain_j")])).iloc[0] or (det_saved - det_overhead))

    fig, ax = plt.subplots(figsize=(3.60, 2.48))
    x = np.arange(3)
    labels = ["Gross Saved", "Overhead", "Net Gain"]

    ax.bar(x[0], det_saved, color=PALETTE[2])
    ax.bar(x[1], -det_overhead, bottom=det_saved, color=PALETTE[3])
    ax.bar(x[2], det_net, color=PALETTE[0])

    ax.plot([0.4, 1.6], [det_saved, det_saved], linestyle="--", color="#444444", linewidth=0.8)
    ax.plot([1.4, 2.0], [det_saved - det_overhead, det_net], linestyle="--", color="#444444", linewidth=0.8)
    ax.axhline(0.0, color="#777777", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy (J)")
    svg_path, _, _ = _save_fig(fig, "Fig18_DET_Waterfall", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig18",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="exp=E6_preferred; categories=gross_saved|overhead|net_gain",
    )


def plot_fig19_ablation(quick_dir: Path, out_dir: Path, trace_rows: list[dict[str, Any]], run_tag: str, cmd: str) -> None:
    contract_path = quick_dir / "fig19_ablation_contract_summary.csv"
    if contract_path.exists():
        df = _load_csv(contract_path)
        if df.empty:
            return
        for col in ["latency_ms", "energy_j", "acc_drop_pp", "speedup_vs_astra", "energy_ratio_vs_astra"]:
            if col in df.columns:
                df[col] = _to_num(df[col])
        df = df.dropna(subset=["variant_label", "latency_ms", "energy_j", "acc_drop_pp"])
        if df.empty:
            return

        order = ["ASTRA", "HOPS", "HOPS+MESO", "HOPS+PHY", "FULLER"]
        df["variant_label"] = pd.Categorical(df["variant_label"], categories=order, ordered=True)
        df = df.sort_values("variant_label")
        x = np.arange(len(df))
        colors = ["#4e79a7", "#59a14f", "#9c755f", "#f28e2b", "#e15759"]
        hatches = ["", "//", "..", "xx", "///"]
        acc_values = df["acc_drop_pp"].to_numpy(dtype=float)
        acc_is_flat = bool(np.isfinite(acc_values).all()) and float(acc_values.max() - acc_values.min()) < 1e-9
        acc_evidence = df.get("accuracy_evidence", pd.Series("", index=df.index)).astype(str).str.lower()
        rounded_values = pd.Series(np.round(acc_values, 9))
        dominant_value = float(rounded_values.value_counts().index[0]) if not rounded_values.empty else math.nan
        dominant_count = int(rounded_values.value_counts().iloc[0]) if not rounded_values.empty else 0
        shared_contract_mask = np.zeros(len(df), dtype=bool)
        shared_contract_value: float | None = None
        acc_mode = "variant_rows"
        if acc_is_flat:
            acc_mode = "flat_contract"
        elif len(df) >= 3 and dominant_count >= 2 and dominant_count < len(df):
            shared_contract_mask = np.isclose(acc_values, dominant_value, atol=1e-9)
            if int(shared_contract_mask.sum()) >= 2 and int((~shared_contract_mask).sum()) >= 1:
                shared_contract_value = dominant_value
                acc_mode = "shared_contract_plateau"

        fig, axes = plt.subplots(1, 3, figsize=(7.04, 2.62), sharex=True)
        panels = [
            ("latency_ms", "Latency (ms)"),
            ("energy_j", "Energy (J)"),
            ("acc_drop_pp", "Accuracy Drop (pp)"),
        ]
        for ax, (metric, ylabel) in zip(axes, panels):
            if metric == "acc_drop_pp" and acc_is_flat:
                ax.plot(x, df[metric], linestyle="--", color="#666666", linewidth=1.0, zorder=1)
                ax.scatter(
                    x,
                    df[metric],
                    s=36,
                    marker="o",
                    facecolors="white",
                    edgecolors="#333333",
                    linewidths=0.9,
                    zorder=3,
                )
                center = float(df[metric].iloc[0])
                pad = max(0.02, abs(center) * 0.15 + 0.02)
                ax.set_ylim(center - pad, center + pad)
                ax.text(
                    0.04,
                    0.92,
                    "shared contract value",
                    transform=ax.transAxes,
                    fontsize=6.8,
                    color="#555555",
                    va="top",
                )
            elif metric == "acc_drop_pp" and acc_mode == "shared_contract_plateau" and shared_contract_value is not None:
                bars = ax.bar(x, df[metric], color="#f7f7f7", edgecolor="#444444", linewidth=0.65)
                for idx, bar in enumerate(bars):
                    if shared_contract_mask[idx]:
                        bar.set_hatch("//")
                    else:
                        bar.set_facecolor("#e15759")
                        bar.set_edgecolor("#6f1d1b")
                        bar.set_hatch("///")
                shared_positions = x[shared_contract_mask]
                if len(shared_positions) > 0:
                    ax.hlines(
                        shared_contract_value,
                        float(shared_positions.min()) - 0.34,
                        float(shared_positions.max()) + 0.34,
                        colors="#666666",
                        linestyles="--",
                        linewidth=0.9,
                        zorder=4,
                    )
                    ax.annotate(
                        f"shared contract plateau\n{shared_contract_value:.2f} pp",
                        xy=(float(shared_positions.mean()), shared_contract_value),
                        xytext=(0.04, 0.94),
                        textcoords="axes fraction",
                        ha="left",
                        va="top",
                        fontsize=6.4,
                        color="#555555",
                        arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 0.7},
                    )
                for idx in np.where(~shared_contract_mask)[0]:
                    y_val = float(df.iloc[idx][metric])
                    ax.text(
                        x[idx],
                        y_val + max(1.0, y_val * 0.02),
                        f"{y_val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color="#6f1d1b",
                    )
                max_y = float(np.nanmax(acc_values))
                ax.set_ylim(0.0, max_y * 1.12 if max_y > 0 else 1.0)
            else:
                bars = ax.bar(x, df[metric], color=colors, edgecolor="#222222", linewidth=0.5)
                for bar, hatch in zip(bars, hatches):
                    bar.set_hatch(hatch)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(df["variant_label"].astype(str), rotation=24, ha="right", fontsize=7.1)
            ax.tick_params(axis="y", labelsize=7.3)
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
            ax.set_axisbelow(True)

        fig.text(
            0.5,
            0.015,
            (
                "Governed active-route ablation: ASTRA -> HOPS -> HOPS+MESO -> HOPS+PHY -> FULLER. "
                + (
                    "Accuracy is flat in the current contract table and is rendered explicitly."
                    if acc_mode == "flat_contract"
                    else f"Accuracy is evidence-scoped: {'/'.join(df.loc[shared_contract_mask, 'variant_label'].astype(str))} share the current mapped value; {'/'.join(df.loc[~shared_contract_mask, 'variant_label'].astype(str))} remains the separate row."
                    if acc_mode == "shared_contract_plateau"
                    else "Accuracy panel reflects variant-specific rows from the current contract table."
                )
            ),
            ha="center",
            fontsize=7.0,
            color="#666666",
        )
        svg_path, _, _ = _save_fig(fig, "Fig19_Ablation", out_dir)

        _record_trace(
            trace_rows,
            fig_id="Fig19",
            figure_file=svg_path,
            input_csvs=[contract_path],
            run_tag=run_tag,
            command=cmd,
            params_summary=(
                "panels=latency_ms|energy_j|acc_drop_pp; baseline=ASTRA; "
                f"source=contract_ablation_summary; accuracy_mode={acc_mode}"
            ),
            notes=(
                "Rebuilt to the active fuller ablation contract with ASTRA/HOPS/HOPS+MESO/HOPS+PHY/FULLER ordering; "
                "the accuracy panel is explicitly rendered when the contract table is flat."
                if acc_mode == "flat_contract"
                else f"Rebuilt to the active fuller ablation contract with evidence-scoped accuracy: {'/'.join(df.loc[shared_contract_mask, 'variant_label'].astype(str))} share the current mapped value in the active table, while {'/'.join(df.loc[~shared_contract_mask, 'variant_label'].astype(str))} remains the separate row."
                if acc_mode == "shared_contract_plateau"
                else "Rebuilt to the active fuller ablation contract with variant-specific rows from the current table."
            ),
        )
        return

    path = quick_dir / "ablation_summary.csv"
    df = _load_csv(path)
    if df.empty:
        return
    keep = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    df = df[df["experiment_id"].isin(keep)].copy()
    if df.empty:
        return
    acc_col = "acc_drop_pp_mean" if "acc_drop_pp_mean" in df.columns else _pick_col(df, ["measured_acc_drop_pp_mean"])
    if acc_col is None:
        return
    for col in ["speedup_vs_E0", "energy_ratio_vs_E0", acc_col]:
        df[col] = _to_num(df[col])
    df = df.dropna(subset=["speedup_vs_E0", "energy_ratio_vs_E0", acc_col])
    if df.empty:
        return
    df["experiment_id"] = pd.Categorical(df["experiment_id"], categories=keep, ordered=True)
    df = df.sort_values("experiment_id")
    fig, ax1 = plt.subplots(figsize=(3.60, 2.48))
    x = np.arange(len(df))
    w = 0.34
    b1 = ax1.bar(x - w / 2, df["speedup_vs_E0"], width=w, color=PALETTE[0], hatch="///", edgecolor="black", label="speedup")
    b2 = ax1.bar(x + w / 2, df["energy_ratio_vs_E0"], width=w, color="#b0b0b0", hatch="...", edgecolor="black", label="energy ratio")
    ax1.set_ylabel("Speedup / Energy Ratio (x)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["experiment_id"].astype(str))
    ax2 = ax1.twinx()
    evidence = df.get("accuracy_evidence", pd.Series("", index=df.index)).astype(str).str.lower()
    modeled_mask = evidence.str.contains("modeled").to_numpy(dtype=bool)
    ax2.plot(x, df[acc_col], linestyle="--", color="#d62728", linewidth=1.2, alpha=0.85)
    measured_pts = ax2.scatter(x[~modeled_mask], df.loc[~modeled_mask, acc_col], s=28, marker="o", facecolors="#d62728", edgecolors="#d62728", linewidths=0.8, label="measured / inherited acc", zorder=4)
    modeled_pts = ax2.scatter(x[modeled_mask], df.loc[modeled_mask, acc_col], s=30, marker="o", facecolors="white", edgecolors="#d62728", linewidths=1.2, label="modeled acc", zorder=5)
    ax2.set_ylabel("Accuracy Drop (pp)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.legend([b1, b2, measured_pts, modeled_pts], ["speedup vs E0", "energy ratio vs E0", "measured / inherited acc", "modeled acc"], loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=7.0)
    fig.text(0.5, 0.015, "Hollow red markers indicate modeled accuracy rows.", ha="center", fontsize=7.0, color="#666666")
    svg_path, _, _ = _save_fig(fig, "Fig19_Ablation", out_dir)
    _record_trace(trace_rows, fig_id="Fig19", figure_file=svg_path, input_csvs=[path], run_tag=run_tag, command=cmd, params_summary=f"acc_col={acc_col}; hollow_markers=modeled_accuracy", notes="Measured/inherited accuracy and modeled accuracy are rendered with separate marker treatments.")


def plot_appf1_workload_generalization(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
) -> None:
    agg_path = quick_dir / "appf1_internal_model_scale_context.csv"
    path = agg_path if agg_path.exists() else (quick_dir / "task_generalization_summary.csv")
    df = _load_csv(path)
    if df.empty:
        return

    for col in ["primary_metric_value", "tops_w", "latency_ms"]:
        df[col] = _to_num(df[col])
    df = df.dropna(subset=["primary_metric_value", "tops_w", "latency_ms"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(3.62, 2.82))
    df["experiment_id"] = df["experiment_id"].astype(str)
    exp_order = [exp for exp in ["E0", "E1", "E2", "E3", "E4", "E5", "E6"] if exp in set(df["experiment_id"])]
    model_markers = {"mobilevit_s": "o", "mobilevit_xs": "s", "mobilevit_xxs": "^"}
    metric_name = (
        df.get("primary_metric_name", pd.Series(dtype=str)).dropna().astype(str).mode().iloc[0]
        if "primary_metric_name" in df.columns and not df["primary_metric_name"].dropna().empty
        else "Primary Metric"
    )
    x_label = "Top-1 Accuracy (%)" if metric_name.lower() == "top1" else f"{metric_name} Value"
    for model, sub in df.groupby(df["model"].astype(str)):
        ordered = sub.copy()
        ordered["experiment_id"] = pd.Categorical(ordered["experiment_id"], categories=exp_order, ordered=True)
        ordered = ordered.sort_values("experiment_id")
        ax.plot(
            ordered["primary_metric_value"],
            ordered["tops_w"],
            color="#bbbbbb",
            linewidth=0.9,
            zorder=1,
        )
    for _, row in df.iterrows():
        exp = str(row.get("experiment_id", ""))
        marker = model_markers.get(str(row.get("model", "")), "o")
        ax.scatter(
            row["primary_metric_value"],
            row["tops_w"],
            s=34,
            marker=marker,
            color=EXP_COLOR.get(exp, PALETTE[0]),
            alpha=0.82,
            linewidths=0.4,
            edgecolors="#222222",
            zorder=3,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("TOPS/W")
    ax.grid(axis="both", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    exp_handles = [
        plt.Line2D([0], [0], linestyle="", marker="o", markersize=4.8, color=EXP_COLOR.get(exp, PALETTE[0]), label=exp)
        for exp in exp_order
    ]
    if exp_handles:
        ax.legend(exp_handles, [h.get_label() for h in exp_handles], frameon=False, fontsize=7.0, ncol=min(4, len(exp_handles)), loc="upper left")

    note_parts: list[str] = [
        "single_workload=W0_mobilevit_imagenet",
        "context_only=internal_model_scale",
    ]
    fig.text(
        0.5,
        0.015,
        "Internal MobileViT ImageNet model-scale context only; single workload, not workload generalization evidence.",
        ha="center",
        fontsize=7.0,
        color="#666666",
    )
    audit_note = quick_dir / "audits" / "task_generalization_external_anchors_excluded_from_paper_20260328.md"
    if audit_note.exists():
        note_parts.append(f"audit artifact: {audit_note}")

    svg_path, _, _ = _save_fig(fig, "AppF1_WorkloadGeneralization", out_dir)
    _record_trace(
        trace_rows,
        fig_id="AppF1",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary=f"x=primary_metric_value({metric_name}); y=tops_w; grouped_by=experiment_id,model; single_workload_context",
        notes="; ".join(note_parts),
    )


def plot_appf2_module_breakdown(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
) -> None:
    path = quick_dir / "module_breakdown_by_block.csv"
    df = _load_csv(path)
    if df.empty:
        return

    df["module_group"] = df["module_group"].astype(str).str.lower().replace({"attn": "attention"})
    df = df[df["module_group"].isin(MODULE_ORDER)].copy()
    if df.empty:
        return

    for col in ["module_energy_j", "energy_j"]:
        df[col] = _to_num(df[col])

    # Prefer run-level all_models rows for clean decomposition.
    if "model" in df.columns and (df["model"] == "all_models").any():
        work = df[df["model"] == "all_models"].copy()
    else:
        work = df.copy()

    agg = (
        work.groupby(["experiment_id", "module_group"], as_index=False)["module_energy_j"]
        .mean()
    )
    if agg.empty:
        return

    exp_order = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    exps = [e for e in exp_order if e in set(agg["experiment_id"].astype(str))]
    if not exps:
        exps = sorted(agg["experiment_id"].astype(str).unique().tolist())

    mat = np.zeros((len(MODULE_ORDER), len(exps)), dtype=float)
    for i, mod in enumerate(MODULE_ORDER):
        for j, exp in enumerate(exps):
            sub = agg[(agg["module_group"] == mod) & (agg["experiment_id"].astype(str) == exp)]
            if not sub.empty:
                mat[i, j] = float(sub["module_energy_j"].mean())

    fig, ax = plt.subplots(figsize=(3.62, 2.90))
    x = np.arange(len(exps))
    bottom = np.zeros(len(exps), dtype=float)
    for i, mod in enumerate(MODULE_ORDER):
        vals = mat[i]
        bars = ax.bar(x, vals, bottom=bottom, width=0.68, color=MODULE_COLOR[mod], label=mod)
        bottom += vals
        for b in bars:
            b.set_linewidth(0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(exps)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Module Energy (J)")
    ax.legend(frameon=False, ncol=2, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, 1.08))

    svg_path, _, _ = _save_fig(fig, "AppF2_ModuleBreakdown", out_dir)

    # Gate-X3 check: module sum vs total energy within 1%.
    gate_note = ""
    if "energy_j" in work.columns:
        run_energy = (
            work.groupby("experiment_id", as_index=False)["energy_j"].mean().set_index("experiment_id")["energy_j"].to_dict()
        )
        errs = []
        for exp in exps:
            total_mod = float(mat[:, exps.index(exp)].sum())
            base = float(run_energy.get(exp, 0.0))
            if base > 0:
                errs.append(abs(total_mod - base) / base)
        if errs:
            max_err = max(errs)
            gate_note = f"Gate-X3 max_rel_err={max_err:.4f}"
            if max_err > 0.01:
                print(f"[render][warn] Gate-X3 failed for AppF2: max_rel_err={max_err:.4f}")

    _record_trace(
        trace_rows,
        fig_id="AppF2",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="x=experiment_id; stack=module_group; y=module_energy_j",
        notes=gate_note,
    )


def plot_appf3_batch_scaling(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
) -> None:
    path = quick_dir / "quickscan_batch_seq_scaling.csv"
    df = _load_csv(path)
    if df.empty:
        return

    for col in ["batch_size", "sequence_length", "throughput_images_s"]:
        df[col] = _to_num(df[col])
    df = df.dropna(subset=["batch_size", "throughput_images_s", "sequence_length"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(3.60, 2.48))
    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    used_rows: list[pd.DataFrame] = []
    ref_seq_map: dict[str, float] = {}
    for i, model in enumerate(models):
        sub = df[df["model"].astype(str) == model].copy()
        if sub.empty:
            continue
        # Batch-scaling curve uses a fixed sequence length per model.
        seq_counts = sub["sequence_length"].value_counts(dropna=True)
        if seq_counts.empty:
            continue
        ref_seq = float(seq_counts.index[0])
        ref_seq_map[model] = ref_seq
        sub = sub[sub["sequence_length"] == ref_seq]
        if sub.empty:
            continue
        sub = (
            sub.groupby("batch_size", as_index=False)[["throughput_images_s"]]
            .mean()
            .sort_values("batch_size")
        )
        if sub.empty:
            continue
        sub["model"] = model
        used_rows.append(sub)
        ax.plot(
            sub["batch_size"],
            sub["throughput_images_s"],
            marker="o",
            linewidth=1.4,
            color=PALETTE[i % len(PALETTE)],
            label=model,
        )

    if not used_rows:
        plt.close(fig)
        return
    used_df = pd.concat(used_rows, ignore_index=True)

    if used_df["batch_size"].nunique() < 2:
        ax.text(0.02, 0.95, "single-point batch reference", transform=ax.transAxes, va="top", fontsize=7, color="#666666")

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Throughput (images/s)")
    ax.legend(frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=min(3, max(1, len(models))))
    svg_path, _, _ = _save_fig(fig, "AppF3_BatchScaling", out_dir)

    # Gate-X4 partial check.
    min_unique_per_model = int(
        used_df.groupby("model", as_index=False)["batch_size"].nunique()["batch_size"].min()
    )
    seq_note = ",".join(
        f"{m}:{int(v) if float(v).is_integer() else v}"
        for m, v in sorted(ref_seq_map.items())
    )
    gate_note = (
        f"Gate-X4 unique_batch={int(used_df['batch_size'].nunique())}; "
        f"min_model_unique_batch={min_unique_per_model}; ref_seq={seq_note}"
    )
    _record_trace(
        trace_rows,
        fig_id="AppF3",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="x=batch_size; y=throughput_images_s; series=model",
        notes=gate_note,
    )


def plot_appf4_sequence_scaling(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
) -> None:
    path = quick_dir / "quickscan_batch_seq_scaling.csv"
    df = _load_csv(path)
    if df.empty:
        return

    for col in ["batch_size", "sequence_length", "latency_ms", "throughput_tokens_s"]:
        df[col] = _to_num(df[col])
    df = df.dropna(subset=["batch_size", "sequence_length", "latency_ms", "throughput_tokens_s"])
    if df.empty:
        return

    # Sequence-scaling curve uses a fixed batch size (prefer modal batch).
    ref_batch_note = "mixed"
    batch_counts = df["batch_size"].value_counts(dropna=True)
    if not batch_counts.empty:
        ref_batch = float(batch_counts.index[0])
        sub = df[df["batch_size"] == ref_batch]
        if sub["sequence_length"].nunique() >= 2:
            df = sub
            ref_batch_note = str(int(ref_batch) if ref_batch.is_integer() else ref_batch)

    # Aggregate across models for sequence trend.
    agg = (
        df.groupby("sequence_length", as_index=False)[["latency_ms", "throughput_tokens_s"]]
        .mean()
        .sort_values("sequence_length")
    )
    if agg.empty:
        return

    fig, ax1 = plt.subplots(figsize=(3.60, 2.48))
    l1 = ax1.plot(agg["sequence_length"], agg["latency_ms"], marker="o", color=PALETTE[0], label="latency_ms")[0]
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Latency (ms)", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])

    ax2 = ax1.twinx()
    l2 = ax2.plot(agg["sequence_length"], agg["throughput_tokens_s"], marker="s", linestyle="--", color=PALETTE[3], label="throughput_tokens_s")[0]
    ax2.set_ylabel("Token Throughput (tokens/s)", color=PALETTE[3])
    ax2.tick_params(axis="y", labelcolor=PALETTE[3])

    if agg["sequence_length"].nunique() < 2:
        ax1.text(0.02, 0.95, "single-point sequence reference", transform=ax1.transAxes, va="top", fontsize=7, color="#666666")

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=7.4)
    svg_path, _, _ = _save_fig(fig, "AppF4_SequenceScaling", out_dir)

    gate_note = (
        f"Gate-X4 unique_seq={int(agg['sequence_length'].nunique())}; "
        f"ref_batch={ref_batch_note}"
    )
    _record_trace(
        trace_rows,
        fig_id="AppF4",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary="x=sequence_length; y=latency_ms|throughput_tokens_s",
        notes=gate_note,
    )


def plot_appf5_wdm_mrr_quant(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
) -> None:
    path = quick_dir / "quickscan_mrr_wdm_quant.csv"
    df = _load_csv(path)
    if df.empty:
        return

    for col in ["N_wdm", "mrr_tile_k", "quant_bits", "energy_j"]:
        df[col] = _to_num(df[col])
    df = df.dropna(subset=["N_wdm", "mrr_tile_k", "quant_bits", "energy_j"])
    if df.empty:
        return

    mrr_vals = sorted(df["mrr_tile_k"].dropna().unique().tolist())
    qbits = sorted(df["quant_bits"].dropna().unique().tolist())
    q_colors = {q: PALETTE[i % len(PALETTE)] for i, q in enumerate(qbits)}
    note_parts: list[str] = []
    fixed_bits = ",".join(str(int(q)) for q in qbits) if qbits else "n/a"
    fixed_mrr = ",".join(str(int(v)) for v in mrr_vals) if mrr_vals else "n/a"

    if len(qbits) == 1 and len(mrr_vals) == 1:
        fig, ax = plt.subplots(figsize=(3.62, 2.52))
        sub = df.sort_values("N_wdm")
        ax.plot(
            sub["N_wdm"],
            sub["energy_j"],
            marker="o",
            linewidth=1.3,
            color=PALETTE[0],
            markerfacecolor="white",
            markeredgewidth=1.0,
        )
        ax.set_xlabel("N_wdm")
        ax.set_ylabel("Energy (J)")
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)
        ax.grid(axis="x", visible=False)
        fig.text(
            0.5,
            0.015,
            "Current freeze primarily varies N_wdm; quant_bits and/or mrr_tile_k remain fixed reference settings in this support figure.",
            ha="center",
            fontsize=7.0,
            color="#666666",
        )
        note_parts.append(f"support_only_fixed_refs quant_bits={fixed_bits} mrr_tile_k={fixed_mrr}")
        params_summary = f"x=N_wdm; y=energy_j; fixed_quant_bits={fixed_bits}; fixed_mrr_tile_k={fixed_mrr}"
    else:
        ncols = min(3, max(1, len(mrr_vals)))
        nrows = int(math.ceil(len(mrr_vals) / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.6 * ncols, 2.5 * nrows), squeeze=False)

        for idx, mrr in enumerate(mrr_vals):
            ax = axes[idx // ncols][idx % ncols]
            sub_m = df[df["mrr_tile_k"] == mrr]
            for q in qbits:
                sub_q = sub_m[sub_m["quant_bits"] == q].sort_values("N_wdm")
                if sub_q.empty:
                    continue
                ax.plot(sub_q["N_wdm"], sub_q["energy_j"], marker="o", linewidth=1.2, color=q_colors[q], label=f"{int(q)}-bit")
            ax.set_title(f"k_tile = {int(mrr)}", fontsize=8.2)
            ax.set_xlabel("N_wdm")
            ax.set_ylabel("Energy (J)")

        for j in range(len(mrr_vals), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        legend_handles = [
            plt.Line2D([0], [0], color=q_colors[q], marker="o", linewidth=1.2, label=f"{int(q)}-bit")
            for q in qbits
        ]
        fig.legend(
            legend_handles,
            [f"{int(q)}-bit" for q in qbits],
            frameon=False,
            fontsize=7.2,
            ncol=min(3, max(1, len(qbits))),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
        )
        if len(qbits) == 1 or len(mrr_vals) == 1:
            fig.text(
                0.5,
                0.015,
                "Current freeze primarily varies N_wdm; quant_bits and/or mrr_tile_k remain fixed reference settings in this support figure.",
                ha="center",
                fontsize=7.0,
                color="#666666",
            )
            note_parts.append(f"support_only_fixed_refs quant_bits={fixed_bits} mrr_tile_k={fixed_mrr}")
        params_summary = "x=N_wdm,hue=quant_bits,facet=mrr_tile_k"

    svg_path, _, _ = _save_fig(fig, "AppF5_WDM_MRR_Quant", out_dir)

    _record_trace(
        trace_rows,
        fig_id="AppF5",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary=params_summary,
        notes="; ".join(note_parts),
    )


def plot_appf6_broadcast_vs_onetoone(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
) -> None:
    fan_path = quick_dir / "quickscan_e1_fanout_sweep.csv"
    ov_path = quick_dir / "quickpack_e0_e6_overview.csv"
    fan = _load_csv(fan_path)
    ov = _load_csv(ov_path)
    if fan.empty or ov.empty:
        return

    fan["fanout"] = _to_num(fan.get("fanout", fan.get("fanout_cfg")))
    fan["energy_j"] = _to_num(fan["energy_j"])
    fan["latency_ms"] = _to_num(fan["latency_ms"])
    fan = fan.dropna(subset=["fanout", "energy_j", "latency_ms"]).sort_values("fanout")
    if fan.empty:
        return

    e0 = ov[ov["experiment_id"] == "E0"]
    baseline_energy = float(_to_num(e0["energy_j"]).mean()) if not e0.empty else float("nan")
    baseline_latency = float(_to_num(e0["latency_ms"]).mean()) if not e0.empty else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(6.92, 2.56))
    ax0, ax1 = axes
    energy_delta_pct = ((fan["energy_j"] / baseline_energy) - 1.0) * 100.0 if math.isfinite(baseline_energy) and baseline_energy > 0 else fan["energy_j"] * 0.0
    latency_delta_pct = ((fan["latency_ms"] / baseline_latency) - 1.0) * 100.0 if math.isfinite(baseline_latency) and baseline_latency > 0 else fan["latency_ms"] * 0.0
    chosen_fanout = 4.0 if (fan["fanout"] == 4.0).any() else float(fan["fanout"].median())

    ax0.plot(
        fan["fanout"],
        energy_delta_pct,
        marker="o",
        color=PALETTE[0],
        label="energy delta",
        linewidth=1.4,
        markersize=4.6,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax0.axhline(0.0, linestyle="--", color="#555555", label="E0 one-to-one baseline", linewidth=1.2)
    ax0.axvline(chosen_fanout, linestyle=":", color="#777777", linewidth=0.9)
    ax0.set_xlabel("Fanout")
    ax0.set_ylabel("Energy Delta vs E0 (%)")
    ax0.grid(axis="y", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)
    ax0.grid(axis="x", visible=False)

    ax1.plot(
        fan["fanout"],
        latency_delta_pct,
        marker="s",
        color=PALETTE[3],
        label="latency delta",
        linewidth=1.3,
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax1.axhline(0.0, linestyle="--", color="#555555", label="E0 one-to-one baseline", linewidth=1.2)
    ax1.axvline(chosen_fanout, linestyle=":", color="#777777", linewidth=0.9)
    ax1.set_xlabel("Fanout")
    ax1.set_ylabel("Latency Delta vs E0 (%)")
    ax1.grid(axis="y", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)
    ax1.grid(axis="x", visible=False)

    legend_handles = [
        plt.Line2D([0], [0], color=PALETTE[0], marker="o", markerfacecolor="white", markeredgewidth=1.0, linewidth=1.4, label="delta curve"),
        plt.Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.2, label="zero delta"),
        plt.Line2D([0], [0], color="#777777", linestyle=":", linewidth=0.9, label="chosen fanout"),
    ]
    fig.legend(legend_handles, ["delta curve", "zero delta", "chosen fanout"], frameon=False, fontsize=7.2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.text(
        0.5,
        0.015,
        "Support-only delta context: the retained broadcast sweep stays near the E0 baseline, with the chosen fanout marked explicitly.",
        ha="center",
        fontsize=7.0,
        color="#666666",
    )
    svg_path, _, _ = _save_fig(fig, "AppF6_BroadcastVsOneToOne", out_dir)

    _record_trace(
        trace_rows,
        fig_id="AppF6",
        figure_file=svg_path,
        input_csvs=[fan_path, ov_path],
        run_tag=run_tag,
        command=cmd,
        params_summary=f"x=fanout; left=energy_delta_pct_vs_E0; right=latency_delta_pct_vs_E0; chosen_fanout={chosen_fanout:g}",
        notes="Support-only delta comparison between the broadcast sweep and the E0 one-to-one baseline.",
    )


def plot_appf7_noise_surface(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
) -> None:
    path = quick_dir / "noise_robustness_surface.csv"
    df = _load_csv(path)
    if df.empty:
        return

    # Keep eval set by default for reporting stability.
    if "workload_id" in df.columns:
        eval_rows = df[df["workload_id"].astype(str).str.contains("_eval", na=False)]
        if not eval_rows.empty:
            df = eval_rows

    for col in ["noise_sigma_lsb", "crosstalk_alpha", "primary_metric_drop", "quant_bits"]:
        df[col] = _to_num(df[col])

    # Gate: single quant_bits in one panel.
    qb_mode = float(df["quant_bits"].dropna().mode().iloc[0]) if not df["quant_bits"].dropna().empty else 8.0
    panel = df[df["quant_bits"] == qb_mode].copy()
    panel = panel.dropna(subset=["noise_sigma_lsb", "crosstalk_alpha", "primary_metric_drop"])
    if panel.empty:
        return

    per_model = (
        panel.groupby(["model", "crosstalk_alpha", "noise_sigma_lsb"], as_index=False)
        .agg(
            primary_metric_drop_model_mean=("primary_metric_drop", "mean"),
            replicate_count=("primary_metric_drop", "size"),
        )
    )
    agg = (
        per_model.groupby(["crosstalk_alpha", "noise_sigma_lsb"], as_index=False)
        .agg(
            primary_metric_drop=("primary_metric_drop_model_mean", "mean"),
            model_count=("model", "nunique"),
            replicate_count_total=("replicate_count", "sum"),
            replicate_count_per_model_min=("replicate_count", "min"),
            replicate_count_per_model_max=("replicate_count", "max"),
        )
    )
    pivot = agg.pivot(index="crosstalk_alpha", columns="noise_sigma_lsb", values="primary_metric_drop").sort_index(ascending=False)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(3.62, 3.08))
    fig.subplots_adjust(bottom=0.27)
    _heatmap(
        pivot,
        annot=False,
        cmap="magma",
        ax=ax,
        cbar_kws={"label": "Primary Metric Drop (pp)"},
    )
    budget = 1.0
    for yi, alpha in enumerate(pivot.index):
        for xi, sigma in enumerate(pivot.columns):
            val = float(pivot.loc[alpha, sigma])
            if val >= budget:
                ax.add_patch(plt.Rectangle((xi, yi), 1, 1, fill=False, edgecolor="#a50026", linewidth=1.1))
    ax.text(0.02, -0.18, f"quant_bits={int(qb_mode)}; outlined >= {budget:.1f}pp", transform=ax.transAxes, fontsize=7.2)
    ax.set_xlabel("Gaussian Noise $\\sigma$ (LSB)")
    ax.set_ylabel("Crosstalk $\\alpha$")
    model_count = int(_to_num(agg["model_count"]).max()) if "model_count" in agg.columns else 1
    duplicate_profile_cells = (
        int((_to_num(agg["replicate_count_per_model_max"]).fillna(0) > 1).sum())
        if "replicate_count_per_model_max" in agg.columns
        else 0
    )
    ax.text(
        0.02,
        -0.29,
        "Mean over per-model coordinate means (S/XS/XXS); repeated representative anchors collapsed.\n"
        f"quant_bits={int(qb_mode)}; outlined >= {budget:.1f}pp; full_model_coverage_cells={len(agg)}/{len(agg)}; duplicate_profile_cells={duplicate_profile_cells}",
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
    )
    mesh = ax.collections[0]
    cbar = mesh.colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=7.6, pad=1)
        cbar.set_label("Primary Metric Drop (pp)", fontsize=7.8, labelpad=4)
    svg_path, _, _ = _save_fig(fig, "AppF7_NoiseRobustnessSurface", out_dir)

    _record_trace(
        trace_rows,
        fig_id="AppF7",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary=f"quant_bits={int(qb_mode)}; aggregation_rule=per_model_mean_then_cross_model_mean",
        notes=f"budget contour: 1.0pp; model_count={model_count}; duplicate_profile_cells={duplicate_profile_cells}",
    )


def plot_appf8_hpat_cross_platform(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
) -> None:
    p_cmp = quick_dir / "hpat_cpu_gpu_compare.csv"
    df = _load_csv(p_cmp)
    if df.empty:
        return
    note_parts: list[str] = []

    for col in [
        "latency_ms",
        "energy_j",
        "avg_power_w",
        "tops_w",
        "peak_tops",
        "area_mm2",
        "batch_size",
        "input_size",
    ]:
        if col in df.columns:
            df[col] = _to_num(df[col])

    df = _clean_platform_context_rows(df, note_parts=note_parts, gate_name="Gate-X6")
    if df.empty:
        print("[render] skip AppF8: no rows remain after platform-context cleanup")
        return

    measured = df[df.get("evidence_tier", "").astype(str).str.lower() == "measured"].copy()
    modeled = df[df.get("evidence_tier", "").astype(str).str.lower() == "modeled"].copy()
    if measured.empty or modeled.empty:
        print("[render] skip AppF8: missing measured or modeled tier after cleanup")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.02, 2.84))
    ax_a, ax_b = axes
    p_colors = {"CPU": "#4e79a7", "GPU": "#59a14f", "HPAT": "#f28e2b"}
    model_short = {"mobilevit_xxs": "XXS", "mobilevit_xs": "XS", "mobilevit_s": "S"}

    for _, row in measured.iterrows():
        lat = float(row.get("latency_ms", np.nan))
        ene = float(row.get("energy_j", np.nan))
        pwr = float(row.get("avg_power_w", np.nan))
        if not math.isfinite(lat) or not math.isfinite(ene):
            continue
        platform = str(row.get("platform_class", "unknown")).upper()
        size = 28.0 + (max(0.0, pwr) * 2.0 if math.isfinite(pwr) else 0.0)
        ax_a.scatter(lat, ene, s=size, color=p_colors.get(platform, "#999999"), alpha=0.84, edgecolors="#222222", linewidths=0.45)
        ax_a.annotate(model_short.get(str(row.get("model", "")), str(row.get("model", ""))), (lat, ene), xytext=(3, 3), textcoords="offset points", fontsize=6.8, color="#333333")

    ax_a.set_title("Measured Local Devices", fontsize=8.1)
    ax_a.set_xlabel("Latency (ms)")
    ax_a.set_ylabel("Energy (J)")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.grid(axis="both", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)

    peak_series = _to_num(modeled.get("peak_tops", pd.Series(np.nan, index=modeled.index)))
    topsw_series = _to_num(modeled.get("tops_w", pd.Series(np.nan, index=modeled.index)))
    modeled = modeled.loc[peak_series.gt(0.0) & topsw_series.gt(0.0)].copy()
    if modeled.empty:
        print("[render] skip AppF8: no modeled efficiency rows remain")
        plt.close(fig)
        return
    note_parts.append("tier_separated_panels=measured_local_latency_energy|modeled_fuller_efficiency")
    note_parts.append("panelB_area_unavailable_fixed_marker_size")
    for _, row in modeled.iterrows():
        x = float(row.get("peak_tops", np.nan))
        y = float(row.get("tops_w", np.nan))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        ax_b.scatter(x, y, s=46.0, marker="s", color=p_colors.get("HPAT", "#f28e2b"), alpha=0.84, edgecolors="#222222", linewidths=0.45)
        ax_b.annotate(model_short.get(str(row.get("model", "")), str(row.get("model", ""))), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6.8, color="#333333")

    ax_b.set_title(f"Modeled {ACCELERATOR_SHORT_NAME} Endpoint", fontsize=8.1)
    ax_b.set_xlabel("Peak TOPS")
    ax_b.set_ylabel("TOPS/W")
    ax_b.grid(axis="both", color="#d9d9d9", linewidth=0.4, linestyle="--", alpha=0.6)

    gpu_legend_label = "GPU measured"
    cpu_rows = measured[measured["platform_class"].astype(str).str.upper() == "CPU"]
    gpu_rows = measured[measured["platform_class"].astype(str).str.upper() == "GPU"]
    if not gpu_rows.empty:
        gpu_legend_label = _platform_display_name(gpu_rows.iloc[0], gpu_legend_label)

    platform_handles = []
    if not cpu_rows.empty:
        platform_handles.append(
            plt.Line2D(
                [0],
                [0],
                linestyle="",
                marker="o",
                markersize=5.3,
                color=p_colors["CPU"],
                label=_platform_display_name(cpu_rows.iloc[0], "CPU measured"),
            )
        )
    if not gpu_rows.empty:
        platform_handles.append(
            plt.Line2D([0], [0], linestyle="", marker="o", markersize=5.3, color=p_colors["GPU"], label=gpu_legend_label)
        )
    platform_handles.append(
        plt.Line2D([0], [0], linestyle="", marker="s", markersize=5.3, color=p_colors["HPAT"], label=ACCELERATOR_SHORT_NAME)
    )
    fig.legend(
        platform_handles,
        [h.get_label() for h in platform_handles],
        frameon=False,
        fontsize=7.2,
        ncol=len(platform_handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.text(
        0.5,
        0.015,
        f"Panels are evidence-tier separated: measured local-device context at left, modeled {ACCELERATOR_SHORT_NAME} efficiency context at right.",
        ha="center",
        fontsize=7.0,
        color="#666666",
    )

    svg_path, _, _ = _save_fig(fig, "AppF8_HPATvsCrossPlatform", out_dir)

    _record_trace(
        trace_rows,
        fig_id="AppF8",
        figure_file=svg_path,
        input_csvs=[p_cmp],
        run_tag=run_tag,
        command=cmd,
        params_summary="panelA=measured_local_latency_energy(size=avg_power_w); panelB=modeled_fuller_peak_tops_vs_tops_w; tier_separated_context",
        notes="; ".join(note_parts),
    )


def plot_appf9_mtl_cpu_gpu_triplet(
    quick_dir: Path,
    out_dir: Path,
    trace_rows: list[dict[str, Any]],
    run_tag: str,
    cmd: str,
) -> None:
    path = quick_dir / "hpat_cpu_gpu_compare.csv"
    df = _load_csv(path)
    if df.empty:
        return

    for col in ["latency_ms", "energy_j", "avg_power_w", "batch_size", "input_size", "throughput_images_s"]:
        if col in df.columns:
            df[col] = _to_num(df[col])

    platform_map = {
        "CPU": "CPU",
        "GPU": "GPU",
        "HPAT": "FULLER",
        "MTL": "FULLER",
    }
    df["plot_platform"] = df.get("platform_class", pd.Series(dtype=str)).astype(str).str.upper().map(platform_map)
    df = df[df["plot_platform"].isin(["CPU", "GPU", "FULLER"])].copy()
    if df.empty:
        return

    for metric in ["latency_ms", "energy_j", "avg_power_w", "throughput_images_s"]:
        if metric not in df.columns:
            return
    df = df.dropna(subset=["model", "latency_ms", "energy_j", "avg_power_w", "throughput_images_s"])
    if df.empty:
        return

    note_parts: list[str] = []
    df = _clean_platform_context_rows(df, note_parts=note_parts, gate_name="Gate-X6")
    if df.empty:
        print("[render] skip Fig20: no rows remain after platform-context cleanup")
        return

    agg = df.groupby(["model", "plot_platform"], as_index=False)[["latency_ms", "energy_j", "avg_power_w", "throughput_images_s"]].mean()
    if agg.empty:
        return

    preferred_models = ["mobilevit_xxs", "mobilevit_xs", "mobilevit_s"]
    model_vals = sorted(agg["model"].astype(str).unique().tolist())
    model_order = [m for m in preferred_models if m in model_vals] + [m for m in model_vals if m not in preferred_models]
    if not model_order:
        return

    def _model_label(model: str) -> str:
        if model.startswith("mobilevit_"):
            suffix = model.split("mobilevit_", 1)[1]
            return f"MobileViT-{suffix.upper()}"
        return model

    platform_order = [platform for platform in ["CPU", "GPU"] if not df[df["plot_platform"] == platform].empty]
    platform_order.append("FULLER")
    platform_label = {"FULLER": ACCELERATOR_DEVICE_NAME}
    for platform in platform_order:
        if platform == "FULLER":
            continue
        sub = df[df["plot_platform"] == platform]
        if not sub.empty:
            label = _device_display_name(sub.iloc[0])
            platform_label[platform] = label
        else:
            platform_label[platform] = platform
    platform_color = {"CPU": "#4e79a7", "GPU": "#59a14f", "FULLER": "#f28e2b"}
    platform_marker = {"CPU": "o", "GPU": "o", "FULLER": "s"}

    metrics = [
        ("latency_ms", "Latency (ms)"),
        ("energy_j", "Energy (J)"),
        ("avg_power_w", "Avg Power (W)"),
        ("throughput_images_s", "Throughput (img/s)"),
    ]

    if len(model_order) == 1:
        # Each panel uses a different physical unit, so sharing a log-scaled y-axis
        # can suppress lower-magnitude metrics such as energy.
        fig, axes = plt.subplots(1, 4, figsize=(9.45, 2.95), sharey=False)
        platform_short = {"CPU": "CPU", "GPU": "GPU", "FULLER": ACCELERATOR_SHORT_NAME}
        model = model_order[0]

        def _fmt_metric_value(metric: str, value: float) -> str:
            if metric == "latency_ms":
                return f"{value:.3f} ms" if value < 1.0 else f"{value:.2f} ms"
            if metric == "energy_j":
                return f"{value:.4f} J" if value < 0.1 else f"{value:.3f} J"
            if metric == "avg_power_w":
                return f"{value:.3f} W" if value < 10.0 else f"{value:.2f} W"
            if metric == "throughput_images_s":
                return f"{value:.1f} img/s" if value < 100.0 else f"{value:.0f} img/s"
            return f"{value:.3g}"

        for col_idx, (metric, xlabel) in enumerate(metrics):
            ax = axes[col_idx]
            vals = []
            x = np.arange(len(platform_order), dtype=float)
            for platform in platform_order:
                sub = agg[(agg["model"].astype(str) == model) & (agg["plot_platform"] == platform)][metric]
                vals.append(float(sub.iloc[0]) if not sub.empty else float("nan"))

            positive_vals = [value for value in vals if math.isfinite(value) and value > 0.0]
            if not positive_vals:
                continue
            ymin = min(positive_vals) / 2.2
            ymax = max(positive_vals) * 2.0

            bars = ax.bar(
                x,
                vals,
                width=0.62,
                color=[platform_color[p] for p in platform_order],
                edgecolor="#222222",
                linewidth=0.5,
                zorder=3,
            )
            for idx, bar in enumerate(bars):
                value = vals[idx]
                platform = platform_order[idx]
                if not math.isfinite(value) or value <= 0.0:
                    continue
                if platform == "FULLER":
                    bar.set_hatch("///")
                ax.annotate(
                    _fmt_metric_value(metric, value),
                    (bar.get_x() + bar.get_width() / 2.0, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.3,
                )

            ax.set_yscale("log")
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel(xlabel)
            ax.set_xticks(x)
            ax.set_xticklabels([platform_short[p] for p in platform_order], fontsize=7.2)
            ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.45, zorder=1)
            ax.tick_params(axis="y", labelsize=7.0)
            if col_idx == 0:
                ax.set_ylabel(_model_label(model))

        legend_handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=platform_color[p],
                edgecolor="#222222",
                linewidth=0.5,
                hatch="///" if p == "FULLER" else "",
                label=platform_label[p],
            )
            for p in platform_order
        ]
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            frameon=False,
            fontsize=6.0,
            ncol=len(legend_handles),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
        )
        fig.text(
            0.5,
            0.02,
            (
                f"Single-model direct comparison for {_model_label(model)}: "
                "CPU/GPU bars are measured local-device context, while "
                f"{ACCELERATOR_SHORT_NAME} is the modeled accelerator row under the active freeze."
            ),
            ha="center",
            fontsize=7.0,
            color="#666666",
        )
        fig.subplots_adjust(left=0.075, right=0.995, top=0.78, bottom=0.22, wspace=0.38)
        params_summary = "direct_platform_bar_compare; x=CPU|GPU|FULLER; cols=latency_ms|energy_j|avg_power_w|throughput_images_s; logy=all; cpu_gpu_measured_fuller_modeled=true"
    else:
        platform_hatch = {"CPU": "", "GPU": "", "FULLER": "///"}
        fig, axes = plt.subplots(1, 4, figsize=(9.15, 2.95), sharex=False)
        x = np.arange(len(model_order), dtype=float)
        bar_w = 0.22

        for col_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[col_idx]
            for idx, platform in enumerate(platform_order):
                vals: list[float] = []
                for model in model_order:
                    sub = agg[(agg["model"].astype(str) == model) & (agg["plot_platform"] == platform)][metric]
                    vals.append(float(sub.iloc[0]) if not sub.empty else float("nan"))
                offset = (idx - (len(platform_order) - 1) / 2.0) * bar_w
                bars = ax.bar(
                    x + offset,
                    vals,
                    width=bar_w,
                    color=platform_color[platform],
                    edgecolor="#222222",
                    linewidth=0.5,
                    label=platform_label[platform],
                )
                if platform_hatch[platform]:
                    for bar in bars:
                        bar.set_hatch(platform_hatch[platform])
            ax.set_title(ylabel, fontsize=7.1, pad=16)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels([_model_label(m) for m in model_order], fontsize=7.1)
            ax.tick_params(axis="y", labelsize=7.2)
            ax.set_yscale("log")
            ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.45)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=platform_color[p], edgecolor="#222222", hatch=platform_hatch[p], linewidth=0.5)
            for p in platform_order
        ]
        fig.legend(
            legend_handles,
            [platform_label[p] for p in platform_order],
            frameon=False,
            fontsize=6.1,
            ncol=len(platform_order),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.075),
        )
        params_summary = "grouped_platform_bars; cols=latency_ms|energy_j|avg_power_w|throughput_images_s; logy=all; actual_device_names_in_legend=true"

    exact_device_note = _host_and_device_note(df)

    gate_msgs: list[str] = []
    missing: list[str] = []
    for model in model_order:
        for platform in platform_order:
            has_row = not agg[(agg["model"].astype(str) == model) & (agg["plot_platform"] == platform)].empty
            if not has_row:
                missing.append(f"{model}:{platform}")
    if missing:
        gate_msgs.append(f"Gate-X6 missing_pairs={','.join(missing)}")

    bad_setup: list[str] = []
    for model, sub in df.groupby(df["model"].astype(str)):
        cfg = sub[["batch_size", "input_size"]].dropna().drop_duplicates()
        if len(cfg) > 1:
            bad_setup.append(str(model))
    if bad_setup:
        gate_msgs.append(f"Gate-X6 warning inconsistent setup models={','.join(sorted(set(bad_setup)))}")

    gate_note = "; ".join(note_parts + gate_msgs)
    if gate_note:
        print(f"[render][warn] {gate_note}")

    fig.subplots_adjust(left=0.06, right=0.995, top=0.66, bottom=0.18, wspace=0.38)

    svg_path, _, _ = _save_fig(fig, "Fig20_DeviceComparison", out_dir)

    _record_trace(
        trace_rows,
        fig_id="Fig20",
        figure_file=svg_path,
        input_csvs=[path],
        run_tag=run_tag,
        command=cmd,
        params_summary=params_summary,
        notes="; ".join(
            [
                item
                for item in [
                    gate_note,
                    exact_device_note,
                    "CPU/GPU bars are measured local-device context rows; MTL-FULLER is the modeled accelerator row.",
                    "Fig20 renamed from legacy AppF9 and promoted to manuscript main tier for direct accelerator-vs-device comparison.",
                ]
                if item
            ]
        ),
    )


def _write_traceability(
    trace_rows: list[dict[str, Any]],
    out_dir: Path,
    run_tag: str,
    selected_fig_ids: set[str] | None = None,
) -> Path:
    out = out_dir / "figure_traceability.csv"
    review_out = _review_dir(run_tag) / "figure_traceability.csv"
    cols = [
        "figure_id",
        "manuscript_tier",
        "figure_file",
        "input_csvs",
        "script_entry",
        "command",
        "run_tag",
        "generated_at",
        "params_summary",
        "literature_style_anchors",
        "literature_anchor_scope",
        "notes",
    ]
    new_df = pd.DataFrame(trace_rows, columns=cols)
    existing_path = out if out.exists() else review_out
    if selected_fig_ids and existing_path.exists():
        old_df = pd.read_csv(existing_path)
        old_df = old_df[~old_df["figure_id"].isin(selected_fig_ids)]
        merged = pd.concat([old_df, new_df], ignore_index=True)
    else:
        merged = new_df
    if not merged.empty and "figure_id" in merged.columns:
        merged["__order"] = merged["figure_id"].map(_trace_sort_key)
        merged = merged.sort_values(["__order", "figure_id"]).drop(columns="__order")
    review_out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    merged.to_csv(review_out, index=False)
    return out


def _default_out_dir(quick_dir: Path) -> Path:
    run_tag = quick_dir.name
    return ROOT.parent / "figures" / f"paper_figures_{run_tag}"


def main() -> None:
    global WARN_MISSING_CSV
    MISSING_CSVS.clear()
    parser = argparse.ArgumentParser(description="Render optimized paper figures (Fig7-Fig20 + App-F1..App-F8)")
    parser.add_argument("--quick_dir", type=Path, default=DEFAULT_QUICK_DIR)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--radar_csv", type=Path, default=None, help="Optional explicit fig_a_related_work_radar_scores.csv")
    parser.add_argument(
        "--fig8_det_summary_csv",
        type=Path,
        default=None,
        help="Optional repaired DET-k summary csv for main-text Fig8.",
    )
    parser.add_argument(
        "--warn_missing_csv",
        action="store_true",
        help="Print each missing CSV path (default: summarize missing optional CSVs).",
    )
    parser.add_argument(
        "--fig_ids",
        nargs="*",
        default=None,
        help="Optional subset of figure ids to render and refresh in traceability.",
    )
    args = parser.parse_args()
    WARN_MISSING_CSV = args.warn_missing_csv

    quick_dir = args.quick_dir
    out_dir = args.out_dir or _default_out_dir(quick_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = quick_dir.name
    cmd = f"python3 experiments/tools/render_paper_figures.py --quick_dir {quick_dir} --out_dir {out_dir}"
    if args.radar_csv is not None:
        cmd += f" --radar_csv {args.radar_csv}"
    if args.fig8_det_summary_csv is not None:
        cmd += f" --fig8_det_summary_csv {args.fig8_det_summary_csv}"
    if args.warn_missing_csv:
        cmd += " --warn_missing_csv"
    selected_fig_ids = (
        {LEGACY_FIG_ID_ALIAS.get(fig_id, fig_id) for fig_id in args.fig_ids}
        if args.fig_ids
        else None
    )
    if args.fig_ids:
        cmd += " --fig_ids " + " ".join(args.fig_ids)

    print(f"[render] quick_dir={quick_dir}")
    print(f"[render] out_dir={out_dir}")

    trace_rows: list[dict[str, Any]] = []
    should_render = lambda fig_id: selected_fig_ids is None or fig_id in selected_fig_ids

    # Fig7-Fig20
    if should_render("Fig7"):
        plot_fig7_radar(quick_dir, out_dir, trace_rows, run_tag, cmd, args.radar_csv)
    if should_render("Fig8"):
        plot_fig8_det_acc(quick_dir, out_dir, trace_rows, run_tag, cmd, args.fig8_det_summary_csv)
    if should_render("Fig9"):
        plot_fig9_prefix_error(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig10"):
        plot_fig10_energy_breakdown(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig11"):
        plot_fig11_timeline(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig12"):
        plot_fig12_phy_sweep(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig13"):
        plot_fig13_heatmap(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig14"):
        plot_fig14_fanout(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig15"):
        plot_fig15_pareto(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig16"):
        plot_fig16_alignment(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig17"):
        plot_fig17_overall_pareto(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig18"):
        plot_fig18_waterfall(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig19"):
        plot_fig19_ablation(quick_dir, out_dir, trace_rows, run_tag, cmd)

    # App-F1..App-F8
    if should_render("AppF1"):
        plot_appf1_workload_generalization(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("AppF2"):
        plot_appf2_module_breakdown(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("AppF3"):
        plot_appf3_batch_scaling(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("AppF4"):
        plot_appf4_sequence_scaling(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("AppF5"):
        plot_appf5_wdm_mrr_quant(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("AppF6"):
        plot_appf6_broadcast_vs_onetoone(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("AppF7"):
        plot_appf7_noise_surface(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("AppF8"):
        plot_appf8_hpat_cross_platform(quick_dir, out_dir, trace_rows, run_tag, cmd)
    if should_render("Fig20"):
        plot_appf9_mtl_cpu_gpu_triplet(quick_dir, out_dir, trace_rows, run_tag, cmd)

    if MISSING_CSVS and not WARN_MISSING_CSV:
        print(
            "[render] skipped missing optional csvs="
            f"{len(MISSING_CSVS)} (use --warn_missing_csv to list paths)"
        )

    trace_path = _write_traceability(trace_rows, out_dir, run_tag, selected_fig_ids)
    manifest_paths = _write_manifests(trace_rows, out_dir, run_tag, selected_fig_ids)
    print(f"[render] traceability={trace_path}")
    if manifest_paths:
        print(f"[render] manifests={len(manifest_paths)}")
    print(f"[render] generated={len(trace_rows)} figures")


if __name__ == "__main__":
    main()
