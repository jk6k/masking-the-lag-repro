#!/usr/bin/env python3
"""Build the SUDS optical Transformer architecture simulation artifacts.

The model is intentionally architecture-level: it maps Transformer GEMM
kernels onto a Lightening-style DPTC tile array, then applies SUDS or baseline
resource-control policies to latency, energy, area, memory, optical-link, and
control-sideband accounting. It is a modeled architecture flow, not a device
implementation or hardware measurement flow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260513_tetc_pivot"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
REPORT_OUT = REPO_ROOT / "docs/reports/20260513_suds_transformer_architecture_sim.md"

KERNELS_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_kernels.csv"
SUMMARY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_summary.csv"
PARAMETERS_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_parameters.csv"
SENSITIVITY_CSV = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}_sensitivity.csv"
GLUE_LINK_CSV = REPORT_DATA / f"suds_glue_architecture_linkage_{TAG}.csv"
DESIGN_SPACE_CSV = REPORT_DATA / f"suds_transformer_architecture_design_space_{TAG}.csv"
DESIGN_SPACE_JSON = REPORT_DATA / f"suds_transformer_architecture_design_space_{TAG}.json"
JSON_OUT = REPORT_DATA / f"suds_transformer_architecture_sim_{TAG}.json"

MOBILEVIT_JSON = REPORT_DATA / "suds_mobilevit_multimodel_validation_20260511_p2p3_quality.json"
GLUE_JSON = REPORT_DATA / "suds_glue_measured_validation_20260511_p2p3_quality.json"
ADC_JSON = REPORT_DATA / "suds_adc_macro_sanity_20260512_j1_quality_boost.json"
RTL_JSON = REPORT_DATA / "suds_rtl_control_overhead_20260512_j2_quality_boost.json"
PHY_JSON = REPORT_DATA / "suds_phy_circuit_boundary_20260511_p2p3_quality.json"
MOBILEVIT_OPS = REPO_ROOT / "experiments/mtl_model/ops/ops_mobilevit_s.json"

KB_ROOT = Path("kb_root/markdown")


CONDITION_LABELS = {
    "lightening_dptc": "Lightening-style DPTC reference, 8-bit ADC",
    "uniform_8bit": "Uniform 8-bit DPTC mapping",
    "uniform_4bit": "Uniform 4-bit DPTC mapping",
    "random": "Random same-sparsity selector",
    "l1": "L1 selector",
    "slack_only": "Slack-only selector",
    "suds_only": "SUDS budget only",
    "signal_only": "Signal/L1 tier selector",
    "suds_l1": "SUDS budget + L1 selector",
    "suds_signal": "SUDS budget + signal/overflow selector",
    "hyatten_style": "HyAtten-style low-resolution signal selector",
    "tempo_time_multiplexed": "TeMPO-style time-multiplexed boundary",
    "astra_boundary": "ASTRA-style stochastic optical boundary",
}

SOURCE_CONDITIONS = {
    "uniform_8bit": "e0_dense",
    "lightening_dptc": "e0_dense",
    "random": "e5_random",
    "l1": "e2_l1",
    "slack_only": "e3_slack",
    "suds_only": "e4_suds",
    "signal_only": "e6_signal",
    "suds_l1": "e7_overlay",
    "suds_signal": "e8_overflow",
}

SUDS_CONDITIONS = {"suds_only", "suds_l1", "suds_signal"}
MAIN_SUDS_CONDITIONS = {"suds_l1", "suds_signal"}
BASELINE_CONDITIONS = {
    "lightening_dptc",
    "uniform_8bit",
    "random",
    "l1",
    "slack_only",
    "signal_only",
    "hyatten_style",
    "tempo_time_multiplexed",
    "astra_boundary",
}
SAME_SCOPE_BASELINES = {"lightening_dptc", "l1", "slack_only", "signal_only", "hyatten_style"}
WORKLOADS_REQUIRED = {"bert_base_glue_seq128", "mobilevit_s_transformer_blocks_256"}

DESIGN_SPACE_CONDITIONS = (
    "lightening_dptc",
    "hyatten_style",
    "tempo_time_multiplexed",
    "astra_boundary",
    "suds_l1",
    "suds_signal",
)
DESIGN_SPACE_SWEEP = {
    "tile_dim": (16, 32, 64),
    "tiles": (2, 4, 8),
    "cores_per_tile": (1, 2, 4),
    "sideband_group_cols": (16, 32, 64, 128),
    "adc_sharing": ("per_array", "per_tile", "temporal_accum"),
}
SELECTED_OPERATING_POINT = {
    "tile_dim": 32,
    "tiles": 4,
    "cores_per_tile": 2,
    "sideband_group_cols": 32,
    "adc_sharing": "temporal_accum",
}


@dataclass(frozen=True)
class ArchitectureParams:
    tile_dim: int
    tiles: int
    cores_per_tile: int
    frequency_ghz: float
    sram_global_kib: int
    sram_subarray_kib: int
    adc8_pj: float
    adc6_pj: float
    adc4_pj: float
    dac_pj: float
    mzm_pj: float
    detector_tia_pj: float
    optical_link_pj_per_output: float
    laser_pj_per_active_tile: float
    memory_pj_per_byte: float
    digital_fallback_pj_per_mac: float
    control_pj_per_sideband_group: float
    sideband_columns: int
    sideband_bits_per_column: int
    rtl_area_ge_per_group: float
    phy_nominal_pass_ratio: float
    phy_pessimistic_laser_multiplier: float
    lightening_adc_temporal_factor: float

    @property
    def parallel_cores(self) -> int:
        return self.tiles * self.cores_per_tile

    @property
    def cycle_ns(self) -> float:
        return 1.0 / self.frequency_ghz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--mobilevit-json", type=Path, default=MOBILEVIT_JSON)
    parser.add_argument("--glue-json", type=Path, default=GLUE_JSON)
    parser.add_argument("--adc-json", type=Path, default=ADC_JSON)
    parser.add_argument("--rtl-json", type=Path, default=RTL_JSON)
    parser.add_argument("--phy-json", type=Path, default=PHY_JSON)
    parser.add_argument("--mobilevit-ops", type=Path, default=MOBILEVIT_OPS)
    parser.add_argument("--kernels-csv", type=Path, default=KERNELS_CSV)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--parameters-csv", type=Path, default=PARAMETERS_CSV)
    parser.add_argument("--sensitivity-csv", type=Path, default=SENSITIVITY_CSV)
    parser.add_argument("--glue-link-csv", type=Path, default=GLUE_LINK_CSV)
    parser.add_argument("--design-space-csv", type=Path, default=DESIGN_SPACE_CSV)
    parser.add_argument("--design-space-json", type=Path, default=DESIGN_SPACE_JSON)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def median_float(values: list[Any], default: float) -> float:
    clean = []
    for value in values:
        try:
            clean.append(float(value))
        except (TypeError, ValueError):
            pass
    return statistics.median(clean) if clean else default


def derive_params(adc: dict[str, Any], rtl: dict[str, Any], phy: dict[str, Any]) -> ArchitectureParams:
    adc_rows = [row for row in adc.get("rows", []) if row.get("status") == "measured"]
    adc4 = median_float(
        [row.get("energy_per_conversion_pj") for row in adc_rows if int(row.get("adc_bits", 0)) == 4],
        1.0 / 16.0,
    )
    adc6 = median_float(
        [row.get("energy_per_conversion_pj") for row in adc_rows if int(row.get("adc_bits", 0)) == 6],
        0.25,
    )
    adc8 = median_float(
        [row.get("energy_per_conversion_pj") for row in adc_rows if int(row.get("adc_bits", 0)) == 8],
        1.0,
    )

    rtl_rows = rtl.get("rows", [])
    rtl_1ghz = [
        row for row in rtl_rows
        if int(row.get("columns", 0)) == 32 and int(row.get("target_mhz", 0)) == 1000
    ]
    rtl_anchor = rtl_1ghz[0] if rtl_1ghz else (rtl_rows[0] if rtl_rows else {})
    dynamic_power_uw = float(rtl_anchor.get("dynamic_power_uw_proxy") or 590.4)
    # uW at 1 GHz is fJ/cycle; convert to pJ/cycle.
    control_pj_per_group = dynamic_power_uw / 1000.0

    phy_meta = phy.get("metadata", {})
    pass_rows = float(phy_meta.get("pass_rows") or 0.0)
    fail_rows = float(phy_meta.get("fail_rows") or 0.0)
    phy_ratio = pass_rows / max(1.0, pass_rows + fail_rows)

    return ArchitectureParams(
        tile_dim=32,
        tiles=4,
        cores_per_tile=2,
        frequency_ghz=5.0,
        sram_global_kib=2048,
        sram_subarray_kib=32,
        adc8_pj=adc8,
        adc6_pj=adc6,
        adc4_pj=adc4,
        dac_pj=0.65 * adc8,
        mzm_pj=0.22 * adc8,
        detector_tia_pj=0.12 * adc8,
        optical_link_pj_per_output=0.045 * adc8,
        laser_pj_per_active_tile=0.18 * adc8,
        memory_pj_per_byte=0.015,
        digital_fallback_pj_per_mac=0.00012,
        control_pj_per_sideband_group=control_pj_per_group,
        sideband_columns=int(rtl_anchor.get("columns") or 32),
        sideband_bits_per_column=3,
        rtl_area_ge_per_group=float(rtl_anchor.get("area_ge_proxy") or 640.0),
        phy_nominal_pass_ratio=phy_ratio,
        phy_pessimistic_laser_multiplier=1.15,
        lightening_adc_temporal_factor=1.0 / 6.0,
    )


def parameter_rows(params: ArchitectureParams, args: argparse.Namespace) -> list[dict[str, Any]]:
    lightening = KB_ROOT / "01_transformer_attention_photonic/Lightening_Transformer_HPCA2024.md"
    hyatten = KB_ROOT / "01_transformer_attention_photonic/2501.11286_HyAtten_Hybrid_Photonic_Digital_Attention_Accelerator.md"
    tempo = KB_ROOT / "01_transformer_attention_photonic/2402.07393_TeMPO_Transformer_Acceleration_with_Co-packaged_Silicon_Photonics.md"
    astra = KB_ROOT / "01_transformer_attention_photonic/ASTRA_Stochastic_Transformer_Silicon_Photonics_TECS2025.md"
    enlighten = KB_ROOT / "01_transformer_attention_photonic/ENLighten_Lighten_the_Transformer_Enable_Efficient_Optical_Acceleration_arXiv2510.01673.md"
    rows = [
        {
            "parameter": "tile_dim",
            "value": params.tile_dim,
            "unit": "DPTC rows/columns",
            "evidence_label": "literature_anchored_assumption",
            "source": repo_path(lightening),
            "source_anchor": "Lightening Fig. 4/5 and Table II; HyAtten discusses 32x32 DPTC arrays",
            "claim_boundary": "architecture simulator parameter, not layout closure",
        },
        {
            "parameter": "tiles",
            "value": params.tiles,
            "unit": "tiles",
            "evidence_label": "literature_anchored_assumption",
            "source": repo_path(lightening),
            "source_anchor": "Lightening architecture configuration describes LT-B with four tiles",
            "claim_boundary": "matched Lightening-style baseline setting",
        },
        {
            "parameter": "cores_per_tile",
            "value": params.cores_per_tile,
            "unit": "DPTC/tile",
            "evidence_label": "literature_anchored_assumption",
            "source": repo_path(lightening),
            "source_anchor": "Lightening architecture configuration describes two DPTC per LT-B tile",
            "claim_boundary": "matched Lightening-style baseline setting",
        },
        {
            "parameter": "frequency_ghz",
            "value": params.frequency_ghz,
            "unit": "GHz",
            "evidence_label": "literature_anchored_assumption",
            "source": repo_path(lightening),
            "source_anchor": "Lightening high-level architecture states DPTCs are clocked at 5 GHz",
            "claim_boundary": "architecture simulator frequency, not measured local hardware",
        },
        {
            "parameter": "sram_global_kib",
            "value": params.sram_global_kib,
            "unit": "KiB",
            "evidence_label": "literature_anchored_assumption",
            "source": repo_path(lightening),
            "source_anchor": "Lightening memory hierarchy describes 2 MB global SRAM",
            "claim_boundary": "architecture simulator parameter",
        },
        {
            "parameter": "sram_subarray_kib",
            "value": params.sram_subarray_kib,
            "unit": "KiB",
            "evidence_label": "literature_anchored_assumption",
            "source": repo_path(lightening),
            "source_anchor": "Lightening memory hierarchy follows 32 KB SRAM sub-arrays",
            "claim_boundary": "architecture simulator parameter",
        },
        {
            "parameter": "adc8_pj",
            "value": params.adc8_pj,
            "unit": "pJ/conversion",
            "evidence_label": "ngspice_or_fallback_macro_calibration",
            "source": repo_path(args.adc_json),
            "source_anchor": "median measured 8-bit ADC macro row",
            "claim_boundary": "macro calibration only; not PDK, extracted physical design, or hardware signoff",
        },
        {
            "parameter": "adc6_pj",
            "value": params.adc6_pj,
            "unit": "pJ/conversion",
            "evidence_label": "ngspice_or_fallback_macro_calibration",
            "source": repo_path(args.adc_json),
            "source_anchor": "median measured 6-bit ADC macro row",
            "claim_boundary": "macro calibration only",
        },
        {
            "parameter": "adc4_pj",
            "value": params.adc4_pj,
            "unit": "pJ/conversion",
            "evidence_label": "ngspice_or_fallback_macro_calibration",
            "source": repo_path(args.adc_json),
            "source_anchor": "median measured 4-bit ADC macro row",
            "claim_boundary": "macro calibration only",
        },
        {
            "parameter": "dac_pj",
            "value": params.dac_pj,
            "unit": "pJ/conversion",
            "evidence_label": "literature_scaled_assumption",
            "source": repo_path(lightening),
            "source_anchor": "Lightening Table III and bit-width/frequency power scaling statement",
            "claim_boundary": "architecture-level conversion accounting",
        },
        {
            "parameter": "hyatten_low_resolution_fraction",
            "value": 0.85,
            "unit": "fraction",
            "evidence_label": "literature_anchored_assumption",
            "source": repo_path(hyatten),
            "source_anchor": "HyAtten reports over 85 percent of analog signals can use low-resolution converters",
            "claim_boundary": "matched selector baseline, not local HyAtten implementation",
        },
        {
            "parameter": "control_pj_per_sideband_group",
            "value": params.control_pj_per_sideband_group,
            "unit": "pJ/group",
            "evidence_label": "rtl_synthesis_proxy",
            "source": repo_path(args.rtl_json),
            "source_anchor": "1 GHz, 32-column sideband control row",
            "claim_boundary": "Yosys/proxy sideband accounting; not liberty-backed timing or P&R",
        },
        {
            "parameter": "phy_nominal_pass_ratio",
            "value": params.phy_nominal_pass_ratio,
            "unit": "fraction",
            "evidence_label": "parametric_boundary",
            "source": repo_path(args.phy_json),
            "source_anchor": "PHY boundary sweep pass/fail summary",
            "claim_boundary": "parametric optical-link boundary only",
        },
        {
            "parameter": "phy_pessimistic_laser_multiplier",
            "value": params.phy_pessimistic_laser_multiplier,
            "unit": "x",
            "evidence_label": "parametric_boundary",
            "source": repo_path(astra),
            "source_anchor": "ASTRA link/power discussion and local PHY boundary sweep stress case",
            "claim_boundary": "sensitivity setting, not optical-device closure",
        },
        {
            "parameter": "lightening_adc_temporal_factor",
            "value": params.lightening_adc_temporal_factor,
            "unit": "x ADC energy",
            "evidence_label": "literature_anchored_assumption",
            "source": repo_path(lightening),
            "source_anchor": "Lightening reports temporal accumulation before ADC with about 6x ADC cost reduction",
            "claim_boundary": "matched architecture baseline feature",
        },
        {
            "parameter": "tempo_time_multiplexing_boundary",
            "value": 1.0,
            "unit": "modeled boundary row enabled",
            "evidence_label": "literature_boundary",
            "source": repo_path(tempo),
            "source_anchor": "TeMPO uses time-multiplexed dynamic photonic tensor cores and hierarchical temporal integration",
            "claim_boundary": "boundary baseline only; not a local TeMPO implementation",
        },
        {
            "parameter": "astra_stochastic_boundary",
            "value": 1.0,
            "unit": "modeled boundary row enabled",
            "evidence_label": "literature_boundary",
            "source": repo_path(astra),
            "source_anchor": "ASTRA uses stochastic signed optical multipliers, homodyne VDPEs, and temporal analog accumulation",
            "claim_boundary": "boundary baseline only; stochastic fabric is outside the selected SUDS DPTC fabric",
        },
        {
            "parameter": "enlighten_l1_boundary",
            "value": 1.0,
            "unit": "selector boundary enabled",
            "evidence_label": "literature_boundary",
            "source": repo_path(enlighten),
            "source_anchor": "ENLighten uses PTC-column sparsity, L1-style selection, densification, and adaptive operating granularity",
            "claim_boundary": "used to explain local selector boundary results, not as a claimed SUDS architecture",
        },
        {
            "parameter": "selected_sideband_group_cols",
            "value": SELECTED_OPERATING_POINT["sideband_group_cols"],
            "unit": "columns/group",
            "evidence_label": "selected_design_point",
            "source": repo_path(args.rtl_json),
            "source_anchor": "32-column sideband control row is the RTL/proxy calibration anchor",
            "claim_boundary": "selected operating point for architecture simulation",
        },
        {
            "parameter": "selected_adc_sharing",
            "value": SELECTED_OPERATING_POINT["adc_sharing"],
            "unit": "mode",
            "evidence_label": "selected_design_point",
            "source": repo_path(lightening),
            "source_anchor": "output-stationary DPTC temporal accumulation before ADC",
            "claim_boundary": "selected operating point for architecture simulation",
        },
    ]
    for row in rows:
        row["tag"] = args.tag
    return rows


def generate_bert_ops(seq_len: int = 128) -> list[dict[str, Any]]:
    hidden = 768
    heads = 12
    head_dim = hidden // heads
    intermediate = 3072
    ops: list[dict[str, Any]] = []
    for layer in range(12):
        prefix = f"bert.encoder.layer.{layer}"
        ops.extend(
            [
                {
                    "name": f"{prefix}.attention.self.qkv",
                    "type": "linear",
                    "m": seq_len,
                    "d": hidden,
                    "n": 3 * hidden,
                    "kernel_class": "mha_qkv_projection",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.attention.self.qk_scores",
                    "type": "linear",
                    "m": heads * seq_len,
                    "d": head_dim,
                    "n": seq_len,
                    "kernel_class": "mha_qk_scores",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.attention.self.av_context",
                    "type": "linear",
                    "m": heads * seq_len,
                    "d": seq_len,
                    "n": head_dim,
                    "kernel_class": "mha_av_context",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.attention.output.dense",
                    "type": "linear",
                    "m": seq_len,
                    "d": hidden,
                    "n": hidden,
                    "kernel_class": "mha_output_projection",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.intermediate.dense",
                    "type": "linear",
                    "m": seq_len,
                    "d": hidden,
                    "n": intermediate,
                    "kernel_class": "ffn_expand",
                    "layer_index": layer,
                },
                {
                    "name": f"{prefix}.output.dense",
                    "type": "linear",
                    "m": seq_len,
                    "d": intermediate,
                    "n": hidden,
                    "kernel_class": "ffn_project",
                    "layer_index": layer,
                },
            ]
        )
    return ops


def load_mobilevit_transformer_ops(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    ops: list[dict[str, Any]] = []
    for op in payload.get("ops", []):
        name = str(op.get("name", ""))
        op_type = str(op.get("type", ""))
        if "global_rep" not in name:
            continue
        if op_type != "linear":
            continue
        kernel_class = "transformer_linear"
        if ".pre_norm_mha." in name:
            if name.endswith(".qkv_proj"):
                kernel_class = "mha_qkv_projection"
            elif name.endswith(".attn_scores"):
                kernel_class = "mha_qk_scores"
            elif name.endswith(".attn_output"):
                kernel_class = "mha_av_context"
            elif name.endswith(".out_proj"):
                kernel_class = "mha_output_projection"
        elif ".pre_norm_ffn." in name:
            if name.endswith(".1"):
                kernel_class = "ffn_expand"
            elif name.endswith(".4"):
                kernel_class = "ffn_project"
        layer_index = extract_mobilevit_layer_index(name)
        ops.append(
            {
                "name": name,
                "type": "linear",
                "m": int(op["m"]),
                "d": int(op["d"]),
                "n": int(op["n"]),
                "kernel_class": kernel_class,
                "layer_index": layer_index,
            }
        )
    if not ops:
        raise SystemExit(f"no MobileViT transformer linear ops found in {path}")
    return ops


def extract_mobilevit_layer_index(name: str) -> int:
    # Stable stage/block order for scheduler slack.  Examples:
    # layer_3.1.global_rep.0..., layer_4.1.global_rep.3...
    try:
        stage = int(name.split("layer_", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        stage = 0
    try:
        block = int(name.split("global_rep.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        block = 0
    return stage * 10 + block


def workload_defs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    mobilevit_ops = load_mobilevit_transformer_ops(args.mobilevit_ops)
    return {
        "bert_base_glue_seq128": {
            "workload": "bert_base_glue_seq128",
            "workload_family": "NLP Transformer encoder",
            "model": "bert_base",
            "dataset_or_split": "GLUE validation, seq_len=128",
            "ops": generate_bert_ops(seq_len=128),
            "accuracy_source": repo_path(args.glue_json),
            "architecture_source": "generated canonical BERT-base encoder GEMM schedule",
        },
        "mobilevit_s_transformer_blocks_256": {
            "workload": "mobilevit_s_transformer_blocks_256",
            "workload_family": "vision Transformer-family blocks",
            "model": "mobilevit_s",
            "dataset_or_split": "ImageNet validation, MobileViT transformer blocks only",
            "ops": mobilevit_ops,
            "accuracy_source": repo_path(args.mobilevit_json),
            "architecture_source": repo_path(args.mobilevit_ops),
        },
    }


def source_profile_rows(mobilevit: dict[str, Any], glue: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    profiles: dict[str, dict[str, dict[str, Any]]] = {
        "mobilevit_s_transformer_blocks_256": {},
        "bert_base_glue_seq128": {},
    }

    mobile_rows = [
        row for row in mobilevit.get("rows", [])
        if row.get("row_type") == "per_seed" and row.get("model") == "mobilevit_s"
    ]
    for condition in sorted({str(row.get("condition")) for row in mobile_rows}):
        items = [row for row in mobile_rows if row.get("condition") == condition]
        if not items:
            continue
        profiles["mobilevit_s_transformer_blocks_256"][condition] = profile_from_items(
            items,
            accuracy_metric="top1",
            delta_metric="delta_top1",
            evidence_label="measured_mps_imagenet",
            promotion_decision="appendix",
        )

    glue_rows = [
        row for row in glue.get("per_seed", [])
        if row.get("split") in {"validation", "validation_matched"}
    ]
    for condition in sorted({str(row.get("condition")) for row in glue_rows}):
        items = [row for row in glue_rows if row.get("condition") == condition]
        if not items:
            continue
        profiles["bert_base_glue_seq128"][condition] = profile_from_items(
            items,
            accuracy_metric="primary_metric",
            delta_metric="delta_primary_metric",
            evidence_label="measured_mps_glue",
            promotion_decision="appendix",
        )

    return profiles


def profile_from_items(
    items: list[dict[str, Any]],
    *,
    accuracy_metric: str,
    delta_metric: str,
    evidence_label: str,
    promotion_decision: str,
) -> dict[str, Any]:
    def mean(key: str, default: float = 0.0) -> float:
        vals = []
        for item in items:
            try:
                vals.append(float(item.get(key)))
            except (TypeError, ValueError):
                pass
        return sum(vals) / len(vals) if vals else default

    devices = sorted({str(item.get("device", "")) for item in items if item.get("device")})
    git_hashes = sorted({str(item.get("git_hash", "")) for item in items if item.get("git_hash")})
    return {
        "keep_ratio": mean("mapped_keep_ratio", 1.0),
        "degrade_ratio": mean("mapped_degrade_ratio", 0.0),
        "prune_ratio": mean("mapped_prune_ratio", 0.0),
        "adc_energy_ratio_vs_e0": mean("adc_energy_ratio_vs_e0", 1.0),
        "accuracy_metric": accuracy_metric,
        "accuracy": mean(accuracy_metric, math.nan),
        "delta_accuracy": mean(delta_metric, math.nan),
        "accuracy_evidence_label": evidence_label,
        "promotion_decision": promotion_decision,
        "device": ",".join(devices),
        "git_hash": ",".join(git_hashes),
        "n_rows": len(items),
    }


def condition_profile(
    workload: str,
    condition: str,
    profiles: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    source_condition = SOURCE_CONDITIONS.get(condition)
    if source_condition:
        row = profiles.get(workload, {}).get(source_condition)
        if row:
            out = dict(row)
            out["source_condition"] = source_condition
            return out

    if condition == "random":
        dense = profiles.get(workload, {}).get("e0_dense", {})
        l1 = profiles.get(workload, {}).get("e2_l1", {})
        prune_ratio = float(l1.get("prune_ratio", 0.30))
        return {
            "keep_ratio": max(0.0, 1.0 - prune_ratio),
            "degrade_ratio": 0.0,
            "prune_ratio": prune_ratio,
            "adc_energy_ratio_vs_e0": max(0.0, 1.0 - prune_ratio),
            "accuracy_metric": dense.get("accuracy_metric", ""),
            "accuracy": math.nan,
            "delta_accuracy": math.nan,
            "accuracy_evidence_label": "unmeasured_random_architecture_boundary",
            "promotion_decision": "boundary",
            "device": dense.get("device", ""),
            "git_hash": dense.get("git_hash", ""),
            "n_rows": 0,
            "source_condition": "fixed_binary_sparsity_random_architecture_only",
        }

    if condition == "signal_only":
        dense = profiles.get(workload, {}).get("e0_dense", {})
        signal = profiles.get(workload, {}).get("e6_signal") or profiles.get(workload, {}).get("e8_overflow")
        if signal:
            return {
                "keep_ratio": signal["keep_ratio"],
                "degrade_ratio": signal["degrade_ratio"],
                "prune_ratio": signal["prune_ratio"],
                "adc_energy_ratio_vs_e0": signal["adc_energy_ratio_vs_e0"],
                "accuracy_metric": dense.get("accuracy_metric", signal.get("accuracy_metric", "")),
                "accuracy": signal.get("accuracy", math.nan) if signal.get("accuracy_evidence_label") != "measured_mps_glue" else math.nan,
                "delta_accuracy": signal.get("delta_accuracy", math.nan) if signal.get("accuracy_evidence_label") != "measured_mps_glue" else math.nan,
                "accuracy_evidence_label": (
                    signal.get("accuracy_evidence_label", "unmeasured_signal_architecture_boundary")
                    if "mobilevit" in workload
                    else "unmeasured_signal_architecture_boundary"
                ),
                "promotion_decision": "appendix" if "mobilevit" in workload else "boundary",
                "device": dense.get("device", ""),
                "git_hash": dense.get("git_hash", ""),
                "n_rows": signal.get("n_rows", 0) if "mobilevit" in workload else 0,
                "source_condition": signal.get("source_condition", "signal_architecture_only"),
            }

    if condition == "uniform_4bit":
        dense = profiles.get(workload, {}).get("e0_dense", {})
        return {
            "keep_ratio": 0.0,
            "degrade_ratio": 1.0,
            "prune_ratio": 0.0,
            "adc_energy_ratio_vs_e0": 1.0 / 16.0,
            "accuracy_metric": dense.get("accuracy_metric", ""),
            "accuracy": math.nan,
            "delta_accuracy": math.nan,
            "accuracy_evidence_label": "unmeasured_accuracy_boundary",
            "promotion_decision": "boundary",
            "device": dense.get("device", ""),
            "git_hash": dense.get("git_hash", ""),
            "n_rows": 0,
            "source_condition": "uniform_4bit_architecture_only",
        }

    if condition == "hyatten_style":
        dense = profiles.get(workload, {}).get("e0_dense", {})
        return {
            "keep_ratio": 0.15,
            "degrade_ratio": 0.85,
            "prune_ratio": 0.0,
            "adc_energy_ratio_vs_e0": 0.15 + 0.85 / 16.0,
            "accuracy_metric": dense.get("accuracy_metric", ""),
            "accuracy": math.nan,
            "delta_accuracy": math.nan,
            "accuracy_evidence_label": "literature_baseline_unmeasured_locally",
            "promotion_decision": "boundary",
            "device": dense.get("device", ""),
            "git_hash": dense.get("git_hash", ""),
            "n_rows": 0,
            "source_condition": "hyatten_low_resolution_fraction",
        }

    if condition in {"tempo_time_multiplexed", "astra_boundary"}:
        dense = profiles.get(workload, {}).get("e0_dense", {})
        return {
            "keep_ratio": 1.0,
            "degrade_ratio": 0.0,
            "prune_ratio": 0.0,
            "adc_energy_ratio_vs_e0": 1.0,
            "accuracy_metric": dense.get("accuracy_metric", ""),
            "accuracy": math.nan,
            "delta_accuracy": math.nan,
            "accuracy_evidence_label": "literature_architecture_boundary_unmeasured_locally",
            "promotion_decision": "boundary",
            "device": dense.get("device", ""),
            "git_hash": dense.get("git_hash", ""),
            "n_rows": 0,
            "source_condition": f"{condition}_literature_boundary",
        }

    raise SystemExit(f"cannot derive profile for workload={workload} condition={condition}")


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def adc_temporal_factor(params: ArchitectureParams, adc_sharing_mode: str) -> float:
    if adc_sharing_mode == "per_array":
        return 1.0
    if adc_sharing_mode == "per_tile":
        return max(params.lightening_adc_temporal_factor, 1.0 / max(1, params.cores_per_tile))
    if adc_sharing_mode == "temporal_accum":
        return params.lightening_adc_temporal_factor
    raise SystemExit(f"unsupported adc_sharing mode: {adc_sharing_mode}")


def params_for_design(
    params: ArchitectureParams,
    *,
    tile_dim: int,
    tiles: int,
    cores_per_tile: int,
    sideband_group_cols: int,
) -> ArchitectureParams:
    return replace(
        params,
        tile_dim=tile_dim,
        tiles=tiles,
        cores_per_tile=cores_per_tile,
        sideband_columns=sideband_group_cols,
    )


def is_selected_operating_point(row: dict[str, Any]) -> bool:
    return (
        int(row.get("tile_dim", -1)) == SELECTED_OPERATING_POINT["tile_dim"]
        and int(row.get("tiles", -1)) == SELECTED_OPERATING_POINT["tiles"]
        and int(row.get("cores_per_tile", -1)) == SELECTED_OPERATING_POINT["cores_per_tile"]
        and int(row.get("sideband_group_cols", -1)) == SELECTED_OPERATING_POINT["sideband_group_cols"]
        and str(row.get("adc_sharing_mode", "")) == SELECTED_OPERATING_POINT["adc_sharing"]
    )


def schedule_ops(
    workload: str,
    workload_meta: dict[str, Any],
    params: ArchitectureParams,
) -> list[dict[str, Any]]:
    ops = workload_meta["ops"]
    max_layer = max(int(op["layer_index"]) for op in ops) if ops else 0
    rows = []
    running_ns = 0.0
    for index, op in enumerate(ops):
        m, d, n = int(op["m"]), int(op["d"]), int(op["n"])
        tile_m = ceil_div(m, params.tile_dim)
        tile_d = ceil_div(d, params.tile_dim)
        tile_n = ceil_div(n, params.tile_dim)
        dptc_tiles = tile_m * tile_d * tile_n
        output_groups = tile_m * tile_n
        output_values = output_groups * params.tile_dim * params.tile_dim
        operand_values = tile_m * tile_d * params.tile_dim * params.tile_dim
        operand_values += tile_d * tile_n * params.tile_dim * params.tile_dim
        cycles = ceil_div(dptc_tiles, params.parallel_cores)
        latency_ns = cycles * params.cycle_ns
        layer_norm = 0.0 if max_layer <= 0 else int(op["layer_index"]) / max_layer
        # This is a scheduler-derived slack proxy: each kernel has a concrete
        # issue time and a downstream stage deadline.  It preserves the serial
        # DPTC column gradient while adding a layer/stage component.
        slack_window_ns = max(1.0, latency_ns * 1.35 + params.cycle_ns * params.tile_dim)
        deadline_ns = running_ns + latency_ns + slack_window_ns
        median_arrival_ns = running_ns + latency_ns * 0.5
        scheduler_slack_norm = max(0.0, min(1.0, (deadline_ns - median_arrival_ns) / (slack_window_ns + latency_ns)))
        scheduler_slack_norm = max(0.0, min(1.0, 0.35 * layer_norm + 0.65 * scheduler_slack_norm))
        rows.append(
            {
                "tag": TAG,
                "workload": workload,
                "model": workload_meta["model"],
                "dataset_or_split": workload_meta["dataset_or_split"],
                "kernel_id": f"{workload}:k{index:03d}",
                "kernel_index": index,
                "kernel_name": op["name"],
                "kernel_class": op["kernel_class"],
                "layer_index": int(op["layer_index"]),
                "m": m,
                "d": d,
                "n": n,
                "macs": m * d * n,
                "tile_m": tile_m,
                "tile_d": tile_d,
                "tile_n": tile_n,
                "dptc_tiles": dptc_tiles,
                "output_groups": output_groups,
                "output_values": output_values,
                "operand_values": operand_values,
                "dptc_cycles": cycles,
                "base_latency_ns": latency_ns,
                "schedule_start_ns": running_ns,
                "schedule_deadline_ns": deadline_ns,
                "scheduler_slack_norm": scheduler_slack_norm,
                "slack_source": "dptc_photonic_tile_schedule",
                "mapping_evidence": workload_meta["architecture_source"],
            }
        )
        running_ns += latency_ns
    return rows


def simulate_condition(
    schedule: list[dict[str, Any]],
    workload_meta: dict[str, Any],
    workload: str,
    condition: str,
    profile: dict[str, Any],
    params: ArchitectureParams,
    *,
    sensitivity_case: str,
    adc_sharing_mode: str = "temporal_accum",
) -> dict[str, Any]:
    scales = sensitivity_scales(sensitivity_case)
    keep = float(profile["keep_ratio"])
    degrade = float(profile["degrade_ratio"])
    prune = float(profile["prune_ratio"])
    active = max(0.0, min(1.0, keep + degrade))

    if condition in {"uniform_8bit", "lightening_dptc"}:
        keep, degrade, prune, active = 1.0, 0.0, 0.0, 1.0
    elif condition == "uniform_4bit":
        keep, degrade, prune, active = 0.0, 1.0, 0.0, 1.0
    elif condition == "hyatten_style":
        keep, degrade, prune, active = 0.15, 0.85, 0.0, 1.0
    elif condition in {"tempo_time_multiplexed", "astra_boundary"}:
        keep, degrade, prune, active = 1.0, 0.0, 0.0, 1.0

    temporal_factor = adc_temporal_factor(params, adc_sharing_mode)
    if condition == "uniform_8bit":
        temporal_factor = 1.0

    total_macs = sum(int(row["macs"]) for row in schedule)
    total_tiles = sum(int(row["dptc_tiles"]) for row in schedule)
    total_output_values = sum(int(row["output_values"]) for row in schedule)
    total_operand_values = sum(int(row["operand_values"]) for row in schedule)
    base_latency_ns = sum(float(row["base_latency_ns"]) for row in schedule)
    total_output_groups = sum(int(row["output_groups"]) for row in schedule)
    bytes_moved = sum((int(row["m"]) * int(row["d"]) + int(row["d"]) * int(row["n"]) + int(row["m"]) * int(row["n"])) * 2 for row in schedule)

    prune_relief = scales["prune_compute_relief"]
    compute_active = 1.0 - prune * prune_relief
    if condition == "hyatten_style":
        compute_active = 0.85
    if condition == "tempo_time_multiplexed":
        compute_active = 0.95
    if condition == "astra_boundary":
        compute_active = 0.90
    if condition in {"uniform_8bit", "uniform_4bit", "lightening_dptc"}:
        compute_active = 1.0

    adc_pj = (keep * params.adc8_pj + degrade * params.adc4_pj) * total_output_values * temporal_factor
    if condition == "hyatten_style":
        adc_pj = 0.85 * params.adc4_pj * total_output_values * 0.5
    elif condition == "tempo_time_multiplexed":
        adc_pj *= 0.78
    elif condition == "astra_boundary":
        adc_pj *= 0.42
    adc_pj *= scales["adc_scale"]

    dac_mzm_pj = total_operand_values * (params.dac_pj + params.mzm_pj) * compute_active
    detector_pj = total_output_values * params.detector_tia_pj * active
    optical_link_pj = total_output_values * params.optical_link_pj_per_output * active * scales["optical_link_scale"]
    laser_pj = total_tiles * params.laser_pj_per_active_tile * compute_active * scales["laser_scale"]
    memory_pj = bytes_moved * params.memory_pj_per_byte * (0.72 + 0.28 * compute_active) * scales["memory_scale"]
    if condition == "tempo_time_multiplexed":
        dac_mzm_pj *= 0.72
        detector_pj *= 0.82
        optical_link_pj *= 0.88
        laser_pj *= 0.86
        memory_pj *= 0.95
    elif condition == "astra_boundary":
        dac_mzm_pj *= 0.30
        detector_pj *= 0.85
        optical_link_pj *= 0.82
        laser_pj *= 0.75
        memory_pj *= 0.93

    control_groups = ceil_div(total_output_groups * params.tile_dim, max(1, params.sideband_columns))
    control_multiplier = 1.0 if condition in SUDS_CONDITIONS or condition in {"slack_only", "hyatten_style"} else 0.25
    if condition in {"tempo_time_multiplexed", "astra_boundary"}:
        control_multiplier = 0.5
    control_pj = (
        control_groups
        * params.control_pj_per_sideband_group
        * control_multiplier
        * scales["control_scale"]
    )

    digital_fallback_pj = 0.0
    if condition == "hyatten_style":
        digital_fallback_pj = total_macs * 0.15 * params.digital_fallback_pj_per_mac * scales["digital_scale"]
    elif condition == "astra_boundary":
        digital_fallback_pj = total_macs * 0.04 * params.digital_fallback_pj_per_mac * scales["digital_scale"]

    energy_pj = (
        adc_pj
        + dac_mzm_pj
        + detector_pj
        + optical_link_pj
        + laser_pj
        + memory_pj
        + control_pj
        + digital_fallback_pj
    )

    conversion_parallelism = 1.0
    if condition == "hyatten_style":
        conversion_parallelism = 2.8
    latency_ns = base_latency_ns * (0.80 + 0.20 * compute_active)
    latency_ns += (adc_pj / max(1.0, total_output_values * params.adc8_pj)) * params.cycle_ns / conversion_parallelism
    latency_ns += control_groups * params.cycle_ns * control_multiplier * 0.002
    if condition == "tempo_time_multiplexed":
        latency_ns *= 1.18
    elif condition == "astra_boundary":
        latency_ns *= 1.34
    if sensitivity_case == "pessimistic":
        latency_ns *= 1.08

    area_mm2 = 0.0
    tile_area_scale = (params.tile_dim / 32.0) ** 1.8
    dptc_area = params.parallel_cores * 0.145 * tile_area_scale
    adc_area = (
        (keep * 1.0 + degrade * 0.25)
        * total_output_groups
        / max(1, len(schedule))
        * 0.00018
        * (1.0 / adc_temporal_factor(params, adc_sharing_mode)) ** 0.25
    )
    if condition == "hyatten_style":
        adc_area = 0.85 * total_output_groups / max(1, len(schedule)) * 0.00012
    elif condition == "tempo_time_multiplexed":
        dptc_area *= 0.78
        adc_area *= 0.70
    elif condition == "astra_boundary":
        dptc_area *= 0.82
        adc_area *= 0.55
    control_area = control_groups * params.rtl_area_ge_per_group * 1.0e-6 * control_multiplier
    memory_area = params.sram_global_kib * 0.00018
    area_mm2 = dptc_area + adc_area + control_area + memory_area

    row = {
        "tag": TAG,
        "sensitivity_case": sensitivity_case,
        "workload": workload,
        "workload_family": workload_meta["workload_family"],
        "model": workload_meta["model"],
        "dataset_or_split": workload_meta["dataset_or_split"],
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "condition_family": condition_family(condition),
        "tile_dim": params.tile_dim,
        "tiles": params.tiles,
        "cores_per_tile": params.cores_per_tile,
        "parallel_cores": params.parallel_cores,
        "sideband_group_cols": params.sideband_columns,
        "adc_sharing_mode": adc_sharing_mode,
        "source_condition": profile.get("source_condition", ""),
        "n_kernels": len(schedule),
        "macs": total_macs,
        "dptc_tiles": total_tiles,
        "output_values": total_output_values,
        "memory_moved_bytes": bytes_moved,
        "control_groups": control_groups,
        "keep_ratio": keep,
        "degrade_ratio": degrade,
        "prune_ratio": prune,
        "active_compute_ratio": compute_active,
        "latency_ns": latency_ns,
        "energy_pj": energy_pj,
        "edp_pj_ns": energy_pj * latency_ns,
        "area_mm2": area_mm2,
        "adc_energy_pj": adc_pj,
        "dac_mzm_energy_pj": dac_mzm_pj,
        "detector_tia_energy_pj": detector_pj,
        "laser_energy_pj": laser_pj,
        "memory_energy_pj": memory_pj,
        "optical_link_energy_pj": optical_link_pj,
        "control_energy_pj": control_pj,
        "optical_link_pj": optical_link_pj,
        "control_overhead_pj": control_pj,
        "digital_fallback_energy_pj": digital_fallback_pj,
        "accuracy_metric": profile.get("accuracy_metric", ""),
        "accuracy": profile.get("accuracy", math.nan),
        "delta_accuracy": profile.get("delta_accuracy", math.nan),
        "accuracy_evidence_label": profile.get("accuracy_evidence_label", ""),
        "promotion_decision": profile.get("promotion_decision", ""),
        "device": profile.get("device", ""),
        "git_hash": profile.get("git_hash", ""),
        "architecture_evidence_label": "modeled_system_ppa",
        "claim_boundary": (
            "DPTC architecture simulation with calibrated/proxy parameters; "
            "not fabrication, physical-design, PDK, or device-solver signoff"
        ),
    }
    return row


def sensitivity_scales(case: str) -> dict[str, float]:
    if case == "nominal":
        return {
            "adc_scale": 1.0,
            "memory_scale": 1.0,
            "optical_link_scale": 1.0,
            "laser_scale": 1.0,
            "control_scale": 1.0,
            "digital_scale": 1.0,
            "prune_compute_relief": 1.0,
        }
    if case == "pessimistic":
        return {
            "adc_scale": 2.0,
            "memory_scale": 1.25,
            "optical_link_scale": 1.15,
            "laser_scale": 1.15,
            "control_scale": 3.0,
            "digital_scale": 1.4,
            "prune_compute_relief": 0.75,
        }
    raise SystemExit(f"unsupported sensitivity case: {case}")


def condition_family(condition: str) -> str:
    if condition in SUDS_CONDITIONS:
        return "suds"
    if condition == "hyatten_style":
        return "hyatten_style"
    if condition == "tempo_time_multiplexed":
        return "tempo_boundary"
    if condition == "astra_boundary":
        return "astra_boundary"
    if condition == "lightening_dptc":
        return "lightening_style"
    if condition.startswith("uniform"):
        return "uniform"
    return "selector_ablation"


def normalize_rows(rows: list[dict[str, Any]]) -> None:
    by_workload_case: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["workload"]),
            str(row["sensitivity_case"]),
            str(row.get("design_id", "reference")),
        )
        if row["condition"] == "lightening_dptc":
            by_workload_case[key] = row
    for row in rows:
        baseline = by_workload_case.get(
            (
                str(row["workload"]),
                str(row["sensitivity_case"]),
                str(row.get("design_id", "reference")),
            )
        )
        if not baseline:
            row["energy_ratio_vs_lightening"] = math.nan
            row["latency_ratio_vs_lightening"] = math.nan
            row["edp_ratio_vs_lightening"] = math.nan
            row["energy_improvement_vs_lightening_pct"] = math.nan
            row["edp_improvement_vs_lightening_pct"] = math.nan
            continue
        row["energy_ratio_vs_lightening"] = float(row["energy_pj"]) / max(1.0, float(baseline["energy_pj"]))
        row["latency_ratio_vs_lightening"] = float(row["latency_ns"]) / max(1.0e-12, float(baseline["latency_ns"]))
        row["edp_ratio_vs_lightening"] = float(row["edp_pj_ns"]) / max(1.0, float(baseline["edp_pj_ns"]))
        row["energy_improvement_vs_lightening_pct"] = (1.0 - row["energy_ratio_vs_lightening"]) * 100.0
        row["edp_improvement_vs_lightening_pct"] = (1.0 - row["edp_ratio_vs_lightening"]) * 100.0


def build_kernel_condition_rows(
    schedules: dict[str, list[dict[str, Any]]],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_lookup = {
        (row["workload"], row["condition"], row["sensitivity_case"]): row
        for row in summary_rows
    }
    rows = []
    for workload, schedule in schedules.items():
        for condition in CONDITION_LABELS:
            summary = summary_lookup[(workload, condition, "nominal")]
            active = float(summary["active_compute_ratio"])
            keep = float(summary["keep_ratio"])
            degrade = float(summary["degrade_ratio"])
            prune = float(summary["prune_ratio"])
            for kernel in schedule:
                out = dict(kernel)
                out.update(
                    {
                        "condition": condition,
                        "condition_label": CONDITION_LABELS[condition],
                        "keep_ratio": keep,
                        "degrade_ratio": degrade,
                        "prune_ratio": prune,
                        "active_compute_ratio": active,
                        "condition_latency_ns": float(kernel["base_latency_ns"]) * (0.80 + 0.20 * active),
                        "condition_dptc_tiles": int(round(int(kernel["dptc_tiles"]) * active)),
                        "condition_output_values": int(round(int(kernel["output_values"]) * active)),
                    }
                )
                rows.append(out)
    return rows


def build_glue_link_rows(
    glue: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_lookup = {
        row["condition"]: row for row in summary_rows
        if row["workload"] == "bert_base_glue_seq128" and row["sensitivity_case"] == "nominal"
    }
    condition_map = {
        "e0_dense": "lightening_dptc",
        "e2_l1": "l1",
        "e3_slack": "slack_only",
        "e4_suds": "suds_only",
        "e7_overlay": "suds_l1",
        "e8_overflow": "suds_signal",
    }
    rows = []
    for row in glue.get("per_seed", []):
        mapped = condition_map.get(str(row.get("condition", "")))
        if not mapped:
            continue
        arch = summary_lookup[mapped]
        rows.append(
            {
                "tag": TAG,
                "task": row.get("task", ""),
                "split": row.get("split", ""),
                "seed": row.get("seed", ""),
                "condition": row.get("condition", ""),
                "architecture_condition": mapped,
                "device": row.get("device", ""),
                "git_hash": row.get("git_hash", ""),
                "command": row.get("command", ""),
                "primary_metric_name": row.get("primary_metric_name", ""),
                "primary_metric": row.get("primary_metric", ""),
                "delta_primary_metric": row.get("delta_primary_metric", ""),
                "original_slack_source": row.get("slack_source", ""),
                "linked_schedule_source": "dptc_photonic_tile_schedule",
                "architecture_summary_artifact": repo_path(SUMMARY_CSV),
                "architecture_json_artifact": repo_path(JSON_OUT),
                "architecture_workload": "bert_base_glue_seq128",
                "architecture_energy_pj": arch["energy_pj"],
                "architecture_latency_ns": arch["latency_ns"],
                "architecture_edp_pj_ns": arch["edp_pj_ns"],
                "profile_link_status": "pass",
                "profile_link_note": (
                    "Existing governed MPS GLUE row is linked to the hardware-derived "
                    "BERT DPTC schedule and matching architecture condition. The accuracy "
                    "measurement remains measured; energy remains architecture-modeled."
                ),
            }
        )
    return rows


def build_design_space_rows(
    workloads: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, dict[str, Any]]],
    params: ArchitectureParams,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tile_dim in DESIGN_SPACE_SWEEP["tile_dim"]:
        for tiles in DESIGN_SPACE_SWEEP["tiles"]:
            for cores_per_tile in DESIGN_SPACE_SWEEP["cores_per_tile"]:
                for sideband_cols in DESIGN_SPACE_SWEEP["sideband_group_cols"]:
                    design_params = params_for_design(
                        params,
                        tile_dim=int(tile_dim),
                        tiles=int(tiles),
                        cores_per_tile=int(cores_per_tile),
                        sideband_group_cols=int(sideband_cols),
                    )
                    for adc_sharing in DESIGN_SPACE_SWEEP["adc_sharing"]:
                        design_id = (
                            f"td{tile_dim}_t{tiles}_c{cores_per_tile}_"
                            f"sg{sideband_cols}_{adc_sharing}"
                        )
                        for workload, meta in workloads.items():
                            schedule = schedule_ops(workload, meta, design_params)
                            for condition in DESIGN_SPACE_CONDITIONS:
                                profile = condition_profile(workload, condition, profiles)
                                row = simulate_condition(
                                    schedule,
                                    meta,
                                    workload,
                                    condition,
                                    profile,
                                    design_params,
                                    sensitivity_case="nominal",
                                    adc_sharing_mode=str(adc_sharing),
                                )
                                row.update(
                                    {
                                        "design_id": design_id,
                                        "design_space_scope": "tile_control_adc_sweep",
                                        "selected_operating_point": is_selected_operating_point(row),
                                        "selection_reason": selected_operating_point_reason(row),
                                        "pareto_front": False,
                                    }
                                )
                                rows.append(row)
    normalize_rows(rows)
    mark_pareto_front(rows)
    return rows


def selected_operating_point_reason(row: dict[str, Any]) -> str:
    if not is_selected_operating_point(row):
        return ""
    return (
        "32x32 tile preserves Lightening/HyAtten DPTC comparability; four tiles "
        "and two cores per tile match the LT-B reference; 32-column sideband "
        "uses the local RTL calibration anchor; temporal accumulation preserves "
        "the output-stationary DPTC conversion assumption."
    )


def mark_pareto_front(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["workload"]), str(row["condition"])), []).append(row)
    for group_rows in grouped.values():
        for row in group_rows:
            dominated = False
            energy = float(row["energy_pj"])
            latency = float(row["latency_ns"])
            area = float(row["area_mm2"])
            for other in group_rows:
                if other is row:
                    continue
                other_energy = float(other["energy_pj"])
                other_latency = float(other["latency_ns"])
                other_area = float(other["area_mm2"])
                no_worse = (
                    other_energy <= energy
                    and other_latency <= latency
                    and other_area <= area
                )
                strictly_better = (
                    other_energy < energy
                    or other_latency < latency
                    or other_area < area
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            row["pareto_front"] = not dominated


def build_decision(summary_rows: list[dict[str, Any]], parameter_rows_: list[dict[str, Any]]) -> dict[str, Any]:
    nominal = [row for row in summary_rows if row["sensitivity_case"] == "nominal"]
    pessimistic = [row for row in summary_rows if row["sensitivity_case"] == "pessimistic"]
    workloads = sorted({row["workload"] for row in nominal})
    conditions = sorted({row["condition"] for row in nominal})
    has_required_workloads = WORKLOADS_REQUIRED.issubset(set(workloads))
    has_baselines = set(CONDITION_LABELS).issubset(set(conditions))
    has_non_adc_terms = all(
        float(row["memory_energy_pj"]) > 0.0
        and float(row["optical_link_energy_pj"]) > 0.0
        and float(row["control_energy_pj"]) >= 0.0
        for row in nominal
    )
    has_calibration_ties = {"adc8_pj", "control_pj_per_sideband_group", "phy_nominal_pass_ratio"}.issubset(
        {str(row["parameter"]) for row in parameter_rows_}
    )

    by_workload: dict[str, dict[str, Any]] = {}
    for workload in workloads:
        case_rows = [row for row in pessimistic if row["workload"] == workload]
        suds_candidates = [row for row in case_rows if row["condition"] in MAIN_SUDS_CONDITIONS]
        best_suds = min(suds_candidates, key=lambda row: float(row["edp_pj_ns"]))
        lightening = next(row for row in case_rows if row["condition"] == "lightening_dptc")
        boundary_baselines = [row for row in case_rows if row["condition"] in SAME_SCOPE_BASELINES]
        stronger_boundary = [
            row["condition"] for row in boundary_baselines
            if float(row["edp_pj_ns"]) < float(best_suds["edp_pj_ns"])
        ]
        edp_improvement = (1.0 - float(best_suds["edp_pj_ns"]) / float(lightening["edp_pj_ns"])) * 100.0
        energy_improvement = (1.0 - float(best_suds["energy_pj"]) / float(lightening["energy_pj"])) * 100.0
        by_workload[workload] = {
            "best_suds_condition": best_suds["condition"],
            "reference_same_scope_baseline": "lightening_dptc",
            "stronger_boundary_conditions": stronger_boundary,
            "pessimistic_energy_improvement_pct": energy_improvement,
            "pessimistic_edp_improvement_pct": edp_improvement,
            "advantage_preserved": edp_improvement > 0.0,
        }

    advantage_preserved = all(row["advantage_preserved"] for row in by_workload.values())
    status = "pass" if (
        has_required_workloads
        and has_baselines
        and has_non_adc_terms
        and has_calibration_ties
        and advantage_preserved
    ) else "partial"
    blockers = []
    if not has_required_workloads:
        blockers.append("missing_required_transformer_workload")
    if not has_baselines:
        blockers.append("missing_required_baseline")
    if not has_non_adc_terms:
        blockers.append("system_cost_terms_incomplete")
    if not has_calibration_ties:
        blockers.append("parameter_calibration_ties_incomplete")
    if not advantage_preserved:
        blockers.append("pessimistic_advantage_not_preserved")
    return {
        "architecture_sim_status": status,
        "blockers": blockers,
        "workloads": workloads,
        "conditions": conditions,
        "system_cost_terms": [
            "adc",
            "dac_mzm",
            "detector_tia",
            "laser",
            "memory",
            "optical_link",
            "control_sideband",
            "digital_fallback",
        ],
        "pessimistic_advantage_by_workload": by_workload,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(val) for val in value]
    return value


def write_json(
    path: Path,
    *,
    tag: str,
    params: ArchitectureParams,
    decision: dict[str, Any],
    parameter_rows_: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    glue_link_rows: list[dict[str, Any]],
    design_space_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    selected_design_rows = [row for row in design_space_rows if row.get("selected_operating_point")]
    pareto_rows = [row for row in design_space_rows if row.get("pareto_front")]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_transformer_architecture_sim_{tag}",
            "evidence_label": "modeled_system_ppa",
            "promotion_decision": "architecture_evidence_ready"
            if decision["architecture_sim_status"] == "pass"
            else "architecture_evidence_partial",
            "git_hash": git_hash(),
            "claim_boundary_note": (
                "Architecture-level DPTC Transformer simulator. Energy, latency, area, memory, optical-link, "
                "and control values are modeled or calibrated/proxy values. The artifact does not claim "
                "fabrication, physical-design, device-solver, bench-energy, or deployment evidence."
            ),
            "regeneration_command": (
                ".venv311-mps/bin/python experiments/tools/build_suds_transformer_architecture_sim.py"
            ),
            "source_artifacts": {
                "mobilevit_json": repo_path(args.mobilevit_json),
                "glue_json": repo_path(args.glue_json),
                "adc_json": repo_path(args.adc_json),
                "rtl_json": repo_path(args.rtl_json),
                "phy_json": repo_path(args.phy_json),
                "mobilevit_ops": repo_path(args.mobilevit_ops),
            },
            "source_artifact_sha256": {
                "mobilevit_json": sha256_path(args.mobilevit_json),
                "glue_json": sha256_path(args.glue_json),
                "adc_json": sha256_path(args.adc_json),
                "rtl_json": sha256_path(args.rtl_json),
                "phy_json": sha256_path(args.phy_json),
                "mobilevit_ops": sha256_path(args.mobilevit_ops),
            },
        },
        "architecture_parameters": params.__dict__,
        "decision": decision,
        "design_space_summary": {
            "sweep": DESIGN_SPACE_SWEEP,
            "selected_operating_point": SELECTED_OPERATING_POINT,
            "design_space_rows": len(design_space_rows),
            "pareto_rows": len(pareto_rows),
            "selected_rows": len(selected_design_rows),
            "design_space_csv": repo_path(args.design_space_csv),
            "design_space_json": repo_path(args.design_space_json),
        },
        "selected_design_rows": selected_design_rows,
        "parameter_rows": parameter_rows_,
        "summary_rows": summary_rows,
        "kernel_rows": kernel_rows,
        "glue_link_rows": glue_link_rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_design_space_json(path: Path, *, tag: str, design_space_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_rows = [row for row in design_space_rows if row.get("selected_operating_point")]
    pareto_rows = [row for row in design_space_rows if row.get("pareto_front")]
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_transformer_architecture_design_space_{tag}",
            "evidence_label": "modeled_design_space_ppa",
            "promotion_decision": "architecture_design_space_ready",
            "sweep": DESIGN_SPACE_SWEEP,
            "selected_operating_point": SELECTED_OPERATING_POINT,
            "pareto_scope": "within each workload and condition over energy, latency, and area",
        },
        "summary": {
            "rows": len(design_space_rows),
            "pareto_rows": len(pareto_rows),
            "selected_rows": len(selected_rows),
            "conditions": list(DESIGN_SPACE_CONDITIONS),
        },
        "selected_rows": selected_rows,
        "pareto_rows": pareto_rows,
        "rows": design_space_rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 2) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(val):
        return "n/a"
    return f"{val:.{digits}f}"


def write_report(
    path: Path,
    *,
    tag: str,
    decision: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    parameter_rows_: list[dict[str, Any]],
    design_space_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nominal = [row for row in summary_rows if row["sensitivity_case"] == "nominal"]
    selected_design_rows = [
        row for row in design_space_rows
        if row.get("selected_operating_point") and row.get("condition") in {"suds_l1", "suds_signal", "lightening_dptc"}
    ]
    pareto_count = sum(1 for row in design_space_rows if row.get("pareto_front"))
    report = f"""# SUDS Transformer Architecture Simulator

Tag: `{tag}`
Evidence label: `modeled_system_ppa`
Promotion decision: `{'architecture_evidence_ready' if decision['architecture_sim_status'] == 'pass' else 'architecture_evidence_partial'}`

## Scope

This artifact maps BERT-base GLUE encoder kernels and MobileViT-S Transformer
blocks onto a Lightening-style DPTC optical tile array, then compares uniform,
selector, SUDS, Lightening-style, and HyAtten-style policies. It reports
architecture-modeled system PPA terms: conversion, DAC/MZM, detector/TIA,
laser, memory movement, optical link, sideband control, and digital fallback.

It is modeled architecture evidence. It does not claim fabrication,
physical-design, device-solver signoff, bench-energy, or deployment evidence.

## Readiness

- Architecture simulator status: `{decision['architecture_sim_status']}`
- Blockers: `{';'.join(decision['blockers']) or 'none'}`
- Workloads: `{','.join(decision['workloads'])}`
- Conditions: `{','.join(decision['conditions'])}`
- Design-space rows: `{len(design_space_rows)}`; Pareto rows: `{pareto_count}`

## Architecture Design Space and Selected Operating Point

Sweep dimensions: `tile_dim={{16,32,64}}`, `tiles={{2,4,8}}`,
`cores_per_tile={{1,2,4}}`, `sideband_group_cols={{16,32,64,128}}`,
and `adc_sharing={{per_array,per_tile,temporal_accum}}`.

| Dimension | Selected value | Rationale |
|---|---:|---|
| Tile dimension | 32 | Matches the Lightening/HyAtten DPTC comparison surface and avoids inventing a different array fabric for the main claim. |
| Tiles | 4 | Matches the LT-B-style reference and keeps inter-tile broadcast accounting comparable. |
| Cores per tile | 2 | Matches the LT-B-style reference while preserving a meaningful per-tile accumulation point. |
| Sideband group columns | 32 | Uses the local RTL sideband calibration anchor instead of extrapolating control cost from a different group size. |
| ADC sharing | temporal_accum | Preserves output-stationary DPTC temporal accumulation as the selected conversion-fabric point. |

Selected-point nominal rows:

| Workload | Condition | Energy ratio vs Lightening | EDP ratio vs Lightening | Area mm2 | Memory pJ | Optical-link pJ | Control pJ |
|---|---|---:|---:|---:|---:|---:|---:|
"""
    for row in sorted(selected_design_rows, key=lambda item: (item["workload"], item["condition"])):
        report += (
            f"| `{row['workload']}` | `{row['condition']}` | "
            f"{fmt(row['energy_ratio_vs_lightening'], 3)} | {fmt(row['edp_ratio_vs_lightening'], 3)} | "
            f"{fmt(row['area_mm2'], 3)} | {fmt(row['memory_energy_pj'], 1)} | "
            f"{fmt(row['optical_link_energy_pj'], 1)} | {fmt(row['control_energy_pj'], 1)} |\n"
        )

    report += """
Boundary rows are retained only as matched architecture context. TeMPO-style
time multiplexing and ASTRA-style stochastic optical rows can define alternate
conversion/readout fabrics, but they are not treated as the selected SUDS DPTC
fabric. Likewise, signal-only/L1/HyAtten wins are boundary evidence for a local
selector beating a scheduler-budgeted composition, not a reason to relabel
SUDS-only as the main method.

## Nominal PPA Summary

| Workload | Condition | Energy ratio vs Lightening | EDP ratio vs Lightening | Latency ns | Energy pJ | Accuracy evidence | Delta |
|---|---|---:|---:|---:|---:|---|---:|
"""
    for row in sorted(nominal, key=lambda item: (item["workload"], item["condition"])):
        report += (
            f"| `{row['workload']}` | {row['condition_label']} | "
            f"{fmt(row['energy_ratio_vs_lightening'], 3)} | {fmt(row['edp_ratio_vs_lightening'], 3)} | "
            f"{fmt(row['latency_ns'], 2)} | {fmt(row['energy_pj'], 1)} | "
            f"`{row['accuracy_evidence_label']}` | {fmt(row['delta_accuracy'], 2)} |\n"
        )

    report += """
## Pessimistic Gate

| Workload | Best SUDS | Reference baseline | Energy improvement | EDP improvement | Preserved | Boundary stronger conditions |
|---|---|---|---:|---:|---|
"""
    for workload, row in decision["pessimistic_advantage_by_workload"].items():
        report += (
            f"| `{workload}` | `{row['best_suds_condition']}` | `{row['reference_same_scope_baseline']}` | "
            f"{fmt(row['pessimistic_energy_improvement_pct'], 2)}% | "
            f"{fmt(row['pessimistic_edp_improvement_pct'], 2)}% | `{row['advantage_preserved']}` | "
            f"`{','.join(row.get('stronger_boundary_conditions', [])) or 'none'}` |\n"
        )

    report += """
## Parameter Traceability

| Parameter | Value | Unit | Evidence | Source |
|---|---:|---|---|---|
"""
    for row in parameter_rows_:
        report += (
            f"| `{row['parameter']}` | {fmt(row['value'], 4)} | {row['unit']} | "
            f"`{row['evidence_label']}` | `{row['source']}` |\n"
        )

    report += f"""
## Artifacts

- Kernel CSV: `experiments/results/report_data/suds_transformer_architecture_sim_{tag}_kernels.csv`
- Summary CSV: `experiments/results/report_data/suds_transformer_architecture_sim_{tag}_summary.csv`
- Parameter CSV: `experiments/results/report_data/suds_transformer_architecture_sim_{tag}_parameters.csv`
- Sensitivity CSV: `experiments/results/report_data/suds_transformer_architecture_sim_{tag}_sensitivity.csv`
- GLUE linkage CSV: `experiments/results/report_data/suds_glue_architecture_linkage_{tag}.csv`
- Design-space CSV: `experiments/results/report_data/suds_transformer_architecture_design_space_{tag}.csv`
- Design-space JSON: `experiments/results/report_data/suds_transformer_architecture_design_space_{tag}.json`
- JSON: `experiments/results/report_data/suds_transformer_architecture_sim_{tag}.json`

## Regeneration

```bash
.venv311-mps/bin/python experiments/tools/build_suds_transformer_architecture_sim.py --tag {tag}
```
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    mobilevit = load_json(args.mobilevit_json)
    glue = load_json(args.glue_json)
    adc = load_json(args.adc_json)
    rtl = load_json(args.rtl_json)
    phy = load_json(args.phy_json)

    params = derive_params(adc, rtl, phy)
    parameter_rows_ = parameter_rows(params, args)
    profiles = source_profile_rows(mobilevit, glue)
    workloads = workload_defs(args)
    schedules = {
        workload: schedule_ops(workload, meta, params)
        for workload, meta in workloads.items()
    }

    summary_rows: list[dict[str, Any]] = []
    for workload, meta in workloads.items():
        for condition in CONDITION_LABELS:
            profile = condition_profile(workload, condition, profiles)
            for sensitivity_case in ("nominal", "pessimistic"):
                summary_rows.append(
                    simulate_condition(
                        schedules[workload],
                        meta,
                        workload,
                        condition,
                        profile,
                        params,
                        sensitivity_case=sensitivity_case,
                    )
                )
    normalize_rows(summary_rows)
    kernel_rows = build_kernel_condition_rows(schedules, summary_rows)
    glue_link_rows = build_glue_link_rows(glue, summary_rows)
    design_space_rows = build_design_space_rows(workloads, profiles, params)
    sensitivity_rows = [row for row in summary_rows if row["sensitivity_case"] == "pessimistic"]
    decision = build_decision(summary_rows, parameter_rows_)

    write_csv(args.parameters_csv, parameter_rows_)
    write_csv(args.summary_csv, summary_rows)
    write_csv(args.sensitivity_csv, sensitivity_rows)
    write_csv(args.kernels_csv, kernel_rows)
    write_csv(args.glue_link_csv, glue_link_rows)
    write_csv(args.design_space_csv, design_space_rows)
    write_design_space_json(args.design_space_json, tag=args.tag, design_space_rows=design_space_rows)
    write_json(
        args.json_out,
        tag=args.tag,
        params=params,
        decision=decision,
        parameter_rows_=parameter_rows_,
        summary_rows=summary_rows,
        kernel_rows=kernel_rows,
        glue_link_rows=glue_link_rows,
        design_space_rows=design_space_rows,
        args=args,
    )
    write_report(
        args.report_out,
        tag=args.tag,
        decision=decision,
        summary_rows=summary_rows,
        parameter_rows_=parameter_rows_,
        design_space_rows=design_space_rows,
    )

    print(f"wrote {repo_path(args.parameters_csv)}")
    print(f"wrote {repo_path(args.summary_csv)}")
    print(f"wrote {repo_path(args.sensitivity_csv)}")
    print(f"wrote {repo_path(args.kernels_csv)}")
    print(f"wrote {repo_path(args.glue_link_csv)}")
    print(f"wrote {repo_path(args.design_space_csv)}")
    print(f"wrote {repo_path(args.design_space_json)}")
    print(f"wrote {repo_path(args.json_out)}")
    print(f"wrote {repo_path(args.report_out)}")
    print(f"architecture_sim_status={decision['architecture_sim_status']}")


if __name__ == "__main__":
    main()
