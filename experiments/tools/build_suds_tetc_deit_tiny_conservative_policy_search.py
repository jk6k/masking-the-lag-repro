#!/usr/bin/env python3
"""Build the R13 DeiT-Tiny conservative policy-search artifact.

The search is governed as MPS-only. It first screens conservative no-prune
policies on a small ImageNet validation slice labelled `screening_only`, then
selects at most two candidates for full ImageNet validation with three seeds.
Architecture joins reuse the existing R9 DeiT-Tiny rows and simulator surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from build_suds_tetc_workload_expansion import ADC_JSON, PHY_JSON, RTL_JSON, workload_defs
from build_suds_transformer_architecture_sim import (
    derive_params,
    load_json,
    schedule_ops,
    simulate_condition,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATE = "2026-05-17"
TAG = "20260517_r13"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
IMAGENET_VAL = REPO_ROOT / "<private_imagenet_val>"
R9_WORKLOAD_JSON = REPORT_DATA / "suds_tetc_workload_expansion_20260513_tetc_pivot.json"
R12_DEIT_JSON = REPORT_DATA / "suds_tetc_deit_tiny_accuracy_20260514_r12_reinforcement.json"
CSV_OUT = REPORT_DATA / f"suds_tetc_deit_tiny_conservative_policy_search_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_deit_tiny_conservative_policy_search_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260517_suds_tetc_deit_tiny_conservative_policy_search.md"

SEEDS = (0, 1, 2)
SCREENING_SEED = 0
SCREENING_SAMPLES = 2048
BATCH_SIZE = 64
MODEL_NAME = "deit_tiny_patch16_224"
R12_DEGRADE_NOISE_STD = 0.003
R12_PRUNE_NOISE_STD = 0.05
FULL_COMMAND = (
    "caffeinate -dimsu .venv311-mps/bin/python "
    "experiments/tools/build_suds_tetc_deit_tiny_conservative_policy_search.py "
    "--tag 20260517_r13 --device mps"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--device", default="mps", choices=["mps"])
    parser.add_argument("--imagenet-val", type=Path, default=IMAGENET_VAL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--screening-samples", type=int, default=SCREENING_SAMPLES)
    parser.add_argument("--full-sample-count", type=int, default=0)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    return parser.parse_args()


def repo_path(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_mps() -> dict[str, Any]:
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    if fallback != "0":
        raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be forced to 0")
    if not torch.backends.mps.is_built():
        raise SystemExit("MPS backend is not built in this PyTorch install")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS backend is not available")
    probe = torch.ones((2, 2), device="mps")
    return {
        "device": "mps",
        "torch_version": getattr(torch, "__version__", "unknown"),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_probe_sum": float((probe @ probe).sum().item()),
        "PYTORCH_ENABLE_MPS_FALLBACK": fallback,
    }


def load_dataset(path: Path) -> datasets.ImageFolder:
    if not path.exists():
        raise SystemExit(f"ImageNet validation directory not found: {path}")
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    dataset = datasets.ImageFolder(str(path), transform=transform)
    if len(dataset.classes) != 1000:
        raise SystemExit(f"Expected 1000 ImageNet classes, got {len(dataset.classes)}")
    return dataset


def sha256_indices(indices: list[int]) -> str:
    h = hashlib.sha256()
    h.update(",".join(str(i) for i in indices).encode("utf-8"))
    return h.hexdigest()


def subset_dataset(dataset: datasets.ImageFolder, count: int) -> tuple[Any, str]:
    if count <= 0 or count >= len(dataset):
        indices = list(range(len(dataset)))
        return dataset, sha256_indices(indices)
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
    return Subset(dataset, indices), sha256_indices(indices)


def load_model(device: torch.device) -> nn.Module:
    import timm

    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.to(device)
    model.eval()
    return model


def r9_suds_keep_ratio() -> float:
    if not R9_WORKLOAD_JSON.is_file():
        return 0.15
    data = json.loads(R9_WORKLOAD_JSON.read_text(encoding="utf-8"))
    rows = [
        row
        for row in data.get("rows", [])
        if row.get("model") == MODEL_NAME
        and row.get("condition") == "suds_pareto"
        and int(row.get("batch_size", 0)) == 4
    ]
    return float(rows[0].get("keep_ratio", 0.15)) if rows else 0.15


def candidate_specs() -> dict[str, dict[str, Any]]:
    signal_safe_keep = r9_suds_keep_ratio()
    return {
        "e0_dense": {
            "label": "dense 8-bit reference",
            "search_role": "dense_baseline",
            "keep_ratio": 1.0,
            "degrade_ratio": 0.0,
            "prune_ratio": 0.0,
            "degrade_noise_std": 0.0,
            "prune_noise_std": 0.0,
            "selector": "dense_reference",
        },
        "e2_l1": {
            "label": "R12g L1 boundary proxy",
            "search_role": "existing_boundary",
            "keep_ratio": 0.5,
            "degrade_ratio": 0.46924,
            "prune_ratio": 0.03076,
            "degrade_noise_std": R12_DEGRADE_NOISE_STD,
            "prune_noise_std": R12_PRUNE_NOISE_STD,
            "selector": "r12g_column_norm",
        },
        "no_prune_keep90": {
            "label": "No-prune top-90% keep, standard degrade",
            "search_role": "conservative_candidate",
            "keep_ratio": 0.90,
            "degrade_ratio": 0.10,
            "prune_ratio": 0.0,
            "degrade_noise_std": R12_DEGRADE_NOISE_STD,
            "prune_noise_std": 0.0,
            "selector": "column_l1_topk",
        },
        "no_prune_keep95": {
            "label": "No-prune top-95% keep, standard degrade",
            "search_role": "conservative_candidate",
            "keep_ratio": 0.95,
            "degrade_ratio": 0.05,
            "prune_ratio": 0.0,
            "degrade_noise_std": R12_DEGRADE_NOISE_STD,
            "prune_noise_std": 0.0,
            "selector": "column_l1_topk",
        },
        "light_degrade_keep90": {
            "label": "No-prune top-90% keep, light degrade",
            "search_role": "conservative_candidate",
            "keep_ratio": 0.90,
            "degrade_ratio": 0.10,
            "prune_ratio": 0.0,
            "degrade_noise_std": 0.0015,
            "prune_noise_std": 0.0,
            "selector": "column_l1_topk",
        },
        "light_degrade_keep95": {
            "label": "No-prune top-95% keep, light degrade",
            "search_role": "conservative_candidate",
            "keep_ratio": 0.95,
            "degrade_ratio": 0.05,
            "prune_ratio": 0.0,
            "degrade_noise_std": 0.0015,
            "prune_noise_std": 0.0,
            "selector": "column_l1_topk",
        },
        "degrade_only_signal_safe": {
            "label": "No-prune R9 SUDS-ratio signal-safe degrade",
            "search_role": "conservative_candidate",
            "keep_ratio": signal_safe_keep,
            "degrade_ratio": max(0.0, 1.0 - signal_safe_keep),
            "prune_ratio": 0.0,
            "degrade_noise_std": 0.00075,
            "prune_noise_std": 0.0,
            "selector": "r9_suds_pareto_ratio_column_l1_topk",
        },
    }


def topk_keep_mask(col_norms: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    n_cols = int(col_norms.numel())
    keep_count = max(0, min(n_cols, int(math.ceil(n_cols * keep_ratio))))
    mask = torch.zeros(n_cols, dtype=torch.bool, device=col_norms.device)
    if keep_count > 0:
        mask[torch.topk(col_norms, keep_count, largest=True).indices] = True
    return mask


def apply_policy_to_weight(
    weight: torch.Tensor,
    policy_name: str,
    policy: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    rng = torch.Generator(device=weight.device)
    rng.manual_seed(seed)

    with torch.no_grad():
        if weight.dim() < 2:
            return {
                "perturbed": weight.clone(),
                "keep_ratio": 1.0,
                "degrade_ratio": 0.0,
                "prune_ratio": 0.0,
            }

        n_cols = int(weight.shape[1])
        perturbed = weight.clone()
        if policy_name == "e0_dense":
            return {
                "perturbed": perturbed,
                "keep_ratio": 1.0,
                "degrade_ratio": 0.0,
                "prune_ratio": 0.0,
            }

        col_norms = weight.norm(p=1, dim=0)
        if policy_name == "e2_l1":
            norm_thresh = col_norms.median() * 0.5
            keep_mask = col_norms > norm_thresh * 2
            degrade_mask = (col_norms > norm_thresh) & ~keep_mask
            prune_mask = ~(keep_mask | degrade_mask)
        else:
            keep_mask = topk_keep_mask(col_norms, float(policy["keep_ratio"]))
            degrade_mask = ~keep_mask
            prune_mask = torch.zeros_like(keep_mask)

        n_degrade = int(degrade_mask.sum().item())
        n_prune = int(prune_mask.sum().item())
        if n_degrade > 0:
            degrade_noise = torch.randn(
                (weight.shape[0], n_degrade), generator=rng, device=weight.device
            ) * float(policy["degrade_noise_std"])
            perturbed[:, degrade_mask] += degrade_noise
        if n_prune > 0:
            prune_noise = torch.randn(
                (weight.shape[0], n_prune), generator=rng, device=weight.device
            ) * float(policy["prune_noise_std"])
            perturbed[:, prune_mask] += prune_noise

        return {
            "perturbed": perturbed,
            "keep_ratio": float(keep_mask.sum().item()) / n_cols if n_cols else 1.0,
            "degrade_ratio": n_degrade / n_cols if n_cols else 0.0,
            "prune_ratio": n_prune / n_cols if n_cols else 0.0,
        }


def perturb_model_weights(
    model: nn.Module,
    policy_name: str,
    policy: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    total_keep = 0.0
    total_degrade = 0.0
    total_prune = 0.0
    total_cols = 0
    per_layer: dict[str, dict[str, float]] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        result = apply_policy_to_weight(module.weight.data, policy_name, policy, seed)
        module.weight.data.copy_(result["perturbed"])
        n_cols = int(module.weight.shape[1])
        total_cols += n_cols
        total_keep += float(result["keep_ratio"]) * n_cols
        total_degrade += float(result["degrade_ratio"]) * n_cols
        total_prune += float(result["prune_ratio"]) * n_cols
        per_layer[name] = {
            "keep_ratio": round(float(result["keep_ratio"]), 6),
            "degrade_ratio": round(float(result["degrade_ratio"]), 6),
            "prune_ratio": round(float(result["prune_ratio"]), 6),
        }

    if total_cols <= 0:
        return {"keep_ratio": 1.0, "degrade_ratio": 0.0, "prune_ratio": 0.0, "per_layer": {}}
    return {
        "keep_ratio": total_keep / total_cols,
        "degrade_ratio": total_degrade / total_cols,
        "prune_ratio": total_prune / total_cols,
        "per_layer": per_layer,
    }


def evaluate_deit(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, Any]:
    correct = 0
    total = 0
    t0 = time.time()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))
    elapsed = time.time() - t0
    return {
        "correct": correct,
        "total": total,
        "top1_accuracy_pct": round(correct / total * 100.0, 4) if total else 0.0,
        "elapsed_s": round(elapsed, 1),
        "images_per_second": round(total / elapsed, 1) if elapsed > 0 else 0.0,
    }


def source_hashes() -> dict[str, str]:
    hashes = {"script": sha256_path(Path(__file__).resolve())}
    if R9_WORKLOAD_JSON.is_file():
        hashes["r9_workload_json"] = sha256_path(R9_WORKLOAD_JSON)
    if R12_DEIT_JSON.is_file():
        hashes["r12_deit_json"] = sha256_path(R12_DEIT_JSON)
    return hashes


def run_one_eval(
    *,
    model: nn.Module,
    clean_state: dict[str, torch.Tensor],
    dataloader: DataLoader,
    device: torch.device,
    policy_name: str,
    policy: dict[str, Any],
    seed: int,
    stage: str,
    sample_count: int,
    sample_indices_sha256: str,
    batch_size: int,
    source_hashes_map: dict[str, str],
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.load_state_dict({key: value.to(device) for key, value in clean_state.items()})
    perturb_stats = perturb_model_weights(model, policy_name, policy, seed)
    result = evaluate_deit(model, dataloader, device)
    row = {
        "tag": tag,
        "roadmap_item": "R13_3_deit_tiny_conservative_policy_search",
        "row_type": "model_evaluation",
        "stage": stage,
        "model": MODEL_NAME,
        "dataset": "imagenet-1k/validation",
        "sample_count": sample_count,
        "sample_indices_sha256": sample_indices_sha256,
        "policy": policy_name,
        "policy_label": policy["label"],
        "search_role": policy["search_role"],
        "seed": seed,
        "top1_accuracy_pct": result["top1_accuracy_pct"],
        "correct": result["correct"],
        "total": result["total"],
        "elapsed_s": result["elapsed_s"],
        "images_per_second": result["images_per_second"],
        "keep_ratio": round(float(perturb_stats["keep_ratio"]), 6),
        "degrade_ratio": round(float(perturb_stats["degrade_ratio"]), 6),
        "prune_ratio": round(float(perturb_stats["prune_ratio"]), 6),
        "target_keep_ratio": round(float(policy["keep_ratio"]), 6),
        "target_degrade_ratio": round(float(policy["degrade_ratio"]), 6),
        "target_prune_ratio": round(float(policy["prune_ratio"]), 6),
        "degrade_noise_std": policy["degrade_noise_std"],
        "prune_noise_std": policy["prune_noise_std"],
        "selector": policy["selector"],
        "batch_size": batch_size,
        "device": str(device),
        "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""),
        "git_hash": git_hash(),
        "command": FULL_COMMAND,
        "source_r9_workload_json_sha256": source_hashes_map.get("r9_workload_json", ""),
        "source_r12_deit_json_sha256": source_hashes_map.get("r12_deit_json", ""),
        "script_sha256": source_hashes_map.get("script", ""),
    }
    return row, perturb_stats.get("per_layer", {})


def add_reference_deltas(rows: list[dict[str, Any]], *, stage: str) -> None:
    dense_rows = [row for row in rows if row["stage"] == stage and row["policy"] == "e0_dense"]
    if not dense_rows:
        return
    ref_mean = float(np.mean([float(row["top1_accuracy_pct"]) for row in dense_rows]))
    for row in rows:
        if row["stage"] != stage:
            continue
        row["reference_mean_top1_pct"] = round(ref_mean, 4)
        row["delta_top1_pp"] = round(float(row["top1_accuracy_pct"]) - ref_mean, 4)


def select_candidates(screening_rows: list[dict[str, Any]], policies: dict[str, dict[str, Any]]) -> list[str]:
    dense = [row for row in screening_rows if row["policy"] == "e0_dense"]
    ref = float(dense[0]["top1_accuracy_pct"]) if dense else 0.0
    ranked: list[dict[str, Any]] = []
    for row in screening_rows:
        policy_name = row["policy"]
        if policies[policy_name]["search_role"] != "conservative_candidate":
            continue
        delta = float(row["top1_accuracy_pct"]) - ref
        row["reference_mean_top1_pct"] = round(ref, 4)
        row["delta_top1_pp"] = round(delta, 4)
        ranked.append(
            {
                "policy": policy_name,
                "screen_pass": abs(delta) <= 1.0,
                "target_degrade_ratio": float(policies[policy_name]["degrade_ratio"]),
                "degrade_noise_std": float(policies[policy_name]["degrade_noise_std"]),
                "screen_delta": delta,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["screen_pass"],
            item["target_degrade_ratio"],
            -item["degrade_noise_std"],
            item["screen_delta"],
        ),
        reverse=True,
    )
    return [item["policy"] for item in ranked[:2]]


def architecture_join_rows(
    *,
    policies: dict[str, dict[str, Any]],
    full_rows: list[dict[str, Any]],
    tag: str,
    source_hashes_map: dict[str, str],
) -> list[dict[str, Any]]:
    params = derive_params(load_json(ADC_JSON), load_json(RTL_JSON), load_json(PHY_JSON))
    rows: list[dict[str, Any]] = []
    means: dict[str, dict[str, float]] = {}
    for policy_name in policies:
        policy_rows = [
            row
            for row in full_rows
            if row.get("policy") == policy_name and row.get("stage") == "full_validation"
        ]
        if not policy_rows:
            continue
        means[policy_name] = {
            "accuracy": float(np.mean([float(row["top1_accuracy_pct"]) for row in policy_rows])),
            "delta_accuracy": float(np.mean([float(row.get("delta_top1_pp", 0.0)) for row in policy_rows])),
            "n_rows": float(len(policy_rows)),
        }

    for workload, meta in workload_defs().items():
        if meta.get("model") != MODEL_NAME:
            continue
        schedule = schedule_ops(workload, meta, params)
        dense_profile = {
            "keep_ratio": 1.0,
            "degrade_ratio": 0.0,
            "prune_ratio": 0.0,
            "accuracy_metric": "top1",
            "accuracy": math.nan,
            "delta_accuracy": 0.0,
            "accuracy_evidence_label": "r13_deit_dense_reference",
            "promotion_decision": "reference",
            "device": "mps",
            "git_hash": git_hash(),
            "n_rows": 3,
            "source_condition": "e0_dense_r13_full_validation",
        }
        baseline = simulate_condition(
            schedule,
            meta,
            workload,
            "lightening_dptc",
            dense_profile,
            params,
            sensitivity_case="nominal",
            adc_sharing_mode="temporal_accum",
        )
        for policy_name, policy in policies.items():
            if policy_name == "e0_dense" or policy_name not in means:
                continue
            profile = {
                "keep_ratio": float(policy["keep_ratio"]),
                "degrade_ratio": float(policy["degrade_ratio"]),
                "prune_ratio": float(policy["prune_ratio"]),
                "accuracy_metric": "top1",
                "accuracy": means[policy_name]["accuracy"],
                "delta_accuracy": means[policy_name]["delta_accuracy"],
                "accuracy_evidence_label": "measured_mps_deit_tiny_r13_conservative_search",
                "promotion_decision": "secondary_support_candidate",
                "device": "mps",
                "git_hash": git_hash(),
                "n_rows": int(means[policy_name]["n_rows"]),
                "source_condition": f"{policy_name}_r13_conservative_search",
            }
            row = simulate_condition(
                schedule,
                meta,
                workload,
                "suds_pareto",
                profile,
                params,
                sensitivity_case="nominal",
                adc_sharing_mode="temporal_accum",
            )
            row.update(
                {
                    "tag": tag,
                    "roadmap_item": "R13_3_deit_tiny_conservative_policy_search",
                    "row_type": "architecture_join",
                    "stage": "r9_architecture_join",
                    "condition": policy_name,
                    "condition_label": policy["label"],
                    "condition_family": "r13_deit_conservative_policy",
                    "policy": policy_name,
                    "policy_label": policy["label"],
                    "search_role": policy["search_role"],
                    "sample_count": 50000,
                    "seed": "aggregate_3_seed",
                    "device": "mps",
                    "command": FULL_COMMAND,
                    "source_r9_workload_json_sha256": source_hashes_map.get("r9_workload_json", ""),
                    "source_r12_deit_json_sha256": source_hashes_map.get("r12_deit_json", ""),
                    "script_sha256": source_hashes_map.get("script", ""),
                    "top1_accuracy_pct": round(means[policy_name]["accuracy"], 4),
                    "delta_top1_pp": round(means[policy_name]["delta_accuracy"], 4),
                    "reference_lightening_edp_pj_ns": baseline["edp_pj_ns"],
                    "energy_ratio_vs_lightening": row["energy_pj"] / baseline["energy_pj"],
                    "latency_ratio_vs_lightening": row["latency_ns"] / baseline["latency_ns"],
                    "edp_ratio_vs_lightening": row["edp_pj_ns"] / baseline["edp_pj_ns"],
                }
            )
            row["energy_improvement_vs_lightening_pct"] = 100.0 * (
                1.0 - float(row["energy_ratio_vs_lightening"])
            )
            row["edp_improvement_vs_lightening_pct"] = 100.0 * (
                1.0 - float(row["edp_ratio_vs_lightening"])
            )
            rows.append(row)
    return rows


def aggregate_full(
    full_rows: list[dict[str, Any]],
    arch_rows: list[dict[str, Any]],
    policies: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for policy_name, policy in policies.items():
        if policy_name == "e0_dense":
            continue
        rows = [
            row
            for row in full_rows
            if row.get("policy") == policy_name and row.get("stage") == "full_validation"
        ]
        if not rows:
            continue
        deltas = [float(row.get("delta_top1_pp", 0.0)) for row in rows]
        top1 = [float(row["top1_accuracy_pct"]) for row in rows]
        policy_arch = [row for row in arch_rows if row.get("policy") == policy_name]
        edp_improvements = [
            float(row["edp_improvement_vs_lightening_pct"]) for row in policy_arch
        ]
        accuracy_pass = bool(deltas) and abs(float(np.mean(deltas))) <= 1.0 and max(abs(d) for d in deltas) <= 1.0
        positive_edp = bool(edp_improvements) and min(edp_improvements) > 0.0
        aggregates.append(
            {
                "policy": policy_name,
                "policy_label": policy["label"],
                "search_role": policy["search_role"],
                "seed_count": len(rows),
                "sample_count_per_seed": rows[0]["sample_count"] if rows else 0,
                "mean_top1_accuracy_pct": round(float(np.mean(top1)), 4),
                "mean_delta_top1_pp": round(float(np.mean(deltas)), 4),
                "worst_seed_delta_top1_pp": round(min(deltas), 4),
                "max_abs_delta_top1_pp": round(max(abs(d) for d in deltas), 4),
                "accuracy_within_1pp": accuracy_pass,
                "min_edp_improvement_vs_lightening_pct": round(min(edp_improvements), 4) if edp_improvements else None,
                "mean_edp_improvement_vs_lightening_pct": round(float(np.mean(edp_improvements)), 4) if edp_improvements else None,
                "positive_edp_improvement": positive_edp,
                "meets_5pct_target": bool(edp_improvements) and min(edp_improvements) >= 5.0,
                "success_acceptance": accuracy_pass and positive_edp,
            }
        )
    aggregates.sort(
        key=lambda row: (
            bool(row["success_acceptance"]),
            float(row.get("min_edp_improvement_vs_lightening_pct") or -1.0),
            -float(row["max_abs_delta_top1_pp"]),
        ),
        reverse=True,
    )
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    screening_rows: list[dict[str, Any]],
    selected: list[str],
    aggregates: list[dict[str, Any]],
    arch_rows: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    selected_text = ", ".join(f"`{item}`" for item in selected) if selected else "`none`"
    lines = [
        "# SUDS TETC R13-3 DeiT-Tiny Conservative Policy Search",
        "",
        f"Date: `{DATE}`",
        "Plan item: `R13-3`",
        f"Acceptance state: `{summary['acceptance_state']}`",
        f"Decision: `{summary['decision']}`",
        "",
        "## Scope",
        "",
        "This artifact attempts a bounded DeiT-Tiny policy search without",
        "changing the existing `suds_pareto` headline row. It is an MPS-only",
        "accuracy measurement plus R9 architecture join, not a silicon, layout,",
        "device-solver, timing-closure, or bench-energy claim.",
        "",
        "## Screening",
        "",
        f"Screening label: `screening_only`; sample count: `{summary['screening_sample_count']}`; seed: `{SCREENING_SEED}`.",
        "",
        "| Policy | Top-1 (%) | Delta (pp) | Selected for full validation |",
        "|---|---:|---:|---|",
    ]
    for row in screening_rows:
        if row.get("policy") == "e0_dense":
            continue
        lines.append(
            f"| {row['policy']} | {fmt(row.get('top1_accuracy_pct'), 4)} | "
            f"{fmt(row.get('delta_top1_pp'), 4)} | {'yes' if row['policy'] in selected else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Full Validation",
            "",
            f"Selected candidates: {selected_text}. Each selected policy was run",
            "on the full 50,000-image validation split with seeds `0`, `1`, and",
            "`2`; dense reference rows were rerun under the same command.",
            "",
            "| Policy | Seeds | Mean Top-1 (%) | Mean delta (pp) | Worst seed delta (pp) | Min EDP improvement vs Lightening (%) | Decision |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row['policy']} | {row['seed_count']} | {fmt(row['mean_top1_accuracy_pct'], 4)} | "
            f"{fmt(row['mean_delta_top1_pp'], 4)} | {fmt(row['worst_seed_delta_top1_pp'], 4)} | "
            f"{fmt(row['min_edp_improvement_vs_lightening_pct'], 3)} | "
            f"{'pass' if row['success_acceptance'] else 'boundary'} |"
        )
    lines.extend(
        [
            "",
            "## R9 Architecture Join",
            "",
            "| Workload | Batch | Policy | EDP ratio vs Lightening | EDP improvement (%) |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for row in arch_rows:
        lines.append(
            f"| {row['workload']} | {row.get('batch_size', 'n/a')} | {row['policy']} | "
            f"{fmt(row.get('edp_ratio_vs_lightening'), 4)} | "
            f"{fmt(row.get('edp_improvement_vs_lightening_pct'), 3)} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            summary["claim_text"],
            "",
            "The earlier R12g `e2_l1` DeiT-Tiny row remains a recorded boundary;",
            "R13-3 only supports the explicitly measured conservative no-prune",
            "policy if its accuracy and modeled-EDP checks pass.",
            "",
            "## Artifacts",
            "",
            f"- CSV: `{repo_path(csv_path)}`",
            f"- JSON: `{repo_path(json_path)}`",
            f"- Report: `{repo_path(path)}`",
            f"- R9 architecture source: `{repo_path(R9_WORKLOAD_JSON)}`",
            f"- R12 DeiT boundary source: `{repo_path(R12_DEIT_JSON)}`",
            "",
            "## Regeneration",
            "",
            "```bash",
            "caffeinate -dimsu .venv311-mps/bin/python \\",
            "  experiments/tools/build_suds_tetc_deit_tiny_conservative_policy_search.py \\",
            "  --tag 20260517_r13 \\",
            "  --device mps",
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.csv_out = Path(str(args.csv_out).replace(TAG, args.tag))
    args.json_out = Path(str(args.json_out).replace(TAG, args.tag))

    mps_meta = require_mps()
    policies = candidate_specs()
    hashes = source_hashes()

    print("R13-3 DeiT-Tiny conservative policy search")
    print(f"Device: {args.device}")
    print(f"MPS metadata: {mps_meta}")
    print(f"ImageNet val: {args.imagenet_val}")

    dataset = load_dataset(args.imagenet_val)
    screening_dataset, screening_hash = subset_dataset(dataset, args.screening_samples)
    full_count = args.full_sample_count if args.full_sample_count > 0 else len(dataset)
    full_dataset, full_hash = subset_dataset(dataset, full_count)
    print(f"Loaded dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    print(f"Screening samples: {len(screening_dataset)}")
    print(f"Full validation samples: {len(full_dataset)}")

    device = torch.device(args.device)
    model = load_model(device)
    param_count = sum(p.numel() for p in model.parameters())
    clean_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    screening_loader = DataLoader(
        screening_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    full_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_rows: list[dict[str, Any]] = []
    layer_breakdowns: dict[str, Any] = {}

    print("\nScreening candidates...")
    screening_policy_names = ["e0_dense"] + [
        name for name, spec in policies.items() if spec["search_role"] == "conservative_candidate"
    ]
    for policy_name in screening_policy_names:
        print(f"  screening_only {policy_name}")
        row, layers = run_one_eval(
            model=model,
            clean_state=clean_state,
            dataloader=screening_loader,
            device=device,
            policy_name=policy_name,
            policy=policies[policy_name],
            seed=SCREENING_SEED,
            stage="screening_only",
            sample_count=len(screening_dataset),
            sample_indices_sha256=screening_hash,
            batch_size=args.batch_size,
            source_hashes_map=hashes,
            tag=args.tag,
        )
        all_rows.append(row)
        layer_breakdowns[f"screening_only/{policy_name}/seed_{SCREENING_SEED}"] = layers
        print(
            f"    top1={row['top1_accuracy_pct']:.4f}% "
            f"keep={row['keep_ratio']:.3f} degrade={row['degrade_ratio']:.3f} "
            f"prune={row['prune_ratio']:.3f}"
        )

    add_reference_deltas(all_rows, stage="screening_only")
    selected = select_candidates([row for row in all_rows if row["stage"] == "screening_only"], policies)
    print(f"\nSelected for full validation: {selected}")

    print("\nFull validation...")
    for policy_name in ["e0_dense"] + selected:
        for seed in SEEDS:
            print(f"  full_validation {policy_name}/seed={seed}")
            row, layers = run_one_eval(
                model=model,
                clean_state=clean_state,
                dataloader=full_loader,
                device=device,
                policy_name=policy_name,
                policy=policies[policy_name],
                seed=seed,
                stage="full_validation",
                sample_count=len(full_dataset),
                sample_indices_sha256=full_hash,
                batch_size=args.batch_size,
                source_hashes_map=hashes,
                tag=args.tag,
            )
            all_rows.append(row)
            layer_breakdowns[f"full_validation/{policy_name}/seed_{seed}"] = layers
            print(
                f"    top1={row['top1_accuracy_pct']:.4f}% "
                f"({row['correct']}/{row['total']}) elapsed={row['elapsed_s']:.1f}s"
            )

    add_reference_deltas(all_rows, stage="full_validation")
    selected_policies = {"e0_dense": policies["e0_dense"]}
    selected_policies.update({name: policies[name] for name in selected})
    full_rows = [row for row in all_rows if row["stage"] == "full_validation"]
    arch_rows = architecture_join_rows(
        policies=selected_policies,
        full_rows=full_rows,
        tag=args.tag,
        source_hashes_map=hashes,
    )
    aggregates = aggregate_full(full_rows, arch_rows, selected_policies)

    successful = [row for row in aggregates if row["success_acceptance"]]
    if successful:
        winner = successful[0]
        acceptance_state = "pass"
        decision = "deit_tiny_secondary_support"
        claim_text = (
            f"R13-3 finds `{winner['policy']}` as a conservative secondary "
            f"DeiT-Tiny support point: mean Top-1 delta "
            f"{winner['mean_delta_top1_pp']:.4f} pp, worst-seed delta "
            f"{winner['worst_seed_delta_top1_pp']:.4f} pp, and minimum modeled "
            f"EDP improvement {winner['min_edp_improvement_vs_lightening_pct']:.3f}% "
            "against the same-scope Lightening-style reference."
        )
    else:
        winner = None
        acceptance_state = "deit_tiny_policy_search_boundary"
        decision = "boundary_recorded"
        claim_text = (
            "R13-3 did not find a conservative DeiT-Tiny operating point that "
            "simultaneously satisfies the one-percentage-point accuracy budget "
            "and positive modeled EDP-improvement check."
        )

    summary = {
        "date": DATE,
        "tag": args.tag,
        "model": MODEL_NAME,
        "model_parameters": param_count,
        "dataset": "imagenet-1k/validation",
        "dataset_size": len(dataset),
        "screening_sample_count": len(screening_dataset),
        "full_validation_sample_count": len(full_dataset),
        "seeds": list(SEEDS),
        "selected_candidates": selected,
        "acceptance_state": acceptance_state,
        "decision": decision,
        "claim_text": claim_text,
        "winner": winner,
        "blockers": [] if successful else ["no_selected_policy_met_success_acceptance"],
    }

    csv_rows = all_rows + arch_rows
    write_csv(args.csv_out, csv_rows)
    payload = {
        "metadata": {
            "tag": args.tag,
            "artifact_id": f"suds_tetc_deit_tiny_conservative_policy_search_{args.tag}",
            "roadmap_item": "R13_3_deit_tiny_conservative_policy_search",
            "evidence_label": "deit_tiny_mps_conservative_policy_search",
            "regeneration_command": FULL_COMMAND,
            "git_hash": git_hash(),
            "model_source": "timm (facebook/deit-tiny-patch16-224)",
            "dataset_source": repo_path(args.imagenet_val),
            "source_hashes": hashes,
        },
        "mps": mps_meta,
        "summary": summary,
        "screening_rows": [row for row in all_rows if row["stage"] == "screening_only"],
        "full_validation_rows": full_rows,
        "architecture_join_rows": arch_rows,
        "aggregates": aggregates,
        "layer_breakdowns": layer_breakdowns,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_report(
        args.report_out,
        summary=summary,
        screening_rows=[row for row in all_rows if row["stage"] == "screening_only"],
        selected=selected,
        aggregates=aggregates,
        arch_rows=arch_rows,
        csv_path=args.csv_out,
        json_path=args.json_out,
    )

    print("\nSummary")
    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {args.csv_out}")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.report_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
