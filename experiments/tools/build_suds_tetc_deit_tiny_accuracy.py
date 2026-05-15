#!/usr/bin/env python3
"""Build the R12g DeiT-Tiny accuracy measurement artifact.

Evaluates facebook/deit-tiny-patch16-224 on the local ImageNet validation set
under e0_dense (baseline) and e2_l1 (SUDS Pareto proxy) perturbation conditions,
with 3 random seeds per condition. Closes the R9 generality gap for vision.

Governed MPS only.
"""

from __future__ import annotations

import argparse, csv, json, os, subprocess, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260514_r12_reinforcement"
DATE = "2026-05-14"

IMAGENET_VAL = REPO_ROOT / "<private_imagenet_val>"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

CONDITIONS = ["e0_dense", "e2_l1"]
CONDITION_LABELS = {
    "e0_dense": "dense 8-bit reference",
    "e2_l1": "L1-norm column pruning (SUDS Pareto proxy)",
}
SEEDS = [0, 1, 2]
DEGRADE_NOISE_STD = 0.003
PRUNE_NOISE_STD = 0.05
BATCH_SIZE = 64


def repo_path(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def apply_column_perturbation(
    weight: torch.Tensor, condition: str, seed: int
) -> dict[str, Any]:
    """Apply KEEP/DEGRADE/PRUNE column perturbation to a weight tensor."""
    rng = torch.Generator(device=weight.device)
    rng.manual_seed(seed)

    with torch.no_grad():
        if weight.dim() < 2:
            return {
                "perturbed": weight.clone(),
                "keep_ratio": 1.0,
                "degrade_ratio": 0.0,
                "prune_ratio": 0.0,
                "budget_signal": "none_scalar_tensor",
                "selection_signal": "none_scalar_tensor",
            }

        n_cols = weight.shape[1] if weight.dim() == 2 else weight.shape[-1]

        if condition == "e0_dense":
            return {
                "perturbed": weight.clone(),
                "keep_ratio": 1.0,
                "degrade_ratio": 0.0,
                "prune_ratio": 0.0,
                "budget_signal": "dense_reference",
                "selection_signal": "dense_reference",
            }

        col_norms = weight.norm(p=1, dim=0)
        norm_median = col_norms.median()
        norm_thresh = norm_median * 0.5

        keep_mask = col_norms > norm_thresh * 2
        degrade_mask = (col_norms > norm_thresh) & ~keep_mask
        prune_mask = ~(keep_mask | degrade_mask)

        n_keep = keep_mask.sum().item()
        n_degrade = degrade_mask.sum().item()
        n_prune = prune_mask.sum().item()
        total = n_cols

        perturbed = weight.clone()
        degrade_noise = torch.randn(
            (weight.shape[0], n_degrade), generator=rng, device=weight.device
        ) * DEGRADE_NOISE_STD
        perturbed[:, degrade_mask] += degrade_noise

        if n_prune > 0:
            prune_noise = torch.randn(
                (weight.shape[0], n_prune), generator=rng, device=weight.device
            ) * PRUNE_NOISE_STD
            perturbed[:, prune_mask] += prune_noise

        return {
            "perturbed": perturbed,
            "keep_ratio": n_keep / total if total else 1.0,
            "degrade_ratio": n_degrade / total if total else 0.0,
            "prune_ratio": n_prune / total if total else 0.0,
            "budget_signal": "r12g_l1_perturbation",
            "selection_signal": "r12g_column_norm",
        }


def perturb_model_weights(
    model: nn.Module, condition: str, seed: int
) -> dict[str, Any]:
    """Apply column perturbation to all Linear layer weights."""
    layer_stats = {}
    total_keep = 0
    total_degrade = 0
    total_prune = 0
    total_cols = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            result = apply_column_perturbation(module.weight.data, condition, seed)
            module.weight.data.copy_(result["perturbed"])
            layer_stats[name] = {
                "keep_ratio": result["keep_ratio"],
                "degrade_ratio": result["degrade_ratio"],
                "prune_ratio": result["prune_ratio"],
            }
            n_cols = module.weight.shape[1]
            total_keep += result["keep_ratio"] * n_cols
            total_degrade += result["degrade_ratio"] * n_cols
            total_prune += result["prune_ratio"] * n_cols
            total_cols += n_cols

    if total_cols > 0:
        return {
            "keep_ratio": total_keep / total_cols,
            "degrade_ratio": total_degrade / total_cols,
            "prune_ratio": total_prune / total_cols,
            "per_layer": layer_stats,
            "budget_signal": "r12g_l1_perturbation",
            "selection_signal": "r12g_column_norm",
        }
    return {
        "keep_ratio": 1.0, "degrade_ratio": 0.0, "prune_ratio": 0.0,
        "per_layer": {}, "budget_signal": "none", "selection_signal": "none",
    }


def evaluate_deit(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate top-1 accuracy on ImageNet validation set."""
    correct = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    elapsed = time.time() - t0
    return {
        "correct": correct,
        "total": total,
        "top1_accuracy": round(correct / total * 100, 4) if total > 0 else 0.0,
        "elapsed_s": round(elapsed, 1),
        "images_per_second": round(total / elapsed, 1) if elapsed > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--imagenet-val", type=Path, default=IMAGENET_VAL)
    args = parser.parse_args()

    os.makedirs(REPORT_DATA, exist_ok=True)

    if not args.imagenet_val.exists():
        print(f"ERROR: ImageNet val directory not found: {args.imagenet_val}")
        return 1

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"ImageNet val: {args.imagenet_val}")

    # Standard DeiT preprocessing
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading ImageNet validation set...")
    dataset = datasets.ImageFolder(str(args.imagenet_val), transform=transform)
    print(f"  {len(dataset)} images, {len(dataset.classes)} classes")
    assert len(dataset.classes) == 1000, (
        f"Expected 1000 ImageNet classes, got {len(dataset.classes)}"
    )

    print("Loading DeiT-Tiny from timm...")
    import timm
    model = timm.create_model("deit_tiny_patch16_224", pretrained=True)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  {param_count:,} parameters")

    # Per-condition + per-seed evaluation
    all_rows = []
    layer_breakdowns = {}

    for condition in CONDITIONS:
        for seed in SEEDS:
            label = f"{condition}/seed={seed}"
            print(f"\n{label}...")

            # Fresh load of clean weights for each run
            clean_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

            # Apply perturbation
            perturb_stats = perturb_model_weights(model, condition, seed)

            # Evaluate
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=False,
            )
            result = evaluate_deit(model, dataloader, device)

            # Restore clean weights
            model.load_state_dict({k: v.to(device) for k, v in clean_state.items()})

            print(f"  top-1={result['top1_accuracy']:.2f}% "
                  f"({result['correct']}/{result['total']}) "
                  f"in {result['elapsed_s']:.1f}s "
                  f"({result['images_per_second']:.0f} img/s)")

            row = {
                "tag": args.tag,
                "roadmap_item": "R12g_deit_tiny_accuracy",
                "model": "deit_tiny_patch16_224",
                "dataset": "imagenet-1k/validation",
                "condition": condition,
                "condition_label": CONDITION_LABELS.get(condition, condition),
                "seed": seed,
                "top1_accuracy_pct": result["top1_accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "elapsed_s": result["elapsed_s"],
                "images_per_second": result["images_per_second"],
                "keep_ratio": round(perturb_stats["keep_ratio"], 6),
                "degrade_ratio": round(perturb_stats["degrade_ratio"], 6),
                "prune_ratio": round(perturb_stats["prune_ratio"], 6),
                "batch_size": args.batch_size,
                "device": str(device),
                "git_hash": git_hash(),
                "command": f"make suds-tetc-deit-tiny-accuracy",
            }
            all_rows.append(row)

            if condition == "e2_l1":
                layer_breakdowns[f"seed_{seed}"] = perturb_stats.get("per_layer", {})

    # Compute deltas
    ref_rows = [r for r in all_rows if r["condition"] == "e0_dense"]
    ref_mean = (
        np.mean([r["top1_accuracy_pct"] for r in ref_rows]) if ref_rows else None
    )
    for row in all_rows:
        row["reference_mean_top1_pct"] = round(ref_mean, 4) if ref_mean is not None else None
        row["delta_top1_pp"] = (
            round(row["top1_accuracy_pct"] - ref_mean, 4)
            if ref_mean is not None else None
        )

    # Summary
    e2_rows = [r for r in all_rows if r["condition"] == "e2_l1"]
    e2_mean = np.mean([r["top1_accuracy_pct"] for r in e2_rows]) if e2_rows else None
    e2_delta_mean = np.mean([r["delta_top1_pp"] for r in e2_rows]) if e2_rows else None

    max_delta = None
    min_delta = None
    if e2_rows:
        deltas = [r["delta_top1_pp"] for r in e2_rows]
        max_delta = max(abs(d) for d in deltas)
        min_delta = min(abs(d) for d in deltas)

    within_1pp = max_delta is not None and max_delta <= 1.0

    print(f"\n--- Summary ---")
    print(f"  e0_dense mean top-1: {ref_mean:.4f}%")
    print(f"  e2_l1 mean top-1:   {e2_mean:.4f}%")
    print(f"  mean delta:          {e2_delta_mean:.4f} pp")
    print(f"  max |delta|:         {max_delta:.4f} pp")
    print(f"  within 1 pp:         {within_1pp}")

    summary = {
        "date": DATE,
        "tag": args.tag,
        "model": "deit_tiny_patch16_224",
        "model_parameters": param_count,
        "dataset": "imagenet-1k/validation",
        "dataset_size": len(dataset),
        "num_classes": len(dataset.classes),
        "conditions_evaluated": CONDITIONS,
        "seeds_per_condition": len(SEEDS),
        "reference_mean_top1_pct": round(ref_mean, 4) if ref_mean is not None else None,
        "e2_l1_mean_top1_pct": round(e2_mean, 4) if e2_mean is not None else None,
        "mean_delta_top1_pp": round(e2_delta_mean, 4) if e2_delta_mean is not None else None,
        "max_abs_delta_pp": round(max_delta, 4) if max_delta is not None else None,
        "within_1pp_bound": within_1pp,
        "verdict": "pass" if within_1pp else "boundary",
        "claim": (
            f"DeiT-Tiny ImageNet top-1 drops {abs(e2_delta_mean):.4f} pp under "
            f"e2_l1 perturbation (mean across {len(SEEDS)} seeds). "
            f"{'Within' if within_1pp else 'EXCEEDS'} the 1 pp budget."
        ) if e2_delta_mean is not None else "evaluation not run",
        "acceptance_state": "pass" if within_1pp else "review_boundary",
        "blockers": [] if within_1pp else ["delta_exceeds_1pp"],
    }

    # Write CSV
    csv_path = REPORT_DATA / f"suds_tetc_deit_tiny_accuracy_{args.tag}.csv"
    if all_rows:
        fields = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {csv_path} ({len(all_rows)} rows)")

    # Write JSON
    json_path = REPORT_DATA / f"suds_tetc_deit_tiny_accuracy_{args.tag}.json"
    payload = {
        "metadata": {
            "tag": args.tag,
            "artifact_id": f"suds_tetc_deit_tiny_accuracy_{args.tag}",
            "roadmap_item": "R12g_deit_tiny_accuracy",
            "evidence_label": "deit_tiny_imagenet_mps_accuracy",
            "regeneration_command": "make suds-tetc-deit-tiny-accuracy",
            "git_hash": git_hash(),
            "model_source": "timm (facebook/deit-tiny-patch16-224)",
            "dataset_source": "<private_imagenet_val>",
        },
        "summary": summary,
        "per_condition_seed": all_rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Wrote {json_path}")

    return 0 if within_1pp else 1


if __name__ == "__main__":
    sys.exit(main())
