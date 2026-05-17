"""Capture-manifest helpers for bounded bitstream slice replay."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None


CAPTURE_MANIFEST_FIELDS = [
    "capture_id",
    "model",
    "module_key",
    "operand_role",
    "tensor_shape",
    "dtype",
    "encoding_mode",
    "multiplier_mode",
    "accumulator_mode",
    "stream_length",
    "generator",
    "call_count",
    "value_count",
    "min_value",
    "max_value",
    "mean_value",
    "positive_fraction",
    "negative_fraction",
    "zero_fraction",
    "operand_values_json",
    "notes",
]


def summarize_operand_stats(values: Iterable[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "value_count": 0,
            "min_value": 0.0,
            "max_value": 0.0,
            "mean_value": 0.0,
            "positive_fraction": 0.0,
            "negative_fraction": 0.0,
            "zero_fraction": 0.0,
        }
    total = len(numbers)
    positive = sum(1 for value in numbers if value > 0.0)
    negative = sum(1 for value in numbers if value < 0.0)
    zero = total - positive - negative
    return {
        "value_count": total,
        "min_value": min(numbers),
        "max_value": max(numbers),
        "mean_value": sum(numbers) / float(total),
        "positive_fraction": positive / float(total),
        "negative_fraction": negative / float(total),
        "zero_fraction": zero / float(total),
    }


def build_capture_record(
    *,
    capture_id: str,
    model: str,
    module_key: str,
    operand_role: str,
    tensor_shape: Iterable[int],
    dtype: str,
    encoding_mode: str,
    multiplier_mode: str,
    accumulator_mode: str,
    stream_length: int,
    generator: str,
    operand_values: Iterable[float] | None = None,
    notes: str | None = None,
) -> dict[str, object]:
    operand_values_list = [float(value) for value in (operand_values or [])]
    row = {
        "capture_id": capture_id,
        "model": model,
        "module_key": module_key,
        "operand_role": operand_role,
        "tensor_shape": json.dumps([int(dim) for dim in tensor_shape], ensure_ascii=False),
        "dtype": dtype,
        "encoding_mode": encoding_mode,
        "multiplier_mode": multiplier_mode,
        "accumulator_mode": accumulator_mode,
        "stream_length": int(stream_length),
        "generator": generator,
        "call_count": 1,
        "operand_values_json": json.dumps(operand_values_list, ensure_ascii=False),
        "notes": notes or "",
    }
    row.update(
        summarize_operand_stats(operand_values_list)
    )
    return row


def parse_operand_values(row: dict[str, object]) -> list[float]:
    payload = str(row.get("operand_values_json") or "").strip()
    if not payload:
        return []
    decoded = json.loads(payload)
    if not isinstance(decoded, list):
        return []
    return [float(value) for value in decoded]


def write_capture_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAPTURE_MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CAPTURE_MANIFEST_FIELDS})
    return path


def capture_torch_module_outputs(
    model,
    *,
    module_keys: Iterable[str],
    model_inputs,
    capture_id_prefix: str,
    encoding_mode: str,
    multiplier_mode: str,
    accumulator_mode: str,
    stream_length: int,
    generator: str,
    max_values: int = 256,
) -> list[dict[str, object]]:
    """Capture bounded output samples from selected torch modules via hooks."""
    if torch is None:
        raise RuntimeError("PyTorch is required for capture_torch_module_outputs.")

    requested = {str(key) for key in module_keys}
    captures: list[dict[str, object]] = []
    handles = []

    def _tensor_values(value) -> list[float]:
        if not torch.is_tensor(value):
            return []
        flat = value.detach().cpu().reshape(-1)
        if max_values > 0:
            flat = flat[:max_values]
        return [float(item) for item in flat.tolist()]

    def _build_hook(module_key: str):
        def hook(module, inputs, output) -> None:
            if torch.is_tensor(output):
                captures.append(
                    build_capture_record(
                        capture_id=f"{capture_id_prefix}:{module_key}",
                        model=model.__class__.__name__,
                        module_key=module_key,
                        operand_role="output",
                        tensor_shape=tuple(int(dim) for dim in output.shape),
                        dtype=str(output.dtype),
                        encoding_mode=encoding_mode,
                        multiplier_mode=multiplier_mode,
                        accumulator_mode=accumulator_mode,
                        stream_length=stream_length,
                        generator=generator,
                        operand_values=_tensor_values(output),
                    )
                )
        return hook

    for module_name, module in model.named_modules():
        if module_name in requested:
            handles.append(module.register_forward_hook(_build_hook(module_name)))

    try:
        with torch.inference_mode():
            if isinstance(model_inputs, tuple):
                model(*model_inputs)
            else:
                model(model_inputs)
    finally:
        for handle in handles:
            handle.remove()
    return captures
