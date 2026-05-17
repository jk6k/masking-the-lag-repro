"""Calibration and sweep helpers for bounded bitstream kernels."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
import statistics
from pathlib import Path
from typing import Iterable, Mapping

from .generators import resolve_generator_default_policy
from .kernels import estimate_dot_product


def run_stream_length_sweep(
    lhs_values: Iterable[float],
    rhs_values: Iterable[float],
    *,
    stream_lengths: Iterable[int],
    generator: str = "bernoulli",
    encoding_mode: str = "bipolar",
    multiplier_mode: str = "xnor",
    seed: int | None = 0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stream_length in stream_lengths:
        result = estimate_dot_product(
            lhs_values,
            rhs_values,
            stream_length=int(stream_length),
            generator=generator,
            encoding_mode=encoding_mode,
            multiplier_mode=multiplier_mode,
            seed=seed,
        )
        rows.append(
            {
                "stream_length": int(stream_length),
                "generator": generator,
                "encoding_mode": encoding_mode,
                "multiplier_mode": multiplier_mode,
                "estimated_value": result["estimated_value"],
                "exact_value": result["exact_value"],
                "abs_error": result["abs_error"],
                "total_count": result["total_count"],
            }
        )
    return rows


def _load_summary_payload(summary_json_path: str | Path) -> tuple[Path, dict[str, object]]:
    path = Path(summary_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Summary JSON must contain an object payload: {path}")
    return path, payload


def _stringify_values(values: Iterable[object]) -> str:
    return ";".join(sorted({str(value) for value in values if str(value).strip()}))


def _extract_policy_evidence_row(summary_json_path: str | Path) -> dict[str, object]:
    path, payload = _load_summary_payload(summary_json_path)
    semantics = payload.get("semantics") or {}
    if not isinstance(semantics, dict):
        semantics = {}
    replay_rows = [
        row
        for row in (payload.get("replay") or [])
        if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "replayed"
    ]
    abs_raw_errors = [
        float(row["abs_error_vs_raw_exact"])
        for row in replay_rows
        if row.get("abs_error_vs_raw_exact") is not None
    ]
    abs_captured_errors = [
        float(row["abs_error_vs_captured_output"])
        for row in replay_rows
        if row.get("abs_error_vs_captured_output") is not None
    ]
    relative_errors = []
    overestimates = []
    replay_kinds = sorted(
        {
            str(row.get("replay_kind") or "").strip()
            for row in replay_rows
            if str(row.get("replay_kind") or "").strip()
        }
    )
    for row in replay_rows:
        raw_exact = row.get("raw_exact_dot_product")
        abs_error = row.get("abs_error_vs_raw_exact")
        estimated = row.get("rescaled_estimated_dot_product")
        if raw_exact is not None and abs_error is not None:
            denom = max(1.0, abs(float(raw_exact)))
            relative_errors.append(float(abs_error) / denom)
        if raw_exact is not None and estimated is not None:
            overestimates.append(1.0 if float(estimated) > float(raw_exact) else 0.0)

    return {
        "summary_json": str(path),
        "artifact_name": path.name,
        "workload_class": str(payload.get("model") or "").strip(),
        "model_family": str(payload.get("model") or "").strip(),
        "sample_index": payload.get("sample_index"),
        "sample_label": payload.get("sample_label"),
        "stream_length": int(semantics.get("stream_length") or 0),
        "generator": str(semantics.get("generator") or "").strip(),
        "capture_row_count": int(payload.get("capture_row_count") or 0),
        "replay_row_count": len(replay_rows),
        "replay_kinds": replay_kinds,
        "sample_count": 1,
        "median_abs_error_vs_raw_exact": statistics.median(abs_raw_errors)
        if abs_raw_errors
        else 0.0,
        "max_abs_error_vs_raw_exact": max(abs_raw_errors) if abs_raw_errors else 0.0,
        "median_abs_error_vs_captured_output": statistics.median(abs_captured_errors)
        if abs_captured_errors
        else 0.0,
        "max_abs_error_vs_captured_output": max(abs_captured_errors)
        if abs_captured_errors
        else 0.0,
        "median_relative_error_vs_raw_exact": statistics.median(relative_errors)
        if relative_errors
        else 0.0,
        "overestimate_fraction": (sum(overestimates) / len(overestimates)) if overestimates else 0.0,
    }


def aggregate_generator_policy_evidence_matrix(
    summary_json_paths: Iterable[str | Path],
) -> list[dict[str, object]]:
    raw_rows = [_extract_policy_evidence_row(summary_json_path) for summary_json_path in summary_json_paths]
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        workload_class = str(row.get("workload_class") or "").strip()
        stream_length = int(row.get("stream_length") or 0)
        generator = str(row.get("generator") or "").strip()
        if not workload_class or stream_length <= 0 or not generator:
            continue
        grouped[(workload_class, stream_length, generator)].append(row)

    rows: list[dict[str, object]] = []
    for (workload_class, stream_length, generator), group_rows in sorted(grouped.items()):
        row_metrics = [float(row["median_abs_error_vs_raw_exact"]) for row in group_rows]
        row_relative_errors = [float(row["median_relative_error_vs_raw_exact"]) for row in group_rows]
        row_overestimates = [float(row["overestimate_fraction"]) for row in group_rows]
        sample_indices = sorted(
            {
                int(row["sample_index"])
                for row in group_rows
                if row.get("sample_index") is not None and str(row.get("sample_index")).strip() != ""
            }
        )
        replay_kinds = sorted({kind for row in group_rows for kind in row.get("replay_kinds", [])})
        rows.append(
            {
                "workload_class": workload_class,
                "model_family": workload_class,
                "stream_length": stream_length,
                "generator": generator,
                "summary_count": len(group_rows),
                "sample_count": sum(int(row.get("sample_count") or 0) for row in group_rows),
                "sample_indices": sample_indices,
                "capture_row_count_total": sum(int(row["capture_row_count"]) for row in group_rows),
                "replay_row_count_total": sum(int(row["replay_row_count"]) for row in group_rows),
                "replay_kinds": replay_kinds,
                "median_abs_error_vs_raw_exact": statistics.median(row_metrics) if row_metrics else 0.0,
                "max_abs_error_vs_raw_exact": max(float(row["max_abs_error_vs_raw_exact"]) for row in group_rows)
                if group_rows
                else 0.0,
                "median_abs_error_vs_captured_output": statistics.median(
                    float(row["median_abs_error_vs_captured_output"]) for row in group_rows
                )
                if group_rows
                else 0.0,
                "max_abs_error_vs_captured_output": max(
                    float(row["max_abs_error_vs_captured_output"]) for row in group_rows
                )
                if group_rows
                else 0.0,
                "median_relative_error_vs_raw_exact": statistics.median(row_relative_errors)
                if row_relative_errors
                else 0.0,
                "overestimate_fraction": statistics.mean(row_overestimates) if row_overestimates else 0.0,
            }
        )

    ranked_rows: list[dict[str, object]] = []
    region_groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        region_groups[(str(row["workload_class"]), int(row["stream_length"]))].append(row)

    for (workload_class, stream_length), region_rows in sorted(region_groups.items()):
        ranked_region_rows = sorted(
            region_rows,
            key=lambda row: (
                float(row["median_abs_error_vs_raw_exact"]),
                str(row["generator"]),
            ),
        )
        best_metric = min(float(row["median_abs_error_vs_raw_exact"]) for row in ranked_region_rows)
        best_rows = [
            row
            for row in ranked_region_rows
            if abs(float(row["median_abs_error_vs_raw_exact"]) - best_metric) <= 1e-12
        ]
        runner_up_metric = None
        for row in ranked_region_rows:
            metric = float(row["median_abs_error_vs_raw_exact"])
            if metric > best_metric + 1e-12:
                runner_up_metric = metric
                break
        for index, row in enumerate(ranked_region_rows, start=1):
            metric = float(row["median_abs_error_vs_raw_exact"])
            row_rank = 1 + sum(
                1
                for other in ranked_region_rows
                if float(other["median_abs_error_vs_raw_exact"]) + 1e-12 < metric
            )
            row["generator_rank"] = row_rank
            row["best_generator"] = best_rows[0]["generator"] if len(best_rows) == 1 else ""
            row["best_generator_count"] = len(best_rows)
            row["best_generator_metric"] = best_metric
            row["runner_up_metric"] = runner_up_metric
            row["metric_margin_vs_best"] = metric - best_metric
            row["region_policy_state"] = (
                "default_with_supporting_assumptions"
                if len(best_rows) == 1 and row_rank == 1
                else ("supporting_comparator" if len(best_rows) == 1 else "mixed_unresolved")
            )
            row["default_generator"] = row["generator"] if len(best_rows) == 1 and row_rank == 1 else ""
            row["region_key"] = f"{workload_class}|{stream_length}"
            ranked_rows.append(row)

    return ranked_rows


def write_generator_policy_evidence_matrix_csv(
    out_csv: str | Path,
    summary_json_paths: Iterable[str | Path],
) -> list[dict[str, object]]:
    rows = aggregate_generator_policy_evidence_matrix(summary_json_paths)
    fieldnames = [
        "workload_class",
        "model_family",
        "stream_length",
        "generator",
        "summary_count",
        "sample_count",
        "sample_indices",
        "capture_row_count_total",
        "replay_row_count_total",
        "replay_kinds",
        "median_abs_error_vs_raw_exact",
        "max_abs_error_vs_raw_exact",
        "median_abs_error_vs_captured_output",
        "max_abs_error_vs_captured_output",
        "median_relative_error_vs_raw_exact",
        "overestimate_fraction",
        "generator_rank",
        "best_generator",
        "best_generator_count",
        "best_generator_metric",
        "runner_up_metric",
        "metric_margin_vs_best",
        "region_policy_state",
        "default_generator",
        "region_key",
    ]
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _stringify_values(value)
                    if isinstance(value, (list, tuple, set))
                    else value
                    for field, value in row.items()
                    if field in fieldnames
                }
            )
    return rows


def summarize_generator_policy_pack(
    summary_json_paths: Iterable[str | Path],
) -> dict[str, object]:
    matrix_rows = aggregate_generator_policy_evidence_matrix(summary_json_paths)
    policy = resolve_generator_default_policy(matrix_rows)
    model_families = sorted({str(row["model_family"]) for row in matrix_rows if str(row["model_family"]).strip()})
    slice_families = sorted(
        {
            kind
            for row in matrix_rows
            for kind in row.get("replay_kinds", [])
            if str(kind).strip()
        }
    )
    stream_lengths = sorted({int(row["stream_length"]) for row in matrix_rows if int(row["stream_length"]) > 0})
    generators = sorted({str(row["generator"]) for row in matrix_rows if str(row["generator"]).strip()})
    return {
        "policy_kind": policy["policy_kind"],
        "policy_state": policy["policy_state"],
        "repository_default_generator": policy["repository_default_generator"],
        "summary_json_count": sum(int(row.get("sample_count") or 0) for row in matrix_rows),
        "sample_count": sum(int(row.get("sample_count") or 0) for row in matrix_rows),
        "model_families": model_families,
        "slice_families": slice_families,
        "stream_lengths": stream_lengths,
        "generators": generators,
        "regional_default_generators": policy["regional_default_generators"],
        "regional_default_generator_counts": policy["regional_default_generator_counts"],
        "resolved_region_count": policy["resolved_region_count"],
        "region_count": len(policy["regions"]),
    }


def write_generator_policy_pack_summary_json(
    out_json: str | Path,
    summary_json_paths: Iterable[str | Path],
) -> dict[str, object]:
    summary = summarize_generator_policy_pack(summary_json_paths)
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def aggregate_smoke_summary_envelopes(
    summary_json_paths: Iterable[str | Path],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_json_path in summary_json_paths:
        path = Path(summary_json_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        semantics = payload.get("semantics") or {}
        replay_rows = [
            row
            for row in (payload.get("replay") or [])
            if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "replayed"
        ]
        abs_raw_errors = [
            float(row["abs_error_vs_raw_exact"])
            for row in replay_rows
            if row.get("abs_error_vs_raw_exact") is not None
        ]
        abs_captured_errors = [
            float(row["abs_error_vs_captured_output"])
            for row in replay_rows
            if row.get("abs_error_vs_captured_output") is not None
        ]
        relative_errors = []
        overestimates = []
        for row in replay_rows:
            raw_exact = row.get("raw_exact_dot_product")
            abs_error = row.get("abs_error_vs_raw_exact")
            estimated = row.get("rescaled_estimated_dot_product")
            if raw_exact is not None and abs_error is not None:
                denom = max(1.0, abs(float(raw_exact)))
                relative_errors.append(float(abs_error) / denom)
            if raw_exact is not None and estimated is not None:
                overestimates.append(1.0 if float(estimated) > float(raw_exact) else 0.0)

        rows.append(
            {
                "summary_json": str(path),
                "artifact_name": path.name,
                "generator": str(semantics.get("generator") or ""),
                "stream_length": int(semantics.get("stream_length") or 0),
                "capture_row_count": int(payload.get("capture_row_count") or 0),
                "replay_row_count": len(replay_rows),
                "median_abs_error_vs_raw_exact": statistics.median(abs_raw_errors)
                if abs_raw_errors
                else 0.0,
                "max_abs_error_vs_raw_exact": max(abs_raw_errors) if abs_raw_errors else 0.0,
                "median_abs_error_vs_captured_output": statistics.median(abs_captured_errors)
                if abs_captured_errors
                else 0.0,
                "max_abs_error_vs_captured_output": max(abs_captured_errors)
                if abs_captured_errors
                else 0.0,
                "median_relative_error": statistics.median(relative_errors)
                if relative_errors
                else 0.0,
                "overestimate_fraction": (sum(overestimates) / len(overestimates))
                if overestimates
                else 0.0,
            }
        )
    return rows


def write_smoke_calibration_pack_csv(
    out_csv: str | Path,
    summary_json_paths: Iterable[str | Path],
) -> list[dict[str, object]]:
    rows = aggregate_smoke_summary_envelopes(summary_json_paths)
    fieldnames = [
        "summary_json",
        "artifact_name",
        "generator",
        "stream_length",
        "capture_row_count",
        "replay_row_count",
        "median_abs_error_vs_raw_exact",
        "max_abs_error_vs_raw_exact",
        "median_abs_error_vs_captured_output",
        "max_abs_error_vs_captured_output",
        "median_relative_error",
        "overestimate_fraction",
    ]
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return rows
