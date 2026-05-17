#!/usr/bin/env python3
"""Prepare an annotated measured-accuracy source for a bitstream phase1 config."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT_DIR / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from accuracy.annotate_bitstream_accuracy_csv import annotate_accuracy_rows
from accuracy.bitstream_semantics import (
    BITSTREAM_BRIDGE_MEASUREMENT_TRUTH_CLASS,
    BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS,
    normalize_bitstream_semantics,
)
from tools.check_bitstream_measured_row_eligibility import (
    assess_row_eligibility,
    select_target_row,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_csv_with_fieldnames(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _parse_match_filters(raw_filters: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in raw_filters:
        if "=" not in item:
            raise SystemExit(f"Invalid --match filter: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --match filter: {item!r}")
        parsed[key] = value.strip()
    return parsed


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _to_int(value: Any, default: int | None = None) -> int | None:
    if value in {None, ""}:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_execution_semantics_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve the bitstream semantics fields needed by this preparation tool."""
    raw_cfg = cfg.get("bitstream") or {}
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise SystemExit("Expected optional 'bitstream' config to be a mapping.")

    requested_semantics = str(raw_cfg.get("execution_semantics") or "").strip().lower()
    default_semantics = str(raw_cfg.get("default_execution_semantics") or "proxy").strip().lower()
    if default_semantics not in {"proxy", "bitstream"}:
        raise SystemExit(f"Unsupported default execution semantics: {default_semantics!r}")
    if requested_semantics not in {"", "proxy", "bitstream"}:
        raise SystemExit(f"Unsupported execution semantics: {requested_semantics!r}")
    if requested_semantics:
        resolved_semantics = requested_semantics
    elif _to_bool(raw_cfg.get("enabled"), False):
        resolved_semantics = "bitstream"
    else:
        resolved_semantics = default_semantics

    if resolved_semantics == "proxy":
        return {
            "execution_semantics": "proxy",
            "bitstream_encoding_mode": None,
            "bitstream_multiplier_mode": None,
            "bitstream_stream_length": None,
            "bitstream_generator": None,
            "bitstream_accumulator_mode": None,
            "bitstream_calibration_source": None,
        }

    stream_length = _to_int(raw_cfg.get("stream_length"), None)
    if stream_length is None or stream_length <= 0:
        raise SystemExit(
            "Bitstream execution semantics require a positive bitstream.stream_length."
        )
    return {
        "execution_semantics": "bitstream",
        "bitstream_encoding_mode": str(raw_cfg.get("encoding_mode") or "bipolar").strip().lower(),
        "bitstream_multiplier_mode": str(raw_cfg.get("multiplier_mode") or "xnor").strip().lower(),
        "bitstream_stream_length": stream_length,
        "bitstream_generator": str(raw_cfg.get("generator") or "bernoulli").strip().lower(),
        "bitstream_accumulator_mode": str(raw_cfg.get("accumulator_mode") or "bitcount").strip().lower(),
        "bitstream_calibration_source": (
            str(raw_cfg.get("calibration_source")).strip()
            if raw_cfg.get("calibration_source") not in {None, ""}
            else None
        ),
    }


def _default_match_filters(cfg: dict[str, Any]) -> dict[str, str]:
    filters: dict[str, str] = {"baseline": "false"}
    models = list((cfg.get("models") or {}).get("keys") or [])
    if len(models) == 1 and str(models[0]).strip():
        filters["model"] = str(models[0]).strip()
    workload_id = str((cfg.get("data") or {}).get("workload_id") or "").strip()
    if workload_id:
        filters["workload"] = workload_id
    return filters


def _detect_context_run_id(
    rows: list[dict[str, Any]],
    *,
    match_filters: dict[str, str],
) -> str | None:
    matched_source_run_ids: list[str] = []
    for row in rows:
        matched = True
        for key, expected in match_filters.items():
            if str(row.get(key) or "").strip().lower() != str(expected).strip().lower():
                matched = False
                break
        if not matched:
            continue
        source_run_id = str(row.get("source_run_id") or row.get("run_id") or "").strip()
        if source_run_id and source_run_id not in matched_source_run_ids:
            matched_source_run_ids.append(source_run_id)
    if len(matched_source_run_ids) == 1:
        return matched_source_run_ids[0]
    return None


def _row_identity(row: dict[str, Any]) -> tuple[str, ...] | None:
    run_id = str(row.get("run_id") or "").strip()
    if run_id:
        return ("run_id", run_id)
    fallback = tuple(
        str(row.get(field) or "").strip()
        for field in (
            "source_run_id",
            "baseline",
            "workload",
            "model",
            "measurement_window",
            "seed",
        )
    )
    return ("row",) + fallback if any(fallback) else None


def _merge_existing_annotations(
    rows: list[dict[str, str]],
    *,
    output_csv: Path,
) -> list[dict[str, str]]:
    if not rows or not output_csv.exists():
        return rows
    existing_fieldnames, existing_rows = _read_csv_with_fieldnames(output_csv)
    if not existing_rows:
        return rows

    existing_by_identity: dict[tuple[str, ...], dict[str, str]] = {}
    for existing_row in existing_rows:
        identity = _row_identity(existing_row)
        if identity is not None:
            existing_by_identity[identity] = existing_row

    fieldnames = list(rows[0].keys())
    for field in existing_fieldnames:
        if field not in fieldnames:
            fieldnames.append(field)

    merged_rows: list[dict[str, str]] = []
    for row in rows:
        merged = dict(row)
        existing = existing_by_identity.get(_row_identity(row))
        if existing is not None:
            for field, value in existing.items():
                if field in merged and str(merged.get(field) or "").strip():
                    continue
                if str(value or "").strip():
                    merged[field] = value
        merged_rows.append({field: str(merged.get(field) or "") for field in fieldnames})
    return merged_rows


def _drop_explicit_accuracy_coupling(cfg: dict[str, Any]) -> None:
    accuracy_cfg = cfg.setdefault("accuracy", {})
    if not isinstance(accuracy_cfg, dict):
        raise SystemExit("Expected config.accuracy to be a mapping.")
    for key in (
        "coupling",
        "accuracy_coupling",
        "coupling_evidence_type",
        "accuracy_coupling_evidence_type",
        "coupling_metric",
        "coupling_source",
        "coupling_reason",
        "coupling_execution_semantics",
        "coupling_bitstream_generator",
        "coupling_bitstream_stream_length",
    ):
        accuracy_cfg.pop(key, None)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_eligibility_markdown(
    path: Path,
    *,
    input_csv: Path,
    payload: dict[str, Any],
) -> None:
    lines = [
        "# Bitstream Measured Source Preparation Eligibility",
        "",
        f"- input_csv: `{input_csv}`",
        f"- run_id: `{payload['run_id']}`",
        f"- measurement_window: `{payload['measurement_window']}`",
        f"- promotable_measured_row_eligible: `{payload['promotable_measured_row_eligible']}`",
        f"- bitstream_measurement_truth_class: `{payload['bitstream_measurement_truth_class']}`",
        f"- bitstream_runtime_claim_surface_status: `{payload['bitstream_runtime_claim_surface_status']}`",
        f"- bitstream_truth_class_authorization_status: `{payload['bitstream_truth_class_authorization_status']}`",
    ]
    blockers = list(payload.get("blockers") or [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assess_measured_row_eligibility(
    *,
    rows: list[dict[str, str]],
    match_filters: dict[str, str],
    input_csv: Path,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict[str, Any]:
    run_id = match_filters.get("run_id")
    measurement_window = match_filters.get("measurement_window", "quantized_eval_pass")
    target_row = select_target_row(
        rows,
        run_id=run_id,
        measurement_window=measurement_window,
    )
    payload = assess_row_eligibility(target_row)
    if output_json is not None:
        _write_json(output_json, payload)
    if output_md is not None:
        _write_eligibility_markdown(output_md, input_csv=input_csv, payload=payload)
    return payload


def _prepared_config_copy(
    cfg: dict[str, Any],
    *,
    annotated_csv: Path,
    semantics: dict[str, Any],
    context_run_id: str | None,
    contract_note: str,
    measurement_truth_class: str,
    authorization_note: str,
    authorization_status: str,
    drop_explicit_coupling: bool,
) -> dict[str, Any]:
    prepared = copy.deepcopy(cfg)
    prepared.setdefault("accuracy", {})
    if not isinstance(prepared["accuracy"], dict):
        raise SystemExit("Expected config.accuracy to be a mapping.")
    accuracy_cfg = prepared["accuracy"]
    accuracy_cfg["source_csv"] = str(annotated_csv)
    if context_run_id:
        accuracy_cfg["context_run_id"] = context_run_id
    measurement_contract = dict(accuracy_cfg.get("measurement_contract") or {})
    measurement_contract.update(
        {
            "source": str(annotated_csv),
            "execution_semantics": semantics["execution_semantics"],
            "bitstream_generator": semantics["bitstream_generator"],
            "bitstream_stream_length": semantics["bitstream_stream_length"],
            "bitstream_measurement_truth_class": measurement_truth_class,
        }
    )
    if authorization_note:
        measurement_contract["bitstream_truth_class_authorization_note"] = (
            authorization_note
        )
    if authorization_status:
        measurement_contract["bitstream_truth_class_authorization_status"] = (
            authorization_status
        )
    if contract_note:
        measurement_contract["note"] = contract_note
    accuracy_cfg["measurement_contract"] = measurement_contract
    if drop_explicit_coupling:
        _drop_explicit_accuracy_coupling(prepared)
    run_cfg = prepared.setdefault("run", {})
    existing_notes = str(run_cfg.get("notes") or "").strip()
    bridge_note = f"measured_accuracy_bridge:{annotated_csv.name}"
    if bridge_note not in existing_notes:
        run_cfg["notes"] = f"{existing_notes}; {bridge_note}".strip("; ").strip()
    return prepared


def prepare_bitstream_measured_accuracy_source(
    *,
    phase1_config: Path,
    input_csv: Path,
    output_csv: Path,
    output_config: Path | None = None,
    raw_match_filters: list[str] | None = None,
    explicit_context_run_id: str | None = None,
    contract_note: str = "",
    measurement_truth_class: str = BITSTREAM_BRIDGE_MEASUREMENT_TRUTH_CLASS,
    extra_annotation_fields: dict[str, Any] | None = None,
    drop_explicit_coupling: bool = True,
    eligibility_report_json: Path | None = None,
    eligibility_report_md: Path | None = None,
) -> dict[str, Any]:
    cfg = _load_yaml(phase1_config)
    execution_semantics_cfg = _resolve_execution_semantics_cfg(cfg)
    if execution_semantics_cfg["execution_semantics"] != "bitstream":
        raise SystemExit(
            f"Config {phase1_config} does not resolve to bitstream execution semantics."
        )
    semantics_cfg = normalize_bitstream_semantics(
        {
            "execution_semantics": execution_semantics_cfg["execution_semantics"],
            "encoding_mode": execution_semantics_cfg["bitstream_encoding_mode"],
            "multiplier_mode": execution_semantics_cfg["bitstream_multiplier_mode"],
            "accumulator_mode": execution_semantics_cfg["bitstream_accumulator_mode"],
            "stream_length": execution_semantics_cfg["bitstream_stream_length"],
            "generator": execution_semantics_cfg["bitstream_generator"],
            "calibration_source": execution_semantics_cfg["bitstream_calibration_source"],
        }
    )
    match_filters = _default_match_filters(cfg)
    match_filters.update(_parse_match_filters(list(raw_match_filters or [])))
    rows = _read_csv(input_csv)
    rows = _merge_existing_annotations(rows, output_csv=output_csv)
    eligibility_payload: dict[str, Any] | None = None
    if (
        measurement_truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
        or eligibility_report_json is not None
        or eligibility_report_md is not None
    ):
        eligibility_payload = _assess_measured_row_eligibility(
            rows=rows,
            match_filters=match_filters,
            input_csv=input_csv,
            output_json=eligibility_report_json,
            output_md=eligibility_report_md,
        )
        if (
            measurement_truth_class == BITSTREAM_MODEL_LEVEL_MEASURED_TRUTH_CLASS
            and not bool(eligibility_payload.get("promotable_measured_row_eligible"))
        ):
            blockers = ", ".join(str(item) for item in eligibility_payload.get("blockers") or [])
            raise SystemExit(
                "Refusing to prepare a model-level measured source from a non-promotable row. "
                f"Blockers: {blockers or 'unknown'}"
            )
    fieldnames, annotated_rows, matched_count = annotate_accuracy_rows(
        rows,
        semantics_cfg=semantics_cfg,
        row_filters=match_filters,
        contract_note=contract_note,
        measurement_truth_class=measurement_truth_class,
        extra_fields=extra_annotation_fields,
    )
    _write_csv(output_csv, fieldnames, annotated_rows)
    context_run_id = explicit_context_run_id or _detect_context_run_id(
        rows,
        match_filters=match_filters,
    )
    if output_config is not None:
        prepared_cfg = _prepared_config_copy(
            cfg,
            annotated_csv=output_csv,
            semantics={
                "execution_semantics": execution_semantics_cfg["execution_semantics"],
                "bitstream_generator": execution_semantics_cfg["bitstream_generator"],
                "bitstream_stream_length": execution_semantics_cfg["bitstream_stream_length"],
            },
            context_run_id=context_run_id,
            contract_note=contract_note,
            measurement_truth_class=measurement_truth_class,
            authorization_note=str(
                (eligibility_payload or {}).get("bitstream_truth_class_authorization_note")
                or ""
            ),
            authorization_status=str(
                (eligibility_payload or {}).get("bitstream_truth_class_authorization_status")
                or ""
            ),
            drop_explicit_coupling=drop_explicit_coupling,
        )
        _write_yaml(output_config, prepared_cfg)
    return {
        "matched_count": matched_count,
        "context_run_id": context_run_id,
        "output_csv": str(output_csv),
        "output_config": str(output_config) if output_config is not None else "",
        "match_filters": match_filters,
        "eligibility_report_json": str(eligibility_report_json or ""),
        "eligibility_report_md": str(eligibility_report_md or ""),
        "eligibility_payload": eligibility_payload or {},
        "extra_annotation_fields": dict(extra_annotation_fields or {}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate a measured accuracy CSV for a bitstream phase1 config.",
    )
    parser.add_argument("--phase1_config", type=Path, required=True)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_config", type=Path, default=None)
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Additional row filter in key=value form. Default filters use baseline=false plus config model/workload/split when available.",
    )
    parser.add_argument("--context_run_id", default=None)
    parser.add_argument("--contract_note", default="")
    parser.add_argument(
        "--measurement_truth_class",
        default=BITSTREAM_BRIDGE_MEASUREMENT_TRUTH_CLASS,
    )
    parser.add_argument("--eligibility_report_json", type=Path, default=None)
    parser.add_argument("--eligibility_report_md", type=Path, default=None)
    parser.add_argument(
        "--keep_explicit_coupling",
        action="store_true",
        help="Preserve existing accuracy.coupling fields when writing --output_config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prepare_bitstream_measured_accuracy_source(
        phase1_config=args.phase1_config,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_config=args.output_config,
        raw_match_filters=args.match,
        explicit_context_run_id=args.context_run_id,
        contract_note=args.contract_note,
        measurement_truth_class=args.measurement_truth_class,
        drop_explicit_coupling=not args.keep_explicit_coupling,
        eligibility_report_json=args.eligibility_report_json,
        eligibility_report_md=args.eligibility_report_md,
    )
    print(
        "Prepared bitstream measured-accuracy source: "
        f"matched_rows={result['matched_count']} output_csv={result['output_csv']} "
        f"context_run_id={result['context_run_id'] or ''}"
    )
    if result["output_config"]:
        print(f"Prepared config copy: {result['output_config']}")


if __name__ == "__main__":
    main()
