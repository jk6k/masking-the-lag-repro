"""Shared authorization helpers for model-level measured bitstream truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZATION_MARKER = (
    "bitstream_model_level_measured_authorized: true"
)
BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZED_STATUS = "authorized"
BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZED_RUN_ID_KEY = "authorized_run_id"


def resolve_truth_class_authorization_note(
    raw: object,
    *,
    search_roots: tuple[Path, ...] = (),
) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(root / path for root in search_roots)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def authorization_note_allows_model_level_measured(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZATION_MARKER in payload


def build_truth_class_authorization_note_text(
    *,
    authorized_run_id: str,
    extra_fields: dict[str, object] | None = None,
) -> str:
    run_id = str(authorized_run_id or "").strip()
    if not run_id:
        raise ValueError("authorized_run_id must be non-empty.")
    lines = [
        "# Bitstream Model-Level Measured Authorization",
        "",
        BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZATION_MARKER,
        f"{BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZED_RUN_ID_KEY}: {run_id}",
    ]
    for key, value in (extra_fields or {}).items():
        normalized_key = str(key or "").strip().lower()
        normalized_value = str(value or "").strip()
        if not normalized_key or not normalized_value:
            continue
        lines.append(f"{normalized_key}: {normalized_value}")
    return "\n".join(lines) + "\n"


def write_truth_class_authorization_note(
    path: Path,
    *,
    authorized_run_id: str,
    extra_fields: dict[str, object] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_truth_class_authorization_note_text(
            authorized_run_id=authorized_run_id,
            extra_fields=extra_fields,
        ),
        encoding="utf-8",
    )
    return path


def parse_truth_class_authorization_note(path: Path | None) -> dict[str, Any]:
    resolved_path = str(path) if path is not None else ""
    if path is None:
        return {
            "resolved_path": resolved_path,
            "marker_present": False,
            "authorized_run_id": "",
            "fields": {},
        }
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError:
        return {
            "resolved_path": resolved_path,
            "marker_present": False,
            "authorized_run_id": "",
            "fields": {},
        }

    fields: dict[str, str] = {}
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower()
        if normalized_key and normalized_key not in fields:
            fields[normalized_key] = value.strip()
    return {
        "resolved_path": resolved_path,
        "marker_present": BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZATION_MARKER in payload,
        "authorized_run_id": str(fields.get(BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZED_RUN_ID_KEY) or "").strip(),
        "fields": fields,
    }


def assess_truth_class_authorization(
    path: Path | None,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    payload = parse_truth_class_authorization_note(path)
    expected = str(expected_run_id or "").strip()
    status = "authorized"
    if path is None:
        status = "missing"
    elif not payload["marker_present"]:
        status = "marker_missing"
    elif expected:
        authorized_run_id = str(payload["authorized_run_id"] or "").strip()
        if not authorized_run_id:
            status = "run_id_missing"
        elif authorized_run_id != expected:
            status = "run_id_mismatch"
    return {
        **payload,
        "expected_run_id": expected,
        "status": status,
        "authorized": status == BITSTREAM_MODEL_LEVEL_MEASURED_AUTHORIZED_STATUS,
    }
