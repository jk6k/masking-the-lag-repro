"""Lightweight local project progress snapshot helpers."""

from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from exp_common.assets import collect_dataset_status, collect_weight_status
from exp_common.runtime import resolve_reporting_device_preference


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROGRESS_DIR = ROOT / "experiments" / "results" / "dev" / "progress_sync"
DEFAULT_RECENT_LIMIT = 8
_ENDPOINT_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_endpoint_name(name: str | None) -> str:
    normalized = _ENDPOINT_TOKEN_RE.sub("-", str(name or "").strip()).strip("-._")
    return normalized.lower() or "unknown-endpoint"


def default_endpoint_name() -> str:
    return sanitize_endpoint_name(socket.gethostname().split(".", 1)[0])


def snapshot_path(endpoint: str, out_dir: Path = DEFAULT_PROGRESS_DIR) -> Path:
    return Path(out_dir) / f"{sanitize_endpoint_name(endpoint)}.json"


def _safe_git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path)


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _recent_paths(root: Path, patterns: Iterable[str], *, limit: int) -> list[dict[str, object]]:
    if limit <= 0 or not root.exists():
        return []

    by_path: dict[Path, Path] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                by_path[path.resolve()] = path

    ordered = sorted(
        by_path.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    rows: list[dict[str, object]] = []
    for path in ordered[:limit]:
        stat = path.stat()
        rows.append(
            {
                "path": _repo_rel(path),
                "size_bytes": stat.st_size,
                "mtime_utc": _iso_utc(stat.st_mtime),
            }
        )
    return rows


def collect_recent_outputs(*, limit: int = DEFAULT_RECENT_LIMIT) -> list[dict[str, object]]:
    per_group = max(1, limit // 3)
    groups = [
        (
            "dev_outputs",
            ROOT / "experiments" / "results" / "dev",
            ("**/*.csv", "**/*.json", "**/*.md"),
        ),
        (
            "quick_reports",
            ROOT / "experiments" / "results" / "quick_reports",
            ("*/compliance_report.json",),
        ),
        (
            "phase1_runs",
            ROOT / "experiments" / "results" / "runs",
            ("*/phase1_summary.csv", "*/master_metrics.csv"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for label, root, patterns in groups:
        for row in _recent_paths(root, patterns, limit=per_group):
            row["label"] = label
            rows.append(row)

    rows.sort(key=lambda row: str(row.get("mtime_utc", "")), reverse=True)
    return rows[:limit]


def collect_asset_summary() -> dict[str, object]:
    dataset_rows = collect_dataset_status()
    weight_rows = collect_weight_status()
    dataset_missing = [str(row["key"]) for row in dataset_rows if not bool(row["exists"])]
    weight_missing = [str(row["key"]) for row in weight_rows if not bool(row["exists"])]
    return {
        "datasets": {
            "present": len(dataset_rows) - len(dataset_missing),
            "total": len(dataset_rows),
            "missing": dataset_missing,
        },
        "weights": {
            "present": len(weight_rows) - len(weight_missing),
            "total": len(weight_rows),
            "missing": weight_missing,
        },
    }


def collect_git_summary(*, sample_limit: int = 8) -> dict[str, object]:
    branch = _safe_git_output("rev-parse", "--abbrev-ref", "HEAD")
    commit = _safe_git_output("rev-parse", "--short", "HEAD")
    porcelain = _safe_git_output("status", "--short") or ""
    lines = [line for line in porcelain.splitlines() if line.strip()]
    modified = 0
    untracked = 0
    sample_paths: list[str] = []

    for line in lines:
        code = line[:2]
        path = line[3:].strip()
        if code == "??":
            untracked += 1
        else:
            modified += 1
        if len(sample_paths) < sample_limit:
            sample_paths.append(f"{code} {path}")

    ahead = behind = None
    upstream = _safe_git_output("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    if upstream:
        counts = _safe_git_output("rev-list", "--left-right", "--count", f"{upstream}...HEAD")
        if counts:
            try:
                behind_str, ahead_str = counts.split()
                behind = int(behind_str)
                ahead = int(ahead_str)
            except Exception:
                ahead = behind = None

    return {
        "branch": branch or "unknown",
        "commit": commit or "unknown",
        "dirty": bool(lines),
        "modified_count": modified,
        "untracked_count": untracked,
        "ahead_count": ahead,
        "behind_count": behind,
        "sample_paths": sample_paths,
    }


def build_progress_snapshot(
    *,
    endpoint: str | None = None,
    note: str | None = None,
    recent_limit: int = DEFAULT_RECENT_LIMIT,
) -> dict[str, object]:
    resolved_device, device_note = resolve_reporting_device_preference("auto")
    now = datetime.now(tz=timezone.utc)
    return {
        "endpoint": sanitize_endpoint_name(endpoint or default_endpoint_name()),
        "captured_at_utc": now.isoformat(),
        "host": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "repo": {
            "root": str(ROOT),
            "auto_device": resolved_device,
            "auto_device_note": device_note,
        },
        "git": collect_git_summary(),
        "assets": collect_asset_summary(),
        "recent_outputs": collect_recent_outputs(limit=recent_limit),
        "note": str(note or "").strip(),
    }


def write_progress_snapshot(
    snapshot: dict[str, object],
    *,
    out_dir: Path = DEFAULT_PROGRESS_DIR,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(str(snapshot.get("endpoint") or default_endpoint_name()), out_dir=out_dir)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_progress_snapshots(out_dir: Path = DEFAULT_PROGRESS_DIR) -> list[dict[str, object]]:
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    rows.sort(key=lambda row: str(row.get("captured_at_utc", "")), reverse=True)
    return rows


def render_progress_status(snapshots: Iterable[dict[str, object]]) -> str:
    rows = list(snapshots)
    if not rows:
        return "[progress] no snapshots available"

    lines: list[str] = []
    for row in rows:
        endpoint = str(row.get("endpoint") or "unknown")
        captured = str(row.get("captured_at_utc") or "unknown-time")
        repo = row.get("repo") if isinstance(row.get("repo"), dict) else {}
        git = row.get("git") if isinstance(row.get("git"), dict) else {}
        assets = row.get("assets") if isinstance(row.get("assets"), dict) else {}
        datasets = assets.get("datasets") if isinstance(assets.get("datasets"), dict) else {}
        weights = assets.get("weights") if isinstance(assets.get("weights"), dict) else {}
        lines.append(
            "[progress] "
            f"{endpoint} | {captured} | auto={repo.get('auto_device', 'unknown')} | "
            f"{git.get('branch', 'unknown')}@{git.get('commit', 'unknown')} | "
            f"dirty={git.get('dirty', False)} | "
            f"datasets={datasets.get('present', 0)}/{datasets.get('total', 0)} | "
            f"weights={weights.get('present', 0)}/{weights.get('total', 0)}"
        )
        note = str(row.get("note") or "").strip()
        if note:
            lines.append(f"  note: {note}")
        sample_paths = git.get("sample_paths") if isinstance(git.get("sample_paths"), list) else []
        if sample_paths:
            lines.append(f"  changes: {', '.join(str(item) for item in sample_paths[:3])}")
        recent = row.get("recent_outputs") if isinstance(row.get("recent_outputs"), list) else []
        if recent:
            preview = []
            for item in recent[:3]:
                if not isinstance(item, dict):
                    continue
                preview.append(f"{item.get('label', 'artifact')}:{item.get('path', '-')}")
            if preview:
                lines.append(f"  recent: {', '.join(preview)}")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_PROGRESS_DIR",
    "DEFAULT_RECENT_LIMIT",
    "build_progress_snapshot",
    "collect_asset_summary",
    "collect_git_summary",
    "collect_recent_outputs",
    "default_endpoint_name",
    "load_progress_snapshots",
    "render_progress_status",
    "sanitize_endpoint_name",
    "snapshot_path",
    "write_progress_snapshot",
]
