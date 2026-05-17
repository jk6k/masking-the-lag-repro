"""Resolve a repo-local Python interpreter across Mac/Windows/Linux layouts."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Iterable


def _existing_path(path: Path) -> str | None:
    if path.is_file():
        return str(path)
    return None


def _which(name: str) -> str | None:
    return shutil.which(name)


def iter_repo_python_candidates(
    repo_root: Path,
    *,
    system: str | None = None,
    env: dict[str, str] | None = None,
) -> Iterable[str]:
    normalized_system = (system or platform.system()).strip().lower()
    active_env = env or dict(os.environ)

    explicit = (active_env.get("HPAT_REPO_PYTHON") or "").strip()
    if explicit:
        yield explicit

    virtual_env = (active_env.get("VIRTUAL_ENV") or "").strip()
    if virtual_env:
        if normalized_system == "windows":
            yield str(Path(virtual_env) / "Scripts" / "python.exe")
        else:
            yield str(Path(virtual_env) / "bin" / "python")

    if normalized_system == "darwin":
        for candidate in (
            repo_root / ".venv311-mps" / "bin" / "python",
            repo_root / ".venv-macos" / "bin" / "python",
            repo_root / ".venv" / "bin" / "python",
        ):
            found = _existing_path(candidate)
            if found:
                yield found
        for name in ("python3.11", "python3", "python"):
            found = _which(name)
            if found:
                yield found
        return

    if normalized_system == "windows":
        for candidate in (
            repo_root / ".venv" / "Scripts" / "python.exe",
            repo_root / ".venv311-mps" / "Scripts" / "python.exe",
        ):
            found = _existing_path(candidate)
            if found:
                yield found
        for name in ("python", "py"):
            found = _which(name)
            if found:
                yield found
        return

    for candidate in (
        repo_root / ".venv" / "bin" / "python",
        repo_root / ".venv311-mps" / "bin" / "python",
        repo_root / ".venv-macos" / "bin" / "python",
    ):
        found = _existing_path(candidate)
        if found:
            yield found
    for name in ("python3", "python"):
        found = _which(name)
        if found:
            yield found


def resolve_repo_python(
    repo_root: Path,
    *,
    system: str | None = None,
    env: dict[str, str] | None = None,
) -> str:
    for candidate in iter_repo_python_candidates(repo_root, system=system, env=env):
        if candidate:
            return candidate
    return "python3"


__all__ = [
    "iter_repo_python_candidates",
    "resolve_repo_python",
]
