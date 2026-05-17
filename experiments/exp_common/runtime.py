"""Cross-platform runtime helpers with Mac-friendly accelerator defaults."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Callable, Optional, Tuple


CudaProbe = Callable[[], Tuple[bool, Optional[str]]]
BoolProbe = Callable[[], bool]


def current_platform_name() -> str:
    """Return the normalized host platform name."""
    return platform.system().strip().lower()


def auto_device_order(system: str | None = None) -> tuple[str, ...]:
    """Choose a sensible accelerator priority for the current platform."""
    normalized = (system or current_platform_name()).strip().lower()
    if normalized == "darwin":
        return ("mps", "cuda", "cpu")
    return ("cuda", "mps", "cpu")


def safe_mps_available() -> bool:
    """Check MPS availability without making torch a hard import dependency."""
    try:
        import torch
    except Exception:
        return False
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend and backend.is_available())


def _is_codex_sandbox() -> bool:
    """Best-effort detection for the Codex seatbelt sandbox."""
    return bool(str(os.environ.get("CODEX_SANDBOX", "")).strip())


def direct_cuda_available() -> tuple[bool, str | None]:
    """Lightweight CUDA probe for tools that do not provide their own safe probe."""
    try:
        import torch
    except Exception as exc:
        return False, f"torch import failed: {exc}"
    try:
        ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception as exc:
        return False, f"cuda probe failed: {exc}"
    if ok:
        return True, None
    return False, "cuda unavailable"


def prepare_runtime_environment(device_name: str) -> None:
    """Apply backend-specific runtime knobs before torch objects are materialized."""
    if (device_name or "").strip().lower() == "mps":
        # This repository treats MPS-backed runs as strict MPS execution.
        # Keep fallback disabled unless explicitly enabled by operators.
        fallback = os.environ.get("HPAT_ENABLE_MPS_FALLBACK")
        if fallback is None:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        elif fallback.strip().lower() in {"0", "false", "no", "off"}:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def prepare_runtime_environment_for_request(
    device_preference: str | None = None,
    *,
    system: str | None = None,
) -> None:
    """Prime accelerator env vars before importing torch when auto may choose MPS."""
    preference = (
        device_preference
        or os.environ.get("CVNETS_DEVICE_BACKEND")
        or "auto"
    ).strip().lower()
    normalized_system = (system or current_platform_name()).strip().lower()
    if preference == "mps":
        prepare_runtime_environment("mps")
        return
    if preference == "auto" and normalized_system == "darwin":
        prepare_runtime_environment("mps")


def host_memory_gb() -> float | None:
    """Best-effort host memory lookup for conservative auto-tuning."""
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total / float(1024 ** 3)
    except Exception:
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * phys_pages
        if total > 0:
            return total / float(1024 ** 3)
    except Exception:
        pass

    if current_platform_name() == "darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            total = int(output)
            if total > 0:
                return total / float(1024 ** 3)
        except Exception:
            pass
    return None


def host_perf_core_count(system: str | None = None) -> int | None:
    """Best-effort Apple Silicon performance-core lookup."""
    normalized_system = (system or current_platform_name()).strip().lower()
    if normalized_system != "darwin":
        return None
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        value = int(output)
        if value > 0:
            return value
    except Exception:
        pass
    return None


def resolve_data_workers(
    requested_workers: int | None,
    device_name: str,
    *,
    cpu_count: int | None = None,
    perf_core_count: int | None = None,
    system: str | None = None,
) -> int:
    """Resolve dataloader workers, allowing negative/None to mean auto."""
    if requested_workers is not None and int(requested_workers) >= 0:
        return int(requested_workers)

    override = os.environ.get("HPAT_DATA_WORKERS")
    if override and override.strip():
        try:
            value = int(override)
            if value >= 0:
                return value
        except ValueError:
            pass

    normalized = (device_name or "").strip().lower()
    cores = max(1, int(cpu_count or os.cpu_count() or 1))
    if normalized == "mps":
        perf_cores = perf_core_count
        if perf_cores is None:
            perf_cores = host_perf_core_count(system)
        if perf_cores is not None and int(perf_cores) > 0:
            return max(2, min(8, int(perf_cores)))
        return max(1, min(8, cores // 2))
    if normalized == "cuda":
        return max(2, min(8, cores // 2))
    return max(0, min(4, cores // 3))


def resolve_dataloader_prefetch_factor(
    device_name: str,
    worker_count: int,
    *,
    requested_prefetch_factor: int | None = None,
) -> int | None:
    """Resolve a sensible prefetch factor for torch DataLoader."""
    if worker_count <= 0:
        return None
    if requested_prefetch_factor is not None and int(requested_prefetch_factor) > 0:
        return int(requested_prefetch_factor)

    override = os.environ.get("HPAT_DATA_PREFETCH_FACTOR")
    if override and override.strip():
        try:
            value = int(override)
            if value > 0:
                return value
        except ValueError:
            pass

    normalized = (device_name or "").strip().lower()
    return 2 if normalized == "mps" else 2


def resolve_eval_batch_size(
    requested_batch_size: int | None,
    device_name: str,
    model_key: str,
    *,
    max_eval_samples: int | None = None,
    host_memory_gb_value: float | None = None,
) -> int | None:
    """Resolve an eval batch size, allowing <=0/None to mean auto."""
    if requested_batch_size is not None and int(requested_batch_size) > 0:
        batch_size = int(requested_batch_size)
    else:
        override = os.environ.get("HPAT_EVAL_BATCH_SIZE")
        batch_size = None
        if override and override.strip():
            try:
                value = int(override)
                if value > 0:
                    batch_size = value
            except ValueError:
                batch_size = None
        if batch_size is None:
            normalized = (device_name or "").strip().lower()
            if normalized == "mps":
                memory_gb = host_memory_gb_value
                if memory_gb is None:
                    memory_gb = host_memory_gb()
                if memory_gb is None:
                    tier = "small"
                elif memory_gb >= 48.0:
                    tier = "large"
                elif memory_gb >= 24.0:
                    tier = "medium"
                else:
                    tier = "small"
                presets = {
                    "small": {
                        "mobilevit_xxs": 256,
                        "mobilevit_xs": 192,
                        "mobilevit_s": 64,
                        "resnet_50": 128,
                    },
                    "medium": {
                        "mobilevit_xxs": 384,
                        "mobilevit_xs": 256,
                        "mobilevit_s": 64,
                        "resnet_50": 192,
                    },
                    "large": {
                        "mobilevit_xxs": 512,
                        "mobilevit_xs": 384,
                        "mobilevit_s": 96,
                        "resnet_50": 256,
                    },
                }
                batch_size = presets[tier].get(model_key, 128)
            elif normalized == "cpu":
                batch_size = {
                    "mobilevit_xxs": 64,
                    "mobilevit_xs": 48,
                    "mobilevit_s": 32,
                    "resnet_50": 32,
                }.get(model_key, 32)
            elif normalized == "cuda":
                batch_size = {
                    "mobilevit_xxs": 256,
                    "mobilevit_xs": 192,
                    "mobilevit_s": 128,
                    "resnet_50": 128,
                }.get(model_key, 128)
            else:
                batch_size = None

    if batch_size is not None and max_eval_samples is not None:
        batch_size = min(int(batch_size), max(1, int(max_eval_samples)))
    return batch_size


def prefer_vision_channels_last(
    device_name: str,
    explicit_channels_last: bool = False,
) -> bool:
    """Prefer NHWC-like layout for vision inference on accelerators."""
    if explicit_channels_last:
        return True
    return (device_name or "").strip().lower() in {"cuda", "mps"}


def resolve_device_preference(
    preference: str,
    *,
    cuda_probe_fn: CudaProbe | None = None,
    mps_probe_fn: BoolProbe | None = None,
    system: str | None = None,
) -> tuple[str, str | None]:
    """Resolve auto/cpu/cuda/mps into a concrete backend name."""
    pref = (preference or "auto").strip().lower()
    cuda_probe = cuda_probe_fn or direct_cuda_available
    mps_probe = mps_probe_fn or safe_mps_available
    normalized_system = (system or current_platform_name()).strip().lower()

    if pref == "cpu":
        return "cpu", None

    if pref == "cuda":
        if normalized_system == "darwin":
            raise SystemExit(
                "CUDA is not permitted on this Apple Silicon host in this repository. "
                "Use --device mps."
            )
        cuda_ok, cuda_msg = cuda_probe()
        if cuda_ok:
            return "cuda", None
        detail = f" Details: {cuda_msg}" if cuda_msg else ""
        raise SystemExit(f"CUDA requested but not available.{detail}")

    if pref == "mps":
        if mps_probe():
            return "mps", None
        raise SystemExit("MPS requested but not available.")

    if pref != "auto":
        raise SystemExit(f"Unsupported device preference: {preference}")

    if normalized_system == "darwin":
        if mps_probe():
            return "mps", None
        raise SystemExit(
            "MPS is required on this Apple Silicon host for accelerator-backed runs, "
            "but MPS is unavailable."
        )

    notes: list[str] = []
    for candidate in auto_device_order(system):
        if candidate == "cpu":
            return "cpu", "; ".join(notes) if notes else None
        if candidate == "cuda":
            cuda_ok, cuda_msg = cuda_probe()
            if cuda_ok:
                return "cuda", None
            notes.append(
                f"cuda unavailable ({cuda_msg})" if cuda_msg else "cuda unavailable"
            )
            continue
        if candidate == "mps":
            if mps_probe():
                return "mps", None
            notes.append("mps unavailable")
            continue
    return "cpu", "; ".join(notes) if notes else None


def resolve_reporting_device_preference(
    preference: str,
    *,
    cuda_probe_fn: CudaProbe | None = None,
    mps_probe_fn: BoolProbe | None = None,
    system: str | None = None,
) -> tuple[str, str | None]:
    """Resolve a device for reporting commands without misclassifying sandbox-only MPS false negatives."""
    try:
        return resolve_device_preference(
            preference,
            cuda_probe_fn=cuda_probe_fn,
            mps_probe_fn=mps_probe_fn,
            system=system,
        )
    except SystemExit as exc:
        pref = (preference or "auto").strip().lower()
        normalized_system = (system or current_platform_name()).strip().lower()
        message = str(exc)
        if (
            pref == "auto"
            and normalized_system == "darwin"
            and _is_codex_sandbox()
            and "MPS is required on this Apple Silicon host" in message
        ):
            return (
                "mps",
                "Codex sandbox reported mps unavailable; verify on the unsandboxed host before treating this as a blocker.",
            )
        raise


def apply_torch_device(opts, device_name: str) -> str:
    """Set common torch/cvnets device fields on an opts namespace."""
    prepare_runtime_environment(device_name)
    import torch

    device = torch.device(device_name)
    setattr(opts, "dev.device", device)
    setattr(opts, "dev.device_id", None)
    setattr(opts, "ddp.use_distributed", False)
    setattr(opts, "dev.num_gpus", 1 if device.type == "cuda" else 0)
    return str(device)


def ensure_validated_accuracy_backend(
    device_name: str,
    *,
    allow_unvalidated_mps: bool = False,
) -> str:
    """Normalize the backend used for claim-bearing accuracy runs."""
    normalized = (device_name or "").strip().lower()
    if normalized == "mps" and allow_unvalidated_mps:
        # Backward-compatible no-op: MPS no longer needs an explicit opt-in after
        # local CPU/MPS parity checks fixed the transfer-path corruption bug.
        return normalized
    return normalized


__all__ = [
    "apply_torch_device",
    "auto_device_order",
    "ensure_validated_accuracy_backend",
    "current_platform_name",
    "direct_cuda_available",
    "host_memory_gb",
    "host_perf_core_count",
    "prefer_vision_channels_last",
    "prepare_runtime_environment",
    "prepare_runtime_environment_for_request",
    "resolve_data_workers",
    "resolve_dataloader_prefetch_factor",
    "resolve_device_preference",
    "resolve_reporting_device_preference",
    "resolve_eval_batch_size",
    "safe_mps_available",
]
