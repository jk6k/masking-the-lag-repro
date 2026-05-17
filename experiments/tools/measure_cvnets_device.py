#!/usr/bin/env python3
"""Measure local CPU/MPS device latency, power, and energy for CVNets models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from exp_common.model_specs import MODEL_SPECS

DEFAULT_TORCH_EVAL_SCRIPT = ROOT / "experiments" / "accuracy" / "eval_cvnets_imagenet_noise.py"
DEFAULT_MLX_EVAL_SCRIPT = ROOT / "experiments" / "accuracy" / "eval_mlx_imagenet_noise.py"
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "results" / "report_data"
OUTPUT_FIELDS = [
    "workload_id",
    "model",
    "latency_ms",
    "avg_power_w",
    "energy_j",
    "latency_source",
    "latency_measurement_window",
    "power_measurement_window",
    "energy_derivation",
    "comparison_boundary",
    "comparison_kind",
    "benchmark_equivalence",
    "measurement_evidence_type",
    "subprocess_elapsed_s",
    "quantized_eval_pass_elapsed_s",
    "quantized_eval_processed_samples",
    "quantized_eval_top1",
    "quantized_eval_top1_delta",
    "accuracy_results_csv",
    "batch_size",
    "sequence_length",
    "max_eval_samples",
    "host_name",
    "device_model",
    "accuracy_backend",
    "framework",
    "precision_mode",
    "power_sampler",
    "profiler_interval_ms",
]


@dataclass
class SamplerHandle:
    mode: str
    process: subprocess.Popen[str] | None = None
    admin_pid: int | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected YAML mapping in {path}")
    return payload


def _write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def _build_eval_command(
    *,
    python_bin: str,
    eval_script: Path,
    accuracy_backend: str,
    imagenet_val: str,
    model: str,
    device: str,
    weights_dir: str | None,
    weights_npz: str | None,
    results_csv: Path,
    run_id: str,
    workload_id: str,
    batch_size: int,
    max_eval_samples: int,
    quant_bits: int,
) -> list[str]:
    if accuracy_backend == "torch":
        command = [
            python_bin,
            str(eval_script),
            "--imagenet_val",
            imagenet_val,
            "--opencv_pipeline",
            "--models",
            model,
            "--device",
            device,
            "--results_csv",
            str(results_csv),
            "--run_id",
            run_id,
            "--workload",
            workload_id,
            "--profile",
            "device_measure_quantized",
            "--sweep_resolution",
            "single_point",
            "--eval_batch_size",
            str(batch_size),
            "--max_eval_samples",
            str(max_eval_samples),
            "--workers",
            "0",
            "--quant_bits",
            str(quant_bits),
            "--gaussian_noise_std",
            "0",
            "--crosstalk_alpha",
            "0",
            "--enable_attention",
        ]
        if weights_dir:
            command.extend(["--weights_dir", weights_dir])
        return command
    if accuracy_backend == "mlx":
        if device != "mps":
            raise SystemExit("MLX device measurement only supports --device mps.")
        command = [
            python_bin,
            str(eval_script),
            "--imagenet_val",
            imagenet_val,
            "--opencv_pipeline",
            "--models",
            model,
            "--device",
            "mps",
            "--results_csv",
            str(results_csv),
            "--run_id",
            run_id,
            "--workload",
            workload_id,
            "--profile",
            "device_measure_quantized",
            "--sweep_resolution",
            "single_point",
            "--eval_batch_size",
            str(batch_size),
            "--max_eval_samples",
            str(max_eval_samples),
            "--workers",
            "0",
            "--quant_bits",
            str(quant_bits),
            "--noise_sigma_lsb",
            "0",
            "--crosstalk_alpha",
            "0",
            "--enable_attention",
        ]
        if weights_npz:
            command.extend(["--weights_npz", weights_npz])
        elif weights_dir:
            command.extend(["--weights_dir", weights_dir])
        return command
    raise SystemExit(f"Unsupported accuracy backend: {accuracy_backend}")


def _load_weights_npz_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    exports = payload.get("exports")
    if isinstance(exports, list):
        mapping: dict[str, str] = {}
        for item in exports:
            if not isinstance(item, dict):
                continue
            model = str(item.get("model") or "").strip()
            weights_npz = str(item.get("weights_npz") or item.get("output_path") or "").strip()
            if model and weights_npz:
                mapping[model] = weights_npz
        return mapping
    model = str(payload.get("model") or "").strip()
    weights_npz = str(payload.get("weights_npz") or payload.get("output_path") or "").strip()
    if model and weights_npz:
        return {model: weights_npz}
    raise SystemExit(f"Unable to parse MLX weights manifest: {path}")


def _resolve_accuracy_backend(requested_backend: str | None, *, device: str) -> str:
    backend = (requested_backend or "").strip().lower()
    if backend:
        return backend
    if device == "mps":
        return "mlx"
    raise SystemExit(
        "CPU device measurement has no MLX backend. Explicit --accuracy_backend torch is "
        "maintenance-only and must not be relied on by the active final-freeze path."
    )


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_powermetrics_command(
    *,
    sampler: str,
    interval_ms: int,
    output_path: Path,
    use_sudo: bool,
    interactive_sudo: bool,
    device: str,
) -> list[str] | None:
    if sampler == "none":
        return None
    samplers = "cpu_power"
    if device == "mps":
        samplers = "gpu_power"
    command = [
        "powermetrics",
        "--samplers",
        samplers,
        "-i",
        str(max(50, interval_ms)),
        "--show-all",
    ]
    if use_sudo:
        command = ["sudo"] + ([] if interactive_sudo else ["-n"]) + command
    return command


def _applescript_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _command_with_absolute_binary(command: list[str]) -> list[str]:
    if not command:
        return command
    resolved = list(command)
    if resolved[0] == "powermetrics":
        resolved[0] = "/usr/bin/powermetrics"
    elif resolved[0] == "nohup":
        resolved[0] = "/usr/bin/nohup"
    return resolved


def _build_osascript_admin_background_command(
    *,
    command: list[str],
    output_path: Path,
) -> list[str]:
    bg_command = _command_with_absolute_binary(command)
    payload = (
        f"{' '.join(shlex.quote(part) for part in bg_command)} "
        f"> {shlex.quote(str(output_path))} 2>&1 < /dev/null & echo $!"
    )
    shell_command = f"/bin/sh -c {shlex.quote(payload)}"
    script = f'do shell script "{_applescript_escape(shell_command)}" with administrator privileges'
    return ["osascript", "-e", script]


def _build_osascript_admin_stop_command(pid: int) -> list[str]:
    shell_command = (
        f"/bin/kill -INT {int(pid)} >/dev/null 2>&1 || true; "
        f"/bin/sleep 1; "
        f"/bin/kill -TERM {int(pid)} >/dev/null 2>&1 || true"
    )
    script = f'do shell script "{_applescript_escape(shell_command)}" with administrator privileges'
    return ["osascript", "-e", script]


def _sampler_needs_tty(command: list[str] | None) -> bool:
    if not command:
        return False
    return bool(command and command[0] == "sudo" and "-n" not in command)


def _spawn_sampler(
    command: list[str] | None,
    *,
    output_path: Path,
    osascript_admin: bool,
) -> SamplerHandle | None:
    if not command:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if osascript_admin:
        if command[0] == "sudo":
            raise SystemExit("osascript admin powermetrics cannot be combined with sudo launcher mode")
        result = subprocess.check_output(
            _build_osascript_admin_background_command(command=command, output_path=output_path),
            text=True,
        ).strip()
        try:
            pid = int(result.splitlines()[-1].strip())
        except (IndexError, ValueError) as exc:
            raise SystemExit(f"Failed to capture osascript powermetrics PID from {result!r}") from exc
        return SamplerHandle(mode="osascript_admin", admin_pid=pid)
    handle = output_path.open("w", encoding="utf-8")
    needs_tty = _sampler_needs_tty(command)
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=None if needs_tty else os.setsid,
    )
    return SamplerHandle(mode="process", process=process)


def _stop_sampler(handle: SamplerHandle | None) -> None:
    if handle is None:
        return
    if handle.mode == "osascript_admin":
        if handle.admin_pid is None:
            return
        subprocess.run(_build_osascript_admin_stop_command(handle.admin_pid), check=True)
        time.sleep(1.0)
        return
    process = handle.process
    if process is None:
        return
    try:
        pgid = os.getpgid(process.pid)
        if pgid == process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            process.terminate()
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                return
        process.wait(timeout=5)


def _parse_powermetrics_average_power_w(path: Path) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "powermetrics must be invoked as the superuser" in text.lower():
        raise SystemExit(
            "powermetrics requires superuser privileges on this host; rerun "
            "measure_cvnets_device with --use_sudo_powermetrics"
        )
    values_mw: list[float] = []
    patterns = (
        r"(?:CPU|GPU)\s+Power:\s*([0-9.]+)\s*mW",
        r"(?:CPU|GPU)\s+HW active frequency.*?([0-9.]+)\s*mW",
    )
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            try:
                values_mw.append(float(match))
            except ValueError:
                continue
    if not values_mw:
        return None
    return sum(values_mw) / len(values_mw) / 1000.0


def _host_device_model(device: str) -> str:
    machine = platform.machine().strip() or "unknown"
    chip = ""
    if platform.system().lower() == "darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            chip = ""
    if device == "cpu":
        return chip or machine
    return f"Apple-{device.upper()}:{chip or machine}"


def _measure_once(
    *,
    command: list[str],
    sampler_command: list[str] | None,
    sampler_log: Path,
    osascript_admin_powermetrics: bool,
    max_eval_samples: int,
) -> tuple[float, float | None]:
    sampler = _spawn_sampler(
        sampler_command,
        output_path=sampler_log,
        osascript_admin=osascript_admin_powermetrics,
    )
    started = time.perf_counter()
    try:
        subprocess.run(command, cwd=str(ROOT), check=True)
    finally:
        elapsed_s = time.perf_counter() - started
        _stop_sampler(sampler)
    avg_power_w = _parse_powermetrics_average_power_w(sampler_log)
    return elapsed_s, avg_power_w


def _load_quantized_eval_measurement(
    *,
    path: Path,
    model: str,
    quant_bits: int,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if str(row.get("baseline") or "").strip().lower() == "true":
            continue
        if str(row.get("model") or "").strip() != model:
            continue
        row_bits = _to_float(row.get("quant_bits"), None)
        if row_bits is None or int(row_bits) != int(quant_bits):
            continue
        gaussian = _to_float(row.get("gaussian_noise_std"), None)
        crosstalk = _to_float(row.get("crosstalk_alpha"), None)
        if gaussian not in {0.0, None}:
            continue
        if crosstalk not in {0.0, None}:
            continue
        latency_ms = _to_float(row.get("latency_ms_per_sample"), None)
        pass_elapsed_s = _to_float(row.get("measured_pass_elapsed_s"), None)
        processed_samples = _to_float(row.get("measured_processed_samples"), None)
        if latency_ms is None and pass_elapsed_s is not None and processed_samples and processed_samples > 0:
            latency_ms = (pass_elapsed_s / processed_samples) * 1000.0
        return {
            "latency_ms": latency_ms,
            "pass_elapsed_s": pass_elapsed_s,
            "processed_samples": int(processed_samples) if processed_samples is not None else None,
            "top1": _to_float(row.get("top1"), None),
            "top1_delta": _to_float(row.get("top1_delta"), None),
            "measurement_window": str(row.get("measurement_window") or "").strip() or "quantized_eval_pass",
        }
    return None


def _derive_metric_fields(
    *,
    subprocess_elapsed_s: float,
    avg_power_w: float | None,
    max_eval_samples: int,
    quantized_measurement: dict[str, Any] | None,
    latency_policy: str,
) -> dict[str, Any]:
    latency_ms = (subprocess_elapsed_s / float(max(1, int(max_eval_samples)))) * 1000.0
    latency_source = "whole_subprocess_wall_clock_div_max_eval_samples"
    latency_measurement_window = "whole_subprocess_div_samples"
    energy_j = None if avg_power_w is None else avg_power_w * (latency_ms / 1e3)
    energy_derivation = "avg_power_w * latency_ms / 1e3"
    comparison_boundary = "local_real_device_subprocess_latency_with_host_runtime_power"
    quantized_eval_pass_elapsed_s = None
    quantized_eval_processed_samples = None
    quantized_eval_top1 = None
    quantized_eval_top1_delta = None

    if quantized_measurement is not None:
        quantized_eval_pass_elapsed_s = quantized_measurement.get("pass_elapsed_s")
        quantized_eval_processed_samples = quantized_measurement.get("processed_samples")
        quantized_eval_top1 = quantized_measurement.get("top1")
        quantized_eval_top1_delta = quantized_measurement.get("top1_delta")

    if latency_policy == "quantized_eval_pass":
        if quantized_measurement is None:
            raise SystemExit(
                "Requested quantized_eval_pass latency policy, but no matching quantized "
                "measurement row was found in accuracy_results_csv."
            )
        measured_latency_ms = quantized_measurement.get("latency_ms")
        pass_elapsed_s = quantized_measurement.get("pass_elapsed_s")
        processed_samples = quantized_measurement.get("processed_samples")
        if measured_latency_ms is None:
            if pass_elapsed_s is None or processed_samples is None or processed_samples <= 0:
                raise SystemExit(
                    "quantized_eval_pass latency policy requires quantized latency or "
                    "elapsed/processed sample fields in accuracy_results_csv."
                )
            measured_latency_ms = (float(pass_elapsed_s) / float(processed_samples)) * 1000.0
        latency_ms = float(measured_latency_ms)
        latency_source = "accuracy_results_csv.quantized_eval_pass_div_processed_samples"
        latency_measurement_window = "quantized_eval_pass_div_processed_samples"
        comparison_boundary = "local_real_device_quantized_eval_pass_with_host_runtime_power"
        if avg_power_w is None:
            energy_j = None
            energy_derivation = "avg_power_w unavailable"
        elif pass_elapsed_s is not None and processed_samples is not None and processed_samples > 0:
            energy_j = avg_power_w * (float(pass_elapsed_s) / float(processed_samples))
            energy_derivation = "avg_power_w * quantized_eval_pass_elapsed_s / quantized_eval_processed_samples"
        else:
            energy_j = avg_power_w * (latency_ms / 1e3)
            energy_derivation = "avg_power_w * latency_ms / 1e3"

    return {
        "latency_ms": latency_ms,
        "latency_source": latency_source,
        "latency_measurement_window": latency_measurement_window,
        "energy_j": energy_j,
        "energy_derivation": energy_derivation,
        "comparison_boundary": comparison_boundary,
        "quantized_eval_pass_elapsed_s": quantized_eval_pass_elapsed_s,
        "quantized_eval_processed_samples": quantized_eval_processed_samples,
        "quantized_eval_top1": quantized_eval_top1,
        "quantized_eval_top1_delta": quantized_eval_top1_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure local CPU/MPS device metrics for CVNets evaluation.")
    parser.add_argument("--imagenet_val", required=True)
    parser.add_argument("--model", default="mobilevit_s", choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--device", required=True, choices=["cpu", "mps"])
    parser.add_argument(
        "--accuracy_backend",
        choices=["torch", "mlx"],
        default=None,
        help="Defaults to MLX for --device mps. CPU has no MLX path; explicit torch is maintenance-only.",
    )
    parser.add_argument("--results_csv", type=Path, required=True)
    parser.add_argument("--weights_dir", default=None)
    parser.add_argument("--weights_npz", default=None)
    parser.add_argument("--weights_npz_manifest", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval_script", type=Path, default=None)
    parser.add_argument("--workload_id", default="W0_mobilevit_imagenet")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=197)
    parser.add_argument("--max_eval_samples", type=int, default=64)
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--precision_mode", default="int8_eval")
    parser.add_argument("--latency_policy", choices=["whole_subprocess", "quantized_eval_pass"], default="whole_subprocess")
    parser.add_argument("--power_sampler", choices=["none", "powermetrics"], default="powermetrics")
    parser.add_argument("--profiler_interval_ms", type=int, default=200)
    parser.add_argument("--use_sudo_powermetrics", action="store_true")
    parser.add_argument("--interactive_sudo_powermetrics", action="store_true")
    parser.add_argument("--osascript_admin_powermetrics", action="store_true")
    parser.add_argument("--sampler_log", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.osascript_admin_powermetrics and (args.use_sudo_powermetrics or args.interactive_sudo_powermetrics):
        raise SystemExit(
            "--osascript_admin_powermetrics cannot be combined with "
            "--use_sudo_powermetrics/--interactive_sudo_powermetrics"
        )
    resolved_accuracy_backend = _resolve_accuracy_backend(args.accuracy_backend, device=args.device)

    run_id = f"device_measure_{args.device}_{args.model}"
    accuracy_results_csv = args.results_csv.with_name(f"{args.results_csv.stem}_accuracy.csv")
    sampler_log = args.sampler_log or args.results_csv.with_name(f"{args.results_csv.stem}_powermetrics.log")
    eval_script = args.eval_script
    if eval_script is None:
        eval_script = (
            DEFAULT_MLX_EVAL_SCRIPT
            if resolved_accuracy_backend == "mlx"
            else DEFAULT_TORCH_EVAL_SCRIPT
        )
    weights_npz_by_model = _load_weights_npz_manifest(args.weights_npz_manifest)
    weights_npz = args.weights_npz
    if resolved_accuracy_backend == "mlx" and not weights_npz:
        weights_npz = weights_npz_by_model.get(args.model)

    command = _build_eval_command(
        python_bin=args.python,
        eval_script=eval_script,
        accuracy_backend=resolved_accuracy_backend,
        imagenet_val=args.imagenet_val,
        model=args.model,
        device=args.device,
        weights_dir=args.weights_dir,
        weights_npz=weights_npz,
        results_csv=accuracy_results_csv,
        run_id=run_id,
        workload_id=args.workload_id,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
        quant_bits=args.quant_bits,
    )
    sampler_command = _build_powermetrics_command(
        sampler=args.power_sampler,
        interval_ms=args.profiler_interval_ms,
        output_path=sampler_log,
        use_sudo=(args.use_sudo_powermetrics or args.interactive_sudo_powermetrics),
        interactive_sudo=args.interactive_sudo_powermetrics,
        device=args.device,
    )
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        if sampler_command:
            print(" ".join(shlex.quote(part) for part in sampler_command))
        return

    subprocess_elapsed_s, avg_power_w = _measure_once(
        command=command,
        sampler_command=sampler_command,
        sampler_log=sampler_log,
        osascript_admin_powermetrics=args.osascript_admin_powermetrics,
        max_eval_samples=max(1, int(args.max_eval_samples)),
    )
    quantized_measurement = _load_quantized_eval_measurement(
        path=accuracy_results_csv,
        model=args.model,
        quant_bits=args.quant_bits,
    )
    metric_fields = _derive_metric_fields(
        subprocess_elapsed_s=subprocess_elapsed_s,
        avg_power_w=avg_power_w,
        max_eval_samples=args.max_eval_samples,
        quantized_measurement=quantized_measurement,
        latency_policy=args.latency_policy,
    )
    if args.power_sampler == "powermetrics" and avg_power_w is None:
        raise SystemExit(
            "powermetrics completed without usable power samples; check sampler "
            "permissions or rerun with --use_sudo_powermetrics"
        )
    power_measurement_window = (
        "whole_subprocess_powermetrics"
        if args.power_sampler == "powermetrics"
        else "none"
    )
    row = {
        "workload_id": args.workload_id,
        "model": args.model,
        "latency_ms": metric_fields["latency_ms"],
        "avg_power_w": avg_power_w,
        "energy_j": metric_fields["energy_j"],
        "latency_source": metric_fields["latency_source"],
        "latency_measurement_window": metric_fields["latency_measurement_window"],
        "power_measurement_window": power_measurement_window,
        "energy_derivation": metric_fields["energy_derivation"],
        "comparison_boundary": metric_fields["comparison_boundary"],
        "comparison_kind": "contextual_real_device_reference",
        "benchmark_equivalence": False,
        "measurement_evidence_type": "measured",
        "subprocess_elapsed_s": subprocess_elapsed_s,
        "quantized_eval_pass_elapsed_s": metric_fields["quantized_eval_pass_elapsed_s"],
        "quantized_eval_processed_samples": metric_fields["quantized_eval_processed_samples"],
        "quantized_eval_top1": metric_fields["quantized_eval_top1"],
        "quantized_eval_top1_delta": metric_fields["quantized_eval_top1_delta"],
        "accuracy_results_csv": str(accuracy_results_csv),
        "batch_size": int(args.batch_size),
        "sequence_length": int(args.sequence_length),
        "max_eval_samples": int(args.max_eval_samples),
        "host_name": platform.node() or "unknown",
        "device_model": _host_device_model(args.device),
        "accuracy_backend": resolved_accuracy_backend,
        "framework": "mlx" if resolved_accuracy_backend == "mlx" else "cvnets+pytorch",
        "precision_mode": args.precision_mode,
        "power_sampler": args.power_sampler,
        "profiler_interval_ms": int(args.profiler_interval_ms),
    }
    _write_csv(args.results_csv, row)
    print(f"[measure-cvnets-device] wrote {args.results_csv}")


if __name__ == "__main__":
    main()
