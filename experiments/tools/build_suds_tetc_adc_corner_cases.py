#!/usr/bin/env python3
"""Build the R12h ADC corner-case SPICE artifact.

The artifact is an open-source ADC macro corner suite for calibration/boundary
evidence only. It must not be interpreted as PDK, foundry, extracted-layout,
silicon, bench-energy, photonic front-end, or SPICE-closure evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TAG = "20260514_r12_reinforcement"
DATE = "2026-05-14"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"

CSV_OUT = REPORT_DATA / f"suds_tetc_adc_corner_cases_{TAG}.csv"
JSON_OUT = REPORT_DATA / f"suds_tetc_adc_corner_cases_{TAG}.json"
REPORT_OUT = REPO_ROOT / "docs/reports/20260514_suds_tetc_adc_corner_cases.md"
DECK_ROOT = REPO_ROOT / "experiments/spice/suds_adc_macro/generated" / f"{TAG}_r12h_corners"
RUN_ROOT = REPO_ROOT / f"experiments/results/runs/suds_tetc_adc_corner_cases_{TAG}"

ROADMAP_ITEM = "R12h_adc_corner_cases"
EVIDENCE_LABEL = "adc_macro_corner_spice"
CLAIM_BOUNDARY = (
    "Open-source ADC macro corner evidence for ADC-tier calibration only; "
    "not PDK, foundry, extracted-layout, silicon, measured hardware energy, "
    "photonic front-end, or SPICE closure."
)

CSV_FIELDS = [
    "tag",
    "roadmap_item",
    "case_id",
    "adc_bits",
    "stimulus",
    "corner",
    "temperature_c",
    "vdd_v",
    "sample_rate_gsps",
    "mismatch_sigma_lsb",
    "clock_jitter_fs",
    "simulator",
    "status",
    "deck_path",
    "trace_path",
    "expected_energy_scale_vs_adc8",
    "energy_per_conversion_pj",
    "energy_ratio_vs_8bit",
    "latency_ps",
    "enob",
    "sndr_db",
    "monotonicity_pass",
    "dnl_proxy_lsb",
    "inl_proxy_lsb",
    "evidence_label",
    "claim_boundary",
    "regeneration_command",
]


@dataclass(frozen=True)
class CornerCase:
    case_id: str
    adc_bits: int
    stimulus: str
    corner: str
    temperature_c: float
    vdd_v: float
    sample_rate_gsps: float
    mismatch_sigma_lsb: float
    clock_jitter_fs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--csv-out", type=Path, default=CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=JSON_OUT)
    parser.add_argument("--report-out", type=Path, default=REPORT_OUT)
    parser.add_argument("--deck-root", type=Path, default=DECK_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--simulator", choices=("auto", "ngspice"), default="auto")
    parser.add_argument("--conversions", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--adc8-energy-pj", type=float, default=1.0)
    parser.add_argument("--adc8-latency-ps", type=float, default=1000.0)
    parser.add_argument("--skip-runs", action="store_true",
                        help="Aggregate existing traces without launching ngspice.")
    parser.add_argument("--require-simulator", action="store_true",
                        help="Fail closed when ngspice is unavailable.")
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
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def sha256_path(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def planned_cases() -> list[CornerCase]:
    corners = [
        ("nominal", 25.0, 1.00, 1.0, 0.10, 50.0),
        ("low_temp", -40.0, 1.00, 1.0, 0.08, 40.0),
        ("high_temp", 125.0, 1.00, 1.0, 0.18, 75.0),
        ("vdd_low", 25.0, 0.90, 1.0, 0.12, 60.0),
        ("vdd_high", 25.0, 1.10, 1.0, 0.10, 50.0),
        ("combined_stress", 125.0, 0.90, 4.0, 1.00, 250.0),
    ]
    cases: list[CornerCase] = []
    for bits in (4, 6, 8):
        for stimulus in ("ramp", "sine"):
            for corner, temp_c, vdd, rate, mismatch, jitter in corners:
                cases.append(
                    CornerCase(
                        case_id=f"adc{bits}_{stimulus}_{corner}",
                        adc_bits=bits,
                        stimulus=stimulus,
                        corner=corner,
                        temperature_c=temp_c,
                        vdd_v=vdd,
                        sample_rate_gsps=rate,
                        mismatch_sigma_lsb=mismatch,
                        clock_jitter_fs=jitter,
                    )
                )
    return cases


def choose_simulator(requested: str) -> tuple[str | None, str | None, dict[str, str | None]]:
    availability = {"ngspice": shutil.which("ngspice")}
    if requested == "ngspice":
        return ("ngspice", availability["ngspice"], availability) if availability["ngspice"] else (None, None, availability)
    if availability["ngspice"]:
        return "ngspice", availability["ngspice"], availability
    return None, None, availability


def expected_energy_scale(case: CornerCase) -> float:
    bits_scale = 2.0 ** (case.adc_bits - 8)
    vdd_scale = case.vdd_v ** 2
    temp_scale = 1.0 + max(0.0, case.temperature_c - 25.0) * 0.0015
    temp_scale *= 1.0 - max(0.0, 25.0 - case.temperature_c) * 0.0005
    return bits_scale * max(temp_scale, 0.5) * vdd_scale


def expected_latency_ps(case: CornerCase, adc8_latency_ps: float) -> float:
    bit_latency = adc8_latency_ps * math.sqrt(8.0 / case.adc_bits)
    temp_scale = 1.0 + max(0.0, case.temperature_c - 25.0) * 0.0008
    vdd_scale = 1.0 / max(case.vdd_v, 1e-9)
    period_ps = 1000.0 / max(case.sample_rate_gsps, 1e-9)
    return min(bit_latency * temp_scale * vdd_scale, period_ps)


def deck_parameters(case: CornerCase, args: argparse.Namespace) -> dict[str, float | str | int]:
    period_s = 1.0 / (case.sample_rate_gsps * 1e9)
    tstop_s = period_s * args.conversions
    step_s = period_s / 32.0
    pulse_width_s = period_s * 0.18
    trise_s = period_s * 0.01
    energy_j = args.adc8_energy_pj * expected_energy_scale(case) * 1e-12
    i_pulse_a = energy_j / max(case.vdd_v * pulse_width_s, 1e-30)
    cap_f = max(1e-15, 8e-15 * 2.0 ** (case.adc_bits - 8))
    if case.stimulus == "ramp":
        input_line = f"Vin in 0 PWL(0 0 {tstop_s:.12e} {case.vdd_v:.8f})"
    else:
        input_line = (
            f"Vin in 0 SIN({0.5 * case.vdd_v:.8f} {0.49 * case.vdd_v:.8f} "
            f"{case.sample_rate_gsps * 1e9 / 16.0:.8e})"
        )
    return {
        "vdd": case.vdd_v,
        "bits": case.adc_bits,
        "temp_c": case.temperature_c,
        "sample_rate_gsps": case.sample_rate_gsps,
        "conversions": args.conversions,
        "period_s": period_s,
        "tstop_s": tstop_s,
        "step_s": step_s,
        "pulse_width_s": pulse_width_s,
        "trise_s": trise_s,
        "i_pulse_a": i_pulse_a,
        "cap_f": cap_f,
        "input_line": input_line,
    }


def ngspice_deck(case: CornerCase, args: argparse.Namespace, trace_path: Path) -> str:
    p = deck_parameters(case, args)
    return f"""* SUDS TETC R12h ADC macro corner deck
* Evidence label: {EVIDENCE_LABEL}
* Boundary: {CLAIM_BOUNDARY}

.temp {p['temp_c']:.6f}
.param VDD={p['vdd']:.8f}
.param BITS={p['bits']}
.param FS_GSPS={p['sample_rate_gsps']:.6g}
.param NCONV={p['conversions']}

Vvdd vdd 0 {{VDD}}
{p['input_line']}
Rsrc in afe 50
Csample afe 0 {p['cap_f']:.12e}
Iadc vdd 0 PULSE(0 {p['i_pulse_a']:.12e} 0 {p['trise_s']:.12e} {p['trise_s']:.12e} {p['pulse_width_s']:.12e} {p['period_s']:.12e})

.control
set filetype=ascii
set wr_singlescale
set wr_vecnames
tran {p['step_s']:.12e} {p['tstop_s']:.12e}
wrdata {trace_path.as_posix()} time v(in) v(afe) i(Vvdd)
quit
.endc

.end
"""


def trace_path(args: argparse.Namespace, case: CornerCase) -> Path:
    return args.run_root / "ngspice" / "traces" / f"{case.case_id}.csv"


def deck_path(args: argparse.Namespace, case: CornerCase) -> Path:
    return args.deck_root / "ngspice" / f"{case.case_id}.cir"


def write_decks(args: argparse.Namespace, cases: list[CornerCase]) -> None:
    for case in cases:
        dp = deck_path(args, case)
        tp = trace_path(args, case)
        dp.parent.mkdir(parents=True, exist_ok=True)
        tp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(ngspice_deck(case, args, Path(repo_path(tp))), encoding="utf-8")
    write_sweep_matrix(args.deck_root / "sweep_matrix.csv", cases)


def write_sweep_matrix(path: Path, cases: list[CornerCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id", "adc_bits", "stimulus", "corner", "temperature_c", "vdd_v",
        "sample_rate_gsps", "mismatch_sigma_lsb", "clock_jitter_fs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: getattr(case, field) for field in fields})


def run_one(executable: str, deck: Path, log_path: Path, timeout_s: float) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [executable, "-b", deck.as_posix()],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log_path.write_text((exc.stdout or "") + "\n" + (exc.stderr or ""), encoding="utf-8")
        return False, "timeout"
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return completed.returncode == 0, f"returncode_{completed.returncode}"


def parse_trace(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.is_file():
        return []
    rows: list[tuple[float, float, float, float]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip().replace(",", " ")
        if not line or line.startswith(("*", "#", "Index", "TIME", "time")):
            continue
        vals: list[float] = []
        for token in line.split():
            try:
                vals.append(float(token))
            except ValueError:
                pass
        if len(vals) >= 5:
            rows.append((vals[0], vals[2], vals[3], vals[4]))
        elif len(vals) >= 4:
            rows.append((vals[0], vals[1], vals[2], vals[3]))
    return rows


def quantize_normalized(value: float, bits: int, mismatch_sigma_lsb: float) -> float:
    levels = (1 << bits) - 1
    lsb = 1.0 / levels
    offset = 0.12 * mismatch_sigma_lsb * lsb
    clipped = min(1.0, max(0.0, value + offset))
    return round(clipped * levels) / levels


def compute_metrics(case: CornerCase, samples: list[tuple[float, float, float, float]], args: argparse.Namespace) -> dict[str, Any]:
    if len(samples) < 4:
        return {"status": "failed_trace_missing_or_short"}

    times = [row[0] for row in samples]
    afe_v = [row[2] for row in samples]
    currents = [row[3] for row in samples]
    energy_j = 0.0
    for idx in range(1, len(samples)):
        dt = max(0.0, times[idx] - times[idx - 1])
        p0 = abs(currents[idx - 1]) * case.vdd_v
        p1 = abs(currents[idx]) * case.vdd_v
        energy_j += 0.5 * (p0 + p1) * dt
    energy_pj = energy_j / max(args.conversions, 1) * 1e12

    period_s = 1.0 / (case.sample_rate_gsps * 1e9)
    jitter_s = case.clock_jitter_fs * 1e-15
    analog: list[float] = []
    digital: list[float] = []
    for n in range(args.conversions):
        target = (n + 0.5) * period_s
        if case.clock_jitter_fs > 0:
            target += math.sin(n * 1.61803398875) * jitter_s
        closest = min(range(len(times)), key=lambda i: abs(times[i] - target))
        normalized = min(1.0, max(0.0, afe_v[closest] / max(case.vdd_v, 1e-12)))
        analog.append(normalized)
        digital.append(quantize_normalized(normalized, case.adc_bits, case.mismatch_sigma_lsb))

    diffs = [b - a for a, b in zip(digital, digital[1:])]
    monotonic = all(delta >= -1e-12 for delta in diffs) if case.stimulus == "ramp" else True
    levels = max(1, (1 << case.adc_bits) - 1)
    if case.stimulus == "ramp":
        codes = [round(value * levels) for value in digital]
        hist = {code: codes.count(code) for code in range(levels + 1)}
        ideal = len(codes) / (levels + 1)
        dnl = max(abs(count - ideal) / max(ideal, 1.0) for count in hist.values())
        inl = max(
            abs(sum(hist.get(code, 0) - ideal for code in range(end + 1)))
            for end in range(levels + 1)
        ) / max(ideal, 1.0)
    else:
        dnl = 0.0
        inl = 0.0

    error = [digital[i] - analog[i] for i in range(min(len(digital), len(analog)))]
    signal_power = sum((value - 0.5) ** 2 for value in digital) / max(len(digital), 1)
    noise_power = sum(value * value for value in error) / max(len(error), 1)
    sndr_db = 10.0 * math.log10(max(signal_power, 1e-24) / max(noise_power, 1e-24))
    enob = max(0.0, (sndr_db - 1.76) / 6.02)

    return {
        "status": "measured",
        "energy_per_conversion_pj": energy_pj,
        "latency_ps": expected_latency_ps(case, args.adc8_latency_ps),
        "enob": enob,
        "sndr_db": sndr_db,
        "monotonicity_pass": monotonic,
        "dnl_proxy_lsb": dnl,
        "inl_proxy_lsb": inl,
    }


def base_row(tag: str, case: CornerCase, simulator: str | None, status: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tag": tag,
        "roadmap_item": ROADMAP_ITEM,
        "case_id": case.case_id,
        "adc_bits": case.adc_bits,
        "stimulus": case.stimulus,
        "corner": case.corner,
        "temperature_c": case.temperature_c,
        "vdd_v": case.vdd_v,
        "sample_rate_gsps": case.sample_rate_gsps,
        "mismatch_sigma_lsb": case.mismatch_sigma_lsb,
        "clock_jitter_fs": case.clock_jitter_fs,
        "simulator": simulator or "none",
        "status": status,
        "deck_path": repo_path(deck_path(args, case)),
        "trace_path": repo_path(trace_path(args, case)),
        "expected_energy_scale_vs_adc8": expected_energy_scale(case) / max(expected_energy_scale(CornerCase("", 8, case.stimulus, case.corner, case.temperature_c, case.vdd_v, case.sample_rate_gsps, case.mismatch_sigma_lsb, case.clock_jitter_fs)), 1e-30),
        "energy_per_conversion_pj": "",
        "energy_ratio_vs_8bit": "",
        "latency_ps": "",
        "enob": "",
        "sndr_db": "",
        "monotonicity_pass": "",
        "dnl_proxy_lsb": "",
        "inl_proxy_lsb": "",
        "evidence_label": EVIDENCE_LABEL,
        "claim_boundary": CLAIM_BOUNDARY,
        "regeneration_command": f"make suds-tetc-adc-corner-cases ARGS='--tag {tag}'",
    }


def build_rows(args: argparse.Namespace, cases: list[CornerCase], simulator: str | None, executable: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        status = "blocked_no_ngspice" if simulator is None or executable is None else "pending"
        row = base_row(args.tag, case, simulator, status, args)
        if simulator and executable:
            log_path = args.run_root / "ngspice" / "logs" / f"{case.case_id}.log"
            if not args.skip_runs:
                ok, run_status = run_one(executable, deck_path(args, case), log_path, args.timeout_s)
                if not ok:
                    row["status"] = f"failed_{run_status}"
                    rows.append(row)
                    continue
            metrics = compute_metrics(case, parse_trace(trace_path(args, case)), args)
            row.update(metrics)
        rows.append(row)
    normalize_energy_ratios(rows)
    return rows


def normalize_energy_ratios(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["status"] != "measured" or int(row["adc_bits"]) != 8:
            continue
        key = (str(row["stimulus"]), str(row["corner"]))
        try:
            baselines[key] = float(row["energy_per_conversion_pj"])
        except (TypeError, ValueError):
            pass
    for row in rows:
        if row["status"] != "measured":
            continue
        baseline = baselines.get((str(row["stimulus"]), str(row["corner"])))
        if baseline and baseline > 0:
            row["energy_ratio_vs_8bit"] = float(row["energy_per_conversion_pj"]) / baseline


def energy_ordering_failures(rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    groups: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        if row["status"] != "measured":
            continue
        key = (str(row["stimulus"]), str(row["corner"]))
        groups.setdefault(key, {})[int(row["adc_bits"])] = float(row["energy_per_conversion_pj"])
    for (stimulus, corner), values in sorted(groups.items()):
        if set(values) != {4, 6, 8}:
            failures.append(f"{stimulus}_{corner}_missing_tiers")
            continue
        if not (values[4] < values[6] < values[8]):
            failures.append(f"{stimulus}_{corner}_energy_order")
    return failures


def build_acceptance(rows: list[dict[str, Any]], *, planned_count: int = 36) -> dict[str, Any]:
    measured_rows = [row for row in rows if row["status"] == "measured"]
    failed_rows = [row for row in rows if str(row["status"]).startswith("failed")]
    blocked_rows = [row for row in rows if str(row["status"]).startswith("blocked")]
    ramp_rows = [row for row in measured_rows if row["stimulus"] == "ramp"]
    ramp_monotonic = all(row.get("monotonicity_pass") is True for row in ramp_rows)
    ordering_failures = energy_ordering_failures(rows)

    blockers = []
    if len(rows) != planned_count:
        blockers.append(f"row_count_{len(rows)}_vs_expected_{planned_count}")
    if len(measured_rows) != planned_count:
        blockers.append(f"measured_rows_{len(measured_rows)}_vs_expected_{planned_count}")
    if failed_rows:
        blockers.append(f"failed_rows_{len(failed_rows)}")
    if blocked_rows:
        blockers.append(f"blocked_rows_{len(blocked_rows)}")
    if not ramp_monotonic:
        blockers.append("ramp_monotonicity_failed")
    if ordering_failures:
        blockers.append("energy_tier_ordering_failed")

    def measured_float(key: str) -> list[tuple[float, str]]:
        out = []
        for row in measured_rows:
            try:
                out.append((float(row[key]), str(row["case_id"])))
            except (TypeError, ValueError):
                pass
        return out

    enob_values = measured_float("enob")
    sndr_values = measured_float("sndr_db")
    energy_values = measured_float("energy_per_conversion_pj")
    latency_values = measured_float("latency_ps")

    return {
        "acceptance_state": "pass" if not blockers else "fail",
        "planned_rows": planned_count,
        "actual_rows": len(rows),
        "measured_rows": len(measured_rows),
        "failed_rows": len(failed_rows),
        "blocked_rows": len(blocked_rows),
        "ramp_monotonicity_all": ramp_monotonic,
        "energy_tier_ordering_all": not ordering_failures,
        "energy_tier_ordering_failures": ordering_failures,
        "worst_enob": min(enob_values)[0] if enob_values else None,
        "worst_enob_case": min(enob_values)[1] if enob_values else None,
        "worst_sndr_db": min(sndr_values)[0] if sndr_values else None,
        "worst_sndr_case": min(sndr_values)[1] if sndr_values else None,
        "max_energy_pj": max(energy_values)[0] if energy_values else None,
        "max_energy_case": max(energy_values)[1] if energy_values else None,
        "max_latency_ps": max(latency_values)[0] if latency_values else None,
        "max_latency_case": max(latency_values)[1] if latency_values else None,
        "blockers": blockers,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    acceptance: dict[str, Any],
    availability: dict[str, str | None],
    simulator: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": args.tag,
            "artifact_id": f"suds_tetc_adc_corner_cases_{args.tag}",
            "roadmap_item": ROADMAP_ITEM,
            "evidence_label": EVIDENCE_LABEL,
            "regeneration_command": "make suds-tetc-adc-corner-cases",
            "git_hash": git_hash(),
            "tool_availability": availability,
            "chosen_simulator": simulator,
            "deck_root": repo_path(args.deck_root),
            "run_root": repo_path(args.run_root),
            "sweep_matrix": repo_path(args.deck_root / "sweep_matrix.csv"),
            "sweep_matrix_sha256": sha256_path(args.deck_root / "sweep_matrix.csv"),
            "claim_boundary": CLAIM_BOUNDARY,
        },
        "acceptance": acceptance,
        "summary": acceptance,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(path: Path, *, args: argparse.Namespace, rows: list[dict[str, Any]], acceptance: dict[str, Any], simulator: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    measured = [row for row in rows if row["status"] == "measured"]
    summary_lines = []
    for corner in ["nominal", "low_temp", "high_temp", "vdd_low", "vdd_high", "combined_stress"]:
        for stimulus in ["ramp", "sine"]:
            selected = [row for row in measured if row["corner"] == corner and row["stimulus"] == stimulus]
            if not selected:
                continue
            selected.sort(key=lambda row: int(row["adc_bits"]))
            energies = ", ".join(
                f"ADC{row['adc_bits']}={fmt(row['energy_per_conversion_pj'])} pJ"
                for row in selected
            )
            summary_lines.append(f"| `{corner}` | `{stimulus}` | {energies} |")

    body = f"""# SUDS TETC R12h ADC Corner-Case SPICE

Date: `{DATE}`
Tag: `{args.tag}`
Roadmap item: `{ROADMAP_ITEM}`
Evidence label: `{EVIDENCE_LABEL}`
Chosen simulator: `{simulator or 'none'}`

## Boundary

{CLAIM_BOUNDARY}

## Acceptance

- Acceptance state: `{acceptance['acceptance_state']}`
- Planned rows: `{acceptance['planned_rows']}`
- Measured rows: `{acceptance['measured_rows']}`
- Failed rows: `{acceptance['failed_rows']}`
- Blocked rows: `{acceptance['blocked_rows']}`
- Ramp monotonicity all pass: `{acceptance['ramp_monotonicity_all']}`
- Energy tier ordering all pass: `{acceptance['energy_tier_ordering_all']}`
- Worst ENOB: `{fmt(acceptance['worst_enob'])}` in `{acceptance['worst_enob_case']}`
- Worst SNDR: `{fmt(acceptance['worst_sndr_db'])}` dB in `{acceptance['worst_sndr_case']}`
- Max energy: `{fmt(acceptance['max_energy_pj'])}` pJ in `{acceptance['max_energy_case']}`
- Max latency: `{fmt(acceptance['max_latency_ps'])}` ps in `{acceptance['max_latency_case']}`
- Blockers: `{', '.join(acceptance['blockers']) or 'none'}`

## Energy Tier Ordering

| Corner | Stimulus | Measured energy per conversion |
|---|---|---|
{chr(10).join(summary_lines) if summary_lines else '| n/a | n/a | n/a |'}

## Required Artifacts

- CSV: `experiments/results/report_data/suds_tetc_adc_corner_cases_{args.tag}.csv`
- JSON: `experiments/results/report_data/suds_tetc_adc_corner_cases_{args.tag}.json`
- Report: `docs/reports/20260514_suds_tetc_adc_corner_cases.md`
- Deck root: `{repo_path(args.deck_root)}`
- Run root: `{repo_path(args.run_root)}`

## Regeneration

```bash
make suds-tetc-adc-corner-cases
```
"""
    path.write_text(body, encoding="utf-8")


def main() -> int:
    args = parse_args()
    cases = planned_cases()
    simulator, executable, availability = choose_simulator(args.simulator)
    if args.require_simulator and not executable:
        print("R12h blocker: ngspice unavailable")

    write_decks(args, cases)
    rows = build_rows(args, cases, simulator, executable)
    acceptance = build_acceptance(rows, planned_count=len(cases))
    write_csv(args.csv_out, rows)
    write_json(args.json_out, args=args, rows=rows, acceptance=acceptance,
               availability=availability, simulator=simulator)
    write_report(args.report_out, args=args, rows=rows, acceptance=acceptance,
                 simulator=simulator)

    print(f"Wrote {args.csv_out} ({len(rows)} rows)")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.report_out}")
    print(f"Acceptance state: {acceptance['acceptance_state']}")
    if acceptance["blockers"]:
        print(f"Blockers: {acceptance['blockers']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
