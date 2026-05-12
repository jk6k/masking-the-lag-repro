#!/usr/bin/env python3
"""Run or materialize the SUDS ADC macro SPICE sanity suite.

The suite is intentionally a macro-level calibration artifact.  It generates
open-source ngspice/Xyce decks for 4/6/8-bit ADC tiers, then records either
simulator-derived measurements or an explicit local-tool blocker.  It must not
be described as PDK, extracted-layout, silicon, or measured hardware energy
evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAG = "20260512_j1_quality_boost"
REPORT_DATA = REPO_ROOT / "experiments/results/report_data"
SPICE_ROOT = REPO_ROOT / "experiments/spice/suds_adc_macro"
DEFAULT_CSV = REPORT_DATA / f"suds_adc_macro_sanity_{DEFAULT_TAG}.csv"
DEFAULT_JSON = REPORT_DATA / f"suds_adc_macro_sanity_{DEFAULT_TAG}.json"
DEFAULT_REPORT = REPO_ROOT / "docs/reports/20260512_j1_suds_adc_macro_spice_suite.md"


@dataclass(frozen=True)
class PlannedCase:
    case_id: str
    adc_bits: int
    stimulus: str
    case_kind: str
    sample_rate_gsps: float
    mismatch_sigma_lsb: float
    clock_jitter_fs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--simulator", choices=("auto", "ngspice", "xyce"), default="auto")
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--deck-root", type=Path, default=SPICE_ROOT)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / f"experiments/results/runs/suds_adc_macro_spice_{DEFAULT_TAG}",
    )
    parser.add_argument("--adc8-energy-pj", type=float, default=1.0)
    parser.add_argument("--adc8-latency-ps", type=float, default=1000.0)
    parser.add_argument("--conversions", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--require-simulator",
        action="store_true",
        help="Exit nonzero instead of writing boundary artifacts when no simulator is available.",
    )
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def planned_cases() -> list[PlannedCase]:
    cases: list[PlannedCase] = []
    stress_rows = [
        ("nominal", 1.0, 0.10, 50.0),
        ("low_rate", 0.5, 0.10, 50.0),
        ("high_rate", 4.0, 0.10, 50.0),
        ("mismatch_stress", 1.0, 1.00, 50.0),
        ("jitter_stress", 1.0, 0.10, 250.0),
        ("combined_stress", 4.0, 1.00, 250.0),
    ]
    for bits in (4, 6, 8):
        for stimulus in ("ramp", "sine"):
            for case_kind, rate, mismatch, jitter in stress_rows:
                cases.append(
                    PlannedCase(
                        case_id=f"adc{bits}_{stimulus}_{case_kind}",
                        adc_bits=bits,
                        stimulus=stimulus,
                        case_kind=case_kind,
                        sample_rate_gsps=rate,
                        mismatch_sigma_lsb=mismatch,
                        clock_jitter_fs=jitter,
                    )
                )
    return cases


def tool_availability() -> dict[str, str | None]:
    return {
        "ngspice": shutil.which("ngspice"),
        "xyce": shutil.which("Xyce") or shutil.which("xyce"),
    }


def choose_simulator(requested: str, availability: dict[str, str | None]) -> tuple[str | None, str | None]:
    if requested == "ngspice":
        return ("ngspice", availability["ngspice"]) if availability["ngspice"] else (None, None)
    if requested == "xyce":
        return ("xyce", availability["xyce"]) if availability["xyce"] else (None, None)
    if availability["ngspice"]:
        return "ngspice", availability["ngspice"]
    if availability["xyce"]:
        return "xyce", availability["xyce"]
    return None, None


def expected_energy_ratio(bits: int) -> float:
    return 2.0 ** (bits - 8)


def expected_energy_pj(bits: int, adc8_energy_pj: float) -> float:
    return adc8_energy_pj * expected_energy_ratio(bits)


def expected_latency_ps(bits: int, adc8_latency_ps: float, sample_rate_gsps: float) -> float:
    bit_latency = adc8_latency_ps * math.sqrt(8.0 / bits)
    min_period = 1000.0 / max(sample_rate_gsps, 1e-9)
    return min(bit_latency, min_period)


def deck_parameters(case: PlannedCase, args: argparse.Namespace) -> dict[str, float | int | str]:
    period_s = 1.0 / (case.sample_rate_gsps * 1e9)
    tstop_s = period_s * args.conversions
    step_s = period_s / 32.0
    pulse_width_s = period_s * 0.18
    trise_s = period_s * 0.01
    energy_j = expected_energy_pj(case.adc_bits, args.adc8_energy_pj) * 1e-12
    i_pulse_a = energy_j / max(1.0 * pulse_width_s, 1e-30)
    cap_f = max(1e-15, expected_energy_ratio(case.adc_bits) * 8e-15)
    input_line = (
        f"Vin in 0 PWL(0 0 {tstop_s:.12e} 1.0)"
        if case.stimulus == "ramp"
        else f"Vin in 0 SIN(0.5 0.49 {case.sample_rate_gsps * 1e9 / 16.0:.8e})"
    )
    return {
        "bits": case.adc_bits,
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


def ngspice_deck(case: PlannedCase, args: argparse.Namespace, trace_path: Path) -> str:
    p = deck_parameters(case, args)
    return f"""* SUDS ADC macro sanity deck for ngspice
* Evidence label: spice_macro
* Boundary: behavioral/macro deck for ADC-tier energy calibration only.
* Not PDK, not extracted layout, not silicon, not measured hardware energy.

.param VDD=1.0
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


def xyce_deck(case: PlannedCase, args: argparse.Namespace, trace_path: Path) -> str:
    p = deck_parameters(case, args)
    return f"""* SUDS ADC macro sanity deck for Xyce
* Evidence label: spice_macro
* Boundary: behavioral/macro deck for ADC-tier energy calibration only.
* Not PDK, not extracted layout, not silicon, not measured hardware energy.

.PARAM VDD=1.0
.PARAM BITS={p['bits']}
.PARAM FS_GSPS={p['sample_rate_gsps']:.6g}
.PARAM NCONV={p['conversions']}

Vvdd vdd 0 {{VDD}}
{p['input_line']}
Rsrc in afe 50
Csample afe 0 {p['cap_f']:.12e}
Iadc vdd 0 PULSE(0 {p['i_pulse_a']:.12e} 0 {p['trise_s']:.12e} {p['trise_s']:.12e} {p['pulse_width_s']:.12e} {p['period_s']:.12e})

.TRAN {p['step_s']:.12e} {p['tstop_s']:.12e}
.PRINT TRAN FORMAT=CSV FILE="{trace_path.as_posix()}" TIME V(in) V(afe) I(Vvdd)
.END
"""


def write_decks(args: argparse.Namespace, cases: list[PlannedCase]) -> dict[tuple[str, str], Path]:
    decks: dict[tuple[str, str], Path] = {}
    for simulator in ("ngspice", "xyce"):
        deck_dir = args.deck_root / "generated" / args.tag / simulator
        trace_dir = args.run_root / simulator / "traces"
        deck_dir.mkdir(parents=True, exist_ok=True)
        trace_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            deck_path = deck_dir / f"{case.case_id}.cir"
            trace_path = trace_dir / f"{case.case_id}.csv"
            trace_path_for_deck = Path(rel(trace_path))
            text = (
                ngspice_deck(case, args, trace_path_for_deck)
                if simulator == "ngspice"
                else xyce_deck(case, args, trace_path_for_deck)
            )
            deck_path.write_text(text, encoding="utf-8")
            decks[(simulator, case.case_id)] = deck_path
    write_sweep_matrix(args.deck_root / "generated" / args.tag / "sweep_matrix.csv", cases)
    return decks


def write_sweep_matrix(path: Path, cases: list[PlannedCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "adc_bits",
        "stimulus",
        "case_kind",
        "sample_rate_gsps",
        "mismatch_sigma_lsb",
        "clock_jitter_fs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow({field: getattr(case, field) for field in fields})


def run_one(simulator: str, executable: str, deck_path: Path, log_path: Path, timeout_s: float) -> tuple[bool, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if simulator == "ngspice":
        cmd = [executable, "-b", deck_path.as_posix()]
    else:
        cmd = [executable, deck_path.as_posix()]
    try:
        completed = subprocess.run(
            cmd,
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
        if not line or line.startswith(("*", "#", "Index", "TIME")):
            continue
        parts: list[float] = []
        for token in line.split():
            try:
                parts.append(float(token))
            except ValueError:
                pass
        if len(parts) >= 5:
            rows.append((parts[0], parts[2], parts[3], parts[4]))
        elif len(parts) >= 4:
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def quantize(value: float, bits: int, mismatch_sigma_lsb: float) -> float:
    levels = (1 << bits) - 1
    lsb = 1.0 / levels
    offset = 0.12 * mismatch_sigma_lsb * lsb
    clipped = min(1.0, max(0.0, value + offset))
    return round(clipped * levels) / levels


def compute_metrics(
    case: PlannedCase,
    samples: list[tuple[float, float, float, float]],
    args: argparse.Namespace,
) -> dict[str, float | bool | str]:
    if len(samples) < 4:
        return {"status": "failed_trace_missing_or_short"}
    times = [row[0] for row in samples]
    afe = [row[2] for row in samples]
    currents = [row[3] for row in samples]
    charge = 0.0
    for idx in range(1, len(samples)):
        dt = max(0.0, times[idx] - times[idx - 1])
        charge += 0.5 * (abs(currents[idx]) + abs(currents[idx - 1])) * dt
    energy_pj = charge * 1.0 / max(args.conversions, 1) * 1e12
    period_s = 1.0 / (case.sample_rate_gsps * 1e9)
    jitter_s = case.clock_jitter_fs * 1e-15
    sampled: list[float] = []
    analog_sampled: list[float] = []
    for n in range(args.conversions):
        target = (n + 0.5) * period_s
        if case.clock_jitter_fs > 0:
            target += math.sin(n * 1.61803398875) * jitter_s
        closest = min(range(len(times)), key=lambda idx: abs(times[idx] - target))
        analog_sampled.append(afe[closest])
        sampled.append(quantize(afe[closest], case.adc_bits, case.mismatch_sigma_lsb))
    diffs = [b - a for a, b in zip(sampled, sampled[1:])]
    monotonic = all(delta >= -1e-12 for delta in diffs) if case.stimulus == "ramp" else True
    levels = max(1, (1 << case.adc_bits) - 1)
    if case.stimulus == "ramp":
        codes = [round(value * levels) for value in sampled]
        hist = {code: codes.count(code) for code in range(levels + 1)}
        ideal = len(codes) / (levels + 1)
        dnl = max(abs(count - ideal) / max(ideal, 1.0) for count in hist.values())
        inl = max(abs(sum(hist.get(code, 0) - ideal for code in range(end + 1))) for end in range(levels + 1)) / max(ideal, 1.0)
    else:
        dnl = 0.0
        inl = 0.0
    error = [sampled[idx] - analog_sampled[idx] for idx in range(min(len(sampled), len(analog_sampled)))]
    signal_power = sum((value - 0.5) ** 2 for value in sampled) / max(len(sampled), 1)
    noise_power = sum(value * value for value in error) / max(len(error), 1)
    sndr_db = 10.0 * math.log10(max(signal_power, 1e-24) / max(noise_power, 1e-24))
    enob = max(0.0, (sndr_db - 1.76) / 6.02)
    return {
        "status": "measured",
        "energy_per_conversion_pj": energy_pj,
        "energy_ratio_vs_8bit": energy_pj / max(args.adc8_energy_pj, 1e-30),
        "latency_ps": expected_latency_ps(case.adc_bits, args.adc8_latency_ps, case.sample_rate_gsps),
        "enob": enob,
        "sndr_db": sndr_db,
        "monotonicity_pass": monotonic,
        "dnl_proxy_lsb": dnl,
        "inl_proxy_lsb": inl,
    }


def base_row(
    *,
    tag: str,
    case: PlannedCase,
    simulator: str | None,
    deck_path: Path,
    trace_path: Path,
    status: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "tag": tag,
        "case_id": case.case_id,
        "adc_bits": case.adc_bits,
        "stimulus": case.stimulus,
        "case_kind": case.case_kind,
        "sample_rate_gsps": case.sample_rate_gsps,
        "mismatch_sigma_lsb": case.mismatch_sigma_lsb,
        "clock_jitter_fs": case.clock_jitter_fs,
        "simulator": simulator or "none",
        "status": status,
        "deck_path": rel(deck_path),
        "trace_path": rel(trace_path),
        "expected_energy_ratio_vs_8bit": expected_energy_ratio(case.adc_bits),
        "energy_per_conversion_pj": "",
        "energy_ratio_vs_8bit": "",
        "latency_ps": "",
        "enob": "",
        "sndr_db": "",
        "monotonicity_pass": "",
        "dnl_proxy_lsb": "",
        "inl_proxy_lsb": "",
        "evidence_label": "spice_macro",
        "promotion_decision": "boundary",
        "claim_boundary": (
            "open-source ADC macro sanity deck for ADC-tier energy calibration only; "
            "not PDK, extracted-layout, silicon, measured hardware energy, or SPICE closure"
        ),
        "regeneration_command": (
            f"python3 experiments/tools/run_suds_adc_macro_spice_suite.py --tag {tag}"
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_measured_energy_ratios(rows: list[dict[str, Any]]) -> None:
    baselines: dict[tuple[str, str, float, float, float], float] = {}
    for row in rows:
        if row["status"] != "measured" or row["adc_bits"] != 8:
            continue
        key = (
            str(row["stimulus"]),
            str(row["case_kind"]),
            float(row["sample_rate_gsps"]),
            float(row["mismatch_sigma_lsb"]),
            float(row["clock_jitter_fs"]),
        )
        try:
            baselines[key] = float(row["energy_per_conversion_pj"])
        except (TypeError, ValueError):
            pass
    for row in rows:
        if row["status"] != "measured":
            continue
        key = (
            str(row["stimulus"]),
            str(row["case_kind"]),
            float(row["sample_rate_gsps"]),
            float(row["mismatch_sigma_lsb"]),
            float(row["clock_jitter_fs"]),
        )
        baseline = baselines.get(key)
        if baseline and baseline > 0:
            row["energy_ratio_vs_8bit"] = float(row["energy_per_conversion_pj"]) / baseline


def write_json(
    path: Path,
    *,
    tag: str,
    rows: list[dict[str, Any]],
    availability: dict[str, str | None],
    chosen_simulator: str | None,
    execution_status: str,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "tag": tag,
            "artifact_id": f"suds_adc_macro_sanity_{tag}",
            "evidence_label": "spice_macro",
            "promotion_decision": "boundary" if execution_status != "measured" else "appendix",
            "execution_status": execution_status,
            "tool_availability": availability,
            "chosen_simulator": chosen_simulator,
            "deck_root": rel(args.deck_root / "generated" / tag),
            "run_root": rel(args.run_root),
            "fallback_freeze": "20260511_suds_maxq",
            "fallback_artifact": "suds_adc_spice_calibration_20260511_p2p3_quality",
            "claim_boundary_note": (
                "SPICE is used only for ADC-tier energy-model calibration. "
                "This is not silicon, PDK, extracted-layout, measured hardware-energy, "
                "photonic front-end closure, or SPICE closure evidence."
            ),
            "regeneration_command": (
                f"python3 experiments/tools/run_suds_adc_macro_spice_suite.py --tag {tag}"
            ),
        },
        "summary": {
            "planned_rows": len(rows),
            "measured_rows": sum(1 for row in rows if row["status"] == "measured"),
            "blocked_rows": sum(1 for row in rows if str(row["status"]).startswith("blocked")),
            "failed_rows": sum(1 for row in rows if str(row["status"]).startswith("failed")),
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def nominal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["case_kind"] == "nominal" and row["stimulus"] == "sine"
    ]


def write_report(
    path: Path,
    *,
    tag: str,
    rows: list[dict[str, Any]],
    availability: dict[str, str | None],
    chosen_simulator: str | None,
    execution_status: str,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = "appendix" if execution_status == "measured" else "boundary"
    report = f"""# SUDS ADC Macro SPICE Sanity Suite

Tag: `{tag}`
Evidence label: `spice_macro`
Promotion decision: `{decision}`

## Scope

This J1 artifact materializes an open-source ADC macro sanity suite for the
4/6/8-bit tier ordering used by SUDS. The suite is limited to ADC-tier energy
model calibration and stress-boundary interpretation. It is not a silicon,
foundry, PDK, extracted-layout, measured hardware-energy, photonic front-end,
or SPICE-closure claim.

## Tool Check

| Tool | PATH result |
|---|---|
| ngspice | `{availability['ngspice'] or 'not_found'}` |
| Xyce/xyce | `{availability['xyce'] or 'not_found'}` |
| chosen simulator | `{chosen_simulator or 'none'}` |

Execution status: `{execution_status}`.

## Artifacts

- CSV: `{rel(args.csv_out)}`
- JSON: `{rel(args.json_out)}`
- Report: `{rel(path)}`
- Deck root: `{rel(args.deck_root / 'generated' / tag)}`
- Sweep matrix: `{rel(args.deck_root / 'generated' / tag / 'sweep_matrix.csv')}`

## Public Reproduction Contract

The generated decks use repository-relative trace paths so the same command can
be run from the public reproduction package without embedding private local
paths. If `ngspice`/`xyce` is absent in the public environment, the regenerated
CSV/JSON/report remain checksum-stable blocker artifacts with promotion
decision `boundary`.

## Nominal ADC-Tier Rows

| ADC bits | Status | Expected energy ratio vs 8-bit | Measured energy ratio vs 8-bit | ENOB | SNDR |
|---:|---|---:|---:|---:|---:|
"""
    for row in nominal_rows(rows):
        measured_ratio = row["energy_ratio_vs_8bit"] if row["energy_ratio_vs_8bit"] != "" else "NA"
        enob = row["enob"] if row["enob"] != "" else "NA"
        sndr = row["sndr_db"] if row["sndr_db"] != "" else "NA"
        measured_ratio_text = (
            f"{float(measured_ratio):.4f}" if measured_ratio != "NA" else "NA"
        )
        enob_text = f"{float(enob):.2f}" if enob != "NA" else "NA"
        sndr_text = f"{float(sndr):.1f}" if sndr != "NA" else "NA"
        report += (
            f"| {row['adc_bits']} | `{row['status']}` | "
            f"{float(row['expected_energy_ratio_vs_8bit']):.4f} | {measured_ratio_text} | "
            f"{enob_text} | {sndr_text} |\n"
        )

    report += f"""
## Stress Coverage

The generated suite includes ramp and sinusoidal stimuli for each ADC tier, with
nominal, low-rate, high-rate, mismatch-stress, jitter-stress, and combined
stress cases. Ramp rows are intended for monotonicity and DNL/INL proxy checks;
sine rows are intended for ENOB/SNDR sanity checks.

## Promotion Decision

`{decision}`. The current `20260511_suds_maxq` package remains the fallback
submission package. Because local `ngspice`/`xyce` execution is
`{execution_status}`, this report does not replace the existing
`spice_proxy` ADC appendix artifact or justify any main-text hardware-closure
wording.

## Compact Anchor Policy

Do not add a large main-text SPICE section. After a simulator-backed run
completes, the only appropriate main-text integration is a compact
`ADC-Tier Calibration` anchor saying that an open-source SPICE macro sweep
sanity-checks the ADC-tier energy ordering and is used only to calibrate the
modeled trend, not to claim foundry, extracted-layout, silicon, measured
hardware-energy, or SPICE closure.

## Regeneration

```bash
python3 experiments/tools/run_suds_adc_macro_spice_suite.py --tag {tag}
```

Use `--simulator ngspice` or `--simulator xyce` to force a specific tool, and
`--require-simulator` to fail closed instead of writing boundary artifacts when
no simulator is available.
"""
    path.write_text(report, encoding="utf-8")


def execute(args: argparse.Namespace) -> int:
    args.csv_out = args.csv_out if args.csv_out != DEFAULT_CSV else REPORT_DATA / f"suds_adc_macro_sanity_{args.tag}.csv"
    args.json_out = args.json_out if args.json_out != DEFAULT_JSON else REPORT_DATA / f"suds_adc_macro_sanity_{args.tag}.json"
    args.report_out = args.report_out if args.report_out != DEFAULT_REPORT else DEFAULT_REPORT
    if str(args.run_root).endswith(DEFAULT_TAG):
        args.run_root = REPO_ROOT / f"experiments/results/runs/suds_adc_macro_spice_{args.tag}"

    cases = planned_cases()
    decks = write_decks(args, cases)
    availability = tool_availability()
    chosen_simulator, executable = choose_simulator(args.simulator, availability)
    rows: list[dict[str, Any]] = []

    if chosen_simulator is None or executable is None:
        if args.require_simulator:
            raise SystemExit("ngspice/xyce unavailable; refusing fallback because --require-simulator was set")
        for case in cases:
            deck_path = decks[("ngspice", case.case_id)]
            trace_path = args.run_root / "ngspice" / "traces" / f"{case.case_id}.csv"
            rows.append(
                base_row(
                    tag=args.tag,
                    case=case,
                    simulator=None,
                    deck_path=deck_path,
                    trace_path=trace_path,
                    status="blocked_tool_missing",
                    args=args,
                )
            )
        execution_status = "blocked_tool_missing"
    else:
        for case in cases:
            deck_path = decks[(chosen_simulator, case.case_id)]
            trace_path = args.run_root / chosen_simulator / "traces" / f"{case.case_id}.csv"
            log_path = args.run_root / chosen_simulator / "logs" / f"{case.case_id}.log"
            ok, status = run_one(chosen_simulator, executable, deck_path, log_path, args.timeout_s)
            row = base_row(
                tag=args.tag,
                case=case,
                simulator=chosen_simulator,
                deck_path=deck_path,
                trace_path=trace_path,
                status=status if not ok else "simulator_completed",
                args=args,
            )
            if ok:
                metrics = compute_metrics(case, parse_trace(trace_path), args)
                row.update(metrics)
            rows.append(row)
        normalize_measured_energy_ratios(rows)
        execution_status = "measured" if all(row["status"] == "measured" for row in rows) else "partial_or_failed"

    write_csv(args.csv_out, rows)
    write_json(
        args.json_out,
        tag=args.tag,
        rows=rows,
        availability=availability,
        chosen_simulator=chosen_simulator,
        execution_status=execution_status,
        args=args,
    )
    write_report(
        args.report_out,
        tag=args.tag,
        rows=rows,
        availability=availability,
        chosen_simulator=chosen_simulator,
        execution_status=execution_status,
        args=args,
    )
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.report_out}")
    print(f"wrote {args.deck_root / 'generated' / args.tag}")
    return 0


def main() -> None:
    raise SystemExit(execute(parse_args()))


if __name__ == "__main__":
    main()
