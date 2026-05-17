#!/usr/bin/env python3
"""Prepare and preflight the active FULLER phase1 canonical bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from . import prepare_true_sc_e0_e6_bundle as legacy_bundle
except ImportError:
    import prepare_true_sc_e0_e6_bundle as legacy_bundle  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "configs" / "fuller_phase1_canonical_bundle_20260421.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and preflight the active FULLER phase1 canonical bundle."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument(
        "--write-scope",
        choices=(legacy_bundle.WRITE_SCOPE_FULL, legacy_bundle.WRITE_SCOPE_REPORT_DATA_ONLY),
        default=legacy_bundle.WRITE_SCOPE_FULL,
        help="Write full outputs or only owned report-data artifacts.",
    )
    parser.add_argument("--inspect-existing-preflight", action="store_true")
    parser.add_argument("--run-phase1-preflight", action="store_true")
    args = parser.parse_args()

    bundle_path = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    if args.inspect_existing_preflight:
        payload = legacy_bundle.inspect_existing_preflight(
            bundle_path=bundle_path,
            write_scope=str(args.write_scope),
        )
    else:
        payload = legacy_bundle.build_bundle_artifacts(
            bundle_path=bundle_path,
            python_bin=str(args.python_bin),
            run_phase1_preflight=bool(args.run_phase1_preflight),
            write_scope=str(args.write_scope),
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
