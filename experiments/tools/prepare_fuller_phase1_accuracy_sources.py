#!/usr/bin/env python3
"""Prepare conservative accuracy-source overlays and scaffolds for FULLER phase1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import prepare_true_sc_e0_e6_accuracy_sources as legacy_sources
except ImportError:
    import prepare_true_sc_e0_e6_accuracy_sources as legacy_sources  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "configs" / "fuller_phase1_canonical_bundle_20260421.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare conservative accuracy-source overlays and scaffolds for FULLER phase1."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--accuracy_launch_root", type=Path, default=None)
    parser.add_argument("--inspect-existing-source-prep", action="store_true")
    parser.add_argument(
        "--write-scope",
        choices=(legacy_sources.WRITE_SCOPE_FULL, legacy_sources.WRITE_SCOPE_REPORT_DATA_ONLY),
        default=legacy_sources.WRITE_SCOPE_FULL,
        help="Choose whether to materialize the full source-prep surface or only owned report-data artifacts.",
    )
    args = parser.parse_args()

    bundle_path = args.bundle if args.bundle.is_absolute() else ROOT / args.bundle
    if args.inspect_existing_source_prep:
        payload = legacy_sources.inspect_existing_source_prep(
            bundle_path=bundle_path,
            accuracy_launch_root=args.accuracy_launch_root,
            write_scope=str(args.write_scope),
        )
    else:
        payload = legacy_sources.build_accuracy_source_artifacts(
            bundle_path=bundle_path,
            accuracy_launch_root=args.accuracy_launch_root,
            write_scope=str(args.write_scope),
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
