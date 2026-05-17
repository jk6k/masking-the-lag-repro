#!/usr/bin/env python3
"""Mirror a manifest-defined image subset into the Linux filesystem."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def _resolve_src(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _load_rows(manifest_csv: Path) -> list[dict[str, str]]:
    with manifest_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit(f"Manifest must contain path,label columns: {manifest_csv}")
        return [{"path": str(row["path"]).strip(), "label": str(row["label"]).strip()} for row in reader if str(row.get("path") or "").strip()]


def _copy_one(src: Path, dst: Path) -> tuple[bool, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if dst.stat().st_size == src.stat().st_size:
                return False, dst.stat().st_size
        except FileNotFoundError:
            pass
    shutil.copy2(src, dst)
    return True, dst.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror a manifest-defined image subset into Linux storage.")
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--source_root", type=Path, default=ROOT_DIR / "experiments" / "datasets" / "imagenet" / "val")
    parser.add_argument("--out_root", type=Path, required=True, help="Destination root for mirrored images.")
    parser.add_argument("--out_manifest_csv", type=Path, required=True, help="Rewritten manifest with Linux-local paths.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = _load_rows(args.manifest_csv)
    source_root = args.source_root
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    args.out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    copy_jobs: list[tuple[Path, Path, str]] = []
    rewritten_rows: list[dict[str, str]] = []
    for row in rows:
        src = _resolve_src(row["path"])
        try:
            rel = src.relative_to(source_root)
        except ValueError as exc:
            raise SystemExit(f"Path {src} is not under source_root {source_root}") from exc
        dst = out_root / rel
        copy_jobs.append((src, dst, row["label"]))
        rewritten_rows.append({"path": str(dst), "label": row["label"]})

    copied_files = 0
    reused_files = 0
    copied_bytes = 0
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(_copy_one, src, dst): (src, dst) for src, dst, _ in copy_jobs}
        for idx, future in enumerate(as_completed(futures), start=1):
            src, dst = futures[future]
            changed, size_bytes = future.result()
            if changed:
                copied_files += 1
                copied_bytes += size_bytes
            else:
                reused_files += 1
            if idx % 1000 == 0 or idx == len(futures):
                print(
                    f"[mirror-manifest] done={idx}/{len(futures)} copied={copied_files} reused={reused_files}",
                    flush=True,
                )

    with args.out_manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rewritten_rows)

    print(f"Wrote mirrored manifest: {args.out_manifest_csv}")
    print(f"Mirrored root: {out_root}")
    print(f"Copied files: {copied_files}")
    print(f"Reused files: {reused_files}")
    print(f"Copied GiB: {copied_bytes / (1024 ** 3):.2f}")


if __name__ == "__main__":
    main()
