#!/usr/bin/env python3
"""Narrow hygiene audit for local knowledge-base Markdown mirrors."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKDOWN_ROOT = REPO_ROOT / "original_papers" / "markdown"

METADATA_RE = re.compile(r"^-\s+`([^`]+)`:\s+`(.*)`\s*$")
SOURCE_PAGE_RE = re.compile(r"^<!--\s*source-page:\s*(\d+)\s*-->\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FUSED_HEADING_RE = re.compile(r"(?<!^)#{1,6}\s+[A-Z][A-Za-z]")
FENCED_CODE_RE = re.compile(r"^\s*(?:```|~~~)")
FUSED_HEADING_START_RE = re.compile(r"^#{1,6}\S")
CODE_HASH_DIRECTIVE_RE = re.compile(
    r"^#{1,6}(?:define|include|if|ifdef|ifndef|elif|else|endif|export|pragma|error|warning)\b",
    re.IGNORECASE,
)
CODE_HASH_INSTANCE_RE = re.compile(r"^#{3,6}Instance\b")
CODE_HASH_IDENTIFIER_RE = re.compile(r"^#{1,6}[A-Za-z_][A-Za-z0-9_]*(?:[._-][A-Za-z0-9_]+)*")
CODE_CONTEXT_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*\s*\(|[;={}$]|::|\.[A-Za-z0-9_]+\b)")
INLINE_CODE_CONTEXT_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*\s*\(|[;{}$]|::|\.[A-Za-z0-9_]+\b)")

NAVIGATION_FILENAMES = frozenset({"SUMMARY.MD", "INDEX.MD", "README.MD", "CHAPTER_INDEX.MD"})

FIRST_PAGE_BOILERPLATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("arxiv_header", re.compile(r"\barxiv\s*:\s*\d", re.IGNORECASE)),
    ("preprint_header", re.compile(r"\b(?:arxiv\s+)?preprint\b", re.IGNORECASE)),
    ("copyright_notice", re.compile(r"\b(?:copyright|all rights reserved|©)\b", re.IGNORECASE)),
    ("licensed_download", re.compile(r"\b(?:licensed use|downloaded from|authorized licensed use)\b", re.IGNORECASE)),
    ("doi_header", re.compile(r"^\s*(?:doi|digital object identifier)\s*[:/]", re.IGNORECASE)),
    ("received_accepted_header", re.compile(r"\breceived\b.{0,80}\baccepted\b", re.IGNORECASE)),
)

BROAD_TOKEN_COMPACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("wellknown", re.compile(r"\bwellknown\b", re.IGNORECASE)),
    ("threedimensional", re.compile(r"\bthreedimensional\b", re.IGNORECASE)),
    ("highdimensional", re.compile(r"\bhighdimensional\b", re.IGNORECASE)),
    ("attentionhead", re.compile(r"\battentionheads?\b", re.IGNORECASE)),
    ("matrixmatrix", re.compile(r"\bmatrixmatrix\b", re.IGNORECASE)),
    ("dotproduct", re.compile(r"\bdotproduct\b", re.IGNORECASE)),
    ("chiptochip", re.compile(r"\bchiptochip\b", re.IGNORECASE)),
    ("endtoend", re.compile(r"\bendtoend\b", re.IGNORECASE)),
    ("stateoftheart", re.compile(r"\bstateoftheart\b", re.IGNORECASE)),
)


@dataclasses.dataclass(frozen=True)
class Finding:
    kind: str
    line: int
    detail: str
    excerpt: str


@dataclasses.dataclass(frozen=True)
class FileReport:
    path: str
    source_page_anchors: int
    declared_page_count: int | None
    findings: tuple[Finding, ...]


@dataclasses.dataclass(frozen=True)
class HygieneReport:
    markdown_root: str
    scanned_files: int
    skipped_quarantine_files: int
    files: tuple[FileReport, ...]

    @property
    def finding_count(self) -> int:
        return sum(len(file_report.findings) for file_report in self.files)

    @property
    def affected_file_count(self) -> int:
        return sum(1 for file_report in self.files if file_report.findings)

    def counts_by_kind(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for file_report in self.files:
            counts.update(finding.kind for finding in file_report.findings)
        return counts

    def files_by_kind(self) -> dict[str, int]:
        by_kind: dict[str, set[str]] = {}
        for file_report in self.files:
            for finding in file_report.findings:
                by_kind.setdefault(finding.kind, set()).add(file_report.path)
        return {kind: len(paths) for kind, paths in sorted(by_kind.items())}

    def anchor_summary(self) -> dict[str, int]:
        anchor_files = [file_report for file_report in self.files if file_report.source_page_anchors > 0]
        return {
            "files_with_anchors": len(anchor_files),
            "total_source_page_anchors": sum(file_report.source_page_anchors for file_report in self.files),
            "files_with_declared_page_count": sum(
                1 for file_report in self.files if file_report.declared_page_count is not None
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "markdown_root": self.markdown_root,
            "scanned_files": self.scanned_files,
            "skipped_quarantine_files": self.skipped_quarantine_files,
            "finding_count": self.finding_count,
            "affected_file_count": self.affected_file_count,
            "counts_by_kind": dict(sorted(self.counts_by_kind().items())),
            "files_by_kind": self.files_by_kind(),
            "anchor_summary": self.anchor_summary(),
            "files": [
                {
                    "path": file_report.path,
                    "source_page_anchors": file_report.source_page_anchors,
                    "declared_page_count": file_report.declared_page_count,
                    "findings": [dataclasses.asdict(finding) for finding in file_report.findings],
                }
                for file_report in self.files
                if file_report.findings
            ],
        }

    def render_text(self, *, max_examples: int = 5) -> str:
        counts = self.counts_by_kind()
        anchors = self.anchor_summary()
        lines = [
            "KB Markdown hygiene audit",
            f"root: {self.markdown_root}",
            f"scanned_files: {self.scanned_files}",
            f"skipped_quarantine_files: {self.skipped_quarantine_files}",
            f"affected_files: {self.affected_file_count}",
            f"findings: {self.finding_count}",
            (
                "source_page_anchors: "
                f"{anchors['total_source_page_anchors']} across {anchors['files_with_anchors']} files "
                f"({anchors['files_with_declared_page_count']} files declare page_count)"
            ),
        ]
        if counts:
            lines.append("counts_by_kind:")
            for kind, count in sorted(counts.items()):
                file_count = self.files_by_kind().get(kind, 0)
                lines.append(f"  - {kind}: {count} findings in {file_count} files")
        else:
            lines.append("counts_by_kind: none")

        examples = [
            (file_report, finding)
            for file_report in self.files
            for finding in file_report.findings
        ][:max_examples]
        if examples:
            lines.append("examples:")
            for file_report, finding in examples:
                lines.append(f"  - {file_report.path}:{finding.line} [{finding.kind}] {finding.excerpt}")
        return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit original_papers/markdown for known Markdown hygiene defects."
    )
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--include-quarantine", action="store_true", help="Include paths under a quarantine directory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--max-examples", type=int, default=5, help="Number of text examples to show.")
    return parser.parse_args(argv)


def _is_quarantine(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return "quarantine" in relative.parts


def _is_navigation_file(path: Path) -> bool:
    return path.name.upper() in NAVIGATION_FILENAMES


def iter_markdown_files(root: Path, *, include_quarantine: bool) -> tuple[list[Path], int]:
    skipped = 0
    files: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        if not include_quarantine and _is_quarantine(path, root):
            skipped += 1
            continue
        files.append(path)
    return files, skipped


def parse_metadata(lines: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines[:120]:
        match = METADATA_RE.match(line)
        if match:
            metadata[match.group(1)] = match.group(2)
    return metadata


def parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _first_page_lines(lines: Sequence[str]) -> Iterable[tuple[int, str]]:
    seen_first_anchor = False
    for idx, line in enumerate(lines, start=1):
        page_match = SOURCE_PAGE_RE.match(line.strip())
        if page_match:
            page = int(page_match.group(1))
            if page == 1:
                seen_first_anchor = True
                continue
            if seen_first_anchor and page > 1:
                break
        if seen_first_anchor or idx <= 120:
            yield idx, line
        if not seen_first_anchor and idx >= 120:
            break


def _compact_excerpt(text: str, *, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _is_suspicious_numeric_heading(text: str) -> bool:
    stripped = text.strip()
    if not any(ch.isdigit() for ch in stripped):
        return False
    alpha_chars = [ch for ch in stripped if ch.isalpha()]
    digit_chars = [ch for ch in stripped if ch.isdigit()]
    if not alpha_chars:
        return True
    if len(stripped) <= 18 and len(digit_chars) >= len(alpha_chars):
        return True
    uppercase_text = stripped.upper()
    layout_terms = {"TABLE", "FIG", "FIGURE", "DOTA", "RBE", "SRBO"}
    return bool(layout_terms.intersection(re.findall(r"[A-Z]+", uppercase_text))) and len(stripped) <= 36


def _is_code_like_hash_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False
    if CODE_HASH_DIRECTIVE_RE.match(stripped) or CODE_HASH_INSTANCE_RE.match(stripped):
        return True
    return bool(CODE_HASH_IDENTIFIER_RE.match(stripped) and CODE_CONTEXT_RE.search(stripped))


def _is_code_like_hash_context(line: str) -> bool:
    if _is_code_like_hash_line(line):
        return True
    if not FUSED_HEADING_RE.search(line):
        return False
    before_hash = line.split("#", 1)[0]
    return bool(INLINE_CODE_CONTEXT_RE.search(before_hash))


def audit_file(path: Path, root: Path) -> FileReport:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    metadata = parse_metadata(lines)
    declared_page_count = parse_int(metadata.get("page_count"))
    source_page_anchors = sum(1 for line in lines if SOURCE_PAGE_RE.match(line.strip()))
    has_source_metadata = bool(metadata.get("source_pdf_relative_path") or declared_page_count is not None)
    findings: list[Finding] = []
    in_fenced_code = False

    for line_number, line in enumerate(lines, start=1):
        if FENCED_CODE_RE.match(line):
            in_fenced_code = not in_fenced_code
            continue

        heading_match = HEADING_RE.match(line)
        if heading_match and _is_suspicious_numeric_heading(heading_match.group(2)):
            findings.append(
                Finding(
                    kind="suspicious_numeric_layout_heading",
                    line=line_number,
                    detail="Heading is numeric/layout-heavy.",
                    excerpt=_compact_excerpt(line),
                )
            )
        elif (
            not in_fenced_code
            and line.startswith("#")
            and not heading_match
            and FUSED_HEADING_START_RE.match(line)
            and not _is_code_like_hash_line(line)
        ):
            findings.append(
                Finding(
                    kind="fused_heading_marker",
                    line=line_number,
                    detail="Heading marker is fused to following text.",
                    excerpt=_compact_excerpt(line),
                )
            )
        elif (
            not in_fenced_code
            and not line.startswith("#")
            and FUSED_HEADING_RE.search(line)
            and not _is_code_like_hash_context(line)
        ):
            findings.append(
                Finding(
                    kind="fused_heading_marker",
                    line=line_number,
                    detail="Heading marker appears fused into body text.",
                    excerpt=_compact_excerpt(line),
                )
            )

        for token_name, pattern in BROAD_TOKEN_COMPACTION_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        kind="broad_token_compaction",
                        line=line_number,
                        detail=token_name,
                        excerpt=_compact_excerpt(line),
                    )
                )

    for line_number, line in _first_page_lines(lines):
        for pattern_name, pattern in FIRST_PAGE_BOILERPLATE_PATTERNS:
            if pattern.search(line):
                findings.append(
                    Finding(
                        kind="first_page_boilerplate_leakage",
                        line=line_number,
                        detail=pattern_name,
                        excerpt=_compact_excerpt(line),
                    )
                )
                break

    is_navigation_file = _is_navigation_file(path)
    if not is_navigation_file and has_source_metadata and source_page_anchors == 0:
        findings.append(
            Finding(
                kind="missing_source_page_anchors",
                line=1,
                detail="Source mirror declares source metadata but has no source-page anchors.",
                excerpt=path.name,
            )
        )
    elif not is_navigation_file and declared_page_count is not None and source_page_anchors != declared_page_count:
        findings.append(
            Finding(
                kind="source_page_anchor_count_mismatch",
                line=1,
                detail=f"page_count={declared_page_count}, source_page_anchors={source_page_anchors}",
                excerpt=path.name,
            )
        )

    relative = path.relative_to(root).as_posix()
    return FileReport(
        path=relative,
        source_page_anchors=source_page_anchors,
        declared_page_count=declared_page_count,
        findings=tuple(findings),
    )


def audit_markdown_root(root: Path, *, include_quarantine: bool = False) -> HygieneReport:
    files, skipped = iter_markdown_files(root, include_quarantine=include_quarantine)
    file_reports = tuple(audit_file(path, root) for path in files)
    return HygieneReport(
        markdown_root=str(root),
        scanned_files=len(files),
        skipped_quarantine_files=skipped,
        files=file_reports,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = audit_markdown_root(args.markdown_root, include_quarantine=args.include_quarantine)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.render_text(max_examples=args.max_examples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
