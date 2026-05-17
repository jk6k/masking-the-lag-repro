#!/usr/bin/env python3
"""Audit manuscript citation claim windows against the local Markdown KB."""

from __future__ import annotations

import argparse
import bisect
import csv
import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

try:  # Test imports add experiments/ to sys.path; direct CLI runs use this file's dir.
    from tools import kb_search
except ImportError:  # pragma: no cover - exercised by direct script execution.
    import kb_search  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAPTERS_DIR = REPO_ROOT / "peper_writing" / "thesis" / "chapters"
DEFAULT_CITATION_MAP = REPO_ROOT / "peper_writing" / "citation_key_map_20260429.csv"
DEFAULT_CSV_OUT = REPO_ROOT / "peper_writing" / "citation_claim_audit_20260429.csv"
DEFAULT_MD_OUT = REPO_ROOT / "peper_writing" / "citation_claim_audit_20260429.md"
DEFAULT_TOP_K = 10

CITE_RE = re.compile(
    r"\\(?P<cmd>[A-Za-z]*cite[A-Za-z]*|cite)\*?"
    r"(?:\s*\[[^\[\]]*\])*"
    r"\s*\{(?P<keys>[^{}]+)\}",
    flags=re.DOTALL,
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-'.]*")
EXTERNAL_STATUSES = {"external_or_bib_only"}
LOCAL_STATUSES = {"local_exact", "local_probable"}


@dataclasses.dataclass(frozen=True)
class CitationMapEntry:
    citation_key: str
    bib_title: str = ""
    bib_year: str = ""
    bib_venue: str = ""
    status: str = ""
    expected_md_path: str = ""
    theme_dir: str = ""
    source_page_policy: str = ""
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class ClaimWindow:
    claim_id: str
    tex_path: str
    line: int
    citation_keys: tuple[str, ...]
    claim_text: str
    context_type: str
    local_keys: tuple[str, ...] = ()
    external_keys: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Candidate:
    rank: int
    score: float
    md_path: str
    source_page: int | None
    heading_path: str
    snippet: str


@dataclasses.dataclass(frozen=True)
class AuditCheck:
    claim_id: str
    tex_path: str
    line: int
    citation_key: str
    citation_keys: tuple[str, ...]
    claim_text: str
    context_type: str
    local_keys: tuple[str, ...]
    external_keys: tuple[str, ...]
    map_status: str
    expected_md_path: str
    theme_dir: str
    verdict: str
    verdict_reason: str
    candidates: tuple[Candidate, ...]


@dataclasses.dataclass(frozen=True)
class EnvironmentSpan:
    name: str
    start: int
    begin_end: int
    end_start: int
    end: int


def strip_latex_comments(text: str) -> str:
    return "".join(strip_latex_comment_line(line) for line in text.splitlines(keepends=True))


def strip_latex_comment_line(line: str) -> str:
    newline = "\n" if line.endswith("\n") else ""
    body = line[:-1] if newline else line
    for index, char in enumerate(body):
        if char != "%":
            continue
        slash_count = 0
        cursor = index - 1
        while cursor >= 0 and body[cursor] == "\\":
            slash_count += 1
            cursor -= 1
        if slash_count % 2 == 0:
            return body[:index] + newline
    return line


def line_starts_for(text: str) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(text) if char == "\n")
    return starts


def line_for_offset(line_starts: Sequence[int], offset: int) -> int:
    return bisect.bisect_right(line_starts, offset)


def report_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_citation_keys(raw_keys: str) -> tuple[str, ...]:
    keys = [key.strip() for key in raw_keys.replace("\n", " ").split(",")]
    return tuple(key for key in keys if key)


def load_citation_map(path: Path) -> dict[str, CitationMapEntry]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        entries: dict[str, CitationMapEntry] = {}
        for row in reader:
            key = (row.get("citation_key") or "").strip()
            if not key:
                continue
            entries[key] = CitationMapEntry(
                citation_key=key,
                bib_title=(row.get("bib_title") or "").strip(),
                bib_year=(row.get("bib_year") or "").strip(),
                bib_venue=(row.get("bib_venue") or "").strip(),
                status=(row.get("status") or "").strip(),
                expected_md_path=(row.get("expected_md_path") or "").strip(),
                theme_dir=(row.get("theme_dir") or "").strip(),
                source_page_policy=(row.get("source_page_policy") or "").strip(),
                notes=(row.get("notes") or "").strip(),
            )
    return entries


def key_buckets(
    keys: Sequence[str],
    citation_map: dict[str, CitationMapEntry],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    local_keys: list[str] = []
    external_keys: list[str] = []
    for key in keys:
        entry = citation_map.get(key)
        if not entry:
            continue
        status = entry.status.lower()
        if status in EXTERNAL_STATUSES:
            external_keys.append(key)
        elif status in LOCAL_STATUSES or entry.expected_md_path:
            local_keys.append(key)
    return tuple(local_keys), tuple(external_keys)


def find_environment_containing(
    text: str,
    offset: int,
    names: Sequence[str],
) -> EnvironmentSpan | None:
    best: EnvironmentSpan | None = None
    for name in names:
        begin_re = re.compile(r"\\begin\{" + re.escape(name) + r"\}")
        end_re = re.compile(r"\\end\{" + re.escape(name) + r"\}")
        for begin_match in begin_re.finditer(text):
            if begin_match.start() > offset:
                break
            end_match = end_re.search(text, begin_match.end())
            if not end_match or end_match.start() < offset:
                continue
            candidate = EnvironmentSpan(
                name=name,
                start=begin_match.start(),
                begin_end=begin_match.end(),
                end_start=end_match.start(),
                end=end_match.end(),
            )
            if best is None or candidate.start > best.start:
                best = candidate
    return best


def read_command_argument(text: str, command_start: int) -> tuple[str, int, int] | None:
    command_match = re.match(r"\\[A-Za-z]+\*?", text[command_start:])
    if not command_match:
        return None
    index = command_start + command_match.end()
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] == "[":
            closing = text.find("]", index + 1)
            if closing == -1:
                return None
            index = closing + 1
            continue
        break
    if index >= len(text) or text[index] != "{":
        return None
    depth = 1
    cursor = index + 1
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[index + 1 : cursor], index + 1, cursor
        cursor += 1
    return None


def extract_captions(text: str, base_offset: int = 0) -> list[tuple[str, int, int]]:
    captions: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\\caption\*?", text):
        parsed = read_command_argument(text, match.start())
        if parsed is None:
            continue
        caption, start, end = parsed
        captions.append((caption, base_offset + start, base_offset + end))
    return captions


def find_caption_containing(
    text: str,
    offset: int,
    table_span: EnvironmentSpan | None,
) -> tuple[str, int, int] | None:
    if table_span is None:
        search_text = text
        base_offset = 0
    else:
        search_text = text[table_span.start : table_span.end]
        base_offset = table_span.start
    for caption, start, end in extract_captions(search_text, base_offset=base_offset):
        if start <= offset <= end:
            return caption, start, end
    return None


def table_caption_text(text: str, table_span: EnvironmentSpan | None) -> str:
    if table_span is None:
        return ""
    captions = [
        strip_latex_to_text(caption)
        for caption, _, _ in extract_captions(text[table_span.start : table_span.end], base_offset=table_span.start)
    ]
    return ". ".join(caption for caption in captions if caption)


def extract_table_row_latex(text: str, offset: int, tabular_span: EnvironmentSpan | None) -> str:
    if tabular_span is None:
        line_start = text.rfind("\n", 0, offset) + 1
        line_end = text.find("\n", offset)
        if line_end == -1:
            line_end = len(text)
        return text[line_start:line_end]

    body = text[tabular_span.begin_end : tabular_span.end_start]
    relative_offset = max(0, offset - tabular_span.begin_end)
    previous_break = body.rfind(r"\\", 0, relative_offset)
    next_break = body.find(r"\\", relative_offset)
    start = previous_break + 2 if previous_break != -1 else 0
    end = next_break if next_break != -1 else len(body)
    row = body[start:end]
    row = re.sub(r"^\s*(?:\{[^{}]*\}\s*)+", "", row)
    row = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline)(?:\{[^{}]*\})?", " ", row)
    return row


def looks_like_table_row(text: str, offset: int) -> bool:
    line_start = text.rfind("\n", 0, offset) + 1
    line_end = text.find("\n", offset)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    return "&" in line and (r"\\" in line or line.rstrip().endswith("\\"))


def paragraph_bounds(text: str, offset: int) -> tuple[int, int]:
    start = text.rfind("\n\n", 0, offset)
    start = 0 if start == -1 else start + 2
    end = text.find("\n\n", offset)
    end = len(text) if end == -1 else end
    return start, end


def sentence_spans(block: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(block):
        end = match.start()
        if block[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if block[start:].strip():
        spans.append((start, len(block)))
    return spans or [(0, len(block))]


def sentence_latex_at(text: str, offset: int) -> str:
    start, end = paragraph_bounds(text, offset)
    block = text[start:end]
    relative_offset = max(0, offset - start)
    spans = sentence_spans(block)
    chosen_index = 0
    for index, (span_start, span_end) in enumerate(spans):
        if span_start <= relative_offset <= span_end:
            chosen_index = index
            break
    claim_start, claim_end = spans[chosen_index]
    stripped = strip_latex_to_text(block[claim_start:claim_end])
    if chosen_index > 0 and len(WORD_RE.findall(stripped)) < 8:
        previous_start, _ = spans[chosen_index - 1]
        claim_start = previous_start
    return block[claim_start:claim_end]


def strip_latex_to_text(latex: str) -> str:
    text = latex
    text = CITE_RE.sub("", text)
    text = re.sub(r"\\(?:label|index|footnote)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]+\}", " ", text)
    text = re.sub(r"\\(?:ref|pageref|autoref|cref|Cref)\*?\{([^{}]+)\}", r"\1", text)
    text = text.replace(r"\(", " ").replace(r"\)", " ")
    text = text.replace(r"\[", " ").replace(r"\]", " ")
    text = re.sub(r"\$(.*?)\$", r"\1", text)
    replacements = {
        r"\&": "&",
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"~": " ",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    for _ in range(6):
        new_text = re.sub(
            r"\\[A-Za-z]+\*?(?:\[[^\[\]]*\])?\{([^{}]*)\}",
            r"\1",
            text,
        )
        if new_text == text:
            break
        text = new_text
    text = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline)\b(?:\{[^{}]*\})?", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    text = text.replace("&", " | ")
    text = text.replace(r"\\", " ")
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("--", "-")
    return re.sub(r"\s+", " ", text).strip(" .;")


def claim_context_for_citation(text: str, offset: int) -> tuple[str, str]:
    table_span = find_environment_containing(text, offset, ("table", "table*", "sidewaystable", "longtable"))
    tabular_span = find_environment_containing(text, offset, ("tabular", "tabular*", "tabularx", "longtable"))

    caption = find_caption_containing(text, offset, table_span)
    if caption is not None:
        caption_latex, caption_start, _ = caption
        return strip_latex_to_text(sentence_latex_at(caption_latex, offset - caption_start)), "table_caption"

    if tabular_span is not None or looks_like_table_row(text, offset):
        row_latex = extract_table_row_latex(text, offset, tabular_span)
        caption_text = table_caption_text(text, table_span)
        row_text = strip_latex_to_text(row_latex)
        if caption_text and row_text:
            return f"{caption_text}. {row_text}", "table_row"
        return caption_text or row_text, "table_row"

    return strip_latex_to_text(sentence_latex_at(text, offset)), "sentence"


def extract_claim_windows_from_tex(
    tex_path: Path,
    citation_map: dict[str, CitationMapEntry] | None = None,
    *,
    start_index: int = 1,
) -> list[ClaimWindow]:
    citation_map = citation_map or {}
    text = strip_latex_comments(tex_path.read_text(encoding="utf-8", errors="replace"))
    line_starts = line_starts_for(text)
    windows: list[ClaimWindow] = []
    for match in CITE_RE.finditer(text):
        command = match.group("cmd").lower()
        if command == "nocite":
            continue
        keys = parse_citation_keys(match.group("keys"))
        if not keys:
            continue
        claim_text, context_type = claim_context_for_citation(text, match.start())
        local_keys, external_keys = key_buckets(keys, citation_map)
        windows.append(
            ClaimWindow(
                claim_id=f"C{start_index + len(windows):04d}",
                tex_path=report_path(tex_path),
                line=line_for_offset(line_starts, match.start()),
                citation_keys=keys,
                claim_text=claim_text,
                context_type=context_type,
                local_keys=local_keys,
                external_keys=external_keys,
            )
        )
    return windows


def extract_claim_windows(
    tex_paths: Iterable[Path],
    citation_map: dict[str, CitationMapEntry] | None = None,
) -> list[ClaimWindow]:
    windows: list[ClaimWindow] = []
    next_index = 1
    for tex_path in sorted(tex_paths):
        file_windows = extract_claim_windows_from_tex(tex_path, citation_map, start_index=next_index)
        windows.extend(file_windows)
        next_index += len(file_windows)
    return windows


def normalize_md_path(raw_path: str) -> str:
    raw_path = raw_path.strip().replace("\\", "/")
    while raw_path.startswith("./"):
        raw_path = raw_path[2:]
    if not raw_path:
        return ""
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return raw_path


def paths_match(expected: str, actual: str) -> bool:
    expected_norm = normalize_md_path(expected)
    actual_norm = normalize_md_path(actual)
    if not expected_norm or not actual_norm:
        return False
    if expected_norm == actual_norm:
        return True
    return expected_norm.endswith("/" + actual_norm) or actual_norm.endswith("/" + expected_norm)


def has_exact_source_page_support(entry: CitationMapEntry | None) -> bool:
    if entry is None or entry.status.lower() != "local_exact":
        return False
    policy = entry.source_page_policy.lower()
    if not policy:
        return False
    blocked = {"missing", "unknown", "none", "no_source_page"}
    return "exact" in policy and not any(token in policy for token in blocked)


def search_args(query: str, top_k: int, theme: str | None, include_quarantine: bool) -> argparse.Namespace:
    return argparse.Namespace(
        query=query,
        theme=theme,
        year_min=None,
        year_max=None,
        status="ACTIVE",
        top_k=top_k,
        include_quarantine=include_quarantine,
    )


def run_kb_search(
    chunks: Sequence[kb_search.Chunk],
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    theme: str | None = None,
    include_quarantine: bool = False,
    use_fts: bool | None = None,
) -> list[kb_search.SearchResult]:
    if not query.strip() or not chunks:
        return []

    def run_once(theme_value: str | None, limit: int) -> list[kb_search.SearchResult]:
        args = search_args(query, limit, theme_value, include_quarantine)
        should_use_fts = kb_search.fts5_available() if use_fts is None else use_fts
        if should_use_fts:
            try:
                return kb_search.search_fts(chunks, args)
            except Exception:
                return kb_search.search_fallback(chunks, args)
        return kb_search.search_fallback(chunks, args)

    primary = run_once(theme, top_k)
    if not theme:
        return primary[:top_k]

    merged: list[kb_search.SearchResult] = []
    seen: set[str] = set()
    for result in primary + run_once(None, top_k):
        if result.md_path in seen:
            continue
        seen.add(result.md_path)
        merged.append(dataclasses.replace(result, rank=len(merged) + 1))
        if len(merged) >= top_k:
            break
    return merged


def candidate_from_result(result: kb_search.SearchResult, rank: int) -> Candidate:
    return Candidate(
        rank=rank,
        score=result.score,
        md_path=result.md_path,
        source_page=result.source_page,
        heading_path=result.heading_path,
        snippet=result.snippet,
    )


def candidates_for_claim(
    chunks: Sequence[kb_search.Chunk],
    claim: ClaimWindow,
    entry: CitationMapEntry | None,
    *,
    top_k: int,
    include_quarantine: bool,
    use_fts: bool | None,
) -> tuple[Candidate, ...]:
    theme = entry.theme_dir if entry and entry.theme_dir else None
    results = run_kb_search(
        chunks,
        claim.claim_text,
        top_k=top_k,
        theme=theme,
        include_quarantine=include_quarantine,
        use_fts=use_fts,
    )
    return tuple(candidate_from_result(result, rank=index + 1) for index, result in enumerate(results))


def classify_verdict(
    entry: CitationMapEntry | None,
    candidates: Sequence[Candidate],
) -> tuple[str, str]:
    if entry and entry.status.lower() in EXTERNAL_STATUSES:
        return "warn_external", "citation map marks this key external_or_bib_only"
    if entry and entry.expected_md_path and any(paths_match(entry.expected_md_path, candidate.md_path) for candidate in candidates):
        return "pass", "expected local Markdown path appears in deterministic top results"
    if has_exact_source_page_support(entry):
        return "pass", "citation map marks an exact local source-page support policy"
    if candidates:
        if entry is None:
            return "review", "citation key is absent from the map; deterministic candidates need manual review"
        if entry.expected_md_path:
            return "review", "deterministic candidates exist, but the mapped local source is not in top results"
        return "review", "deterministic candidates exist for a non-local or unresolved map entry"
    if entry is None:
        return "fail", "citation key is absent from the map and deterministic retrieval found no candidates"
    if entry.expected_md_path:
        return "fail", "mapped local source was not retrieved and no alternative candidates were found"
    return "fail", "citation map does not provide local evidence and deterministic retrieval found no candidates"


def audit_claims(
    claims: Sequence[ClaimWindow],
    citation_map: dict[str, CitationMapEntry],
    chunks: Sequence[kb_search.Chunk],
    *,
    top_k: int = DEFAULT_TOP_K,
    include_quarantine: bool = False,
    use_fts: bool | None = None,
) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    for claim in claims:
        for key in claim.citation_keys:
            entry = citation_map.get(key)
            if entry and entry.status.lower() in EXTERNAL_STATUSES:
                candidates: tuple[Candidate, ...] = ()
            else:
                candidates = candidates_for_claim(
                    chunks,
                    claim,
                    entry,
                    top_k=top_k,
                    include_quarantine=include_quarantine,
                    use_fts=use_fts,
                )
            verdict, reason = classify_verdict(entry, candidates)
            checks.append(
                AuditCheck(
                    claim_id=claim.claim_id,
                    tex_path=claim.tex_path,
                    line=claim.line,
                    citation_key=key,
                    citation_keys=claim.citation_keys,
                    claim_text=claim.claim_text,
                    context_type=claim.context_type,
                    local_keys=claim.local_keys,
                    external_keys=claim.external_keys,
                    map_status=entry.status if entry else "missing_from_map",
                    expected_md_path=entry.expected_md_path if entry else "",
                    theme_dir=entry.theme_dir if entry else "",
                    verdict=verdict,
                    verdict_reason=reason,
                    candidates=candidates,
                )
            )
    return checks


def candidates_json(candidates: Sequence[Candidate]) -> str:
    return json.dumps([dataclasses.asdict(candidate) for candidate in candidates], ensure_ascii=False)


CSV_FIELDS = [
    "claim_id",
    "tex_path",
    "line",
    "citation_key",
    "citation_keys",
    "claim_text",
    "context_type",
    "local_keys",
    "external_keys",
    "map_status",
    "expected_md_path",
    "theme_dir",
    "verdict",
    "verdict_reason",
    "candidates_json",
]


def audit_check_to_csv_row(check: AuditCheck) -> dict[str, object]:
    return {
        "claim_id": check.claim_id,
        "tex_path": check.tex_path,
        "line": check.line,
        "citation_key": check.citation_key,
        "citation_keys": ",".join(check.citation_keys),
        "claim_text": check.claim_text,
        "context_type": check.context_type,
        "local_keys": ",".join(check.local_keys),
        "external_keys": ",".join(check.external_keys),
        "map_status": check.map_status,
        "expected_md_path": check.expected_md_path,
        "theme_dir": check.theme_dir,
        "verdict": check.verdict,
        "verdict_reason": check.verdict_reason,
        "candidates_json": candidates_json(check.candidates),
    }


def write_csv_report(checks: Sequence[AuditCheck], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for check in checks:
            writer.writerow(audit_check_to_csv_row(check))


def write_markdown_report(checks: Sequence[AuditCheck], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verdict_counts = Counter(check.verdict for check in checks)
    lines = [
        "# Citation Claim Audit",
        "",
        "Generated by `experiments/tools/audit_manuscript_citations.py`.",
        "",
        "## Summary",
        "",
        "| Verdict | Count |",
        "| --- | ---: |",
    ]
    for verdict in ("pass", "warn_external", "review", "fail"):
        lines.append(f"| `{verdict}` | {verdict_counts.get(verdict, 0)} |")
    lines.extend(["", "## Checks", ""])
    for check in checks:
        lines.extend(
            [
                f"### {check.claim_id} `{check.citation_key}` - `{check.verdict}`",
                "",
                f"- Source: `{check.tex_path}:{check.line}`",
                f"- Context type: `{check.context_type}`",
                f"- Citation keys: `{','.join(check.citation_keys)}`",
                f"- Map status: `{check.map_status}`",
                f"- Expected Markdown: `{check.expected_md_path or '(none)'}`",
                f"- Reason: {check.verdict_reason}",
                f"- Claim: {check.claim_text}",
                "",
            ]
        )
        if check.candidates:
            lines.extend(["Candidates:", ""])
            for candidate in check.candidates:
                page = candidate.source_page if candidate.source_page is not None else "?"
                heading = candidate.heading_path or "(no heading)"
                snippet = candidate.snippet or "(no snippet)"
                lines.append(
                    f"{candidate.rank}. `{candidate.md_path}` page `{page}`; heading `{heading}`; "
                    f"score `{candidate.score:.6f}`; snippet: {snippet}"
                )
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract manuscript citation claim windows and audit them with deterministic KB retrieval."
    )
    parser.add_argument("--chapters-dir", type=Path, default=DEFAULT_CHAPTERS_DIR)
    parser.add_argument("--chapter-glob", default="*.tex")
    parser.add_argument("--citation-map", type=Path, default=DEFAULT_CITATION_MAP)
    parser.add_argument("--markdown-root", type=Path, default=kb_search.DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--include-quarantine", action="store_true")
    parser.add_argument("--no-retrieval", action="store_true", help="Only parse citation windows; audit rows become map-only.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    tex_paths = sorted(args.chapters_dir.glob(args.chapter_glob))
    if not tex_paths:
        raise SystemExit(f"No TeX files matched {args.chapters_dir / args.chapter_glob}")

    citation_map = load_citation_map(args.citation_map)
    claims = extract_claim_windows(tex_paths, citation_map)
    chunks: list[kb_search.Chunk] = []
    if not args.no_retrieval:
        markdown_root = args.markdown_root.resolve()
        if not markdown_root.exists():
            raise SystemExit(f"Markdown root does not exist: {markdown_root}")
        chunks = kb_search.load_chunks(markdown_root, include_quarantine=args.include_quarantine)

    checks = audit_claims(
        claims,
        citation_map,
        chunks,
        top_k=args.top_k,
        include_quarantine=args.include_quarantine,
    )
    write_csv_report(checks, args.csv_out)
    write_markdown_report(checks, args.md_out)

    counts = Counter(check.verdict for check in checks)
    print(
        "audited "
        f"{len(claims)} claim windows / {len(checks)} citation-key checks; "
        + ", ".join(f"{verdict}={counts.get(verdict, 0)}" for verdict in ("pass", "warn_external", "review", "fail"))
    )
    print(f"csv: {args.csv_out}")
    print(f"md: {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
