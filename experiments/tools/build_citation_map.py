#!/usr/bin/env python3
"""Build a conservative citation-key to local Markdown evidence map."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIB_PATH = REPO_ROOT / "peper_writing" / "thesis" / "references.bib"
DEFAULT_TEX_ROOT = REPO_ROOT / "peper_writing" / "thesis"
DEFAULT_MARKDOWN_ROOT = REPO_ROOT / "original_papers" / "markdown"
DEFAULT_CSV_OUTPUT = REPO_ROOT / "peper_writing" / "citation_key_map_20260429.csv"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "peper_writing" / "citation_key_map_20260429.md"

MAP_FIELDS = (
    "citation_key",
    "bib_title",
    "bib_year",
    "bib_venue",
    "status",
    "expected_md_path",
    "theme_dir",
    "source_page_policy",
    "notes",
)

KNOWN_EXTERNAL_KEYS = {
    "deng2009imagenet": "Known external dataset/reference; no dedicated local Markdown mirror expected in Phase 1.",
    "sze2017efficientDNN": "Known external survey/reference; no dedicated local Markdown mirror expected in Phase 1.",
    "shastri2021photonicsAI": "Known external photonics-AI review; no dedicated local Markdown mirror expected in Phase 1.",
}

METADATA_RE = re.compile(r"^-\s+`([^`]+)`:\s+`(.*)`\s*$")
HEADING_RE = re.compile(r"^#\s+(.+?)\s*$")
CITE_RE = re.compile(
    r"\\(?P<command>[A-Za-z]*cite[A-Za-z]*)(?:\*)?\s*(?:\[[^\[\]{}]*\]\s*)*\{(?P<keys>[^{}]+)\}",
    re.DOTALL,
)
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "via",
    "with",
}
TITLE_METADATA_FIELDS = (
    "title",
    "short_title",
    "canonical_title",
    "full_title",
)
EVIDENCE_METADATA_FIELDS = (
    "contribution",
    "key_claims",
    "method_tags",
    "hardware_tags",
    "model_tags",
    "metrics",
    "limitations",
    "best_used_for",
)


@dataclasses.dataclass(frozen=True)
class BibEntry:
    key: str
    entry_type: str
    fields: dict[str, str]
    line: int

    @property
    def title(self) -> str:
        return self.fields.get("title", "")

    @property
    def year(self) -> str:
        return self.fields.get("year", "")

    @property
    def venue(self) -> str:
        for field in ("booktitle", "journal", "venue", "publisher", "note"):
            if self.fields.get(field):
                return self.fields[field]
        return ""


@dataclasses.dataclass(frozen=True)
class CitationUse:
    key: str
    tex_path: str
    line: int
    command: str


@dataclasses.dataclass(frozen=True)
class MarkdownRecord:
    path: Path
    md_path: str
    title: str
    metadata: dict[str, str]
    theme_dir: str
    status: str
    source_page_policy: str
    title_candidates: tuple[str, ...]
    evidence_candidates: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CandidateScore:
    record: MarkdownRecord
    score: float
    exact: bool
    matched_text: str
    matched_field: str


@dataclasses.dataclass(frozen=True)
class CitationMapRow:
    citation_key: str
    bib_title: str
    bib_year: str
    bib_venue: str
    status: str
    expected_md_path: str
    theme_dir: str
    source_page_policy: str
    notes: str

    def as_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in MAP_FIELDS}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map cited BibTeX keys to local knowledge-base Markdown evidence surfaces."
    )
    parser.add_argument("--bib-path", type=Path, default=DEFAULT_BIB_PATH)
    parser.add_argument("--tex-root", type=Path, default=DEFAULT_TEX_ROOT)
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--include-quarantine", action="store_true")
    parser.add_argument(
        "--include-uncited-bib",
        action="store_true",
        help="Also include bibliography entries that are not cited in the TeX tree.",
    )
    return parser.parse_args(argv)


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _strip_tex_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        cut_at = len(line)
        index = 0
        while True:
            index = line.find("%", index)
            if index < 0:
                break
            slash_count = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                slash_count += 1
                cursor -= 1
            if slash_count % 2 == 0:
                cut_at = index
                break
            index += 1
        cleaned_lines.append(line[:cut_at])
    return "\n".join(cleaned_lines)


def parse_tex_citations_from_text(text: str, *, tex_path: str = "<memory>") -> list[CitationUse]:
    stripped = _strip_tex_comments(text)
    uses: list[CitationUse] = []
    for match in CITE_RE.finditer(stripped):
        command = match.group("command")
        if command.lower() == "nocite":
            continue
        line = _line_for_offset(stripped, match.start())
        for raw_key in match.group("keys").replace("\n", " ").split(","):
            key = raw_key.strip()
            if key and key != "*":
                uses.append(CitationUse(key=key, tex_path=tex_path, line=line, command=command))
    return uses


def parse_tex_citations(tex_root: Path) -> list[CitationUse]:
    uses: list[CitationUse] = []
    for path in sorted(tex_root.rglob("*.tex")):
        rel_path = path.as_posix()
        uses.extend(parse_tex_citations_from_text(path.read_text(encoding="utf-8"), tex_path=rel_path))
    return uses


def _find_matching_delimiter(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    in_quote = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError(f"unterminated BibTeX entry starting at byte {start}")


def _top_level_comma(text: str) -> int:
    brace_depth = 0
    paren_depth = 0
    in_quote = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "," and brace_depth == 0 and paren_depth == 0:
            return index
    return -1


def _parse_bib_value(fields_text: str, start: int) -> tuple[str, int]:
    brace_depth = 0
    paren_depth = 0
    in_quote = False
    escaped = False
    index = start
    while index < len(fields_text):
        char = fields_text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char == '"':
            in_quote = not in_quote
            index += 1
            continue
        if not in_quote:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth = max(0, brace_depth - 1)
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(0, paren_depth - 1)
            elif char == "," and brace_depth == 0 and paren_depth == 0:
                break
        index += 1
    return fields_text[start:index].strip(), index


def _outer_pair_encloses(raw: str, open_char: str, close_char: str) -> bool:
    if len(raw) < 2 or raw[0] != open_char or raw[-1] != close_char:
        return False
    if open_char == "{" and close_char == "}":
        try:
            return _find_matching_delimiter(raw, 0, "{", "}") == len(raw) - 1
        except ValueError:
            return False
    return True


def clean_bib_value(raw: str) -> str:
    value = raw.strip().rstrip(",").strip()
    if "#" in value:
        value = " ".join(part.strip() for part in value.split("#") if part.strip())
    changed = True
    while changed:
        changed = False
        if _outer_pair_encloses(value, "{", "}"):
            value = value[1:-1].strip()
            changed = True
        if _outer_pair_encloses(value, '"', '"'):
            value = value[1:-1].strip()
            changed = True
    return latex_to_text(value)


def parse_bibtex(text: str) -> dict[str, BibEntry]:
    entries: dict[str, BibEntry] = {}
    cursor = 0
    while True:
        match = re.search(r"@([A-Za-z]+)\s*([{\(])", text[cursor:])
        if not match:
            break
        entry_type = match.group(1).lower()
        open_char = match.group(2)
        entry_start = cursor + match.start()
        body_start = cursor + match.end() - 1
        close_char = "}" if open_char == "{" else ")"
        body_end = _find_matching_delimiter(text, body_start, open_char, close_char)
        body = text[body_start + 1 : body_end]
        cursor = body_end + 1

        comma = _top_level_comma(body)
        if comma < 0:
            continue
        key = body[:comma].strip()
        if not key:
            continue
        fields_text = body[comma + 1 :]
        fields: dict[str, str] = {}
        field_cursor = 0
        while field_cursor < len(fields_text):
            while field_cursor < len(fields_text) and fields_text[field_cursor] in " \t\r\n,":
                field_cursor += 1
            name_match = re.match(r"([A-Za-z][A-Za-z0-9_:\-]*)\s*=", fields_text[field_cursor:])
            if not name_match:
                field_cursor += 1
                continue
            name = name_match.group(1).lower()
            value_start = field_cursor + name_match.end()
            raw_value, value_end = _parse_bib_value(fields_text, value_start)
            fields[name] = clean_bib_value(raw_value)
            field_cursor = value_end + 1
        entries[key] = BibEntry(key=key, entry_type=entry_type, fields=fields, line=_line_for_offset(text, entry_start))
    return entries


def parse_bibtex_file(path: Path) -> dict[str, BibEntry]:
    return parse_bibtex(path.read_text(encoding="utf-8"))


def latex_to_text(value: str) -> str:
    text = value.replace("~", " ")
    text = text.replace("--", "-")
    text = re.sub(r"\\['`^\"~=.]?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\([&%_$#{}])", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", lambda match: match.group(1) or "", text)
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(value: str) -> str:
    text = latex_to_text(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stem_token(token: str) -> str:
    if token.endswith("ically") and len(token) > 8:
        return token[:-6] + "ic"
    if token.endswith("ization") and len(token) > 10:
        return token[:-7] + "ize"
    return token


def title_tokens(value: str) -> set[str]:
    normalized = normalize_title(value)
    return {_stem_token(token) for token in WORD_RE.findall(normalized) if token not in STOPWORDS and len(token) > 1}


def _metadata_from_lines(lines: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines[:100]:
        match = METADATA_RE.match(line.strip())
        if match:
            metadata[match.group(1)] = match.group(2)
    return metadata


def _first_heading(lines: Sequence[str]) -> str:
    for line in lines[:40]:
        match = HEADING_RE.match(line.strip())
        if match:
            return latex_to_text(match.group(1))
    return ""


def _source_page_policy(metadata: dict[str, str]) -> str:
    page_count = metadata.get("page_count", "")
    source_pdf = metadata.get("source_pdf_relative_path", "")
    if page_count and source_pdf:
        return f"source-page anchors expected; page_count={page_count}"
    if source_pdf:
        return "source-page anchors expected when available"
    return "no source PDF metadata"


def _candidate_from_path(path: Path) -> str:
    return re.sub(r"[_\-]+", " ", path.stem)


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def load_markdown_records(markdown_root: Path, *, include_quarantine: bool = False) -> list[MarkdownRecord]:
    records: list[MarkdownRecord] = []
    for path in sorted(markdown_root.rglob("*.md")):
        rel = path.relative_to(markdown_root).as_posix()
        if not include_quarantine and rel.startswith("quarantine/"):
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        metadata = _metadata_from_lines(lines)
        heading = _first_heading(lines)
        theme_dir = metadata.get("theme_dir") or path.parent.name
        status = metadata.get("status", "")
        title_candidates: list[str] = []
        if heading:
            title_candidates.append(heading)
        for field in TITLE_METADATA_FIELDS:
            value = metadata.get(field, "")
            if value:
                title_candidates.append(value)
        title_candidates.append(_candidate_from_path(path))
        source_pdf = metadata.get("source_pdf_relative_path", "")
        if source_pdf:
            title_candidates.append(_candidate_from_path(Path(source_pdf)))
        evidence_candidates = [metadata[field] for field in EVIDENCE_METADATA_FIELDS if metadata.get(field)]
        records.append(
            MarkdownRecord(
                path=path,
                md_path=_repo_relative_path(path),
                title=heading or metadata.get("short_title", "") or path.stem,
                metadata=metadata,
                theme_dir=theme_dir,
                status=status,
                source_page_policy=_source_page_policy(metadata),
                title_candidates=tuple(dict.fromkeys(title_candidates)),
                evidence_candidates=tuple(evidence_candidates),
            )
        )
    return records


def _candidate_score(query: str, candidate: str, *, evidence: bool) -> float:
    query_norm = normalize_title(query)
    candidate_norm = normalize_title(candidate)
    if not query_norm or not candidate_norm:
        return 0.0
    if query_norm == candidate_norm:
        return 1.0
    query_tokens = title_tokens(query_norm)
    candidate_tokens = title_tokens(candidate_norm)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = query_tokens & candidate_tokens
    containment = len(overlap) / len(query_tokens)
    reverse_containment = len(overlap) / len(candidate_tokens)
    jaccard = len(overlap) / len(query_tokens | candidate_tokens)
    phrase_bonus = 0.0
    if query_norm in candidate_norm or candidate_norm in query_norm:
        phrase_bonus = 0.12
    elif any(token in candidate_norm for token in query_tokens if len(token) >= 8):
        phrase_bonus = 0.04
    base = 0.70 * containment + 0.25 * jaccard + 0.05 * reverse_containment + phrase_bonus
    if evidence:
        base *= 0.86
    return min(0.99, base)


def score_markdown_records(title: str, records: Sequence[MarkdownRecord]) -> list[CandidateScore]:
    scores: list[CandidateScore] = []
    for record in records:
        best: CandidateScore | None = None
        for candidate in record.title_candidates:
            score = _candidate_score(title, candidate, evidence=False)
            exact = normalize_title(title) == normalize_title(candidate)
            candidate_score = CandidateScore(
                record=record,
                score=score,
                exact=exact,
                matched_text=candidate,
                matched_field="title",
            )
            if best is None or candidate_score.score > best.score:
                best = candidate_score
        for candidate in record.evidence_candidates:
            score = _candidate_score(title, candidate, evidence=True)
            candidate_score = CandidateScore(
                record=record,
                score=score,
                exact=False,
                matched_text=candidate,
                matched_field="metadata",
            )
            if best is None or candidate_score.score > best.score:
                best = candidate_score
        if best and best.score > 0:
            if record.status.upper() == "ACTIVE":
                best = dataclasses.replace(best, score=min(1.0, best.score + 0.015))
            elif record.status.upper() in {"SUPERSEDED", "DUPLICATE"}:
                best = dataclasses.replace(best, score=max(0.0, best.score - 0.05))
            scores.append(best)
    return sorted(scores, key=lambda item: (-item.score, item.record.md_path))


def _near_candidates(scores: Sequence[CandidateScore], best_score: float, *, delta: float = 0.05) -> list[CandidateScore]:
    return [score for score in scores if best_score - score.score <= delta]


def _status_and_notes(entry: BibEntry | None, scores: Sequence[CandidateScore], key: str) -> tuple[str, CandidateScore | None, str]:
    if entry is None:
        return "needs_manual_review", None, "Cited key is missing from references.bib."
    if not entry.title:
        return "needs_manual_review", None, "BibTeX entry has no title field."
    if not scores or scores[0].score < 0.48:
        if key in KNOWN_EXTERNAL_KEYS:
            return "external_or_bib_only", None, KNOWN_EXTERNAL_KEYS[key]
        return "needs_source_ingest", None, "No conservative title/metadata match found in local Markdown root."

    best = scores[0]
    if key in KNOWN_EXTERNAL_KEYS and not (best.exact or best.score >= 0.80):
        return "external_or_bib_only", None, KNOWN_EXTERNAL_KEYS[key]

    near = _near_candidates(scores, best.score)
    distinct_near = {candidate.record.md_path for candidate in near}
    near_details = "; ".join(
        f"{Path(candidate.record.md_path).name}={candidate.score:.2f}"
        for candidate in near[:4]
    )
    if best.exact and best.record.status.upper() == "ACTIVE":
        if len(distinct_near) > 1 and any(candidate.score >= 0.97 for candidate in near[1:]):
            return "local_probable", best, f"Exact title match with duplicate/near duplicate candidates: {near_details}."
        return "local_exact", best, f"Exact normalized title match against Markdown {best.matched_field}."
    if len(distinct_near) > 1 and best.score < 0.72:
        return "needs_manual_review", best, f"Ambiguous local candidates: {near_details}."
    active_competitor_scores = [
        score.score
        for score in scores[1:]
        if score.record.status.upper() not in {"SUPERSEDED", "DUPLICATE"}
    ]
    second_score = active_competitor_scores[0] if active_competitor_scores else 0.0
    if best.score >= 0.58 or (best.score >= 0.50 and best.score - second_score >= 0.09):
        note = f"Unique conservative non-exact match via {best.matched_field} score={best.score:.2f}."
        if len(distinct_near) > 1:
            note += f" Nearby candidates retained for review: {near_details}."
        return "local_probable", best, note
    if key in KNOWN_EXTERNAL_KEYS:
        return "external_or_bib_only", None, KNOWN_EXTERNAL_KEYS[key]
    return "needs_manual_review", best, f"Weak local candidate score={best.score:.2f}: {Path(best.record.md_path).name}."


def build_citation_rows(
    bib_entries: dict[str, BibEntry],
    citation_uses: Sequence[CitationUse],
    markdown_records: Sequence[MarkdownRecord],
    *,
    include_uncited_bib: bool = False,
) -> list[CitationMapRow]:
    cited_order: list[str] = []
    seen: set[str] = set()
    for use in citation_uses:
        if use.key not in seen:
            cited_order.append(use.key)
            seen.add(use.key)
    if include_uncited_bib:
        for key in sorted(bib_entries):
            if key not in seen:
                cited_order.append(key)
                seen.add(key)

    rows: list[CitationMapRow] = []
    for key in cited_order:
        entry = bib_entries.get(key)
        scores = score_markdown_records(entry.title if entry else "", markdown_records) if entry else []
        status, best, notes = _status_and_notes(entry, scores, key)
        record = best.record if best else None
        rows.append(
            CitationMapRow(
                citation_key=key,
                bib_title=entry.title if entry else "",
                bib_year=entry.year if entry else "",
                bib_venue=entry.venue if entry else "",
                status=status,
                expected_md_path=record.md_path if record and status != "external_or_bib_only" else "",
                theme_dir=record.theme_dir if record and status != "external_or_bib_only" else "",
                source_page_policy=record.source_page_policy if record and status != "external_or_bib_only" else "not local",
                notes=notes,
            )
        )
    return rows


def write_csv(rows: Sequence[CitationMapRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAP_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def write_markdown(rows: Sequence[CitationMapRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row.status for row in rows)
    lines = [
        "# Citation Key Map 2026-04-29",
        "",
        "Generated by `experiments/tools/build_citation_map.py`.",
        "",
        "## Summary",
        "",
        f"- cited keys mapped: `{len(rows)}`",
    ]
    for status in sorted(counts):
        lines.append(f"- `{status}`: `{counts[status]}`")
    lines.extend(
        [
            "",
            "## Map",
            "",
            "| citation_key | status | bib_title | expected_md_path | notes |",
            "|---|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row.citation_key),
                    _md_cell(row.status),
                    _md_cell(row.bib_title),
                    _md_cell(row.expected_md_path),
                    _md_cell(row.notes),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_from_paths(
    *,
    bib_path: Path,
    tex_root: Path,
    markdown_root: Path,
    include_quarantine: bool = False,
    include_uncited_bib: bool = False,
) -> list[CitationMapRow]:
    bib_entries = parse_bibtex_file(bib_path)
    citation_uses = parse_tex_citations(tex_root)
    markdown_records = load_markdown_records(markdown_root, include_quarantine=include_quarantine)
    return build_citation_rows(
        bib_entries,
        citation_uses,
        markdown_records,
        include_uncited_bib=include_uncited_bib,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    rows = build_from_paths(
        bib_path=args.bib_path,
        tex_root=args.tex_root,
        markdown_root=args.markdown_root,
        include_quarantine=args.include_quarantine,
        include_uncited_bib=args.include_uncited_bib,
    )
    write_csv(rows, args.csv_output)
    write_markdown(rows, args.markdown_output)

    counts = Counter(row.status for row in rows)
    print(f"wrote {len(rows)} rows to {args.csv_output}")
    print(f"wrote Markdown report to {args.markdown_output}")
    for status in sorted(counts):
        print(f"{status}: {counts[status]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
