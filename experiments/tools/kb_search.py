#!/usr/bin/env python3
"""Read-only deterministic search over local knowledge-base Markdown mirrors."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKDOWN_ROOT = REPO_ROOT / "original_papers" / "markdown"
METADATA_RE = re.compile(r"^-\s+`([^`]+)`:\s+`(.*)`\s*$")
SOURCE_PAGE_RE = re.compile(r"^<!--\s*source-page:\s*(\d+)\s*-->\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-'.]*")
STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "find",
    "for",
    "from",
    "i",
    "in",
    "is",
    "of",
    "on",
    "or",
    "papers",
    "paper",
    "read",
    "relevant",
    "should",
    "that",
    "the",
    "to",
    "use",
    "what",
    "where",
    "which",
    "with",
}
MIN_CHUNK_WORDS = 60
MAX_CHUNK_WORDS = 420
OVERLAP_WORDS = 40
DEFAULT_TOP_K = 5
SEARCH_METADATA_FIELDS = (
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
class Chunk:
    chunk_id: int
    doc_id: str
    title: str
    short_title: str
    theme: str
    status: str
    venue: str | None
    year: int | None
    md_path: str
    source_pdf_relative_path: str | None
    metadata_text: str
    source_page: int | None
    heading_path: str
    text: str
    chunk_word_count: int
    quarantine: bool


@dataclasses.dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    doc_id: str
    title: str
    short_title: str
    status: str
    theme: str
    venue: str | None
    year: int | None
    md_path: str
    source_pdf_relative_path: str | None
    source_page: int | None
    heading_path: str
    chunk_word_count: int
    snippet: str


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search original_papers/markdown Markdown mirrors with SQLite FTS5 when available."
    )
    parser.add_argument("query", nargs="*", help="Search query.")
    parser.add_argument("--query", dest="query_text", help="Search query. Alternative to positional terms.")
    parser.add_argument("--theme", help="Filter to a theme_dir value such as 01_transformer_attention_photonic.")
    parser.add_argument("--year-min", type=int, help="Filter to papers with year >= this value.")
    parser.add_argument("--status", default="ACTIVE", help="Filter by metadata status. Use ALL to disable. Default: ACTIVE.")
    parser.add_argument("--year-max", type=int, help="Filter to papers with year <= this value.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help=f"Number of results to return. Default: {DEFAULT_TOP_K}.")
    parser.add_argument("--include-quarantine", action="store_true", help="Include quarantined markdown mirrors.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON array instead of text.")
    parser.add_argument("--jsonl", action="store_true", help="Emit one JSON object per result.")
    parser.add_argument(
        "--markdown-root",
        type=Path,
        default=DEFAULT_MARKDOWN_ROOT,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.query_text and args.query:
        parser.error("provide query either as positional terms or --query, not both")
    if args.query_text:
        args.query = [args.query_text]
    if not args.query:
        parser.error("query is required")
    return args


def parse_metadata(lines: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines[:80]:
        match = METADATA_RE.match(line)
        if match:
            metadata[match.group(1)] = match.group(2)
    return metadata


def parse_year(raw: str | None) -> int | None:
    if not raw:
        return None
    match = re.search(r"\d{4}", raw)
    return int(match.group(0)) if match else None


def clean_heading(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().strip("#").strip())


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def normalize_text(lines: Iterable[str]) -> str:
    blocks: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("> Source page contains non-text"):
            continue
        if METADATA_RE.match(stripped) or SOURCE_PAGE_RE.match(stripped):
            continue
        blocks.append(stripped)
    return re.sub(r"\s+", " ", " ".join(blocks)).strip()


def split_text_by_word_bounds(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    words = tokenize(text)
    if len(words) <= max_words:
        return [text] if text else []

    pieces: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        pieces.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - OVERLAP_WORDS)
    return pieces


def flush_section(
    chunks: list[Chunk],
    *,
    next_id: int,
    doc_id: str,
    title: str,
    short_title: str,
    theme: str,
    status: str,
    venue: str | None,
    year: int | None,
    md_path: str,
    source_pdf_relative_path: str | None,
    metadata_text: str,
    source_page: int | None,
    heading_path: str,
    section_lines: list[str],
    quarantine: bool,
) -> int:
    text = normalize_text(section_lines)
    word_count = len(tokenize(text))
    if word_count < MIN_CHUNK_WORDS and chunks:
        previous = chunks[-1]
        if (
            previous.md_path == md_path
            and previous.chunk_word_count + word_count <= MAX_CHUNK_WORDS
        ):
            merged_text = f"{previous.text} {text}".strip()
            merged = dataclasses.replace(
                previous,
                text=merged_text,
                chunk_word_count=len(tokenize(merged_text)),
            )
            chunks[-1] = merged
            return next_id
    if word_count < MIN_CHUNK_WORDS:
        return next_id
    for piece in split_text_by_word_bounds(text):
        if not piece:
            continue
        chunk_word_count = len(tokenize(piece))
        chunks.append(
            Chunk(
                chunk_id=next_id,
                doc_id=doc_id,
                title=title,
                short_title=short_title,
                theme=theme,
                status=status,
                venue=venue,
                year=year,
                md_path=md_path,
                source_pdf_relative_path=source_pdf_relative_path,
                metadata_text=metadata_text,
                source_page=source_page,
                heading_path=heading_path,
                text=piece,
                chunk_word_count=chunk_word_count,
                quarantine=quarantine,
            )
        )
        next_id += 1
    return next_id


def chunks_for_file(path: Path, root: Path, start_id: int) -> tuple[list[Chunk], int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    metadata = parse_metadata(lines)
    rel_path = path.relative_to(root).as_posix()
    repo_rel_path = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else rel_path
    doc_id = rel_path.removesuffix(".md")
    theme = metadata.get("theme_dir") or rel_path.split("/", 1)[0]
    title = metadata.get("short_title") or (clean_heading(lines[0]) if lines else path.stem)
    short_title = metadata.get("short_title") or title
    status = metadata.get("status") or ("QUARANTINED" if "quarantine/" in rel_path else "ACTIVE")
    venue = metadata.get("venue")
    year = parse_year(metadata.get("year"))
    source_pdf_relative_path = metadata.get("source_pdf_relative_path")
    metadata_text = " ".join(
        value for field in SEARCH_METADATA_FIELDS if (value := metadata.get(field))
    )
    quarantine = rel_path.startswith("quarantine/")

    chunks: list[Chunk] = []
    next_id = start_id
    source_page: int | None = None
    heading_stack: list[tuple[int, str]] = []
    section_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        page_match = SOURCE_PAGE_RE.match(stripped)
        heading_match = HEADING_RE.match(stripped)
        if page_match or heading_match:
            heading_path = " > ".join(item[1] for item in heading_stack)
            next_id = flush_section(
                chunks,
                next_id=next_id,
                doc_id=doc_id,
                title=title,
                short_title=short_title,
                theme=theme,
                status=status,
                venue=venue,
                year=year,
                md_path=repo_rel_path,
                source_pdf_relative_path=source_pdf_relative_path,
                metadata_text=metadata_text,
                source_page=source_page,
                heading_path=heading_path,
                section_lines=section_lines,
                quarantine=quarantine,
            )
            section_lines = []
            if page_match:
                source_page = int(page_match.group(1))
            elif heading_match:
                level = len(heading_match.group(1))
                heading = clean_heading(heading_match.group(2))
                heading_stack = [item for item in heading_stack if item[0] < level]
                heading_stack.append((level, heading))
                section_lines.append(line)
            continue
        section_lines.append(line)

    heading_path = " > ".join(item[1] for item in heading_stack)
    next_id = flush_section(
        chunks,
        next_id=next_id,
        doc_id=doc_id,
        title=title,
        short_title=short_title,
        theme=theme,
        status=status,
        venue=venue,
        year=year,
        md_path=repo_rel_path,
        source_pdf_relative_path=source_pdf_relative_path,
        metadata_text=metadata_text,
        source_page=source_page,
        heading_path=heading_path,
        section_lines=section_lines,
        quarantine=quarantine,
    )
    return chunks, next_id


def iter_markdown_files(root: Path, include_quarantine: bool) -> Iterable[Path]:
    for path in sorted(root.rglob("*.md")):
        if path.name in {"INDEX.md", "README.md", "SUMMARY.md"}:
            continue
        rel_path = path.relative_to(root).as_posix()
        if not include_quarantine and rel_path.startswith("quarantine/"):
            continue
        yield path


def load_chunks(root: Path, include_quarantine: bool) -> list[Chunk]:
    chunks: list[Chunk] = []
    next_id = 1
    for path in iter_markdown_files(root, include_quarantine):
        file_chunks, next_id = chunks_for_file(path, root, next_id)
        chunks.extend(file_chunks)
    return chunks


def fts5_available() -> bool:
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        conn.close()
        return True
    except sqlite3.Error:
        return False


def fts_query(query: str) -> str:
    terms = re.findall(r"[A-Za-z0-9]+", query.lower())
    terms = [term for term in terms if term and term not in STOPWORDS]
    return " OR ".join(f"{term}*" for term in terms)


def build_fts_index(chunks: Sequence[Chunk]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_id INTEGER PRIMARY KEY,
            doc_id TEXT NOT NULL,
            title TEXT NOT NULL,
            short_title TEXT NOT NULL,
            theme TEXT NOT NULL,
            status TEXT NOT NULL,
            venue TEXT,
            year INTEGER,
            md_path TEXT NOT NULL,
            source_pdf_relative_path TEXT,
            metadata_text TEXT NOT NULL,
            source_page INTEGER,
            heading_path TEXT NOT NULL,
            text TEXT NOT NULL,
            chunk_word_count INTEGER NOT NULL,
            quarantine INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            doc_id, title, short_title, theme, heading_path, metadata_text, text,
            content='chunks',
            content_rowid='chunk_id',
            tokenize='unicode61'
        )
        """
    )
    rows = [
        (
            chunk.chunk_id,
            chunk.doc_id,
            chunk.title,
            chunk.short_title,
            chunk.theme,
            chunk.status,
            chunk.venue,
            chunk.year,
            chunk.md_path,
            chunk.source_pdf_relative_path,
            chunk.metadata_text,
            chunk.source_page,
            chunk.heading_path,
            chunk.text,
            chunk.chunk_word_count,
            int(chunk.quarantine),
        )
        for chunk in chunks
    ]
    conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
    return conn


def sql_filters(args: argparse.Namespace) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if args.theme:
        clauses.append("c.theme = ?")
        params.append(args.theme)
    if args.year_min is not None:
        clauses.append("c.year IS NOT NULL AND c.year >= ?")
        params.append(args.year_min)
    if args.year_max is not None:
        clauses.append("c.year IS NOT NULL AND c.year <= ?")
        params.append(args.year_max)
    if args.status and args.status.upper() != "ALL":
        clauses.append("c.status = ?")
        params.append(args.status)
    if not args.include_quarantine:
        clauses.append("c.quarantine = 0")
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def search_fts(chunks: Sequence[Chunk], args: argparse.Namespace) -> list[SearchResult]:
    match_query = fts_query(args.query)
    if not match_query:
        return []
    conn = build_fts_index(chunks)
    where_sql, filter_params = sql_filters(args)
    sql = f"""
        SELECT c.doc_id, c.title, c.short_title, c.status, c.theme, c.venue, c.year,
               c.md_path, c.source_pdf_relative_path, c.source_page, c.heading_path,
               c.chunk_word_count, c.metadata_text, c.text,
               bm25(chunks_fts, 3.0, 5.0, 5.0, 1.5, 2.0, 4.0, 1.0) AS raw_score
        FROM chunks_fts
        JOIN chunks c ON c.chunk_id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?{where_sql}
        ORDER BY raw_score ASC, c.md_path ASC, c.source_page ASC, c.heading_path ASC
        LIMIT ?
    """
    fetch_limit = max(args.top_k * 20, 100)
    rows = conn.execute(sql, [match_query, *filter_params, fetch_limit]).fetchall()
    candidates: list[tuple[float, sqlite3.Row]] = []
    seen_paths: set[str] = set()
    for row in rows:
        if row["md_path"] in seen_paths:
            continue
        seen_paths.add(row["md_path"])
        score = round(-float(row["raw_score"]) + path_score_adjustment(row["md_path"]), 6)
        candidates.append((score, row))
    candidates.sort(key=lambda item: (-item[0], item[1]["md_path"], item[1]["source_page"] or -1, item[1]["heading_path"]))

    results: list[SearchResult] = []
    for score, row in candidates[: args.top_k]:
        snippet_text = " ".join(part for part in [row["title"], row["heading_path"], row["text"]] if part)
        results.append(
            SearchResult(
                rank=len(results) + 1,
                score=score,
                doc_id=row["doc_id"],
                title=row["title"],
                short_title=row["short_title"],
                status=row["status"],
                theme=row["theme"],
                venue=row["venue"],
                year=row["year"],
                md_path=row["md_path"],
                source_pdf_relative_path=row["source_pdf_relative_path"],
                source_page=row["source_page"],
                heading_path=row["heading_path"],
                chunk_word_count=row["chunk_word_count"],
                snippet=make_snippet(snippet_text, args.query),
            )
        )
    return results


def passes_filters(chunk: Chunk, args: argparse.Namespace) -> bool:
    if args.theme and chunk.theme != args.theme:
        return False
    if args.year_min is not None and (chunk.year is None or chunk.year < args.year_min):
        return False
    if args.year_max is not None and (chunk.year is None or chunk.year > args.year_max):
        return False
    if args.status and args.status.upper() != "ALL" and chunk.status != args.status:
        return False
    if not args.include_quarantine and chunk.quarantine:
        return False
    return True


def search_fallback(chunks: Sequence[Chunk], args: argparse.Namespace) -> list[SearchResult]:
    terms = [term.lower() for term in WORD_RE.findall(args.query)]
    terms = [term for term in terms if term and term not in STOPWORDS]
    if not terms:
        return []
    scored: list[tuple[float, Chunk]] = []
    for chunk in chunks:
        if not passes_filters(chunk, args):
            continue
        haystacks = [
            chunk.doc_id.lower(),
            chunk.title.lower(),
            chunk.theme.lower(),
            chunk.heading_path.lower(),
            chunk.metadata_text.lower(),
            chunk.text.lower(),
        ]
        score = 0.0
        for term in terms:
            score += haystacks[0].count(term) * 3.0
            score += haystacks[1].count(term) * 5.0
            score += haystacks[2].count(term) * 1.5
            score += haystacks[3].count(term) * 2.0
            score += haystacks[4].count(term) * 4.0
            score += haystacks[5].count(term)
        score += path_score_adjustment(chunk.md_path)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].md_path, item[1].source_page or -1, item[1].heading_path))
    results: list[SearchResult] = []
    seen_paths: set[str] = set()
    for score, chunk in scored:
        if chunk.md_path in seen_paths:
            continue
        seen_paths.add(chunk.md_path)
        results.append(
            SearchResult(
                rank=len(results) + 1,
                score=round(score, 6),
                doc_id=chunk.doc_id,
                title=chunk.title,
                short_title=chunk.short_title,
                status=chunk.status,
                theme=chunk.theme,
                venue=chunk.venue,
                year=chunk.year,
                md_path=chunk.md_path,
                source_pdf_relative_path=chunk.source_pdf_relative_path,
                source_page=chunk.source_page,
                heading_path=chunk.heading_path,
                chunk_word_count=chunk.chunk_word_count,
                snippet=make_snippet(" ".join(part for part in [chunk.title, chunk.heading_path, chunk.text] if part), args.query),
            )
        )
        if len(results) >= args.top_k:
            break
    return results


def path_score_adjustment(md_path: str) -> float:
    if "08_books_and_design_references/" not in md_path:
        return 0.0
    if "/__chapters/" in md_path:
        if md_path.endswith("/CHAPTER_INDEX.md"):
            return 0.5
        return 2.0
    if md_path.endswith("Silicon_Photonics_Design_From_Devices_to_Systems_CN_Translation.md"):
        return -2.0
    return 0.0


def make_snippet(text: str, query: str, width: int = 260) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= width:
        return compact
    terms = [re.escape(term) for term in WORD_RE.findall(query)]
    match = re.search("|".join(terms), compact, flags=re.IGNORECASE) if terms else None
    center = match.start() if match else 0
    start = max(0, center - width // 3)
    end = min(len(compact), start + width)
    start = max(0, end - width)
    snippet = compact[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(compact):
        snippet += "..."
    return snippet


def emit_text(results: Sequence[SearchResult]) -> None:
    for result in results:
        page = result.source_page if result.source_page is not None else "?"
        heading = result.heading_path or "(no heading)"
        year = result.year if result.year is not None else "?"
        print(f"{result.rank}. score={result.score:.6f} | {result.status} | {year} | {result.title}")
        print(f"   theme: {result.theme}")
        print(f"   path: {result.md_path}")
        print(f"   source_page: {page}")
        print(f"   heading_path: {heading}")
        print(f"   chunk_words: {result.chunk_word_count}")
        print(f"   snippet: {result.snippet}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    args.query = " ".join(args.query)
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")
    root = args.markdown_root.resolve()
    if not root.exists():
        raise SystemExit(f"Markdown root does not exist: {root}")

    chunks = load_chunks(root, include_quarantine=args.include_quarantine)
    results = search_fts(chunks, args) if fts5_available() else search_fallback(chunks, args)
    if args.json:
        print(json.dumps([dataclasses.asdict(result) for result in results], indent=2, ensure_ascii=False))
    elif args.jsonl:
        for result in results:
            print(json.dumps(dataclasses.asdict(result), ensure_ascii=False))
    else:
        emit_text(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
