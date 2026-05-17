#!/usr/bin/env python3
"""Hybrid deterministic/embedding reranker for citation-grounding candidates."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:  # pragma: no cover - exercised by script execution, not package import.
    from . import build_kb_embedding_index as rag_index
    from . import kb_search
except ImportError:  # pragma: no cover
    import build_kb_embedding_index as rag_index  # type: ignore
    import kb_search  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CITATION_MAP_CSV = REPO_ROOT / "peper_writing" / "citation_key_map_20260429.csv"
DEFAULT_CLAIM_AUDIT_CSV = REPO_ROOT / "peper_writing" / "citation_claim_audit_20260429.csv"
DEFAULT_CHUNKS_JSONL = REPO_ROOT / ".cache" / "kb_rag" / "chunks.jsonl"
DEFAULT_EMBEDDINGS_SQLITE = REPO_ROOT / ".cache" / "kb_rag" / "embeddings.sqlite"
DEFAULT_MANIFEST = REPO_ROOT / ".cache" / "kb_rag" / "embedding_manifest.json"
DEFAULT_REPORT = REPO_ROOT / "peper_writing" / "rag_citation_rerank_eval_20260429.md"
DEFAULT_MARKDOWN_ROOT = REPO_ROOT / "original_papers" / "markdown"
DEFAULT_MODEL_NAME = rag_index.DEFAULT_MODEL_NAME

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-'.]*")
REFERENCE_SECTION_HEADINGS = {
    "bibliography",
    "literature cited",
    "reference list",
    "references",
    "references and notes",
    "works cited",
}
REFERENCE_TEXT_RE = re.compile(
    r"(?:^\s*(?:references?|bibliography)\s*$|^\s*\[\d+\]\s+|^\s*\d+\.\s+[A-Z][^.]{10,120}\.\s)",
    re.IGNORECASE,
)
STOPWORDS = kb_search.STOPWORDS | {
    "also",
    "can",
    "into",
    "than",
    "these",
    "this",
    "using",
    "via",
    "we",
}


@dataclasses.dataclass(frozen=True)
class CitationMapEntry:
    citation_key: str
    status: str = ""
    expected_md_path: str | None = None
    theme_dir: str | None = None
    bib_title: str | None = None
    bib_year: int | None = None
    source_page_policy: str | None = None
    notes: str | None = None


@dataclasses.dataclass(frozen=True)
class Candidate:
    chunk_id: str
    md_path: str
    title: str
    theme_dir: str
    source_page: int | None
    heading_path: str
    snippet: str
    chunk_text: str = ""
    status: str = "ACTIVE"
    year: int | None = None
    deterministic_score: float = 0.0
    embedding_score: float = 0.0
    sources: tuple[str, ...] = ()
    quarantine: bool = False
    metadata: Mapping[str, object] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    score: float
    components: Mapping[str, float]


@dataclasses.dataclass(frozen=True)
class ClaimInput:
    claim_id: str
    claim_text: str
    citation_keys: tuple[str, ...]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank citation QA candidates with deterministic guardrails.")
    parser.add_argument("--claim-audit-csv", type=Path, default=DEFAULT_CLAIM_AUDIT_CSV)
    parser.add_argument("--citation-map-csv", type=Path, default=DEFAULT_CITATION_MAP_CSV)
    parser.add_argument("--chunks-jsonl", type=Path, default=DEFAULT_CHUNKS_JSONL)
    parser.add_argument("--embeddings-sqlite", type=Path, default=DEFAULT_EMBEDDINGS_SQLITE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--deterministic-top-k", type=int, default=10)
    parser.add_argument("--embedding-top-k", type=int, default=10)
    parser.add_argument("--final-top-k", type=int, default=5)
    parser.add_argument("--include-quarantine", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable ranking JSON to stdout.")
    return parser.parse_args(argv)


def normalize_md_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = str(path).replace("\\", "/").strip()
    if not raw:
        return ""
    try:
        path_obj = Path(raw)
        if path_obj.is_absolute():
            return path_obj.resolve().relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        pass
    return raw.removeprefix("./")


def _parse_int(raw: object) -> int | None:
    if raw in (None, ""):
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        match = re.search(r"\d{4}", str(raw))
        return int(match.group(0)) if match else None


def parse_citation_keys(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    cleaned = re.sub(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]*)\}", r"\1", raw)
    return tuple(key.strip() for key in re.split(r"[,;\s]+", cleaned) if key.strip())


def load_citation_map(path: Path) -> dict[str, CitationMapEntry]:
    if not path.exists():
        return {}
    entries: dict[str, CitationMapEntry] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("citation_key") or row.get("key") or "").strip()
            if not key:
                continue
            entries[key] = CitationMapEntry(
                citation_key=key,
                status=(row.get("status") or "").strip(),
                expected_md_path=normalize_md_path(row.get("expected_md_path") or row.get("md_path")),
                theme_dir=(row.get("theme_dir") or row.get("theme") or "").strip() or None,
                bib_title=(row.get("bib_title") or row.get("title") or "").strip() or None,
                bib_year=_parse_int(row.get("bib_year") or row.get("year")),
                source_page_policy=(row.get("source_page_policy") or "").strip() or None,
                notes=(row.get("notes") or "").strip() or None,
            )
    return entries


def load_claims(path: Path) -> list[ClaimInput]:
    if not path.exists():
        return []
    claims: list[ClaimInput] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            claim_text = (
                row.get("claim_text")
                or row.get("claim")
                or row.get("context")
                or row.get("query")
                or ""
            ).strip()
            if not claim_text:
                continue
            # The claim-audit CSV is one row per cited-key check; keep reranking
            # scoped to that row's key even when the surrounding claim has a
            # multi-key citation group.
            keys = parse_citation_keys(row.get("citation_key") or row.get("key") or row.get("citation_keys"))
            claims.append(
                ClaimInput(
                    claim_id=(row.get("claim_id") or row.get("id") or f"claim_{index:04d}").strip(),
                    claim_text=claim_text,
                    citation_keys=keys,
                )
            )
    return claims


def load_chunk_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def candidate_from_record(record: Mapping[str, object], *, source: str, score: float = 0.0) -> Candidate:
    text = str(record.get("chunk_text") or record.get("text") or "")
    snippet = str(record.get("snippet") or kb_search.make_snippet(text, ""))
    return Candidate(
        chunk_id=str(record.get("chunk_id") or record.get("id") or normalize_md_path(record.get("md_path"))),
        md_path=normalize_md_path(record.get("md_path")),
        title=str(record.get("title") or record.get("short_title") or ""),
        theme_dir=str(record.get("theme_dir") or record.get("theme") or ""),
        source_page=_parse_int(record.get("source_page")),
        heading_path=str(record.get("heading_path") or ""),
        snippet=snippet,
        chunk_text=text,
        status=str(record.get("status") or "ACTIVE"),
        year=_parse_int(record.get("year")),
        deterministic_score=score if source == "deterministic" else float(record.get("deterministic_score") or 0.0),
        embedding_score=score if source == "embedding" else float(record.get("embedding_score") or 0.0),
        sources=(source,),
        quarantine=bool(record.get("quarantine")) or "quarantine/" in normalize_md_path(record.get("md_path")),
        metadata=record,
    )


def candidate_from_search_result(result: kb_search.SearchResult) -> Candidate:
    return Candidate(
        chunk_id=f"{result.md_path}:{result.source_page or '?'}:{result.heading_path}",
        md_path=normalize_md_path(result.md_path),
        title=result.title,
        theme_dir=result.theme,
        source_page=result.source_page,
        heading_path=result.heading_path,
        snippet=result.snippet,
        chunk_text=result.snippet,
        status=result.status,
        year=result.year,
        deterministic_score=result.score,
        sources=("deterministic",),
        quarantine="quarantine/" in normalize_md_path(result.md_path),
        metadata={"doc_id": result.doc_id, "chunk_word_count": result.chunk_word_count},
    )


def is_active_candidate(candidate: Candidate, *, include_quarantine: bool = False) -> bool:
    if not include_quarantine and (candidate.quarantine or "quarantine/" in candidate.md_path):
        return False
    return candidate.status.upper() == "ACTIVE"


def token_counts(text: str) -> Counter[str]:
    return Counter(
        token.lower()
        for token in WORD_RE.findall(text)
        if token.lower() not in STOPWORDS and len(token) > 1
    )


def cosine_counts(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def deterministic_candidates(
    claim_text: str,
    chunks: Sequence[kb_search.Chunk],
    *,
    top_k: int,
    include_quarantine: bool = False,
    theme: str | None = None,
) -> list[Candidate]:
    args = argparse.Namespace(
        query=claim_text,
        theme=theme,
        year_min=None,
        year_max=None,
        status="ACTIVE",
        top_k=top_k,
        include_quarantine=include_quarantine,
    )
    results = kb_search.search_fts(chunks, args) if kb_search.fts5_available() else kb_search.search_fallback(chunks, args)
    return [candidate_from_search_result(result) for result in results]


def embedding_candidates_from_records(
    claim_text: str,
    records: Sequence[Mapping[str, object]],
    *,
    top_k: int,
    include_quarantine: bool = False,
    theme: str | None = None,
) -> list[Candidate]:
    query_counts = token_counts(claim_text)
    scored: list[tuple[float, Mapping[str, object]]] = []
    for record in records:
        candidate = candidate_from_record(record, source="embedding", score=0.0)
        if not is_active_candidate(candidate, include_quarantine=include_quarantine):
            continue
        if theme and candidate.theme_dir != theme:
            continue
        text = " ".join(
            [
                candidate.title,
                candidate.heading_path,
                str(record.get("metadata_text") or ""),
                candidate.chunk_text,
            ]
        )
        score = cosine_counts(query_counts, token_counts(text))
        if score > 0:
            scored.append((round(score, 6), record))
    scored.sort(key=lambda item: (-item[0], normalize_md_path(item[1].get("md_path")), str(item[1].get("source_page") or "")))
    return [candidate_from_record(record, source="embedding", score=score) for score, record in scored[:top_k]]


def record_lookup(records: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    return {str(record.get("chunk_id")): record for record in records}


def embedding_lookup(embeddings: Sequence[rag_index.EmbeddingRecord]) -> dict[str, rag_index.EmbeddingRecord]:
    return {record.chunk_id: record for record in embeddings}


def encode_query_vectors(claims: Sequence[ClaimInput], *, model_name: str, device: str) -> dict[str, object]:
    if not claims:
        return {}
    _torch, sentence_transformer_cls, _blocker = rag_index._load_embedding_backend(device, required=True)
    model = sentence_transformer_cls(model_name, device=device)
    unique_queries = sorted({claim.claim_text for claim in claims if claim.claim_text.strip()})
    encoded = model.encode(
        [rag_index.e5_query_text(query) for query in unique_queries],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    matrix = rag_index.normalize_rows(encoded)
    return {query: matrix[index] for index, query in enumerate(unique_queries)}


def vector_candidates_from_embeddings(
    claim_text: str,
    records: Sequence[Mapping[str, object]],
    embeddings: Sequence[rag_index.EmbeddingRecord],
    query_vector: object,
    *,
    top_k: int,
    include_quarantine: bool = False,
    theme: str | None = None,
) -> list[Candidate]:
    import numpy as np

    record_by_chunk = record_lookup(records)
    query = np.asarray(query_vector, dtype=np.float32)
    scored: list[tuple[float, Mapping[str, object]]] = []
    for embedding in embeddings:
        record = record_by_chunk.get(embedding.chunk_id)
        if record is None:
            continue
        candidate = candidate_from_record(record, source="embedding", score=0.0)
        if not is_active_candidate(candidate, include_quarantine=include_quarantine):
            continue
        if theme and candidate.theme_dir != theme:
            continue
        score = float(np.dot(query, np.asarray(embedding.vector, dtype=np.float32)))
        scored.append((round(score, 6), record))
    scored.sort(key=lambda item: (-item[0], normalize_md_path(item[1].get("md_path")), str(item[1].get("source_page") or "")))
    return [candidate_from_record(record, source="embedding", score=score) for score, record in scored[:top_k]]


def mapped_path_candidates_from_records(
    claim_text: str,
    citation_keys: Sequence[str],
    citation_map: Mapping[str, CitationMapEntry],
    records: Sequence[Mapping[str, object]],
    *,
    top_per_path: int = 3,
    include_quarantine: bool = False,
) -> list[Candidate]:
    expected_paths = _normalized_expected_paths(_matching_entries(citation_keys, citation_map))
    if not expected_paths:
        return []

    query_counts = token_counts(claim_text)
    candidates: list[Candidate] = []
    for expected_path in sorted(expected_paths):
        scored: list[tuple[float, Mapping[str, object]]] = []
        for record in records:
            if normalize_md_path(record.get("md_path")) != expected_path:
                continue
            candidate = candidate_from_record(record, source="embedding", score=0.0)
            if not is_active_candidate(candidate, include_quarantine=include_quarantine):
                continue
            text = " ".join(
                [
                    candidate.title,
                    candidate.heading_path,
                    str(record.get("metadata_text") or ""),
                    candidate.chunk_text,
                ]
            )
            score = round(cosine_counts(query_counts, token_counts(text)), 6)
            scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("source_page") or ""), str(item[1].get("heading_path") or "")))
        for score, record in scored[:top_per_path]:
            candidate = candidate_from_record(record, source="embedding", score=score)
            candidates.append(dataclasses.replace(candidate, sources=("citation_map",), embedding_score=score))
    return candidates


def mapped_path_candidates_from_embeddings(
    query_vector: object,
    citation_keys: Sequence[str],
    citation_map: Mapping[str, CitationMapEntry],
    records: Sequence[Mapping[str, object]],
    embeddings: Sequence[rag_index.EmbeddingRecord],
    *,
    top_per_path: int = 3,
    include_quarantine: bool = False,
) -> list[Candidate]:
    import numpy as np

    expected_paths = _normalized_expected_paths(_matching_entries(citation_keys, citation_map))
    if not expected_paths:
        return []
    embedding_by_chunk = embedding_lookup(embeddings)
    query = np.asarray(query_vector, dtype=np.float32)
    scored_by_path: dict[str, list[tuple[float, Mapping[str, object]]]] = {path: [] for path in expected_paths}
    for record in records:
        md_path = normalize_md_path(record.get("md_path"))
        if md_path not in expected_paths:
            continue
        candidate = candidate_from_record(record, source="embedding", score=0.0)
        if not is_active_candidate(candidate, include_quarantine=include_quarantine):
            continue
        embedding = embedding_by_chunk.get(str(record.get("chunk_id")))
        if embedding is None:
            continue
        score = float(np.dot(query, np.asarray(embedding.vector, dtype=np.float32)))
        scored_by_path[md_path].append((round(score, 6), record))
    candidates: list[Candidate] = []
    for expected_path in sorted(scored_by_path):
        scored = scored_by_path[expected_path]
        scored.sort(key=lambda item: (-item[0], str(item[1].get("source_page") or ""), str(item[1].get("heading_path") or "")))
        for score, record in scored[:top_per_path]:
            candidate = candidate_from_record(record, source="embedding", score=score)
            candidates.append(dataclasses.replace(candidate, sources=("citation_map",), embedding_score=score))
    return candidates


def merge_candidates(
    deterministic: Sequence[Candidate],
    embedding: Sequence[Candidate],
    *,
    include_quarantine: bool = False,
) -> list[Candidate]:
    merged: dict[tuple[str, int | None, str], Candidate] = {}
    for candidate in [*deterministic, *embedding]:
        if not is_active_candidate(candidate, include_quarantine=include_quarantine):
            continue
        key = (candidate.md_path, candidate.source_page, candidate.heading_path)
        previous = merged.get(key)
        if previous is None:
            merged[key] = candidate
            continue
        sources = tuple(sorted(set(previous.sources) | set(candidate.sources)))
        merged[key] = dataclasses.replace(
            previous,
            deterministic_score=max(previous.deterministic_score, candidate.deterministic_score),
            embedding_score=max(previous.embedding_score, candidate.embedding_score),
            snippet=previous.snippet if len(previous.snippet) >= len(candidate.snippet) else candidate.snippet,
            chunk_text=previous.chunk_text if len(previous.chunk_text) >= len(candidate.chunk_text) else candidate.chunk_text,
            sources=sources,
        )
    return list(merged.values())


def is_reference_section(candidate: Candidate) -> bool:
    heading_segments = [segment.strip().lower() for segment in re.split(r"\s*>\s*", candidate.heading_path) if segment.strip()]
    if any(segment in REFERENCE_SECTION_HEADINGS for segment in heading_segments):
        return True
    text = candidate.chunk_text or candidate.snippet
    return bool(REFERENCE_TEXT_RE.search(text.strip()))


def is_chapter_chunk(candidate: Candidate) -> bool:
    scope = str(candidate.metadata.get("document_scope") or candidate.metadata.get("record_type") or "").lower()
    if scope in {"chapter", "book_chapter"}:
        return True
    return "/__chapters/" in candidate.md_path


def is_full_book_chunk(candidate: Candidate) -> bool:
    scope = str(candidate.metadata.get("document_scope") or candidate.metadata.get("record_type") or "").lower()
    if scope in {"full_book", "book"}:
        return True
    if "/__chapters/" in candidate.md_path:
        return False
    return candidate.md_path.startswith("08_books_and_design_references/") and candidate.md_path.endswith(".md")


def book_family(candidate: Candidate) -> str:
    explicit = candidate.metadata.get("book_id") or candidate.metadata.get("source_pdf_relative_path")
    if explicit:
        return str(explicit)
    path = candidate.md_path
    if "/__chapters/" in path:
        prefix, rest = path.split("/__chapters/", 1)
        book_dir = rest.split("/", 1)[0]
        return f"{prefix}/{book_dir.removesuffix('.md')}"
    parent, _, filename = path.rpartition("/")
    stem = filename.removesuffix(".md")
    return f"{parent}/{stem}" if parent else stem


def has_competing_chapter(candidate: Candidate, candidates: Sequence[Candidate]) -> bool:
    if not is_full_book_chunk(candidate):
        return False
    family = book_family(candidate)
    return any(
        other is not candidate
        and is_chapter_chunk(other)
        and book_family(other) == family
        for other in candidates
    )


def _matching_entries(
    citation_keys: Sequence[str],
    citation_map: Mapping[str, CitationMapEntry],
) -> list[CitationMapEntry]:
    return [citation_map[key] for key in citation_keys if key in citation_map]


def _normalized_expected_paths(entries: Sequence[CitationMapEntry]) -> set[str]:
    return {normalize_md_path(entry.expected_md_path) for entry in entries if normalize_md_path(entry.expected_md_path)}


def _entry_themes(entries: Sequence[CitationMapEntry]) -> set[str]:
    return {entry.theme_dir for entry in entries if entry.theme_dir}


def rank_candidates(
    candidates: Sequence[Candidate],
    *,
    citation_keys: Sequence[str] = (),
    citation_map: Mapping[str, CitationMapEntry] | None = None,
    include_quarantine: bool = False,
) -> list[RankedCandidate]:
    active = [candidate for candidate in candidates if is_active_candidate(candidate, include_quarantine=include_quarantine)]
    if not active:
        return []

    citation_map = citation_map or {}
    entries = _matching_entries(citation_keys, citation_map)
    expected_paths = _normalized_expected_paths(entries)
    themes = _entry_themes(entries)
    has_local_exact = any(entry.status in {"local_exact", "local_probable"} and entry.expected_md_path for entry in entries)

    max_det = max((abs(candidate.deterministic_score) for candidate in active), default=0.0)
    max_emb = max((abs(candidate.embedding_score) for candidate in active), default=0.0)

    ranked: list[RankedCandidate] = []
    for candidate in active:
        normalized_path = normalize_md_path(candidate.md_path)
        exact_path_match = 1.0 if normalized_path in expected_paths else 0.0
        source_page_bonus = 0.75 if candidate.source_page is not None else 0.0
        theme_consistency = 0.75 if themes and candidate.theme_dir in themes else 0.0
        deterministic_component = (candidate.deterministic_score / max_det) if max_det else 0.0
        embedding_component = (candidate.embedding_score / max_emb) if max_emb else 0.0
        constraint_component = -1.25 if has_local_exact and not exact_path_match else 0.0
        reference_penalty = -3.0 if is_reference_section(candidate) else 0.0
        full_book_penalty = -2.5 if has_competing_chapter(candidate, active) else 0.0
        exact_path_bonus = 8.0 * exact_path_match

        components = {
            "exact_path_bonus": exact_path_bonus,
            "source_page_bonus": source_page_bonus,
            "theme_consistency": theme_consistency,
            "deterministic_score": 1.25 * deterministic_component,
            "embedding_score": 1.0 * embedding_component,
            "citation_constraint": constraint_component,
            "reference_section_penalty": reference_penalty,
            "full_book_penalty": full_book_penalty,
        }
        score = round(sum(components.values()), 6)
        ranked.append(RankedCandidate(candidate=candidate, score=score, components=components))

    ranked.sort(
        key=lambda item: (
            -item.score,
            -item.components.get("exact_path_bonus", 0.0),
            -(1 if item.candidate.source_page is not None else 0),
            item.candidate.md_path,
            item.candidate.source_page or -1,
            item.candidate.heading_path,
        )
    )
    return ranked


def choose_theme(citation_keys: Sequence[str], citation_map: Mapping[str, CitationMapEntry]) -> str | None:
    themes = _entry_themes(_matching_entries(citation_keys, citation_map))
    return sorted(themes)[0] if len(themes) == 1 else None


def verdict_for(claim: ClaimInput, ranked: Sequence[RankedCandidate], citation_map: Mapping[str, CitationMapEntry]) -> str:
    entries = _matching_entries(claim.citation_keys, citation_map)
    if entries and all(entry.status == "external_or_bib_only" for entry in entries):
        return "warn_external"
    if not ranked:
        return "review"
    expected_paths = _normalized_expected_paths(entries)
    top_path = normalize_md_path(ranked[0].candidate.md_path)
    if expected_paths and top_path in expected_paths:
        return "pass"
    return "review"


def evaluate_claims(
    claims: Sequence[ClaimInput],
    *,
    citation_map: Mapping[str, CitationMapEntry],
    chunks: Sequence[kb_search.Chunk],
    chunk_records: Sequence[Mapping[str, object]],
    deterministic_top_k: int,
    embedding_top_k: int,
    final_top_k: int,
    embedding_records: Sequence[rag_index.EmbeddingRecord] = (),
    query_vectors: Mapping[str, object] | None = None,
    include_quarantine: bool = False,
) -> list[dict[str, object]]:
    evaluations: list[dict[str, object]] = []
    for claim in claims:
        theme = choose_theme(claim.citation_keys, citation_map)
        deterministic = deterministic_candidates(
            claim.claim_text,
            chunks,
            top_k=deterministic_top_k,
            include_quarantine=include_quarantine,
            theme=theme,
        )
        query_vector = (query_vectors or {}).get(claim.claim_text)
        if embedding_records and query_vector is not None:
            embedding = vector_candidates_from_embeddings(
                claim.claim_text,
                chunk_records,
                embedding_records,
                query_vector,
                top_k=embedding_top_k,
                include_quarantine=include_quarantine,
                theme=theme,
            )
            mapped = mapped_path_candidates_from_embeddings(
                query_vector,
                claim.citation_keys,
                citation_map,
                chunk_records,
                embedding_records,
                include_quarantine=include_quarantine,
            )
        else:
            embedding = []
            mapped = mapped_path_candidates_from_records(
                claim.claim_text,
                claim.citation_keys,
                citation_map,
                chunk_records,
                include_quarantine=include_quarantine,
            )
        merged = merge_candidates([*deterministic, *mapped], embedding, include_quarantine=include_quarantine)
        ranked = rank_candidates(
            merged,
            citation_keys=claim.citation_keys,
            citation_map=citation_map,
            include_quarantine=include_quarantine,
        )[:final_top_k]
        evaluations.append(
            {
                "claim": claim,
                "verdict": verdict_for(claim, ranked, citation_map),
                "ranked": ranked,
                "deterministic_count": len(deterministic),
                "embedding_count": len(embedding),
                "mapped_count": len(mapped),
                "merged_count": len(merged),
            }
        )
    return evaluations


def render_report(
    evaluations: Sequence[Mapping[str, object]],
    *,
    claim_audit_csv: Path,
    citation_map_csv: Path,
    chunks_jsonl: Path,
    embeddings_sqlite: Path,
    embedding_status: str,
) -> str:
    lines = [
        "# RAG Citation Hybrid Rerank Evaluation 2026-04-29",
        "",
        "## Inputs",
        "",
        f"- claim_audit_csv: `{claim_audit_csv}`",
        f"- citation_map_csv: `{citation_map_csv}`",
        f"- chunks_jsonl: `{chunks_jsonl}`",
        f"- embeddings_sqlite: `{embeddings_sqlite}`",
        f"- embedding_status: `{embedding_status}`",
        "",
    ]
    if not evaluations:
        lines.extend(
            [
                "## Status",
                "",
                "No claim audit rows were available, so no claim-level reranking was run.",
                "",
            ]
        )
        return "\n".join(lines)

    verdict_counts = Counter(str(item["verdict"]) for item in evaluations)
    lines.extend(["## Summary", ""])
    for verdict, count in sorted(verdict_counts.items()):
        lines.append(f"- {verdict}: {count}")
    lines.append("")

    lines.extend(["## Claim Rankings", ""])
    for item in evaluations:
        claim = item["claim"]
        assert isinstance(claim, ClaimInput)
        ranked = item["ranked"]
        assert isinstance(ranked, list)
        keys = ", ".join(claim.citation_keys) if claim.citation_keys else "(none)"
        lines.extend(
            [
                f"### {claim.claim_id}",
                "",
                f"- citation_keys: `{keys}`",
                f"- verdict: `{item['verdict']}`",
                f"- candidate_counts: deterministic={item['deterministic_count']}, mapped={item['mapped_count']}, embedding={item['embedding_count']}, merged={item['merged_count']}",
                f"- claim: {claim.claim_text}",
                "",
            ]
        )
        if not ranked:
            lines.extend(["No active candidates found.", ""])
            continue
        lines.append("| rank | score | path | source_page | heading | reasons |")
        lines.append("|---:|---:|---|---:|---|---|")
        for rank, ranked_candidate in enumerate(ranked, start=1):
            assert isinstance(ranked_candidate, RankedCandidate)
            candidate = ranked_candidate.candidate
            reasons = ", ".join(
                f"{key}={value:.2f}" for key, value in ranked_candidate.components.items() if abs(value) > 1e-9
            )
            page = candidate.source_page if candidate.source_page is not None else ""
            heading = candidate.heading_path.replace("|", "\\|") or "(no heading)"
            path = candidate.md_path.replace("|", "\\|")
            lines.append(f"| {rank} | {ranked_candidate.score:.3f} | `{path}` | {page} | {heading} | {reasons} |")
        lines.append("")
    return "\n".join(lines)


def ranked_to_dict(item: RankedCandidate) -> dict[str, object]:
    return {
        "score": item.score,
        "components": dict(item.components),
        "candidate": {
            "chunk_id": item.candidate.chunk_id,
            "md_path": item.candidate.md_path,
            "title": item.candidate.title,
            "theme_dir": item.candidate.theme_dir,
            "source_page": item.candidate.source_page,
            "heading_path": item.candidate.heading_path,
            "status": item.candidate.status,
            "deterministic_score": item.candidate.deterministic_score,
            "embedding_score": item.candidate.embedding_score,
            "sources": list(item.candidate.sources),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.deterministic_top_k < 1 or args.embedding_top_k < 1 or args.final_top_k < 1:
        raise SystemExit("top-k values must be >= 1")

    citation_map = load_citation_map(args.citation_map_csv)
    claims = load_claims(args.claim_audit_csv)
    chunk_records = rag_index.load_chunk_records(args.chunks_jsonl)
    chunks = kb_search.load_chunks(args.markdown_root.resolve(), include_quarantine=args.include_quarantine) if claims else []
    if claims and not chunk_records:
        chunk_records = [
            {
                "chunk_id": chunk.chunk_id,
                "md_path": chunk.md_path,
                "title": chunk.title,
                "theme_dir": chunk.theme,
                "source_page": chunk.source_page,
                "heading_path": chunk.heading_path,
                "chunk_text": chunk.text,
                "status": chunk.status,
                "year": chunk.year,
                "quarantine": chunk.quarantine,
                "metadata_text": chunk.metadata_text,
            }
            for chunk in chunks
        ]

    embedding_records: list[rag_index.EmbeddingRecord] = []
    query_vectors: dict[str, object] = {}
    embedding_status = "not_loaded"
    if claims and chunk_records and args.manifest.exists():
        manifest = rag_index.read_manifest(args.manifest)
        embedding_status = str(manifest.get("embedding_status") or "unknown")
        if embedding_status == "generated":
            rag_index.validate_index_manifest(
                manifest,
                chunk_records,
                expected_model_name=args.model_name,
                embeddings_sqlite=args.embeddings_sqlite,
            )
            embedding_records = rag_index.load_embedding_records(args.embeddings_sqlite)
            query_vectors = encode_query_vectors(claims, model_name=args.model_name, device=args.device)

    evaluations = evaluate_claims(
        claims,
        citation_map=citation_map,
        chunks=chunks,
        chunk_records=chunk_records,
        deterministic_top_k=args.deterministic_top_k,
        embedding_top_k=args.embedding_top_k,
        final_top_k=args.final_top_k,
        embedding_records=embedding_records,
        query_vectors=query_vectors,
        include_quarantine=args.include_quarantine,
    ) if claims else []

    report = render_report(
        evaluations,
        claim_audit_csv=args.claim_audit_csv,
        citation_map_csv=args.citation_map_csv,
        chunks_jsonl=args.chunks_jsonl,
        embeddings_sqlite=args.embeddings_sqlite,
        embedding_status=embedding_status,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8")

    if args.json:
        payload = [
            {
                "claim_id": item["claim"].claim_id,
                "verdict": item["verdict"],
                "ranked": [ranked_to_dict(ranked) for ranked in item["ranked"]],
            }
            for item in evaluations
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"RAG citation rerank report written: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
