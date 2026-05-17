#!/usr/bin/env python3
"""AI-facing hybrid vector RAG query over the local Markdown knowledge base."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

try:  # pragma: no cover - exercised by script execution, not package import.
    from . import build_kb_embedding_index as rag_index
    from . import kb_search
    from . import rag_citation_rerank as rerank
except ImportError:  # pragma: no cover
    import build_kb_embedding_index as rag_index  # type: ignore
    import kb_search  # type: ignore
    import rag_citation_rerank as rerank  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKDOWN_ROOT = REPO_ROOT / "original_papers" / "markdown"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "kb_rag"
DEFAULT_CHUNKS_JSONL = DEFAULT_CACHE_DIR / "chunks.jsonl"
DEFAULT_EMBEDDINGS_SQLITE = DEFAULT_CACHE_DIR / "embeddings.sqlite"
DEFAULT_MANIFEST = DEFAULT_CACHE_DIR / "embedding_manifest.json"
DEFAULT_CITATION_MAP = REPO_ROOT / "peper_writing" / "citation_key_map_20260429.csv"
DEFAULT_MODEL_NAME = rag_index.DEFAULT_MODEL_NAME
DEFAULT_DEVICE = "mps"


@dataclasses.dataclass(frozen=True)
class VectorIndex:
    records: tuple[dict[str, object], ...]
    embeddings: tuple[rag_index.EmbeddingRecord, ...]
    manifest: Mapping[str, object]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the local KB RAG vector index with deterministic guardrails."
    )
    parser.add_argument("query_terms", nargs="*", help="Query text. Alternative to --query.")
    parser.add_argument("--query", dest="query_text", help="Query text.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--vector-top-k", type=int, default=40)
    parser.add_argument("--deterministic-top-k", type=int, default=10)
    parser.add_argument("--theme")
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--citation-map", type=Path, default=DEFAULT_CITATION_MAP)
    parser.add_argument("--citation-key", action="append", default=[])
    parser.add_argument("--chunks-jsonl", type=Path, default=DEFAULT_CHUNKS_JSONL)
    parser.add_argument("--embeddings-sqlite", type=Path, default=DEFAULT_EMBEDDINGS_SQLITE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--include-quarantine", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args(argv)
    if args.query_text and args.query_terms:
        parser.error("provide query either as positional terms or --query, not both")
    query = args.query_text or " ".join(args.query_terms)
    if not query.strip():
        parser.error("query is required")
    args.query = query.strip()
    if args.top_k < 1 or args.vector_top_k < 1 or args.deterministic_top_k < 0:
        parser.error("top-k values must be positive; deterministic-top-k may be 0")
    return args


def load_vector_index(
    *,
    chunks_jsonl: Path,
    embeddings_sqlite: Path,
    manifest_path: Path,
    model_name: str,
) -> VectorIndex:
    records = rag_index.load_chunk_records(chunks_jsonl)
    if not records:
        raise SystemExit(f"Chunk JSONL is empty or missing: {chunks_jsonl}")
    manifest = rag_index.read_manifest(manifest_path)
    rag_index.validate_index_manifest(
        manifest,
        records,
        expected_model_name=model_name,
        embeddings_sqlite=embeddings_sqlite,
    )
    embeddings = rag_index.load_embedding_records(embeddings_sqlite)
    if len(embeddings) != len(records):
        raise SystemExit("Embedding SQLite row count does not match chunk JSONL row count.")
    return VectorIndex(records=tuple(records), embeddings=tuple(embeddings), manifest=manifest)


def encode_query(query: str, *, model_name: str, device: str):
    _torch, sentence_transformer_cls, _blocker = rag_index._load_embedding_backend(device, required=True)
    model = sentence_transformer_cls(model_name, device=device)
    embedding = model.encode(
        [rag_index.e5_query_text(query)],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return rag_index.normalize_rows(embedding)[0]


def record_passes_filters(
    record: Mapping[str, object],
    *,
    include_quarantine: bool,
    theme: str | None,
    year_min: int | None,
    year_max: int | None,
) -> bool:
    if not include_quarantine and (record.get("quarantine") or "quarantine/" in str(record.get("md_path", ""))):
        return False
    if str(record.get("status") or "").upper() != "ACTIVE":
        return False
    if theme and str(record.get("theme_dir") or record.get("theme") or "") != theme:
        return False
    year = record.get("year")
    year_int = int(year) if isinstance(year, int) or (isinstance(year, str) and year.isdigit()) else None
    if year_min is not None and (year_int is None or year_int < year_min):
        return False
    if year_max is not None and (year_int is None or year_int > year_max):
        return False
    return True


def _embedding_lookup(index: VectorIndex) -> dict[str, rag_index.EmbeddingRecord]:
    return {record.chunk_id: record for record in index.embeddings}


def _record_lookup(index: VectorIndex) -> dict[str, Mapping[str, object]]:
    return {str(record.get("chunk_id")): record for record in index.records}


def vector_candidates(
    query_vector: object,
    index: VectorIndex,
    *,
    top_k: int,
    include_quarantine: bool = False,
    theme: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[rerank.Candidate]:
    import numpy as np

    query = np.asarray(query_vector, dtype=np.float32)
    record_by_chunk = _record_lookup(index)
    scored: list[tuple[float, Mapping[str, object]]] = []
    for embedding in index.embeddings:
        record = record_by_chunk.get(embedding.chunk_id)
        if record is None:
            continue
        if not record_passes_filters(
            record,
            include_quarantine=include_quarantine,
            theme=theme,
            year_min=year_min,
            year_max=year_max,
        ):
            continue
        score = float(np.dot(query, np.asarray(embedding.vector, dtype=np.float32)))
        scored.append((round(score, 6), record))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("md_path")), str(item[1].get("source_page") or "")))
    return [
        rerank.candidate_from_record(record, source="embedding", score=score)
        for score, record in scored[:top_k]
    ]


def mapped_vector_candidates(
    query_vector: object,
    citation_keys: Sequence[str],
    citation_map: Mapping[str, rerank.CitationMapEntry],
    index: VectorIndex,
    *,
    top_per_path: int = 3,
    include_quarantine: bool = False,
) -> list[rerank.Candidate]:
    import numpy as np

    expected_paths = rerank._normalized_expected_paths(rerank._matching_entries(citation_keys, citation_map))
    if not expected_paths:
        return []
    query = np.asarray(query_vector, dtype=np.float32)
    embedding_by_chunk = _embedding_lookup(index)
    scored_by_path: dict[str, list[tuple[float, Mapping[str, object]]]] = {path: [] for path in expected_paths}
    for record in index.records:
        md_path = rerank.normalize_md_path(record.get("md_path"))
        if md_path not in expected_paths:
            continue
        if not record_passes_filters(record, include_quarantine=include_quarantine, theme=None, year_min=None, year_max=None):
            continue
        embedding = embedding_by_chunk.get(str(record.get("chunk_id")))
        if embedding is None:
            continue
        score = float(np.dot(query, np.asarray(embedding.vector, dtype=np.float32)))
        scored_by_path[md_path].append((round(score, 6), record))

    candidates: list[rerank.Candidate] = []
    for expected_path in sorted(scored_by_path):
        scored = scored_by_path[expected_path]
        scored.sort(key=lambda item: (-item[0], str(item[1].get("source_page") or ""), str(item[1].get("heading_path") or "")))
        for score, record in scored[:top_per_path]:
            candidate = rerank.candidate_from_record(record, source="embedding", score=score)
            candidates.append(dataclasses.replace(candidate, sources=("citation_map",), embedding_score=score))
    return candidates


def deterministic_candidates(
    query: str,
    chunks: Sequence[kb_search.Chunk],
    *,
    top_k: int,
    include_quarantine: bool,
    theme: str | None,
    year_min: int | None,
    year_max: int | None,
) -> list[rerank.Candidate]:
    if top_k == 0:
        return []
    args = argparse.Namespace(
        query=query,
        theme=theme,
        year_min=year_min,
        year_max=year_max,
        status="ACTIVE",
        top_k=top_k,
        include_quarantine=include_quarantine,
    )
    results = kb_search.search_fts(chunks, args) if kb_search.fts5_available() else kb_search.search_fallback(chunks, args)
    return [rerank.candidate_from_search_result(result) for result in results]


def hybrid_search(
    query: str,
    *,
    index: VectorIndex,
    query_vector: object,
    chunks: Sequence[kb_search.Chunk],
    citation_map: Mapping[str, rerank.CitationMapEntry] | None = None,
    citation_keys: Sequence[str] = (),
    top_k: int = 5,
    vector_top_k: int = 40,
    deterministic_top_k: int = 10,
    include_quarantine: bool = False,
    theme: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> list[rerank.RankedCandidate]:
    citation_map = citation_map or {}
    theme_value = theme or rerank.choose_theme(citation_keys, citation_map)
    vector = vector_candidates(
        query_vector,
        index,
        top_k=vector_top_k,
        include_quarantine=include_quarantine,
        theme=theme_value,
        year_min=year_min,
        year_max=year_max,
    )
    deterministic = deterministic_candidates(
        query,
        chunks,
        top_k=deterministic_top_k,
        include_quarantine=include_quarantine,
        theme=theme_value,
        year_min=year_min,
        year_max=year_max,
    )
    mapped = mapped_vector_candidates(
        query_vector,
        citation_keys,
        citation_map,
        index,
        include_quarantine=include_quarantine,
    )
    merged = rerank.merge_candidates([*deterministic, *mapped], vector, include_quarantine=include_quarantine)
    return rerank.rank_candidates(
        merged,
        citation_keys=citation_keys,
        citation_map=citation_map,
        include_quarantine=include_quarantine,
    )[:top_k]


def result_to_dict(rank: int, result: rerank.RankedCandidate) -> dict[str, object]:
    candidate = result.candidate
    return {
        "rank": rank,
        "hybrid_score": result.score,
        "vector_score": candidate.embedding_score,
        "deterministic_score": candidate.deterministic_score,
        "md_path": candidate.md_path,
        "source_page": candidate.source_page,
        "heading_path": candidate.heading_path,
        "title": candidate.title,
        "theme_dir": candidate.theme_dir,
        "year": candidate.year,
        "snippet": candidate.snippet,
        "score_components": dict(result.components),
        "sources": list(candidate.sources),
    }


def emit_text(results: Sequence[rerank.RankedCandidate]) -> None:
    for rank, result in enumerate(results, start=1):
        item = result_to_dict(rank, result)
        page = item["source_page"] if item["source_page"] is not None else "?"
        heading = item["heading_path"] or "(no heading)"
        print(f"{rank}. hybrid={item['hybrid_score']:.6f} vector={item['vector_score']:.6f} det={item['deterministic_score']:.6f}")
        print(f"   title: {item['title']}")
        print(f"   path: {item['md_path']}")
        print(f"   loc: source-page {page} > {heading}")
        print(f"   theme: {item['theme_dir']} | year: {item['year'] or '?'} | sources: {','.join(item['sources'])}")
        print(f"   snippet: {item['snippet']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    index = load_vector_index(
        chunks_jsonl=args.chunks_jsonl,
        embeddings_sqlite=args.embeddings_sqlite,
        manifest_path=args.manifest,
        model_name=args.model_name,
    )
    query_vector = encode_query(args.query, model_name=args.model_name, device=args.device)
    chunks = kb_search.load_chunks(args.markdown_root.resolve(), include_quarantine=args.include_quarantine)
    citation_map = rerank.load_citation_map(args.citation_map)
    results = hybrid_search(
        args.query,
        index=index,
        query_vector=query_vector,
        chunks=chunks,
        citation_map=citation_map,
        citation_keys=tuple(args.citation_key),
        top_k=args.top_k,
        vector_top_k=args.vector_top_k,
        deterministic_top_k=args.deterministic_top_k,
        include_quarantine=args.include_quarantine,
        theme=args.theme,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    if args.json:
        print(json.dumps([result_to_dict(rank, result) for rank, result in enumerate(results, start=1)], indent=2, ensure_ascii=False))
    elif args.jsonl:
        for rank, result in enumerate(results, start=1):
            print(json.dumps(result_to_dict(rank, result), ensure_ascii=False))
    else:
        emit_text(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
