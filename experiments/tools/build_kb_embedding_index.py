#!/usr/bin/env python3
"""Build the derived KB chunk cache and MPS-only E5 embedding index."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:  # pragma: no cover - exercised by script execution, not package import.
    from . import kb_search
except ImportError:  # pragma: no cover
    import kb_search  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKDOWN_ROOT = REPO_ROOT / "original_papers" / "markdown"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "kb_rag"
DEFAULT_CHUNKS_JSONL = DEFAULT_CACHE_DIR / "chunks.jsonl"
DEFAULT_EMBEDDINGS_SQLITE = DEFAULT_CACHE_DIR / "embeddings.sqlite"
DEFAULT_MANIFEST = DEFAULT_CACHE_DIR / "embedding_manifest.json"
DEFAULT_MODEL_NAME = "intfloat/e5-large-v2"
DEFAULT_DEVICE = "mps"
EMBEDDING_SCHEMA_VERSION = 2


@dataclasses.dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    md_path: str
    source_page: int | None
    heading_path: str
    model_name: str
    dimension: int
    vector: object


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache active kb_search chunks and, when requested, generate an "
            "MPS-only E5 vector index for AI/agent RAG."
        )
    )
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--chunks-jsonl", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--embeddings-sqlite", type=Path, default=None)
    parser.add_argument("--include-quarantine", action="store_true", help="Include quarantined Markdown mirrors.")
    parser.add_argument(
        "--status",
        default="ACTIVE",
        help="Chunk status to include. Use ALL to disable. Default: ACTIVE.",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=("none", "auto", "generate"),
        default="none",
        help=(
            "none writes only chunks/manifest; auto generates only when dependencies "
            "exist and MPS is available; generate requires dependencies and MPS."
        ),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Local embedding device. Must be mps.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--json", action="store_true", help="Emit manifest JSON to stdout.")
    return parser.parse_args(argv)


def normalize_repo_path(path: str | Path) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        return str(path).replace("\\", "/").removeprefix("./")


def chunk_to_record(chunk: kb_search.Chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "md_path": normalize_repo_path(chunk.md_path),
        "theme_dir": chunk.theme,
        "theme": chunk.theme,
        "title": chunk.title,
        "short_title": chunk.short_title,
        "status": chunk.status,
        "venue": chunk.venue,
        "year": chunk.year,
        "source_pdf_relative_path": chunk.source_pdf_relative_path,
        "source_page": chunk.source_page,
        "heading_path": chunk.heading_path,
        "chunk_text": chunk.text,
        "text": chunk.text,
        "chunk_word_count": chunk.chunk_word_count,
        "quarantine": chunk.quarantine,
        "metadata_text": chunk.metadata_text,
    }


def _is_active_record(record: Mapping[str, object], *, status: str, include_quarantine: bool) -> bool:
    if not include_quarantine and (record.get("quarantine") or "quarantine/" in str(record.get("md_path", ""))):
        return False
    if status.upper() != "ALL" and str(record.get("status", "")).upper() != status.upper():
        return False
    return True


def build_chunk_records(
    markdown_root: Path = DEFAULT_MARKDOWN_ROOT,
    *,
    include_quarantine: bool = False,
    status: str = "ACTIVE",
) -> list[dict[str, object]]:
    chunks = kb_search.load_chunks(markdown_root, include_quarantine=include_quarantine)
    records = [chunk_to_record(chunk) for chunk in chunks]
    return [
        record
        for record in records
        if _is_active_record(record, status=status, include_quarantine=include_quarantine)
    ]


def corpus_hash(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def load_chunk_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def e5_passage_text(record: Mapping[str, object]) -> str:
    parts = [
        str(record.get("title") or record.get("short_title") or ""),
        str(record.get("heading_path") or ""),
        str(record.get("metadata_text") or ""),
        str(record.get("chunk_text") or record.get("text") or ""),
    ]
    body = " ".join(part.strip() for part in parts if part and part.strip())
    return f"passage: {body}"


def e5_query_text(query: str) -> str:
    return f"query: {query.strip()}"


def _dependency_blocker_message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def ensure_mps_available(torch_module: object) -> None:
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        raise SystemExit("MPS is unavailable for local embedding generation; refusing CPU fallback.")


def _load_embedding_backend(device: str, *, required: bool):
    if device != "mps":
        raise SystemExit("Local embedding generation is governed to --device mps; CPU fallback is not allowed.")
    try:
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency dependent.
        if required:
            raise SystemExit(f"Embedding dependencies unavailable: {_dependency_blocker_message(exc)}") from exc
        return None, None, _dependency_blocker_message(exc)

    ensure_mps_available(torch)
    return torch, SentenceTransformer, None


def normalize_rows(embeddings: object):
    import numpy as np

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D embedding matrix, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def vector_to_blob(vector: object) -> bytes:
    import numpy as np

    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes, dimension: int):
    import numpy as np

    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.size != dimension:
        raise ValueError(f"Embedding blob has dimension {vector.size}, expected {dimension}")
    return vector


def encode_passages(sentence_transformer_cls: object, records: Sequence[Mapping[str, object]], *, model_name: str, device: str, batch_size: int):
    model = sentence_transformer_cls(model_name, device=device)
    texts = [e5_passage_text(record) for record in records]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return normalize_rows(embeddings)


def write_embedding_rows(
    records: Sequence[Mapping[str, object]],
    embeddings: object,
    *,
    sqlite_path: Path,
    model_name: str,
) -> dict[str, object]:
    matrix = normalize_rows(embeddings)
    if len(records) != matrix.shape[0]:
        raise ValueError(f"Embedding row count {matrix.shape[0]} does not match chunk count {len(records)}")
    dimension = int(matrix.shape[1]) if matrix.ndim == 2 else 0

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.execute("DROP TABLE IF EXISTS embeddings")
    conn.execute("DROP TABLE IF EXISTS metadata")
    conn.execute(
        """
        CREATE TABLE embeddings (
            chunk_id TEXT PRIMARY KEY,
            md_path TEXT NOT NULL,
            source_page INTEGER,
            heading_path TEXT NOT NULL,
            model_name TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_embeddings_path ON embeddings(md_path)")
    conn.execute(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("schema_version", str(EMBEDDING_SCHEMA_VERSION)))
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("model_name", model_name))
    conn.execute("INSERT INTO metadata VALUES (?, ?)", ("dimension", str(dimension)))
    rows = [
        (
            str(record.get("chunk_id")),
            str(record.get("md_path")),
            record.get("source_page"),
            str(record.get("heading_path") or ""),
            model_name,
            dimension,
            vector_to_blob(vector),
        )
        for record, vector in zip(records, matrix, strict=True)
    ]
    conn.executemany("INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return {
        "embedding_status": "generated",
        "embedding_blocker": None,
        "model_name": model_name,
        "device": DEFAULT_DEVICE,
        "dimension": dimension,
        "embedding_rows": len(rows),
        "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
        "embedding_storage": "sqlite_float32_blob",
    }


def write_embeddings_sqlite(
    records: Sequence[Mapping[str, object]],
    *,
    sqlite_path: Path,
    model_name: str,
    device: str,
    batch_size: int,
    required: bool,
) -> dict[str, object]:
    _torch, sentence_transformer_cls, blocker = _load_embedding_backend(device, required=required)
    if sentence_transformer_cls is None:
        return {
            "embedding_status": "stubbed_dependency_unavailable",
            "embedding_blocker": blocker,
            "model_name": model_name,
            "device": device,
            "dimension": None,
            "embedding_rows": 0,
            "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
            "embedding_storage": "sqlite_float32_blob",
        }

    embeddings = encode_passages(
        sentence_transformer_cls,
        records,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    info = write_embedding_rows(records, embeddings, sqlite_path=sqlite_path, model_name=model_name)
    return {**info, "device": device}


def load_embedding_records(sqlite_path: Path) -> list[EmbeddingRecord]:
    if not sqlite_path.exists():
        return []
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT chunk_id, md_path, source_page, heading_path, model_name, dimension, embedding
        FROM embeddings
        ORDER BY chunk_id
        """
    ).fetchall()
    conn.close()
    return [
        EmbeddingRecord(
            chunk_id=str(row["chunk_id"]),
            md_path=str(row["md_path"]),
            source_page=int(row["source_page"]) if row["source_page"] is not None else None,
            heading_path=str(row["heading_path"] or ""),
            model_name=str(row["model_name"]),
            dimension=int(row["dimension"]),
            vector=blob_to_vector(row["embedding"], int(row["dimension"])),
        )
        for row in rows
    ]


def read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Embedding manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_index_manifest(
    manifest: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    expected_model_name: str,
    embeddings_sqlite: Path,
) -> None:
    if manifest.get("embedding_status") != "generated":
        raise SystemExit(f"Vector index is not generated: embedding_status={manifest.get('embedding_status')}")
    if manifest.get("model_name") != expected_model_name:
        raise SystemExit(
            f"Vector index model mismatch: manifest={manifest.get('model_name')} expected={expected_model_name}"
        )
    current_hash = corpus_hash(records)
    if manifest.get("corpus_hash") != current_hash:
        raise SystemExit("Vector index is stale: chunk corpus hash does not match embedding manifest.")
    if int(manifest.get("chunk_count") or -1) != len(records):
        raise SystemExit("Vector index is stale: chunk count does not match embedding manifest.")
    embedding_rows = int(manifest.get("embedding_rows") or 0)
    if embedding_rows != len(records):
        raise SystemExit("Vector index row count does not match chunk count.")
    if not embeddings_sqlite.exists():
        raise SystemExit(f"Embedding SQLite file does not exist: {embeddings_sqlite}")


def build_manifest(
    *,
    markdown_root: Path,
    chunks_jsonl: Path,
    embeddings_sqlite: Path,
    records: Sequence[Mapping[str, object]],
    include_quarantine: bool,
    status: str,
    embedding_info: Mapping[str, object],
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": REPO_ROOT.as_posix(),
        "markdown_root": normalize_repo_path(markdown_root),
        "chunks_jsonl": normalize_repo_path(chunks_jsonl),
        "embeddings_sqlite": normalize_repo_path(embeddings_sqlite),
        "chunk_count": len(records),
        "active_only": status.upper() == "ACTIVE" and not include_quarantine,
        "include_quarantine": include_quarantine,
        "status_filter": status,
        "corpus_hash": corpus_hash(records),
        **dict(embedding_info),
    }


def write_manifest(manifest: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cache_dir = args.cache_dir
    chunks_jsonl = args.chunks_jsonl or cache_dir / "chunks.jsonl"
    manifest_path = args.manifest or cache_dir / "embedding_manifest.json"
    embeddings_sqlite = args.embeddings_sqlite or cache_dir / "embeddings.sqlite"

    markdown_root = args.markdown_root.resolve()
    if not markdown_root.exists():
        raise SystemExit(f"Markdown root does not exist: {markdown_root}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    records = build_chunk_records(
        markdown_root,
        include_quarantine=args.include_quarantine,
        status=args.status,
    )
    write_jsonl(records, chunks_jsonl)

    embedding_info: dict[str, object]
    if args.embedding_mode == "none":
        embedding_info = {
            "embedding_status": "not_requested",
            "embedding_blocker": None,
            "model_name": args.model_name,
            "device": args.device,
            "dimension": None,
            "embedding_rows": 0,
            "embedding_schema_version": EMBEDDING_SCHEMA_VERSION,
            "embedding_storage": "sqlite_float32_blob",
        }
    else:
        embedding_info = write_embeddings_sqlite(
            records,
            sqlite_path=embeddings_sqlite,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            required=args.embedding_mode == "generate",
        )

    manifest = build_manifest(
        markdown_root=markdown_root,
        chunks_jsonl=chunks_jsonl,
        embeddings_sqlite=embeddings_sqlite,
        records=records,
        include_quarantine=args.include_quarantine,
        status=args.status,
        embedding_info=embedding_info,
    )
    write_manifest(manifest, manifest_path)

    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "KB RAG cache written: "
            f"{len(records)} chunks -> {normalize_repo_path(chunks_jsonl)}; "
            f"manifest -> {normalize_repo_path(manifest_path)}; "
            f"embedding_status={manifest['embedding_status']}; "
            f"embedding_rows={manifest['embedding_rows']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
