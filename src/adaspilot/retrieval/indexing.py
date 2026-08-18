"""Build and load the local dense text index."""

from __future__ import annotations

import json
from pathlib import Path

from adaspilot.config import Settings
from adaspilot.domain.ingestion import ChunkRecord
from adaspilot.ingestion.io import load_chunks, resolve_current_build
from adaspilot.retrieval.embeddings import FastEmbedTextEmbedder, TextEmbedder
from adaspilot.retrieval.qdrant_store import QdrantChunkStore
from adaspilot.retrieval.text import DenseRetriever


def collection_name(settings: Settings, corpus_version: str) -> str:
    return f"{settings.text_collection_prefix}_{corpus_version}"


def build_text_index(
    settings: Settings,
    *,
    embedder: TextEmbedder | None = None,
    in_memory: bool = False,
) -> tuple[QdrantChunkStore, list[ChunkRecord]]:
    build_dir, build = resolve_current_build(settings.processed_data_dir)
    chunks = load_chunks(build_dir, "child")
    active_embedder = embedder or FastEmbedTextEmbedder(
        settings.embedding_model,
        settings.embedding_dimension,
    )
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), 128):
        vectors.extend(
            active_embedder.embed_documents([chunk.text for chunk in chunks[start : start + 128]])
        )
    store = QdrantChunkStore(
        collection_name=collection_name(settings, build.corpus_version),
        vector_size=active_embedder.dimension,
        path=None if in_memory else settings.qdrant_path,
        in_memory=in_memory,
    )
    store.rebuild(chunks, vectors)
    if not in_memory:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "corpus_version": build.corpus_version,
            "collection_name": store.collection_name,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": active_embedder.dimension,
            "chunk_count": len(chunks),
        }
        (settings.index_dir / "text_index.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
    return store, chunks


def load_dense_retriever(
    settings: Settings,
    *,
    embedder: TextEmbedder | None = None,
) -> tuple[DenseRetriever, QdrantChunkStore, Path]:
    build_dir, build = resolve_current_build(settings.processed_data_dir)
    active_embedder = embedder or FastEmbedTextEmbedder(
        settings.embedding_model,
        settings.embedding_dimension,
    )
    store = QdrantChunkStore(
        collection_name=collection_name(settings, build.corpus_version),
        vector_size=active_embedder.dimension,
        path=settings.qdrant_path,
    )
    if not store.client.collection_exists(store.collection_name):
        store.close()
        raise FileNotFoundError("text index is missing; run `make build-index`")
    return DenseRetriever(active_embedder, store), store, build_dir
