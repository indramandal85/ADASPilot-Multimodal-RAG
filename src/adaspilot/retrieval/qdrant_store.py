"""Local Qdrant storage for text chunks."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient, models

from adaspilot.domain.ingestion import ChunkRecord
from adaspilot.domain.retrieval import SearchHit


def _point_id(record_id: str) -> str:
    """Map any stable record ID to a stable Qdrant-compatible UUID."""

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"adaspilot:{record_id}"))


class QdrantChunkStore:
    """One local collection per corpus version; no server or Docker required."""

    def __init__(
        self,
        *,
        collection_name: str,
        vector_size: int,
        path: Path | None = None,
        in_memory: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = (
            QdrantClient(":memory:")
            if in_memory
            else QdrantClient(path=str(path) if path is not None else None)
        )

    def rebuild(self, chunks: Sequence[ChunkRecord], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size, distance=models.Distance.COSINE
            ),
        )
        points = [
            models.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=list(vector),
                payload={
                    "record_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id,
                    "document_id": chunk.document_id,
                    "document_version": chunk.document_version,
                    "corpus_version": chunk.corpus_version,
                    "page_number": chunk.page_number,
                    "modality": chunk.modality.value,
                    "text": chunk.text,
                    "asset_id": chunk.asset_id,
                    "table_id": chunk.table_id,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), 128):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + 128],
                wait=True,
            )

    def search(self, vector: Sequence[float], limit: int) -> list[SearchHit]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            limit=limit,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    record_id=str(payload["record_id"]),
                    score=float(point.score),
                    document_id=str(payload["document_id"]),
                    page_number=int(payload["page_number"]),
                    text=str(payload.get("text", "")),
                    parent_id=(str(payload["parent_id"]) if payload.get("parent_id") else None),
                    asset_id=(str(payload["asset_id"]) if payload.get("asset_id") else None),
                    source="dense",
                )
            )
        return hits

    def close(self) -> None:
        self.client.close()
