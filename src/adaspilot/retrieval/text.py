"""Dense, sparse, hybrid, reranked, parent-aware, and conditional-HyDE retrieval."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from adaspilot.domain.ingestion import ChunkRecord
from adaspilot.domain.retrieval import SearchHit
from adaspilot.retrieval.embeddings import TextEmbedder
from adaspilot.retrieval.qdrant_store import QdrantChunkStore


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[SearchHit]: ...


class DenseRetriever:
    def __init__(self, embedder: TextEmbedder, store: QdrantChunkStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        return self.store.search(self.embedder.embed_query(query), top_k)


class SparseRetriever:
    def __init__(self, chunks: Sequence[ChunkRecord]) -> None:
        self.chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
        hits: list[SearchHit] = []
        for index in ranked[:top_k]:
            chunk = self.chunks[index]
            hits.append(
                SearchHit(
                    record_id=chunk.chunk_id,
                    score=float(scores[index]),
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    parent_id=chunk.parent_id,
                    asset_id=chunk.asset_id,
                    source="sparse",
                )
            )
        return hits


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        *,
        candidate_k: int = 30,
        rrf_constant: int = 60,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        rankings = [
            self.dense.retrieve(query, self.candidate_k),
            self.sparse.retrieve(query, self.candidate_k),
        ]
        combined: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, hit in enumerate(ranking, start=1):
                combined.setdefault(hit.record_id, hit)
                scores[hit.record_id] = scores.get(hit.record_id, 0.0) + 1.0 / (
                    self.rrf_constant + rank
                )
        ordered = sorted(scores, key=lambda record_id: scores[record_id], reverse=True)
        return [
            combined[record_id].model_copy(update={"score": scores[record_id], "source": "hybrid"})
            for record_id in ordered[:top_k]
        ]


class Reranker(Protocol):
    def scores(self, query: str, passages: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(score) for score in self._model.rerank(query, list(passages))]


class LexicalReranker:
    def scores(self, query: str, passages: Sequence[str]) -> list[float]:
        query_tokens = set(tokenize(query))
        return [
            len(query_tokens & set(tokenize(passage))) / max(1, len(query_tokens))
            for passage in passages
        ]


class RerankingRetriever:
    def __init__(self, base: Retriever, reranker: Reranker, candidate_k: int = 30) -> None:
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        candidates = self.base.retrieve(query, self.candidate_k)
        scores = self.reranker.scores(query, [hit.text for hit in candidates])
        reranked = [
            hit.model_copy(update={"score": score, "source": "reranked"})
            for hit, score in zip(candidates, scores, strict=True)
        ]
        return sorted(reranked, key=lambda hit: hit.score, reverse=True)[:top_k]


class ParentContextRetriever:
    def __init__(self, base: Retriever, parents: Sequence[ChunkRecord]) -> None:
        self.base = base
        self.parents = {parent.chunk_id: parent for parent in parents}

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        child_hits = self.base.retrieve(query, max(top_k * 2, top_k))
        parent_hits: list[SearchHit] = []
        seen: set[str] = set()
        for hit in child_hits:
            parent = self.parents.get(hit.parent_id or "")
            record_id = parent.chunk_id if parent else hit.record_id
            if record_id in seen:
                continue
            seen.add(record_id)
            parent_hits.append(
                hit.model_copy(
                    update={
                        "record_id": record_id,
                        "text": parent.text if parent else hit.text,
                        "source": "parent",
                    }
                )
            )
            if len(parent_hits) == top_k:
                break
        return parent_hits


class HyDEGenerator(Protocol):
    def generate(self, query: str) -> str: ...


class TemplateHyDEGenerator:
    """Offline fallback that makes the expansion visible and deterministic."""

    def generate(self, query: str) -> str:
        return (
            "A technical ADAS document answering the question would explain the system, "
            f"its evidence, measurements, limitations, and evaluation results. Question: {query}"
        )


class ConditionalHyDERetriever:
    def __init__(self, base: Retriever, generator: HyDEGenerator, score_threshold: float) -> None:
        self.base = base
        self.generator = generator
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        initial = self.base.retrieve(query, top_k)
        needs_hyde = not initial or initial[0].score < self.score_threshold
        if not needs_hyde:
            return initial
        hypothetical = self.generator.generate(query)
        expanded = self.base.retrieve(f"{query}\n{hypothetical}", top_k)
        return [hit.model_copy(update={"source": "conditional_hyde"}) for hit in expanded]
