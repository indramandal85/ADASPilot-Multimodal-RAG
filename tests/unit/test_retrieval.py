"""Dense, hybrid, reranking, parent, and HyDE retrieval tests."""

from __future__ import annotations

from adaspilot.domain.ingestion import ChunkKind, ChunkRecord, ContentModality
from adaspilot.retrieval.embeddings import HashingTextEmbedder
from adaspilot.retrieval.qdrant_store import QdrantChunkStore
from adaspilot.retrieval.text import (
    ConditionalHyDERetriever,
    DenseRetriever,
    HybridRetriever,
    LexicalReranker,
    ParentContextRetriever,
    RerankingRetriever,
    SparseRetriever,
    TemplateHyDEGenerator,
)


def _chunk(identifier: str, text: str, page: int, parent_id: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"child_{identifier:0<24}" if parent_id else f"parent_{identifier:0<24}",
        parent_id=parent_id,
        document_id="sample_doc",
        document_version="doc_123",
        corpus_version="corpus_123",
        page_number=page,
        kind=ChunkKind.CHILD if parent_id else ChunkKind.PARENT,
        modality=ContentModality.TEXT,
        text=text,
    )


def test_dense_and_advanced_retrieval_chain() -> None:
    parents = [
        _chunk("1", "Radar measures range and relative velocity for ADAS perception.", 1),
        _chunk("2", "Camera imagery provides texture and colour information.", 2),
    ]
    children = [
        _chunk("3", "Radar measures range and velocity.", 1, parents[0].chunk_id),
        _chunk("4", "Camera provides colour texture.", 2, parents[1].chunk_id),
    ]
    embedder = HashingTextEmbedder(64)
    store = QdrantChunkStore(
        collection_name="test_chunks",
        vector_size=64,
        in_memory=True,
    )
    store.rebuild(children, embedder.embed_documents([chunk.text for chunk in children]))
    dense = DenseRetriever(embedder, store)
    sparse = SparseRetriever(children)
    hybrid = HybridRetriever(dense, sparse, candidate_k=2)
    reranked = RerankingRetriever(hybrid, LexicalReranker(), candidate_k=2)
    parent = ParentContextRetriever(reranked, parents)
    hyde = ConditionalHyDERetriever(parent, TemplateHyDEGenerator(), score_threshold=10)

    assert dense.retrieve("radar range", 1)[0].page_number == 1
    assert hybrid.retrieve("radar velocity", 1)[0].source == "hybrid"
    assert reranked.retrieve("camera colour", 1)[0].page_number == 2
    assert parent.retrieve("radar", 1)[0].record_id == parents[0].chunk_id
    assert hyde.retrieve("radar range", 1)[0].source == "conditional_hyde"
    store.close()
