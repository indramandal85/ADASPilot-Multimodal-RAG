"""Small composition tests for configured retrieval components."""

from __future__ import annotations

from pathlib import Path

from adaspilot.config import Settings
from adaspilot.domain.ingestion import ChunkKind, ChunkRecord, ContentModality
from adaspilot.domain.retrieval import SearchHit
from adaspilot.retrieval.answers import ExtractiveAnswerer, OpenAICompatibleAnswerer
from adaspilot.retrieval.runtime import (
    make_advanced_retriever,
    make_answerer,
    make_hyde_generator,
    make_visual_embedder,
)
from adaspilot.retrieval.text import (
    ConditionalHyDERetriever,
    LexicalReranker,
    ParentContextRetriever,
    TemplateHyDEGenerator,
)


class EmptyDenseRetriever:
    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        return []


class FakeVisualEmbedder:
    def __init__(self, *args: object) -> None:
        self.args = args


def _chunks(kind: str) -> list[ChunkRecord]:
    parent_id = None if kind == "parent" else "parent_1"
    return [
        ChunkRecord(
            chunk_id=f"{kind}_1",
            parent_id=parent_id,
            document_id="sample_doc",
            document_version="doc_1",
            corpus_version="corpus_1",
            page_number=1,
            kind=ChunkKind.PARENT if kind == "parent" else ChunkKind.CHILD,
            modality=ContentModality.TEXT,
            text="Radar measures range and velocity.",
        )
    ]


def test_answerer_and_hyde_configuration() -> None:
    default_settings = Settings()
    assert isinstance(make_answerer(default_settings), ExtractiveAnswerer)
    assert isinstance(make_hyde_generator(default_settings), TemplateHyDEGenerator)

    llm_settings = Settings(
        llm_base_url="http://localhost:1234/v1",
        llm_model="local-model",
        llm_api_key="secret",
    )
    assert isinstance(make_answerer(llm_settings), OpenAICompatibleAnswerer)
    assert isinstance(make_hyde_generator(llm_settings), OpenAICompatibleAnswerer)


def test_advanced_and_visual_composition(monkeypatch: object) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.retrieval.runtime.load_chunks", lambda _path, kind: _chunks(kind)
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.retrieval.runtime.FastEmbedReranker", lambda _model: LexicalReranker()
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.retrieval.runtime.FastEmbedClipVisualEmbedder", FakeVisualEmbedder
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.retrieval.runtime.ColPaliVisualEmbedder", FakeVisualEmbedder
    )

    retained_retriever = make_advanced_retriever(
        Settings(retrieval_candidate_k=2), EmptyDenseRetriever(), Path(".")
    )
    experimental_hyde = make_advanced_retriever(
        Settings(retrieval_candidate_k=2, enable_hyde=True),
        EmptyDenseRetriever(),
        Path("."),
    )
    assert isinstance(retained_retriever, ParentContextRetriever)
    assert isinstance(experimental_hyde, ConditionalHyDERetriever)

    clip = make_visual_embedder(Settings(visual_backend="clip"))
    colpali = make_visual_embedder(Settings(visual_backend="colpali"))
    assert len(clip.args) == 2  # type: ignore[attr-defined]
    assert len(colpali.args) == 1  # type: ignore[attr-defined]
