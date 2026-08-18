"""Composition test for the real Modules 6-8 workflow factory."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from adaspilot.config import Settings
from adaspilot.orchestration.direct import DirectCorrectiveWorkflow
from adaspilot.orchestration.planning import PlanExecuteWorkflow
from adaspilot.orchestration.react import ReactResearchAgent
from adaspilot.orchestration.runtime import build_workflows
from adaspilot.retrieval.answers import ExtractiveAnswerer


class EmptyRetriever:
    def retrieve(self, query: str, top_k: int) -> list[object]:
        return []


class FakeStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_build_and_close_workflow_bundle(monkeypatch: object, tmp_path: Path) -> None:
    store = FakeStore()
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.orchestration.runtime.load_dense_retriever",
        lambda _settings: (EmptyRetriever(), store, tmp_path),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.orchestration.runtime.make_advanced_retriever",
        lambda _settings, dense, _build: dense,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.orchestration.runtime.make_answerer",
        lambda _settings, _usage: ExtractiveAnswerer(),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.orchestration.runtime.resolve_current_build",
        lambda _path: (tmp_path, SimpleNamespace(corpus_version="corpus_test")),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.orchestration.runtime.load_corpus_manifest",
        lambda _path: SimpleNamespace(documents=[]),
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.orchestration.runtime.load_pages", lambda _path: []
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.orchestration.runtime.load_documents", lambda _path: []
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.orchestration.runtime.load_chunks", lambda _path, _kind: []
    )
    settings = Settings(
        processed_data_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        corpus_manifest_path=tmp_path / "manifest.json",
    )

    bundle = build_workflows(settings, allowed_document_ids={"sample_doc"})

    assert isinstance(bundle.direct, DirectCorrectiveWorkflow)
    assert isinstance(bundle.react, ReactResearchAgent)
    assert isinstance(bundle.plan, PlanExecuteWorkflow)
    bundle.close()
    assert store.closed
