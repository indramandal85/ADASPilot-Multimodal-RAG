"""Text index construction and E0-E4 experiment tests."""

from __future__ import annotations

import json
from pathlib import Path

from adaspilot.config import Settings
from adaspilot.domain.evaluation import EvaluationExample
from adaspilot.domain.ingestion import (
    ChunkKind,
    ChunkRecord,
    ContentModality,
    CorpusBuild,
)
from adaspilot.ingestion.io import write_json, write_jsonl
from adaspilot.retrieval.embeddings import HashingTextEmbedder
from adaspilot.retrieval.experiments import build_experiments, run_experiments
from adaspilot.retrieval.indexing import build_text_index, load_dense_retriever
from adaspilot.retrieval.text import LexicalReranker, TemplateHyDEGenerator


def _chunk(identifier: str, text: str, page: int, parent_id: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"chunk_{identifier:0<26}",
        parent_id=parent_id,
        document_id="sample_doc",
        document_version="doc_123",
        corpus_version="corpus_test",
        page_number=page,
        kind=ChunkKind.CHILD if parent_id else ChunkKind.PARENT,
        modality=ContentModality.TEXT,
        text=text,
    )


def _settings(tmp_path: Path) -> tuple[Settings, list[ChunkRecord], list[ChunkRecord]]:
    processed = tmp_path / "processed"
    build_dir = processed / "corpus_test"
    parents = [
        _chunk("parent1", "Radar measures range and relative velocity.", 1),
        _chunk("parent2", "A camera identifies lanes and traffic signs.", 2),
    ]
    children = [
        _chunk("child1", "Radar measures range and velocity.", 1, parents[0].chunk_id),
        _chunk("child2", "Camera identifies traffic signs.", 2, parents[1].chunk_id),
    ]
    build = CorpusBuild(
        corpus_id="test_corpus",
        corpus_version="corpus_test",
        source_manifest_sha256="a" * 64,
        document_count=1,
        unique_document_count=1,
        page_count=2,
        table_count=0,
        visual_asset_count=0,
        parent_chunk_count=2,
        child_chunk_count=2,
        ingestion_config={},
    )
    write_json(build_dir / "corpus.json", build)
    write_jsonl(build_dir / "parent_chunks.jsonl", parents)
    write_jsonl(build_dir / "child_chunks.jsonl", children)
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "current.json").write_text(
        json.dumps({"corpus_version": "corpus_test"}), encoding="utf-8"
    )
    settings = Settings(
        processed_data_dir=processed,
        index_dir=tmp_path / "index",
        qdrant_path=tmp_path / "index" / "qdrant",
        embedding_dimension=64,
        retrieval_candidate_k=2,
    )
    return settings, children, parents


def _example() -> EvaluationExample:
    return EvaluationExample.model_validate(
        {
            "question_id": "q_radar_test",
            "question": "What does radar measure?",
            "question_type": "factual",
            "should_answer": True,
            "expected_response_status": "answered",
            "expected_execution_mode": "direct",
            "reference_answer": "Radar measures range and velocity.",
            "evidence": [
                {
                    "document_id": "sample_doc",
                    "page_number": 1,
                    "supporting_excerpt": "Radar measures range and velocity.",
                }
            ],
        }
    )


def test_build_load_index_and_run_all_experiments(tmp_path: Path) -> None:
    settings, children, parents = _settings(tmp_path)
    embedder = HashingTextEmbedder(64)
    store, indexed = build_text_index(settings, embedder=embedder)
    assert indexed == children
    assert json.loads((settings.index_dir / "text_index.json").read_text())["chunk_count"] == 2
    store.close()

    dense, loaded_store, build_dir = load_dense_retriever(settings, embedder=embedder)
    experiments = build_experiments(
        dense=dense,
        child_chunks=children,
        parent_chunks=parents,
        reranker=LexicalReranker(),
        hyde_generator=TemplateHyDEGenerator(),
        candidate_k=2,
        rrf_constant=60,
        hyde_score_threshold=10,
    )
    results = run_experiments(experiments, [_example()], minimum_gain=0)

    assert build_dir.name == "corpus_test"
    assert [result.experiment_id for result in results] == ["E0", "E1", "E2", "E3", "E4"]
    assert results[0].retained
    assert all(result.metrics.query_count == 1 for result in results)
    loaded_store.close()


def test_load_index_reports_missing_collection(tmp_path: Path) -> None:
    settings, _, _ = _settings(tmp_path)
    try:
        load_dense_retriever(settings, embedder=HashingTextEmbedder(64))
    except FileNotFoundError as exc:
        assert "make build-index" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("missing index was not reported")
