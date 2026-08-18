"""Controlled E0-E4 retrieval experiments."""

from __future__ import annotations

from collections.abc import Sequence

from adaspilot.domain.evaluation import EvaluationExample
from adaspilot.domain.ingestion import ChunkRecord
from adaspilot.domain.retrieval import ExperimentResult
from adaspilot.evaluation.retrieval import evaluate_retriever
from adaspilot.retrieval.text import (
    ConditionalHyDERetriever,
    HybridRetriever,
    HyDEGenerator,
    ParentContextRetriever,
    Reranker,
    RerankingRetriever,
    Retriever,
    SparseRetriever,
)


def build_experiments(
    *,
    dense: Retriever,
    child_chunks: Sequence[ChunkRecord],
    parent_chunks: Sequence[ChunkRecord],
    reranker: Reranker,
    hyde_generator: HyDEGenerator,
    candidate_k: int,
    rrf_constant: int,
    hyde_score_threshold: float,
) -> list[tuple[str, str, Retriever]]:
    sparse = SparseRetriever(child_chunks)
    hybrid = HybridRetriever(
        dense,
        sparse,
        candidate_k=candidate_k,
        rrf_constant=rrf_constant,
    )
    reranked = RerankingRetriever(hybrid, reranker, candidate_k=candidate_k)
    parent = ParentContextRetriever(reranked, parent_chunks)
    hyde = ConditionalHyDERetriever(parent, hyde_generator, hyde_score_threshold)
    return [
        ("E0", "Dense retrieval", dense),
        ("E1", "Dense + BM25 with reciprocal-rank fusion", hybrid),
        ("E2", "Hybrid retrieval + cross-encoder reranking", reranked),
        ("E3", "E2 + parent context", parent),
        ("E4", "E3 + conditional HyDE", hyde),
    ]


def run_experiments(
    experiments: Sequence[tuple[str, str, Retriever]],
    examples: Sequence[EvaluationExample],
    *,
    minimum_gain: float,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    previous_recall = 0.0
    for index, (experiment_id, description, retriever) in enumerate(experiments):
        metrics = evaluate_retriever(retriever, examples, top_k=10)
        delta = metrics.recall_at_5 - previous_recall if index else 0.0
        retained = index == 0 or delta >= minimum_gain
        results.append(
            ExperimentResult(
                experiment_id=experiment_id,  # type: ignore[arg-type]
                description=description,
                metrics=metrics,
                delta_recall_at_5=delta,
                retained=retained,
            )
        )
        previous_recall = metrics.recall_at_5
    return results
