"""Run the E0-E4 controlled text-retrieval experiments."""

from __future__ import annotations

import json

from adaspilot.config import get_settings
from adaspilot.domain.evaluation import BenchmarkSplit
from adaspilot.evaluation.io import load_evaluation_examples
from adaspilot.ingestion.io import load_chunks
from adaspilot.retrieval.experiments import build_experiments, run_experiments
from adaspilot.retrieval.indexing import load_dense_retriever
from adaspilot.retrieval.runtime import make_hyde_generator
from adaspilot.retrieval.text import FastEmbedReranker


def main() -> None:
    settings = get_settings()
    dense, store, build_dir = load_dense_retriever(settings)
    try:
        examples = [
            example
            for example in load_evaluation_examples(settings.benchmark_path)
            if example.split == BenchmarkSplit.DEVELOPMENT
        ]
        experiments = build_experiments(
            dense=dense,
            child_chunks=load_chunks(build_dir, "child"),
            parent_chunks=load_chunks(build_dir, "parent"),
            reranker=FastEmbedReranker(settings.reranker_model),
            hyde_generator=make_hyde_generator(settings),  # type: ignore[arg-type]
            candidate_k=settings.retrieval_candidate_k,
            rrf_constant=settings.rrf_constant,
            hyde_score_threshold=settings.hyde_score_threshold,
        )
        results = run_experiments(
            experiments,
            examples,
            minimum_gain=settings.experiment_minimum_gain,
        )
    finally:
        store.close()
    payload = [result.model_dump(mode="json") for result in results]
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "text_experiments.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
