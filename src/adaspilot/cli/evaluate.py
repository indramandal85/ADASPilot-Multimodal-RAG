"""Evaluate dense answers or the visual index on a frozen split."""

from __future__ import annotations

import argparse
import json

from adaspilot.config import get_settings
from adaspilot.domain.evaluation import BenchmarkSplit
from adaspilot.evaluation.io import load_evaluation_examples
from adaspilot.evaluation.retrieval import evaluate_answers, evaluate_retriever
from adaspilot.evaluation.visual import evaluate_visual_retrieval
from adaspilot.ingestion.io import resolve_current_build
from adaspilot.retrieval.indexing import load_dense_retriever
from adaspilot.retrieval.runtime import make_answerer, make_visual_embedder
from adaspilot.retrieval.visual import load_visual_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ADASPilot retrieval")
    parser.add_argument("--mode", choices=["baseline", "visual"], default="baseline")
    parser.add_argument("--split", choices=["development", "test"], default="development")
    arguments = parser.parse_args()
    settings = get_settings()
    split = BenchmarkSplit(arguments.split)
    examples = [
        example
        for example in load_evaluation_examples(settings.benchmark_path)
        if example.split == split
    ]
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    if arguments.mode == "visual":
        _, build = resolve_current_build(settings.processed_data_dir)
        index = load_visual_index(
            settings.index_dir / f"visual_{build.corpus_version}",
            make_visual_embedder(settings),
        )
        payload = evaluate_visual_retrieval(index, examples).model_dump(mode="json")
    else:
        dense, store, _ = load_dense_retriever(settings)
        try:
            payload = {
                "retrieval": evaluate_retriever(dense, examples).model_dump(mode="json"),
                "answers": evaluate_answers(
                    dense,
                    make_answerer(settings),
                    examples,
                ).model_dump(mode="json"),
            }
        finally:
            store.close()
    output = settings.results_dir / f"{arguments.mode}_{arguments.split}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
