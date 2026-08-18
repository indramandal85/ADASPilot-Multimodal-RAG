"""Ask the dense, advanced, or fused multimodal retriever."""

from __future__ import annotations

import argparse

from adaspilot.config import get_settings
from adaspilot.ingestion.io import resolve_current_build
from adaspilot.retrieval.answers import OpenAICompatibleAnswerer, answer_to_json
from adaspilot.retrieval.indexing import load_dense_retriever
from adaspilot.retrieval.runtime import make_advanced_retriever, make_answerer, make_visual_embedder
from adaspilot.retrieval.text import Retriever
from adaspilot.retrieval.visual import FusedMultimodalRetriever, load_visual_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask ADASPilot over the ingested corpus")
    parser.add_argument("query")
    parser.add_argument(
        "--mode",
        choices=["dense", "advanced", "multimodal"],
        default="dense",
    )
    arguments = parser.parse_args()
    settings = get_settings()
    dense, store, build_dir = load_dense_retriever(settings)
    try:
        retriever: Retriever = dense
        if arguments.mode in {"advanced", "multimodal"}:
            retriever = make_advanced_retriever(settings, dense, build_dir)
        if arguments.mode == "multimodal":
            _, build = resolve_current_build(settings.processed_data_dir)
            visual = load_visual_index(
                settings.index_dir / f"visual_{build.corpus_version}",
                make_visual_embedder(settings),
            )
            retriever = FusedMultimodalRetriever(
                retriever,
                visual,
                rrf_constant=settings.rrf_constant,
            )
        hits = retriever.retrieve(arguments.query, settings.dense_top_k)
        answerer = make_answerer(settings)
        if arguments.mode == "multimodal" and isinstance(answerer, OpenAICompatibleAnswerer):
            answer = answerer.answer_visual(arguments.query, hits, build_dir)
        else:
            answer = answerer.answer(arguments.query, hits)
        print(answer_to_json(answer))
    finally:
        store.close()


if __name__ == "__main__":
    main()
