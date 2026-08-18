"""Composition helpers used by the small command-line interfaces."""

from __future__ import annotations

from pathlib import Path

from adaspilot.config import Settings
from adaspilot.ingestion.io import load_chunks
from adaspilot.observability.usage import UsageTracker
from adaspilot.retrieval.answers import Answerer, ExtractiveAnswerer, OpenAICompatibleAnswerer
from adaspilot.retrieval.text import (
    ConditionalHyDERetriever,
    FastEmbedReranker,
    HybridRetriever,
    ParentContextRetriever,
    RerankingRetriever,
    Retriever,
    SparseRetriever,
    TemplateHyDEGenerator,
)
from adaspilot.retrieval.visual import (
    ColPaliVisualEmbedder,
    FastEmbedClipVisualEmbedder,
    VisualEmbedder,
)
from adaspilot.safety.guardrails import ImageGuard


def make_answerer(settings: Settings, usage_tracker: UsageTracker | None = None) -> Answerer:
    if settings.llm_base_url and settings.llm_model and settings.llm_api_key:
        return OpenAICompatibleAnswerer(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout_seconds=settings.llm_timeout_seconds,
            image_guard=ImageGuard(settings.max_image_bytes, settings.max_image_pixels),
            usage_tracker=usage_tracker,
        )
    return ExtractiveAnswerer()


def make_hyde_generator(settings: Settings) -> object:
    answerer = make_answerer(settings)
    if isinstance(answerer, OpenAICompatibleAnswerer):
        return answerer
    return TemplateHyDEGenerator()


def make_advanced_retriever(
    settings: Settings,
    dense: Retriever,
    build_dir: Path,
) -> Retriever:
    children = load_chunks(build_dir, "child")
    parents = load_chunks(build_dir, "parent")
    sparse = SparseRetriever(children)
    hybrid = HybridRetriever(
        dense,
        sparse,
        candidate_k=settings.retrieval_candidate_k,
        rrf_constant=settings.rrf_constant,
    )
    reranked = RerankingRetriever(
        hybrid,
        FastEmbedReranker(settings.reranker_model),
        candidate_k=settings.retrieval_candidate_k,
    )
    parent = ParentContextRetriever(reranked, parents)
    if not settings.enable_hyde:
        return parent
    generator = make_hyde_generator(settings)
    return ConditionalHyDERetriever(
        parent,
        generator,  # type: ignore[arg-type]
        settings.hyde_score_threshold,
    )


def make_visual_embedder(settings: Settings) -> VisualEmbedder:
    if settings.visual_backend == "colpali":
        return ColPaliVisualEmbedder(settings.colpali_model)
    return FastEmbedClipVisualEmbedder(
        settings.visual_image_model,
        settings.visual_text_model,
    )
