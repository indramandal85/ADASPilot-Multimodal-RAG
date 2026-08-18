"""Visual routing, scoring, fusion, and metrics tests."""

from __future__ import annotations

import numpy as np

from adaspilot.domain.retrieval import SearchHit
from adaspilot.retrieval.visual import (
    FusedMultimodalRetriever,
    VisualIndex,
    VisualQueryDetector,
)


class FakeVisualEmbedder:
    name = "fake"

    def embed_images(self, paths: object) -> list[np.ndarray]:
        return []

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0] if "radar" in query else [0.0, 1.0])

    def score(self, query: np.ndarray, document: np.ndarray) -> float:
        return float(query @ document)


class TextRetriever:
    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        return [
            SearchHit(
                record_id="child_text",
                score=0.8,
                document_id="doc",
                page_number=2,
                text="text evidence",
                source="dense",
            )
        ][:top_k]


def test_visual_index_and_fused_routing() -> None:
    index = VisualIndex(
        records=[
            {
                "record_id": "page_1",
                "document_id": "doc",
                "page_number": 1,
                "image_path": "page.png",
                "text": "radar figure",
            },
            {
                "record_id": "page_2",
                "document_id": "doc",
                "page_number": 2,
                "image_path": "page2.png",
                "text": "camera figure",
            },
        ],
        embeddings=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
        embedder=FakeVisualEmbedder(),
    )
    detector = VisualQueryDetector()
    fused = FusedMultimodalRetriever(TextRetriever(), index, detector)

    assert detector.is_visual("What is shown in the radar figure?")
    assert not detector.is_visual("What does radar measure?")
    assert index.search("radar figure", 1)[0].page_number == 1
    assert fused.retrieve("What is shown in the radar figure?", 2)[0].source == "fused"
    assert fused.retrieve("What does radar measure?", 1)[0].source == "dense"
