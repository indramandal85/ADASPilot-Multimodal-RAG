"""File-backed page/figure index and visual metric tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from adaspilot.domain.evaluation import EvaluationExample
from adaspilot.domain.ingestion import AssetKind, PageBox, PageRecord, VisualAssetRecord
from adaspilot.evaluation.visual import evaluate_visual_retrieval
from adaspilot.ingestion.io import write_jsonl
from adaspilot.retrieval.visual import VisualIndex, build_visual_index, load_visual_index


class FileNameEmbedder:
    name = "filename-test"

    def embed_images(self, paths: Sequence[Path]) -> list[np.ndarray]:
        return [
            np.array([1.0, 0.0], dtype=np.float32)
            if "figure" in path.name
            else np.array([0.0, 1.0], dtype=np.float32)
            for path in paths
        ]

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def score(self, query: np.ndarray, document: np.ndarray) -> float:
        return float(query @ document)


class ReuseOnlyEmbedder(FileNameEmbedder):
    def embed_images(self, paths: Sequence[Path]) -> list[np.ndarray]:
        raise AssertionError("unchanged visual records should use cached embeddings")


def _visual_example() -> EvaluationExample:
    return EvaluationExample.model_validate(
        {
            "question_id": "q_visual_radar",
            "question": "What is visible in the radar figure?",
            "question_type": "visual",
            "verification_status": "verified",
            "should_answer": True,
            "expected_response_status": "answered",
            "expected_execution_mode": "direct",
            "required_modalities": ["visual"],
            "reference_answer": "The figure shows a radar sensor.",
            "grading_notes": ["Use the figure."],
            "evidence": [
                {
                    "document_id": "sample_doc",
                    "page_number": 1,
                    "asset_id": "asset_figure",
                }
            ],
            "verified_by": "test",
            "verified_at": "2026-08-17T00:00:00Z",
        }
    )


def test_build_load_and_evaluate_visual_index(tmp_path: Path) -> None:
    build_dir = tmp_path / "build"
    page_path = build_dir / "pages" / "page.png"
    figure_path = build_dir / "figures" / "radar_figure.png"
    page_path.parent.mkdir(parents=True)
    figure_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20), "white").save(page_path)
    Image.new("RGB", (20, 20), "black").save(figure_path)
    page = PageRecord(
        page_id="page_1",
        document_id="sample_doc",
        document_version="doc_1",
        corpus_version="corpus_1",
        page_number=1,
        width=100,
        height=100,
        text="Radar sensor page",
        text_block_count=1,
        rendered_image_path="pages/page.png",
        visual_asset_ids=["asset_figure"],
    )
    asset = VisualAssetRecord(
        asset_id="asset_figure",
        document_id="sample_doc",
        page_number=1,
        kind=AssetKind.FIGURE,
        bounding_box=PageBox(x0=1, y0=1, x1=50, y1=50),
        image_path="figures/radar_figure.png",
        caption="Figure 1. Radar sensor.",
    )
    write_jsonl(build_dir / "pages.jsonl", [page])
    write_jsonl(build_dir / "visual_assets.jsonl", [asset])

    embedder = FileNameEmbedder()
    output_dir = tmp_path / "visual_index"
    index = build_visual_index(build_dir=build_dir, output_dir=output_dir, embedder=embedder)
    loaded = load_visual_index(output_dir, embedder)
    metrics = evaluate_visual_retrieval(loaded, [_visual_example()])

    assert len(index.records) == 2
    assert loaded.search("radar figure", 1)[0].asset_id == "asset_figure"
    assert metrics.visual_recall_at_1 == 1
    assert metrics.multimodal_faithfulness == 1

    reused = build_visual_index(
        build_dir=build_dir,
        output_dir=tmp_path / "visual_second",
        embedder=ReuseOnlyEmbedder(),
    )
    assert len(reused.embeddings) == 2

    empty = evaluate_visual_retrieval(loaded, [])
    assert empty.query_count == 0

    try:
        VisualIndex(records=[], embeddings=[np.array([1.0])], embedder=embedder)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("mismatched visual index was accepted")
