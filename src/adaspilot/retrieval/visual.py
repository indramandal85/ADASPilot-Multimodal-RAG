"""Page/figure visual indexing, ColPali-family scoring, routing, and fusion."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from adaspilot.domain.retrieval import SearchHit
from adaspilot.ingestion.io import load_pages, load_visual_assets
from adaspilot.retrieval.text import Retriever


class VisualEmbedder(Protocol):
    name: str

    def embed_images(self, paths: Sequence[Path]) -> list[np.ndarray]: ...

    def embed_query(self, query: str) -> np.ndarray: ...

    def score(self, query: np.ndarray, document: np.ndarray) -> float: ...


class FastEmbedClipVisualEmbedder:
    """Lightweight dual-encoder fallback for laptops without the ColPali extra."""

    def __init__(self, image_model: str, text_model: str) -> None:
        from fastembed import ImageEmbedding, TextEmbedding

        self.name = f"clip:{image_model}"
        self._images = ImageEmbedding(model_name=image_model)
        self._text = TextEmbedding(model_name=text_model)

    def embed_images(self, paths: Sequence[Path]) -> list[np.ndarray]:
        return [
            np.asarray(vector, dtype=np.float32)
            for vector in self._images.embed([str(path) for path in paths])
        ]

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(next(iter(self._text.query_embed(query))), dtype=np.float32)

    def score(self, query: np.ndarray, document: np.ndarray) -> float:
        query_norm = query / max(float(np.linalg.norm(query)), 1e-12)
        document_norm = document / max(float(np.linalg.norm(document)), 1e-12)
        return float(query_norm @ document_norm)


class ColPaliVisualEmbedder:
    """ColSmol/ColPali multi-vector embeddings with ColBERT-style MaxSim."""

    def __init__(self, model_name: str) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from colpali_engine.models import (  # type: ignore[import-not-found]
                ColIdefics3,
                ColIdefics3Processor,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "ColPali support is optional; run `make install-visual` first"
            ) from exc
        self._torch = torch
        self.name = f"colpali:{model_name}"
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32
        self._model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        self._processor = ColIdefics3Processor.from_pretrained(model_name)

    def embed_images(self, paths: Sequence[Path]) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        for path in paths:
            with Image.open(path) as image:
                batch = self._processor.process_images([image.convert("RGB")]).to(
                    self._model.device
                )
            with self._torch.no_grad():
                output = self._model(**batch)[0].float().cpu().numpy()
            embeddings.append(output)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        batch = self._processor.process_queries([query]).to(self._model.device)
        with self._torch.no_grad():
            output = self._model(**batch)[0].float().cpu().numpy()
            return np.asarray(output, dtype=np.float32)

    def score(self, query: np.ndarray, document: np.ndarray) -> float:
        similarities = query @ document.T
        return float(similarities.max(axis=1).sum())


class VisualIndex:
    def __init__(
        self,
        *,
        records: list[dict[str, object]],
        embeddings: list[np.ndarray],
        embedder: VisualEmbedder,
    ) -> None:
        if len(records) != len(embeddings):
            raise ValueError("visual records and embeddings must have the same length")
        self.records = records
        self.embeddings = embeddings
        self.embedder = embedder

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_embedding = self.embedder.embed_query(query)
        scored = [
            (self.embedder.score(query_embedding, embedding), record)
            for record, embedding in zip(self.records, self.embeddings, strict=True)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchHit(
                record_id=str(record["record_id"]),
                score=score,
                document_id=str(record["document_id"]),
                page_number=int(str(record["page_number"])),
                text=str(record.get("text", "")),
                asset_id=(str(record["asset_id"]) if record.get("asset_id") else None),
                image_path=str(record["image_path"]),
                source="visual",
            )
            for score, record in scored[:top_k]
        ]


def build_visual_index(
    *,
    build_dir: Path,
    output_dir: Path,
    embedder: VisualEmbedder,
) -> VisualIndex:
    pages = load_pages(build_dir)
    assets = load_visual_assets(build_dir)
    records: list[dict[str, object]] = [
        {
            "record_id": page.page_id,
            "document_id": page.document_id,
            "page_number": page.page_number,
            "asset_id": None,
            "image_path": page.rendered_image_path,
            "text": page.text[:1000],
            "kind": "page",
        }
        for page in pages
    ]
    records.extend(
        {
            "record_id": asset.asset_id,
            "document_id": asset.document_id,
            "page_number": asset.page_number,
            "asset_id": asset.asset_id,
            "image_path": asset.image_path,
            "text": "\n".join(part for part in [asset.caption, asset.ocr_text] if part),
            "kind": asset.kind.value,
        }
        for asset in assets
    )
    paths = [build_dir / str(record["image_path"]) for record in records]
    cached_embeddings = _load_cached_embeddings(output_dir, embedder.name)
    pending: list[tuple[int, Path]] = []
    ordered_embeddings: list[np.ndarray | None] = []
    for index, (record, path) in enumerate(zip(records, paths, strict=True)):
        cached = cached_embeddings.get(str(record["record_id"]))
        ordered_embeddings.append(cached)
        if cached is None:
            pending.append((index, path))
    for start in range(0, len(pending), 8):
        batch = pending[start : start + 8]
        vectors = embedder.embed_images([path for _, path in batch])
        for (index, _), vector in zip(batch, vectors, strict=True):
            ordered_embeddings[index] = vector
    if any(item is None for item in ordered_embeddings):
        raise ValueError("visual embedder returned fewer vectors than requested")
    embeddings = [item for item in ordered_embeddings if item is not None]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"embedder": embedder.name, "record_count": len(records), "records": records}
    (output_dir / "visual_index.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "visual_embeddings.npz",
        embeddings=np.asarray(embeddings, dtype=object),
    )
    return VisualIndex(records=records, embeddings=embeddings, embedder=embedder)


def _load_cached_embeddings(output_dir: Path, embedder_name: str) -> dict[str, np.ndarray]:
    """Reuse stable page/figure vectors from earlier corpus versions."""

    cache: dict[str, np.ndarray] = {}
    candidates = sorted(
        output_dir.parent.glob("visual_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate == output_dir:
            continue
        metadata_path = candidate / "visual_index.json"
        embeddings_path = candidate / "visual_embeddings.npz"
        if not metadata_path.is_file() or not embeddings_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("embedder") != embedder_name:
                continue
            data = np.load(embeddings_path, allow_pickle=True)
            vectors = data["embeddings"]
            for record, vector in zip(metadata["records"], vectors, strict=True):
                cache.setdefault(
                    str(record["record_id"]),
                    np.asarray(vector, dtype=np.float32),
                )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return cache


def load_visual_index(output_dir: Path, embedder: VisualEmbedder) -> VisualIndex:
    metadata = json.loads((output_dir / "visual_index.json").read_text(encoding="utf-8"))
    if metadata["embedder"] != embedder.name:
        raise ValueError("visual index embedder does not match the configured backend")
    data = np.load(output_dir / "visual_embeddings.npz", allow_pickle=True)
    embeddings = [np.asarray(item, dtype=np.float32) for item in data["embeddings"]]
    return VisualIndex(records=list(metadata["records"]), embeddings=embeddings, embedder=embedder)


class VisualQueryDetector:
    _KEYWORDS = {
        "figure",
        "fig.",
        "image",
        "illustration",
        "diagram",
        "plot",
        "chart",
        "curve",
        "layout",
        "shown",
        "pictured",
        "visual",
    }

    def is_visual(self, query: str) -> bool:
        lowered = query.lower()
        tokens = set(re.findall(r"[a-z]+\.?", lowered))
        return bool(tokens & self._KEYWORDS)


class FusedMultimodalRetriever:
    def __init__(
        self,
        text: Retriever,
        visual: VisualIndex,
        detector: VisualQueryDetector | None = None,
        rrf_constant: int = 60,
    ) -> None:
        self.text = text
        self.visual = visual
        self.detector = detector or VisualQueryDetector()
        self.rrf_constant = rrf_constant

    def retrieve(self, query: str, top_k: int) -> list[SearchHit]:
        text_hits = self.text.retrieve(query, max(top_k * 2, top_k))
        if not self.detector.is_visual(query):
            return text_hits[:top_k]
        visual_hits = self.visual.search(query, max(top_k * 2, top_k))
        combined: dict[tuple[str, int], SearchHit] = {}
        scores: dict[tuple[str, int], float] = {}
        for ranking in [text_hits, visual_hits]:
            for rank, hit in enumerate(ranking, start=1):
                key = (hit.document_id, hit.page_number)
                if key not in combined or hit.source == "visual":
                    combined[key] = hit
                scores[key] = scores.get(key, 0.0) + 1 / (self.rrf_constant + rank)
        ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
        return [
            combined[key].model_copy(update={"score": scores[key], "source": "fused"})
            for key in ordered[:top_k]
        ]
