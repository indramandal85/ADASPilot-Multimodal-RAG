"""Small embedding interfaces with local FastEmbed and deterministic test implementations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class TextEmbedder(Protocol):
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class FastEmbedTextEmbedder:
    """CPU-friendly BGE embeddings through Qdrant's ONNX FastEmbed runtime."""

    def __init__(self, model_name: str, dimension: int) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.passage_embed(list(texts))
        return [vector.astype(np.float32).tolist() for vector in vectors]

    def embed_query(self, query: str) -> list[float]:
        vector = next(iter(self._model.query_embed(query)))
        return list(vector.astype(np.float32).tolist())


class HashingTextEmbedder:
    """Dependency-free deterministic embedder used only for unit tests and smoke checks."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)
