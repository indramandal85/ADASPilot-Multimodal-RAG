"""Ingestion contract and chunking tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adaspilot.domain.ingestion import IngestionConfig, PageBox
from adaspilot.ingestion.chunking import chunk_page, stable_id


def test_ingestion_config_rejects_invalid_chunk_sizes() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        IngestionConfig(child_chunk_chars=300, child_chunk_overlap=300)
    with pytest.raises(ValidationError, match="larger"):
        IngestionConfig(parent_chunk_chars=500, child_chunk_chars=600)


def test_page_box_requires_ordered_coordinates() -> None:
    with pytest.raises(ValidationError, match="maximums"):
        PageBox(x0=10, y0=10, x1=5, y1=20)


def test_chunk_page_creates_stable_parent_child_links() -> None:
    config = IngestionConfig(parent_chunk_chars=500, child_chunk_chars=150, child_chunk_overlap=20)
    text = "First ADAS sentence explains radar. " * 12
    first_parents, first_children = chunk_page(
        document_id="document_one",
        document_version="doc_123",
        corpus_version="corpus_123",
        page_number=2,
        text=text,
        config=config,
    )
    second_parents, second_children = chunk_page(
        document_id="document_one",
        document_version="doc_123",
        corpus_version="corpus_123",
        page_number=2,
        text=text,
        config=config,
    )

    assert first_parents == second_parents
    assert first_children == second_children
    assert len(first_parents) >= 1
    assert all(
        child.parent_id in {parent.chunk_id for parent in first_parents} for child in first_children
    )
    assert stable_id("test", "a", 1) == stable_id("test", "a", 1)
