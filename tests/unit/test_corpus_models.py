"""Corpus manifest model tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adaspilot.domain.corpus import CorpusDocument, CorpusManifest


def test_verified_document_requires_hash() -> None:
    with pytest.raises(ValidationError, match="expected_sha256"):
        CorpusDocument(
            document_id="doc_one",
            title="Example document",
            source_kind="public_url",
            source_uri="https://example.com/doc.pdf",
            status="verified",
        )


def test_verified_document_requires_page_count() -> None:
    with pytest.raises(ValidationError, match="page_count"):
        CorpusDocument(
            document_id="doc_one",
            title="Example document",
            source_kind="public_url",
            source_uri="https://example.com/doc.pdf",
            status="verified",
            expected_sha256="a" * 64,
        )


def test_manifest_rejects_duplicate_document_ids() -> None:
    document = CorpusDocument(
        document_id="doc_one",
        title="Example document",
        source_kind="public_url",
        source_uri="https://example.com/doc.pdf",
    )
    with pytest.raises(ValidationError, match="duplicate document_id"):
        CorpusManifest(
            corpus_id="adas_test",
            description="A test corpus containing duplicate documents.",
            documents=[document, document],
        )
