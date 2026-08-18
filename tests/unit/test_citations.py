"""IEEE citation formatting tests."""

from __future__ import annotations

from adaspilot.domain.citations import doi_url, format_ieee
from adaspilot.domain.ingestion import CitationMetadata, DocumentRecord


def _document(metadata: CitationMetadata | None) -> DocumentRecord:
    return DocumentRecord(
        document_id="sample_doc",
        document_version="doc_v1",
        corpus_version="corpus_v1",
        source_path="data/corpus/raw/sample_doc.pdf",
        source_sha256="a" * 64,
        page_count=10,
        citation_metadata=metadata,
    )


def test_ieee_formatter_uses_authors_venue_year_and_doi() -> None:
    document = _document(
        CitationMetadata(
            title="Radar-Based ADAS",
            authors=["Arun Kumar", "Bina Singh"],
            year=2023,
            venue="IEEE Transactions on Vehicular Technology",
            doi="10.1109/TVT.2023.12345",
        )
    )

    citation = format_ieee(document, 2)

    assert citation.startswith('[2] A. Kumar and B. Singh, "Radar-Based ADAS"')
    assert "IEEE Transactions on Vehicular Technology, 2023" in citation
    assert citation.endswith("doi: 10.1109/TVT.2023.12345.")
    assert doi_url(document) == "https://doi.org/10.1109/TVT.2023.12345"


def test_ieee_formatter_falls_back_to_filename() -> None:
    assert format_ieee(_document(None), 1) == "[1] Sample Doc."
