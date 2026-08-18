"""End-to-end ingestion test using a tiny generated PDF."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pymupdf
import pytest

from adaspilot.domain.ingestion import IngestionConfig
from adaspilot.ingestion import IngestionPipeline
from adaspilot.ingestion.io import load_chunks, load_documents, load_pages, resolve_current_build


def _write_pdf(path: Path) -> str:
    document = pymupdf.open()
    document.set_metadata(
        {
            "title": "Sample ADAS Document",
            "author": "Indra Mandal; Alex Researcher",
            "creationDate": "D:20240817000000Z",
        }
    )
    page = document.new_page(width=400, height=500)
    page.insert_text((40, 60), "Figure 1. ADAS sensor diagram.", fontsize=11)
    page.insert_text(
        (40, 100),
        "Radar measures range and velocity for assisted-driving perception.",
        fontsize=11,
    )
    page.insert_text(
        (40, 125),
        "The camera provides semantic information about lanes and objects.",
        fontsize=11,
    )
    document.save(path)
    document.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_versions_pages_and_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pdf_path = raw_dir / "sample_doc.pdf"
    digest = _write_pdf(pdf_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "corpus_id": "test_corpus",
                "domain": "adas",
                "description": "Small deterministic ingestion test corpus.",
                "documents": [
                    {
                        "document_id": "sample_doc",
                        "title": "Sample ADAS Document",
                        "source_kind": "local_file",
                        "source_uri": str(pdf_path),
                        "expected_sha256": digest,
                        "page_count": 1,
                        "status": "verified",
                    }
                ],
            }
        )
    )
    processed = tmp_path / "processed"
    pipeline = IngestionPipeline(
        manifest_path=manifest_path,
        raw_corpus_dir=raw_dir,
        processed_dir=processed,
        config=IngestionConfig(enable_ocr=False),
    )

    first = pipeline.run()
    second = pipeline.run()
    build_dir, current = resolve_current_build(processed)

    assert first == second == current
    assert first.page_count == 1
    assert first.child_chunk_count > 0
    assert len(load_pages(build_dir)) == 1
    assert load_chunks(build_dir, "child")
    metadata = load_documents(build_dir)[0].citation_metadata
    assert metadata is not None
    assert metadata.title == "Sample ADAS Document"
    assert metadata.authors == ["Indra Mandal", "Alex Researcher"]
    assert metadata.year == 2024
    assert (build_dir / load_pages(build_dir)[0].rendered_image_path).exists()


def test_pipeline_rejects_hash_mismatch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_pdf(raw_dir / "sample_doc.pdf")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "corpus_id": "test_corpus",
                "domain": "adas",
                "description": "Hash mismatch test corpus.",
                "documents": [
                    {
                        "document_id": "sample_doc",
                        "title": "Sample ADAS Document",
                        "source_kind": "local_file",
                        "source_uri": "sample_doc.pdf",
                        "expected_sha256": "a" * 64,
                        "page_count": 1,
                        "status": "verified",
                    }
                ],
            }
        )
    )
    pipeline = IngestionPipeline(
        manifest_path=manifest,
        raw_corpus_dir=raw_dir,
        processed_dir=tmp_path / "processed",
        config=IngestionConfig(enable_ocr=False),
    )

    try:
        pipeline.run()
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("hash mismatch was not rejected")


def test_pipeline_reuses_unchanged_document_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pdf_path = raw_dir / "sample_doc.pdf"
    digest = _write_pdf(pdf_path)
    manifest_path = tmp_path / "manifest.json"

    def write_manifest(description: str) -> None:
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "corpus_id": "test_corpus",
                    "domain": "adas",
                    "description": description,
                    "documents": [
                        {
                            "document_id": "sample_doc",
                            "title": "Sample ADAS Document",
                            "source_kind": "local_file",
                            "source_uri": str(pdf_path),
                            "expected_sha256": digest,
                            "page_count": 1,
                            "status": "verified",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    write_manifest("First manifest state.")
    processed = tmp_path / "processed"
    pipeline = IngestionPipeline(
        manifest_path=manifest_path,
        raw_corpus_dir=raw_dir,
        processed_dir=processed,
        config=IngestionConfig(enable_ocr=False),
    )
    first = pipeline.run()
    write_manifest("Second manifest state.")
    monkeypatch.setattr(
        "adaspilot.ingestion.pipeline.extract_page",
        lambda **_: (_ for _ in ()).throw(AssertionError("page was re-extracted")),
    )

    second = pipeline.run()
    build_dir, _ = resolve_current_build(processed)

    assert second.corpus_version != first.corpus_version
    assert len(load_pages(build_dir)) == 1
    assert (build_dir / load_pages(build_dir)[0].rendered_image_path).is_file()
