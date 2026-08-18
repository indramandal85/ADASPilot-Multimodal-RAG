"""Safe corpus-upload staging tests."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from adaspilot.api.corpus import (
    corpus_needs_rebuild,
    deactivate_document,
    ingest_staged_files,
    stage_pdf,
)
from adaspilot.config import Settings
from adaspilot.evaluation.io import load_corpus_manifest


def _pdf_bytes() -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "ADAS upload test")
    payload = document.tobytes()
    document.close()
    return payload


def test_pdf_upload_is_hashed_and_staged_without_changing_manifest(tmp_path: Path) -> None:
    settings = Settings(upload_dir=tmp_path, _env_file=None)
    payload = _pdf_bytes()

    first = stage_pdf(settings, "../My paper.pdf", payload)
    second = stage_pdf(settings, "../My paper.pdf", payload)

    assert first.status == "staged"
    assert second.status == "already_staged"
    assert first.page_count == 1
    assert ".." not in first.stored_name
    assert (tmp_path / first.stored_name).is_file()


def test_upload_rejects_non_pdf_content(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only valid PDF"):
        stage_pdf(Settings(upload_dir=tmp_path, _env_file=None), "notes.txt", b"not a pdf")


def test_missing_processed_build_requires_rebuild(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"schema_version":"1.0","corpus_id":"adas_test","domain":"adas",'
        '"description":"A rebuild-state test corpus.","documents":[]}',
        encoding="utf-8",
    )
    settings = Settings(
        corpus_manifest_path=manifest_path,
        processed_data_dir=tmp_path / "processed",
        _env_file=None,
    )

    assert corpus_needs_rebuild(settings) is True


def test_staged_pdf_is_added_to_manifest_and_raw_corpus(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"schema_version":"1.0","corpus_id":"adas_test","domain":"adas",'
        '"description":"A small corpus used for API lifecycle tests.","documents":[]}',
        encoding="utf-8",
    )
    settings = Settings(
        corpus_manifest_path=manifest_path,
        raw_corpus_dir=tmp_path / "raw",
        upload_dir=tmp_path / "uploads",
        _env_file=None,
    )
    stage_pdf(settings, "My ADAS Paper.pdf", _pdf_bytes())

    result = ingest_staged_files(settings)
    manifest = load_corpus_manifest(manifest_path)

    assert result.ingested == ["My_ADAS_Paper.pdf"]
    assert not result.errors
    assert len(manifest.documents) == 1
    assert manifest.documents[0].status.value == "verified"
    assert (settings.raw_corpus_dir / f"{manifest.documents[0].document_id}.pdf").is_file()


def test_deactivate_keeps_file_and_marks_manifest_inactive(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pdf_path = raw_dir / "sample_doc.pdf"
    payload = _pdf_bytes()
    pdf_path.write_bytes(payload)
    digest = __import__("hashlib").sha256(payload).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        (
            '{"schema_version":"1.0","corpus_id":"adas_test","domain":"adas",'
            '"description":"A small corpus used for deactivate tests.","documents":[{'
            '"document_id":"sample_doc","title":"Sample Document",'
            '"source_kind":"local_file","source_uri":"sample_doc.pdf",'
            f'"expected_sha256":"{digest}","page_count":1,"status":"verified"}}]}}'
        ),
        encoding="utf-8",
    )
    settings = Settings(
        corpus_manifest_path=manifest_path,
        raw_corpus_dir=raw_dir,
        _env_file=None,
    )

    result = deactivate_document(settings, "sample_doc")

    assert result.status == "deactivated"
    assert load_corpus_manifest(manifest_path).documents[0].status.value == "inactive"
    assert pdf_path.is_file()


def test_uploading_an_inactive_pdf_reactivates_existing_document(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    upload_dir = tmp_path / "uploads"
    raw_dir.mkdir()
    payload = _pdf_bytes()
    pdf_path = raw_dir / "sample_doc.pdf"
    pdf_path.write_bytes(payload)
    digest = __import__("hashlib").sha256(payload).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        (
            '{"schema_version":"1.0","corpus_id":"adas_test","domain":"adas",'
            '"description":"A small corpus used for reactivate tests.","documents":[{'
            '"document_id":"sample_doc","title":"Sample Document",'
            '"source_kind":"local_file","source_uri":"sample_doc.pdf",'
            f'"expected_sha256":"{digest}","page_count":1,"status":"inactive"}}]}}'
        ),
        encoding="utf-8",
    )
    settings = Settings(
        corpus_manifest_path=manifest_path,
        raw_corpus_dir=raw_dir,
        upload_dir=upload_dir,
        _env_file=None,
    )
    staged = stage_pdf(settings, "Sample Document.pdf", payload)

    result = ingest_staged_files(settings)
    manifest = load_corpus_manifest(manifest_path)

    assert result.ingested == []
    assert result.reactivated == ["Sample Document"]
    assert not result.errors
    assert manifest.documents[0].status.value == "verified"
    assert not (upload_dir / staged.stored_name).exists()
    assert pdf_path.is_file()
