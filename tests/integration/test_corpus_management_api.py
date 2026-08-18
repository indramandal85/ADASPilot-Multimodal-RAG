"""Corpus management and IEEE citation API contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import cast

import httpx
import pymupdf
import pytest

from adaspilot.api import app as api_app_module
from adaspilot.api.app import create_app
from adaspilot.api.runtime import RuntimeManager
from adaspilot.config import Environment, Settings
from adaspilot.domain.ingestion import (
    CitationMetadata,
    CorpusBuild,
    DocumentRecord,
    IngestionConfig,
)
from adaspilot.ingestion.io import write_json, write_jsonl


class CorpusOnlyManager:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def _settings(tmp_path: Path) -> Settings:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    payload = _pdf_bytes("Existing ADAS document")
    digest = hashlib.sha256(payload).hexdigest()
    (raw_dir / "sample_doc.pdf").write_bytes(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "corpus_id": "adas_test",
                "domain": "adas",
                "description": "A small corpus used for API endpoint tests.",
                "documents": [
                    {
                        "document_id": "sample_doc",
                        "title": "Sample ADAS Document",
                        "source_kind": "local_file",
                        "source_uri": "sample_doc.pdf",
                        "expected_sha256": digest,
                        "page_count": 1,
                        "status": "verified",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    processed = tmp_path / "processed"
    build_dir = processed / "corpus_test"
    build = CorpusBuild(
        corpus_id="adas_test",
        corpus_version="corpus_test",
        source_manifest_sha256="a" * 64,
        ingestion_config=IngestionConfig(),
        document_count=1,
        unique_document_count=1,
        page_count=1,
        table_count=0,
        visual_asset_count=0,
        parent_chunk_count=1,
        child_chunk_count=1,
    )
    write_json(build_dir / "corpus.json", build)
    write_jsonl(
        build_dir / "documents.jsonl",
        [
            DocumentRecord(
                document_id="sample_doc",
                document_version="doc_test",
                corpus_version="corpus_test",
                source_path=(raw_dir / "sample_doc.pdf").as_posix(),
                source_sha256=digest,
                page_count=1,
                citation_metadata=CitationMetadata(
                    title="Sample ADAS Document",
                    authors=["Indra Mandal"],
                    year=2026,
                    venue="ADAS Test Review",
                ),
            )
        ],
    )
    processed.mkdir(exist_ok=True)
    (processed / "current.json").write_text(
        json.dumps({"corpus_version": "corpus_test"}),
        encoding="utf-8",
    )
    return Settings(
        environment=Environment.TEST,
        corpus_manifest_path=manifest_path,
        raw_corpus_dir=raw_dir,
        upload_dir=tmp_path / "uploads",
        processed_data_dir=processed,
        index_dir=tmp_path / "index",
        _env_file=None,
    )


@pytest.mark.asyncio
async def test_manage_documents_and_get_ieee_citation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    manager = CorpusOnlyManager()
    monkeypatch.setattr(
        api_app_module,
        "_rebuild_corpus",
        lambda _settings, _progress=None: "corpus_new",
    )
    app = create_app(
        settings=settings,
        runtime_manager=cast(RuntimeManager, manager),
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        status = await client.get("/corpus/status")
        citation = await client.get("/corpus/documents/sample_doc/citation", params={"ref": 2})
        upload = await client.post(
            "/corpus/upload",
            files={"file": ("new-paper.pdf", _pdf_bytes("A new ADAS paper"), "application/pdf")},
        )
        processed = await client.post("/corpus/ingest-staged")
        for _ in range(50):
            job = await client.get("/corpus/ingest-status")
            if job.json()["status"] != "running":
                break
            await asyncio.sleep(0.01)
        removed = await client.delete("/corpus/documents/sample_doc")
        missing = await client.delete("/corpus/documents/unknown_doc")

    assert status.status_code == 200
    assert status.json()["documents"][0]["document_id"] == "sample_doc"
    assert citation.status_code == 200
    assert citation.json()["citation"].startswith('[2] I. Mandal, "Sample ADAS Document"')
    assert upload.json()["status"] == "staged"
    assert processed.status_code == 202
    assert processed.json()["status"] == "running"
    assert job.json()["status"] == "succeeded"
    assert job.json()["ingested"] == ["new-paper.pdf"]
    assert job.json()["corpus_version"] == "corpus_new"
    assert removed.json() == {"document_id": "sample_doc", "status": "deactivated"}
    assert missing.status_code == 404
    assert manager.close_count == 2
