"""Corpus status, safe PDF staging, and explicit corpus lifecycle changes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pymupdf

from adaspilot.api.models import (
    CorpusDocumentStatus,
    CorpusStatusResponse,
    DocumentActionResponse,
    IngestStagedResponse,
    UploadResponse,
)
from adaspilot.config import Settings
from adaspilot.domain.corpus import (
    CorpusDocument,
    CorpusManifest,
    DocumentStatus,
    SourceKind,
)
from adaspilot.evaluation.io import load_corpus_manifest
from adaspilot.ingestion.io import resolve_current_build


def corpus_status(settings: Settings) -> CorpusStatusResponse:
    _, build = resolve_current_build(settings.processed_data_dir)
    manifest = load_corpus_manifest(settings.corpus_manifest_path)
    staged = list(settings.upload_dir.glob("*.pdf")) if settings.upload_dir.exists() else []
    return CorpusStatusResponse(
        corpus_id=build.corpus_id,
        corpus_version=build.corpus_version,
        document_count=build.document_count,
        page_count=build.page_count,
        table_count=build.table_count,
        visual_asset_count=build.visual_asset_count,
        parent_chunk_count=build.parent_chunk_count,
        child_chunk_count=build.child_chunk_count,
        staged_upload_count=len(staged),
        documents=[
            CorpusDocumentStatus(
                document_id=item.document_id,
                title=item.title,
                page_count=item.page_count,
                status=item.status.value,
            )
            for item in manifest.documents
        ],
    )


def stage_pdf(settings: Settings, filename: str, content: bytes) -> UploadResponse:
    digest, page_count, _ = _inspect_pdf(settings, filename, content)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(filename).name).strip("._")
    safe_name = safe_name or "document.pdf"
    stored_name = f"{digest[:12]}_{safe_name}"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.upload_dir / stored_name
    duplicate = destination.exists()
    if not duplicate:
        destination.write_bytes(content)
    return UploadResponse(
        filename=filename,
        stored_name=stored_name,
        sha256=digest,
        page_count=page_count,
        status="already_staged" if duplicate else "staged",
        next_step="Use Process uploaded files to ingest and rebuild the local indexes.",
    )


def _inspect_pdf(settings: Settings, filename: str, content: bytes) -> tuple[str, int, str | None]:
    if len(content) > settings.max_upload_bytes:
        raise ValueError(f"PDF exceeds the {settings.max_upload_bytes}-byte upload limit")
    if not filename.lower().endswith(".pdf") or not content.startswith(b"%PDF-"):
        raise ValueError("Only valid PDF files can be staged")
    try:
        document = pymupdf.open(stream=content, filetype="pdf")
        page_count = document.page_count
        title = str((document.metadata or {}).get("title") or "").strip() or None
        document.close()
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError("The uploaded PDF could not be opened safely") from exc
    if page_count < 1:
        raise ValueError("The uploaded PDF has no pages")
    digest = hashlib.sha256(content).hexdigest()
    return digest, page_count, title


def _write_manifest(path: Path, manifest: CorpusManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def deactivate_document(settings: Settings, document_id: str) -> DocumentActionResponse:
    manifest = load_corpus_manifest(settings.corpus_manifest_path)
    selected = next(
        (document for document in manifest.documents if document.document_id == document_id),
        None,
    )
    if selected is None:
        raise KeyError(document_id)
    selected.status = DocumentStatus.INACTIVE
    _write_manifest(settings.corpus_manifest_path, manifest)
    return DocumentActionResponse(document_id=document_id, status="deactivated")


def _document_id(filename: str, digest: str, used_ids: set[str]) -> str:
    original = re.sub(r"^[a-f0-9]{12}_", "", Path(filename).stem, flags=re.I)
    slug = re.sub(r"[^a-z0-9]+", "_", original.lower()).strip("_")
    slug = (slug or f"document_{digest[:8]}")[:63].strip("_")
    if len(slug) < 3:
        slug = f"doc_{slug}_{digest[:6]}"[:63]
    candidate = slug
    if candidate in used_ids:
        candidate = f"{slug[:54]}_{digest[:8]}"
    return candidate


def ingest_staged_files(settings: Settings) -> IngestStagedResponse:
    """Validate staged PDFs, move them into raw storage, and update the manifest."""

    manifest = load_corpus_manifest(settings.corpus_manifest_path)
    documents_by_hash = {
        document.expected_sha256: document
        for document in manifest.documents
        if document.expected_sha256
    }
    used_ids = {document.document_id for document in manifest.documents}
    settings.raw_corpus_dir.mkdir(parents=True, exist_ok=True)
    ingested: list[str] = []
    reactivated: list[str] = []
    errors: list[str] = []
    additions: list[CorpusDocument] = []
    staged_paths = sorted(settings.upload_dir.glob("*.pdf")) if settings.upload_dir.exists() else []

    for staged_path in staged_paths:
        try:
            content = staged_path.read_bytes()
            digest, page_count, embedded_title = _inspect_pdf(settings, staged_path.name, content)
            existing_document = documents_by_hash.get(digest)
            if existing_document is not None:
                if existing_document.status == DocumentStatus.INACTIVE:
                    raw_path = settings.raw_corpus_dir / f"{existing_document.document_id}.pdf"
                    raw_matches = (
                        raw_path.exists()
                        and hashlib.sha256(raw_path.read_bytes()).hexdigest() == digest
                    )
                    if not raw_matches:
                        staged_path.replace(raw_path)
                    else:
                        staged_path.unlink()
                    existing_document.status = DocumentStatus.VERIFIED
                    reactivated.append(existing_document.title)
                else:
                    # The verified raw PDF is already authoritative. Removing this redundant
                    # staged copy prevents the same warning from appearing on every rerun.
                    staged_path.unlink()
                    errors.append(f"{staged_path.name}: already active in the corpus")
                continue
            document_id = _document_id(staged_path.name, digest, used_ids)
            destination = settings.raw_corpus_dir / f"{document_id}.pdf"
            if destination.exists():
                errors.append(f"{staged_path.name}: destination already exists")
                continue
            original_name = re.sub(r"^[a-f0-9]{12}_", "", staged_path.name, flags=re.I)
            fallback_title = Path(original_name).stem.replace("_", " ").strip().title()
            staged_path.replace(destination)
            additions.append(
                CorpusDocument(
                    document_id=document_id,
                    title=embedded_title or fallback_title or document_id,
                    source_kind=SourceKind.LOCAL_FILE,
                    source_uri=destination.as_posix(),
                    expected_sha256=digest,
                    page_count=page_count,
                    status=DocumentStatus.VERIFIED,
                    notes="Added through the local ADASPilot document sidebar.",
                )
            )
            documents_by_hash[digest] = additions[-1]
            used_ids.add(document_id)
            ingested.append(original_name)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{staged_path.name}: {exc}")

    if additions or reactivated:
        manifest.documents.extend(additions)
        _write_manifest(settings.corpus_manifest_path, manifest)
    return IngestStagedResponse(
        ingested=ingested,
        reactivated=reactivated,
        errors=errors,
    )


def corpus_needs_rebuild(settings: Settings) -> bool:
    """Return whether the current processed build predates the manifest state."""

    try:
        _, build = resolve_current_build(settings.processed_data_dir)
    except (FileNotFoundError, KeyError, ValueError):
        return True
    manifest_hash = hashlib.sha256(settings.corpus_manifest_path.read_bytes()).hexdigest()
    return build.source_manifest_sha256 != manifest_hash
