"""End-to-end deterministic corpus ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from adaspilot.domain.corpus import CorpusDocument, CorpusManifest, DocumentStatus
from adaspilot.domain.ingestion import (
    ChunkKind,
    ChunkRecord,
    CitationMetadata,
    ContentModality,
    CorpusBuild,
    DocumentRecord,
    IngestionConfig,
    PageRecord,
    TableRecord,
    VisualAssetRecord,
)
from adaspilot.evaluation.io import load_corpus_manifest
from adaspilot.ingestion.chunking import chunk_page, stable_id
from adaspilot.ingestion.io import read_json, read_jsonl, write_json, write_jsonl
from adaspilot.ingestion.pdf import extract_page


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus_version(manifest: CorpusManifest, manifest_hash: str, config: IngestionConfig) -> str:
    active_documents = [
        document for document in manifest.documents if document.status != DocumentStatus.INACTIVE
    ]
    inputs = {
        "manifest_sha256": manifest_hash,
        "document_hashes": sorted(
            document.expected_sha256 for document in active_documents if document.expected_sha256
        ),
        "ingestion_config": config.model_dump(mode="json"),
    }
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return f"corpus_{digest[:16]}"


_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


@dataclass(frozen=True)
class _CachedArtifacts:
    build_dir: Path
    pages: list[PageRecord]
    tables: list[TableRecord]
    assets: list[VisualAssetRecord]
    parents: list[ChunkRecord]
    children: list[ChunkRecord]


def _metadata_value(metadata: dict[str, object], name: str) -> str:
    value = metadata.get(name) or metadata.get(name.capitalize()) or ""
    return str(value).strip()


def _authors(raw_authors: str) -> list[str]:
    if not raw_authors:
        return []
    separator = ";" if ";" in raw_authors else ","
    return [part.strip() for part in raw_authors.split(separator) if part.strip()]


def _clean_doi(value: str) -> str:
    return value.rstrip(".,;:)]}")


def extract_citation_metadata(
    pdf: pymupdf.Document,
    selected: CorpusDocument,
) -> CitationMetadata:
    """Combine embedded PDF metadata with authoritative manifest fallbacks."""

    metadata = dict(pdf.metadata or {})
    title = _metadata_value(metadata, "title") or selected.title
    raw_authors = _metadata_value(metadata, "author")
    creation_date = _metadata_value(metadata, "creationDate")
    year_match = re.search(r"(?:D:)?(19\d{2}|20\d{2})", creation_date)
    embedded_year = int(year_match.group(1)) if year_match else None
    year = selected.publication_year or embedded_year
    doi = None
    fields = [
        _metadata_value(metadata, "subject"),
        _metadata_value(metadata, "keywords"),
    ]
    if pdf.page_count:
        fields.append(pdf[0].get_text("text")[:8000])
    for field in fields:
        match = _DOI_PATTERN.search(field)
        if match:
            doi = _clean_doi(match.group(0))
            break
    return CitationMetadata(
        title=title or None,
        authors=_authors(raw_authors),
        year=year,
        venue=selected.publisher or None,
        doi=doi,
    )


class IngestionPipeline:
    """Build page, table, visual, and chunk manifests for one corpus version."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        raw_corpus_dir: Path,
        processed_dir: Path,
        config: IngestionConfig,
    ) -> None:
        self.manifest_path = manifest_path
        self.raw_corpus_dir = raw_corpus_dir
        self.processed_dir = processed_dir
        self.config = config

    def run(self) -> CorpusBuild:
        manifest = load_corpus_manifest(self.manifest_path)
        manifest_hash = _manifest_hash(self.manifest_path)
        corpus_version = _corpus_version(manifest, manifest_hash, self.config)
        build_dir = self.processed_dir / corpus_version
        existing = build_dir / "corpus.json"
        if existing.exists():
            self._write_current(corpus_version)
            return read_json(existing, CorpusBuild)

        documents: list[DocumentRecord] = []
        pages: list[PageRecord] = []
        tables: list[TableRecord] = []
        assets: list[VisualAssetRecord] = []
        parents: list[ChunkRecord] = []
        children: list[ChunkRecord] = []
        first_document_by_hash: dict[str, str] = {}

        for selected in manifest.documents:
            if selected.status == DocumentStatus.INACTIVE:
                continue
            pdf_path = self.raw_corpus_dir / f"{selected.document_id}.pdf"
            if not pdf_path.exists():
                raise FileNotFoundError(f"missing corpus PDF: {pdf_path}")
            actual_hash = sha256_file(pdf_path)
            if selected.expected_sha256 and actual_hash != selected.expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {selected.document_id}: "
                    f"expected {selected.expected_sha256}, got {actual_hash}"
                )
            duplicate_of = first_document_by_hash.get(actual_hash)
            if duplicate_of is None:
                first_document_by_hash[actual_hash] = selected.document_id
            document_version = f"doc_{actual_hash[:16]}"
            with pymupdf.open(pdf_path) as pdf:
                if selected.page_count and pdf.page_count != selected.page_count:
                    raise ValueError(
                        f"page-count mismatch for {selected.document_id}: "
                        f"expected {selected.page_count}, got {pdf.page_count}"
                    )
                documents.append(
                    DocumentRecord(
                        document_id=selected.document_id,
                        document_version=document_version,
                        corpus_version=corpus_version,
                        source_path=pdf_path.as_posix(),
                        source_sha256=actual_hash,
                        page_count=pdf.page_count,
                        duplicate_of=duplicate_of,
                        citation_metadata=extract_citation_metadata(pdf, selected),
                    )
                )
                if duplicate_of is not None:
                    continue
                cached = self._find_cached_artifacts(
                    document_id=selected.document_id,
                    document_version=document_version,
                    target_build_dir=build_dir,
                )
                if cached is not None:
                    self._copy_cached_images(cached, build_dir, selected.document_id)
                    pages.extend(
                        page.model_copy(update={"corpus_version": corpus_version})
                        for page in cached.pages
                    )
                    tables.extend(cached.tables)
                    assets.extend(cached.assets)
                    parents.extend(
                        chunk.model_copy(update={"corpus_version": corpus_version})
                        for chunk in cached.parents
                    )
                    children.extend(
                        chunk.model_copy(update={"corpus_version": corpus_version})
                        for chunk in cached.children
                    )
                    continue
                for pdf_page in pdf:
                    extracted = extract_page(
                        page=pdf_page,
                        document_id=selected.document_id,
                        document_version=document_version,
                        corpus_version=corpus_version,
                        build_dir=build_dir,
                        config=self.config,
                    )
                    pages.append(extracted.page)
                    tables.extend(extracted.tables)
                    assets.extend(extracted.visual_assets)
                    page_parents, page_children = chunk_page(
                        document_id=selected.document_id,
                        document_version=document_version,
                        corpus_version=corpus_version,
                        page_number=extracted.page.page_number,
                        text=extracted.page.text,
                        config=self.config,
                    )
                    parents.extend(page_parents)
                    children.extend(page_children)
                    for table in extracted.tables:
                        text = "\n\n".join(part for part in [table.caption, table.markdown] if part)
                        parent_id = stable_id("parent", table.table_id, text)
                        parents.append(
                            ChunkRecord(
                                chunk_id=parent_id,
                                document_id=selected.document_id,
                                document_version=document_version,
                                corpus_version=corpus_version,
                                page_number=table.page_number,
                                kind=ChunkKind.PARENT,
                                modality=ContentModality.TABLE,
                                text=text,
                                table_id=table.table_id,
                            )
                        )
                        children.append(
                            ChunkRecord(
                                chunk_id=stable_id("child", parent_id, text),
                                parent_id=parent_id,
                                document_id=selected.document_id,
                                document_version=document_version,
                                corpus_version=corpus_version,
                                page_number=table.page_number,
                                kind=ChunkKind.CHILD,
                                modality=ContentModality.TABLE,
                                text=text,
                                table_id=table.table_id,
                            )
                        )
                    for asset in extracted.visual_assets:
                        text = "\n\n".join(part for part in [asset.caption, asset.ocr_text] if part)
                        if not text:
                            continue
                        parent_id = stable_id("parent", asset.asset_id, text)
                        parents.append(
                            ChunkRecord(
                                chunk_id=parent_id,
                                document_id=selected.document_id,
                                document_version=document_version,
                                corpus_version=corpus_version,
                                page_number=asset.page_number,
                                kind=ChunkKind.PARENT,
                                modality=ContentModality.VISUAL,
                                text=text,
                                asset_id=asset.asset_id,
                            )
                        )
                        children.append(
                            ChunkRecord(
                                chunk_id=stable_id("child", parent_id, text),
                                parent_id=parent_id,
                                document_id=selected.document_id,
                                document_version=document_version,
                                corpus_version=corpus_version,
                                page_number=asset.page_number,
                                kind=ChunkKind.CHILD,
                                modality=ContentModality.VISUAL,
                                text=text,
                                asset_id=asset.asset_id,
                            )
                        )

        build = CorpusBuild(
            corpus_id=manifest.corpus_id,
            corpus_version=corpus_version,
            source_manifest_sha256=manifest_hash,
            ingestion_config=self.config,
            document_count=len(documents),
            unique_document_count=len(first_document_by_hash),
            page_count=len(pages),
            table_count=len(tables),
            visual_asset_count=len(assets),
            parent_chunk_count=len(parents),
            child_chunk_count=len(children),
        )
        write_jsonl(build_dir / "documents.jsonl", documents)
        write_jsonl(build_dir / "pages.jsonl", pages)
        write_jsonl(build_dir / "tables.jsonl", tables)
        write_jsonl(build_dir / "visual_assets.jsonl", assets)
        write_jsonl(build_dir / "parent_chunks.jsonl", parents)
        write_jsonl(build_dir / "child_chunks.jsonl", children)
        write_json(build_dir / "corpus.json", build)
        self._write_current(corpus_version)
        return build

    def _find_cached_artifacts(
        self,
        *,
        document_id: str,
        document_version: str,
        target_build_dir: Path,
    ) -> _CachedArtifacts | None:
        candidates = sorted(
            self.processed_dir.glob("corpus_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if candidate == target_build_dir or not (candidate / "corpus.json").is_file():
                continue
            try:
                previous_build = read_json(candidate / "corpus.json", CorpusBuild)
                if previous_build.ingestion_config != self.config:
                    continue
                previous_documents = read_jsonl(
                    candidate / "documents.jsonl",
                    DocumentRecord,
                )
                previous_document = next(
                    (
                        item
                        for item in previous_documents
                        if item.document_id == document_id
                        and item.document_version == document_version
                        and item.duplicate_of is None
                    ),
                    None,
                )
                if previous_document is None:
                    continue
                cached = _CachedArtifacts(
                    build_dir=candidate,
                    pages=[
                        item
                        for item in read_jsonl(candidate / "pages.jsonl", PageRecord)
                        if item.document_id == document_id
                    ],
                    tables=[
                        item
                        for item in read_jsonl(candidate / "tables.jsonl", TableRecord)
                        if item.document_id == document_id
                    ],
                    assets=[
                        item
                        for item in read_jsonl(candidate / "visual_assets.jsonl", VisualAssetRecord)
                        if item.document_id == document_id
                    ],
                    parents=[
                        item
                        for item in read_jsonl(candidate / "parent_chunks.jsonl", ChunkRecord)
                        if item.document_id == document_id
                    ],
                    children=[
                        item
                        for item in read_jsonl(candidate / "child_chunks.jsonl", ChunkRecord)
                        if item.document_id == document_id
                    ],
                )
                if len(cached.pages) != previous_document.page_count:
                    continue
                image_paths = [item.rendered_image_path for item in cached.pages]
                image_paths.extend(item.image_path for item in cached.assets)
                if all((candidate / image_path).is_file() for image_path in image_paths):
                    return cached
            except (FileNotFoundError, OSError, ValueError):
                continue
        return None

    @staticmethod
    def _copy_cached_images(
        cached: _CachedArtifacts,
        target_build_dir: Path,
        document_id: str,
    ) -> None:
        for folder in ("pages", "assets"):
            source = cached.build_dir / folder / document_id
            if source.is_dir():
                shutil.copytree(
                    source,
                    target_build_dir / folder / document_id,
                    dirs_exist_ok=True,
                )

    def _write_current(self, corpus_version: str) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        pointer = {"corpus_version": corpus_version}
        (self.processed_dir / "current.json").write_text(
            json.dumps(pointer, indent=2) + "\n",
            encoding="utf-8",
        )
