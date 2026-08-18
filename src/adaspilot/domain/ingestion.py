"""Contracts produced by deterministic multimodal ingestion."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AssetKind(StrEnum):
    FIGURE = "figure"
    PLOT = "plot"
    DIAGRAM = "diagram"
    TABLE_IMAGE = "table_image"
    EMBEDDED_IMAGE = "embedded_image"


class ChunkKind(StrEnum):
    PARENT = "parent"
    CHILD = "child"


class ContentModality(StrEnum):
    TEXT = "text"
    TABLE = "table"
    VISUAL = "visual"


class PageBox(BaseModel):
    """Coordinates in PDF page space."""

    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(gt=0)
    y1: float = Field(gt=0)

    @model_validator(mode="after")
    def coordinates_are_ordered(self) -> PageBox:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("page-box maximums must exceed minimums")
        return self


class IngestionConfig(BaseModel):
    """Settings that affect deterministic corpus output."""

    schema_version: Literal["1.0"] = "1.0"
    extractor_revision: Literal["modules_2_5_v3", "modules_2_5_v4", "modules_2_5_v5"] = (
        "modules_2_5_v5"
    )
    render_dpi: int = Field(default=120, ge=72, le=300)
    parent_chunk_chars: int = Field(default=1800, ge=500, le=6000)
    child_chunk_chars: int = Field(default=520, ge=150, le=2000)
    child_chunk_overlap: int = Field(default=80, ge=0, le=500)
    minimum_ocr_chars: int = Field(default=40, ge=0, le=1000)
    extract_tables: bool = True
    extract_visuals: bool = True
    enable_ocr: bool = True

    @model_validator(mode="after")
    def chunk_sizes_are_consistent(self) -> IngestionConfig:
        if self.child_chunk_overlap >= self.child_chunk_chars:
            raise ValueError("child_chunk_overlap must be smaller than child_chunk_chars")
        if self.child_chunk_chars > self.parent_chunk_chars:
            raise ValueError("child chunks cannot be larger than parent chunks")
        return self


class CitationMetadata(BaseModel):
    """Bibliographic fields extracted from a PDF and its verified manifest entry."""

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1900, le=2100)
    venue: str | None = None
    doi: str | None = None


class DocumentRecord(BaseModel):
    document_id: str
    document_version: str
    corpus_version: str
    source_path: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_count: int = Field(ge=1)
    duplicate_of: str | None = None
    citation_metadata: CitationMetadata | None = None


class PageRecord(BaseModel):
    page_id: str
    document_id: str
    document_version: str
    corpus_version: str
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    text: str
    text_block_count: int = Field(ge=0)
    rendered_image_path: str
    ocr_used: bool = False
    table_ids: list[str] = Field(default_factory=list)
    visual_asset_ids: list[str] = Field(default_factory=list)


class TableRecord(BaseModel):
    table_id: str
    document_id: str
    page_number: int = Field(ge=1)
    bounding_box: PageBox
    caption: str | None = None
    rows: list[list[str]]
    markdown: str


class OCRToken(BaseModel):
    text: str
    confidence: float = Field(ge=-1, le=100)
    bounding_box: PageBox


class VisualAssetRecord(BaseModel):
    asset_id: str
    document_id: str
    page_number: int = Field(ge=1)
    kind: AssetKind
    bounding_box: PageBox
    image_path: str
    caption: str | None = None
    ocr_text: str | None = None
    ocr_tokens: list[OCRToken] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    chunk_id: str
    parent_id: str | None = None
    document_id: str
    document_version: str
    corpus_version: str
    page_number: int = Field(ge=1)
    kind: ChunkKind
    modality: ContentModality
    text: str = Field(min_length=1)
    table_id: str | None = None
    asset_id: str | None = None


class CorpusBuild(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    corpus_id: str
    corpus_version: str
    source_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    ingestion_config: IngestionConfig
    document_count: int = Field(ge=1)
    unique_document_count: int = Field(ge=1)
    page_count: int = Field(ge=1)
    table_count: int = Field(ge=0)
    visual_asset_count: int = Field(ge=0)
    parent_chunk_count: int = Field(ge=0)
    child_chunk_count: int = Field(ge=0)
