"""Small HTTP request and corpus-status contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from adaspilot.domain.serving import ExecutionMode


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    mode: ExecutionMode = ExecutionMode.AUTO


class CorpusDocumentStatus(BaseModel):
    document_id: str
    title: str
    page_count: int | None
    status: str


class CorpusStatusResponse(BaseModel):
    corpus_id: str
    corpus_version: str
    document_count: int
    page_count: int
    table_count: int
    visual_asset_count: int
    parent_chunk_count: int
    child_chunk_count: int
    staged_upload_count: int
    documents: list[CorpusDocumentStatus]


class UploadResponse(BaseModel):
    filename: str
    stored_name: str
    sha256: str
    page_count: int
    status: str
    next_step: str


class DocumentActionResponse(BaseModel):
    document_id: str
    status: str


class IngestStagedResponse(BaseModel):
    ingested: list[str] = Field(default_factory=list)
    reactivated: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    corpus_version: str | None = None


class IngestionJobResponse(BaseModel):
    status: Literal["idle", "running", "succeeded", "failed"] = "idle"
    message: str = "No corpus processing job has been started."
    ingested: list[str] = Field(default_factory=list)
    reactivated: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    corpus_version: str | None = None


class CitationResponse(BaseModel):
    citation: str
    doi_url: str | None = None
