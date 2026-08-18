"""Corpus-manifest contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Identifier = str


class SourceKind(StrEnum):
    LOCAL_FILE = "local_file"
    PUBLIC_URL = "public_url"


class DocumentStatus(StrEnum):
    PLANNED = "planned"
    DOWNLOADED = "downloaded"
    VERIFIED = "verified"
    INACTIVE = "inactive"


class CorpusDocument(BaseModel):
    """One source document selected for the ADAS benchmark corpus."""

    document_id: Identifier = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    title: str = Field(min_length=3, max_length=300)
    source_kind: SourceKind
    source_uri: str = Field(min_length=1)
    publisher: str | None = Field(default=None, max_length=200)
    publication_year: int | None = Field(default=None, ge=1900, le=2100)
    license: str | None = Field(default=None, max_length=200)
    expected_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    page_count: int | None = Field(default=None, ge=1)
    status: DocumentStatus = DocumentStatus.PLANNED
    notes: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def verified_document_has_hash(self) -> CorpusDocument:
        if self.status == DocumentStatus.VERIFIED:
            if self.expected_sha256 is None:
                raise ValueError("a verified corpus document must include expected_sha256")
            if self.page_count is None:
                raise ValueError("a verified corpus document must include page_count")
        return self


class CorpusManifest(BaseModel):
    """Versioned collection of source documents."""

    schema_version: Literal["1.0"] = "1.0"
    corpus_id: Identifier = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    domain: Literal["adas"] = "adas"
    description: str = Field(min_length=10, max_length=1000)
    documents: list[CorpusDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def document_ids_are_unique(self) -> CorpusManifest:
        ids = [document.document_id for document in self.documents]
        duplicates = sorted({document_id for document_id in ids if ids.count(document_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate document_id values: {duplicates}")
        return self
