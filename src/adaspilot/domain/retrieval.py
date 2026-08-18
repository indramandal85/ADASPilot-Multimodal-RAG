"""Small contracts shared by text and visual retrieval."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"


class SearchHit(BaseModel):
    record_id: str
    score: float
    document_id: str
    page_number: int = Field(ge=0)
    text: str = ""
    parent_id: str | None = None
    asset_id: str | None = None
    image_path: str | None = None
    source: str


class AnswerCitation(BaseModel):
    document_id: str
    page_number: int = Field(ge=0)
    record_id: str
    asset_id: str | None = None


class StructuredAnswer(BaseModel):
    answer: str
    status: AnswerStatus
    citations: list[AnswerCitation] = Field(default_factory=list)
    retrieval_mode: str


class RetrievalMetrics(BaseModel):
    query_count: int = Field(ge=0)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)


class AnswerMetrics(BaseModel):
    answer_count: int = Field(ge=0)
    token_f1: float = Field(ge=0, le=1)
    citation_recall: float = Field(ge=0, le=1)
    abstention_accuracy: float = Field(ge=0, le=1)


class ExperimentResult(BaseModel):
    experiment_id: Literal["E0", "E1", "E2", "E3", "E4"]
    description: str
    metrics: RetrievalMetrics
    delta_recall_at_5: float
    retained: bool


class VisualMetrics(BaseModel):
    query_count: int = Field(ge=0)
    visual_recall_at_1: float = Field(ge=0, le=1)
    visual_recall_at_5: float = Field(ge=0, le=1)
    multimodal_faithfulness: float = Field(ge=0, le=1)
