"""Evaluation benchmark contracts and verification rules."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QuestionType(StrEnum):
    FACTUAL = "factual"
    TABLE = "table"
    VISUAL = "visual"
    COMPARISON = "comparison"
    MULTI_STEP = "multi_step"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"


class VerificationStatus(StrEnum):
    DRAFT = "draft"
    VERIFIED = "verified"
    REJECTED = "rejected"


class ExpectedExecutionMode(StrEnum):
    DIRECT = "direct"
    REACT = "react"
    PLAN_EXECUTE = "plan_execute"


class ExpectedResponseStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"


class Modality(StrEnum):
    TEXT = "text"
    TABLE = "table"
    VISUAL = "visual"
    METADATA = "metadata"
    CALCULATION = "calculation"


class BenchmarkSplit(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"


class BoundingBox(BaseModel):
    """Bounding box in source-page coordinates."""

    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(gt=0)
    y1: float = Field(gt=0)

    @model_validator(mode="after")
    def coordinates_are_ordered(self) -> BoundingBox:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding-box maximums must exceed minimums")
        return self


class EvidenceReference(BaseModel):
    """Page-level source evidence for an evaluation example."""

    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    page_number: int = Field(ge=1)
    asset_id: str | None = Field(default=None, min_length=3, max_length=128)
    bounding_box: BoundingBox | None = None
    supporting_excerpt: str | None = Field(default=None, min_length=3, max_length=2000)


class EvaluationExample(BaseModel):
    """One benchmark question and its human-verification state."""

    schema_version: Literal["1.0"] = "1.0"
    question_id: str = Field(pattern=r"^q_[a-z0-9][a-z0-9_-]{2,63}$")
    question: str = Field(min_length=8, max_length=2000)
    question_type: QuestionType
    split: BenchmarkSplit = BenchmarkSplit.DEVELOPMENT
    verification_status: VerificationStatus = VerificationStatus.DRAFT
    should_answer: bool
    expected_response_status: ExpectedResponseStatus
    expected_execution_mode: ExpectedExecutionMode
    required_modalities: set[Modality] = Field(default_factory=set)
    reference_answer: str | None = Field(default=None, min_length=3, max_length=5000)
    grading_notes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)
    verified_by: str | None = Field(default=None, min_length=2, max_length=100)
    verified_at: datetime | None = None

    @model_validator(mode="after")
    def validate_answer_contract(self) -> EvaluationExample:
        if self.should_answer and self.expected_response_status != ExpectedResponseStatus.ANSWERED:
            raise ValueError("answerable examples must expect an answered response")
        if (
            not self.should_answer
            and self.expected_response_status == ExpectedResponseStatus.ANSWERED
        ):
            raise ValueError("non-answerable examples cannot expect an answered response")
        if self.question_type == QuestionType.UNANSWERABLE and (
            self.should_answer or self.expected_response_status != ExpectedResponseStatus.ABSTAINED
        ):
            raise ValueError("unanswerable examples must expect abstention")
        if self.question_type == QuestionType.ADVERSARIAL and self.should_answer:
            raise ValueError("adversarial benchmark examples must not request an answer")
        return self

    @model_validator(mode="after")
    def validate_verified_example(self) -> EvaluationExample:
        if self.verification_status != VerificationStatus.VERIFIED:
            return self

        if self.verified_by is None or self.verified_at is None:
            raise ValueError("verified examples require verified_by and verified_at")
        if not self.grading_notes:
            raise ValueError("verified examples require grading_notes")
        if self.should_answer:
            if self.reference_answer is None:
                raise ValueError("verified answerable examples require a reference_answer")
            if not self.evidence:
                raise ValueError("verified answerable examples require source evidence")
        else:
            if self.evidence:
                raise ValueError(
                    "verified non-answerable examples must not contain supporting evidence"
                )
            if self.reference_answer is not None:
                raise ValueError("verified non-answerable examples cannot have a reference_answer")

        if self.question_type == QuestionType.VISUAL and self.should_answer:
            if Modality.VISUAL not in self.required_modalities:
                raise ValueError("verified visual examples must require the visual modality")
            has_visual_reference = any(
                reference.asset_id is not None or reference.bounding_box is not None
                for reference in self.evidence
            )
            if not has_visual_reference:
                raise ValueError("verified visual examples require an asset_id or bounding_box")
        elif self.should_answer:
            has_excerpt = any(reference.supporting_excerpt for reference in self.evidence)
            if not has_excerpt:
                raise ValueError("verified text/table examples require a supporting_excerpt")
        return self


class CategoryTarget(BaseModel):
    """Required and recommended verified counts for one question category."""

    minimum: int = Field(ge=0)
    recommended: int = Field(ge=0)

    @model_validator(mode="after")
    def recommended_is_not_below_minimum(self) -> CategoryTarget:
        if self.recommended < self.minimum:
            raise ValueError("recommended target cannot be below minimum")
        return self


class BenchmarkTargets(BaseModel):
    """Versioned completion target for the project benchmark."""

    schema_version: Literal["1.0"] = "1.0"
    minimum_verified_total: int = Field(ge=1)
    maximum_verified_total: int = Field(ge=1)
    recommended_verified_total: int = Field(ge=1)
    categories: dict[QuestionType, CategoryTarget]

    @model_validator(mode="after")
    def totals_and_categories_are_consistent(self) -> BenchmarkTargets:
        if not (
            self.minimum_verified_total
            <= self.recommended_verified_total
            <= self.maximum_verified_total
        ):
            raise ValueError("verified totals must satisfy minimum <= recommended <= maximum")
        missing = set(QuestionType) - set(self.categories)
        if missing:
            raise ValueError(f"targets missing question categories: {sorted(missing)}")
        minimum_sum = sum(target.minimum for target in self.categories.values())
        recommended_sum = sum(target.recommended for target in self.categories.values())
        if minimum_sum != self.minimum_verified_total:
            raise ValueError("category minimums must sum to minimum_verified_total")
        if recommended_sum != self.recommended_verified_total:
            raise ValueError("category recommendations must sum to recommended_verified_total")
        return self
