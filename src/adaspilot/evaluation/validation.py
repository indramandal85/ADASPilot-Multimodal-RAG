"""Cross-record validation and benchmark completion reporting."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from adaspilot.domain.corpus import CorpusManifest, DocumentStatus
from adaspilot.domain.evaluation import (
    BenchmarkTargets,
    EvaluationExample,
    QuestionType,
    VerificationStatus,
)


class ValidationIssue(BaseModel):
    level: Literal["warning", "error"]
    code: str
    message: str
    record_id: str | None = None


class DataValidationReport(BaseModel):
    valid: bool
    document_count: int
    example_count: int
    verified_count: int
    draft_count: int
    rejected_count: int
    category_counts: dict[QuestionType, int]
    verified_category_counts: dict[QuestionType, int]
    issues: list[ValidationIssue] = Field(default_factory=list)


def _target_issue_level(require_target: bool) -> Literal["warning", "error"]:
    return "error" if require_target else "warning"


def validate_benchmark(
    manifest: CorpusManifest,
    examples: list[EvaluationExample],
    targets: BenchmarkTargets,
    *,
    require_target: bool = False,
) -> DataValidationReport:
    """Validate benchmark integrity and report progress toward verified targets."""

    issues: list[ValidationIssue] = []
    documents = {document.document_id: document for document in manifest.documents}
    question_ids = [example.question_id for example in examples]
    duplicates = sorted(
        question_id for question_id, count in Counter(question_ids).items() if count > 1
    )
    for question_id in duplicates:
        issues.append(
            ValidationIssue(
                level="error",
                code="duplicate_question_id",
                message="question_id appears more than once",
                record_id=question_id,
            )
        )

    normalized_questions = [" ".join(example.question.lower().split()) for example in examples]
    duplicate_questions = {
        question for question, count in Counter(normalized_questions).items() if count > 1
    }
    for example, normalized in zip(examples, normalized_questions, strict=True):
        if normalized in duplicate_questions:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="duplicate_question_text",
                    message="question text duplicates another benchmark example",
                    record_id=example.question_id,
                )
            )

    if not manifest.documents:
        issues.append(
            ValidationIssue(
                level="warning",
                code="empty_corpus",
                message="corpus manifest has no selected source documents yet",
            )
        )

    for example in examples:
        for evidence in example.evidence:
            document = documents.get(evidence.document_id)
            if document is None:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="unknown_document_reference",
                        message=f"evidence references unknown document {evidence.document_id!r}",
                        record_id=example.question_id,
                    )
                )
                continue
            if example.verification_status == VerificationStatus.VERIFIED:
                if document.status != DocumentStatus.VERIFIED:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="unverified_document_reference",
                            message="verified example references a document that is not verified",
                            record_id=example.question_id,
                        )
                    )
                if document.page_count is not None and evidence.page_number > document.page_count:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="page_number_out_of_range",
                            message=(
                                f"page {evidence.page_number} exceeds document page count "
                                f"{document.page_count}"
                            ),
                            record_id=example.question_id,
                        )
                    )

    status_counts = Counter(example.verification_status for example in examples)
    category_counts = Counter(example.question_type for example in examples)
    verified_examples = [
        example
        for example in examples
        if example.verification_status == VerificationStatus.VERIFIED
    ]
    verified_category_counts = Counter(example.question_type for example in verified_examples)
    verified_count = len(verified_examples)
    target_level = _target_issue_level(require_target)

    if verified_count < targets.minimum_verified_total:
        issues.append(
            ValidationIssue(
                level=target_level,
                code="verified_total_below_minimum",
                message=(
                    f"verified examples {verified_count} are below the minimum "
                    f"target {targets.minimum_verified_total}"
                ),
            )
        )
    if verified_count > targets.maximum_verified_total:
        issues.append(
            ValidationIssue(
                level="error",
                code="verified_total_above_maximum",
                message=(
                    f"verified examples {verified_count} exceed the maximum "
                    f"target {targets.maximum_verified_total}"
                ),
            )
        )

    for question_type, target in targets.categories.items():
        current = verified_category_counts[question_type]
        if current < target.minimum:
            issues.append(
                ValidationIssue(
                    level=target_level,
                    code="verified_category_below_minimum",
                    message=(
                        f"verified {question_type.value} examples {current} are below "
                        f"the minimum target {target.minimum}"
                    ),
                    record_id=question_type.value,
                )
            )

    report_category_counts = {
        question_type: category_counts[question_type] for question_type in QuestionType
    }
    report_verified_counts = {
        question_type: verified_category_counts[question_type] for question_type in QuestionType
    }
    return DataValidationReport(
        valid=not any(issue.level == "error" for issue in issues),
        document_count=len(manifest.documents),
        example_count=len(examples),
        verified_count=verified_count,
        draft_count=status_counts[VerificationStatus.DRAFT],
        rejected_count=status_counts[VerificationStatus.REJECTED],
        category_counts=report_category_counts,
        verified_category_counts=report_verified_counts,
        issues=issues,
    )
