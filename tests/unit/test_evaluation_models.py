"""Evaluation example verification-rule tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from adaspilot.domain.evaluation import EvaluationExample


def test_draft_answerable_example_does_not_fake_ground_truth() -> None:
    example = EvaluationExample(
        question_id="q_example_draft",
        question="What sensor is described in the selected source document?",
        question_type="factual",
        verification_status="draft",
        should_answer=True,
        expected_response_status="answered",
        expected_execution_mode="direct",
        required_modalities={"text"},
    )

    assert example.reference_answer is None
    assert example.evidence == []


def test_verified_answer_requires_reference_and_evidence() -> None:
    with pytest.raises(ValidationError, match="reference_answer"):
        EvaluationExample(
            question_id="q_verified_missing",
            question="What sensor is described in the selected source document?",
            question_type="factual",
            verification_status="verified",
            should_answer=True,
            expected_response_status="answered",
            expected_execution_mode="direct",
            required_modalities={"text"},
            grading_notes=["Must identify the source-supported sensor."],
            verified_by="reviewer",
            verified_at=datetime.now(UTC),
        )


def test_verified_visual_example_requires_asset_or_box() -> None:
    with pytest.raises(ValidationError, match="asset_id or bounding_box"):
        EvaluationExample(
            question_id="q_visual_missing_asset",
            question="Which modules are connected in the architecture figure?",
            question_type="visual",
            verification_status="verified",
            should_answer=True,
            expected_response_status="answered",
            expected_execution_mode="direct",
            required_modalities={"visual"},
            reference_answer="The detector is connected to the tracker.",
            evidence=[{"document_id": "doc_one", "page_number": 3}],
            grading_notes=["Must describe the connected modules."],
            verified_by="reviewer",
            verified_at=datetime.now(UTC),
        )


def test_unanswerable_example_must_abstain() -> None:
    with pytest.raises(ValidationError, match="unanswerable"):
        EvaluationExample(
            question_id="q_bad_unanswerable",
            question="What is an answer that the corpus does not contain?",
            question_type="unanswerable",
            should_answer=True,
            expected_response_status="answered",
            expected_execution_mode="direct",
        )
