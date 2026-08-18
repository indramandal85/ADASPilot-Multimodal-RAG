"""Module 10 report aggregation tests over deterministic fake runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from adaspilot.config import Settings
from adaspilot.domain.evaluation import (
    BenchmarkSplit,
    EvaluationExample,
    EvidenceReference,
    ExpectedExecutionMode,
    ExpectedResponseStatus,
    Modality,
    QuestionType,
    VerificationStatus,
)
from adaspilot.domain.orchestration import EvidenceItem
from adaspilot.domain.retrieval import AnswerCitation, AnswerStatus, StructuredAnswer
from adaspilot.domain.serving import ExecutionMode, ExecutionRecord
from adaspilot.evaluation.system import SystemEvaluator
from adaspilot.observability.usage import UsageTracker
from adaspilot.orchestration.service import WorkflowRunner


class FakeBundle:
    corpus_version = "corpus_test"
    usage_tracker = UsageTracker()


class FakeRunner:
    bundle = FakeBundle()

    def run(self, question: str, requested_mode: ExecutionMode) -> ExecutionRecord:
        mode = requested_mode if requested_mode != ExecutionMode.AUTO else ExecutionMode.DIRECT
        evidence = [] if "unknown" in question else [_evidence()]
        should_abstain = "unknown" in question
        return ExecutionRecord(
            question=question,
            requested_mode=requested_mode,
            mode=mode,
            answer=StructuredAnswer(
                answer="Insufficient evidence." if should_abstain else "Radar measures range.",
                status=AnswerStatus.ABSTAINED if should_abstain else AnswerStatus.ANSWERED,
                citations=(
                    []
                    if should_abstain
                    else [
                        AnswerCitation(
                            document_id="radar_doc",
                            page_number=2,
                            record_id="chunk_1",
                        )
                    ]
                ),
                retrieval_mode=mode.value,
            ),
            evidence=evidence,
            timeline=[],
            terminal_reason="done",
            latency_ms=10.0,
            tool_names=(
                ["hybrid_text_search", "evidence_linked_calculator"]
                if mode == ExecutionMode.REACT
                else (["hybrid_text_search"] if mode == ExecutionMode.PLAN else [])
            ),
            tool_call_count=2 if mode == ExecutionMode.REACT else int(mode == ExecutionMode.PLAN),
            plan_valid=True if mode == ExecutionMode.PLAN else None,
            task_completion_rate=1.0 if mode == ExecutionMode.PLAN else None,
        )


def _evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="chunk_1",
        document_id="radar_doc",
        page_number=2,
        text="Automotive radar measures range and relative velocity.",
        source="text",
        tool_name="hybrid_text_search",
        record_id="chunk_1",
        support_score=0.9,
    )


def _example(
    question_id: str,
    question: str,
    mode: ExpectedExecutionMode,
    question_type: QuestionType,
    modalities: set[Modality],
    *,
    should_answer: bool = True,
) -> EvaluationExample:
    return EvaluationExample(
        question_id=question_id,
        question=question,
        question_type=question_type,
        split=BenchmarkSplit.DEVELOPMENT,
        verification_status=VerificationStatus.VERIFIED,
        should_answer=should_answer,
        expected_response_status=(
            ExpectedResponseStatus.ANSWERED if should_answer else ExpectedResponseStatus.ABSTAINED
        ),
        expected_execution_mode=mode,
        required_modalities=modalities,
        reference_answer="Radar measures range." if should_answer else None,
        grading_notes=["Verified against the page."],
        evidence=(
            [
                EvidenceReference(
                    document_id="radar_doc",
                    page_number=2,
                    supporting_excerpt="Radar measures range.",
                )
            ]
            if should_answer
            else []
        ),
        verified_by="tester",
        verified_at=datetime.now(UTC),
    )


def test_system_evaluator_separates_modes_and_scores_core_metrics(tmp_path: Path) -> None:
    examples = [
        _example(
            "q_direct_case",
            "What does automotive radar measure?",
            ExpectedExecutionMode.DIRECT,
            QuestionType.FACTUAL,
            {Modality.TEXT},
        ),
        _example(
            "q_react_case",
            "Calculate 10 + 5 using the radar table.",
            ExpectedExecutionMode.REACT,
            QuestionType.MULTI_STEP,
            {Modality.TEXT, Modality.CALCULATION},
        ),
        _example(
            "q_plan_case",
            "Compare radar and camera strengths.",
            ExpectedExecutionMode.PLAN_EXECUTE,
            QuestionType.COMPARISON,
            {Modality.TEXT},
        ),
        _example(
            "q_unknown_case",
            "What unknown value is missing from this corpus?",
            ExpectedExecutionMode.DIRECT,
            QuestionType.UNANSWERABLE,
            set(),
            should_answer=False,
        ),
    ]
    evaluator = SystemEvaluator(
        Settings(_env_file=None),
        cast(WorkflowRunner, FakeRunner()),
        build_dir=tmp_path,
    )

    report = evaluator.evaluate(examples, split="development")

    assert report.router.accuracy == 1.0
    assert report.modes[ExecutionMode.DIRECT].example_count == 2
    assert report.modes[ExecutionMode.DIRECT].text_recall_at_5 == 1.0
    assert report.modes[ExecutionMode.DIRECT].abstention_f1 == 1.0
    assert report.modes[ExecutionMode.REACT].tool_call_f1 == 1.0
    assert report.modes[ExecutionMode.PLAN].plan_validity == 1.0
    assert report.modes[ExecutionMode.PLAN].task_completion == 1.0
    assert report.modes[ExecutionMode.DIRECT].ragas.status.value == "not_configured"
    assert report.modes[ExecutionMode.DIRECT].p95_latency_ms == pytest.approx(10.0)
