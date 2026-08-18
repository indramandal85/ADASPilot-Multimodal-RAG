"""Unified serving records for all three workflow implementations."""

from __future__ import annotations

from typing import cast

from adaspilot.domain.orchestration import (
    DirectResult,
    EvidenceItem,
    NodeTrace,
    PlanResult,
    QueryKind,
    ReactResult,
    ResearchMode,
    ResearchPlan,
    ResearchTask,
    TaskResult,
    ToolName,
    ToolObservation,
    WorkflowStatus,
)
from adaspilot.domain.retrieval import AnswerStatus, StructuredAnswer
from adaspilot.domain.serving import ExecutionMode
from adaspilot.observability.usage import UsageTracker
from adaspilot.orchestration.runtime import WorkflowBundle
from adaspilot.orchestration.service import WorkflowRunner


def _answer(mode: str) -> StructuredAnswer:
    return StructuredAnswer(
        answer="Grounded answer.", status=AnswerStatus.ANSWERED, retrieval_mode=mode
    )


def _evidence() -> EvidenceItem:
    return EvidenceItem(
        evidence_id="chunk_1",
        document_id="radar_doc",
        page_number=1,
        text="Evidence.",
        source="text",
        tool_name="hybrid_text_search",
        record_id="chunk_1",
    )


class FakeDirect:
    def run(self, question: str) -> DirectResult:
        return DirectResult(
            question=question,
            query_kind=QueryKind.FACT,
            retry_count=0,
            evidence=[_evidence()],
            answer=_answer("direct"),
            trace=[NodeTrace(node="retrieve", outcome="1 hit", duration_ms=1)],
            terminal_reason="verified_answer",
        )


class FakeReact:
    def run(self, question: str) -> ReactResult:
        return ReactResult(
            question=question,
            status=WorkflowStatus.ANSWERED,
            answer=_answer("react"),
            evidence=[_evidence()],
            observations=[
                ToolObservation(
                    tool_name=ToolName.HYBRID_TEXT_SEARCH,
                    signature="search",
                    evidence_ids=["chunk_1"],
                    new_evidence_count=1,
                    message="Found evidence",
                    duration_ms=2,
                )
            ],
            tool_call_count=1,
            redundant_action_count=0,
            terminal_reason="finished_with_evidence",
            latency_ms=3,
        )


class FakePlan:
    def run(self, question: str) -> PlanResult:
        task = ResearchTask(
            task_id="task_text",
            objective="Find grounded radar evidence.",
            mode=ResearchMode.BOTH,
        )
        return PlanResult(
            question=question,
            status=WorkflowStatus.ANSWERED,
            plan=ResearchPlan(question=question, tasks=[task]),
            task_results=[
                TaskResult(
                    task_id="task_text",
                    completed=True,
                    evidence_ids=["chunk_1"],
                    tool_calls=2,
                    message="complete",
                )
            ],
            evidence=[_evidence()],
            conflicts=[],
            answer=_answer("plan"),
            replan_count=0,
            trace=[],
            terminal_reason="verified_complete_answer",
            latency_ms=4,
        )


class FakeBundle:
    direct = FakeDirect()
    react = FakeReact()
    plan = FakePlan()
    usage_tracker = UsageTracker()
    corpus_version = "corpus_test"


def test_workflow_runner_normalizes_direct_react_and_plan_results() -> None:
    runner = WorkflowRunner(cast(WorkflowBundle, FakeBundle()))

    direct = runner.run("What does radar measure?", ExecutionMode.DIRECT)
    react = runner.run("Calculate the gain.", ExecutionMode.REACT)
    plan = runner.run("Compare radar and camera.", ExecutionMode.PLAN)

    assert direct.timeline[0].label == "retrieve"
    assert react.tool_names == ["hybrid_text_search"]
    assert react.timeline[0].status == "evidence"
    assert plan.plan_valid is True
    assert plan.task_completion_rate == 1.0
    assert plan.tool_names == ["hybrid_text_search", "visual_search"]
