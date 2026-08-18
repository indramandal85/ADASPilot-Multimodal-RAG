"""Module 8 planning, parallel research, replan, conflict, and partial-result tests."""

from __future__ import annotations

import threading
import time

from adaspilot.domain.orchestration import (
    EvidenceItem,
    ResearchMode,
    ResearchPlan,
    ResearchTask,
    TaskResult,
    WorkflowStatus,
)
from adaspilot.domain.retrieval import AnswerStatus
from adaspilot.evaluation.agents import evaluate_plan_run
from adaspilot.orchestration.planning import (
    PlanExecuteWorkflow,
    RuleBasedPlanner,
    detect_conflicts,
)
from adaspilot.retrieval.answers import ExtractiveAnswerer
from adaspilot.safety.guardrails import InputGuard
from adaspilot.safety.plan_validation import PlanValidator


class ScriptedPlanner:
    def __init__(self, plan: ResearchPlan) -> None:
        self.plan = plan

    def create_plan(self, question: str) -> ResearchPlan:
        return self.plan.model_copy(update={"question": question})

    def replan(self, plan: ResearchPlan, failed_task_ids: list[str]) -> ResearchPlan:
        return plan.model_copy(update={"revision": 1})


class FakeResearcher:
    def __init__(self, fail_counts: dict[str, int] | None = None, delay: float = 0) -> None:
        self.fail_counts = fail_counts or {}
        self.calls: dict[str, int] = {}
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.dependency_counts: dict[str, int] = {}

    def run(
        self, task: ResearchTask, dependency_evidence: list[EvidenceItem]
    ) -> tuple[TaskResult, list[EvidenceItem]]:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            self.dependency_counts[task.task_id] = len(dependency_evidence)
            call_number = self.calls.get(task.task_id, 0) + 1
            self.calls[task.task_id] = call_number
            if call_number <= self.fail_counts.get(task.task_id, 0):
                return (
                    TaskResult(
                        task_id=task.task_id,
                        completed=False,
                        tool_calls=1,
                        message="No evidence found",
                    ),
                    [],
                )
            evidence = EvidenceItem(
                evidence_id=f"evidence_{task.task_id}",
                document_id="sample_doc",
                page_number=call_number,
                text=(
                    f"Evidence for {task.objective} explains an ADAS finding with enough "
                    "detail for cited synthesis."
                ),
                source="fake_research",
                tool_name="hybrid_text_search",
                task_id=task.task_id,
                record_id=f"evidence_{task.task_id}",
                support_score=0.9,
            )
            return (
                TaskResult(
                    task_id=task.task_id,
                    completed=True,
                    evidence_ids=[evidence.evidence_id],
                    tool_calls=1,
                    message="Evidence collected",
                ),
                [evidence],
            )
        finally:
            with self.lock:
                self.active -= 1


def _task(identifier: str, dependencies: list[str] | None = None) -> ResearchTask:
    return ResearchTask(
        task_id=identifier,
        objective=f"Research the ADAS evidence required for {identifier}.",
        mode=ResearchMode.TEXT,
        dependencies=dependencies or [],
    )


def _workflow(
    plan: ResearchPlan,
    researcher: FakeResearcher,
    *,
    max_tasks: int = 5,
) -> PlanExecuteWorkflow:
    return PlanExecuteWorkflow(
        planner=ScriptedPlanner(plan),
        researcher=researcher,  # type: ignore[arg-type]
        answerer=ExtractiveAnswerer(),
        input_guard=InputGuard(500),
        plan_validator=PlanValidator(max_tasks=max_tasks, max_depth=3),
        max_parallel_workers=2,
        max_replans=1,
    )


def test_independent_tasks_run_in_bounded_parallel() -> None:
    plan = ResearchPlan(
        question="Compare two systems",
        tasks=[_task("task_one"), _task("task_two"), _task("task_three")],
    )
    researcher = FakeResearcher(delay=0.02)

    result = _workflow(plan, researcher).run("Compare radar, camera, and fusion systems.")
    metrics = evaluate_plan_run(result)

    assert result.status == WorkflowStatus.ANSWERED
    assert result.answer.status == AnswerStatus.ANSWERED
    assert 1 < researcher.max_active <= 2
    assert len(result.task_results) == 3
    assert metrics.task_completion_rate == 1
    assert metrics.required_evidence_coverage == 1
    assert metrics.citation_valid


def test_dependency_order_passes_prior_evidence() -> None:
    plan = ResearchPlan(
        question="Research then calculate",
        tasks=[_task("task_source"), _task("task_followup", ["task_source"])],
    )
    researcher = FakeResearcher()

    result = _workflow(plan, researcher).run("Research a source and then follow it up.")

    assert result.status == WorkflowStatus.ANSWERED
    assert researcher.dependency_counts["task_source"] == 0
    assert researcher.dependency_counts["task_followup"] == 1


def test_failed_task_replans_once_then_succeeds() -> None:
    plan = ResearchPlan(question="Retry research", tasks=[_task("task_retry")])
    researcher = FakeResearcher(fail_counts={"task_retry": 1})

    result = _workflow(plan, researcher).run("Research a difficult ADAS finding.")

    assert result.status == WorkflowStatus.ANSWERED
    assert result.replan_count == 1
    assert researcher.calls["task_retry"] == 2


def test_replan_exhaustion_returns_partial_or_abstains() -> None:
    plan = ResearchPlan(
        question="Partial research",
        tasks=[_task("task_good"), _task("task_fail")],
    )
    researcher = FakeResearcher(fail_counts={"task_fail": 5})

    result = _workflow(plan, researcher).run("Compare available and missing evidence.")

    assert result.status == WorkflowStatus.PARTIAL
    assert result.replan_count == 1
    assert result.terminal_reason == "partial_evidence"


def test_cyclic_and_oversized_plans_are_blocked() -> None:
    cycle = ResearchPlan(
        question="Cycle",
        tasks=[_task("task_a", ["task_b"]), _task("task_b", ["task_a"])],
    )
    oversized = ResearchPlan(
        question="Explosion",
        tasks=[_task(f"task_{index}") for index in range(6)],
    )

    cycle_result = _workflow(cycle, FakeResearcher()).run("Run a cyclic research plan.")
    size_result = _workflow(oversized, FakeResearcher()).run("Run an oversized research plan.")

    assert cycle_result.terminal_reason == "invalid_plan"
    assert size_result.status == WorkflowStatus.BLOCKED
    assert not cycle_result.task_results


def test_conflict_detection_and_rule_planner() -> None:
    evidence = [
        EvidenceItem(
            evidence_id="negative",
            document_id="doc_one",
            page_number=1,
            text="The radar system cannot identify colour.",
            source="test",
            tool_name="search",
            task_id="task_claim",
            record_id="negative",
        ),
        EvidenceItem(
            evidence_id="positive",
            document_id="doc_two",
            page_number=2,
            text="The radar system can identify colour.",
            source="test",
            tool_name="search",
            task_id="task_claim",
            record_id="positive",
        ),
    ]
    planner = RuleBasedPlanner()

    conflicts = detect_conflicts(evidence)
    comparison = planner.create_plan("Compare radar and camera.")
    visual = planner.create_plan("Explain the architecture figure and its caption.")
    revised = planner.replan(comparison, ["task_left"])

    assert len(conflicts) == 1
    assert len(comparison.tasks) == 2
    assert {task.mode for task in visual.tasks} == {ResearchMode.TEXT, ResearchMode.VISUAL}
    assert revised.revision == 1
