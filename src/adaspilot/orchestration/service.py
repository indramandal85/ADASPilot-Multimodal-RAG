"""One public execution service used by both evaluation and FastAPI."""

from __future__ import annotations

import time

from adaspilot.domain.orchestration import ResearchMode, ToolName
from adaspilot.domain.serving import (
    ExecutionMode,
    ExecutionRecord,
    TimelineItem,
)
from adaspilot.orchestration.router import ExecutionRouter
from adaspilot.orchestration.runtime import WorkflowBundle


class WorkflowRunner:
    def __init__(self, bundle: WorkflowBundle, router: ExecutionRouter | None = None) -> None:
        self.bundle = bundle
        self.router = router or ExecutionRouter()

    def run(
        self,
        question: str,
        requested_mode: ExecutionMode = ExecutionMode.AUTO,
    ) -> ExecutionRecord:
        mode = (
            self.router.route(question) if requested_mode == ExecutionMode.AUTO else requested_mode
        )
        started = time.perf_counter()
        if mode == ExecutionMode.DIRECT:
            direct_result = self.bundle.direct.run(question)
            timeline = [
                TimelineItem(
                    label=item.node,
                    status=item.outcome,
                    duration_ms=item.duration_ms,
                )
                for item in direct_result.trace
            ]
            return ExecutionRecord(
                question=question,
                requested_mode=requested_mode,
                mode=mode,
                answer=direct_result.answer,
                evidence=direct_result.evidence,
                timeline=timeline,
                terminal_reason=direct_result.terminal_reason,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if mode == ExecutionMode.REACT:
            react_result = self.bundle.react.run(question)
            timeline = [
                TimelineItem(
                    label=item.tool_name.value,
                    status="evidence" if item.new_evidence_count else "no_gain",
                    detail=item.message,
                    duration_ms=item.duration_ms,
                )
                for item in react_result.observations
            ]
            return ExecutionRecord(
                question=question,
                requested_mode=requested_mode,
                mode=mode,
                answer=react_result.answer,
                evidence=react_result.evidence,
                timeline=timeline,
                terminal_reason=react_result.terminal_reason,
                latency_ms=react_result.latency_ms,
                tool_names=[item.tool_name.value for item in react_result.observations],
                tool_call_count=react_result.tool_call_count,
                repeated_step_count=react_result.redundant_action_count,
            )

        plan_result = self.bundle.plan.run(question)
        completed = sum(item.completed for item in plan_result.task_results)
        total = len(plan_result.plan.tasks)
        timeline = [
            TimelineItem(
                label=task.task_id,
                status=(
                    "complete"
                    if any(
                        item.task_id == task.task_id and item.completed
                        for item in plan_result.task_results
                    )
                    else "incomplete"
                ),
                detail=task.objective,
            )
            for task in plan_result.plan.tasks
        ]
        tool_map = {
            ResearchMode.TEXT: [ToolName.HYBRID_TEXT_SEARCH.value],
            ResearchMode.VISUAL: [ToolName.VISUAL_SEARCH.value],
            ResearchMode.BOTH: [
                ToolName.HYBRID_TEXT_SEARCH.value,
                ToolName.VISUAL_SEARCH.value,
            ],
            ResearchMode.METADATA: [ToolName.DOCUMENT_METADATA.value],
            ResearchMode.CALCULATION: [ToolName.CALCULATOR.value],
        }
        tool_names = [name for task in plan_result.plan.tasks for name in tool_map[task.mode]]
        return ExecutionRecord(
            question=question,
            requested_mode=requested_mode,
            mode=mode,
            answer=plan_result.answer,
            evidence=plan_result.evidence,
            timeline=timeline,
            terminal_reason=plan_result.terminal_reason,
            latency_ms=plan_result.latency_ms,
            tool_names=tool_names,
            tool_call_count=sum(item.tool_calls for item in plan_result.task_results),
            plan_valid=plan_result.terminal_reason != "invalid_plan",
            task_completion_rate=completed / total if total else 0.0,
        )
