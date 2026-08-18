"""Deterministic operational metrics for bounded agent runs."""

from __future__ import annotations

from collections.abc import Sequence

from adaspilot.domain.orchestration import (
    AgentRunMetrics,
    PlanResult,
    PlanRunMetrics,
    ReactResult,
    ToolName,
    WorkflowStatus,
)
from adaspilot.safety.guardrails import CitationValidator


def evaluate_react_run(
    result: ReactResult,
    expected_tools: Sequence[ToolName] | None = None,
) -> AgentRunMetrics:
    call_count = result.tool_call_count
    successful_calls = sum(
        not observation.message.startswith("Tool failed safely")
        for observation in result.observations
    )
    evidence_gain = sum(observation.new_evidence_count for observation in result.observations)
    if expected_tools is None:
        tool_accuracy = successful_calls / call_count if call_count else 0
    else:
        actual = [observation.tool_name for observation in result.observations]
        comparison_length = max(len(actual), len(expected_tools), 1)
        correct = sum(
            actual[index] == expected_tools[index]
            for index in range(min(len(actual), len(expected_tools)))
        )
        tool_accuracy = correct / comparison_length
    return AgentRunMetrics(
        completed=result.status == WorkflowStatus.ANSWERED,
        tool_call_count=call_count,
        tool_call_accuracy=tool_accuracy,
        redundant_action_rate=(result.redundant_action_count / call_count if call_count else 0),
        evidence_gain_per_call=evidence_gain / call_count if call_count else 0,
        latency_ms=result.latency_ms,
        model_call_count=0,
        estimated_cost_usd=0,
    )


def evaluate_plan_run(result: PlanResult) -> PlanRunMetrics:
    total = len(result.plan.tasks)
    completed_ids = {task.task_id for task in result.task_results if task.completed}
    required = {task.task_id for task in result.plan.tasks if task.required}
    evidenced_tasks = {item.task_id for item in result.evidence if item.task_id}
    citation_issues = CitationValidator().validate(result.answer, result.evidence)
    return PlanRunMetrics(
        plan_valid=result.terminal_reason != "invalid_plan",
        task_completion_rate=len(completed_ids) / total if total else 0,
        required_evidence_coverage=(
            len(required & evidenced_tasks) / len(required) if required else 1
        ),
        citation_valid=not citation_issues,
        conflict_count=len(result.conflicts),
        replan_count=result.replan_count,
        latency_ms=result.latency_ms,
        model_call_count=0,
        estimated_cost_usd=0,
    )
