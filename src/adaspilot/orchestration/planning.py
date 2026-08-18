"""Module 8: typed, bounded plan-and-execute research orchestration."""

from __future__ import annotations

import operator
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from adaspilot.domain.orchestration import (
    EvidenceConflict,
    EvidenceItem,
    NodeTrace,
    PlanResult,
    ResearchMode,
    ResearchPlan,
    ResearchTask,
    TaskResult,
    ToolCall,
    ToolName,
    WorkflowStatus,
)
from adaspilot.domain.retrieval import AnswerCitation, AnswerStatus, StructuredAnswer
from adaspilot.orchestration.react import _to_search_hits
from adaspilot.orchestration.tools import ProjectToolbox
from adaspilot.retrieval.answers import Answerer
from adaspilot.safety.guardrails import CitationValidator, GuardrailViolation, InputGuard
from adaspilot.safety.plan_validation import PlanValidationError, PlanValidator


class Planner(Protocol):
    def create_plan(self, question: str) -> ResearchPlan: ...

    def replan(self, plan: ResearchPlan, failed_task_ids: list[str]) -> ResearchPlan: ...


class RuleBasedPlanner:
    """Small local planner; structured-output LLM planners can use the same contract."""

    def create_plan(self, question: str) -> ResearchPlan:
        lowered = question.lower()
        comparison = re.search(
            r"compare\s+(.{2,80}?)\s+(?:and|versus|vs\.?)\s+(.{2,80}?)(?:[?.]|$)",
            question,
            re.I,
        )
        tasks: list[ResearchTask]
        if comparison:
            left, right = comparison.group(1).strip(), comparison.group(2).strip()
            tasks = [
                ResearchTask(
                    task_id="task_left",
                    objective=f"Find cited technical evidence about {left} for this comparison.",
                    mode=ResearchMode.TEXT,
                ),
                ResearchTask(
                    task_id="task_right",
                    objective=f"Find cited technical evidence about {right} for this comparison.",
                    mode=ResearchMode.TEXT,
                ),
            ]
        elif any(word in lowered for word in ["figure", "diagram", "plot", "image"]):
            tasks = [
                ResearchTask(
                    task_id="task_text",
                    objective=f"Find textual evidence needed to answer: {question}",
                    mode=ResearchMode.TEXT,
                ),
                ResearchTask(
                    task_id="task_visual",
                    objective=f"Find visual page or figure evidence needed to answer: {question}",
                    mode=ResearchMode.VISUAL,
                ),
            ]
        else:
            clauses = [part.strip(" ,.") for part in re.split(r"\band\b", question) if part.strip()]
            tasks = [
                ResearchTask(
                    task_id=f"task_{index}",
                    objective=f"Research with cited evidence: {clause}",
                    mode=ResearchMode.TEXT,
                )
                for index, clause in enumerate(clauses[:3], start=1)
            ]
        return ResearchPlan(question=question, tasks=tasks)

    def replan(self, plan: ResearchPlan, failed_task_ids: list[str]) -> ResearchPlan:
        failed = set(failed_task_ids)
        tasks = [
            task.model_copy(
                update={
                    "objective": (
                        f"{task.objective} Retry once using alternative ADAS "
                        "terminology and acronyms."
                        if task.task_id in failed
                        else task.objective
                    )
                }
            )
            for task in plan.tasks
        ]
        return plan.model_copy(update={"tasks": tasks, "revision": 1})


class PlanState(TypedDict):
    question: str
    plan: ResearchPlan
    task_results: list[TaskResult]
    evidence: list[EvidenceItem]
    conflicts: list[EvidenceConflict]
    answer: StructuredAnswer | None
    status: WorkflowStatus
    replan_count: int
    error: str | None
    terminal_reason: str
    trace: Annotated[list[NodeTrace], operator.add]


class Researcher:
    def __init__(self, toolbox: ProjectToolbox) -> None:
        self.toolbox = toolbox

    def run(
        self, task: ResearchTask, dependency_evidence: list[EvidenceItem]
    ) -> tuple[TaskResult, list[EvidenceItem]]:
        calls = self._calls_for_task(task, dependency_evidence)
        evidence: list[EvidenceItem] = []
        messages: list[str] = []
        for role, call in calls[:3]:
            try:
                result, _ = self.toolbox.registry.execute(role, call)
                messages.append(result.message)
                existing = {item.evidence_id for item in evidence}
                evidence.extend(
                    item.model_copy(update={"task_id": task.task_id})
                    for item in result.evidence
                    if item.evidence_id not in existing
                )
            except (GuardrailViolation, TimeoutError, OSError, ValueError) as exc:
                messages.append(f"Tool failed safely: {exc}")
        completed = bool(evidence) or (task.mode == ResearchMode.CALCULATION and bool(messages))
        return (
            TaskResult(
                task_id=task.task_id,
                completed=completed,
                evidence_ids=[item.evidence_id for item in evidence],
                tool_calls=len(calls[:3]),
                message="; ".join(messages) or "No research action was available",
            ),
            evidence,
        )

    @staticmethod
    def _calls_for_task(
        task: ResearchTask, dependency_evidence: list[EvidenceItem]
    ) -> list[tuple[str, ToolCall]]:
        text = (
            "text_researcher",
            ToolCall(
                name=ToolName.HYBRID_TEXT_SEARCH,
                arguments={"query": task.objective, "top_k": 5},
            ),
        )
        visual = (
            "visual_researcher",
            ToolCall(
                name=ToolName.VISUAL_SEARCH,
                arguments={"query": task.objective, "top_k": 5},
            ),
        )
        if task.mode == ResearchMode.TEXT:
            return [text]
        if task.mode == ResearchMode.VISUAL:
            return [visual]
        if task.mode == ResearchMode.BOTH:
            return [text, visual]
        if task.mode == ResearchMode.METADATA:
            return [("metadata_researcher", ToolCall(name=ToolName.DOCUMENT_METADATA))]
        expression = RuleBasedPlannerExpression.find(task.objective)
        return [
            (
                "calculator",
                ToolCall(
                    name=ToolName.CALCULATOR,
                    arguments={
                        "expression": expression or "0+0",
                        "evidence_ids": [item.evidence_id for item in dependency_evidence],
                    },
                ),
            )
        ]


class RuleBasedPlannerExpression:
    @staticmethod
    def find(text: str) -> str | None:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", text)
        return match.group(0) if match else None


def detect_conflicts(evidence: list[EvidenceItem]) -> list[EvidenceConflict]:
    by_task: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in evidence:
        if item.task_id:
            by_task[item.task_id].append(item)
    conflicts: list[EvidenceConflict] = []
    negative = re.compile(
        r"\b(?:does not|cannot|fails to|no improvement in)\s+([^.;]{3,100})", re.I
    )
    positive = re.compile(r"\b(?:does|can|supports|improves)\s+([^.;]{3,100})", re.I)

    def claim_terms(pattern: re.Pattern[str], text: str) -> set[str]:
        match = pattern.search(text)
        if not match:
            return set()
        stopwords = {"a", "an", "and", "for", "in", "of", "the", "to", "with"}
        return {
            token
            for token in re.findall(r"[a-z]+", match.group(1).lower())[:8]
            if token not in stopwords
        }

    for task_id, items in by_task.items():
        for negative_item in items:
            negative_terms = claim_terms(negative, negative_item.text)
            if len(negative_terms) < 2:
                continue
            for positive_item in items:
                if positive_item.evidence_id == negative_item.evidence_id:
                    continue
                positive_terms = claim_terms(positive, positive_item.text)
                union = negative_terms | positive_terms
                overlap = len(negative_terms & positive_terms) / max(1, len(union))
                if len(positive_terms) >= 2 and overlap >= 0.5:
                    conflicts.append(
                        EvidenceConflict(
                            task_ids=[task_id],
                            evidence_ids=[negative_item.evidence_id, positive_item.evidence_id],
                            reason=(
                                "Retrieved sources contain potentially conflicting "
                                "capability claims."
                            ),
                        )
                    )
                    break
            if conflicts and conflicts[-1].task_ids == [task_id]:
                break
    return conflicts


def select_synthesis_evidence(
    evidence: list[EvidenceItem],
    task_ids: list[str],
    *,
    limit: int = 6,
) -> list[EvidenceItem]:
    """Put the strongest item from every task before additional context."""

    selected: list[EvidenceItem] = []
    for task_id in task_ids:
        candidates = [item for item in evidence if item.task_id == task_id]
        if candidates:
            selected.append(
                max(candidates, key=lambda item: (item.support_score, item.evidence_id))
            )
    selected_ids = {item.evidence_id for item in selected}
    remaining = sorted(
        (item for item in evidence if item.evidence_id not in selected_ids),
        key=lambda item: (item.support_score, item.evidence_id),
        reverse=True,
    )
    return [*selected, *remaining][:limit]


def _board_citations(evidence: list[EvidenceItem]) -> list[AnswerCitation]:
    citations: list[AnswerCitation] = []
    seen: set[tuple[str, int, str, str | None]] = set()
    for item in evidence:
        key = (item.document_id, item.page_number, item.record_id, item.asset_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            AnswerCitation(
                document_id=item.document_id,
                page_number=item.page_number,
                record_id=item.record_id,
                asset_id=item.asset_id,
            )
        )
    return citations


def _plan_abstention(reason: str) -> StructuredAnswer:
    return StructuredAnswer(
        answer=f"I cannot complete the planned research from the available evidence. {reason}",
        status=AnswerStatus.ABSTAINED,
        retrieval_mode="plan_execute",
    )


class PlanExecuteWorkflow:
    def __init__(
        self,
        *,
        planner: Planner,
        researcher: Researcher,
        answerer: Answerer,
        input_guard: InputGuard,
        plan_validator: PlanValidator,
        max_parallel_workers: int = 2,
        max_replans: int = 1,
        recursion_limit: int = 14,
    ) -> None:
        self.planner = planner
        self.researcher = researcher
        self.answerer = answerer
        self.input_guard = input_guard
        self.plan_validator = plan_validator
        self.max_parallel_workers = max_parallel_workers
        self.max_replans = max_replans
        self.recursion_limit = recursion_limit
        self.citation_validator = CitationValidator()
        self.graph = self._build_graph()

    @staticmethod
    def _trace(node: str, outcome: str, started: float) -> list[NodeTrace]:
        return [
            NodeTrace(
                node=node,
                outcome=outcome,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        ]

    def _plan(self, state: PlanState) -> dict[str, Any]:
        started = time.perf_counter()
        plan = self.planner.create_plan(state["question"])
        return {"plan": plan, "trace": self._trace("plan", f"{len(plan.tasks)} tasks", started)}

    def _validate(self, state: PlanState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            self.plan_validator.validate(state["plan"])
            return {"error": None, "trace": self._trace("validate_plan", "valid", started)}
        except PlanValidationError as exc:
            return {
                "error": str(exc),
                "terminal_reason": "invalid_plan",
                "trace": self._trace("validate_plan", "blocked", started),
            }

    def _execute(self, state: PlanState) -> dict[str, Any]:
        started = time.perf_counter()
        completed_ids = {result.task_id for result in state["task_results"] if result.completed}
        results = list(state["task_results"])
        evidence = list(state["evidence"])
        pending = {
            task.task_id: task for task in state["plan"].tasks if task.task_id not in completed_ids
        }
        while pending:
            ready = [
                task for task in pending.values() if set(task.dependencies).issubset(completed_ids)
            ]
            if not ready:
                for task in pending.values():
                    results.append(
                        TaskResult(
                            task_id=task.task_id,
                            completed=False,
                            message="A required dependency did not complete.",
                        )
                    )
                break
            dependency_evidence = {
                task.task_id: [item for item in evidence if item.task_id in set(task.dependencies)]
                for task in ready
            }
            with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
                futures = {
                    executor.submit(
                        self.researcher.run, task, dependency_evidence[task.task_id]
                    ): task
                    for task in ready
                }
                for future in as_completed(futures):
                    task_result, task_evidence = future.result()
                    results = [item for item in results if item.task_id != task_result.task_id]
                    results.append(task_result)
                    existing = {item.evidence_id for item in evidence}
                    evidence.extend(
                        item for item in task_evidence if item.evidence_id not in existing
                    )
                    if task_result.completed:
                        completed_ids.add(task_result.task_id)
            for task in ready:
                pending.pop(task.task_id, None)
        task_ids = [task.task_id for task in state["plan"].tasks]
        task_order = {task_id: index for index, task_id in enumerate(task_ids)}
        results.sort(key=lambda item: task_order.get(item.task_id, len(task_order)))
        evidence = select_synthesis_evidence(evidence, task_ids, limit=len(evidence))
        return {
            "task_results": results,
            "evidence": evidence,
            "trace": self._trace("execute_research", f"{len(results)} results", started),
        }

    def _replan(self, state: PlanState) -> dict[str, Any]:
        started = time.perf_counter()
        failed = [result.task_id for result in state["task_results"] if not result.completed]
        plan = self.planner.replan(state["plan"], failed)
        return {
            "plan": plan,
            "replan_count": state["replan_count"] + 1,
            "trace": self._trace("replan", f"retry {len(failed)} tasks", started),
        }

    def _find_conflicts(self, state: PlanState) -> dict[str, Any]:
        started = time.perf_counter()
        conflicts = detect_conflicts(state["evidence"])
        return {
            "conflicts": conflicts,
            "trace": self._trace("detect_conflicts", f"{len(conflicts)} found", started),
        }

    def _synthesize(self, state: PlanState) -> dict[str, Any]:
        started = time.perf_counter()
        if not state["evidence"]:
            answer = _plan_abstention("No research task returned evidence.")
        else:
            selected = select_synthesis_evidence(
                state["evidence"],
                [task.task_id for task in state["plan"].tasks],
            )
            answer = self.answerer.answer(state["question"], _to_search_hits(selected))
            updates: dict[str, object] = {"retrieval_mode": "plan_execute"}
            if answer.status == AnswerStatus.ANSWERED:
                updates["citations"] = _board_citations(selected)
            answer = answer.model_copy(update=updates)
        return {"answer": answer, "trace": self._trace("synthesize", answer.status.value, started)}

    def _verify(self, state: PlanState) -> dict[str, Any]:
        started = time.perf_counter()
        answer = state["answer"] or _plan_abstention("No synthesis was produced.")
        issues = self.citation_validator.validate(answer, state["evidence"])
        completed = {result.task_id for result in state["task_results"] if result.completed}
        missing_required = [
            task.task_id
            for task in state["plan"].tasks
            if task.required and task.task_id not in completed
        ]
        if issues or answer.status != AnswerStatus.ANSWERED:
            status = WorkflowStatus.ABSTAINED
            answer = _plan_abstention("Citation or synthesis verification failed.")
            reason = "verification_failed"
        elif missing_required or state["conflicts"]:
            status = WorkflowStatus.PARTIAL
            reason = "partial_evidence" if missing_required else "unresolved_conflict"
            answer = answer.model_copy(
                update={
                    "answer": f"Partial answer with unresolved evidence limits: {answer.answer}"
                }
            )
        else:
            status = WorkflowStatus.ANSWERED
            reason = "verified_complete_answer"
        return {
            "answer": answer,
            "status": status,
            "terminal_reason": reason,
            "trace": self._trace("verify", status.value, started),
        }

    def _abstain(self, state: PlanState) -> dict[str, Any]:
        started = time.perf_counter()
        return {
            "answer": _plan_abstention(state.get("error") or "Plan execution was blocked."),
            "status": WorkflowStatus.BLOCKED,
            "trace": self._trace("abstain", "terminal", started),
        }

    @staticmethod
    def _after_validation(state: PlanState) -> Literal["execute", "abstain"]:
        return "abstain" if state.get("error") else "execute"

    def _after_execution(self, state: PlanState) -> Literal["replan", "conflicts"]:
        completed = {result.task_id for result in state["task_results"] if result.completed}
        failed_required = any(
            task.required and task.task_id not in completed for task in state["plan"].tasks
        )
        if failed_required and state["replan_count"] < self.max_replans:
            return "replan"
        return "conflicts"

    def _build_graph(self) -> Any:
        builder = StateGraph(PlanState)
        builder.add_node("plan", self._plan)
        builder.add_node("validate", self._validate)
        builder.add_node("execute", self._execute)
        builder.add_node("replan", self._replan)
        builder.add_node("conflicts", self._find_conflicts)
        builder.add_node("synthesize", self._synthesize)
        builder.add_node("verify", self._verify)
        builder.add_node("abstain", self._abstain)
        builder.add_edge(START, "plan")
        builder.add_edge("plan", "validate")
        builder.add_conditional_edges("validate", self._after_validation)
        builder.add_conditional_edges("execute", self._after_execution)
        builder.add_edge("replan", "validate")
        builder.add_edge("conflicts", "synthesize")
        builder.add_edge("synthesize", "verify")
        builder.add_edge("verify", END)
        builder.add_edge("abstain", END)
        return builder.compile()

    def run(self, question: str) -> PlanResult:
        started = time.perf_counter()
        try:
            normalized = self.input_guard.validate(question)
        except GuardrailViolation as exc:
            empty_plan = ResearchPlan(question=question, tasks=[])
            return PlanResult(
                question=question,
                status=WorkflowStatus.BLOCKED,
                plan=empty_plan,
                task_results=[],
                evidence=[],
                conflicts=[],
                answer=_plan_abstention(str(exc)),
                replan_count=0,
                trace=[NodeTrace(node="validate_input", outcome="blocked", duration_ms=0)],
                terminal_reason="input_blocked",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        empty_plan = ResearchPlan(question=normalized, tasks=[])
        initial: PlanState = {
            "question": normalized,
            "plan": empty_plan,
            "task_results": [],
            "evidence": [],
            "conflicts": [],
            "answer": None,
            "status": WorkflowStatus.ABSTAINED,
            "replan_count": 0,
            "error": None,
            "terminal_reason": "",
            "trace": [],
        }
        try:
            state = self.graph.invoke(initial, {"recursion_limit": self.recursion_limit})
        except GraphRecursionError:
            state = initial | {
                "answer": _plan_abstention("The supervisor reached its recursion limit."),
                "status": WorkflowStatus.BLOCKED,
                "terminal_reason": "recursion_limit",
                "trace": [NodeTrace(node="graph", outcome="recursion_limit", duration_ms=0)],
            }
        return PlanResult(
            question=normalized,
            status=state["status"],
            plan=state["plan"],
            task_results=state["task_results"],
            evidence=state["evidence"],
            conflicts=state["conflicts"],
            answer=state["answer"] or _plan_abstention("No answer was produced."),
            replan_count=state["replan_count"],
            trace=state["trace"],
            terminal_reason=state["terminal_reason"] or "safe_terminal",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
