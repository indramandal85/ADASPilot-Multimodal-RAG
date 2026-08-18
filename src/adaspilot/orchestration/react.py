"""Module 7: a bounded ReAct-style research loop over read-only tools."""

from __future__ import annotations

import re
import time
from typing import Any, Literal, Protocol, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph

from adaspilot.domain.orchestration import (
    EvidenceItem,
    QueryKind,
    ReactResult,
    ToolCall,
    ToolName,
    ToolObservation,
    WorkflowStatus,
)
from adaspilot.domain.retrieval import AnswerStatus, SearchHit, StructuredAnswer
from adaspilot.orchestration.direct import classify_query
from adaspilot.orchestration.tools import ProjectToolbox, normalized_tool_signature
from adaspilot.retrieval.answers import Answerer
from adaspilot.safety.guardrails import CitationValidator, GuardrailViolation, InputGuard


class ReactState(TypedDict):
    question: str
    query_kind: QueryKind
    evidence: list[EvidenceItem]
    observations: list[ToolObservation]
    signature_counts: dict[str, int]
    tool_call_count: int
    redundant_action_count: int
    consecutive_no_gain: int
    answer: StructuredAnswer | None
    status: WorkflowStatus
    terminal_reason: str
    done: bool


class ActionPolicy(Protocol):
    def select(self, state: ReactState) -> ToolCall: ...


class RuleBasedReActPolicy:
    """Explainable local policy; an LLM policy can implement the same protocol later."""

    def __init__(self, *, allow_web_search: bool = False) -> None:
        self.allow_web_search = allow_web_search

    def select(self, state: ReactState) -> ToolCall:
        question = state["question"]
        if not state["observations"]:
            if state["query_kind"] == QueryKind.VISUAL:
                return ToolCall(
                    name=ToolName.VISUAL_SEARCH,
                    arguments={"query": question, "top_k": 5},
                )
            if state["query_kind"] == QueryKind.METADATA:
                return ToolCall(name=ToolName.DOCUMENT_METADATA, arguments={})
            return ToolCall(
                name=ToolName.HYBRID_TEXT_SEARCH,
                arguments={"query": question, "top_k": 6},
            )

        if not state["evidence"]:
            if state["tool_call_count"] == 1 and state["query_kind"] == QueryKind.VISUAL:
                return ToolCall(
                    name=ToolName.HYBRID_TEXT_SEARCH,
                    arguments={"query": f"{question} caption OCR", "top_k": 6},
                )
            if self.allow_web_search and not any(
                observation.tool_name == ToolName.WEB_SEARCH
                for observation in state["observations"]
            ):
                return ToolCall(
                    name=ToolName.WEB_SEARCH,
                    arguments={"query": question},
                )
            return ToolCall(
                name=ToolName.ABSTAIN,
                arguments={"reason": "No allowed tool produced supporting evidence."},
            )

        if state["query_kind"] == QueryKind.CALCULATION:
            expression = self._expression(question)
            if expression:
                return ToolCall(
                    name=ToolName.CALCULATOR,
                    arguments={
                        "expression": expression,
                        "evidence_ids": [item.evidence_id for item in state["evidence"][:3]],
                    },
                )

        if "full page" in question.lower() and not any(
            observation.tool_name == ToolName.PAGE_RETRIEVAL
            for observation in state["observations"]
        ):
            first = state["evidence"][0]
            return ToolCall(
                name=ToolName.PAGE_RETRIEVAL,
                arguments={
                    "document_id": first.document_id,
                    "page_number": first.page_number,
                },
            )

        return ToolCall(
            name=ToolName.FINISH,
            arguments={
                "evidence_ids": [item.evidence_id for item in state["evidence"]],
                "answer_outline": "Answer the question only from the collected evidence.",
            },
        )

    @staticmethod
    def _expression(question: str) -> str | None:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", question)
        return match.group(0) if match else None


def _to_search_hits(evidence: list[EvidenceItem]) -> list[SearchHit]:
    return [
        SearchHit(
            record_id=item.record_id,
            score=item.support_score,
            document_id=item.document_id,
            page_number=item.page_number,
            text=item.text,
            asset_id=item.asset_id,
            image_path=item.image_path,
            source=item.source,
        )
        for item in evidence
    ]


def _react_abstention(reason: str) -> StructuredAnswer:
    return StructuredAnswer(
        answer=f"I cannot complete this research request from the allowed evidence. {reason}",
        status=AnswerStatus.ABSTAINED,
        retrieval_mode="react",
    )


class ReactResearchAgent:
    def __init__(
        self,
        *,
        toolbox: ProjectToolbox,
        answerer: Answerer,
        input_guard: InputGuard,
        policy: ActionPolicy | None = None,
        max_tool_calls: int = 6,
        recursion_limit: int = 8,
    ) -> None:
        self.toolbox = toolbox
        self.answerer = answerer
        self.input_guard = input_guard
        self.policy = policy or RuleBasedReActPolicy(allow_web_search=toolbox.web_search_available)
        self.max_tool_calls = max_tool_calls
        self.recursion_limit = recursion_limit
        self.citation_validator = CitationValidator()
        self.graph = self._build_graph()

    def _finish(self, state: ReactState, call: ToolCall) -> dict[str, Any]:
        requested = {str(value) for value in call.arguments.get("evidence_ids", [])}
        selected = [item for item in state["evidence"] if item.evidence_id in requested]
        if not selected or requested - {item.evidence_id for item in state["evidence"]}:
            return {
                "answer": _react_abstention("Finish requested unknown or empty evidence IDs."),
                "status": WorkflowStatus.ABSTAINED,
                "terminal_reason": "invalid_finish_evidence",
                "done": True,
            }
        answer = self.answerer.answer(state["question"], _to_search_hits(selected))
        issues = self.citation_validator.validate(answer, selected)
        if answer.status != AnswerStatus.ANSWERED or issues:
            return {
                "answer": _react_abstention("The evidence-linked answer failed citation checks."),
                "status": WorkflowStatus.ABSTAINED,
                "terminal_reason": "verification_failed",
                "done": True,
            }
        return {
            "answer": answer.model_copy(update={"retrieval_mode": "react"}),
            "status": WorkflowStatus.ANSWERED,
            "terminal_reason": "finished_with_evidence",
            "done": True,
        }

    def _step(self, state: ReactState) -> dict[str, Any]:
        if state["tool_call_count"] >= self.max_tool_calls:
            return {
                "answer": _react_abstention("The tool-call budget was exhausted."),
                "status": WorkflowStatus.ABSTAINED,
                "terminal_reason": "tool_budget",
                "done": True,
            }
        if state["consecutive_no_gain"] >= 2:
            return {
                "answer": _react_abstention("Two consecutive actions added no evidence."),
                "status": WorkflowStatus.ABSTAINED,
                "terminal_reason": "no_evidence_gain",
                "done": True,
            }

        call = self.policy.select(state)
        if call.name == ToolName.FINISH:
            return self._finish(state, call)
        if call.name == ToolName.ABSTAIN:
            reason = str(call.arguments.get("reason", "The agent chose to abstain."))
            return {
                "answer": _react_abstention(reason),
                "status": WorkflowStatus.ABSTAINED,
                "terminal_reason": "agent_abstained",
                "done": True,
            }

        signature = normalized_tool_signature(call)
        previous_count = state["signature_counts"].get(signature, 0)
        if previous_count >= 2:
            return {
                "answer": _react_abstention("An equivalent tool action repeated too many times."),
                "status": WorkflowStatus.BLOCKED,
                "terminal_reason": "repeated_action",
                "done": True,
            }
        signature_counts = dict(state["signature_counts"])
        signature_counts[signature] = previous_count + 1
        existing_ids = {item.evidence_id for item in state["evidence"]}
        try:
            if call.name == ToolName.WEB_SEARCH:
                corpus_tools = {
                    ToolName.HYBRID_TEXT_SEARCH,
                    ToolName.VISUAL_SEARCH,
                    ToolName.PARENT_CONTEXT,
                    ToolName.PAGE_RETRIEVAL,
                    ToolName.DOCUMENT_METADATA,
                }
                attempted_corpus = any(
                    observation.tool_name in corpus_tools for observation in state["observations"]
                )
                if state["evidence"] or not attempted_corpus:
                    raise GuardrailViolation(
                        "web search is allowed only after corpus retrieval returns no evidence"
                    )
            if call.name == ToolName.CALCULATOR:
                referenced = {str(value) for value in call.arguments.get("evidence_ids", [])}
                if not referenced or referenced - existing_ids:
                    raise GuardrailViolation("calculator referenced unknown evidence IDs")
            result, duration_ms = self.toolbox.registry.execute("react", call)
            new_items = [item for item in result.evidence if item.evidence_id not in existing_ids]
            observation = ToolObservation(
                tool_name=call.name,
                signature=signature,
                evidence_ids=[item.evidence_id for item in result.evidence],
                new_evidence_count=len(new_items),
                message=result.message,
                value=result.value,
                duration_ms=duration_ms,
            )
        except (GuardrailViolation, TimeoutError, OSError, ValueError) as exc:
            new_items = []
            observation = ToolObservation(
                tool_name=call.name,
                signature=signature,
                new_evidence_count=0,
                message=f"Tool failed safely: {exc}",
                duration_ms=0,
            )
        return {
            "evidence": [*state["evidence"], *new_items],
            "observations": [*state["observations"], observation],
            "signature_counts": signature_counts,
            "tool_call_count": state["tool_call_count"] + 1,
            "redundant_action_count": state["redundant_action_count"] + int(previous_count > 0),
            "consecutive_no_gain": 0 if new_items else state["consecutive_no_gain"] + 1,
        }

    @staticmethod
    def _route(state: ReactState) -> Literal["step", "__end__"]:
        return "__end__" if state["done"] else "step"

    def _build_graph(self) -> Any:
        builder = StateGraph(ReactState)
        builder.add_node("step", self._step)
        builder.add_edge(START, "step")
        builder.add_conditional_edges("step", self._route)
        return builder.compile()

    def run(self, question: str) -> ReactResult:
        started = time.perf_counter()
        try:
            normalized = self.input_guard.validate(question)
        except GuardrailViolation as exc:
            answer = _react_abstention(str(exc))
            return ReactResult(
                question=question,
                status=WorkflowStatus.BLOCKED,
                answer=answer,
                evidence=[],
                observations=[],
                tool_call_count=0,
                redundant_action_count=0,
                terminal_reason="input_blocked",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        initial: ReactState = {
            "question": normalized,
            "query_kind": classify_query(normalized),
            "evidence": [],
            "observations": [],
            "signature_counts": {},
            "tool_call_count": 0,
            "redundant_action_count": 0,
            "consecutive_no_gain": 0,
            "answer": None,
            "status": WorkflowStatus.ABSTAINED,
            "terminal_reason": "",
            "done": False,
        }
        try:
            state = self.graph.invoke(initial, {"recursion_limit": self.recursion_limit})
        except GraphRecursionError:
            state = initial | {
                "answer": _react_abstention("The agent reached its recursion limit."),
                "status": WorkflowStatus.BLOCKED,
                "terminal_reason": "recursion_limit",
                "done": True,
            }
        answer = state["answer"] or _react_abstention("No terminal answer was produced.")
        return ReactResult(
            question=normalized,
            status=state["status"],
            answer=answer,
            evidence=state["evidence"],
            observations=state["observations"],
            tool_call_count=state["tool_call_count"],
            redundant_action_count=state["redundant_action_count"],
            terminal_reason=state["terminal_reason"] or "safe_terminal",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
