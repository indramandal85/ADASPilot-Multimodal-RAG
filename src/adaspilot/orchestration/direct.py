"""Module 6: lowest-cost corrective retrieval workflow implemented in LangGraph."""

from __future__ import annotations

import operator
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Annotated, Any, Literal, Protocol, TypedDict, TypeVar

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from adaspilot.domain.orchestration import (
    DirectResult,
    EvidenceItem,
    NodeTrace,
    QueryKind,
)
from adaspilot.domain.retrieval import AnswerStatus, SearchHit, StructuredAnswer
from adaspilot.retrieval.answers import Answerer
from adaspilot.retrieval.text import Retriever
from adaspilot.safety.guardrails import (
    AccessPolicy,
    CitationValidator,
    GuardrailViolation,
    InjectionDetector,
    InputGuard,
    PiiRedactor,
)

ValueT = TypeVar("ValueT")

_EVIDENCE_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "define",
    "describe",
    "does",
    "explain",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
    "why",
}


def _evidence_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in _EVIDENCE_STOPWORDS or len(token) < 2:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        terms.add(token)
    return terms


def _minimum_term_coverage(term_count: int) -> float:
    if term_count <= 2:
        return 1.0
    if term_count <= 4:
        return 0.6
    return 0.4


class QueryRewriter(Protocol):
    def rewrite(self, question: str, query_kind: QueryKind) -> str: ...


class SimpleQueryRewriter:
    def rewrite(self, question: str, query_kind: QueryKind) -> str:
        suffix = {
            QueryKind.VISUAL: "figure diagram caption visual evidence",
            QueryKind.METADATA: "document title author source metadata",
            QueryKind.CALCULATION: "reported values table numerical evidence",
            QueryKind.COMPARISON: "compare differences similarities results",
        }.get(query_kind, "ADAS technical definition evidence")
        return f"{question} {suffix}"


class DirectState(TypedDict):
    original_question: str
    query: str
    query_kind: QueryKind
    retry_count: int
    hits: list[SearchHit]
    evidence: list[EvidenceItem]
    sufficient: bool
    answer: StructuredAnswer | None
    trace: Annotated[list[NodeTrace], operator.add]
    error: str | None
    terminal_reason: str


def _run_with_timeout(function: Callable[[], ValueT], timeout_seconds: float) -> ValueT:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(function)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError("node execution timed out") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def classify_query(question: str) -> QueryKind:
    lowered = question.lower()
    if any(word in lowered for word in ["figure", "diagram", "plot", "image", "shown"]):
        return QueryKind.VISUAL
    if any(word in lowered for word in ["metadata", "title", "author", "published"]):
        return QueryKind.METADATA
    if any(word in lowered for word in ["calculate", "difference", "percentage", "sum"]):
        return QueryKind.CALCULATION
    if any(word in lowered for word in ["compare", "versus", "difference between"]):
        return QueryKind.COMPARISON
    return QueryKind.FACT


def evidence_from_hits(
    hits: list[SearchHit], tool_name: str = "direct_retrieval"
) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            evidence_id=hit.record_id,
            document_id=hit.document_id,
            page_number=hit.page_number,
            text=hit.text,
            source=hit.source,
            tool_name=tool_name,
            record_id=hit.record_id,
            asset_id=hit.asset_id,
            image_path=hit.image_path,
            support_score=hit.score,
        )
        for hit in hits
    ]


def _abstained_answer(mode: str, reason: str) -> StructuredAnswer:
    return StructuredAnswer(
        answer=f"I cannot answer from the available evidence. {reason}",
        status=AnswerStatus.ABSTAINED,
        retrieval_mode=mode,
    )


class DirectCorrectiveWorkflow:
    """Validate -> classify -> retrieve -> grade -> retry once -> verify -> END."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        answerer: Answerer,
        input_guard: InputGuard,
        access_policy: AccessPolicy | None = None,
        rewriter: QueryRewriter | None = None,
        evidence_score_threshold: float = 0.35,
        top_k: int = 8,
        node_timeout_seconds: float = 15,
        recursion_limit: int = 10,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.input_guard = input_guard
        self.access_policy = access_policy or AccessPolicy()
        self.rewriter = rewriter or SimpleQueryRewriter()
        self.evidence_score_threshold = evidence_score_threshold
        self.top_k = top_k
        self.node_timeout_seconds = node_timeout_seconds
        self.recursion_limit = recursion_limit
        self.injection_detector = InjectionDetector()
        self.pii_redactor = PiiRedactor()
        self.citation_validator = CitationValidator()
        self.graph = self._build_graph()

    def _trace(self, node: str, outcome: str, started: float) -> list[NodeTrace]:
        return [
            NodeTrace(
                node=node,
                outcome=outcome,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        ]

    def _validate(self, state: DirectState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            question = self.input_guard.validate(state["original_question"])
            return {"query": question, "trace": self._trace("validate", "valid", started)}
        except GuardrailViolation as exc:
            return {
                "error": str(exc),
                "terminal_reason": "input_blocked",
                "trace": self._trace("validate", "blocked", started),
            }

    def _classify(self, state: DirectState) -> dict[str, Any]:
        started = time.perf_counter()
        kind = classify_query(state["query"])
        return {
            "query_kind": kind,
            "trace": self._trace("classify", kind.value, started),
        }

    def _retrieve(self, state: DirectState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            candidates = _run_with_timeout(
                lambda: self.retriever.retrieve(state["query"], max(self.top_k * 4, self.top_k)),
                self.node_timeout_seconds,
            )
            hits = self.access_policy.filter_hits(candidates)[: self.top_k]
            return {
                "hits": hits,
                "evidence": evidence_from_hits(hits),
                "error": None,
                "trace": self._trace("retrieve", f"{len(hits)} hits", started),
            }
        except (TimeoutError, OSError, ValueError) as exc:
            return {
                "hits": [],
                "evidence": [],
                "error": str(exc),
                "trace": self._trace("retrieve", "failed", started),
            }

    def _grade(self, state: DirectState) -> dict[str, Any]:
        started = time.perf_counter()
        # Always grade against the user's question. The rewrite suffix is retrieval help,
        # not evidence that a passage actually addresses the requested concept.
        query_terms = _evidence_terms(state["original_question"])
        safe_hits = [
            hit.model_copy(update={"text": self.pii_redactor.redact(hit.text)[0]})
            for hit in state["hits"]
            if not self.injection_detector.detect(hit.text)
        ]
        best_overlap = max(
            (
                len(query_terms & _evidence_terms(hit.text)) / max(1, len(query_terms))
                for hit in safe_hits
            ),
            default=0.0,
        )
        best_score = max((hit.score for hit in safe_hits), default=float("-inf"))
        sufficient = bool(safe_hits) and (
            best_score >= self.evidence_score_threshold
            and best_overlap >= _minimum_term_coverage(len(query_terms))
        )
        return {
            "hits": safe_hits,
            "evidence": evidence_from_hits(safe_hits),
            "sufficient": sufficient,
            "trace": self._trace("grade_evidence", "sufficient" if sufficient else "weak", started),
        }

    def _rewrite(self, state: DirectState) -> dict[str, Any]:
        started = time.perf_counter()
        rewritten = self.rewriter.rewrite(state["original_question"], state["query_kind"])
        return {
            "query": rewritten,
            "retry_count": state["retry_count"] + 1,
            "trace": self._trace("rewrite", "retry_1", started),
        }

    def _synthesize(self, state: DirectState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            answer = _run_with_timeout(
                lambda: self.answerer.answer(state["original_question"], state["hits"]),
                self.node_timeout_seconds,
            )
            return {
                "answer": answer,
                "trace": self._trace("generate", answer.status.value, started),
            }
        except (TimeoutError, OSError, ValueError) as exc:
            return {
                "error": str(exc),
                "answer": _abstained_answer("direct_corrective", "Generation failed safely."),
                "trace": self._trace("generate", "failed", started),
            }

    def _verify(self, state: DirectState) -> dict[str, Any]:
        started = time.perf_counter()
        answer = state["answer"] or _abstained_answer("direct_corrective", "No answer produced.")
        issues = self.citation_validator.validate(answer, state["evidence"])
        if answer.status == AnswerStatus.ANSWERED and not issues:
            terminal_reason = "verified_answer"
            outcome = "verified"
        else:
            terminal_reason = "verification_failed"
            outcome = "abstained"
            answer = _abstained_answer(
                "direct_corrective",
                "The generated answer could not be verified against retrieved citations.",
            )
        return {
            "answer": answer,
            "terminal_reason": terminal_reason,
            "trace": self._trace("verify", outcome, started),
        }

    def _abstain(self, state: DirectState) -> dict[str, Any]:
        started = time.perf_counter()
        reason = state.get("error") or "Evidence remained weak after the single allowed retry."
        return {
            "answer": _abstained_answer("direct_corrective", reason),
            "terminal_reason": state.get("terminal_reason") or "insufficient_evidence",
            "trace": self._trace("abstain", "terminal", started),
        }

    @staticmethod
    def _after_validate(state: DirectState) -> Literal["classify", "abstain"]:
        return "abstain" if state.get("error") else "classify"

    @staticmethod
    def _after_grade(state: DirectState) -> Literal["generate", "rewrite", "abstain"]:
        if state["sufficient"]:
            return "generate"
        return "rewrite" if state["retry_count"] == 0 else "abstain"

    def _build_graph(self) -> Any:
        builder = StateGraph(DirectState)
        builder.add_node("validate", self._validate)
        builder.add_node("classify", self._classify)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade", self._grade)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("generate", self._synthesize)
        builder.add_node("verify", self._verify)
        builder.add_node("abstain", self._abstain)
        builder.add_edge(START, "validate")
        builder.add_conditional_edges("validate", self._after_validate)
        builder.add_edge("classify", "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges("grade", self._after_grade)
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", "verify")
        builder.add_edge("verify", END)
        builder.add_edge("abstain", END)
        return builder.compile()

    def run(self, question: str) -> DirectResult:
        initial: DirectState = {
            "original_question": question,
            "query": question,
            "query_kind": QueryKind.FACT,
            "retry_count": 0,
            "hits": [],
            "evidence": [],
            "sufficient": False,
            "answer": None,
            "trace": [],
            "error": None,
            "terminal_reason": "",
        }
        try:
            state = self.graph.invoke(initial, {"recursion_limit": self.recursion_limit})
        except GraphRecursionError:
            state = initial | {
                "answer": _abstained_answer(
                    "direct_corrective", "The workflow reached its recursion limit."
                ),
                "terminal_reason": "recursion_limit",
                "trace": [NodeTrace(node="graph", outcome="recursion_limit", duration_ms=0)],
            }
        answer = state["answer"] or _abstained_answer("direct_corrective", "No answer produced.")
        return DirectResult(
            question=question,
            query_kind=state["query_kind"],
            retry_count=state["retry_count"],
            evidence=state["evidence"],
            answer=answer,
            trace=state["trace"],
            terminal_reason=state["terminal_reason"] or "safe_terminal",
        )
