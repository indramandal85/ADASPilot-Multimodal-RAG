"""Unified deterministic evaluation for direct, ReAct, and plan modes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from adaspilot.config import Settings
from adaspilot.domain.evaluation import (
    EvaluationExample,
    ExpectedExecutionMode,
    Modality,
)
from adaspilot.domain.retrieval import AnswerStatus
from adaspilot.domain.serving import ExecutionMode, ExecutionRecord
from adaspilot.domain.system_evaluation import (
    ExampleEvaluation,
    ModeMetrics,
    RouterMetrics,
    SystemEvaluationReport,
)
from adaspilot.evaluation.ragas_adapter import RagasEvaluator
from adaspilot.observability.langsmith import LangSmithTracer
from adaspilot.observability.usage import UsageSnapshot
from adaspilot.orchestration.router import ExecutionRouter
from adaspilot.orchestration.service import WorkflowRunner


def expected_mode(example: EvaluationExample) -> ExecutionMode:
    return {
        ExpectedExecutionMode.DIRECT: ExecutionMode.DIRECT,
        ExpectedExecutionMode.REACT: ExecutionMode.REACT,
        ExpectedExecutionMode.PLAN_EXECUTE: ExecutionMode.PLAN,
    }[example.expected_execution_mode]


def expected_tools(example: EvaluationExample) -> set[str]:
    tools: set[str] = set()
    if Modality.VISUAL in example.required_modalities:
        tools.add("visual_search")
    if Modality.METADATA in example.required_modalities:
        tools.add("document_metadata_lookup")
    if example.should_answer and (
        {Modality.TEXT, Modality.TABLE, Modality.CALCULATION} & example.required_modalities
    ):
        tools.add("hybrid_text_search")
    if Modality.CALCULATION in example.required_modalities:
        tools.add("evidence_linked_calculator")
    return tools


def _unique_pages(record: ExecutionRecord, limit: int | None = None) -> list[tuple[str, int]]:
    pages: list[tuple[str, int]] = []
    for item in record.evidence:
        page = (item.document_id, item.page_number)
        if page not in pages:
            pages.append(page)
        if limit is not None and len(pages) >= limit:
            break
    return pages


def _relevant_pages(example: EvaluationExample) -> set[tuple[str, int]]:
    return {(item.document_id, item.page_number) for item in example.evidence}


def _first_rank(relevant: set[tuple[str, int]], ranked: list[tuple[str, int]]) -> int | None:
    return next((index for index, page in enumerate(ranked, start=1) if page in relevant), None)


def _ndcg(relevant: set[tuple[str, int]], ranked: list[tuple[str, int]], k: int = 10) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        (1.0 / math.log2(rank + 1)) if page in relevant else 0.0
        for rank, page in enumerate(ranked[:k], start=1)
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _binary_f1(expected: list[bool], predicted: list[bool]) -> float:
    true_positive = sum(
        wanted and actual for wanted, actual in zip(expected, predicted, strict=True)
    )
    false_positive = sum(
        not wanted and actual for wanted, actual in zip(expected, predicted, strict=True)
    )
    false_negative = sum(
        wanted and not actual for wanted, actual in zip(expected, predicted, strict=True)
    )
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    )
    return _f1(precision, recall)


class SystemEvaluator:
    def __init__(
        self,
        settings: Settings,
        runner: WorkflowRunner,
        *,
        build_dir: Path,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.router = ExecutionRouter()
        self.tracer = LangSmithTracer(settings)
        self.ragas = RagasEvaluator(settings, build_dir)

    def evaluate(
        self,
        examples: Sequence[EvaluationExample],
        *,
        split: str,
        use_auto_router: bool = False,
        max_examples_per_mode: int | None = None,
    ) -> SystemEvaluationReport:
        selected = [example for example in examples if example.split.value == split]
        if max_examples_per_mode is not None:
            counts = {
                mode: 0 for mode in (ExecutionMode.DIRECT, ExecutionMode.REACT, ExecutionMode.PLAN)
            }
            limited: list[EvaluationExample] = []
            for example in selected:
                mode = expected_mode(example)
                if counts[mode] < max_examples_per_mode:
                    limited.append(example)
                    counts[mode] += 1
            selected = limited

        router_correct = sum(
            self.router.route(item.question) == expected_mode(item) for item in selected
        )
        by_mode: dict[
            ExecutionMode, list[tuple[EvaluationExample, ExecutionRecord, UsageSnapshot]]
        ] = {
            ExecutionMode.DIRECT: [],
            ExecutionMode.REACT: [],
            ExecutionMode.PLAN: [],
        }
        details: list[ExampleEvaluation] = []
        for example in selected:
            requested = ExecutionMode.AUTO if use_auto_router else expected_mode(example)
            before = self.runner.bundle.usage_tracker.snapshot()
            with self.tracer.run(
                "adaspilot-evaluation-example",
                inputs={"question_id": example.question_id, "mode": requested.value},
                metadata={"split": split, "expected_mode": expected_mode(example).value},
            ):
                record = self.runner.run(example.question, requested)
            usage = self.runner.bundle.usage_tracker.snapshot() - before
            by_mode[record.mode].append((example, record, usage))
            expected_abstain = not example.should_answer
            actual_abstain = record.answer.status == AnswerStatus.ABSTAINED
            pages = _unique_pages(record)
            citations = [(item.document_id, item.page_number) for item in record.answer.citations]
            details.append(
                ExampleEvaluation(
                    question_id=example.question_id,
                    expected_mode=expected_mode(example),
                    actual_mode=record.mode,
                    answered_as_expected=expected_abstain == actual_abstain,
                    retrieved_pages=[f"{document}:p{page}" for document, page in pages],
                    cited_pages=[f"{document}:p{page}" for document, page in citations],
                    expected_tools=sorted(expected_tools(example)),
                    actual_tools=record.tool_names,
                    latency_ms=record.latency_ms,
                    terminal_reason=record.terminal_reason,
                )
            )

        metrics = {mode: self._aggregate(mode, records) for mode, records in by_mode.items()}
        return SystemEvaluationReport(
            generated_at=datetime.now(UTC),
            corpus_version=self.runner.bundle.corpus_version,
            benchmark_split=split,
            execution_policy="auto_router" if use_auto_router else "benchmark_mode",
            router=RouterMetrics(
                example_count=len(selected),
                accuracy=router_correct / len(selected) if selected else 0.0,
            ),
            modes=metrics,
            examples=details,
            notes=[
                "Context precision/recall use verified page IDs and are deterministic.",
                "RAGAS scores remain null unless an explicit judge model is configured.",
                "Token and cost totals cover generation/planning calls, not evaluator calls.",
            ],
        )

    def _aggregate(
        self,
        mode: ExecutionMode,
        rows: list[tuple[EvaluationExample, ExecutionRecord, UsageSnapshot]],
    ) -> ModeMetrics:
        answerable = [(example, record) for example, record, _ in rows if example.should_answer]
        text_rows = [
            (example, record)
            for example, record in answerable
            if Modality.VISUAL not in example.required_modalities
        ]
        visual_rows = [
            (example, record)
            for example, record in answerable
            if Modality.VISUAL in example.required_modalities
        ]
        ranks: list[int | None] = []
        ndcgs: list[float] = []
        context_precisions: list[float] = []
        context_recalls: list[float] = []
        citation_precisions: list[float] = []
        citation_recalls: list[float] = []
        visual_hits: list[float] = []
        for example, record in answerable:
            relevant = _relevant_pages(example)
            ranked = _unique_pages(record, 10)
            top_five = ranked[:5]
            context_precisions.append(
                len(relevant & set(top_five)) / len(top_five) if top_five else 0.0
            )
            context_recalls.append(len(relevant & set(ranked)) / len(relevant) if relevant else 0.0)
            ndcgs.append(_ndcg(relevant, ranked))
            cited = {(item.document_id, item.page_number) for item in record.answer.citations}
            citation_precisions.append(len(relevant & cited) / len(cited) if cited else 0.0)
            citation_recalls.append(len(relevant & cited) / len(relevant) if relevant else 0.0)
        for example, record in text_rows:
            ranks.append(_first_rank(_relevant_pages(example), _unique_pages(record, 10)))
        for example, record in visual_rows:
            relevant_assets = {item.asset_id for item in example.evidence if item.asset_id}
            actual_assets = {
                item.asset_id for item in record.evidence[:5] if item.asset_id is not None
            }
            relevant_pages = _relevant_pages(example)
            visual_hits.append(
                float(
                    bool(
                        relevant_assets & actual_assets
                        or relevant_pages & set(_unique_pages(record, 5))
                    )
                )
            )

        expected_abstain = [not example.should_answer for example, _, _ in rows]
        actual_abstain = [record.answer.status == AnswerStatus.ABSTAINED for _, record, _ in rows]
        expected_tool_sets = [expected_tools(example) for example, _, _ in rows]
        actual_tool_sets = [set(record.tool_names) for _, record, _ in rows]
        tool_precision_values = [
            len(expected & actual) / len(actual) if actual else float(not expected)
            for expected, actual in zip(expected_tool_sets, actual_tool_sets, strict=True)
        ]
        tool_recall_values = [
            len(expected & actual) / len(expected) if expected else 1.0
            for expected, actual in zip(expected_tool_sets, actual_tool_sets, strict=True)
        ]
        has_tools = mode != ExecutionMode.DIRECT and bool(rows)
        tool_precision = fmean(tool_precision_values) if has_tools else None
        tool_recall = fmean(tool_recall_values) if has_tools else None
        plan_rows = [record for _, record, _ in rows if record.plan_valid is not None]
        usage = UsageSnapshot(
            model_calls=sum(item.model_calls for _, _, item in rows),
            input_tokens=sum(item.input_tokens for _, _, item in rows),
            output_tokens=sum(item.output_tokens for _, _, item in rows),
            estimated_cost_usd=sum(item.estimated_cost_usd for _, _, item in rows),
        )
        pairs = [(example, record) for example, record, _ in rows]
        return ModeMetrics(
            example_count=len(rows),
            answerable_count=len(answerable),
            text_recall_at_5=(
                sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks) if ranks else 0.0
            ),
            mrr=fmean(0.0 if rank is None else 1 / rank for rank in ranks) if ranks else 0.0,
            ndcg_at_10=fmean(ndcgs) if ndcgs else 0.0,
            visual_recall_at_5=fmean(visual_hits) if visual_hits else 0.0,
            citation_precision=fmean(citation_precisions) if citation_precisions else 0.0,
            citation_recall=fmean(citation_recalls) if citation_recalls else 0.0,
            context_precision=fmean(context_precisions) if context_precisions else 0.0,
            context_recall=fmean(context_recalls) if context_recalls else 0.0,
            abstention_f1=_binary_f1(expected_abstain, actual_abstain),
            tool_call_precision=tool_precision,
            tool_call_recall=tool_recall,
            tool_call_f1=(
                _f1(tool_precision, tool_recall)
                if tool_precision is not None and tool_recall is not None
                else None
            ),
            plan_validity=(
                fmean(float(bool(record.plan_valid)) for record in plan_rows) if plan_rows else None
            ),
            task_completion=(
                fmean(record.task_completion_rate or 0.0 for record in plan_rows)
                if plan_rows
                else None
            ),
            agent_goal_accuracy=(
                sum(
                    wanted == actual
                    for wanted, actual in zip(expected_abstain, actual_abstain, strict=True)
                )
                / len(rows)
                if rows
                else 0.0
            ),
            repeated_step_rate=(
                sum(record.repeated_step_count for _, record, _ in rows)
                / max(1, sum(record.tool_call_count for _, record, _ in rows))
            ),
            p50_latency_ms=_percentile([record.latency_ms for _, record, _ in rows], 0.50),
            p95_latency_ms=_percentile([record.latency_ms for _, record, _ in rows], 0.95),
            model_calls=usage.model_calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            ragas=self.ragas.evaluate(pairs),
        )
