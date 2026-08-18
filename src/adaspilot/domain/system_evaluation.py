"""Auditable Module 10 evaluation report contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from adaspilot.domain.serving import ExecutionMode


class SemanticEvaluationStatus(StrEnum):
    NOT_CONFIGURED = "not_configured"
    PACKAGE_MISSING = "package_missing"
    COMPLETED = "completed"
    FAILED = "failed"


class SemanticMetrics(BaseModel):
    status: SemanticEvaluationStatus
    context_precision: float | None = Field(default=None, ge=0, le=1)
    context_recall: float | None = Field(default=None, ge=0, le=1)
    faithfulness: float | None = Field(default=None, ge=0, le=1)
    multimodal_faithfulness: float | None = Field(default=None, ge=0, le=1)
    detail: str


class ModeMetrics(BaseModel):
    example_count: int = Field(ge=0)
    answerable_count: int = Field(ge=0)
    text_recall_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)
    visual_recall_at_5: float = Field(ge=0, le=1)
    citation_precision: float = Field(ge=0, le=1)
    citation_recall: float = Field(ge=0, le=1)
    context_precision: float = Field(ge=0, le=1)
    context_recall: float = Field(ge=0, le=1)
    abstention_f1: float = Field(ge=0, le=1)
    tool_call_precision: float | None = Field(default=None, ge=0, le=1)
    tool_call_recall: float | None = Field(default=None, ge=0, le=1)
    tool_call_f1: float | None = Field(default=None, ge=0, le=1)
    plan_validity: float | None = Field(default=None, ge=0, le=1)
    task_completion: float | None = Field(default=None, ge=0, le=1)
    agent_goal_accuracy: float = Field(ge=0, le=1)
    repeated_step_rate: float = Field(ge=0, le=1)
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    model_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    ragas: SemanticMetrics


class ExampleEvaluation(BaseModel):
    question_id: str
    expected_mode: ExecutionMode
    actual_mode: ExecutionMode
    answered_as_expected: bool
    retrieved_pages: list[str]
    cited_pages: list[str]
    expected_tools: list[str]
    actual_tools: list[str]
    latency_ms: float = Field(ge=0)
    terminal_reason: str


class RouterMetrics(BaseModel):
    example_count: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)


class SystemEvaluationReport(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    corpus_version: str
    benchmark_split: str
    execution_policy: Literal["benchmark_mode", "auto_router"]
    router: RouterMetrics
    modes: dict[ExecutionMode, ModeMetrics]
    examples: list[ExampleEvaluation]
    notes: list[str]
