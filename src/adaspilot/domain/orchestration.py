"""Typed contracts shared by the direct, ReAct, and plan workflows."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from adaspilot.domain.retrieval import StructuredAnswer


class QueryKind(StrEnum):
    FACT = "fact"
    VISUAL = "visual"
    METADATA = "metadata"
    CALCULATION = "calculation"
    COMPARISON = "comparison"
    UNSUPPORTED = "unsupported"


class WorkflowStatus(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"


class EvidenceItem(BaseModel):
    evidence_id: str
    document_id: str
    page_number: int = Field(ge=0)
    text: str
    source: str
    source_type: Literal["text", "table", "figure", "calculation", "web"] = "text"
    tool_name: str
    task_id: str | None = None
    record_id: str
    asset_id: str | None = None
    image_path: str | None = None
    support_score: float = 0.0


class NodeTrace(BaseModel):
    node: str
    outcome: str
    duration_ms: float = Field(ge=0)


class DirectResult(BaseModel):
    question: str
    query_kind: QueryKind
    retry_count: int = Field(ge=0, le=1)
    evidence: list[EvidenceItem]
    answer: StructuredAnswer
    trace: list[NodeTrace]
    terminal_reason: str


class ToolName(StrEnum):
    HYBRID_TEXT_SEARCH = "hybrid_text_search"
    VISUAL_SEARCH = "visual_search"
    PARENT_CONTEXT = "parent_context_retrieval"
    PAGE_RETRIEVAL = "page_retrieval"
    DOCUMENT_METADATA = "document_metadata_lookup"
    CALCULATOR = "evidence_linked_calculator"
    WEB_SEARCH = "search_adas_literature"
    FINISH = "finish_with_evidence"
    ABSTAIN = "abstain"


class ToolCall(BaseModel):
    name: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)
    message: str
    value: str | None = None


class ToolObservation(BaseModel):
    tool_name: ToolName
    signature: str
    evidence_ids: list[str] = Field(default_factory=list)
    new_evidence_count: int = Field(ge=0)
    message: str
    value: str | None = None
    duration_ms: float = Field(ge=0)


class ReactResult(BaseModel):
    question: str
    status: WorkflowStatus
    answer: StructuredAnswer
    evidence: list[EvidenceItem]
    observations: list[ToolObservation]
    tool_call_count: int = Field(ge=0)
    redundant_action_count: int = Field(ge=0)
    terminal_reason: str
    latency_ms: float = Field(ge=0)


class ResearchMode(StrEnum):
    TEXT = "text"
    VISUAL = "visual"
    BOTH = "both"
    METADATA = "metadata"
    CALCULATION = "calculation"


class ResearchTask(BaseModel):
    task_id: str = Field(pattern=r"^task_[a-z0-9_-]{1,40}$")
    objective: str = Field(min_length=8, max_length=500)
    mode: ResearchMode
    dependencies: list[str] = Field(default_factory=list)
    required: bool = True


class ResearchPlan(BaseModel):
    question: str
    tasks: list[ResearchTask]
    revision: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def task_ids_are_unique(self) -> ResearchPlan:
        identifiers = [task.task_id for task in self.tasks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("plan task IDs must be unique")
        return self


class TaskResult(BaseModel):
    task_id: str
    completed: bool
    evidence_ids: list[str] = Field(default_factory=list)
    tool_calls: int = Field(default=0, ge=0, le=3)
    message: str


class EvidenceConflict(BaseModel):
    task_ids: list[str]
    evidence_ids: list[str]
    reason: str


class PlanResult(BaseModel):
    question: str
    status: WorkflowStatus
    plan: ResearchPlan
    task_results: list[TaskResult]
    evidence: list[EvidenceItem]
    conflicts: list[EvidenceConflict]
    answer: StructuredAnswer
    replan_count: int = Field(ge=0, le=1)
    trace: list[NodeTrace]
    terminal_reason: str
    latency_ms: float = Field(ge=0)


class AgentRunMetrics(BaseModel):
    completed: bool
    tool_call_count: int = Field(ge=0)
    tool_call_accuracy: float = Field(ge=0, le=1)
    redundant_action_rate: float = Field(ge=0, le=1)
    evidence_gain_per_call: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    model_call_count: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)


class PlanRunMetrics(BaseModel):
    plan_valid: bool
    task_completion_rate: float = Field(ge=0, le=1)
    required_evidence_coverage: float = Field(ge=0, le=1)
    citation_valid: bool
    conflict_count: int = Field(ge=0)
    replan_count: int = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    model_call_count: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)


ExecutionMode = Literal["direct", "react", "plan"]
