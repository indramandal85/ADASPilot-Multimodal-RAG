"""Stable contracts shared by evaluation and the HTTP API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from adaspilot.domain.orchestration import EvidenceItem
from adaspilot.domain.retrieval import StructuredAnswer


class ExecutionMode(StrEnum):
    AUTO = "auto"
    DIRECT = "direct"
    REACT = "react"
    PLAN = "plan"


class TimelineItem(BaseModel):
    label: str
    status: str
    detail: str = ""
    duration_ms: float | None = Field(default=None, ge=0)


class ExecutionRecord(BaseModel):
    question: str
    requested_mode: ExecutionMode
    mode: ExecutionMode
    answer: StructuredAnswer
    evidence: list[EvidenceItem]
    timeline: list[TimelineItem]
    terminal_reason: str
    latency_ms: float = Field(ge=0)
    tool_names: list[str] = Field(default_factory=list)
    tool_call_count: int = Field(default=0, ge=0)
    repeated_step_count: int = Field(default=0, ge=0)
    plan_valid: bool | None = None
    task_completion_rate: float | None = Field(default=None, ge=0, le=1)
