"""Focused tests for routing and provider-reported usage accounting."""

from __future__ import annotations

import pytest

from adaspilot.domain.serving import ExecutionMode
from adaspilot.observability.usage import UsageTracker
from adaspilot.orchestration.router import ExecutionRouter


def test_execution_router_uses_explainable_complexity_rules() -> None:
    router = ExecutionRouter()

    assert router.route("What does automotive radar measure?") == ExecutionMode.DIRECT
    assert router.route("Calculate 15 - 10 and explain the result.") == ExecutionMode.REACT
    assert router.route("Compare radar and camera strengths.") == ExecutionMode.PLAN


def test_usage_tracker_records_provider_usage_and_cost() -> None:
    tracker = UsageTracker(input_cost_per_million=2.0, output_cost_per_million=8.0)
    before = tracker.snapshot()
    tracker.record({"usage": {"prompt_tokens": 1000, "completion_tokens": 500}})
    delta = tracker.snapshot() - before

    assert delta.model_calls == 1
    assert delta.input_tokens == 1000
    assert delta.output_tokens == 500
    assert delta.estimated_cost_usd == pytest.approx(0.006)
