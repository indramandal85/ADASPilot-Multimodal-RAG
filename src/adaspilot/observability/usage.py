"""Thread-safe token and cost accounting for optional model calls."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class UsageSnapshot:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __sub__(self, other: UsageSnapshot) -> UsageSnapshot:
        return UsageSnapshot(
            model_calls=max(0, self.model_calls - other.model_calls),
            input_tokens=max(0, self.input_tokens - other.input_tokens),
            output_tokens=max(0, self.output_tokens - other.output_tokens),
            estimated_cost_usd=max(0.0, self.estimated_cost_usd - other.estimated_cost_usd),
        )


class UsageTracker:
    """Accumulates provider-reported usage without depending on one provider SDK."""

    def __init__(
        self,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self._lock = Lock()
        self._snapshot = UsageSnapshot()

    def record(self, payload: dict[str, Any]) -> None:
        usage = payload.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        cost = (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000
        with self._lock:
            current = self._snapshot
            self._snapshot = UsageSnapshot(
                model_calls=current.model_calls + 1,
                input_tokens=current.input_tokens + input_tokens,
                output_tokens=current.output_tokens + output_tokens,
                estimated_cost_usd=current.estimated_cost_usd + cost,
            )

    def snapshot(self) -> UsageSnapshot:
        with self._lock:
            return self._snapshot
