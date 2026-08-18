"""Explicit opt-in LangSmith tracing; local execution remains the default."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import langsmith as ls

from adaspilot.config import Settings


class LangSmithTracer:
    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.langsmith_tracing and settings.langsmith_api_key is not None
        self.project = settings.langsmith_project
        self.client = (
            ls.Client(
                api_key=settings.langsmith_api_key.get_secret_value(),
                api_url=settings.langsmith_endpoint,
            )
            if self.enabled and settings.langsmith_api_key is not None
            else None
        )

    @contextmanager
    def run(
        self,
        name: str,
        *,
        inputs: dict[str, Any],
        metadata: dict[str, Any],
    ) -> Iterator[None]:
        if not self.enabled or self.client is None:
            with nullcontext():
                yield
            return
        with (
            ls.tracing_context(
                enabled=True,
                project_name=self.project,
                client=self.client,
                metadata=metadata,
            ),
            ls.trace(
                name,
                inputs=inputs,
                metadata=metadata,
                project_name=self.project,
                client=self.client,
            ),
        ):
            yield
