"""Lazy API composition so health and corpus routes stay lightweight."""

from __future__ import annotations

from threading import Lock

from adaspilot.config import Settings
from adaspilot.orchestration.runtime import WorkflowBundle, build_workflows
from adaspilot.orchestration.service import WorkflowRunner


class RuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._bundle: WorkflowBundle | None = None
        self._runner: WorkflowRunner | None = None

    def runner(self) -> WorkflowRunner:
        if self._runner is not None:
            return self._runner
        with self._lock:
            if self._runner is None:
                self._bundle = build_workflows(self.settings)
                self._runner = WorkflowRunner(self._bundle)
        return self._runner

    def close(self) -> None:
        with self._lock:
            if self._bundle is not None:
                self._bundle.close()
            self._bundle = None
            self._runner = None
