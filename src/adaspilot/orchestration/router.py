"""Small explainable router for the three bounded execution modes."""

from __future__ import annotations

import re

from adaspilot.domain.serving import ExecutionMode


class ExecutionRouter:
    """Route simple questions cheaply and reserve agents for multi-part work."""

    _plan_patterns = (
        r"^compare\b",
        r"^how do\b.*\band\b.*\bdiffer\b",
        r"\bmessage passing differ\b",
        r"\bused differently\b",
        r"\bversus\b|\bvs\.?\b",
        r"\brelated but different\b",
        r"\brelated\b.*\bto\b",
        r"\brelate\b.*\bto\b",
        r"\bfit\b.*\btaxonomy\b",
        r"\binstead of relying only\b",
    )
    _react_patterns = (
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bquantify\b",
        r"\bthree[- ]step\b",
        r"\bidentify\b.*\band\b",
        r"\bchoose\b.*\bjustify\b",
        r"\bapply\b.*\bthen\b",
        r"\bapply\b.*\band\b.*\bdetermine\b",
        r"\bhow much\b.*\band\b",
        r"^compare\b.*\b(?:number|total|million|score|improvement)\b",
    )

    def route(self, question: str) -> ExecutionMode:
        normalized = " ".join(question.lower().split())
        if any(re.search(pattern, normalized) for pattern in self._react_patterns):
            return ExecutionMode.REACT
        if any(re.search(pattern, normalized) for pattern in self._plan_patterns):
            return ExecutionMode.PLAN
        return ExecutionMode.DIRECT
