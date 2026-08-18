"""Small explainable guardrails for inputs, tools, images, and citations."""

from __future__ import annotations

import ast
import operator
import re
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from PIL import Image

from adaspilot.domain.orchestration import EvidenceItem, ToolName
from adaspilot.domain.retrieval import SearchHit, StructuredAnswer


class GuardrailViolation(ValueError):
    """Raised when deterministic policy rejects an operation."""


class PiiRedactor:
    _PATTERNS = [
        (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[EMAIL]"),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_ADDRESS]"),
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD_NUMBER]"),
        (re.compile(r"(?<![\d.])(?:\+?\d[\d ()-]{7,}\d)(?![\d.])"), "[PHONE]"),
    ]

    def redact(self, text: str) -> tuple[str, list[str]]:
        redacted = text
        labels: list[str] = []
        for pattern, replacement in self._PATTERNS:
            redacted, count = pattern.subn(replacement, redacted)
            if count:
                labels.append(replacement.strip("[]"))
        return redacted, labels


class InjectionDetector:
    """Pattern detector for user, retrieved-text, caption, and OCR instructions."""

    _RULES = {
        "instruction_override": re.compile(
            r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|system|developer)\b",
            re.I | re.S,
        ),
        "prompt_disclosure": re.compile(
            r"\b(reveal|print|show|leak)\b.{0,40}\b(system prompt|hidden instructions|secrets?)\b",
            re.I | re.S,
        ),
        "role_impersonation": re.compile(r"\b(system|developer)\s*(message|prompt)\s*:", re.I),
        "safety_bypass": re.compile(
            r"\b(disable|bypass|remove|ignore)\b.{0,30}\b(safety|guardrails?|policy|filters?)\b",
            re.I | re.S,
        ),
        "tool_injection": re.compile(
            r"\b(call|execute|invoke|run)\b.{0,30}\b(shell|terminal|tool|command|python)\b",
            re.I | re.S,
        ),
        "answer_hijack": re.compile(
            r"\b(do not answer|instead answer|respond only with|follow these instructions)\b",
            re.I,
        ),
    }

    def detect(self, text: str) -> list[str]:
        return [name for name, pattern in self._RULES.items() if pattern.search(text)]

    def ensure_safe(self, text: str, *, source: str) -> None:
        findings = self.detect(text)
        if findings:
            raise GuardrailViolation(f"{source} prompt injection detected: {', '.join(findings)}")


class InputGuard:
    def __init__(self, max_chars: int, detector: InjectionDetector | None = None) -> None:
        self.max_chars = max_chars
        self.detector = detector or InjectionDetector()
        self.redactor = PiiRedactor()

    def validate(self, text: str) -> str:
        normalized = " ".join(text.split())
        if not normalized:
            raise GuardrailViolation("question cannot be empty")
        if len(normalized) > self.max_chars:
            raise GuardrailViolation(f"question exceeds {self.max_chars} characters")
        self.detector.ensure_safe(normalized, source="user")
        return self.redactor.redact(normalized)[0]


class AccessPolicy:
    """Document allowlist enforced after retrieval and before model/tool output."""

    def __init__(self, allowed_document_ids: set[str] | None = None) -> None:
        self.allowed_document_ids = allowed_document_ids

    def allows(self, document_id: str) -> bool:
        return self.allowed_document_ids is None or document_id in self.allowed_document_ids

    def filter_hits(self, hits: Sequence[SearchHit]) -> list[SearchHit]:
        return [hit for hit in hits if self.allows(hit.document_id)]

    def filter_evidence(self, evidence: Sequence[EvidenceItem]) -> list[EvidenceItem]:
        return [item for item in evidence if self.allows(item.document_id)]


class ToolPermissionPolicy:
    _ROLE_TOOLS = {
        "react": {
            ToolName.HYBRID_TEXT_SEARCH,
            ToolName.VISUAL_SEARCH,
            ToolName.PARENT_CONTEXT,
            ToolName.PAGE_RETRIEVAL,
            ToolName.DOCUMENT_METADATA,
            ToolName.CALCULATOR,
            ToolName.WEB_SEARCH,
        },
        "text_researcher": {
            ToolName.HYBRID_TEXT_SEARCH,
            ToolName.PARENT_CONTEXT,
            ToolName.PAGE_RETRIEVAL,
        },
        "visual_researcher": {ToolName.VISUAL_SEARCH, ToolName.PAGE_RETRIEVAL},
        "metadata_researcher": {ToolName.DOCUMENT_METADATA},
        "calculator": {ToolName.CALCULATOR},
    }

    def ensure_allowed(self, role: str, tool_name: ToolName) -> None:
        if tool_name not in self._ROLE_TOOLS.get(role, set()):
            raise GuardrailViolation(f"role {role!r} cannot call {tool_name.value!r}")


class CitationValidator:
    def validate(self, answer: StructuredAnswer, evidence: Sequence[EvidenceItem]) -> list[str]:
        valid = {
            (item.document_id, item.page_number, item.record_id, item.asset_id) for item in evidence
        }
        issues: list[str] = []
        for citation in answer.citations:
            key = (
                citation.document_id,
                citation.page_number,
                citation.record_id,
                citation.asset_id,
            )
            if key not in valid:
                issues.append(
                    f"unknown citation {citation.document_id}:p{citation.page_number}:"
                    f"{citation.record_id}"
                )
        if answer.status.value == "answered" and not answer.citations:
            issues.append("answered response has no citations")
        return issues


class ImageGuard:
    def __init__(self, max_bytes: int, max_pixels: int) -> None:
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels

    def validate(self, path: Path) -> tuple[int, int]:
        if path.stat().st_size > self.max_bytes:
            raise GuardrailViolation("image exceeds byte-size limit")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    width, height = image.size
                    if width * height > self.max_pixels:
                        raise GuardrailViolation("image exceeds decoded-pixel limit")
                    image.verify()
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise GuardrailViolation("image decompression protection triggered") from exc
        except OSError as exc:
            raise GuardrailViolation("malformed image") from exc
        return width, height


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> float:
    """Evaluate only numeric arithmetic; names, calls, and attributes are rejected."""

    tree = ast.parse(expression, mode="eval")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if abs(left) > 1e12 or abs(right) > 1e12:
                raise GuardrailViolation("calculator input is outside the safe range")
            return float(_BINARY_OPERATORS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return float(_UNARY_OPERATORS[type(node.op)](evaluate(node.operand)))
        raise GuardrailViolation("calculator accepts numeric arithmetic only")

    result = evaluate(tree)
    if not (-1e15 <= result <= 1e15):
        raise GuardrailViolation("calculator result is outside the safe range")
    return result
