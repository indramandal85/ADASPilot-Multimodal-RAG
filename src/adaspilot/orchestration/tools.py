"""Allowlisted read-only research tools shared by ReAct and plan workers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from adaspilot.domain.ingestion import ChunkRecord, DocumentRecord, PageRecord
from adaspilot.domain.orchestration import EvidenceItem, ToolCall, ToolName, ToolResult
from adaspilot.domain.retrieval import SearchHit
from adaspilot.retrieval.text import Retriever
from adaspilot.retrieval.visual import VisualIndex
from adaspilot.safety.guardrails import (
    AccessPolicy,
    GuardrailViolation,
    InjectionDetector,
    PiiRedactor,
    ToolPermissionPolicy,
    safe_calculate,
)

ToolHandler = Callable[[dict[str, Any]], ToolResult]


def normalized_tool_signature(call: ToolCall) -> str:
    payload = json.dumps(call.arguments, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(f"{call.name.value}:{payload}".encode()).hexdigest()[:20]


def hit_to_evidence(
    hit: SearchHit, tool_name: ToolName, task_id: str | None = None
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=hit.record_id,
        document_id=hit.document_id,
        page_number=hit.page_number,
        text=hit.text,
        source=hit.source,
        tool_name=tool_name.value,
        task_id=task_id,
        record_id=hit.record_id,
        asset_id=hit.asset_id,
        image_path=hit.image_path,
        support_score=hit.score,
    )


class ToolRegistry:
    """Executes only registered tools with role permissions and a timeout."""

    def __init__(
        self,
        handlers: dict[ToolName, ToolHandler],
        *,
        timeout_seconds: float = 15,
        permissions: ToolPermissionPolicy | None = None,
    ) -> None:
        self.handlers = handlers
        self.timeout_seconds = timeout_seconds
        self.permissions = permissions or ToolPermissionPolicy()

    def execute(self, role: str, call: ToolCall) -> tuple[ToolResult, float]:
        self.permissions.ensure_allowed(role, call.name)
        handler = self.handlers.get(call.name)
        if handler is None:
            raise GuardrailViolation(f"tool {call.name.value!r} is not registered")
        executor = ThreadPoolExecutor(max_workers=1)
        started = time.perf_counter()
        future = executor.submit(handler, call.arguments)
        try:
            result = future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"tool {call.name.value} timed out") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return result, (time.perf_counter() - started) * 1000


class ProjectToolbox:
    """Binds the allowlisted tool names to local corpus and retrieval objects."""

    def __init__(
        self,
        *,
        text_retriever: Retriever,
        visual_index: VisualIndex | None,
        build_dir: Path,
        pages: Sequence[PageRecord],
        documents: Sequence[DocumentRecord],
        parent_chunks: Sequence[ChunkRecord],
        child_chunks: Sequence[ChunkRecord],
        document_titles: dict[str, str] | None = None,
        web_searcher: Callable[[str], list[EvidenceItem]] | None = None,
        access_policy: AccessPolicy | None = None,
        timeout_seconds: float = 15,
    ) -> None:
        self.text_retriever = text_retriever
        self.visual_index = visual_index
        self.build_dir = build_dir
        self.pages = {(page.document_id, page.page_number): page for page in pages}
        self.documents = {document.document_id: document for document in documents}
        self.parents = {chunk.chunk_id: chunk for chunk in parent_chunks}
        self.children = {chunk.chunk_id: chunk for chunk in child_chunks}
        self.document_titles = document_titles or {}
        self.web_searcher = web_searcher
        self.web_search_available = web_searcher is not None
        self.access_policy = access_policy or AccessPolicy()
        self.detector = InjectionDetector()
        self.pii_redactor = PiiRedactor()
        handlers: dict[ToolName, ToolHandler] = {
            ToolName.HYBRID_TEXT_SEARCH: self._hybrid_text_search,
            ToolName.VISUAL_SEARCH: self._visual_search,
            ToolName.PARENT_CONTEXT: self._parent_context,
            ToolName.PAGE_RETRIEVAL: self._page_retrieval,
            ToolName.DOCUMENT_METADATA: self._document_metadata,
            ToolName.CALCULATOR: self._calculator,
        }
        if self.web_search_available:
            handlers[ToolName.WEB_SEARCH] = self._web_search
        self.registry = ToolRegistry(
            handlers,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _query(arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise GuardrailViolation("search query cannot be empty")
        return query

    @staticmethod
    def _top_k(arguments: dict[str, Any]) -> int:
        return max(1, min(int(arguments.get("top_k", 5)), 10))

    def _safe_evidence(self, hits: Sequence[SearchHit], tool: ToolName) -> list[EvidenceItem]:
        allowed = self.access_policy.filter_hits(hits)
        return [
            hit_to_evidence(
                hit.model_copy(update={"text": self.pii_redactor.redact(hit.text)[0]}), tool
            )
            for hit in allowed
            if not self.detector.detect(hit.text)
        ]

    def _hybrid_text_search(self, arguments: dict[str, Any]) -> ToolResult:
        top_k = self._top_k(arguments)
        hits = self.text_retriever.retrieve(self._query(arguments), top_k * 4)
        evidence = self._safe_evidence(hits, ToolName.HYBRID_TEXT_SEARCH)[:top_k]
        return ToolResult(evidence=evidence, message=f"Retrieved {len(evidence)} text records")

    def _visual_search(self, arguments: dict[str, Any]) -> ToolResult:
        if self.visual_index is None:
            return ToolResult(message="Visual index is unavailable")
        top_k = self._top_k(arguments)
        hits = self.visual_index.search(self._query(arguments), top_k * 4)
        evidence = self._safe_evidence(hits, ToolName.VISUAL_SEARCH)[:top_k]
        return ToolResult(evidence=evidence, message=f"Retrieved {len(evidence)} visual records")

    def _parent_context(self, arguments: dict[str, Any]) -> ToolResult:
        raw_ids = arguments.get("chunk_ids", [])
        if not isinstance(raw_ids, list):
            raise GuardrailViolation("chunk_ids must be a list")
        parent_ids = {
            self.children[str(chunk_id)].parent_id
            for chunk_id in raw_ids[:10]
            if str(chunk_id) in self.children
        }
        evidence = [
            EvidenceItem(
                evidence_id=parent.chunk_id,
                document_id=parent.document_id,
                page_number=parent.page_number,
                text=self.pii_redactor.redact(parent.text)[0],
                source="parent",
                tool_name=ToolName.PARENT_CONTEXT.value,
                record_id=parent.chunk_id,
            )
            for parent_id in parent_ids
            if parent_id is not None
            and (parent := self.parents.get(parent_id)) is not None
            and self.access_policy.allows(parent.document_id)
            and not self.detector.detect(parent.text)
        ]
        return ToolResult(evidence=evidence, message=f"Resolved {len(evidence)} parent records")

    def _page_retrieval(self, arguments: dict[str, Any]) -> ToolResult:
        document_id = str(arguments.get("document_id", ""))
        page_number = int(arguments.get("page_number", 0))
        if not self.access_policy.allows(document_id):
            raise GuardrailViolation("requested document is outside the access policy")
        page = self.pages.get((document_id, page_number))
        if page is None:
            return ToolResult(message="Requested page was not found")
        if self.detector.detect(page.text):
            raise GuardrailViolation("retrieved page contains instruction-like content")
        evidence = EvidenceItem(
            evidence_id=page.page_id,
            document_id=page.document_id,
            page_number=page.page_number,
            text=self.pii_redactor.redact(page.text)[0],
            source="page",
            tool_name=ToolName.PAGE_RETRIEVAL.value,
            record_id=page.page_id,
            image_path=page.rendered_image_path,
        )
        return ToolResult(evidence=[evidence], message="Retrieved one source page")

    def _document_metadata(self, arguments: dict[str, Any]) -> ToolResult:
        requested = str(arguments.get("document_id", "")).strip()
        selected = [
            document
            for document in self.documents.values()
            if self.access_policy.allows(document.document_id)
            and (not requested or document.document_id == requested)
        ][:10]
        evidence = [
            EvidenceItem(
                evidence_id=f"metadata_{document.document_id}",
                document_id=document.document_id,
                page_number=1,
                text=json.dumps(
                    {
                        "title": self.document_titles.get(
                            document.document_id, document.document_id
                        ),
                        "page_count": document.page_count,
                        "source_sha256": document.source_sha256,
                        "document_version": document.document_version,
                    },
                    sort_keys=True,
                ),
                source="metadata",
                tool_name=ToolName.DOCUMENT_METADATA.value,
                record_id=f"metadata_{document.document_id}",
            )
            for document in selected
        ]
        return ToolResult(evidence=evidence, message=f"Retrieved {len(evidence)} metadata records")

    def _calculator(self, arguments: dict[str, Any]) -> ToolResult:
        evidence_ids = arguments.get("evidence_ids", [])
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise GuardrailViolation("calculator requires at least one evidence ID")
        expression = str(arguments.get("expression", ""))
        value = safe_calculate(expression)
        return ToolResult(message="Calculated from cited evidence", value=f"{value:g}")

    def _web_search(self, arguments: dict[str, Any]) -> ToolResult:
        if self.web_searcher is None:
            return ToolResult(message="Bounded web search is unavailable")
        raw = self.web_searcher(self._query(arguments))
        evidence = [
            item.model_copy(update={"text": self.pii_redactor.redact(item.text)[0]})
            for item in raw[:3]
            if item.source_type == "web" and not self.detector.detect(item.text)
        ]
        return ToolResult(
            evidence=evidence,
            message=f"Retrieved {len(evidence)} domain-filtered web records",
        )
