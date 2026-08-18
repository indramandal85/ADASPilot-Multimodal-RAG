"""Structured, cited answer generation for text and visual evidence."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import httpx

from adaspilot.domain.retrieval import (
    AnswerCitation,
    AnswerStatus,
    SearchHit,
    StructuredAnswer,
)
from adaspilot.observability.usage import UsageTracker
from adaspilot.safety.guardrails import ImageGuard


class Answerer(Protocol):
    def answer(self, query: str, hits: Sequence[SearchHit]) -> StructuredAnswer: ...


def _citations(hits: Sequence[SearchHit], limit: int = 3) -> list[AnswerCitation]:
    citations: list[AnswerCitation] = []
    seen: set[tuple[str, int, str]] = set()
    for hit in hits:
        key = (hit.document_id, hit.page_number, hit.record_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            AnswerCitation(
                document_id=hit.document_id,
                page_number=hit.page_number,
                record_id=hit.record_id,
                asset_id=hit.asset_id,
            )
        )
        if len(citations) == limit:
            break
    return citations


class ExtractiveAnswerer:
    """No-LLM baseline that keeps answer evaluation reproducible."""

    def answer(self, query: str, hits: Sequence[SearchHit]) -> StructuredAnswer:
        if not hits:
            return StructuredAnswer(
                answer="The selected corpus does not contain enough evidence to answer.",
                status=AnswerStatus.ABSTAINED,
                retrieval_mode="extractive",
            )
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        candidates: list[tuple[float, str]] = []
        for hit in hits[:5]:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", hit.text):
                sentence = " ".join(sentence.split())
                if len(sentence) < 25:
                    continue
                terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                score = len(query_terms & terms) / max(1, len(query_terms))
                candidates.append((score, sentence))
        selected = [text for _, text in sorted(candidates, reverse=True)[:2]]
        if not selected:
            selected = [" ".join(hits[0].text.split())[:600]]
        return StructuredAnswer(
            answer=" ".join(selected),
            status=AnswerStatus.ANSWERED,
            citations=_citations(hits),
            retrieval_mode="extractive",
        )


class OpenAICompatibleAnswerer:
    """One small HTTP adapter for text generation, HyDE, and vision-grounded answers."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 60,
        image_guard: ImageGuard | None = None,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.image_guard = image_guard
        self.usage_tracker = usage_tracker

    def _complete(self, messages: list[dict[str, object]]) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "temperature": 0},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if self.usage_tracker is not None:
            self.usage_tracker.record(payload)
        return str(payload["choices"][0]["message"]["content"])

    def generate(self, query: str) -> str:
        return self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Write a short hypothetical passage that would answer the retrieval query. "
                        "Do not invent numeric results or citations."
                    ),
                },
                {"role": "user", "content": query},
            ]
        )

    def answer(self, query: str, hits: Sequence[SearchHit]) -> StructuredAnswer:
        if not hits:
            return StructuredAnswer(
                answer="The selected corpus does not contain enough evidence to answer.",
                status=AnswerStatus.ABSTAINED,
                retrieval_mode="llm_text",
            )
        context = "\n\n".join(
            f"[{index}] evidence_id={hit.record_id} | source={hit.document_id} "
            f"page={hit.page_number}\n{hit.text}"
            for index, hit in enumerate(hits[:6], start=1)
        )
        answer = self._complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using only the evidence items listed by exact evidence_id. Every "
                        "factual claim must be supported by an item in that list. Do not mention "
                        "a page, document, or source outside the list. If the evidence is "
                        "insufficient, say so explicitly. Keep the answer concise. Citations are "
                        "attached programmatically from the same evidence board."
                    ),
                },
                {"role": "user", "content": f"Question: {query}\n\nEvidence:\n{context}"},
            ]
        )
        return StructuredAnswer(
            answer=answer,
            status=AnswerStatus.ANSWERED,
            citations=_citations(hits),
            retrieval_mode="llm_text",
        )

    def answer_visual(
        self,
        query: str,
        hits: Sequence[SearchHit],
        build_dir: Path,
    ) -> StructuredAnswer:
        visual_hits = [hit for hit in hits if hit.image_path]
        if not visual_hits:
            return self.answer(query, hits)
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    f"Answer only from these retrieved document pages or figures. Question: {query}"
                ),
            }
        ]
        for hit in visual_hits[:3]:
            image_path = build_dir / str(hit.image_path)
            if self.image_guard is not None:
                self.image_guard.validate(image_path)
            encoded = base64.b64encode(image_path.read_bytes()).decode()
            media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{encoded}"},
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": f"Source: {hit.document_id}, page {hit.page_number}",
                }
            )
        answer = self._complete(
            [
                {
                    "role": "system",
                    "content": "Use only visible evidence and never invent a figure citation.",
                },
                {"role": "user", "content": content},
            ]
        )
        return StructuredAnswer(
            answer=answer,
            status=AnswerStatus.ANSWERED,
            citations=_citations(visual_hits),
            retrieval_mode="llm_vision",
        )


def answer_to_json(answer: StructuredAnswer) -> str:
    return json.dumps(answer.model_dump(mode="json"), indent=2)
