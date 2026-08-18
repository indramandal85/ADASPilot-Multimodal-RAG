"""FastAPI query and SSE contract tests with orchestration replaced at the boundary."""

from __future__ import annotations

from typing import cast

import httpx
import pytest

from adaspilot.api.app import create_app
from adaspilot.api.runtime import RuntimeManager
from adaspilot.config import Environment, Settings
from adaspilot.domain.orchestration import EvidenceItem
from adaspilot.domain.retrieval import AnswerCitation, AnswerStatus, StructuredAnswer
from adaspilot.domain.serving import ExecutionMode, ExecutionRecord, TimelineItem


class FakeRunner:
    def run(self, question: str, mode: ExecutionMode) -> ExecutionRecord:
        evidence = EvidenceItem(
            evidence_id="chunk_1",
            document_id="radar_doc",
            page_number=4,
            text="Radar measures range.",
            source="text",
            tool_name="direct_retrieval",
            record_id="chunk_1",
            support_score=0.9,
        )
        return ExecutionRecord(
            question=question,
            requested_mode=mode,
            mode=ExecutionMode.DIRECT,
            answer=StructuredAnswer(
                answer="Radar measures range.",
                status=AnswerStatus.ANSWERED,
                citations=[
                    AnswerCitation(
                        document_id="radar_doc",
                        page_number=4,
                        record_id="chunk_1",
                    )
                ],
                retrieval_mode="direct_corrective",
            ),
            evidence=[evidence],
            timeline=[TimelineItem(label="retrieve", status="1 hit", duration_ms=2)],
            terminal_reason="verified_answer",
            latency_ms=5,
        )


class FakeManager:
    def runner(self) -> FakeRunner:
        return FakeRunner()

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_query_and_streaming_endpoints_share_one_response_contract() -> None:
    app = create_app(
        settings=Settings(environment=Environment.TEST, stream_chunk_chars=20, _env_file=None),
        runtime_manager=cast(RuntimeManager, FakeManager()),
    )
    transport = httpx.ASGITransport(app=app)
    payload = {"question": "What does radar measure?", "mode": "auto"}

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json=payload)
        streamed = await client.post("/query/stream", json=payload)

    assert response.status_code == 200
    assert response.json()["mode"] == "direct"
    assert response.json()["answer"]["citations"][0]["page_number"] == 4
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert "event: execution" in streamed.text
    assert "event: timeline" in streamed.text
    assert "event: answer" in streamed.text
    assert "event: evidence" in streamed.text
    assert "event: done" in streamed.text
