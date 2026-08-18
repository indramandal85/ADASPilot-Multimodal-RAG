"""Tests for the optional OpenAI-compatible text and vision adapter."""

from __future__ import annotations

from pathlib import Path

from adaspilot.domain.retrieval import AnswerStatus, SearchHit
from adaspilot.retrieval.answers import OpenAICompatibleAnswerer, answer_to_json


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": "Grounded response."}}]}


def test_text_hyde_and_visual_answer(monkeypatch: object, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr("adaspilot.retrieval.answers.httpx.post", fake_post)  # type: ignore[attr-defined]
    answerer = OpenAICompatibleAnswerer(
        base_url="http://localhost:1234/v1/",
        model="local-model",
        api_key="test-key",
    )
    hit = SearchHit(
        record_id="asset_1",
        score=0.9,
        document_id="sample_doc",
        page_number=3,
        text="Radar evidence.",
        asset_id="asset_1",
        image_path="figure.jpg",
        source="visual",
    )
    (tmp_path / "figure.jpg").write_bytes(b"jpeg-bytes")

    assert answerer.generate("radar query") == "Grounded response."
    text_answer = answerer.answer("What is radar?", [hit])
    visual_answer = answerer.answer_visual("Describe the figure", [hit], tmp_path)
    fallback = answerer.answer_visual(
        "What is radar?", [hit.model_copy(update={"image_path": None})], tmp_path
    )
    abstained = answerer.answer("unknown", [])

    assert text_answer.status == AnswerStatus.ANSWERED
    assert visual_answer.retrieval_mode == "llm_vision"
    assert fallback.retrieval_mode == "llm_text"
    assert abstained.status == AnswerStatus.ABSTAINED
    assert '"page_number": 3' in answer_to_json(text_answer)
    assert len(calls) == 4
