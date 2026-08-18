"""Optional structured model policies remain typed and provider-neutral."""

from __future__ import annotations

from adaspilot.domain.orchestration import QueryKind, ToolName, WorkflowStatus
from adaspilot.orchestration.model_adapters import (
    OpenAICompatibleActionPolicy,
    OpenAICompatiblePlanner,
    StructuredModelClient,
)
from adaspilot.orchestration.react import ReactState


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": '{"name":"abstain","arguments":{}}'}}]}


def test_structured_client_requests_json(monkeypatch: object) -> None:
    captured: list[dict[str, object]] = []

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        captured.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "adaspilot.orchestration.model_adapters.httpx.post", fake_post
    )
    client = StructuredModelClient(
        base_url="http://localhost:1234/v1/",
        model="local-model",
        api_key="secret",
        timeout_seconds=2,
    )

    raw = client.complete_json("Return JSON", {"question": "test"})

    assert '"abstain"' in raw
    request = captured[0]["json"]
    assert request["response_format"] == {"type": "json_object"}  # type: ignore[index]


def test_model_action_and_plan_outputs_are_validated(monkeypatch: object) -> None:
    client = StructuredModelClient(
        base_url="http://localhost",
        model="model",
        api_key="key",
        timeout_seconds=2,
    )
    outputs = iter(
        [
            '{"name":"hybrid_text_search","arguments":{"query":"radar","top_k":5}}',
            (
                '{"question":"Compare radar and camera","tasks":['
                '{"task_id":"task_radar","objective":"Research radar evidence safely",'
                '"mode":"text","dependencies":[],"required":true}],"revision":0}'
            ),
            (
                '{"question":"Compare radar and camera","tasks":['
                '{"task_id":"task_radar","objective":"Retry radar evidence safely",'
                '"mode":"text","dependencies":[],"required":true}],"revision":1}'
            ),
        ]
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        client, "complete_json", lambda _system, _payload: next(outputs)
    )
    state: ReactState = {
        "question": "Research radar",
        "query_kind": QueryKind.FACT,
        "evidence": [],
        "observations": [],
        "signature_counts": {},
        "tool_call_count": 0,
        "redundant_action_count": 0,
        "consecutive_no_gain": 0,
        "answer": None,
        "status": WorkflowStatus.ABSTAINED,
        "terminal_reason": "",
        "done": False,
    }
    action = OpenAICompatibleActionPolicy(client).select(state)
    planner = OpenAICompatiblePlanner(client)
    plan = planner.create_plan("Compare radar and camera")
    revised = planner.replan(plan, ["task_radar"])

    assert action.name == ToolName.HYBRID_TEXT_SEARCH
    assert plan.tasks[0].task_id == "task_radar"
    assert revised.revision == 1
