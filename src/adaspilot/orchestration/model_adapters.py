"""Optional structured-output model policies for ReAct actions and research plans."""

from __future__ import annotations

import json
from typing import Any

import httpx

from adaspilot.domain.orchestration import ResearchPlan, ToolCall
from adaspilot.observability.usage import UsageTracker
from adaspilot.orchestration.planning import Planner
from adaspilot.orchestration.react import ActionPolicy, ReactState


class StructuredModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float,
        usage_tracker: UsageTracker | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.usage_tracker = usage_tracker

    def complete_json(self, system: str, payload: dict[str, Any]) -> str:
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload)},
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        response_payload = response.json()
        if self.usage_tracker is not None:
            self.usage_tracker.record(response_payload)
        return str(response_payload["choices"][0]["message"]["content"])


class OpenAICompatibleActionPolicy(ActionPolicy):
    """Select exactly one typed allowlisted action; tool execution stays outside the model."""

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        allow_web_search: bool = False,
    ) -> None:
        self.client = client
        self.allow_web_search = allow_web_search

    def select(self, state: ReactState) -> ToolCall:
        payload = {
            "question": state["question"],
            "query_kind": state["query_kind"].value,
            "tool_call_count": state["tool_call_count"],
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "document_id": item.document_id,
                    "page_number": item.page_number,
                    "text": item.text[:400],
                }
                for item in state["evidence"][:8]
            ],
            "observations": [
                observation.model_dump(mode="json") for observation in state["observations"]
            ],
        }
        allowed = (
            "hybrid_text_search, visual_search, parent_context_retrieval, page_retrieval, "
            "document_metadata_lookup, evidence_linked_calculator, finish_with_evidence, abstain"
        )
        if self.allow_web_search:
            allowed += ", search_adas_literature"
        raw = self.client.complete_json(
            (
                "Return one JSON tool call with keys name and arguments. "
                f"Allowed names: {allowed}. "
                "Use only supplied evidence IDs. Web search is permitted only when a corpus "
                "search has returned no evidence. Do not include reasoning or prose."
            ),
            payload,
        )
        return ToolCall.model_validate_json(raw)


class OpenAICompatiblePlanner(Planner):
    """Create a typed plan only; validation and execution remain deterministic."""

    def __init__(self, client: StructuredModelClient, max_tasks: int = 5) -> None:
        self.client = client
        self.max_tasks = max_tasks

    def create_plan(self, question: str) -> ResearchPlan:
        raw = self.client.complete_json(
            (
                f"Create a research plan with at most {self.max_tasks} tasks. Return JSON matching "
                "{question, tasks, revision}. Each task needs task_id beginning task_, objective, "
                "mode (text, visual, both, metadata, calculation), dependencies, and required. "
                "Do not answer the question."
            ),
            {"question": question, "revision": 0},
        )
        return ResearchPlan.model_validate_json(raw)

    def replan(self, plan: ResearchPlan, failed_task_ids: list[str]) -> ResearchPlan:
        raw = self.client.complete_json(
            (
                "Revise only failed research tasks once. Preserve completed task IDs and valid "
                "dependencies. Return the complete ResearchPlan JSON with revision set to 1."
            ),
            {
                "plan": plan.model_dump(mode="json"),
                "failed_task_ids": failed_task_ids,
            },
        )
        revised = ResearchPlan.model_validate_json(raw)
        return revised.model_copy(update={"revision": 1})
