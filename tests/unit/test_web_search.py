"""Optional web search stays disabled without an explicit key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adaspilot.config import Settings
from adaspilot.tools import web_search
from adaspilot.tools.web_search import search_adas_web


def test_web_search_is_a_noop_without_api_key() -> None:
    assert search_adas_web("ADAS radar", Settings(_env_file=None)) == []


def test_web_search_filters_domains_and_builds_web_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-key"

        def search(self, **_: object) -> dict[str, object]:
            return {
                "results": [
                    {
                        "url": "https://arxiv.org/abs/1234.5678",
                        "content": "A technical ADAS radar result.",
                    },
                    {
                        "url": "https://example.com/blog",
                        "content": "An untrusted blog result.",
                    },
                ]
            }

    monkeypatch.setattr(
        web_search.importlib,
        "import_module",
        lambda _: SimpleNamespace(TavilyClient=FakeClient),
    )

    result = search_adas_web(
        "ADAS radar",
        Settings(tavily_api_key="test-key", _env_file=None),
    )

    assert len(result) == 1
    assert result[0].source_type == "web"
    assert result[0].page_number == 0
    assert result[0].document_id.startswith("web:https://arxiv.org/")


def test_web_search_provider_failure_is_a_safe_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        web_search.importlib,
        "import_module",
        lambda _: (_ for _ in ()).throw(ImportError("missing SDK")),
    )

    assert (
        search_adas_web(
            "ADAS radar",
            Settings(tavily_api_key="test-key", _env_file=None),
        )
        == []
    )
