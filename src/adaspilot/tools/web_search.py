"""Domain-bounded ADAS literature search for the ReAct fallback path."""

from __future__ import annotations

import hashlib
import importlib
import logging
from typing import Any
from urllib.parse import urlparse

from adaspilot.config import Settings
from adaspilot.domain.orchestration import EvidenceItem, ToolName

LOGGER = logging.getLogger(__name__)

ALLOWED_DOMAINS = [
    "arxiv.org",
    "ieeexplore.ieee.org",
    "sae.org",
    "nhtsa.gov",
    "euro-ncap.com",
    "researchgate.net",
]


def _allowed_url(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in ALLOWED_DOMAINS)


def search_adas_web(query: str, settings: Settings) -> list[EvidenceItem]:
    """Return at most three conservatively scored technical web results."""

    if settings.tavily_api_key is None:
        return []
    try:
        module = importlib.import_module("tavily")
        client_class = module.TavilyClient
        client = client_class(api_key=settings.tavily_api_key.get_secret_value())
        payload: dict[str, Any] = client.search(
            query=query,
            include_domains=ALLOWED_DOMAINS,
            max_results=3,
            search_depth="basic",
        )
        evidence: list[EvidenceItem] = []
        for result in payload.get("results", [])[:3]:
            url = str(result.get("url", "")).strip()
            content = " ".join(str(result.get("content", "")).split())[:500]
            if not url or not content or not _allowed_url(url):
                continue
            record_id = f"web_{hashlib.sha256(url.encode()).hexdigest()[:20]}"
            evidence.append(
                EvidenceItem(
                    evidence_id=record_id,
                    document_id=f"web:{url}",
                    page_number=0,
                    text=content,
                    source="web",
                    source_type="web",
                    tool_name=ToolName.WEB_SEARCH.value,
                    record_id=record_id,
                    support_score=0.5,
                )
            )
        return evidence
    except Exception as exc:  # The optional SDK may raise provider-specific exception types.
        LOGGER.warning("Bounded web search is unavailable: %s", exc)
        return []
