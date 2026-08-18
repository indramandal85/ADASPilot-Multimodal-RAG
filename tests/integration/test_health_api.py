"""FastAPI health endpoint integration tests."""

from __future__ import annotations

import httpx
import pytest

from adaspilot.api.app import create_app
from adaspilot.config import Environment, Settings


def _settings() -> Settings:
    return Settings(environment=Environment.TEST, _env_file=None)


@pytest.mark.asyncio
async def test_health_endpoint_reports_application_status() -> None:
    app = create_app(settings=_settings())
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "application": "ADASPilot",
        "version": "0.1.0",
        "status": "ok",
    }
