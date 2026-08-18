"""Run the FastAPI application through the installed console script."""

from __future__ import annotations

import uvicorn

from adaspilot.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "adaspilot.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.environment.value == "development",
    )


if __name__ == "__main__":
    main()
