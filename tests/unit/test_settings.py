"""Configuration contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adaspilot.config import Environment, Settings


def test_settings_have_safe_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment == Environment.DEVELOPMENT
    assert settings.api_host == "127.0.0.1"
    assert settings.api_port == 8000


def test_log_level_is_normalized() -> None:
    settings = Settings(log_level="warning", _env_file=None)

    assert settings.log_level == "WARNING"


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        Settings(log_level="verbose", _env_file=None)
