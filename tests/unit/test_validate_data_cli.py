"""Data-validation CLI tests."""

from __future__ import annotations

from pathlib import Path

from adaspilot.cli.validate_data import run


def _arguments(*extra: str) -> list[str]:
    root = Path(__file__).parents[2]
    return [
        "--manifest",
        str(root / "data/corpus/manifest.json"),
        "--benchmark",
        str(root / "data/evaluation/benchmark.jsonl"),
        "--targets",
        str(root / "data/evaluation/targets.json"),
        *extra,
    ]


def test_cli_accepts_source_verified_benchmark() -> None:
    assert run(_arguments()) == 0


def test_cli_strict_mode_accepts_minimum_verified_target() -> None:
    assert run(_arguments("--require-target")) == 0
