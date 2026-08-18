"""JSON and JSONL readers for versioned corpus and benchmark records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaspilot.domain.corpus import CorpusManifest
from adaspilot.domain.evaluation import BenchmarkTargets, EvaluationExample


def _load_json(path: Path) -> Any:  # noqa: ANN401 - JSON is validated immediately by Pydantic
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_corpus_manifest(path: Path) -> CorpusManifest:
    return CorpusManifest.model_validate(_load_json(path))


def load_benchmark_targets(path: Path) -> BenchmarkTargets:
    return BenchmarkTargets.model_validate(_load_json(path))


def load_evaluation_examples(path: Path) -> list[EvaluationExample]:
    examples: list[EvaluationExample] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc.msg}") from exc
            try:
                examples.append(EvaluationExample.model_validate(payload))
            except ValueError as exc:
                raise ValueError(f"invalid example on {path}:{line_number}: {exc}") from exc
    return examples
