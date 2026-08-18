"""Validate corpus and benchmark files from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from adaspilot.config import get_settings
from adaspilot.evaluation.io import (
    load_benchmark_targets,
    load_corpus_manifest,
    load_evaluation_examples,
)
from adaspilot.evaluation.validation import validate_benchmark


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Validate ADASPilot corpus and evaluation data")
    parser.add_argument("--manifest", type=Path, default=settings.corpus_manifest_path)
    parser.add_argument("--benchmark", type=Path, default=settings.benchmark_path)
    parser.add_argument("--targets", type=Path, default=settings.benchmark_targets_path)
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="Treat missing verified target counts as errors",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_corpus_manifest(args.manifest)
        examples = load_evaluation_examples(args.benchmark)
        targets = load_benchmark_targets(args.targets)
        report = validate_benchmark(
            manifest,
            examples,
            targets,
            require_target=args.require_target,
        )
    except (OSError, ValueError, ValidationError) as exc:
        print(json.dumps({"valid": False, "fatal_error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.valid else 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
