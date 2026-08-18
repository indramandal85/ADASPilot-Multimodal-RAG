"""Run the unified Module 10 evaluation and save one JSON report."""

from __future__ import annotations

import argparse

from adaspilot.config import get_settings
from adaspilot.evaluation.io import load_evaluation_examples
from adaspilot.evaluation.system import SystemEvaluator
from adaspilot.ingestion.io import resolve_current_build, write_json
from adaspilot.orchestration.runtime import build_workflows
from adaspilot.orchestration.service import WorkflowRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all ADASPilot execution modes")
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument(
        "--max-examples-per-mode",
        type=int,
        default=0,
        help="0 evaluates every example in the selected split",
    )
    parser.add_argument(
        "--auto-router",
        action="store_true",
        help="route each example automatically instead of using its benchmark mode",
    )
    arguments = parser.parse_args()
    if arguments.max_examples_per_mode < 0:
        parser.error("--max-examples-per-mode cannot be negative")

    settings = get_settings()
    examples = load_evaluation_examples(settings.benchmark_path)
    build_dir, _ = resolve_current_build(settings.processed_data_dir)
    bundle = build_workflows(settings)
    try:
        report = SystemEvaluator(
            settings,
            WorkflowRunner(bundle),
            build_dir=build_dir,
        ).evaluate(
            examples,
            split=arguments.split,
            use_auto_router=arguments.auto_router,
            max_examples_per_mode=(arguments.max_examples_per_mode or None),
        )
        write_json(settings.evaluation_report_path, report)
        print(report.model_dump_json(indent=2))
        print(f"Saved: {settings.evaluation_report_path}")
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
