"""Run one bounded orchestration mode over the local corpus."""

from __future__ import annotations

import argparse

from adaspilot.config import get_settings
from adaspilot.evaluation.agents import evaluate_plan_run, evaluate_react_run
from adaspilot.orchestration.runtime import build_workflows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded ADASPilot workflow")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["direct", "react", "plan"], default="direct")
    arguments = parser.parse_args()
    bundle = build_workflows(get_settings())
    try:
        if arguments.mode == "direct":
            payload = bundle.direct.run(arguments.query).model_dump_json(indent=2)
        elif arguments.mode == "react":
            react_result = bundle.react.run(arguments.query)
            payload = react_result.model_dump_json(indent=2)
            print(evaluate_react_run(react_result).model_dump_json(indent=2))
        else:
            plan_result = bundle.plan.run(arguments.query)
            payload = plan_result.model_dump_json(indent=2)
            print(evaluate_plan_run(plan_result).model_dump_json(indent=2))
        print(payload)
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
