"""Dependency and size validation for typed research plans."""

from __future__ import annotations

from adaspilot.domain.orchestration import ResearchPlan


class PlanValidationError(ValueError):
    """Raised before executing an invalid or unsafe task graph."""


class PlanValidator:
    def __init__(self, max_tasks: int = 5, max_depth: int = 3) -> None:
        self.max_tasks = max_tasks
        self.max_depth = max_depth

    def validate(self, plan: ResearchPlan) -> None:
        if not plan.tasks:
            raise PlanValidationError("plan must contain at least one task")
        if len(plan.tasks) > self.max_tasks:
            raise PlanValidationError(f"plan exceeds the {self.max_tasks}-task limit")
        tasks = {task.task_id: task for task in plan.tasks}
        for task in plan.tasks:
            missing = set(task.dependencies) - set(tasks)
            if missing:
                raise PlanValidationError(f"task {task.task_id} has unknown dependencies")
            if task.task_id in task.dependencies:
                raise PlanValidationError(f"task {task.task_id} depends on itself")

        visiting: set[str] = set()
        completed: set[str] = set()
        depths: dict[str, int] = {}

        def visit(task_id: str) -> int:
            if task_id in visiting:
                raise PlanValidationError("plan contains a dependency cycle")
            if task_id in completed:
                return depths[task_id]
            visiting.add(task_id)
            dependencies = tasks[task_id].dependencies
            depth = 1 + max((visit(dependency) for dependency in dependencies), default=0)
            visiting.remove(task_id)
            completed.add(task_id)
            depths[task_id] = depth
            return depth

        maximum_depth = max(visit(task_id) for task_id in tasks)
        if maximum_depth > self.max_depth:
            raise PlanValidationError(f"plan exceeds dependency depth {self.max_depth}")
