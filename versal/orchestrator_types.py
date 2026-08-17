"""Small value types shared by the orchestration policy and its callers."""

import math
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Attempt:
    """One policy-ledger row.

    Policy scores remain separate from literal support/query accuracy, and status fields
    distinguish a measured zero from an evaluation that did not run.
    """

    task: str
    depth: int
    outcome: str
    metric: float
    generations: int
    library_key: str | None = None
    decompose_op: str | None = None
    strategy: str | None = None
    failure_stage: str | None = None
    refine_generations: int = 0
    seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    sample_metrics: dict[str, float] = field(default_factory=dict)
    size_metrics: dict[str, float] = field(default_factory=dict)
    resource_metrics: dict[str, float] = field(default_factory=dict)
    report_metric: float | None = None
    report_strategy: str | None = None
    report_representation: str | None = None
    task_metrics: dict[str, float] = field(default_factory=dict)
    strategy_metrics: dict[str, float] = field(default_factory=dict)
    validation_status: str = "not_run"
    validation_metrics: dict[str, float] = field(default_factory=dict)
    diagnostic_observation: dict[str, Any] = field(default_factory=dict)
    support_accuracy: float | None = None
    query_accuracy: float | None = None
    support_status: str = "legacy_missing"
    query_status: str = "legacy_missing"
    representation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task": self.task,
            "depth": self.depth,
            "outcome": self.outcome,
            "metric": self.metric,
            "generations": self.generations,
            "library_key": self.library_key,
            "decompose_op": self.decompose_op,
            "strategy": self.strategy,
            "failure_stage": self.failure_stage,
        }
        if self.refine_generations:
            data["refine_generations"] = self.refine_generations
        if self.seconds:
            data["seconds"] = self.seconds
        if self.stage_seconds:
            data["stage_seconds"] = self.stage_seconds
        if self.sample_metrics:
            data["sample_metrics"] = self.sample_metrics
        if self.size_metrics:
            data["size_metrics"] = self.size_metrics
        if self.resource_metrics:
            data["resource_metrics"] = self.resource_metrics
        if self.report_metric is not None:
            data["report_metric"] = self.report_metric
        if self.report_strategy is not None:
            data["report_strategy"] = self.report_strategy
        if self.report_representation is not None:
            data["report_representation"] = self.report_representation
        if self.task_metrics:
            data["task_metrics"] = self.task_metrics
        if self.strategy_metrics:
            data["strategy_metrics"] = self.strategy_metrics
        if self.validation_status != "not_run" or self.validation_metrics:
            data["validation_status"] = self.validation_status
            data["validation_metrics"] = self.validation_metrics
        if self.diagnostic_observation:
            data["diagnostic_observation"] = self.diagnostic_observation
        if self.support_status != "legacy_missing" or self.query_status != "legacy_missing" or self.support_accuracy is not None or self.query_accuracy is not None:
            data.update(
                {
                    "support_accuracy": self.support_accuracy,
                    "query_accuracy": self.query_accuracy,
                    "support_status": self.support_status,
                    "query_status": self.query_status,
                }
            )
        if self.representation is not None:
            data["representation"] = self.representation
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attempt":
        return cls(
            task=data["task"],
            depth=int(data["depth"]),
            outcome=data["outcome"],
            metric=float(data["metric"]),
            generations=int(data["generations"]),
            library_key=data.get("library_key"),
            decompose_op=data.get("decompose_op"),
            strategy=data.get("strategy"),
            failure_stage=data.get("failure_stage"),
            refine_generations=int(data.get("refine_generations", 0)),
            seconds=float(data.get("seconds", 0.0)),
            stage_seconds=dict(data.get("stage_seconds", {})),
            sample_metrics=dict(data.get("sample_metrics", {})),
            size_metrics=dict(data.get("size_metrics", {})),
            resource_metrics=dict(data.get("resource_metrics", {})),
            report_metric=float(data["report_metric"]) if data.get("report_metric") is not None else None,
            report_strategy=str(data["report_strategy"]) if data.get("report_strategy") is not None else None,
            report_representation=str(data["report_representation"]) if data.get("report_representation") is not None else None,
            task_metrics={str(key): float(value) for key, value in data.get("task_metrics", {}).items()},
            strategy_metrics={str(key): float(value) for key, value in data.get("strategy_metrics", {}).items()},
            validation_status=str(data.get("validation_status", "not_run")),
            validation_metrics={str(key): float(value) for key, value in data.get("validation_metrics", {}).items()},
            diagnostic_observation=dict(data.get("diagnostic_observation", {})),
            support_accuracy=float(data["support_accuracy"]) if data.get("support_accuracy") is not None else None,
            query_accuracy=float(data["query_accuracy"]) if data.get("query_accuracy") is not None else None,
            support_status=str(data.get("support_status", "legacy_missing")),
            query_status=str(data.get("query_status", "legacy_missing")),
            representation=str(data["representation"]) if data.get("representation") is not None else None,
        )


@dataclass(frozen=True)
class Solution:
    key: str | None
    entry_type: str
    metric: float
    report_metric: float | None = None
    task_metrics: dict[str, float] = field(default_factory=dict)
    support_accuracy: float | None = None
    query_accuracy: float | None = None
    support_status: str = "legacy_missing"
    query_status: str = "legacy_missing"


@dataclass(frozen=True)
class RefinementRank:
    """The lexicographic standing of one topology in a refine-on-hit comparison."""

    metric: float
    robustness: float
    complexity: int
    entry_type: str


def refinement_improves(candidate: RefinementRank, incumbent: RefinementRank, *, metric_epsilon: float, robustness_epsilon: float) -> bool:
    """Prefer simpler executable structure inside a non-regressing accuracy band.

    Expanded complexity makes modules and compositions comparable. Pareto selection still explores
    novelty and robustness; this final gate deterministically chooses whether to replace a known-good
    solution and never treats an equal candidate as an improvement.
    """
    if not (math.isfinite(candidate.metric) and math.isfinite(candidate.robustness)):
        return False
    if candidate.metric > incumbent.metric + metric_epsilon:
        return True
    if candidate.metric < incumbent.metric - metric_epsilon:
        return False
    if candidate.complexity < incumbent.complexity:
        return True
    if candidate.complexity > incumbent.complexity:
        return False
    return candidate.robustness > incumbent.robustness + robustness_epsilon


@dataclass
class StallDetector:
    """Stops an evolve phase that has flatlined (no best-fitness gain) or is hopeless (below the
    floor metric at half budget). One instance per evolve phase; it is stateful."""

    stall_generations: int
    stall_epsilon: float
    floor: float
    budget: int
    metric_of: Callable[[Any], float]
    best_fitness: float = -math.inf
    since_improvement: int = 0
    stalled: bool = False

    def __call__(self, generation: int, best: Any) -> bool:
        if best.fitness > self.best_fitness + self.stall_epsilon:
            self.best_fitness = best.fitness
            self.since_improvement = 0
        else:
            self.since_improvement += 1
        if self.since_improvement >= self.stall_generations:
            self.stalled = True
        if generation >= self.budget // 2 and self.metric_of(best) < self.floor:
            self.stalled = True
        return self.stalled


def attempts_to_dicts(attempts: list[Attempt]) -> list[dict[str, Any]]:
    return [attempt.to_dict() for attempt in attempts]


def attempts_from_dicts(data: list[dict[str, Any]]) -> list[Attempt]:
    return [Attempt.from_dict(item) for item in data]
