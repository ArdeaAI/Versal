"""Bounded support-only validation for task-solution admission.

The evolutionary search still uses the task's full support split.  This gate asks a narrower,
operational question afterward: can the selected representation be freshly fitted without one
support example and predict that omitted example?  Query examples never enter fold construction.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from array import array
from dataclasses import dataclass, replace
from typing import Any, Callable

from versal.dataset.icarus import Task
from versal.evolution.composition import CompositionGenome
from versal.evolution.genome import Genome


@dataclass(frozen=True, slots=True)
class CrossValidationConfig:
    enabled: bool = False
    max_folds: int = 5
    min_folds: int = 2
    pass_fraction: float = 0.5
    reserve_fraction: float = 0.15
    reserve_min_seconds: float = 5.0
    reserve_max_seconds: float = 60.0
    seed: int = 0

    @classmethod
    def from_table(cls, table: dict[str, Any] | None, *, seed: int = 0) -> "CrossValidationConfig":
        values = table or {}
        config = cls(
            enabled=bool(values.get("enabled", False)),
            max_folds=max(1, int(values.get("max_folds", 5))),
            min_folds=max(1, int(values.get("min_folds", 2))),
            pass_fraction=float(values.get("pass_fraction", 0.5)),
            reserve_fraction=max(0.0, float(values.get("reserve_fraction", 0.15))),
            reserve_min_seconds=max(0.0, float(values.get("reserve_min_seconds", 5.0))),
            reserve_max_seconds=max(0.0, float(values.get("reserve_max_seconds", 60.0))),
            seed=int(values.get("seed", seed)),
        )
        if not 0.0 <= config.pass_fraction < 1.0:
            raise ValueError("[orchestrator.cross_validation] pass_fraction must be in [0, 1)")
        if config.reserve_max_seconds < config.reserve_min_seconds:
            raise ValueError("cross-validation reserve_max_seconds must be >= reserve_min_seconds")
        return config

    def reserve_seconds(self, available_seconds: float | None) -> float:
        if not self.enabled or available_seconds is None:
            return 0.0
        wanted = available_seconds * self.reserve_fraction
        return min(available_seconds, self.reserve_max_seconds, max(self.reserve_min_seconds, wanted))


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    status: str  # disabled | not_applicable | passed | failed | inconclusive
    folds_planned: int = 0
    folds_completed: int = 0
    folds_passed: int = 0
    seconds: float = 0.0

    @property
    def admits(self) -> bool:
        return self.status in {"disabled", "not_applicable", "passed"}

    def metrics(self) -> dict[str, float]:
        return {
            "cv_folds_planned": float(self.folds_planned),
            "cv_folds_completed": float(self.folds_completed),
            "cv_folds_passed": float(self.folds_passed),
            "cv_pass_fraction": self.folds_passed / max(1, self.folds_completed),
            "cv_seconds": self.seconds,
        }


FoldEvaluator = Callable[[Task, int, float | None], dict[str, float]]


class SupportCrossValidator:
    def __init__(self, config: CrossValidationConfig, *, accept_threshold: float) -> None:
        self.config = config
        self.accept_threshold = accept_threshold

    def _fold_indices(self, task: Task) -> list[int]:
        indices = list(range(len(task.support)))
        if len(indices) <= self.config.max_folds:
            return indices
        salt = f"{self.config.seed}:{task.meta.rung}:{task.meta.name}"
        indices.sort(key=lambda index: hashlib.sha1(f"{salt}:{index}".encode()).digest())
        return sorted(indices[: self.config.max_folds])

    def run(self, task: Task, evaluate_fold: FoldEvaluator, *, deadline: float | None) -> CrossValidationResult:
        if not self.config.enabled:
            return CrossValidationResult("disabled")
        indices = self._fold_indices(task)
        if len(task.support) < 2 or len(indices) < self.config.min_folds:
            return CrossValidationResult("not_applicable", folds_planned=len(indices))
        started = time.perf_counter()
        completed = passed = 0
        for fold_index in indices:
            if deadline is not None and time.perf_counter() >= deadline:
                return CrossValidationResult("inconclusive", len(indices), completed, passed, time.perf_counter() - started)
            fold = Task(task.meta, [pair for index, pair in enumerate(task.support) if index != fold_index], [task.support[fold_index]])
            seed = self._fold_seed(task, fold_index)
            try:
                metrics = evaluate_fold(fold, seed, deadline)
            except TimeoutError:
                return CrossValidationResult("inconclusive", len(indices), completed, passed, time.perf_counter() - started)
            value = self._task_appropriate_value(metrics)
            if value is None:
                return CrossValidationResult("inconclusive", len(indices), completed, passed, time.perf_counter() - started)
            completed += 1
            passed += int(value >= self.accept_threshold)
        status = "passed" if completed >= self.config.min_folds and passed / completed > self.config.pass_fraction else "failed"
        return CrossValidationResult(status, len(indices), completed, passed, time.perf_counter() - started)

    def _fold_seed(self, task: Task, fold_index: int) -> int:
        digest = hashlib.sha256(f"{self.config.seed}:{task.meta.rung}:{task.meta.name}:{fold_index}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    @staticmethod
    def _task_appropriate_value(metrics: dict[str, float]) -> float | None:
        exact = metrics.get("query_task_exact")
        if exact is not None and math.isfinite(float(exact)):
            return float(exact)
        loss = metrics.get("query_loss")
        value = metrics.get("query_accuracy")
        if value is None or not math.isfinite(float(value)) or (loss is not None and not math.isfinite(float(loss))):
            return None
        return float(value)


def fresh_genome_weights(genome: Genome, seed: int) -> Genome:
    """Clone a topology and deterministically reset its task-trained connection weights."""

    rng = random.Random(seed)
    fan_in: dict[int, int] = {}
    for connection in genome.connections:
        if connection.enabled:
            fan_in[connection.out_id] = fan_in.get(connection.out_id, 0) + 1
    tied: dict[int, float] = {}
    connections = []
    for connection in genome.connections:
        sigma = 1.0 / math.sqrt(max(1, fan_in.get(connection.out_id, 1)))
        if connection.tie_group is None:
            weight = rng.gauss(0.0, sigma)
        else:
            weight = tied.setdefault(connection.tie_group, rng.gauss(0.0, sigma))
        connections.append(replace(connection, weight=weight))
    fresh = genome.clone()
    fresh.connections = connections
    fresh.growth_hints = None
    return fresh


def fresh_composition_glue(comp: CompositionGenome, seed: int, *, dense_scale: float | None = None) -> CompositionGenome:
    """Clone composition structure while resetting trainable glue and preserving fixed port maps."""

    rng = random.Random(seed)
    fresh = comp.clone()
    edges = []
    for edge in fresh.edges:
        if edge.port_map is not None or not edge.glue:
            edges.append(edge)
            continue
        source_width = fresh.nodes[edge.in_id].out_width
        if edge.glue_rank > 0:
            sigma = (max(1, source_width) * edge.glue_rank) ** -0.25
        else:
            sigma = dense_scale if dense_scale is not None else 1.0 / math.sqrt(max(1, source_width))
        values = [rng.gauss(0.0, sigma) for _ in range(len(edge.glue))]
        glue = _as_storage(values, edge.glue)
        edges.append(replace(edge, glue=glue))
    fresh.edges = edges
    return fresh


def _as_storage(values: list[float], template: tuple[float, ...] | array) -> tuple[float, ...] | array:
    if isinstance(template, array):
        compact = array("f")
        compact.extend(values)
        return compact
    return tuple(values)
