"""Schedule: an independent, swappable stage that picks which task the run faces next.

The orchestrated trial asks the scheduler for the next pool index before every solve. Schedulers
are stateful (cursors / last pick) so the interleaving is reproducible and survives a checkpoint;
the registry returns configured instances, like speciation. To add a curriculum, register one
class and name it in `[schedule].kind`.
"""

import random
from dataclasses import dataclass, field
from typing import Any

from ardevo.evolution.multitask import TaskEntry
from ardevo.evolution.registry import Registry

SCHEDULE: Registry = Registry("schedule")


@SCHEDULE.register("random")
def _build_random(*, avoid_repeat: bool = True, **_params: object) -> "RandomSchedule":
    return RandomSchedule(avoid_repeat=avoid_repeat)


@SCHEDULE.register("round_robin")
def _build_round_robin(**_params: object) -> "RoundRobinSchedule":
    return RoundRobinSchedule()


@SCHEDULE.register("interleave_rungs")
def _build_interleave(**_params: object) -> "InterleaveRungsSchedule":
    return InterleaveRungsSchedule()


@SCHEDULE.register("regret")
def _build_regret(*, accept_threshold: float = 0.95, explore_fraction: float = 0.25, hard_floor: float = 0.1, solved_weight: float = 0.05, **_params: object) -> "RegretSchedule":
    return RegretSchedule(accept_threshold=accept_threshold, explore_fraction=explore_fraction, hard_floor=hard_floor, solved_weight=solved_weight)


def build_schedule(config: dict[str, Any]) -> Any:
    """Resolve `[schedule].kind` to a configured scheduler instance (params bound by name)."""
    kind = config.get("kind", "random")
    params = {key: value for key, value in config.items() if key not in ("kind", "generations_per_task", "checkpoint_every", "rungs", "tasks_per_rung", "shuffle")}
    return SCHEDULE.get(kind)(**params)


@dataclass
class RandomSchedule:
    """Uniform-random next task, optionally never the same task twice in a row."""

    avoid_repeat: bool = True
    last: int = -1

    def next_index(self, pool: list[TaskEntry], rng: random.Random) -> int:
        size = len(pool)
        choice = rng.randrange(size)
        if self.avoid_repeat and size > 1:
            while choice == self.last:
                choice = rng.randrange(size)
        self.last = choice
        return choice

    def state_dict(self) -> dict[str, Any]:
        return {"last": self.last}

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.last = int(data["last"])


@dataclass
class RoundRobinSchedule:
    """Walk the pool in order, wrapping around."""

    position: int = 0

    def next_index(self, pool: list[TaskEntry], rng: random.Random) -> int:
        index = self.position % len(pool)
        self.position += 1
        return index

    def state_dict(self) -> dict[str, Any]:
        return {"position": self.position}

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.position = int(data["position"])


@dataclass
class RegretSchedule:
    """ACCEL-style frontier prioritization: attempt the rung with the most learnable headroom.

    Regret per rung = accept_threshold minus the best metric any attempt has reached there
    (the trial feeds outcomes back through `observe`). Unattempted rungs outrank everything
    (regret plus an unseen bonus), solved rungs decay to `solved_weight` (revisits are cheap
    library hits anyway), and rungs stuck below `hard_floor` halve their score (a wall with no
    signal should not eat the run: the goldilocks band). `explore_fraction` of picks stay uniform
    random so no rung is ever starved and the rng contract matches the other schedulers. Within a
    rung, tasks advance by cursor exactly like interleave_rungs."""

    accept_threshold: float = 0.95
    explore_fraction: float = 0.25
    hard_floor: float = 0.1
    solved_weight: float = 0.05
    best_metric: dict[int, float] = field(default_factory=dict)
    attempts: dict[int, int] = field(default_factory=dict)
    task_cursors: dict[int, int] = field(default_factory=dict)

    def observe(self, rung: int, metric: float, solved: bool) -> None:
        self.attempts[rung] = self.attempts.get(rung, 0) + 1
        observed = self.accept_threshold if solved else float(metric)
        self.best_metric[rung] = max(self.best_metric.get(rung, 0.0), observed)

    def _score(self, rung: int) -> float:
        if rung not in self.attempts:
            return self.accept_threshold + 1.0  # first contact beats every partially-known rung
        best = self.best_metric.get(rung, 0.0)
        if best >= self.accept_threshold:
            return self.solved_weight
        regret = self.accept_threshold - best
        return regret * 0.5 if best < self.hard_floor else regret

    def next_index(self, pool: list[TaskEntry], rng: random.Random) -> int:
        by_rung: dict[int, list[int]] = {}
        for index, entry in enumerate(pool):
            by_rung.setdefault(entry.rung, []).append(index)
        rungs = sorted(by_rung)
        if rng.random() < self.explore_fraction:
            rung = rungs[rng.randrange(len(rungs))]
        else:
            rung = max(rungs, key=lambda candidate: (self._score(candidate), -candidate))  # ties: lowest rung first
        cursor = self.task_cursors.get(rung, 0)
        self.task_cursors[rung] = cursor + 1
        indices = by_rung[rung]
        return indices[cursor % len(indices)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_metric": {str(rung): value for rung, value in self.best_metric.items()},
            "attempts": {str(rung): count for rung, count in self.attempts.items()},
            "task_cursors": {str(rung): cursor for rung, cursor in self.task_cursors.items()},
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.best_metric = {int(rung): float(value) for rung, value in data["best_metric"].items()}
        self.attempts = {int(rung): int(count) for rung, count in data["attempts"].items()}
        self.task_cursors = {int(rung): int(cursor) for rung, cursor in data["task_cursors"].items()}


@dataclass
class InterleaveRungsSchedule:
    """Round-robin across rungs (one task per rung per cycle), advancing within each rung."""

    rung_cursor: int = 0
    task_cursors: dict[int, int] = field(default_factory=dict)

    def next_index(self, pool: list[TaskEntry], rng: random.Random) -> int:
        by_rung: dict[int, list[int]] = {}
        for index, entry in enumerate(pool):
            by_rung.setdefault(entry.rung, []).append(index)
        rungs = sorted(by_rung)
        rung = rungs[self.rung_cursor % len(rungs)]
        self.rung_cursor += 1
        cursor = self.task_cursors.get(rung, 0)
        self.task_cursors[rung] = cursor + 1
        indices = by_rung[rung]
        return indices[cursor % len(indices)]

    def state_dict(self) -> dict[str, Any]:
        return {"rung_cursor": self.rung_cursor, "task_cursors": {str(rung): cursor for rung, cursor in self.task_cursors.items()}}

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.rung_cursor = int(data["rung_cursor"])
        self.task_cursors = {int(rung): int(cursor) for rung, cursor in data["task_cursors"].items()}
