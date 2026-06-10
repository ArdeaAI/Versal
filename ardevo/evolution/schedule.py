"""Schedule: an independent, swappable stage that picks which task the population faces next.

The continuous trial runs `generations_per_task` generations on one task, then asks the scheduler
for the next one. Schedulers are stateful (cursors / last pick) so the interleaving is reproducible
and survives a checkpoint; the registry returns configured instances, like speciation. To add a
curriculum, register one class and name it in `[schedule].kind`.
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
