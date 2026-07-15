"""Task pool + schedule: entries describe their interface, loading is defensive, order resumes.

All offline: synthetic binary tasks stand in for the rungs (no Hub access)."""

import random
from typing import Any

import torch

from ardevo.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from ardevo.evolution import multitask
from ardevo.evolution.multitask import task_entry
from ardevo.evolution.schedule import build_schedule


def _binary_field(values: list[float]) -> Field:
    return Field(torch.tensor(values, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)


def _binary_task(name: str, width: int, rung: int = 2) -> Task:
    rows = [[float((index >> bit) & 1) for bit in range(width)] for index in range(4)]
    pairs = [(_binary_field(row), _binary_field([float(sum(row) % 2)])) for row in rows]
    return Task(meta=TaskMeta(rung=rung, kind=TaskKind.MAP, name=name, fixed_split=True), support=list(pairs), query=list(pairs))


def test_task_entry_describes_interface() -> None:
    entry = task_entry(_binary_task("parity3", 3))
    assert entry.input_signature.startswith("BINARY")
    assert entry.input_width == 3
    assert entry.output_width == 1


def test_build_pool_report_closes_backend_datasets(monkeypatch) -> None:
    closed: list[int] = []

    class FakeIcarusDataset:
        def __init__(self, *, rungs, **_kwargs: Any) -> None:
            self.rung = int(rungs[0])
            self.tasks = [_binary_task(f"task{self.rung}", 2, rung=self.rung)]

        def __len__(self) -> int:
            return len(self.tasks)

        def __getitem__(self, index: int) -> Task:
            return self.tasks[index]

        def close(self) -> None:
            closed.append(self.rung)

    monkeypatch.setattr(multitask, "IcarusDataset", FakeIcarusDataset)

    report = multitask.build_pool_report("unused", [1, 2], n_samples=4, support_fraction=0.8, tasks_per_rung=1, shuffle=False, seed=0)

    assert [entry.rung for entry in report.entries] == [1, 2]
    # Rungs load on a thread pool, so CLOSE order follows completion order; the guarantee is that
    # every dataset closes and the ENTRY order stays the configured rung order.
    assert sorted(closed) == [1, 2]


def test_fixed_split_query_floor_reloads_native_query_without_trimming_support() -> None:
    calls: list[int] = []
    closed: list[int] = []
    pair = (_binary_field([0.0, 1.0]), _binary_field([1.0]))

    class CappedFixedDataset:
        def __init__(self, *, n_samples: int, **_kwargs: Any) -> None:
            calls.append(n_samples)
            support = [pair] * 5
            native_query = [pair] * 6
            self.tasks = [Task(TaskMeta(3, TaskKind.MAP, "fixed", fixed_split=True), support, native_query[: max(0, n_samples - len(support))])]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> Task:
            return self.tasks[index]

        def close(self) -> None:
            closed.append(1)

    report = multitask.build_pool_report(
        "unused",
        [3],
        n_samples=4,
        support_fraction=0.8,
        tasks_per_rung=1,
        shuffle=False,
        seed=0,
        min_fixed_query_samples=3,
        dataset_factory=CappedFixedDataset,
    )

    assert calls == [4, 8]
    assert len(report.entries[0].task.support) == 5
    assert len(report.entries[0].task.query) == 3
    assert len(closed) == 2


def test_scheduler_state_round_trips() -> None:
    """The orchestrated trial checkpoints scheduler state every task; a rebuilt scheduler must
    continue the exact pick order."""
    pool = [task_entry(_binary_task(f"r{rung}_t{index}", 2, rung=rung)) for rung in (1, 2) for index in range(2)]
    rng = random.Random(0)
    for kind in ("round_robin", "interleave_rungs"):
        scheduler = build_schedule({"kind": kind})
        for _ in range(3):
            scheduler.next_index(pool, rng)
        snapshot = scheduler.state_dict()
        expected = [scheduler.next_index(pool, rng) for _ in range(4)]
        resumed = build_schedule({"kind": kind})
        resumed.load_state_dict(snapshot)
        assert [resumed.next_index(pool, rng) for _ in range(4)] == expected, kind
