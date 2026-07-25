"""Task pool + schedule: entries describe their interface, loading is defensive, order resumes.

All offline: synthetic binary tasks stand in for the rungs (no Hub access)."""

import random
from typing import Any, cast

import torch

from ardevo.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from ardevo.dataset.icarus_streaming import DatasetProvenance, StreamingTaskRef
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


def test_build_pool_report_closes_backend_datasets() -> None:
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

    report = multitask.build_pool_report(
        "unused",
        [1, 2],
        n_samples=4,
        support_fraction=0.8,
        tasks_per_rung=1,
        shuffle=False,
        seed=0,
        dataset_factory=FakeIcarusDataset,
    )

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
    loaded_task = report.entries[0].task
    assert loaded_task is not None
    assert len(loaded_task.support) == 5
    assert len(loaded_task.query) == 3
    assert len(closed) == 2


def test_streaming_manifest_restores_pinned_references_without_reselection() -> None:
    closed: list[bool] = []
    revision = "a" * 40

    class ManifestSource:
        def __init__(self, source: str, *, revision: str | None, pinned_revision: bool = False) -> None:
            assert source == "published/icarus" and revision == "a" * 40 and pinned_revision
            self.provenance = DatasetProvenance(source, revision)

        def select(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201 - must remain unused
            raise AssertionError("resume reselected tasks instead of using task_pool.json")

        def close(self) -> None:
            closed.append(True)

    reference = StreamingTaskRef(rung=17, name="raven.b42", config="rung_17", shard="rung_17/shard-00042.parquet", row_index=3, revision=revision)
    manifest = {
        "dataset": {"source": "published/icarus", "revision": revision, "selection_algorithm": "shard_round_robin_v1"},
        "tasks": [reference.to_dict()],
        "skipped_rungs": [{"rung": 8, "error_type": "Unavailable", "message": "recorded on the original run"}],
    }

    report = multitask.build_pool_report(
        "newer/ignored",
        [17],
        n_samples=8,
        support_fraction=0.8,
        tasks_per_rung=1,
        shuffle=True,
        seed=9,
        task_manifest=manifest,
        streaming_source_factory=cast(Any, ManifestSource),
    )

    assert [entry.reference for entry in report.entries] == [reference]
    assert report.provenance == {"source": "published/icarus", "revision": revision, "selection_algorithm": "shard_round_robin_v1"}
    assert report.skipped == [multitask.SkippedRung(8, "Unavailable", "recorded on the original run")]
    report.close()
    assert closed == [True]


def test_local_manifest_resume_needs_no_hub_revision(tmp_path) -> None:
    class LocalManifestSource:
        def __init__(self, source: str, *, revision: str | None, pinned_revision: bool = False) -> None:
            assert source == str(tmp_path) and revision is None and pinned_revision
            self.provenance = DatasetProvenance(source, None)

        def close(self) -> None:
            return None

    reference = StreamingTaskRef(rung=2, name="local", config="rung_2", shard="rung_2/shard.parquet", row_index=0, revision=None)
    manifest = {"dataset": {"source": str(tmp_path), "revision": None}, "tasks": [reference.to_dict()], "skipped_rungs": []}

    report = multitask.build_pool_report(
        "ignored",
        [2],
        n_samples=4,
        support_fraction=0.8,
        tasks_per_rung=1,
        shuffle=False,
        seed=0,
        task_manifest=manifest,
        streaming_source_factory=cast(Any, LocalManifestSource),
    )

    assert report.entries[0].reference == reference


def test_streaming_discovery_records_failed_rung_and_continues() -> None:
    calls: list[int] = []

    class PartialSource:
        def __init__(self, source: str, *, revision: str | None, pinned_revision: bool = False) -> None:
            assert source == "published/icarus" and revision is None and not pinned_revision
            self.provenance = DatasetProvenance(source, "resolved-sha")

        def select(self, rungs, **_kwargs):  # noqa: ANN001, ANN003, ANN201 - injected source seam
            rung = int(rungs[0])
            calls.append(rung)
            if rung == 3:
                raise OSError("synthetic metadata failure")
            return [StreamingTaskRef(rung, f"task{rung}", f"rung_{rung}", f"rung_{rung}/shard.parquet", 0, "resolved-sha")]

        def close(self) -> None:
            return None

    report = multitask.build_pool_report(
        "published/icarus",
        [3, 4],
        n_samples=8,
        support_fraction=0.8,
        tasks_per_rung=1,
        shuffle=False,
        seed=0,
        streaming_source_factory=cast(Any, PartialSource),
    )

    assert calls == [3, 4]
    assert [entry.name for entry in report.entries] == ["task4"]
    assert len(report.skipped) == 1
    assert report.skipped[0].rung == 3 and report.skipped[0].error_type == "OSError"
    assert report.skipped[0].message == "synthetic metadata failure"


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
