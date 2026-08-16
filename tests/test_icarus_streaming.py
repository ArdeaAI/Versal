"""Offline regressions for the bounded Icarus streaming task pool.

The fixtures are real Parquet shards read through Hugging Face, but they live entirely under
``tmp_path``: these tests must never depend on Hub availability or a populated user cache.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from datasets import Dataset

from versal.dataset.icarus import ICARUS_FEATURES, Axis, Field, Task, TaskKind, TaskMeta, ValueType, serialize_task
from versal.dataset.icarus_streaming import SELECTION_ALGORITHM, DatasetSelectionCancelled, IcarusStreamingSource, OneTaskMaterializer, StreamingTaskRef


def _field(values: list[float]) -> Field:
    return Field(torch.tensor(values, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)


def _task(name: str, *, rung: int = 2, fixed_split: bool = False, support: int = 3, query: int = 3) -> Task:
    pairs = [(_field([float(index % 2), float((index // 2) % 2)]), _field([float(index % 2)])) for index in range(support + query)]
    return Task(TaskMeta(rung, TaskKind.MAP, name, fixed_split=fixed_split), pairs[:support], pairs[support:])


def _write_shard(root: Path, rung: int, ordinal: int, tasks: list[Task]) -> Path:
    directory = root / f"rung_{rung}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shard-{ordinal:05d}.parquet"
    Dataset.from_list([serialize_task(task) for task in tasks], features=ICARUS_FEATURES).to_parquet(path)
    return path


def test_streaming_ref_round_trips_without_losing_pinned_revision() -> None:
    ref = StreamingTaskRef(rung=17, name="raven.b42", config="rung_17", shard="rung_17/shard-00042.parquet", row_index=3, revision="deadbeef")
    assert StreamingTaskRef.from_dict(ref.to_dict()) == ref


def test_local_selection_is_deterministic_and_shard_balanced(tmp_path: Path) -> None:
    for shard in range(4):
        _write_shard(tmp_path, 2, shard, [_task(f"s{shard}.a"), _task(f"s{shard}.b")])

    first = IcarusStreamingSource(str(tmp_path)).select(rungs=[2], n_tasks=6, shuffle=True, seed=19)
    second = IcarusStreamingSource(str(tmp_path)).select(rungs=[2], n_tasks=6, shuffle=True, seed=19)

    assert first == second
    assert len(first) == 6 and len({ref.name for ref in first}) == 6
    assert len({ref.shard for ref in first[:4]}) == 4  # one task per shard before any shard repeats


def test_unshuffled_local_selection_uses_sequential_file_and_row_order(tmp_path: Path) -> None:
    _write_shard(tmp_path, 4, 0, [_task("a0", rung=4), _task("a1", rung=4)])
    _write_shard(tmp_path, 4, 1, [_task("b0", rung=4), _task("b1", rung=4)])

    refs = IcarusStreamingSource(str(tmp_path)).select(rungs=[4], n_tasks=99, shuffle=False, seed=999)

    assert [ref.name for ref in refs] == ["a0", "a1", "b0", "b1"]
    assert [ref.row_index for ref in refs] == [0, 1, 0, 1]


def test_selection_cancellation_stops_before_opening_a_shard(tmp_path: Path, monkeypatch) -> None:
    _write_shard(tmp_path, 4, 0, [_task("never-read", rung=4)])

    def forbidden(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202 - fail-fast test sentinel
        raise AssertionError("cancelled selection opened a shard")

    monkeypatch.setattr("versal.dataset.icarus_streaming.load_dataset", forbidden)
    with pytest.raises(DatasetSelectionCancelled, match="graceful shutdown"):
        IcarusStreamingSource(str(tmp_path)).select(rungs=[4], n_tasks=1, shuffle=False, seed=0, cancelled=lambda: True)


def test_materialize_preserves_fixed_support_and_native_query_floor(tmp_path: Path) -> None:
    _write_shard(tmp_path, 3, 0, [_task("fixed", rung=3, fixed_split=True, support=5, query=6)])
    source = IcarusStreamingSource(str(tmp_path))
    ref = source.select(rungs=[3], n_tasks=1, shuffle=False, seed=0)[0]

    task = source.materialize(ref, n_samples=4, support_fraction=0.8, shuffle=False, seed=0, min_fixed_query_samples=3)

    assert len(task.support) == 5  # authoritative support is never truncated
    assert len(task.query) == 3  # floor is met in the same read, despite n_samples < support size


def test_materialize_caps_and_resplits_bucketed_task(tmp_path: Path) -> None:
    _write_shard(tmp_path, 5, 0, [_task("bucket", rung=5, fixed_split=False, support=3, query=3)])
    source = IcarusStreamingSource(str(tmp_path))
    ref = source.select(rungs=[5], n_tasks=1, shuffle=False, seed=0)[0]

    task = source.materialize(ref, n_samples=4, support_fraction=0.5, shuffle=False, seed=0)

    assert len(task.support) == 2 and len(task.query) == 2


def test_hub_materialization_pins_revision_and_reuses_downloader_cache_path(tmp_path: Path) -> None:
    fixture = _write_shard(tmp_path / "published", 7, 0, [_task("remote", rung=7)])
    cached = tmp_path / "hub-cache" / "rung_7" / fixture.name
    calls: list[dict[str, object]] = []
    misses = 0

    class FakeApi:
        def dataset_info(self, **kwargs):  # noqa: ANN003, ANN201 - injected Hugging Face seam
            assert kwargs == {"repo_id": "example/icarus", "revision": "main"}
            return SimpleNamespace(sha="abc123", siblings=[SimpleNamespace(rfilename=f"rung_7/{fixture.name}")])

    def cached_download(**kwargs) -> str:  # noqa: ANN003 - injected Hugging Face seam
        nonlocal misses
        calls.append(dict(kwargs))
        if not cached.exists():
            misses += 1
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(fixture.read_bytes())
        return str(cached)

    source = IcarusStreamingSource("example/icarus", revision="main", api=cast(Any, FakeApi()), download_file=cached_download)
    ref = StreamingTaskRef(rung=7, name="remote", config="rung_7", shard=f"rung_7/{fixture.name}", row_index=0, revision="abc123")

    assert source.materialize(ref, n_samples=4, support_fraction=0.5, shuffle=False, seed=0).meta.name == "remote"
    assert source.materialize(ref, n_samples=4, support_fraction=0.5, shuffle=False, seed=0).meta.name == "remote"

    expected = {"repo_id": "example/icarus", "filename": f"rung_7/{fixture.name}", "repo_type": "dataset", "revision": "abc123"}
    assert calls == [expected, expected]
    assert misses == 1  # the downloader owns the standard content-addressed cache; Versal adds no second cache
    assert source.provenance.to_dict() == {"source": "example/icarus", "revision": "abc123", "selection_algorithm": SELECTION_ALGORITHM}


def test_manifest_pinned_source_resumes_without_hub_api_access(tmp_path: Path) -> None:
    fixture = _write_shard(tmp_path, 8, 0, [_task("resume", rung=8)])
    commit = "a" * 40

    class ForbiddenApi:
        def __getattr__(self, name: str):
            raise AssertionError(f"resume attempted Hub API access through {name}")

    def cached_download(**kwargs) -> str:  # noqa: ANN003 - injected Hugging Face seam
        assert kwargs == {"repo_id": "example/icarus", "filename": "rung_8/shard.parquet", "repo_type": "dataset", "revision": commit}
        return str(fixture)

    source = IcarusStreamingSource(
        "example/icarus",
        revision=commit,
        pinned_revision=True,
        api=cast(Any, ForbiddenApi()),
        download_file=cached_download,
    )
    ref = StreamingTaskRef(rung=8, name="resume", config="rung_8", shard="rung_8/shard.parquet", row_index=0, revision=commit)

    assert source.materialize(ref, n_samples=4, support_fraction=0.5, shuffle=False, seed=0).meta.name == "resume"
    with pytest.raises(RuntimeError, match="cannot discover new tasks"):
        source.select(rungs=[8], n_tasks=1, shuffle=False, seed=0)
    with pytest.raises(ValueError, match="40-hex"):
        IcarusStreamingSource("example/icarus", revision="main", pinned_revision=True, api=cast(Any, ForbiddenApi()))


def test_one_task_materializer_reuses_immediate_repeat_and_releases_on_switch(tmp_path: Path, monkeypatch) -> None:
    from versal.dataset import icarus_streaming as streaming_module

    _write_shard(tmp_path, 6, 0, [_task("first", rung=6), _task("second", rung=6)])
    source = IcarusStreamingSource(str(tmp_path))
    refs = source.select(rungs=[6], n_tasks=2, shuffle=False, seed=0)
    calls: list[StreamingTaskRef] = []
    original = source.materialize

    def counted(ref: StreamingTaskRef, **kwargs) -> Task:  # noqa: ANN003 - mirrors the source's keyword-only caps
        calls.append(ref)
        return original(ref, **kwargs)

    monkeypatch.setattr(source, "materialize", counted)
    trims: list[bool] = []
    monkeypatch.setattr(streaming_module, "release_unused_host_memory", lambda: trims.append(True))
    materializer = OneTaskMaterializer(source, n_samples=4, support_fraction=0.5, shuffle=False, seed=0, min_fixed_query_samples=0)

    first = materializer.get(refs[0])
    assert materializer.get(refs[0]) is first
    assert calls == [refs[0]]

    second = materializer.get(refs[1])
    assert second.meta.name == "second" and calls == refs
    assert materializer.current_ref == refs[1]

    materializer.release()
    assert materializer.current_ref is None
    assert trims == [True, True]  # switch plus explicit release
