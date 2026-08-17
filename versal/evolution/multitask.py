"""Build the orchestrated trial's schedulable Icarus task pool.

The live path discovers lightweight, revision-pinned Parquet row references and materializes one
task at a time.  Synthetic/offline callers can still provide a map-style ``dataset_factory``;
that compatibility seam deliberately remains eager because its fixtures are already in memory.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from versal.dataset.icarus import Level0Encoder, Task, encode_task, support_loader
from versal.dataset.icarus_streaming import (
    DatasetSelectionCancelled,
    IcarusStreamingSource,
    OneTaskMaterializer,
    StreamingTaskRef,
)
from versal.evaluation import output_features
from versal.utils.logging import Logger

logger = Logger.get_logger()


@dataclass(frozen=True)
class TaskEntry:
    """One schedulable task identity, either eager or backed by a lazy Parquet row reference."""

    rung: int
    name: str
    task: Task | None = None
    input_signature: str = ""
    input_axes: tuple[str, ...] = ()
    input_shape: tuple[int, ...] = ()
    input_width: int = 0
    output_width: int = 0
    reference: StreamingTaskRef | None = None


def task_entry(task: Task) -> TaskEntry:
    """Derive a fully described eager entry from an already materialized task."""
    support_input, _support_output = support_loader(task)
    input_shape = tuple(int(dim) for dim in support_input.data.shape[1:])
    input_axes = tuple(axis.value for axis in support_input.descriptor.axes)
    input_width = 1
    for dim in input_shape:
        input_width *= dim
    signature = f"{support_input.descriptor.value_type.value}|{','.join(input_axes)}"
    encoded = encode_task(task, Level0Encoder(input_width))
    return TaskEntry(
        rung=task.meta.rung,
        name=task.meta.name,
        task=task,
        input_signature=signature,
        input_axes=input_axes,
        input_shape=input_shape,
        input_width=input_width,
        output_width=output_features(encoded),
    )


def reference_entry(reference: StreamingTaskRef) -> TaskEntry:
    """Wrap a payload-free source reference for the existing schedulers."""
    return TaskEntry(rung=reference.rung, name=reference.name, reference=reference)


@dataclass(frozen=True)
class SkippedRung:
    """Why a configured rung produced no schedulable task."""

    rung: int
    error_type: str
    message: str


@dataclass
class PoolReport:
    entries: list[TaskEntry]
    skipped: list[SkippedRung]
    provenance: dict[str, Any] = field(default_factory=dict)
    materializer: OneTaskMaterializer | None = None
    interrupted: bool = False

    def materialize(self, entry: TaskEntry) -> Task:
        """Return an eager task or load the referenced task into the one-task resident slot."""
        if entry.task is not None:
            return entry.task
        if entry.reference is None or self.materializer is None:
            raise RuntimeError(f"task {entry.name!r} has neither an eager payload nor a streaming reference")
        return self.materializer.get(entry.reference)

    def close(self) -> None:
        """Release the active task and projected selection metadata."""
        if self.materializer is not None:
            self.materializer.close()


def build_pool_report(
    source: str,
    rungs: list[int],
    n_samples: int,
    support_fraction: float,
    tasks_per_rung: int,
    shuffle: bool,
    seed: int,
    min_fixed_query_samples: int = 0,
    dataset_factory: Any = None,
    load_workers: int = 4,
    *,
    task_manifest: Mapping[str, Any] | None = None,
    streaming_source_factory: Callable[..., IcarusStreamingSource] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> PoolReport:
    """Discover schedulable tasks without preparing entire rung datasets.

    A supplied ``task_manifest`` restores its exact references against the pinned revision.  The
    ``dataset_factory`` argument retains the old eager path solely as an offline-test/tool seam;
    production callers use Hugging Face streaming and the ordinary Hub cache.
    """
    if dataset_factory is not None:
        return _build_eager_pool_report(
            source,
            rungs,
            n_samples,
            support_fraction,
            tasks_per_rung,
            shuffle,
            seed,
            min_fixed_query_samples,
            dataset_factory,
            load_workers,
        )

    manifest_dataset = dict(task_manifest.get("dataset", {})) if task_manifest is not None else {}
    pinned_source = str(manifest_dataset.get("source", source))
    revision_value = manifest_dataset.get("revision")
    revision = None if revision_value is None else str(revision_value)
    local_manifest = task_manifest is not None and Path(pinned_source).expanduser().is_dir()
    if task_manifest is not None and revision is None and not local_manifest:
        raise ValueError("a resumed Hub task manifest must pin an immutable dataset revision")
    source_factory = streaming_source_factory or IcarusStreamingSource
    streaming_source = source_factory(pinned_source, revision=revision, pinned_revision=task_manifest is not None)
    materializer = OneTaskMaterializer(
        streaming_source,
        n_samples=n_samples,
        support_fraction=support_fraction,
        shuffle=shuffle,
        seed=seed,
        min_fixed_query_samples=min_fixed_query_samples,
    )
    provenance = streaming_source.provenance.to_dict()

    try:
        if task_manifest is not None:
            references = [StreamingTaskRef.from_dict(row) for row in task_manifest.get("tasks", [])]
            skipped = [SkippedRung(rung=int(row["rung"]), error_type=str(row["error_type"]), message=str(row["message"])) for row in task_manifest.get("skipped_rungs", [])]
            return PoolReport(entries=[reference_entry(reference) for reference in references], skipped=skipped, provenance=provenance, materializer=materializer)

        entries: list[TaskEntry] = []
        skipped: list[SkippedRung] = []
        for rung in rungs:
            if cancelled is not None and cancelled():
                return PoolReport(entries=entries, skipped=skipped, provenance=provenance, materializer=materializer, interrupted=True)
            try:
                references = streaming_source.select([rung], n_tasks=tasks_per_rung, shuffle=shuffle, seed=seed, cancelled=cancelled)
            except DatasetSelectionCancelled:
                return PoolReport(entries=entries, skipped=skipped, provenance=provenance, materializer=materializer, interrupted=True)
            except Exception as error:  # network, missing config, malformed shard: record this rung and continue
                logger.warning("skipping rung %s: could not discover it (%s: %s)", rung, type(error).__name__, error)
                skipped.append(SkippedRung(rung=rung, error_type=type(error).__name__, message=str(error)[:300]))
                continue
            if not references:
                skipped.append(SkippedRung(rung=rung, error_type="EmptyRung", message="dataset yielded zero task references"))
                continue
            entries.extend(reference_entry(reference) for reference in references)
        return PoolReport(entries=entries, skipped=skipped, provenance=provenance, materializer=materializer)
    except BaseException:
        materializer.close()
        raise


def _build_eager_pool_report(
    source: str,
    rungs: list[int],
    n_samples: int,
    support_fraction: float,
    tasks_per_rung: int,
    shuffle: bool,
    seed: int,
    min_fixed_query_samples: int,
    dataset_factory: Any,
    load_workers: int,
) -> PoolReport:
    """Compatibility implementation for synthetic map-style fixtures and diagnostics."""

    factory = dataset_factory

    def _load_rung(rung: int) -> tuple[list[TaskEntry], SkippedRung | None]:
        dataset = None
        expanded_dataset = None
        rung_entries: list[TaskEntry] = []
        try:
            dataset = factory(rungs=(rung,), n_tasks=tasks_per_rung, n_samples=n_samples, support_fraction=support_fraction, shuffle_within=shuffle, seed=seed, hf_repo=source)
            tasks = [dataset[index] for index in range(len(dataset))]
            fixed_floor = max(0, int(min_fixed_query_samples))
            needs_native_query = [task for task in tasks if task.meta.fixed_split and len(task.query) < fixed_floor]
            if needs_native_query:
                expanded_cap = max(n_samples, max(len(task.support) + fixed_floor for task in needs_native_query))
                expanded_dataset = factory(
                    rungs=(rung,),
                    n_tasks=tasks_per_rung,
                    n_samples=expanded_cap,
                    support_fraction=support_fraction,
                    shuffle_within=shuffle,
                    seed=seed,
                    hf_repo=source,
                )
                expanded_by_name = {expanded.meta.name: expanded for expanded in (expanded_dataset[index] for index in range(len(expanded_dataset)))}
                tasks = [expanded_by_name.get(task.meta.name, task) if task.meta.fixed_split else task for task in tasks]
            rung_entries.extend(task_entry(task) for task in tasks)
            if not rung_entries:
                return rung_entries, SkippedRung(rung=rung, error_type="EmptyRung", message="dataset loaded but yielded zero tasks")
        except Exception as error:
            logger.warning("skipping rung %s: could not load it (%s: %s)", rung, type(error).__name__, error)
            return rung_entries, SkippedRung(rung=rung, error_type=type(error).__name__, message=str(error)[:300])
        finally:
            if dataset is not None:
                dataset.close()
            if expanded_dataset is not None:
                expanded_dataset.close()
        return rung_entries, None

    if load_workers > 1 and len(rungs) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(load_workers, len(rungs))) as pool:
            results = list(pool.map(_load_rung, rungs))
    else:
        results = [_load_rung(rung) for rung in rungs]
    gc.collect()

    entries: list[TaskEntry] = []
    skipped: list[SkippedRung] = []
    for rung_entries, skip in results:
        entries.extend(rung_entries)
        if skip is not None:
            skipped.append(skip)
    return PoolReport(entries=entries, skipped=skipped, provenance={"source": source, "revision": None, "selection_algorithm": "eager_fixture"})
