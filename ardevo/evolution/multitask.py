"""The orchestrated trial's task pool: load Icarus rungs as schedulable entries.

`task_entry` derives the structural facts a task exposes (I/O signature, shapes, head width) from
Field descriptors only. `build_pool_report` loads every configured rung defensively, recording a
`SkippedRung` row for anything that fails to load instead of killing the run: silence is how a
"full ladder" run quietly stops being one.
"""

import gc
from dataclasses import dataclass
from typing import Any

from ardevo.dataset.icarus import IcarusDataset, Level0Encoder, Task, encode_task, support_loader
from ardevo.evaluation import output_features
from ardevo.utils.logging import Logger

logger = Logger.get_logger()


@dataclass(frozen=True)
class TaskEntry:
    """One schedulable task plus the structural facts needed to shape a search for it."""

    rung: int
    name: str
    task: Task
    input_signature: str
    input_axes: tuple[str, ...]
    input_shape: tuple[int, ...]
    input_width: int
    output_width: int


def task_entry(task: Task) -> TaskEntry:
    """Derive a `TaskEntry` (signature, shapes, head width) from a raw Icarus task."""
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


@dataclass(frozen=True)
class SkippedRung:
    """Why a configured rung produced no tasks: visible in stats.json and the console, never just a
    log line (rung 5 silently never loading is how a 'full ladder' run quietly stops being one)."""

    rung: int
    error_type: str
    message: str


@dataclass
class PoolReport:
    entries: list[TaskEntry]
    skipped: list[SkippedRung]


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
) -> PoolReport:
    """Load every task across the configured rungs as schedulable entries, RUNG BY RUNG.

    Each rung is loaded in its own dataset so one unloadable rung does not kill the whole run:
    a missing config, a network error, or a heavy modality whose binary payload overflows the arrow
    loader's 2 GB chunk limit is recorded as a `SkippedRung` instead of raising. (`source` is the
    hyphen `hf_repo`.) A rung that loads but yields ZERO tasks is also recorded: silence here is
    how coverage gaps hide. `dataset_factory` is the offline-test seam (defaults to IcarusDataset).

    Rungs load on a small thread pool (I/O-bound HF fetches; the arrow cache uses file locks, and
    torch encode work releases the GIL): an 18-rung startup overlaps its downloads instead of
    paying them serially. Results keep the exact configured rung ORDER regardless of completion
    order, so schedules and skip reports are unchanged; `load_workers = 1` is the serial path.
    """
    factory = dataset_factory or IcarusDataset

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
                # Icarus treats n_samples as support + query, while authoritative fixed support is
                # never truncated. Reload the same deterministic task selection with just enough
                # headroom for the requested native query floor. Bucketed tasks keep the original
                # cap and the vendored adapter remains unchanged.
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
        except Exception as error:  # broad: many failure modes (arrow overflow, network, missing config); skip the rung and continue
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
            results = list(pool.map(_load_rung, rungs))  # map preserves the configured rung order
    else:
        results = [_load_rung(rung) for rung in rungs]
    gc.collect()

    entries: list[TaskEntry] = []
    skipped: list[SkippedRung] = []
    for rung_entries, skip in results:
        entries.extend(rung_entries)
        if skip is not None:
            skipped.append(skip)
    return PoolReport(entries=entries, skipped=skipped)
