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
    dataset_factory: Any = None,
) -> PoolReport:
    """Load every task across the configured rungs as schedulable entries, RUNG BY RUNG.

    Each rung is loaded in its own dataset so one unloadable rung does not kill the whole run:
    a missing config, a network error, or a heavy modality whose binary payload overflows the arrow
    loader's 2 GB chunk limit is recorded as a `SkippedRung` instead of raising. (`source` is the
    hyphen `hf_repo`.) A rung that loads but yields ZERO tasks is also recorded: silence here is
    how coverage gaps hide. `dataset_factory` is the offline-test seam (defaults to IcarusDataset).
    """
    factory = dataset_factory or IcarusDataset
    entries: list[TaskEntry] = []
    skipped: list[SkippedRung] = []
    for rung in rungs:
        dataset = None
        loaded = 0
        try:
            dataset = factory(rungs=(rung,), n_tasks=tasks_per_rung, n_samples=n_samples, support_fraction=support_fraction, shuffle_within=shuffle, seed=seed, hf_repo=source)
            for index in range(len(dataset)):
                entries.append(task_entry(dataset[index]))
                loaded += 1
            if loaded == 0:
                skipped.append(SkippedRung(rung=rung, error_type="EmptyRung", message="dataset loaded but yielded zero tasks"))
        except Exception as error:  # broad: many failure modes (arrow overflow, network, missing config); skip the rung and continue
            logger.warning("skipping rung %s: could not load it (%s: %s)", rung, type(error).__name__, error)
            skipped.append(SkippedRung(rung=rung, error_type=type(error).__name__, message=str(error)[:300]))
            continue
        finally:
            if dataset is not None:
                dataset.close()
                gc.collect()
    return PoolReport(entries=entries, skipped=skipped)
