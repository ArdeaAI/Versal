"""Lazy, revision-pinned access to the published Icarus Parquet shards.

The vendored :mod:`versal.dataset.icarus` loader is intentionally map-style: it prepares an
entire Hugging Face configuration before applying its task cap.  That is useful as reference
code, but it is the wrong storage boundary for the very large upper rungs.  This module keeps
the published codec as the source of truth while selecting tasks from projected Parquet metadata
and downloading only the shards that contain selected tasks.
"""

from __future__ import annotations

import gc
import os
import random
import re
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from versal.dataset.icarus import Task, deserialize_task
from versal.utils.memory import release_unused_host_memory

SELECTION_ALGORITHM = "shard_round_robin_v1"
_IDENTITY_COLUMNS = ("name", "rung", "kind", "fixed_split")


class DatasetSelectionCancelled(Exception):
    """Raised between shard reads when graceful shutdown cancels pool discovery."""


@dataclass(frozen=True, slots=True)
class StreamingTaskRef:
    """Stable location of one task row without retaining its tensor payload."""

    rung: int
    name: str
    config: str
    shard: str
    row_index: int
    revision: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest row."""
        return {
            "rung": self.rung,
            "name": self.name,
            "config": self.config,
            "shard": self.shard,
            "row_index": self.row_index,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StreamingTaskRef:
        """Restore a reference written by :meth:`to_dict`."""
        revision = value.get("revision")
        return cls(
            rung=int(value["rung"]),
            name=str(value["name"]),
            config=str(value["config"]),
            shard=str(value["shard"]),
            row_index=int(value["row_index"]),
            revision=None if revision is None else str(revision),
        )


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Dataset identity recorded with a run's selected task manifest."""

    source: str
    revision: str | None
    selection_algorithm: str = SELECTION_ALGORITHM

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible provenance record."""
        return {
            "source": self.source,
            "revision": self.revision,
            "selection_algorithm": self.selection_algorithm,
        }


@dataclass(frozen=True, slots=True)
class _MetadataRow:
    name: str
    rung: int
    row_index: int


class IcarusStreamingSource:
    """Select and materialize Icarus tasks without preparing complete rung datasets.

    Hub sources are resolved to an immutable commit at construction.  Selection reads only the
    four identity columns through a Hugging Face streaming dataset.  Materialization then uses
    ``hf_hub_download`` so the selected Parquet shard participates in the normal content-addressed
    Hub cache, and streams only the referenced row from that local file.

    A local source uses the same ``<source>/rung_N/*.parquet`` layout and needs no revision.
    ``pinned_revision=True`` restores saved task references without doing new discovery; for a Hub
    source it additionally requires and trusts an immutable commit SHA.  That mode materializes
    recorded references but intentionally cannot select new ones.  ``api`` and ``download_file``
    are dependency-injection seams for offline tests.
    """

    def __init__(
        self,
        source: str | os.PathLike[str],
        *,
        revision: str | None = None,
        pinned_revision: bool = False,
        api: HfApi | None = None,
        download_file: Callable[..., str] | None = None,
    ) -> None:
        source_text = os.fspath(source)
        local_path = Path(source_text).expanduser()
        self._local_root = local_path.resolve() if local_path.is_dir() else None
        self._api = api or HfApi()
        self._download_file = download_file or hf_hub_download
        self._metadata_cache: dict[str, tuple[_MetadataRow, ...]] = {}
        self._trust_pinned_refs = False

        if self._local_root is not None:
            if revision is not None:
                raise ValueError("revision applies only to a Hugging Face Hub dataset source")
            self._source = str(self._local_root)
            self._revision = None
            self._repo_files: tuple[str, ...] = ()
            # A local resume also restores exact manifest locators, but there is no Hub commit to
            # resolve. Row identity validation during materialization still detects a moved task.
            self._trust_pinned_refs = pinned_revision
            return

        self._source = source_text
        if pinned_revision:
            revision_text = "" if revision is None else str(revision)
            if re.fullmatch(r"[0-9a-fA-F]{40}", revision_text) is None:
                raise ValueError("pinned_revision requires an immutable 40-hex Hub commit SHA")
            self._revision = revision_text.lower()
            self._repo_files = ()
            self._trust_pinned_refs = True
            return

        info = self._api.dataset_info(repo_id=self._source, revision=revision)
        resolved_revision = getattr(info, "sha", None)
        if not resolved_revision:
            raise RuntimeError(f"Hugging Face did not return a commit SHA for dataset {self._source!r}")
        self._revision = str(resolved_revision)
        siblings = getattr(info, "siblings", None) or ()
        self._repo_files = tuple(sorted(str(sibling.rfilename) for sibling in siblings if getattr(sibling, "rfilename", None)))
        if not self._repo_files:
            self._repo_files = tuple(sorted(self._api.list_repo_files(repo_id=self._source, repo_type="dataset", revision=self._revision)))

    @property
    def source(self) -> str:
        return self._source

    @property
    def revision(self) -> str | None:
        return self._revision

    @property
    def provenance(self) -> DatasetProvenance:
        return DatasetProvenance(source=self._source, revision=self._revision)

    def select(
        self,
        rungs: Sequence[int],
        *,
        n_tasks: int,
        shuffle: bool,
        seed: int | None,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[StreamingTaskRef]:
        """Select at most ``n_tasks`` references per rung from identity-only streams.

        Shuffled selection first permutes the shards, independently permutes the rows within each
        selected shard, and takes one row per shard before taking a second.  This prevents a small
        task cap from selecting a contiguous family from a single large shard.  Unshuffled
        selection remains the intuitive sorted-shard, row-by-row order.
        """
        if n_tasks < 0:
            raise ValueError("n_tasks must be non-negative")
        if self._trust_pinned_refs:
            raise RuntimeError("a manifest-pinned source cannot discover new tasks; restore its recorded task references")
        if n_tasks == 0:
            return []

        base_seed = 0 if seed is None else int(seed)
        selected: list[StreamingTaskRef] = []
        for rung in sorted(set(int(value) for value in rungs)):
            _raise_if_cancelled(cancelled)
            config = f"rung_{rung}"
            shards = self._config_shards(config)
            if not shards:
                continue
            if shuffle:
                rng = random.Random(_stable_seed(base_seed, config))
                rng.shuffle(shards)
                selected.extend(self._select_balanced(rung, config, shards, n_tasks, base_seed, cancelled))
            else:
                selected.extend(self._select_sequential(rung, config, shards, n_tasks, cancelled))
        return selected

    def materialize(
        self,
        ref: StreamingTaskRef,
        *,
        n_samples: int,
        support_fraction: float,
        shuffle: bool,
        seed: int | None,
        min_fixed_query_samples: int = 0,
    ) -> Task:
        """Load, validate, and cap one referenced task in a single payload pass."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if not 0.0 <= support_fraction <= 1.0:
            raise ValueError("support_fraction must be between zero and one")
        if ref.revision != self._revision:
            raise ValueError(f"task reference revision {ref.revision!r} does not match source revision {self._revision!r}")
        expected_config = f"rung_{ref.rung}"
        if ref.config != expected_config or not ref.shard.startswith(f"{expected_config}/"):
            raise ValueError(f"task reference {ref.name!r} does not belong to {expected_config}")
        if self._local_root is None and not self._trust_pinned_refs and ref.shard not in self._repo_files:
            raise ValueError(f"task shard {ref.shard!r} is absent from pinned dataset revision {self._revision}")

        local_shard = self._local_shard(ref)
        row = self._read_payload_row(local_shard, ref.row_index)
        task = deserialize_task(row)
        if task.meta.name != ref.name or task.meta.rung != ref.rung:
            raise RuntimeError(f"task reference no longer matches its row: expected rung {ref.rung} {ref.name!r}, found rung {task.meta.rung} {task.meta.name!r}")

        # Row indices restart in every Parquet shard.  Include the immutable shard locator so two
        # tasks at row zero do not receive the same support/query permutation merely by collision.
        task_seed = _stable_seed(0 if seed is None else int(seed), f"{ref.config}:{ref.shard}", ref.row_index)
        if task.meta.fixed_split:
            query = _shuffled(task.query, task_seed) if shuffle else task.query
            configured_query = max(0, n_samples - len(task.support))
            query_count = min(len(query), max(configured_query, max(0, int(min_fixed_query_samples))))
            return Task(meta=task.meta, support=task.support, query=query[:query_count])

        pool = task.support + task.query
        if shuffle:
            pool = _shuffled(pool, task_seed)
        total = min(n_samples, len(pool))
        if total >= 2:
            support_count = min(max(1, round(support_fraction * total)), total - 1)
            return Task(meta=task.meta, support=pool[:support_count], query=pool[support_count:total])
        return Task(meta=task.meta, support=pool[:total], query=[])

    def close(self) -> None:
        """Release projected metadata retained during task selection."""
        self._metadata_cache.clear()

    def _select_sequential(
        self,
        rung: int,
        config: str,
        shards: list[str],
        limit: int,
        cancelled: Callable[[], bool] | None,
    ) -> list[StreamingTaskRef]:
        selected: list[StreamingTaskRef] = []
        for shard in shards:
            _raise_if_cancelled(cancelled)
            for row in self._metadata_rows(shard, expected_rung=rung):
                selected.append(self._reference(config, shard, row))
                if len(selected) == limit:
                    return selected
        return selected

    def _select_balanced(
        self,
        rung: int,
        config: str,
        shards: list[str],
        limit: int,
        seed: int,
        cancelled: Callable[[], bool] | None,
    ) -> list[StreamingTaskRef]:
        rows_by_shard: list[tuple[str, list[_MetadataRow]]] = []
        selected: list[StreamingTaskRef] = []

        # First visit each shard once.  Stop opening remote metadata as soon as the task cap is met.
        for shard in shards:
            _raise_if_cancelled(cancelled)
            rows = list(self._metadata_rows(shard, expected_rung=rung))
            random.Random(_stable_seed(seed, shard)).shuffle(rows)
            if not rows:
                continue
            rows_by_shard.append((shard, rows))
            selected.append(self._reference(config, shard, rows[0]))
            if len(selected) == limit:
                return selected

        # Large caps take a second row from every shard before a third, preserving balance.
        row_offset = 1
        while len(selected) < limit:
            _raise_if_cancelled(cancelled)
            progressed = False
            for shard, rows in rows_by_shard:
                if row_offset >= len(rows):
                    continue
                selected.append(self._reference(config, shard, rows[row_offset]))
                progressed = True
                if len(selected) == limit:
                    return selected
            if not progressed:
                break
            row_offset += 1
        return selected

    def _reference(self, config: str, shard: str, row: _MetadataRow) -> StreamingTaskRef:
        return StreamingTaskRef(
            rung=row.rung,
            name=row.name,
            config=config,
            shard=shard,
            row_index=row.row_index,
            revision=self._revision,
        )

    def _config_shards(self, config: str) -> list[str]:
        if self._local_root is not None:
            return [path.relative_to(self._local_root).as_posix() for path in sorted((self._local_root / config).glob("*.parquet"))]
        prefix = f"{config}/"
        return [name for name in self._repo_files if name.startswith(prefix) and name.endswith(".parquet")]

    def _metadata_rows(self, shard: str, *, expected_rung: int) -> tuple[_MetadataRow, ...]:
        cached = self._metadata_cache.get(shard)
        if cached is not None:
            return cached

        stream = load_dataset("parquet", data_files=[self._stream_path(shard)], split="train", streaming=True, columns=list(_IDENTITY_COLUMNS))
        rows: list[_MetadataRow] = []
        iterator = iter(stream)
        try:
            for row_index, row in enumerate(iterator):
                row_rung = int(row["rung"])
                if row_rung != expected_rung:
                    raise RuntimeError(f"shard {shard!r} contains rung {row_rung}, expected rung {expected_rung}")
                name = str(row["name"])
                if not name:
                    raise RuntimeError(f"shard {shard!r} contains an empty task name at row {row_index}")
                rows.append(_MetadataRow(name=name, rung=row_rung, row_index=row_index))
        finally:
            _close_iterator(iterator)
        result = tuple(rows)
        self._metadata_cache[shard] = result
        return result

    def _stream_path(self, shard: str) -> str:
        if self._local_root is not None:
            return str(self._local_root / shard)
        return f"hf://datasets/{self._source}@{self._revision}/{shard}"

    def _local_shard(self, ref: StreamingTaskRef) -> str:
        if self._local_root is not None:
            path = (self._local_root / ref.shard).resolve()
            try:
                path.relative_to(self._local_root)
            except ValueError as error:
                raise ValueError(f"task shard {str(path)!r} is outside local dataset source {str(self._local_root)!r}") from error
            if not path.is_file():
                raise FileNotFoundError(path)
            return str(path)
        return self._download_file(repo_id=self._source, filename=ref.shard, repo_type="dataset", revision=self._revision)

    @staticmethod
    def _read_payload_row(shard: str, row_index: int) -> Mapping[str, Any]:
        if row_index < 0:
            raise IndexError(row_index)
        stream = load_dataset("parquet", data_files=[shard], split="train", streaming=True)
        iterator = iter(stream)
        try:
            for current_index, row in enumerate(iterator):
                if current_index == row_index:
                    return row
        finally:
            _close_iterator(iterator)
        raise IndexError(f"row {row_index} does not exist in shard {shard!r}")


class OneTaskMaterializer:
    """Keep at most one source-owned decoded task alive between scheduler selections."""

    def __init__(
        self,
        source: IcarusStreamingSource,
        *,
        n_samples: int,
        support_fraction: float,
        shuffle: bool,
        seed: int | None,
        min_fixed_query_samples: int = 0,
    ) -> None:
        self._source = source
        self._n_samples = n_samples
        self._support_fraction = support_fraction
        self._shuffle = shuffle
        self._seed = seed
        self._min_fixed_query_samples = min_fixed_query_samples
        self._current_ref: StreamingTaskRef | None = None
        self._current_task: Task | None = None

    @property
    def current_ref(self) -> StreamingTaskRef | None:
        return self._current_ref

    def get(self, ref: StreamingTaskRef) -> Task:
        """Return the active task, reusing immediate repeats and releasing task switches."""
        if ref == self._current_ref and self._current_task is not None:
            return self._current_task
        self.release()
        task = self._source.materialize(
            ref,
            n_samples=self._n_samples,
            support_fraction=self._support_fraction,
            shuffle=self._shuffle,
            seed=self._seed,
            min_fixed_query_samples=self._min_fixed_query_samples,
        )
        self._current_ref = ref
        self._current_task = task
        return task

    def release(self) -> None:
        """Drop the source-owned strong reference to the active tensor payload."""
        had_task = self._current_task is not None
        self._current_ref = None
        self._current_task = None
        if had_task:
            gc.collect()
            # Streaming prevents pool growth, but Arrow may retain freed row-group buffers in its
            # allocator. Return those pages before the next large modality is decoded.
            import pyarrow

            pyarrow.default_memory_pool().release_unused()
            release_unused_host_memory()

    def close(self) -> None:
        self.release()
        self._source.close()


def _stable_seed(base: int, label: str, row_index: int = 0) -> int:
    return (base * 1_000_003 + zlib.crc32(label.encode()) * 31 + row_index) & 0xFFFFFFFF


def _shuffled(values: list[Any], seed: int) -> list[Any]:
    order = list(range(len(values)))
    random.Random(seed).shuffle(order)
    return [values[index] for index in order]


def _raise_if_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise DatasetSelectionCancelled("task-pool discovery cancelled by graceful shutdown")


def _close_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()
