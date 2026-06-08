# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch>=2.12.0",
#     "datasets>=4.8.5",
# ]
# ///

"""Icarus dataset - vendored single-file runtime: loader + reference encoder (generated; do not edit here).

Assembled from the icarus/ runtime of the generator repo
https://github.com/ArdeaAI/Icarus-Dataset - edit the source there and rebuild.

Copyright (c) 2026 Ardea AI Corp. MIT License with Attribution.
This software is based on work originally developed by Ardea AI Corp.
"""

import glob
import os
import random
import torch
import torch.nn.functional as F
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from datasets import Dataset, Features, Sequence, Value, load_dataset
from enum import Enum
from typing import Any

# ===== icarus/types.py =====
class Axis(Enum):
    """Semantic role of a tensor dimension. Values are stored verbatim in the dataset."""

    EXAMPLE = "E"
    CHANNEL = "C"
    TIME = "T"
    HEIGHT = "H"
    WIDTH = "W"
    DEPTH = "D"
    EXTRA = "K"


class ValueType(Enum):
    """How the values in a `Field` are interpreted by an encoder/loss."""

    BINARY = "BINARY"
    CATEGORICAL = "CATEGORICAL"
    MULTILABEL = "MULTILABEL"
    CONTINUOUS = "CONTINUOUS"
    ORDINAL = "ORDINAL"


class TaskKind(Enum):
    """Provenance flag. INTERACTIVE marks rungs derived from policy rollouts (rungs 4-5)."""

    MAP = "MAP"
    INTERACTIVE = "INTERACTIVE"


_CLASS_BEARING_VALUE_TYPES = frozenset({ValueType.CATEGORICAL, ValueType.ORDINAL, ValueType.MULTILABEL})


@dataclass(frozen=True, slots=True)
class Field:
    """
    A single typed tensor plus the descriptor an encoder needs to consume it.

    Attributes:
        data: The payload tensor. Its dtype must be one the codec can round-trip.
        axes: One `Axis` per dimension of `data`; `len(axes) == data.ndim`.
        value_type: How values are interpreted (drives encoder/loss dispatch downstream).
        n_classes: Number of classes; required for CATEGORICAL/ORDINAL/MULTILABEL, else None.
        value_range: `(lo, hi)` for CONTINUOUS range-normalization; None when unknown/not applicable.
        mask: Optional boolean padding mask, same shape as `data`, **True where the value is PADDING/
            ignored**, False where it is real (PyTorch `key_padding_mask` convention). None ⇒ all real.
    """

    data: torch.Tensor
    axes: tuple[Axis, ...]
    value_type: ValueType
    n_classes: int | None
    value_range: tuple[float, float] | None
    mask: torch.Tensor | None

    def __post_init__(self) -> None:
        if len(self.axes) != self.data.ndim:
            raise ValueError(f"axes {self.axes} do not match data.ndim={self.data.ndim}")
        if self.value_type in _CLASS_BEARING_VALUE_TYPES:
            if self.n_classes is None or self.n_classes <= 0:
                raise ValueError(f"{self.value_type.value} requires a positive n_classes, got {self.n_classes}")
        elif self.n_classes is not None:
            raise ValueError(f"{self.value_type.value} must not set n_classes, got {self.n_classes}")
        if self.value_range is not None:
            low, high = self.value_range
            if high < low:
                raise ValueError(f"value_range {self.value_range} has high < low")
        if self.mask is not None:
            if self.mask.shape != self.data.shape:
                raise ValueError(f"mask shape {tuple(self.mask.shape)} != data shape {tuple(self.data.shape)}")
            if self.mask.dtype is not torch.bool:
                raise ValueError(f"mask dtype must be torch.bool, got {self.mask.dtype}")


@dataclass(frozen=True, slots=True)
class TaskMeta:
    """Identity of a task: which rung it belongs to, its kind, a stable name, and its split policy.

    `fixed_split=True` means the stored support/query split is authoritative (a native/built-in split,
    e.g. ARC's train demos vs test pairs, or XOR's degenerate support==query) and the loader must NOT
    re-split it. `fixed_split=False` (the default) means the task is a bucketed pool the loader carves
    into support/query at load time via `support_fraction`.

    `class_names` is optional human-readable names for the task's class-bearing TARGET (the i-th name labels
    class/position i of the output CATEGORICAL/MULTILABEL field). It is viewer/eval metadata only - the loader
    and encoder never read it (they stay structural). None when names are unknown or not applicable. Carried
    here (once per task) rather than on every Field so it does not bloat the per-example payload.
    """

    rung: int
    kind: TaskKind
    name: str
    fixed_split: bool = False
    class_names: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Task:
    """
    One self-contained task: a support set to fit on and a query set to score on.

    `support` is NON-EMPTY for every task in every rung; the inner loop trains on it. `query` holds
    the held-out pairs fitness is scored on. Each pair is `(input_Field, output_Field)`.
    """

    meta: TaskMeta
    support: list[tuple[Field, Field]]
    query: list[tuple[Field, Field]]

    def __post_init__(self) -> None:
        if not self.support:
            raise ValueError(f"support must be non-empty for task {self.meta.name!r}")


# ===== icarus/codec.py =====
# Pinned dtype set. The codec refuses anything else so an adapter emitting an unexpected dtype fails
# loudly instead of silently lossy. Names are numpy/torch-canonical and carry no endianness ambiguity
# on the single-arch (little-endian) build pipeline.
_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
_NAME_BY_DTYPE: dict[torch.dtype, str] = {dtype: name for name, dtype in _DTYPE_BY_NAME.items()}

_FIELD_FEATURES: dict[str, Any] = {
    "flat_bytes": Value("binary"),
    "dtype": Value("string"),
    "shape": Sequence(Value("int64")),
    "axes": Sequence(Value("string")),
    "value_type": Value("string"),
    "n_classes": Value("int64"),
    "value_range": Sequence(Value("float64")),
    "mask_bytes": Value("binary"),
}
_PAIR_FEATURES: dict[str, Any] = {"input": _FIELD_FEATURES, "output": _FIELD_FEATURES}

# A list `[feature]` denotes a variable-length list of structs (one row carries many pairs), as
# opposed to `Sequence(dict)` which would transpose into a struct-of-lists.
ICARUS_FEATURES = Features(
    {
        "name": Value("string"),
        "rung": Value("int64"),
        "kind": Value("string"),
        "fixed_split": Value("bool"),
        "class_names": Sequence(Value("string")),  # optional target class names; null when unknown
        "support": [_PAIR_FEATURES],
        "query": [_PAIR_FEATURES],
    }
)


def serialize_field(field: Field) -> dict[str, Any]:
    """Encode a `Field` to a flat record matching `_FIELD_FEATURES`."""
    name = _NAME_BY_DTYPE.get(field.data.dtype)
    if name is None:
        raise ValueError(f"unsupported dtype {field.data.dtype}; allowed: {sorted(_DTYPE_BY_NAME)}")
    # `.numpy()` is a torch tensor method (no `import numpy`); `.tobytes()` yields C-order raw bytes.
    flat_bytes = field.data.detach().contiguous().cpu().numpy().tobytes()
    mask_bytes = None
    if field.mask is not None:
        mask_bytes = field.mask.detach().contiguous().cpu().numpy().tobytes()
    return {
        "flat_bytes": flat_bytes,
        "dtype": name,
        "shape": list(field.data.shape),
        "axes": [axis.value for axis in field.axes],
        "value_type": field.value_type.value,
        "n_classes": field.n_classes,
        "value_range": None if field.value_range is None else [float(field.value_range[0]), float(field.value_range[1])],
        "mask_bytes": mask_bytes,
    }


def deserialize_field(record: Mapping[str, Any]) -> Field:
    """Decode a record produced by `serialize_field` back into a `Field` (lossless)."""
    dtype = _DTYPE_BY_NAME[record["dtype"]]
    shape = tuple(int(dimension) for dimension in record["shape"])
    # bytearray => writable buffer (silences the non-writable warning); clone => owns its storage so
    # the tensor outlives the Arrow buffer it was read from.
    data = torch.frombuffer(bytearray(record["flat_bytes"]), dtype=dtype).reshape(shape).clone()
    axes = tuple(Axis(value) for value in record["axes"])
    value_range_raw = record["value_range"]
    value_range = None if value_range_raw is None else (float(value_range_raw[0]), float(value_range_raw[1]))
    mask = None
    if record["mask_bytes"] is not None:
        mask = torch.frombuffer(bytearray(record["mask_bytes"]), dtype=torch.bool).reshape(shape).clone()
    return Field(
        data=data,
        axes=axes,
        value_type=ValueType(record["value_type"]),
        n_classes=record["n_classes"],
        value_range=value_range,
        mask=mask,
    )


def serialize_task(task: Task) -> dict[str, Any]:
    """Encode a whole `Task` to one dataset row matching `ICARUS_FEATURES`."""
    return {
        "name": task.meta.name,
        "rung": task.meta.rung,
        "kind": task.meta.kind.value,
        "fixed_split": task.meta.fixed_split,
        "class_names": None if task.meta.class_names is None else list(task.meta.class_names),
        "support": [{"input": serialize_field(inp), "output": serialize_field(out)} for inp, out in task.support],
        "query": [{"input": serialize_field(inp), "output": serialize_field(out)} for inp, out in task.query],
    }


def deserialize_task(row: Mapping[str, Any]) -> Task:
    """Decode one dataset row back into a whole `Task` (lossless)."""
    # `.get` tolerates shards written before `class_names` existed, so adding the column does not force a
    # rebuild of rungs that carry no names.
    class_names = row.get("class_names")
    meta = TaskMeta(
        rung=int(row["rung"]),
        kind=TaskKind(row["kind"]),
        name=row["name"],
        fixed_split=bool(row["fixed_split"]),
        class_names=tuple(class_names) if class_names else None,
    )
    support = [(deserialize_field(pair["input"]), deserialize_field(pair["output"])) for pair in row["support"]]
    query = [(deserialize_field(pair["input"]), deserialize_field(pair["output"])) for pair in row["query"]]
    return Task(meta=meta, support=support, query=query)


def configs_for_rung(rung: int) -> list[str]:
    """Map a rung number to its dataset config name. One config per rung (NB360 families are now rungs)."""
    return [f"rung_{rung}"]


def config_for_task(rung: int, name: str) -> str:
    """Route one task to its config name (the inverse of `configs_for_rung`)."""
    return f"rung_{rung}"


# ===== icarus/batching.py =====
@dataclass(frozen=True, slots=True)
class FieldDescriptor:
    """The structural contract an encoder reads - a `Field` with the payload (data/mask) removed."""

    axes: tuple[Axis, ...]
    value_type: ValueType
    n_classes: int | None
    value_range: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class BatchedField:
    """A stack of `B` fields sharing one descriptor; `mask` (True = padding/ignore) is None when nothing is padded."""

    data: torch.Tensor
    mask: torch.Tensor | None
    descriptor: FieldDescriptor


def _descriptor(field: Field) -> FieldDescriptor:
    return FieldDescriptor(axes=field.axes, value_type=field.value_type, n_classes=field.n_classes, value_range=field.value_range)


def _stack(fields: list[Field]) -> BatchedField:
    descriptor = _descriptor(fields[0])
    for other in fields[1:]:
        assert _descriptor(other) == descriptor, f"heterogeneous descriptors in one batch: {descriptor} vs {_descriptor(other)}"

    shapes = {tuple(field.data.shape) for field in fields}
    if len(shapes) == 1:
        data = torch.stack([field.data for field in fields])
        if all(field.mask is None for field in fields):
            return BatchedField(data=data, mask=None, descriptor=descriptor)
        # mask is True where padding; a field without its own mask has none (all False = all real).
        masks = [field.mask if field.mask is not None else torch.zeros_like(field.data, dtype=torch.bool) for field in fields]
        return BatchedField(data=data, mask=torch.stack(masks), descriptor=descriptor)

    # Heterogeneous shapes batch by padding to the max extent. Each field's own pad-mask (or None => all
    # real) delimits its real region; everything outside that region is batch padding (mask True).
    ndims = {field.data.ndim for field in fields}
    assert len(ndims) == 1, f"cannot batch fields of differing rank: {ndims}"
    rank = ndims.pop()
    max_shape = tuple(max(field.data.shape[dim] for field in fields) for dim in range(rank))

    data = torch.zeros((len(fields), *max_shape), dtype=fields[0].data.dtype)
    mask = torch.ones((len(fields), *max_shape), dtype=torch.bool)  # everything is padding until a field fills it
    for index, field in enumerate(fields):
        region = (index, *(slice(0, size) for size in field.data.shape))
        data[region] = field.data
        mask[region] = field.mask if field.mask is not None else False  # None => the field's region is entirely real
    return BatchedField(data=data, mask=mask, descriptor=descriptor)


def support_loader(task: Task) -> tuple[BatchedField, BatchedField]:
    """Batch the task's support pairs into (inputs, outputs). Support is guaranteed non-empty."""
    inputs = [inp for inp, _ in task.support]
    outputs = [out for _, out in task.support]
    return _stack(inputs), _stack(outputs)


def query_loader(task: Task) -> tuple[BatchedField, BatchedField]:
    """Batch the task's query pairs into (inputs, outputs)."""
    assert task.query, "query is empty; nothing to batch"
    inputs = [inp for inp, _ in task.query]
    outputs = [out for _, out in task.query]
    return _stack(inputs), _stack(outputs)


# ===== icarus/loader.py =====
_Plan = tuple[str, int]  # (config name, row index within that config's split)


def _stable_seed(base: int, config: str, row_index: int) -> int:
    """A process-stable per-task seed for the load-time shuffle.

    Uses crc32 of the config name (not Python's salted `hash`) so the same `(seed, config, row)` draws the
    same support/query split + ordering across runs and processes — reproducible sampling, not building.
    """
    return (base * 1_000_003 + zlib.crc32(config.encode()) * 31 + row_index) & 0xFFFFFFFF


def _concatenate(per_rung_plans: list[list[_Plan]]) -> list[_Plan]:
    """Flatten the per-rung task plans in ascending ladder order."""
    return [plan for plans in per_rung_plans for plan in plans]


def _interleave(per_rung_plans: list[list[_Plan]], tasks_per_cycle: int) -> list[_Plan]:
    """Round-robin `tasks_per_cycle` tasks from each rung per cycle (ascending rung order), dropping exhausted rungs."""
    combined: list[_Plan] = []
    start = 0
    while True:
        progressed = False
        for plans in per_rung_plans:
            chunk = plans[start : start + tasks_per_cycle]
            if chunk:
                combined.extend(chunk)
                progressed = True
        if not progressed:
            return combined
        start += tasks_per_cycle


class IcarusDataset(torch.utils.data.Dataset[Task]):
    """
    Map-style torch Dataset yielding whole `Task` objects from the published Icarus dataset.

    Args:
        rungs: Which rungs to include (deduplicated; ascending ladder order unless interleaved).
        n_tasks: Per-rung cap on tasks (MAXES: `min(n_tasks, available)`).
        n_samples: Per-task cap on the TOTAL number of examples (support + query).
        support_fraction: Fraction of each task's `n_samples` used as support for BUCKETED rungs
            (the rest is query). Ignored for `fixed_split` tasks (ARC/XOR), which keep their stored split.
        shuffle_within: Seeded randomization of which tasks are drawn per rung and of the per-task pool order.
        interleave_rungs: 0 = concatenate rungs in ascending order; N>0 = round-robin N tasks/rung per cycle.
        seed: Seed for the deterministic shuffles (None behaves as 0).
        hf_repo: Hub dataset id, OR a local directory laid out as `<dir>/<config>/*.parquet`.
    """

    def __init__(
        self,
        rungs: tuple[int, ...],
        n_tasks: int,
        n_samples: int,
        support_fraction: float = 0.8,
        shuffle_within: bool = False,
        interleave_rungs: int = 0,
        seed: int | None = None,
        hf_repo: str = "Ardea/Icarus_dataset",
    ) -> None:
        self._n_samples = n_samples
        self._support_fraction = support_fraction
        self._shuffle_within = shuffle_within
        self._seed = 0 if seed is None else int(seed)
        self._hf_repo = hf_repo
        self._datasets: dict[str, Dataset] = {}

        rng = random.Random(self._seed)
        per_rung_plans: list[list[_Plan]] = []
        for rung in sorted(set(rungs)):
            candidates: list[_Plan] = []
            for config in configs_for_rung(rung):
                dataset = self._datasets.get(config)
                if dataset is None:
                    dataset = self._maybe_load_config(config)
                    if dataset is None:  # absent in a partially-built local repo
                        continue
                    self._datasets[config] = dataset
                candidates.extend((config, row_index) for row_index in range(len(dataset)))
            if shuffle_within:
                rng.shuffle(candidates)
            per_rung_plans.append(candidates[: min(n_tasks, len(candidates))])

        self._plans = _interleave(per_rung_plans, interleave_rungs) if interleave_rungs > 0 else _concatenate(per_rung_plans)

    def _maybe_load_config(self, config: str) -> Dataset | None:
        """Load one config, or None if it is absent from a local directory build."""
        if os.path.isdir(self._hf_repo):
            files = sorted(glob.glob(os.path.join(self._hf_repo, config, "*.parquet")))
            if not files:
                return None
            return load_dataset("parquet", data_files=files, split="train")
        return load_dataset(self._hf_repo, name=config, split="train")

    def __len__(self) -> int:
        return len(self._plans)

    def __getitem__(self, index: int) -> Task:
        config, row_index = self._plans[index]
        task = deserialize_task(self._datasets[config][row_index])
        if task.meta.fixed_split:
            return self._cap_fixed_split(task, config, row_index)
        return self._resplit(task, config, row_index)

    def _cap_fixed_split(self, task: Task, config: str, row_index: int) -> Task:
        """Native split (ARC/XOR): support travels whole; query trimmed so support+query <= n_samples."""
        query = task.query
        if self._shuffle_within:
            query = _shuffled(query, _stable_seed(self._seed, config, row_index))
        keep = max(0, self._n_samples - len(task.support))
        return Task(meta=task.meta, support=task.support, query=query[:keep])

    def _resplit(self, task: Task, config: str, row_index: int) -> Task:
        """Bucketed rung: recombine the stored pool and re-split by support_fraction, capped at n_samples total."""
        pool = task.support + task.query
        if self._shuffle_within:
            pool = _shuffled(pool, _stable_seed(self._seed, config, row_index))
        total = min(self._n_samples, len(pool))
        if total >= 2:
            support_count = min(max(1, round(self._support_fraction * total)), total - 1)
            return Task(meta=task.meta, support=pool[:support_count], query=pool[support_count:total])
        return Task(meta=task.meta, support=pool[:total], query=[])  # degenerate single-example task


def _shuffled(pairs: list, seed: int) -> list:  # noqa: ANN001 - list of (Field, Field) pairs
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    return [pairs[index] for index in order]


# ===== examples/level0_encoder.py =====
_CLASS_TYPES = (ValueType.CATEGORICAL, ValueType.ORDINAL)
_BINARY_TYPES = (ValueType.BINARY, ValueType.MULTILABEL)


def _normalize_continuous(x: torch.Tensor, descriptor: FieldDescriptor) -> torch.Tensor:
    if descriptor.value_range is not None:
        low, high = descriptor.value_range
        if high > low:
            return (x - low) / (high - low)
    return x


class Level0Encoder:
    """Flatten any field to a fixed-width 1-D vector (+ descriptor). Reference code only."""

    def __init__(self, max_flat_dim: int) -> None:
        self._max_flat_dim = max_flat_dim

    def _fit(self, x: torch.Tensor) -> torch.Tensor:
        batch, width = x.shape
        if width == self._max_flat_dim:
            return x
        if width > self._max_flat_dim:
            return x[:, : self._max_flat_dim]
        return torch.cat([x, torch.zeros(batch, self._max_flat_dim - width)], dim=1)

    def encode(self, batched: BatchedField) -> tuple[torch.Tensor, FieldDescriptor]:
        """Model input: flatten -> normalize CONTINUOUS -> zero masked pads -> fit to max_flat_dim."""
        batch = batched.data.shape[0]
        x = batched.data.reshape(batch, -1).to(torch.float32)
        if batched.descriptor.value_type is ValueType.CONTINUOUS:
            x = _normalize_continuous(x, batched.descriptor)
        if batched.mask is not None:  # mask True = padding; keep (multiply by 1) only the real positions
            x = x * (~batched.mask).reshape(batch, -1).to(torch.float32)
        return self._fit(x), batched.descriptor

    def encode_target(self, batched: BatchedField) -> tuple[torch.Tensor, torch.Tensor | None, FieldDescriptor]:
        """Loss target in the form each loss expects (indices for CE; float for BCE/MSE)."""
        batch = batched.data.shape[0]
        flat = batched.data.reshape(batch, -1)
        mask = None if batched.mask is None else batched.mask.reshape(batch, -1)
        value_type = batched.descriptor.value_type
        if value_type in _CLASS_TYPES:
            return flat.to(torch.long), mask, batched.descriptor
        if value_type in _BINARY_TYPES:
            return flat.to(torch.float32), mask, batched.descriptor
        return _normalize_continuous(flat.to(torch.float32), batched.descriptor), mask, batched.descriptor

    def decode(self, prediction: torch.Tensor, descriptor: FieldDescriptor) -> torch.Tensor:
        """EVAL ONLY - argmax/denorm. Never call this inside the loss path."""
        if descriptor.value_type in _CLASS_TYPES:
            return prediction.argmax(dim=-1)
        if descriptor.value_type in _BINARY_TYPES:
            return (torch.sigmoid(prediction) > 0.5).to(torch.long)
        out = prediction
        if descriptor.value_range is not None:
            low, high = descriptor.value_range
            out = out * (high - low) + low
        return out


def target_positions(target_encoded: torch.Tensor) -> int:
    return target_encoded.shape[1]


def model_output_features(descriptor: FieldDescriptor, n_positions: int) -> int:
    """Width a model head must emit: n_positions * n_classes for class targets, else n_positions."""
    if descriptor.value_type in _CLASS_TYPES:
        assert descriptor.n_classes is not None
        return n_positions * descriptor.n_classes
    return n_positions


def as_logits(raw: torch.Tensor, descriptor: FieldDescriptor, n_positions: int) -> torch.Tensor:
    """Reshape a flat model head output into [B, P, n_classes] for class targets; pass through otherwise."""
    if descriptor.value_type in _CLASS_TYPES:
        assert descriptor.n_classes is not None
        return raw.reshape(raw.shape[0], n_positions, descriptor.n_classes)
    return raw


def loss_fn(prediction: torch.Tensor, target: torch.Tensor, descriptor: FieldDescriptor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Dispatch on value_type; mask zeroes padded positions (normalized by valid count)."""
    value_type = descriptor.value_type
    if value_type in _CLASS_TYPES:
        batch, positions, classes = prediction.shape
        elementwise = F.cross_entropy(prediction.reshape(batch * positions, classes), target.reshape(batch * positions), reduction="none").reshape(batch, positions)
    elif value_type in _BINARY_TYPES:
        elementwise = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    elif value_type is ValueType.CONTINUOUS:
        elementwise = F.mse_loss(prediction, target, reduction="none")
    else:
        raise ValueError(f"unsupported value_type {value_type}")
    if mask is not None:  # mask True = padding; weight only the real (~mask) positions
        weights = (~mask).to(elementwise.dtype)
        return (elementwise * weights).sum() / weights.sum().clamp_min(1.0)
    return elementwise.mean()


_EncodedInput = tuple[torch.Tensor, FieldDescriptor]
_EncodedTarget = tuple[torch.Tensor, torch.Tensor | None, FieldDescriptor]


@dataclass(frozen=True, slots=True)
class EncodedTask:
    """A whole `Task` turned into model-ready tensors. `query_*` are None for a degenerate query-less task."""

    support_input: _EncodedInput
    support_target: _EncodedTarget
    query_input: _EncodedInput | None
    query_target: _EncodedTarget | None


def encode_task(task: Task, encoder: Level0Encoder) -> EncodedTask:
    """Encode a Task fully into input + target tensors for both support and query (batched WITHIN the task).

    A real consuming model brings its own pipeline; this shows that a structural `Task` becomes tensors with
    no per-rung special-casing - everything dispatches on the `Field` descriptor.
    """
    support_input_field, support_output_field = support_loader(task)
    query_input = query_target = None
    if task.query:
        query_input_field, query_output_field = query_loader(task)
        query_input = encoder.encode(query_input_field)
        query_target = encoder.encode_target(query_output_field)
    return EncodedTask(
        support_input=encoder.encode(support_input_field),
        support_target=encoder.encode_target(support_output_field),
        query_input=query_input,
        query_target=query_target,
    )
