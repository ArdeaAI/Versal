"""Task decomposition operators for the recursive orchestrator.

A decompose operator takes an Icarus `Task` and returns subtasks that are THEMSELVES fully
valid `Task` objects (sliced Fields with descriptors and masks intact), so the orchestrator
can recurse on them with zero special-casing. Each subtask carries a `PortSpec` telling the
orchestrator exactly how a mini-model solving the subtask wires back into a composition
solving the parent: which flat output positions it emits, which flat input columns it reads,
or which time window it owns.

Operators follow the lego-block convention used by every other evolutionary stage: they are
registered by name in `DECOMPOSE`, selected and parameterized from config via
`build_decomposers`, and uniform in signature (`op(task, *, rng, **params) -> list[Subtask]`,
returning [] when inapplicable) so the orchestrator can try every configured op blindly.
"""

import functools
import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from ardevo.dataset.icarus import Axis, Field, FieldDescriptor, Task, TaskMeta
from ardevo.evolution.registry import Registry, _bind_prefixed


@dataclass(frozen=True)
class PortSpec:
    """How a subtask's mini-model plugs back into the parent composition.

    `offsets` is a half-open [start, end) range whose unit depends on `role`: flat output
    positions for "output_slice", flat input columns for "input_subset", and time steps for
    "time_window". `descriptor` describes the SLICED side (the subtask's output for
    output_slice/time_window, its input for input_subset) so the orchestrator can size heads
    and encoders without re-deriving structure from the subtask's tensors.
    """

    role: str
    offsets: tuple[int, int]
    width: int
    descriptor: FieldDescriptor


@dataclass(frozen=True)
class Subtask:
    """A fully valid Icarus `Task` plus the port that locates it inside the parent."""

    task: Task
    port: PortSpec


DecomposeOp = Callable[..., list[Subtask]]
DECOMPOSE: Registry[DecomposeOp] = Registry("decompose")


def _chunk_bounds(size: int, n_chunks: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) chunks, as even as possible; the last chunk absorbs the remainder."""
    base = size // n_chunks
    bounds = [(group * base, (group + 1) * base) for group in range(n_chunks - 1)]
    bounds.append(((n_chunks - 1) * base, size))
    return bounds


def _field_descriptor(field: Field) -> FieldDescriptor:
    return FieldDescriptor(axes=field.axes, value_type=field.value_type, n_classes=field.n_classes, value_range=field.value_range)


def _slice_field(field: Field, axis_position: int, start: int, end: int) -> Field:
    """Build a genuinely new Field from one axis slice.

    Going back through the Field constructor (instead of mutating or sharing) is the point:
    `Field.__post_init__` revalidates shapes and mask alignment, so an invalid slice fails
    loudly here rather than deep inside the orchestrator's recursion.
    """
    window = tuple(slice(start, end) if dim == axis_position else slice(None) for dim in range(field.data.ndim))
    return Field(
        data=field.data[window],
        axes=field.axes,
        value_type=field.value_type,
        n_classes=field.n_classes,
        value_range=field.value_range,
        mask=None if field.mask is None else field.mask[window],
    )


def _child_meta(parent: TaskMeta, suffix: str) -> TaskMeta:
    """Subtask identity stays traceable to the parent: same rung/kind/split, dotted name suffix."""
    return TaskMeta(rung=parent.rung, kind=parent.kind, name=f"{parent.name}{suffix}", fixed_split=parent.fixed_split, class_names=parent.class_names)


@DECOMPOSE.register("output_slices")
def output_slices(task: Task, *, rng: random.Random, n_groups: int = 2) -> list[Subtask]:
    """Slice the OUTPUT of every pair along its first axis into n_groups contiguous chunks.

    Inputs pass through whole: each subtask sees the full problem but answers only one chunk
    of it, so mini-model heads concatenate back into the parent's flat output losslessly.
    `rng` is accepted for signature uniformity; this v1 op is deterministic.

    Returns:
        One Subtask per chunk, or [] when the output is 0-d or narrower than n_groups.
    """
    _, first_output = task.support[0]
    if first_output.data.ndim == 0:
        return []
    first_axis_size = first_output.data.shape[0]
    if first_axis_size < n_groups:
        return []
    # Port offsets live in the parent's FLAT output positions (row-major flatten of the whole
    # field), so a chunk of the first axis spans trailing-many flat positions per step.
    trailing = math.prod(first_output.data.shape[1:])
    subtasks: list[Subtask] = []
    for group, (start, end) in enumerate(_chunk_bounds(first_axis_size, n_groups)):
        support = [(inp, _slice_field(out, 0, start, end)) for inp, out in task.support]
        query = [(inp, _slice_field(out, 0, start, end)) for inp, out in task.query]
        child = Task(meta=_child_meta(task.meta, f".out{group}"), support=support, query=query)
        port = PortSpec(role="output_slice", offsets=(start * trailing, end * trailing), width=(end - start) * trailing, descriptor=_field_descriptor(support[0][1]))
        subtasks.append(Subtask(task=child, port=port))
    return subtasks


@DECOMPOSE.register("input_subsets")
def input_subsets(task: Task, *, rng: random.Random, n_subsets: int = 2) -> list[Subtask]:
    """Slice the INPUT of every pair along its first axis into n_subsets contiguous chunks.

    Outputs pass through whole: each subtask predicts the full target from a restricted view,
    which is how the orchestrator probes whether a region of the input suffices on its own.
    `rng` is accepted for signature uniformity; this v1 op is deterministic.

    Returns:
        One Subtask per chunk, or [] when the input is 0-d or narrower than n_subsets.
    """
    first_input, _ = task.support[0]
    if first_input.data.ndim == 0:
        return []
    first_axis_size = first_input.data.shape[0]
    if first_axis_size < n_subsets:
        return []
    trailing = math.prod(first_input.data.shape[1:])
    subtasks: list[Subtask] = []
    for group, (start, end) in enumerate(_chunk_bounds(first_axis_size, n_subsets)):
        support = [(_slice_field(inp, 0, start, end), out) for inp, out in task.support]
        query = [(_slice_field(inp, 0, start, end), out) for inp, out in task.query]
        child = Task(meta=_child_meta(task.meta, f".in{group}"), support=support, query=query)
        port = PortSpec(role="input_subset", offsets=(start * trailing, end * trailing), width=(end - start) * trailing, descriptor=_field_descriptor(support[0][0]))
        subtasks.append(Subtask(task=child, port=port))
    return subtasks


@DECOMPOSE.register("time_windows")
def time_windows(task: Task, *, rng: random.Random, n_windows: int = 2) -> list[Subtask]:
    """Slice BOTH input and output along their TIME axis into n_windows aligned windows.

    Aligned windows preserve the step-for-step correspondence a seq-to-seq task encodes, so a
    mini-model owns a contiguous span of time and its predictions concatenate back along the
    time axis. Offsets are therefore in TIME STEPS, not flat positions.
    `rng` is accepted for signature uniformity; this v1 op is deterministic.

    Returns:
        One Subtask per window, or [] when either side lacks a TIME axis, the time lengths
        disagree, or the time length is shorter than n_windows.
    """
    first_input, first_output = task.support[0]
    if Axis.TIME not in first_input.axes or Axis.TIME not in first_output.axes:
        return []
    input_time_position = first_input.axes.index(Axis.TIME)
    output_time_position = first_output.axes.index(Axis.TIME)
    time_length = first_input.data.shape[input_time_position]
    if first_output.data.shape[output_time_position] != time_length:
        return []
    if time_length < n_windows:
        return []
    subtasks: list[Subtask] = []
    for group, (start, end) in enumerate(_chunk_bounds(time_length, n_windows)):
        support = [(_slice_field(inp, input_time_position, start, end), _slice_field(out, output_time_position, start, end)) for inp, out in task.support]
        query = [(_slice_field(inp, input_time_position, start, end), _slice_field(out, output_time_position, start, end)) for inp, out in task.query]
        child = Task(meta=_child_meta(task.meta, f".t{group}"), support=support, query=query)
        port = PortSpec(role="time_window", offsets=(start, end), width=end - start, descriptor=_field_descriptor(support[0][1]))
        subtasks.append(Subtask(task=child, port=port))
    return subtasks


@DECOMPOSE.register("spatial_patches")
def spatial_patches(task: Task, *, rng: random.Random, n_patches: int = 2) -> list[Subtask]:
    """Slice BOTH input and output along their first shared SPATIAL axis (HEIGHT, else WIDTH) into
    n_patches aligned bands: the grid->grid analogue of `time_windows`, and the ARC-portable
    decomposition. Each subtask transforms a contiguous band of the grid and the bands tile back
    losslessly along that axis. Unlike I/O-flat slicing on entangled tasks, a local grid transform
    genuinely decomposes, because a band's output depends only on the same band's input.

    Returns one Subtask per band, or [] when input and output do not share a spatial axis of equal
    length (e.g. a grid->label classification task, where the output carries no spatial axis).
    """
    first_input, first_output = task.support[0]
    spatial = next((axis for axis in (Axis.HEIGHT, Axis.WIDTH) if axis in first_input.axes and axis in first_output.axes), None)
    if spatial is None:
        return []
    input_position = first_input.axes.index(spatial)
    output_position = first_output.axes.index(spatial)
    length = first_input.data.shape[input_position]
    if first_output.data.shape[output_position] != length or length < n_patches:
        return []
    subtasks: list[Subtask] = []
    for group, (start, end) in enumerate(_chunk_bounds(length, n_patches)):
        support = [(_slice_field(inp, input_position, start, end), _slice_field(out, output_position, start, end)) for inp, out in task.support]
        query = [(_slice_field(inp, input_position, start, end), _slice_field(out, output_position, start, end)) for inp, out in task.query]
        child = Task(meta=_child_meta(task.meta, f".{spatial.value.lower()}{group}"), support=support, query=query)
        port = PortSpec(role="spatial_patch", offsets=(start, end), width=end - start, descriptor=_field_descriptor(support[0][1]))
        subtasks.append(Subtask(task=child, port=port))
    return subtasks


def build_decomposers(table: dict[str, Any]) -> list[tuple[str, Callable[..., list[Subtask]]]]:
    """Resolve the configured decompose ops and bind their `{name}_{param}` prefixed params.

    Mirrors how `build_evolver` binds mutation operators so decomposition stays a config-driven
    lego block: the orchestrator receives named, ready-to-call partials and never reads raw
    config. Unknown names fail loudly via `Registry.get` (KeyError).
    """
    return [(name, functools.partial(DECOMPOSE.get(name), **_bind_prefixed(table, name))) for name in table.get("decompose", [])]
