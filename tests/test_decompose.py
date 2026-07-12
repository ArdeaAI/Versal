"""Decomposition operator tests: every subtask must be a valid Icarus Task that reassembles into its parent."""

import random

import pytest
import torch

from ardevo.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from ardevo.decompose import Subtask, build_decomposers, input_subsets, output_slices, time_windows


def _stacked_outputs(pairs: list[tuple[Field, Field]]) -> torch.Tensor:
    return torch.stack([out.data for _, out in pairs])


def test_output_slices_splits_output_and_reassembles(decomposable_task: Task) -> None:
    subtasks = output_slices(decomposable_task, rng=random.Random(0), n_groups=2)
    assert len(subtasks) == 2
    assert [sub.port.offsets for sub in subtasks] == [(0, 1), (1, 2)]
    for sub in subtasks:
        assert sub.port.role == "output_slice"
        assert sub.port.width == 1
        assert sub.port.descriptor.value_type is ValueType.BINARY
        for (parent_input, _), (sub_input, sub_output) in zip(decomposable_task.support, sub.task.support, strict=True):
            assert torch.equal(parent_input.data, sub_input.data)
            assert sub_output.data.shape == torch.Size([1])
    # Writing each subtask's stacked support targets into its port slot must rebuild the parent
    # targets exactly: the whole point of output_slice ports.
    parent_targets = _stacked_outputs(decomposable_task.support)
    reassembled = torch.zeros_like(parent_targets)
    for sub in subtasks:
        start, end = sub.port.offsets
        reassembled[:, start:end] = _stacked_outputs(sub.task.support)
    assert torch.equal(reassembled, parent_targets)


def test_output_slices_inapplicable_when_groups_exceed_width(decomposable_task: Task) -> None:
    assert output_slices(decomposable_task, rng=random.Random(0), n_groups=3) == []


def test_input_subsets_splits_inputs_only(decomposable_task: Task) -> None:
    subtasks = input_subsets(decomposable_task, rng=random.Random(0), n_subsets=2)
    assert len(subtasks) == 2
    assert [sub.port.offsets for sub in subtasks] == [(0, 4), (4, 8)]
    for sub in subtasks:
        assert sub.port.role == "input_subset"
        parent_pairs = decomposable_task.support + decomposable_task.query
        sub_pairs = sub.task.support + sub.task.query
        for (_, parent_output), (sub_input, sub_output) in zip(parent_pairs, sub_pairs, strict=True):
            assert sub_input.data.shape == torch.Size([4])
            assert torch.equal(parent_output.data, sub_output.data)


def test_time_windows_aligned_and_reassembles(temporal_seq_task: Task) -> None:
    subtasks = time_windows(temporal_seq_task, rng=random.Random(0), n_windows=2)
    assert len(subtasks) == 2
    assert [sub.port.offsets for sub in subtasks] == [(0, 4), (4, 8)]
    for sub in subtasks:
        assert sub.port.role == "time_window"
        assert sub.port.width == 4
        for sub_input, sub_output in sub.task.support + sub.task.query:
            assert sub_input.data.shape[0] == 4
            assert sub_output.data.shape[0] == 4
    # Each subtask is a valid Task already (Field/Task validation ran in the constructor);
    # concatenating window targets along the time axis must rebuild the parent exactly.
    parent_targets = _stacked_outputs(temporal_seq_task.support)
    reassembled = torch.zeros_like(parent_targets)
    for sub in subtasks:
        start, end = sub.port.offsets
        reassembled[:, start:end] = _stacked_outputs(sub.task.support)
    assert torch.equal(reassembled, parent_targets)


def test_time_windows_requires_time_axis_on_both_sides(decomposable_task: Task, temporal_task: Task) -> None:
    assert time_windows(decomposable_task, rng=random.Random(0), n_windows=2) == []
    # temporal_task carries TIME only on the input; its target is (EXTRA,), so no aligned windows exist.
    assert time_windows(temporal_task, rng=random.Random(0), n_windows=2) == []


def test_subtasks_preserve_structure_and_meta(decomposable_task: Task, temporal_seq_task: Task) -> None:
    cases = [
        (output_slices(decomposable_task, rng=random.Random(0), n_groups=2), decomposable_task, ".out"),
        (input_subsets(decomposable_task, rng=random.Random(0), n_subsets=2), decomposable_task, ".in"),
        (time_windows(temporal_seq_task, rng=random.Random(0), n_windows=2), temporal_seq_task, ".t"),
    ]
    for subtasks, parent, suffix in cases:
        assert subtasks
        for group, sub in enumerate(subtasks):
            assert sub.task.support
            assert len(sub.task.support) == len(parent.support)
            assert len(sub.task.query) == len(parent.query)
            assert sub.task.meta.rung == parent.meta.rung
            assert sub.task.meta.kind == parent.meta.kind
            assert sub.task.meta.fixed_split == parent.meta.fixed_split
            assert sub.task.meta.name == f"{parent.meta.name}{suffix}{group}"


def test_build_decomposers_binds_prefixed_params(decomposable_task: Task) -> None:
    table = {"decompose": ["output_slices", "input_subsets"], "output_slices_n_groups": 2, "input_subsets_n_subsets": 2}
    decomposers = build_decomposers(table)
    assert [name for name, _ in decomposers] == ["output_slices", "input_subsets"]
    for _, op in decomposers:
        subtasks = op(decomposable_task, rng=random.Random(0))
        assert len(subtasks) == 2
        assert all(isinstance(sub, Subtask) for sub in subtasks)


def test_build_decomposers_unknown_name_raises() -> None:
    with pytest.raises(KeyError):
        build_decomposers({"decompose": ["nonexistent_op"]})


def test_output_slices_adapts_group_count_to_model_head_budget() -> None:
    grid = torch.zeros(5, 2, dtype=torch.long)
    field = Field(grid, (Axis.HEIGHT, Axis.WIDTH), ValueType.CATEGORICAL, 2, None, None)
    task = Task(TaskMeta(0, TaskKind.MAP, "grid", fixed_split=True), [(field, field)], [(field, field)])
    # One first-axis unit costs 2 positions * 2 logits = 4 features, so an 8-feature cap requires
    # two units per group even when the configured minimum asks for only one group.
    subtasks = output_slices(task, rng=random.Random(0), n_groups=1, max_output_features=8)
    assert len(subtasks) == 3
    assert all(subtask.task.support[0][1].data.shape[0] <= 2 for subtask in subtasks)


def test_output_slices_declines_heterogeneous_support_canvases() -> None:
    small = Field(torch.zeros(2, 2, dtype=torch.long), (Axis.HEIGHT, Axis.WIDTH), ValueType.CATEGORICAL, 2, None, None)
    large = Field(torch.zeros(4, 2, dtype=torch.long), (Axis.HEIGHT, Axis.WIDTH), ValueType.CATEGORICAL, 2, None, None)
    task = Task(TaskMeta(18, TaskKind.MAP, "heterogeneous", fixed_split=True), [(small, small), (large, large)], [(small, small)])
    assert output_slices(task, rng=random.Random(0), n_groups=2) == []


def test_output_slices_carries_masks() -> None:
    input_field = Field(torch.tensor([1.0, 0.0, 1.0, 0.0]), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
    output_mask = torch.tensor([False, True, False, True])
    output_field = Field(torch.tensor([1.0, 0.0, 1.0, 0.0]), (Axis.EXTRA,), ValueType.BINARY, None, None, output_mask)
    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name="masked", fixed_split=True)
    parent = Task(meta=meta, support=[(input_field, output_field)], query=[(input_field, output_field)])

    subtasks = output_slices(parent, rng=random.Random(0), n_groups=2)
    assert len(subtasks) == 2
    for sub, (start, end) in zip(subtasks, [(0, 2), (2, 4)], strict=True):
        assert sub.port.offsets == (start, end)
        for _, sub_output in sub.task.support + sub.task.query:
            assert sub_output.mask is not None
            assert torch.equal(sub_output.mask, output_mask[start:end])
