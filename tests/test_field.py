from __future__ import annotations

import torch

from ardevo.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from ardevo.field import field_contract, field_feature_width, gather_local_multiscale_v1


def _field(data: torch.Tensor, axes: tuple[Axis, ...], mask: torch.Tensor | None = None) -> Field:
    return Field(data, axes, ValueType.CONTINUOUS, None, None, mask)


def _task(query_size: int = 9) -> Task:
    axes = (Axis.CHANNEL, Axis.HEIGHT, Axis.WIDTH)
    support = [(_field(torch.arange(2 * 4 * 5).reshape(2, 4, 5), axes), _field(torch.zeros(1, 4, 5), axes))]
    query = [(_field(torch.zeros(2, query_size, query_size), axes), _field(torch.zeros(1, query_size, query_size), axes))]
    return Task(TaskMeta(999, TaskKind.MAP, "arbitrary"), support, query)


def test_contract_is_support_only_and_symbolic() -> None:
    assert field_contract(_task(9)) == field_contract(_task(101))
    contract = field_contract(_task())
    assert contract is not None and "height" in contract.to_dict()["spatial"]


def test_semantic_axis_permutation_and_lazy_feature_width() -> None:
    task = _task()
    contract = field_contract(task)
    assert contract is not None and contract.input_channels == 2
    original = task.support[0][0]
    permuted = _field(original.data.permute(1, 2, 0), (Axis.HEIGHT, Axis.WIDTH, Axis.CHANNEL))
    sites = torch.tensor([[0, 0], [2, 3]])
    left = gather_local_multiscale_v1(original, sites)
    right = gather_local_multiscale_v1(permuted, sites)
    assert left.shape == (2, field_feature_width(2))
    torch.testing.assert_close(left, right)


def test_rejects_spatial_mismatch_and_time() -> None:
    task = _task()
    inp, out = task.support[0]
    mismatch = _field(torch.zeros(1, 3, 5), out.axes)
    assert field_contract(Task(task.meta, [(inp, mismatch)], [])) is None
    temporal = _field(torch.zeros(2, 4, 5), (Axis.TIME, Axis.HEIGHT, Axis.WIDTH))
    assert field_contract(Task(task.meta, [(temporal, out)], [])) is None
