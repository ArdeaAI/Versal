"""Variable-width targets (the psicov class): support and query targets in one task may have
different natural sizes (per-protein LxL distance maps). Every model head in the system is sized by
the SUPPORT target, so an unfitted query target crashes the first evaluation (2026-07-06 smoke run,
rung 14: mse over 245,025 predictions vs a 167,281 target). The consumer-side fix is
`fit_query_target`: pad the encoded query target to the support width under a True (ignored) mask,
or crop a wider one to the overlap; same-width tasks pass through as the same object."""

import torch

from ardevo.dataset.icarus import Axis, Field, Level0Encoder, Task, TaskKind, TaskMeta, ValueType, encode_task, model_output_features
from ardevo.evaluation import evaluate, fit_query_target


class _FixedHead(torch.nn.Module):
    """Emits a constant [batch, width] output: enough to walk the evaluation path end to end."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.width)


def _continuous_task(support_positions: int, query_positions: int) -> Task:
    def pair(positions: int) -> tuple[Field, Field]:
        x = Field(torch.linspace(0.0, 1.0, positions), (Axis.EXTRA,), ValueType.CONTINUOUS, None, (0.0, 1.0), None)
        y = Field(torch.linspace(0.0, 1.0, positions), (Axis.EXTRA,), ValueType.CONTINUOUS, None, (0.0, 1.0), None)
        return x, y

    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name="variable_width", fixed_split=True)
    return Task(meta=meta, support=[pair(support_positions) for _ in range(4)], query=[pair(query_positions) for _ in range(3)])


def _categorical_task(support_positions: int, query_positions: int, n_classes: int = 3) -> Task:
    def pair(positions: int) -> tuple[Field, Field]:
        x = Field(torch.linspace(0.0, 1.0, positions), (Axis.EXTRA,), ValueType.CONTINUOUS, None, (0.0, 1.0), None)
        y = Field(torch.zeros(positions, dtype=torch.long), (Axis.EXTRA,), ValueType.CATEGORICAL, n_classes, None, None)
        return x, y

    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name="variable_width_classes", fixed_split=True)
    return Task(meta=meta, support=[pair(support_positions) for _ in range(4)], query=[pair(query_positions) for _ in range(3)])


def test_same_width_is_the_same_object() -> None:
    encoded = encode_task(_continuous_task(9, 9), Level0Encoder(9))
    assert fit_query_target(encoded) is encoded


def test_queryless_task_is_the_same_object() -> None:
    task = _continuous_task(9, 9)
    queryless = Task(meta=task.meta, support=task.support, query=[])
    encoded = encode_task(queryless, Level0Encoder(9))
    assert fit_query_target(encoded) is encoded


def test_narrow_query_target_pads_under_ignored_mask() -> None:
    encoded = fit_query_target(encode_task(_continuous_task(9, 4), Level0Encoder(9)))
    assert encoded.query_target is not None
    target, mask, _descriptor = encoded.query_target
    assert target.shape[1] == 9 and mask is not None and mask.shape == target.shape
    assert bool(mask[:, 4:].all()) and not bool(mask[:, :4].any())
    metrics = evaluate(_FixedHead(9), encoded, Level0Encoder(9))
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_wide_query_target_crops_to_the_support_width() -> None:
    encoded = fit_query_target(encode_task(_continuous_task(4, 9), Level0Encoder(4)))
    assert encoded.query_target is not None
    target, _mask, _descriptor = encoded.query_target
    assert target.shape[1] == 4
    metrics = evaluate(_FixedHead(4), encoded, Level0Encoder(4))
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_categorical_padding_keeps_the_logit_reshape_sound() -> None:
    encoded = fit_query_target(encode_task(_categorical_task(4, 2), Level0Encoder(4)))
    assert encoded.query_target is not None
    target, mask, descriptor = encoded.query_target
    assert target.shape[1] == 4 and target.dtype == torch.long
    assert mask is not None and bool(mask[:, 2:].all())
    head_width = model_output_features(descriptor, 4)
    metrics = evaluate(_FixedHead(head_width), encoded, Level0Encoder(4))
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
