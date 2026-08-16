"""Structured grid evaluation without task-name or rung special cases.

The ordinary Icarus encoder flattens a grid and scores cells under a padding mask.  That is a
useful dense training signal, but it is not an ARC answer: an answer also owns its height and
width, and the whole grid must be correct.  This module wraps an ``EncodedTask`` with the missing
shape information and learns a deliberately small integer shape program from support examples.
Query shapes are used only as labels during reporting, never to choose the program.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import torch

from versal.dataset.icarus import Axis, EncodedTask, Field, FieldDescriptor, Level0Encoder, Task, ValueType, as_logits, target_positions
from versal.evaluation import split_metrics_from_raw

if TYPE_CHECKING:
    from versal.evaluation import DecodingEncoder


@dataclass(frozen=True, slots=True)
class ShapeRule:
    """One integer affine output-dimension rule over input height and width."""

    height_coefficient: int
    width_coefficient: int
    bias: int

    def __call__(self, shape: tuple[int, int]) -> int:
        height, width = shape
        return self.height_coefficient * height + self.width_coefficient * width + self.bias

    @property
    def complexity(self) -> int:
        return int(self.height_coefficient != 0) + int(self.width_coefficient != 0) + abs(self.height_coefficient) + abs(self.width_coefficient) + int(self.bias != 0)


@dataclass(frozen=True, slots=True)
class ShapeProgram:
    height: ShapeRule
    width: ShapeRule
    max_height: int
    max_width: int

    def predict(self, shape: tuple[int, int]) -> tuple[int, int]:
        return (
            max(1, min(self.max_height, self.height(shape))),
            max(1, min(self.max_width, self.width(shape))),
        )


def _fit_dimension(inputs: tuple[tuple[int, int], ...], targets: tuple[int, ...]) -> ShapeRule:
    """Return the simplest exact affine rule, or the modal constant when none exists.

    The coefficient range is intentionally small.  It expresses the generic dimension relations
    common to tensor transforms (copy, swap, crop/expand by a constant, double/halve-like integer
    maps) without embedding an ARC operation vocabulary.
    """

    candidates: list[ShapeRule] = []
    bias_limit = max(32, max(targets, default=1))
    for a, b in product(range(-2, 3), repeat=2):
        for bias in range(-bias_limit, bias_limit + 1):
            rule = ShapeRule(a, b, bias)
            if all(rule(shape) == target for shape, target in zip(inputs, targets)):
                candidates.append(rule)
    if candidates:
        return min(
            candidates,
            key=lambda rule: (
                rule.complexity,
                abs(rule.bias),
                abs(rule.height_coefficient) + abs(rule.width_coefficient),
                rule.height_coefficient,
                rule.width_coefficient,
                rule.bias,
            ),
        )
    counts: dict[int, int] = {}
    for target in targets:
        counts[target] = counts.get(target, 0) + 1
    modal = min(counts, key=lambda value: (-counts[value], value)) if counts else 1
    return ShapeRule(0, 0, modal)


def fit_shape_program(
    input_shapes: tuple[tuple[int, int], ...],
    output_shapes: tuple[tuple[int, int], ...],
    *,
    max_height: int,
    max_width: int,
) -> ShapeProgram:
    if not input_shapes or len(input_shapes) != len(output_shapes):
        raise ValueError("shape-program fitting needs paired, non-empty support shapes")
    return ShapeProgram(
        height=_fit_dimension(input_shapes, tuple(shape[0] for shape in output_shapes)),
        width=_fit_dimension(input_shapes, tuple(shape[1] for shape in output_shapes)),
        max_height=max_height,
        max_width=max_width,
    )


def _real_shape(field: Field) -> tuple[int, int]:
    height_axis = field.axes.index(Axis.HEIGHT)
    width_axis = field.axes.index(Axis.WIDTH)
    if field.mask is None:
        return int(field.data.shape[height_axis]), int(field.data.shape[width_axis])
    valid = ~field.mask

    def extent(axis: int) -> int:
        reduce = tuple(index for index in range(valid.ndim) if index != axis)
        occupied = valid.any(dim=reduce) if reduce else valid
        positions = occupied.nonzero(as_tuple=False).flatten()
        return int(positions.max()) + 1 if positions.numel() else 1

    return extent(height_axis), extent(width_axis)


def _support_grid_fields(task: Task) -> tuple[list[Field], list[Field]] | None:
    """Detect structured grids from support pairs without inspecting held-out labels."""

    inputs = [field for field, _target in task.support]
    outputs = [target for _field, target in task.support]
    if not inputs or not outputs:
        return None
    spatial = {Axis.HEIGHT, Axis.WIDTH}
    if any(field.data.ndim != 2 or set(field.axes) != spatial for field in inputs + outputs):
        return None
    if any(field.value_type is not ValueType.CATEGORICAL for field in inputs + outputs):
        return None
    return inputs, outputs


@dataclass(frozen=True, slots=True)
class StructuredGridEncoded:
    """Encoded tensors plus support-derived shape and exact-grid evaluation state."""

    base: EncodedTask
    support_input_shapes: tuple[tuple[int, int], ...]
    support_output_shapes: tuple[tuple[int, int], ...]
    query_input_shapes: tuple[tuple[int, int], ...]
    query_output_shapes: tuple[tuple[int, int], ...]
    support_output_cells: tuple[int, ...]
    query_output_cells: tuple[int, ...]
    support_input_coverage: tuple[float, ...]
    query_input_coverage: tuple[float, ...]
    shape_program: ShapeProgram

    @property
    def support_input(self) -> tuple[torch.Tensor, FieldDescriptor]:
        return self.base.support_input

    @property
    def support_target(self) -> tuple[torch.Tensor, torch.Tensor | None, FieldDescriptor]:
        return self.base.support_target

    @property
    def query_input(self) -> tuple[torch.Tensor, FieldDescriptor] | None:
        return self.base.query_input

    @property
    def query_target(self) -> tuple[torch.Tensor, torch.Tensor | None, FieldDescriptor] | None:
        return self.base.query_target

    def without_query(self) -> "StructuredGridEncoded":
        return StructuredGridEncoded(
            base=EncodedTask(self.base.support_input, self.base.support_target, None, None),
            support_input_shapes=self.support_input_shapes,
            support_output_shapes=self.support_output_shapes,
            query_input_shapes=(),
            query_output_shapes=(),
            support_output_cells=self.support_output_cells,
            query_output_cells=(),
            support_input_coverage=self.support_input_coverage,
            query_input_coverage=(),
            shape_program=self.shape_program,
        )


def _real_cell_count(field: Field) -> int:
    return int((~field.mask).sum()) if field.mask is not None else int(field.data.numel())


def _descriptor(field: Field) -> FieldDescriptor:
    return FieldDescriptor(field.axes, field.value_type, field.n_classes, field.value_range)


def _as_hw(field: Field) -> tuple[torch.Tensor, torch.Tensor | None]:
    height_axis = field.axes.index(Axis.HEIGHT)
    width_axis = field.axes.index(Axis.WIDTH)
    data = field.data.movedim((height_axis, width_axis), (0, 1))
    mask = None if field.mask is None else field.mask.movedim((height_axis, width_axis), (0, 1))
    return data, mask


def _encode_inputs(fields: list[Field], canvas: tuple[int, int], descriptor: FieldDescriptor) -> tuple[tuple[torch.Tensor, FieldDescriptor], tuple[float, ...]]:
    height, width = canvas
    rows: list[torch.Tensor] = []
    coverages: list[float] = []
    for field in fields:
        data, mask = _as_hw(field)
        take_height, take_width = min(height, data.shape[0]), min(width, data.shape[1])
        row = torch.zeros((height, width), dtype=torch.float32)
        copied = data[:take_height, :take_width].to(torch.float32)
        if mask is not None:
            copied = copied.masked_fill(mask[:take_height, :take_width], 0.0)
        row[:take_height, :take_width] = copied
        rows.append(row.reshape(-1))
        covered = take_height * take_width if mask is None else int((~mask[:take_height, :take_width]).sum())
        coverages.append(covered / max(_real_cell_count(field), 1))
    return (torch.stack(rows), descriptor), tuple(coverages)


def _encode_targets(fields: list[Field], canvas: tuple[int, int], descriptor: FieldDescriptor) -> tuple[torch.Tensor, torch.Tensor, FieldDescriptor]:
    height, width = canvas
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for field in fields:
        data, source_mask = _as_hw(field)
        take_height, take_width = min(height, data.shape[0]), min(width, data.shape[1])
        target = torch.zeros((height, width), dtype=torch.long)
        mask = torch.ones((height, width), dtype=torch.bool)
        target[:take_height, :take_width] = data[:take_height, :take_width].to(torch.long)
        if source_mask is None:
            mask[:take_height, :take_width] = False
        else:
            mask[:take_height, :take_width] = source_mask[:take_height, :take_width]
        targets.append(target.reshape(-1))
        masks.append(mask.reshape(-1))
    return torch.stack(targets), torch.stack(masks), descriptor


def encode_structured_grid(task: Task, encoder: Level0Encoder, *, include_query: bool = True) -> StructuredGridEncoded | None:
    """Encode a grid task, optionally without reading or encoding any query target."""

    fields = _support_grid_fields(task)
    if fields is None:
        return None
    support_input_fields, support_output_fields = fields
    support_inputs = tuple(_real_shape(field) for field in support_input_fields)
    support_outputs = tuple(_real_shape(field) for field in support_output_fields)
    query_inputs = tuple(_real_shape(field) for field, _target in task.query) if include_query else ()
    query_output_fields = [target for _field, target in task.query] if include_query else []
    if include_query and any(
        field.data.ndim != 2 or set(field.axes) != {Axis.HEIGHT, Axis.WIDTH} or field.value_type is not ValueType.CATEGORICAL for field in query_output_fields
    ):
        return None
    query_outputs = tuple(_real_shape(field) for field in query_output_fields)
    # ARC's public 30x30 domain bound permits support-derived identity/scale rules to extrapolate;
    # unlike a support-max clamp, it does not make a larger valid query shape impossible.
    observed_shapes = (*support_inputs, *support_outputs)
    max_height = max(30, *(shape[0] for shape in observed_shapes))
    max_width = max(30, *(shape[1] for shape in observed_shapes))
    input_canvas = (
        max(shape[0] for shape in support_inputs),
        max(shape[1] for shape in support_inputs),
    )
    output_canvas = (
        max(shape[0] for shape in support_outputs),
        max(shape[1] for shape in support_outputs),
    )
    input_descriptor = _descriptor(support_input_fields[0])
    output_descriptor = _descriptor(support_output_fields[0])
    support_input, support_input_coverage = _encode_inputs(support_input_fields, input_canvas, input_descriptor)
    support_target = _encode_targets(support_output_fields, output_canvas, output_descriptor)
    query_input = query_target = None
    query_input_coverage: tuple[float, ...] = ()
    if include_query and task.query:
        query_input_fields = [field for field, _target in task.query]
        query_input, query_input_coverage = _encode_inputs(query_input_fields, input_canvas, input_descriptor)
        query_target = _encode_targets(query_output_fields, output_canvas, output_descriptor)
    return StructuredGridEncoded(
        base=EncodedTask(support_input, support_target, query_input, query_target),
        support_input_shapes=support_inputs,
        support_output_shapes=support_outputs,
        query_input_shapes=query_inputs,
        query_output_shapes=query_outputs,
        support_output_cells=tuple(_real_cell_count(field) for field in support_output_fields),
        query_output_cells=tuple(_real_cell_count(field) for field in query_output_fields),
        support_input_coverage=support_input_coverage,
        query_input_coverage=query_input_coverage,
        shape_program=fit_shape_program(support_inputs, support_outputs, max_height=max_height, max_width=max_width),
    )


def _example_exact(predictions: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    correct = predictions == target.to(torch.long)
    if mask is None:
        return correct.reshape(correct.shape[0], -1).all(dim=1)
    valid = ~mask
    return (correct | ~valid).reshape(correct.shape[0], -1).all(dim=1)


def _cell_accuracy(predictions: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> float:
    correct = predictions == target.to(torch.long)
    if mask is None:
        return float(correct.float().mean())
    valid = ~mask
    return float((correct & valid).sum().float() / valid.sum().clamp_min(1))


def _mode(values: torch.Tensor, mask: torch.Tensor | None) -> int:
    flat = values.reshape(-1)
    if mask is not None:
        flat = flat[~mask.reshape(-1)]
    return int(torch.mode(flat.to(torch.long)).values) if flat.numel() else 0


def _baseline_metrics(
    encoded: StructuredGridEncoded,
    *,
    split: str,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    inputs: torch.Tensor,
    input_shapes: tuple[tuple[int, int], ...],
    output_shapes: tuple[tuple[int, int], ...],
    coverage: torch.Tensor,
) -> dict[str, float]:
    support_target, support_mask, _descriptor = encoded.support_target
    constant_value = _mode(support_target, support_mask)
    constant = torch.full_like(target, constant_value)
    positions = target.shape[1]
    if inputs.shape[1] >= positions:
        copied = inputs[:, :positions].round().to(torch.long)
    else:
        copied = torch.zeros_like(target, dtype=torch.long)
        copied[:, : inputs.shape[1]] = inputs.round().to(torch.long)
    mean_coverage = float(coverage.mean()) if coverage.numel() else 0.0
    constant_accuracy = _cell_accuracy(constant, target, mask) * mean_coverage
    copy_accuracy = _cell_accuracy(copied, target, mask) * mean_coverage

    support_modal_shape = min(
        set(encoded.support_output_shapes),
        key=lambda shape: (-encoded.support_output_shapes.count(shape), shape),
    )
    constant_shapes = (support_modal_shape,) * len(output_shapes)
    constant_exact = _example_exact(constant, target, mask)
    copy_exact = _example_exact(copied, target, mask)
    if output_shapes:
        constant_exact &= torch.tensor([predicted == actual for predicted, actual in zip(constant_shapes, output_shapes)], device=constant_exact.device)
        copy_exact &= torch.tensor([predicted == actual for predicted, actual in zip(input_shapes, output_shapes)], device=copy_exact.device)
    complete = coverage >= 1.0
    constant_exact &= complete
    copy_exact &= complete
    return {
        f"{split}_constant_accuracy": constant_accuracy,
        f"{split}_copy_accuracy": copy_accuracy,
        f"{split}_baseline_accuracy": max(constant_accuracy, copy_accuracy),
        f"{split}_constant_exact": float(constant_exact.float().mean()),
        f"{split}_copy_exact": float(copy_exact.float().mean()),
        f"{split}_baseline_exact": max(float(constant_exact.float().mean()), float(copy_exact.float().mean())),
    }


def _structured_split(
    module: torch.nn.Module,
    encoded: StructuredGridEncoded,
    encoded_input: tuple,
    encoded_target: tuple,
    input_shapes: tuple[tuple[int, int], ...],
    output_shapes: tuple[tuple[int, int], ...],
    output_cells: tuple[int, ...],
    input_coverage: tuple[float, ...],
    encoder: "DecodingEncoder",
    split: str,
) -> dict[str, float]:
    x, _input_descriptor = encoded_input
    target, mask, descriptor = encoded_target
    raw = as_logits(module(x), descriptor, target_positions(target))
    accuracy, loss = split_metrics_from_raw(raw, target, mask, descriptor, encoder)
    predictions = encoder.decode(raw, descriptor)
    covered_cell_exact = _example_exact(predictions, target, mask)
    predicted_shapes = tuple(encoded.shape_program.predict(shape) for shape in input_shapes)
    shape_correct = torch.tensor(
        [predicted == actual for predicted, actual in zip(predicted_shapes, output_shapes)],
        dtype=torch.bool,
        device=covered_cell_exact.device,
    )
    if mask is None:
        covered_cells = torch.full((target.shape[0],), target.shape[1], dtype=torch.float32, device=covered_cell_exact.device)
    else:
        covered_cells = (~mask).reshape(mask.shape[0], -1).sum(dim=1).to(device=covered_cell_exact.device, dtype=torch.float32)
    actual_cells = torch.tensor(output_cells, dtype=torch.float32, device=covered_cell_exact.device).clamp_min(1.0)
    coverage = (covered_cells / actual_cells).clamp(max=1.0)
    complete = coverage >= 1.0
    cell_exact = covered_cell_exact & complete
    exact = cell_exact & shape_correct & complete if len(shape_correct) else cell_exact & complete
    baselines = _baseline_metrics(
        encoded,
        split=split,
        target=target,
        mask=mask,
        inputs=x,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        coverage=coverage,
    )
    mean_coverage = float(coverage.mean()) if coverage.numel() else 0.0
    mean_input_coverage = sum(input_coverage) / len(input_coverage) if input_coverage else 0.0
    full_accuracy = accuracy * mean_coverage
    return {
        f"{split}_accuracy": full_accuracy,
        f"{split}_covered_accuracy": accuracy,
        f"{split}_coverage": mean_coverage,
        f"{split}_input_coverage": mean_input_coverage,
        f"{split}_loss": loss,
        f"{split}_covered_cell_exact": float(covered_cell_exact.float().mean()),
        f"{split}_cell_exact": float(cell_exact.float().mean()),
        f"{split}_shape_accuracy": float(shape_correct.float().mean()) if len(shape_correct) else 0.0,
        f"{split}_exact": float(exact.float().mean()),
        f"{split}_task_exact": float(bool(exact.numel()) and bool(exact.all())),
        **baselines,
        f"{split}_gain_over_baseline": full_accuracy - baselines[f"{split}_baseline_accuracy"],
    }


def evaluate_structured_grid(module: torch.nn.Module, encoded: StructuredGridEncoded, encoder: "DecodingEncoder") -> dict[str, float]:
    with torch.no_grad():
        metrics = _structured_split(
            module,
            encoded,
            encoded.support_input,
            encoded.support_target,
            encoded.support_input_shapes,
            encoded.support_output_shapes,
            encoded.support_output_cells,
            encoded.support_input_coverage,
            encoder,
            "support",
        )
        if encoded.query_input is not None and encoded.query_target is not None:
            metrics.update(
                _structured_split(
                    module,
                    encoded,
                    encoded.query_input,
                    encoded.query_target,
                    encoded.query_input_shapes,
                    encoded.query_output_shapes,
                    encoded.query_output_cells,
                    encoded.query_input_coverage,
                    encoder,
                    "query",
                )
            )
        else:
            metrics.update({"query_accuracy": 0.0, "query_loss": float("inf")})
    return metrics
