"""Temporal encoding: rebuild the TIME axis that Level0 flattens away, for the recurrent substrate.

`versal/dataset/icarus.py` is vendored and stays untouched; everything here works from its public
contract (`Field.axes`, `BatchedField`, `FieldDescriptor`). Inputs become `[batch, time, features]`
for `RecurrentGraphNet`; targets stay FLAT in t-major order so the Icarus `loss_fn` / `as_logits`
dispatch and the whole evaluate path run unchanged. Normalization and masking replicate
`Level0Encoder.encode` exactly (the T=1 parity test pins this) so temporal and flat scores are
comparable.
"""

from dataclasses import dataclass

import torch

from versal.dataset.icarus import Axis, BatchedField, EncodedTask, FieldDescriptor, Level0Encoder, Task, ValueType, model_output_features, query_loader, support_loader
from versal.evaluation import evaluate
from versal.evolution.genome import Genome
from versal.reference_depth import DEFAULT_MAX_INLINE_DEPTH
from versal.substrate import SubstrateModule, decode_recurrent

_CLASS_TYPES = (ValueType.CATEGORICAL, ValueType.ORDINAL)


def has_time_axis(descriptor: FieldDescriptor) -> bool:
    return Axis.TIME in descriptor.axes


def _normalize_continuous(x: torch.Tensor, descriptor: FieldDescriptor) -> torch.Tensor:
    # Replicates the Level0 normalization (kept local: importing icarus privates would break when the
    # vendored file is regenerated).
    if descriptor.value_type is ValueType.CONTINUOUS and descriptor.value_range is not None:
        low, high = descriptor.value_range
        if high > low:
            return (x - low) / (high - low)
    return x


def _time_major(tensor: torch.Tensor, descriptor: FieldDescriptor) -> torch.Tensor:
    """Move the TIME axis to dim 1 (after batch) and flatten the rest: [B, T, features_per_step]."""
    time_dim = descriptor.axes.index(Axis.TIME) + 1  # +1 for the batch dim
    moved = tensor.movedim(time_dim, 1)
    return moved.reshape(moved.shape[0], moved.shape[1], -1)


class TemporalEncoder:
    """Encode a TIME-bearing field as a stepped sequence; everything else mirrors Level0."""

    def __init__(self, step_dim: int) -> None:
        self._step_dim = step_dim
        self._reference = Level0Encoder(step_dim)  # decode (argmax/threshold/denorm) is shared verbatim

    def _fit(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape
        if width == self._step_dim:
            return x
        if width > self._step_dim:
            return x[:, :, : self._step_dim]
        return torch.cat([x, torch.zeros(batch, steps, self._step_dim - width)], dim=2)

    def encode(self, batched: BatchedField) -> tuple[torch.Tensor, FieldDescriptor]:
        """Model input: [B, T, step_dim]. Normalize CONTINUOUS, zero masked pads, fit per-step width."""
        if not has_time_axis(batched.descriptor):
            raise ValueError(f"TemporalEncoder.encode needs a TIME axis; got {batched.descriptor.axes}")
        x = _time_major(batched.data.to(torch.float32), batched.descriptor)
        x = _normalize_continuous(x, batched.descriptor)
        if batched.mask is not None:
            x = x * (~_time_major(batched.mask, batched.descriptor).to(torch.bool)).to(torch.float32)
        return self._fit(x), batched.descriptor

    def encode_target(self, batched: BatchedField) -> tuple[torch.Tensor, torch.Tensor | None, FieldDescriptor]:
        """Loss target, FLAT like Level0 but t-major when the target carries TIME, matching the
        seq-to-seq module output layout (`RecurrentGraphNet(mode="all")`)."""
        batch = batched.data.shape[0]
        if has_time_axis(batched.descriptor):
            flat = _time_major(batched.data, batched.descriptor).reshape(batch, -1)
            mask = None if batched.mask is None else _time_major(batched.mask, batched.descriptor).reshape(batch, -1)
        else:
            flat = batched.data.reshape(batch, -1)
            mask = None if batched.mask is None else batched.mask.reshape(batch, -1)
        value_type = batched.descriptor.value_type
        if value_type in _CLASS_TYPES:
            return flat.to(torch.long), mask, batched.descriptor
        if value_type in (ValueType.BINARY, ValueType.MULTILABEL):
            return flat.to(torch.float32), mask, batched.descriptor
        return _normalize_continuous(flat.to(torch.float32), batched.descriptor), mask, batched.descriptor

    def decode(self, prediction: torch.Tensor, descriptor: FieldDescriptor) -> torch.Tensor:
        return self._reference.decode(prediction, descriptor)


def step_features(batched: BatchedField) -> int:
    """Per-step input width: the product of every non-TIME, non-batch dimension."""
    time_dim = batched.descriptor.axes.index(Axis.TIME) + 1
    width = 1
    for dim, size in enumerate(batched.data.shape[1:], start=1):
        if dim != time_dim:
            width *= int(size)
    return width


def encode_temporal_task(task: Task, encoder: TemporalEncoder) -> EncodedTask:
    """Mirror of `encode_task` with stepped inputs and t-major flat targets."""
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


@dataclass
class TemporalTaskAdapter:
    """An evolver `Adapter` whose decode is the stepped recurrent substrate.

    `mode` is derived from the target: seq-to-seq ("all") when the target carries TIME, else
    seq-to-one ("last"). `n_outputs` is the PER-STEP head width; the t-major flatten of the module
    output lines up with the t-major flat target by construction.
    """

    encoded: EncodedTask
    encoder: TemporalEncoder
    n_inputs: int
    n_outputs: int
    mode: str
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH

    def decode(self, genome: Genome) -> SubstrateModule:
        return decode_recurrent(genome, self.n_inputs, self.n_outputs, self.mode, max_inline_depth=self.max_inline_depth)

    def evaluate(self, module: SubstrateModule) -> dict[str, float]:
        return evaluate(module, self.encoded, self.encoder)


def temporal_adapter(task: Task, *, max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH) -> TemporalTaskAdapter:
    """Build the stepped adapter for a TIME-bearing task (natural per-step width, no padding)."""
    support_input_field, support_output_field = support_loader(task)
    width = step_features(support_input_field)
    encoder = TemporalEncoder(step_dim=width)
    encoded = encode_temporal_task(task, encoder)

    target, _mask, descriptor = encoded.support_target
    if has_time_axis(descriptor):
        steps = encoded.support_input[0].shape[1]
        positions_per_step = target.shape[1] // steps
        return TemporalTaskAdapter(encoded, encoder, width, model_output_features(descriptor, positions_per_step), "all", max_inline_depth)
    return TemporalTaskAdapter(encoded, encoder, width, model_output_features(descriptor, target.shape[1]), "last", max_inline_depth)
