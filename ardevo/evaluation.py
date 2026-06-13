"""Evaluation: score a decoded substrate on an Icarus task.

A thin orchestration over the existing torch path in `ardevo/dataset/icarus.py`. The task is
encoded once (the trial caches the result) and every candidate is scored against the cached
tensors. Dispatch on `value_type` lives entirely in the Icarus `loss_fn`/`Level0Encoder`, so
nothing here special-cases a rung.
"""

from typing import TYPE_CHECKING, Protocol, cast

import torch

if TYPE_CHECKING:
    from ardevo.substrate import RefineGraphNet

from ardevo.dataset.icarus import (
    EncodedTask,
    FieldDescriptor,
    Level0Encoder,
    Task,
    ValueType,
    as_logits,
    encode_task,
    loss_fn,
    model_output_features,
    target_positions,
)


class DecodingEncoder(Protocol):
    """The only encoder surface evaluation needs: prediction -> labels/values for accuracy.

    `Level0Encoder` and the temporal encoder both satisfy it, so the same scoring path serves the
    flat and the stepped substrates."""

    def decode(self, prediction: torch.Tensor, descriptor: FieldDescriptor) -> torch.Tensor: ...


# Regression "correct" tolerance: an output counts as correct if it lands within this fraction of the
# target's spread. Exact float equality (the class path) is always wrong for CONTINUOUS targets, which
# would leave continuous rungs (pole/double_pole) with a flat-zero accuracy and no fitness signal.
_CONTINUOUS_TOLERANCE = 0.1


def encode(task: Task, encoder: Level0Encoder) -> EncodedTask:
    """Encode a whole task into cached, model-ready tensors (support + query)."""
    return encode_task(task, encoder)


def input_width(encoded: EncodedTask) -> int:
    """Width of the encoded input vector (the substrate's input-node count)."""
    tensor, _descriptor = encoded.support_input
    return int(tensor.shape[1])


def output_features(encoded: EncodedTask) -> int:
    """Number of output units the substrate must emit for this task's target."""
    target, _mask, descriptor = encoded.support_target
    return model_output_features(descriptor, target_positions(target))


def support_loss(module: torch.nn.Module, encoded: EncodedTask) -> torch.Tensor:
    """Differentiable loss on the support set, for the gradient train operator."""
    x, _descriptor = encoded.support_input
    target, mask, descriptor = encoded.support_target
    raw = as_logits(module(x), descriptor, target_positions(target))
    return loss_fn(raw, target, descriptor, mask)


def support_loss_deep(module: torch.nn.Module, encoded: EncodedTask) -> torch.Tensor:
    """Deep-supervised support loss for the refine substrate (TRM): a loss at EVERY refinement pass
    against the same target, weighted toward the later (more refined) passes and normalized so the
    scale matches the single-pass loss. Backprop flows through the full recursion. Requires a module
    exposing `refine_trace` (RefineGraphNet); callers fall back to `support_loss` otherwise."""
    x, _descriptor = encoded.support_input
    target, mask, descriptor = encoded.support_target
    trace = cast("RefineGraphNet", module).refine_trace(x)  # [batch, steps, n_outputs]
    steps = trace.shape[1]
    total = torch.zeros((), dtype=trace.dtype)
    weight_sum = 0.0
    positions = target_positions(target)
    for step in range(steps):
        weight = float(step + 1)  # later passes carry more weight: the answer should keep improving
        raw = as_logits(trace[:, step], descriptor, positions)
        total = total + weight * loss_fn(raw, target, descriptor, mask)
        weight_sum += weight
    return total / weight_sum


def split_metrics_from_raw(raw: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, descriptor, encoder: DecodingEncoder) -> tuple[float, float]:
    """Accuracy (via decode) and dispatched loss from an ALREADY-COMPUTED logits tensor.

    The single source of the metric math: the per-candidate path and the population/sample-batched
    paths all delegate here so their numbers can never drift apart."""
    loss = loss_fn(raw, target, descriptor, mask)
    if descriptor.value_type is ValueType.CONTINUOUS:
        # Compare in the same (normalized) space the loss uses: within-tolerance fraction, not exact match.
        spread = (target.max() - target.min()).clamp_min(1e-6)
        correct = (raw - target).abs() <= _CONTINUOUS_TOLERANCE * spread
    else:
        predictions = encoder.decode(raw, descriptor)
        correct = predictions == target.to(torch.long)
    if mask is not None:
        valid = ~mask
        accuracy = (correct & valid).sum().float() / valid.sum().clamp_min(1.0)
    else:
        accuracy = correct.float().mean()
    return float(accuracy), float(loss)


def _split_metrics(module: torch.nn.Module, encoded_input: tuple, encoded_target: tuple, encoder: DecodingEncoder) -> tuple[float, float]:
    """Accuracy (via decode) and dispatched loss for one encoded (input, target) split."""
    x, _descriptor = encoded_input
    target, mask, descriptor = encoded_target
    raw = as_logits(module(x), descriptor, target_positions(target))
    return split_metrics_from_raw(raw, target, mask, descriptor, encoder)


def evaluate(module: torch.nn.Module, encoded: EncodedTask, encoder: DecodingEncoder) -> dict[str, float]:
    """Support- and query-set metrics. Support fit rewards capacity (structure improves it even when
    the held-out query, being tiny/non-generalizable, cannot); query fit measures generalization."""
    with torch.no_grad():
        support_accuracy, support_loss_value = _split_metrics(module, encoded.support_input, encoded.support_target, encoder)
        if encoded.query_input is not None and encoded.query_target is not None:
            query_accuracy, query_loss_value = _split_metrics(module, encoded.query_input, encoded.query_target, encoder)
        else:
            query_accuracy, query_loss_value = 0.0, float("inf")
    return {
        "support_accuracy": support_accuracy,
        "support_loss": support_loss_value,
        "query_accuracy": query_accuracy,
        "query_loss": query_loss_value,
    }
