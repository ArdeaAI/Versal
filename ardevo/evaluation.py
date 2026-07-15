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


class EncodedSupport(Protocol):
    """Support tensors shared by ordinary and structured task encodings."""

    @property
    def support_input(self) -> tuple[torch.Tensor, FieldDescriptor]: ...

    @property
    def support_target(self) -> tuple[torch.Tensor, torch.Tensor | None, FieldDescriptor]: ...


# Regression "correct" tolerance: an output counts as correct if it lands within this fraction of the
# target's spread. Exact float equality (the class path) is always wrong for CONTINUOUS targets, which
# would leave continuous rungs (pole/double_pole) with a flat-zero accuracy and no fitness signal.
_CONTINUOUS_TOLERANCE = 0.1


def encode(task: Task, encoder: Level0Encoder) -> EncodedTask:
    """Encode a whole task into cached, model-ready tensors (support + query)."""
    return fit_query_target(encode_task(task, encoder))


def fit_query_target(encoded: EncodedTask) -> EncodedTask:
    """Fit the encoded query target (and its mask) to the support target's position count.

    Tasks with per-example natural sizes (the psicov class: per-protein LxL distance maps) encode
    support and query targets at different widths, but every model head in the system is sized by
    the SUPPORT target, so an unfitted query target crashes the first evaluation (2026-07-06 smoke
    run, rung 14: mse over 245,025 predictions vs a 167,281 target). Padding enters under a True
    mask, so padded positions carry zero loss and zero accuracy weight; a wider query target is
    cropped, scoring the overlap. Same-width tasks return the same object, byte-identical. The
    encoder cannot own this fit because icarus.py is vendored, so it lives at the consumer layer."""
    if encoded.query_target is None:
        return encoded
    target, mask, descriptor = encoded.query_target
    width = encoded.support_target[0].shape[1]
    if target.shape[1] == width:
        return encoded
    if mask is None:
        mask = torch.zeros(target.shape, dtype=torch.bool)
    if target.shape[1] > width:
        target, mask = target[:, :width], mask[:, :width]
    else:
        pad = width - target.shape[1]
        target = torch.cat([target, torch.zeros(target.shape[0], pad, dtype=target.dtype)], dim=1)
        mask = torch.cat([mask, torch.ones(target.shape[0], pad, dtype=torch.bool)], dim=1)
    return EncodedTask(
        support_input=encoded.support_input,
        support_target=encoded.support_target,
        query_input=encoded.query_input,
        query_target=(target, mask, descriptor),
    )


def without_query(encoded: EncodedTask) -> EncodedTask:
    """Return the search-time view of a task, with held-out query tensors inaccessible.

    Callers opt into this view through orchestrator configuration.  Keeping it as a distinct
    object makes accidental query evaluation impossible rather than merely promising not to read
    the resulting metric.
    """

    return EncodedTask(
        support_input=encoded.support_input,
        support_target=encoded.support_target,
        query_input=None,
        query_target=None,
    )


def input_width(encoded: EncodedSupport) -> int:
    """Width of the encoded input vector (the substrate's input-node count)."""
    tensor, _descriptor = encoded.support_input
    return int(tensor.shape[1])


def output_features(encoded: EncodedSupport) -> int:
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
    from ardevo.structured import StructuredGridEncoded, evaluate_structured_grid

    if isinstance(encoded, StructuredGridEncoded):
        return evaluate_structured_grid(module, encoded, encoder)
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
