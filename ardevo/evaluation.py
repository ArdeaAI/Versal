"""Evaluation: score a decoded substrate on an Icarus task.

A thin orchestration over the existing torch path in `ardevo/dataset/icarus.py`. The task is
encoded once (the trial caches the result) and every candidate is scored against the cached
tensors. Dispatch on `value_type` lives entirely in the Icarus `loss_fn`/`Level0Encoder`, so
nothing here special-cases a rung.
"""

import torch

from ardevo.dataset.icarus import (
    EncodedTask,
    Level0Encoder,
    Task,
    ValueType,
    as_logits,
    encode_task,
    loss_fn,
    model_output_features,
    target_positions,
)

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


def _split_metrics(module: torch.nn.Module, encoded_input: tuple, encoded_target: tuple, encoder: Level0Encoder) -> tuple[float, float]:
    """Accuracy (via decode) and dispatched loss for one encoded (input, target) split."""
    x, _descriptor = encoded_input
    target, mask, descriptor = encoded_target
    raw = as_logits(module(x), descriptor, target_positions(target))
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


def evaluate(module: torch.nn.Module, encoded: EncodedTask, encoder: Level0Encoder) -> dict[str, float]:
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
