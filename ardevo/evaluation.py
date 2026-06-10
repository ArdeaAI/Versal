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
    as_logits,
    encode_task,
    loss_fn,
    model_output_features,
    target_positions,
)


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


def evaluate(module: torch.nn.Module, encoded: EncodedTask, encoder: Level0Encoder) -> dict[str, float]:
    """Query-set metrics: accuracy via decode() and the dispatched loss."""
    if encoded.query_input is None or encoded.query_target is None:
        return {"query_accuracy": 0.0, "query_loss": float("inf")}

    x, _descriptor = encoded.query_input
    target, mask, descriptor = encoded.query_target
    with torch.no_grad():
        raw = as_logits(module(x), descriptor, target_positions(target))
        loss = loss_fn(raw, target, descriptor, mask)
        predictions = encoder.decode(raw, descriptor)
        correct = predictions == target.to(torch.long)
        if mask is not None:
            valid = ~mask
            accuracy = (correct & valid).sum().float() / valid.sum().clamp_min(1.0)
        else:
            accuracy = correct.float().mean()
    return {"query_accuracy": float(accuracy), "query_loss": float(loss)}
