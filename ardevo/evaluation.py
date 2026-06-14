"""Evaluation: score a decoded substrate on an Icarus task.

A thin orchestration over the existing torch path in `ardevo/dataset/icarus.py`. The task is
encoded once (the trial caches the result) and every candidate is scored against the cached
tensors. Dispatch on `value_type` lives entirely in the Icarus `loss_fn`/`Level0Encoder`, so
nothing here special-cases a rung.
"""

import hashlib
import math
import random
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, cast

import torch

if TYPE_CHECKING:
    from ardevo.substrate import EquilibriumGraphNet, RefineGraphNet

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


def support_loss_equilibrium(module: torch.nn.Module, encoded: EncodedTask) -> torch.Tensor:
    """Deep-supervised support loss for the equilibrium substrate: a loss at EVERY fixed-point
    iteration against the same target, weighted toward the later (more converged) iterations and
    normalized so the scale matches the single-pass loss. Backprop flows through the full unrolled
    iteration. Requires a module exposing `equilibrium_trace` (EquilibriumGraphNet)."""
    x, _descriptor = encoded.support_input
    target, mask, descriptor = encoded.support_target
    trace = cast("EquilibriumGraphNet", module).equilibrium_trace(x)  # [batch, iters, n_outputs]
    iters = trace.shape[1]
    total = torch.zeros((), dtype=trace.dtype)
    weight_sum = 0.0
    positions = target_positions(target)
    for index in range(iters):
        weight = float(index + 1)  # later iterations carry more weight: the answer should keep converging
        raw = as_logits(trace[:, index], descriptor, positions)
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
    # A genome whose forward blew up (e.g. an exploding refine self-loop) yields a non-finite loss; it
    # is a degenerate genome, not a crash, so report it as the WORST (zero accuracy, huge loss) and let
    # selection cull it. Without this a single NaN propagates into fitness -> speciation -> a hard stop.
    loss_value = float(loss)
    accuracy_value = float(accuracy)
    return (accuracy_value if math.isfinite(accuracy_value) else 0.0), (loss_value if math.isfinite(loss_value) else 1e9)


def _split_metrics(module: torch.nn.Module, encoded_input: tuple, encoded_target: tuple, encoder: DecodingEncoder) -> tuple[float, float]:
    """Accuracy (via decode) and dispatched loss for one encoded (input, target) split."""
    x, _descriptor = encoded_input
    target, mask, descriptor = encoded_target
    raw = as_logits(module(x), descriptor, target_positions(target))
    return split_metrics_from_raw(raw, target, mask, descriptor, encoder)


def _fold_seed(encoded: EncodedTask) -> int:
    """A deterministic seed derived ONLY from the io descriptor (value_type + shapes), never the rung or
    task name, so the inner fold is stable across runs/resume and ARC-portable. Hashed (not Python's
    salted `hash`) so it is identical across processes."""
    x, _descriptor = encoded.support_input
    target, _mask, descriptor = encoded.support_target
    key = f"{descriptor.value_type}|{tuple(int(d) for d in x.shape[1:])}|{tuple(int(d) for d in target.shape[1:])}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def support_fold(encoded: EncodedTask, fraction: float, *, holdout_min: int = 4) -> tuple[list[int], list[int]]:
    """Partition the support rows into (inner_train, inner_holdout), deterministically by io shape.

    The inner holdout is a LEAKAGE-FREE stand-in for the real query: it is carved out of the support
    set the trainer fits, never the accept/query set, so selecting on it pressures generalization
    without ever touching the metric admission scores. `fraction <= 0` or a support set too small to
    keep `holdout_min` rows on BOTH sides returns the whole support as inner-train and an empty
    holdout, so the path is byte-identical to no fold."""
    x, _descriptor = encoded.support_input
    n = int(x.shape[0])
    if fraction <= 0.0 or n < 2 * holdout_min:
        return list(range(n)), []
    holdout_count = min(max(holdout_min, int(round(n * fraction))), n - holdout_min)
    order = list(range(n))
    random.Random(_fold_seed(encoded)).shuffle(order)
    return sorted(order[holdout_count:]), sorted(order[:holdout_count])


def restrict_support(encoded: EncodedTask, rows: list[int]) -> EncodedTask:
    """A view of `encoded` whose SUPPORT split is restricted to `rows` (query untouched). Used to build
    the inner-train task the trainer fits on and the inner-holdout task evaluate scores against."""
    index = torch.tensor(rows, dtype=torch.long)
    x, x_descriptor = encoded.support_input
    target, mask, target_descriptor = encoded.support_target
    return replace(
        encoded,
        support_input=(x.index_select(0, index), x_descriptor),
        support_target=(target.index_select(0, index), mask.index_select(0, index) if mask is not None else None, target_descriptor),
    )


def behavior_descriptor(module: torch.nn.Module, encoded: EncodedTask, *, max_dim: int = 64) -> tuple[float, ...]:
    """A cheap FUNCTIONAL fingerprint of what a module computes: its raw outputs on the support input,
    flattened and deterministically subsampled to a bounded length. Two genomes computing the same
    function get nearby descriptors regardless of topology, which is exactly what novelty search needs
    to escape DECEPTIVE landscapes (structural diversity does not). One no-grad forward; the caller
    only invokes this when novelty selection is enabled."""
    with torch.no_grad():
        x, _descriptor = encoded.support_input
        out = module(x).reshape(-1)
    count = int(out.shape[0])
    if count > max_dim:
        out = out[torch.linspace(0, count - 1, steps=max_dim).round().long()]
    return tuple(float(value) for value in out.tolist())


def evaluate(module: torch.nn.Module, encoded: EncodedTask, encoder: DecodingEncoder, *, holdout: EncodedTask | None = None) -> dict[str, float]:
    """Support- and query-set metrics. Support fit rewards capacity (structure improves it even when
    the held-out query, being tiny/non-generalizable, cannot); query fit measures generalization.

    When `holdout` is given (the inner-fold view, support rows withheld from training), its support
    metrics are reported under `support_holdout_*` so the generalization fitness components get a
    leakage-free generalization signal. Absent, no holdout keys are emitted and the dict is exactly the
    pre-fold contract (byte-identical when no fold is configured)."""
    with torch.no_grad():
        support_accuracy, support_loss_value = _split_metrics(module, encoded.support_input, encoded.support_target, encoder)
        if encoded.query_input is not None and encoded.query_target is not None:
            query_accuracy, query_loss_value = _split_metrics(module, encoded.query_input, encoded.query_target, encoder)
        else:
            query_accuracy, query_loss_value = 0.0, float("inf")
        metrics = {
            "support_accuracy": support_accuracy,
            "support_loss": support_loss_value,
            "query_accuracy": query_accuracy,
            "query_loss": query_loss_value,
        }
        if holdout is not None:
            holdout_accuracy, holdout_loss_value = _split_metrics(module, holdout.support_input, holdout.support_target, encoder)
            metrics["support_holdout_accuracy"] = holdout_accuracy
            metrics["support_holdout_loss"] = holdout_loss_value
    return metrics
