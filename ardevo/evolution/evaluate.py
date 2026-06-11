"""Evaluate operators: an independent stage that turns a scored candidate into metrics.

`standard` delegates to the task adapter (the original behavior, and the default). `weight_samples`
scores the topology with every enabled weight set to each value of a shared sample set: a topology
that performs across weight settings encodes its function in STRUCTURE, which makes it robust and,
critically for the library, composable. `hybrid` reports trained-weight metrics plus the sampling
metrics, so gradient-trained runs still carry the robustness signal for library admission.
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

import torch

from ardevo.evolution.genome import Genome
from ardevo.evolution.registry import Registry
from ardevo.substrate import SubstrateModule

if TYPE_CHECKING:
    from ardevo.evolution.evolver import Adapter

EvaluateOp = Callable[..., dict[str, float]]

EVALUATE: Registry[EvaluateOp] = Registry("evaluate")

DEFAULT_WEIGHT_SAMPLES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


@EVALUATE.register("standard")
def standard(genome: Genome, module: SubstrateModule, adapter: "Adapter", **_params: object) -> dict[str, float]:
    return adapter.evaluate(module)


def _accuracy_key(metrics: dict[str, float]) -> str:
    # Degenerate tasks have no query split (query_loss = inf); judge those on support fit instead.
    return "query_accuracy" if math.isfinite(metrics.get("query_loss", math.inf)) else "support_accuracy"


def _sample_metrics(module: SubstrateModule, adapter: "Adapter", samples: Sequence[float]) -> tuple[dict[str, float], list[dict[str, float]], int]:
    """Evaluate with all weights filled to each shared sample value; restore the weights after.

    The restore is mandatory: in hybrid mode the module holds gradient-trained weights that the
    trial later exports and saves. Returns (robustness metrics, per-sample metrics, best index).
    """
    saved = [parameter.detach().clone() for parameter in module.parameters()]
    per_sample: list[dict[str, float]] = []
    with torch.no_grad():
        for value in samples:
            for parameter in module.parameters():
                parameter.fill_(float(value))
            per_sample.append(adapter.evaluate(module))
        for parameter, original in zip(module.parameters(), saved):
            parameter.copy_(original)

    key = _accuracy_key(per_sample[0])
    accuracies = [metrics[key] for metrics in per_sample]
    mean = sum(accuracies) / len(accuracies)
    variance = sum((value - mean) ** 2 for value in accuracies) / len(accuracies)
    best_index = max(
        range(len(per_sample)),
        key=lambda i: (per_sample[i]["query_accuracy"], per_sample[i]["support_accuracy"], -per_sample[i]["support_loss"]),
    )
    robustness = {
        "mean_sample_accuracy": mean,
        "max_sample_accuracy": max(accuracies),
        "min_sample_accuracy": min(accuracies),
        "mean_sample_loss": sum(metrics["support_loss"] for metrics in per_sample) / len(per_sample),
        "best_sample_weight": float(samples[best_index]),
        "weight_robustness": mean - math.sqrt(variance),
    }
    return robustness, per_sample, best_index


@EVALUATE.register("weight_samples")
def weight_samples(
    genome: Genome,
    module: SubstrateModule,
    adapter: "Adapter",
    *,
    samples: Sequence[float] = DEFAULT_WEIGHT_SAMPLES,
    **_params: object,
) -> dict[str, float]:
    """Pure weight-agnostic scoring: standard metric keys come from the BEST shared-weight sample."""
    robustness, per_sample, best_index = _sample_metrics(module, adapter, samples)
    return {**per_sample[best_index], **robustness}


@EVALUATE.register("hybrid")
def hybrid(
    genome: Genome,
    module: SubstrateModule,
    adapter: "Adapter",
    *,
    samples: Sequence[float] = DEFAULT_WEIGHT_SAMPLES,
    **_params: object,
) -> dict[str, float]:
    """Trained-weight metrics under the standard keys, plus the robustness metrics merged in."""
    trained = adapter.evaluate(module)
    robustness, _per_sample, _best_index = _sample_metrics(module, adapter, samples)
    return {**robustness, **trained}
