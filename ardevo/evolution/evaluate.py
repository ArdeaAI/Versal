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
    from ardevo.evaluation import DecodingEncoder
    from ardevo.evolution.evolver import Adapter
    from ardevo.substrate import GraphNet
    from ardevo.substrate_batched import BatchedGraphNet

EvaluateOp = Callable[..., dict[str, float]]

EVALUATE: Registry[EvaluateOp] = Registry("evaluate")

DEFAULT_WEIGHT_SAMPLES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


@EVALUATE.register("standard")
def standard(genome: Genome, module: SubstrateModule, adapter: "Adapter", **_params: object) -> dict[str, float]:
    return adapter.evaluate(module)


def _accuracy_key(metrics: dict[str, float]) -> str:
    # Degenerate tasks have no query split (query_loss = inf); judge those on support fit instead.
    return "query_accuracy" if math.isfinite(metrics.get("query_loss", math.inf)) else "support_accuracy"


def _stacked_split(batched: "BatchedGraphNet", columns: torch.Tensor | None, encoded_input: tuple, encoded_target: tuple, encoder: "DecodingEncoder") -> list[tuple[float, float]]:
    """One BATCHED forward scores every weight sample at once; per-slice math reuses the shared
    metric helper so numbers cannot drift from the serial path."""
    from ardevo.dataset.icarus import as_logits, target_positions
    from ardevo.evaluation import split_metrics_from_raw

    x, _descriptor = encoded_input
    target, mask, descriptor = encoded_target
    out = batched(x)  # [P, B, n_out_total]
    if columns is not None:
        out = out.index_select(2, columns)
    results: list[tuple[float, float]] = []
    for index in range(out.shape[0]):
        raw = as_logits(out[index], descriptor, target_positions(target))
        results.append(split_metrics_from_raw(raw, target, mask, descriptor, encoder))
    return results


def _stacked_sample_metrics(net: "GraphNet", columns: torch.Tensor | None, adapter: "Adapter", samples: Sequence[float]) -> list[dict[str, float]]:
    """The T2 fast path: 2 batched forwards instead of len(samples) full evaluations. The module's
    own parameters are never touched, so the save/restore dance disappears entirely."""
    from ardevo.substrate_batched import BatchedGraphNet

    population = len(samples)
    batched = BatchedGraphNet([net] * population)
    encoded = adapter.encoded
    encoder = getattr(adapter, "encoder")  # checked by the caller's gate; the Adapter protocol does not require it
    with torch.no_grad():
        fill = torch.tensor([float(value) for value in samples]).view(population, 1, 1)
        batched.weights.data = batched.mask.to(batched.weights.dtype) * fill
        support = _stacked_split(batched, columns, encoded.support_input, encoded.support_target, encoder)
        query = (
            _stacked_split(batched, columns, encoded.query_input, encoded.query_target, encoder) if encoded.query_input is not None and encoded.query_target is not None else None
        )
    per_sample: list[dict[str, float]] = []
    for index in range(population):
        support_accuracy, support_loss = support[index]
        query_accuracy, query_loss = query[index] if query is not None else (0.0, float("inf"))
        per_sample.append({"support_accuracy": support_accuracy, "support_loss": support_loss, "query_accuracy": query_accuracy, "query_loss": query_loss})
    return per_sample


def _sample_metrics(module: SubstrateModule, adapter: "Adapter", samples: Sequence[float], batched_samples: bool = False) -> tuple[dict[str, float], list[dict[str, float]], int]:
    """Evaluate with all TRAINABLE weights filled to each shared sample value; restore after.

    Frozen parameters (macro inners, library entries inside compositions) are deliberately
    excluded: robustness measures exactly the surface evolution and training control. The stacked
    fast path exists but DEFAULTS OFF: `uv run benchmark` measured it 0.2-0.4x at widths
    16-256 because the batched forward does full-width level math (D times the FLOPs of the
    serial path's per-level column slicing); enable it only if the bench shows a win for your
    population shape. Non-batchable modules (recurrent, product, composed) always use the serial
    fill/restore loop. The restore is mandatory on that path: in hybrid
    mode the module holds gradient-trained weights the trial later exports and saves.
    Returns (robustness metrics, per-sample metrics, best index).
    """
    core_net, columns = module.core()
    if batched_samples and core_net is not None and getattr(adapter, "encoder", None) is not None and getattr(adapter, "encoded", None) is not None:
        per_sample = _stacked_sample_metrics(core_net, columns, adapter, samples)
    else:
        trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
        saved = [parameter.detach().clone() for parameter in trainable]
        per_sample = []
        with torch.no_grad():
            for value in samples:
                for parameter in trainable:
                    parameter.fill_(float(value))
                per_sample.append(adapter.evaluate(module))
            for parameter, original in zip(trainable, saved):
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
    batched_samples: bool = False,
    **_params: object,
) -> dict[str, float]:
    """Pure weight-agnostic scoring: standard metric keys come from the BEST shared-weight sample."""
    robustness, per_sample, best_index = _sample_metrics(module, adapter, samples, batched_samples)
    return {**per_sample[best_index], **robustness}


@EVALUATE.register("hybrid")
def hybrid(
    genome: Genome,
    module: SubstrateModule,
    adapter: "Adapter",
    *,
    samples: Sequence[float] = DEFAULT_WEIGHT_SAMPLES,
    batched_samples: bool = False,
    **_params: object,
) -> dict[str, float]:
    """Trained-weight metrics under the standard keys, plus the robustness metrics merged in."""
    trained = adapter.evaluate(module)
    robustness, _per_sample, _best_index = _sample_metrics(module, adapter, samples, batched_samples)
    return {**robustness, **trained}
