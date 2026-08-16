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

from versal.evolution.genome import Genome
from versal.evolution.registry import Registry
from versal.substrate import SubstrateModule

if TYPE_CHECKING:
    from versal.evaluation import DecodingEncoder
    from versal.evolution.evolver import Adapter
    from versal.substrate import GraphNet
    from versal.substrate_batched import BatchedGraphNet

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
    from versal.dataset.icarus import as_logits, target_positions
    from versal.evaluation import split_metrics_from_raw

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
    from versal.substrate_batched import BatchedGraphNet

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


# The stacked path's break-even node count on the compact-column substrate (T2, 2026-07-04:
# 0.56x/0.68x/0.83x/1.04x/1.14x at widths 16/64/256/784/3072). "auto" turns it on from here up.
STACKED_AUTO_MIN_NODES = 768


def _sample_metrics(
    module: SubstrateModule, adapter: "Adapter", samples: Sequence[float], batched_samples: bool | str = False
) -> tuple[dict[str, float], list[dict[str, float]], int]:
    """Evaluate with all TRAINABLE weights filled to each shared sample value; restore after.

    Frozen parameters (macro inners, library entries inside compositions) are deliberately
    excluded: robustness measures exactly the surface evolution and training control. The stacked
    fast path DEFAULTS OFF: on the compact-column substrate `uv run benchmark` measured it below
    break-even until ~768 nodes (0.56x-0.83x at widths 16-256) and only 1.04x-1.14x at image
    widths. `batched_samples = "auto"` enables it exactly where it measured a win (node count >=
    STACKED_AUTO_MIN_NODES); `true` forces it everywhere. Non-batchable modules (recurrent,
    product, composed) always use the serial fill/restore loop. The restore is mandatory on that
    path: in hybrid mode the module holds gradient-trained weights the trial later exports and
    saves. Returns (robustness metrics, per-sample metrics, best index).
    """
    core_net, columns = module.core()
    stacked = batched_samples is True or (batched_samples == "auto" and core_net is not None and core_net.n >= STACKED_AUTO_MIN_NODES)
    if stacked and core_net is not None and getattr(adapter, "encoder", None) is not None and getattr(adapter, "encoded", None) is not None:
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
    batched_samples: bool | str = False,
    **_params: object,
) -> dict[str, float]:
    """Pure weight-agnostic scoring: standard metric keys come from the BEST shared-weight sample."""
    robustness, per_sample, best_index = _sample_metrics(module, adapter, samples, batched_samples)
    return {**per_sample[best_index], **robustness}


def _d4_index_maps(shape: tuple[int, ...]) -> list[torch.Tensor]:
    """Flat-index permutations for the dihedral views a grid supports: 8 for a square spatial
    shape, the 4 shape-preserving ones (identity, both flips, 180 rotation) otherwise. Trailing
    axes (channels) ride along untouched. Entry j of a map is the ORIGINAL flat index whose value
    lands at position j of the transformed layout, so `x.index_select(1, map)` IS the view."""
    height, width = int(shape[0]), int(shape[1])
    trailing = 1
    for dim in shape[2:]:
        trailing *= int(dim)
    base = torch.arange(height * width * trailing).reshape(height, width, trailing)
    views = [base, torch.flip(base, dims=[0]), torch.flip(base, dims=[1]), torch.flip(base, dims=[0, 1])]
    if height == width:
        quarter = torch.rot90(base, 1, dims=(0, 1))
        views += [quarter, torch.flip(quarter, dims=[0]), torch.flip(quarter, dims=[1]), torch.flip(quarter, dims=[0, 1])]
    return [view.reshape(-1) for view in views]


def _voted_raw(module: SubstrateModule, x: torch.Tensor, input_maps: list[torch.Tensor], output_maps: "list[torch.Tensor] | None", n_outputs: int) -> torch.Tensor:
    """Mean raw output over the augmented views, accumulated back in ORIGINAL layout (grid outputs
    are inverse-permuted via index_add, a bijection), so targets and masks apply unchanged."""
    accumulated = torch.zeros(x.shape[0], n_outputs, dtype=x.dtype)
    with torch.no_grad():
        for index, input_map in enumerate(input_maps):
            out = module(x.index_select(1, input_map))
            if output_maps is None:
                accumulated += out
            else:
                accumulated.index_add_(1, output_maps[index], out)
    return accumulated / len(input_maps)


@EVALUATE.register("augmented_vote")
def augmented_vote(
    genome: Genome,
    module: SubstrateModule,
    adapter: "Adapter",
    *,
    samples: Sequence[float] = DEFAULT_WEIGHT_SAMPLES,
    batched_samples: bool | str = False,
    **_params: object,
) -> dict[str, float]:
    """Test-time augmentation voting (the TRM-on-ARC finding: a large share of abstraction-rung
    accuracy lives in augmented-view ensembling, not deeper recursion). Dispatches purely on the
    adapter's structural facts, never on rung identity: a 2-D grid input gets its dihedral views
    (D4 when square), outputs are averaged in raw space; an output whose width is a whole multiple
    of the grid's cell count is treated as a grid and votes are inverse-permuted back to the
    original layout first (the grid-to-grid case), so targets and masks apply unchanged. Non-grid
    adapters fall back to `hybrid` (this op is a drop-in for it: robustness metrics included, and
    the un-voted trained metrics stay visible under `unvoted_*`)."""
    grid = getattr(adapter, "grid_shape", None)
    encoder = getattr(adapter, "encoder", None)
    encoded = getattr(adapter, "encoded", None)
    baseline = hybrid(genome, module, adapter, samples=samples, batched_samples=batched_samples)
    if grid is None or len(grid) < 2 or encoder is None or encoded is None:
        return baseline
    input_maps = _d4_index_maps(tuple(int(dim) for dim in grid))
    if int(input_maps[0].numel()) != int(adapter.n_inputs):
        return baseline  # encoder padded or truncated; the permutation would misalign

    from versal.dataset.icarus import as_logits, target_positions
    from versal.evaluation import split_metrics_from_raw

    cells = int(grid[0]) * int(grid[1])
    output_maps = None
    if adapter.n_outputs % cells == 0:
        output_maps = _d4_index_maps((int(grid[0]), int(grid[1]), adapter.n_outputs // cells))

    def voted_pair(encoded_input: tuple, encoded_target: tuple) -> tuple[float, float]:
        x, _input_descriptor = encoded_input
        target, mask, descriptor = encoded_target
        votes = _voted_raw(module, x, input_maps, output_maps, adapter.n_outputs)
        raw = as_logits(votes, descriptor, target_positions(target))
        return split_metrics_from_raw(raw, target, mask, descriptor, encoder)

    support_accuracy, support_loss = voted_pair(encoded.support_input, encoded.support_target)
    voted = {"support_accuracy": support_accuracy, "support_loss": support_loss, "query_accuracy": 0.0, "query_loss": float("inf")}
    if encoded.query_input is not None and encoded.query_target is not None:
        query_accuracy, query_loss = voted_pair(encoded.query_input, encoded.query_target)
        voted.update({"query_accuracy": query_accuracy, "query_loss": query_loss})
    unvoted = {f"unvoted_{key}": baseline[key] for key in ("support_accuracy", "support_loss", "query_accuracy", "query_loss") if key in baseline}
    return {**baseline, **unvoted, **voted, "vote_views": float(len(input_maps))}


@EVALUATE.register("hybrid")
def hybrid(
    genome: Genome,
    module: SubstrateModule,
    adapter: "Adapter",
    *,
    samples: Sequence[float] = DEFAULT_WEIGHT_SAMPLES,
    batched_samples: bool | str = False,
    **_params: object,
) -> dict[str, float]:
    """Trained-weight metrics under the standard keys, plus the robustness metrics merged in."""
    trained = adapter.evaluate(module)
    robustness, _per_sample, _best_index = _sample_metrics(module, adapter, samples, batched_samples)
    return {**robustness, **trained}
