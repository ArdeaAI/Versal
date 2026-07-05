"""Behavioral novelty: k-NN distance in the network's OUTPUT space over a fixed probe set.

The divergent-selection lever (Mouret/Doncieux lineage) for deceptive landscapes: objective-greedy
selection converges on the mediocre middle (the two-spirals 0.672 plateau), and rewarding
behavioral distance keeps structurally different candidates alive long enough to assemble the
coordinated motif a full solution needs. The descriptor is the network's own function over
task-derived probe inputs (a deterministic stride of the encoded support rows), so no human probe
geometry enters. Everything here is rng-free: the off switch is byte-identical and the on switch
perturbs no mutation/selection stream. Scope: the direct path's Evolver hook only; the archive
lives on `EvolverState`, which is born fresh per task solve and never serialized (task attempts
are checkpoint-atomic), so per-task novelty needs no resume machinery.
"""

import math
from dataclasses import dataclass

import torch

from ardevo.dataset.icarus import EncodedTask

Descriptor = tuple[float, ...]


@dataclass(frozen=True)
class NoveltyConfig:
    """`[evolution.novelty]`: absent table (or enabled = false) builds no config at all."""

    k: int = 15
    archive_cap: int = 256  # 0 = archive-free: novelty against the current population only
    probe_rows: int = 64


def probe_indices(n_rows: int, probe_rows: int) -> list[int]:
    """Deterministic stride subsample: unique, sorted, rng-free (a draw here would perturb the
    shared mutation stream), and stable across generations so descriptors stay comparable."""
    m = min(probe_rows, n_rows)
    return [(index * n_rows) // m for index in range(m)]


def probe_tensor(encoded: EncodedTask, probe_rows: int) -> torch.Tensor | None:
    """The probe inputs, or None when the task shape is out of scope (TIME-axis adapters)."""
    tensor, _descriptor = encoded.support_input
    if tensor.dim() != 2 or tensor.shape[0] == 0:
        return None
    return tensor.index_select(0, torch.tensor(probe_indices(int(tensor.shape[0]), probe_rows), dtype=torch.long))


def compute_descriptor(module: torch.nn.Module, probe: torch.Tensor) -> Descriptor | None:
    """tanh of the raw outputs over the probe set: bounded (so distances normalize), smooth, and
    confidence-preserving where argmax labels would collapse the early near-0.5 population into one
    behavior exactly when the deceptive plateau needs the most behavioral resolution."""
    with torch.no_grad():
        try:
            output = module(probe)
        except (RuntimeError, ValueError, KeyError, IndexError):
            return None  # a candidate whose forward cannot run the probe simply goes unscored
    if output.dim() != 2 or output.shape[0] != probe.shape[0] or torch.isnan(output).any():
        # tanh saturates inf to +-1 (a valid behavior) but passes NaN through; one NaN row would
        # poison every cdist column and could ride the argmax into the archive (NaN wins max()
        # from slot 0), so a NaN forward goes unscored instead.
        return None
    return tuple(torch.tanh(output).flatten().tolist())


def novelty_scores(descriptors: list[Descriptor], archive: list[Descriptor], k: int) -> list[float]:
    """Mean L2 distance to the k nearest neighbors across population + archive, normalized by the
    tanh descriptor-space diameter (2 * sqrt(D)) so scores land in [0, 1] on the same comparable
    scale as the other fitness components."""
    population = torch.tensor(descriptors, dtype=torch.float32)
    union = torch.tensor(descriptors + archive, dtype=torch.float32)
    neighbors = min(k, int(union.shape[0]) - 1)
    if neighbors <= 0:
        return [0.0] * len(descriptors)
    distances = torch.cdist(population, union)
    for index in range(len(descriptors)):
        distances[index, index] = math.inf  # each member sits at its own column in the union
    nearest, _indices = torch.topk(distances, neighbors, dim=1, largest=False)
    diameter = 2.0 * math.sqrt(population.shape[1])
    return [float(value) / diameter for value in nearest.mean(dim=1)]


def archive_insert(archive: list[Descriptor], descriptor: Descriptor, cap: int) -> None:
    """FIFO append with eviction: the simplest deterministic policy (probabilistic insertion draws
    rng; add-if-novel thresholds add a fragile knob). cap = 0 keeps the archive empty, which is the
    stateless multi-objectivization ablation."""
    if cap <= 0:
        return
    archive.append(descriptor)
    while len(archive) > cap:
        archive.pop(0)
