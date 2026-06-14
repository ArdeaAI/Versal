"""Fitness components and their weighted aggregator. Convention: higher fitness is better.

Each component is a registered `(genome, metrics) -> float`. `FitnessAggregator` sums
`w_<name> * component(...)` using the weights from `[fitness]`, so the objective is reshaped
purely from config.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable

from ardevo.evolution.genome import Genome, coordinate_distance
from ardevo.evolution.registry import Registry

# A degenerate genome (non-finite metric, e.g. an exploding refine recursion) gets this floor instead
# of NaN/inf, so it is the worst in selection without poisoning speciation's offspring-share math.
_FITNESS_FLOOR = -1.0e9

# Components receive either a flat `Genome` or a `CompositionGenome`; both expose the structural
# surface the components read (`hidden_ids`, `complexity()`), so the alias stays duck-typed.
FitnessComponent = Callable[[Any, dict[str, float]], float]

FITNESS: Registry[FitnessComponent] = Registry("fitness")


@FITNESS.register("query_accuracy")
def query_accuracy(genome: Genome, metrics: dict[str, float]) -> float:
    return float(metrics.get("query_accuracy", 0.0))


@FITNESS.register("complexity_penalty")
def complexity_penalty(genome: Genome, metrics: dict[str, float]) -> float:
    # Negative so a larger graph lowers fitness; the weight scales the pressure.
    return -float(genome.complexity())


@FITNESS.register("hidden_penalty")
def hidden_penalty(genome: Genome, metrics: dict[str, float]) -> float:
    # Penalize only the HIDDEN-node count, not the unavoidable dense input->output readout. With
    # high-dimensional I/O (e.g. 32->19) the readout alone is hundreds of edges, so a total-edge
    # `complexity_penalty` swamps the fitting signal and punishes any growth; this bounds the shared
    # body instead. This is the complexity bound to use on a continuous / multi-rung run.
    return -float(len(genome.hidden_ids))


@FITNESS.register("negative_query_loss")
def negative_query_loss(genome: Genome, metrics: dict[str, float]) -> float:
    return -float(metrics.get("query_loss", 0.0))


@FITNESS.register("support_accuracy")
def support_accuracy(genome: Genome, metrics: dict[str, float]) -> float:
    # Rewards fitting the training set, which structure (hidden nodes) can improve even when the
    # held-out query is too small to generalize to. Use this to drive topology growth.
    return float(metrics.get("support_accuracy", 0.0))


@FITNESS.register("negative_support_loss")
def negative_support_loss(genome: Genome, metrics: dict[str, float]) -> float:
    return -float(metrics.get("support_loss", 0.0))


# Bounded loss components: map an unbounded loss in [0, inf) to (0, 1] via 1/(1+loss) (1 at zero
# loss, decaying toward 0). Raw `negative_support_loss` in (-inf, 0] otherwise DOMINATES the weighted
# sum and drowns out support_accuracy [0,1] and weight_robustness [-1,1], so selection optimizes
# brittle low-loss modules that then fail the robustness gate. Use these on the orchestrated/library
# path where module robustness and transfer matter; the raw variants stay for the tuned flat configs.


@FITNESS.register("bounded_negative_support_loss")
def bounded_negative_support_loss(genome: Genome, metrics: dict[str, float]) -> float:
    return 1.0 / (1.0 + max(float(metrics.get("support_loss", 0.0)), 0.0))


@FITNESS.register("bounded_negative_query_loss")
def bounded_negative_query_loss(genome: Genome, metrics: dict[str, float]) -> float:
    return 1.0 / (1.0 + max(float(metrics.get("query_loss", 0.0)), 0.0))


# --- generalization components (metrics produced when an inner support fold is active) ---------------
# The orchestrated/library path may NOT select on the real query_accuracy: it is the accept metric and
# the library admission currency, so optimizing it directly is leakage (the search would memorize its
# own held-out test). These components read an INNER fold carved out of TRAINING instead, the leakage-
# free analogue of the proven standalone recipe's w_query_accuracy. All fall back to the support metric
# when no fold is configured, so they are harmless (contribute 0 gap, support-equal loss) at fraction 0.


@FITNESS.register("holdout_accuracy")
def holdout_accuracy(genome: Genome, metrics: dict[str, float]) -> float:
    # Accuracy on the inner fold withheld from training: rewards GENERALIZATION, not memorization.
    return float(metrics.get("support_holdout_accuracy", metrics.get("support_accuracy", 0.0)))


@FITNESS.register("generalization_gap")
def generalization_gap(genome: Genome, metrics: dict[str, float]) -> float:
    # Negative train-minus-holdout gap: a memorizer (high train, low holdout) is penalized; a genome
    # that generalizes (train ~ holdout) is not. Exactly 0 when no fold (holdout defaults to train).
    train = float(metrics.get("support_accuracy", 0.0))
    held = float(metrics.get("support_holdout_accuracy", train))
    return -(train - held)


@FITNESS.register("bounded_negative_holdout_loss")
def bounded_negative_holdout_loss(genome: Genome, metrics: dict[str, float]) -> float:
    loss = float(metrics.get("support_holdout_loss", metrics.get("support_loss", 0.0)))
    return 1.0 / (1.0 + max(loss, 0.0))


# --- weight-robustness components (metrics produced by the weight_samples / hybrid evaluate ops) ---
# All default to 0.0 when the metric is absent so a misconfigured combo degrades instead of crashing.


@FITNESS.register("mean_sample_accuracy")
def mean_sample_accuracy(genome: Genome, metrics: dict[str, float]) -> float:
    return float(metrics.get("mean_sample_accuracy", 0.0))


@FITNESS.register("max_sample_accuracy")
def max_sample_accuracy(genome: Genome, metrics: dict[str, float]) -> float:
    return float(metrics.get("max_sample_accuracy", 0.0))


@FITNESS.register("depth_scaling_score")
def depth_scaling_score(genome: Genome, metrics: dict[str, float]) -> float:
    # Phase 7 (Pillar D): rewards a genome whose accuracy IMPROVES with more recursion iterations (the
    # depth_scaled evaluate stamps it). Makes effective depth an admittable currency so recursion buys
    # reasoning depth instead of refine_steps drifting back to 1. Absent (no depth) -> 0.0, inert.
    return float(metrics.get("depth_scaling_score", 0.0))


@FITNESS.register("weight_robustness")
def weight_robustness(genome: Genome, metrics: dict[str, float]) -> float:
    # mean minus std over the shared-weight samples: rewards topologies whose function survives
    # weight perturbation, the signal that predicts a module will compose and transfer well.
    return float(metrics.get("weight_robustness", 0.0))


@FITNESS.register("negative_mean_sample_loss")
def negative_mean_sample_loss(genome: Genome, metrics: dict[str, float]) -> float:
    return -float(metrics.get("mean_sample_loss", 0.0))


# --- modularity (phase 7, Pillar C): a gradient toward LOCAL/tiled structure on grid tasks ----------
# Computed straight from the genome's coordinate geometry (no evaluate metric needed): the fraction of
# coordinate-comparable edges that are LOCAL (within a small radius). On grid rungs this rewards the
# tiled receptive fields a convolution needs, giving evolution a pull toward weight-tied kernels rather
# than a dense readout. No-op (0.0) on the flat/non-grid path: uncoordinated edges are skipped, so a
# genome with no geometry scores 0 and the term is inert until grid coordinates are stamped.
_MODULARITY_RADIUS = 2.0


@FITNESS.register("modularity_bonus")
def modularity_bonus(genome: Genome, metrics: dict[str, float]) -> float:
    nodes = getattr(genome, "nodes", None)
    if nodes is None or not hasattr(genome, "enabled_connections"):
        return 0.0  # a composition genome carries no node geometry; the bonus only applies to flat grids
    comparable = 0
    local = 0
    for conn in genome.enabled_connections():
        if conn.in_id not in nodes or conn.out_id not in nodes:
            continue
        distance = coordinate_distance(nodes[conn.in_id].coordinate, nodes[conn.out_id].coordinate)
        if math.isinf(distance):
            continue  # uncoordinated or incomparable banks: not a geometry edge
        comparable += 1
        if distance <= _MODULARITY_RADIUS:
            local += 1
    return local / comparable if comparable else 0.0


@dataclass
class FitnessAggregator:
    """Weighted sum of fitness components."""

    components: list[tuple[FitnessComponent, float]]

    def __call__(self, genome: Any, metrics: dict[str, float]) -> float:
        total = sum(weight * component(genome, metrics) for component, weight in self.components)
        # Never let a non-finite component (a degenerate genome) reach selection: floor it to the worst.
        return total if math.isfinite(total) else _FITNESS_FLOOR
