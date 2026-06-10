"""Fitness components and their weighted aggregator. Convention: higher fitness is better.

Each component is a registered `(genome, metrics) -> float`. `FitnessAggregator` sums
`w_<name> * component(...)` using the weights from `[fitness]`, so the objective is reshaped
purely from config.
"""

from dataclasses import dataclass
from typing import Callable

from ardevo.evolution.genome import Genome
from ardevo.evolution.registry import Registry

FitnessComponent = Callable[[Genome, dict[str, float]], float]

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


@dataclass
class FitnessAggregator:
    """Weighted sum of fitness components."""

    components: list[tuple[FitnessComponent, float]]

    def __call__(self, genome: Genome, metrics: dict[str, float]) -> float:
        return sum(weight * component(genome, metrics) for component, weight in self.components)
