"""Fitness components and their weighted aggregator. Convention: higher fitness is better.

Each component is a registered `(genome, metrics) -> float`. `FitnessAggregator` sums
`w_<name> * component(...)` using the weights from `[fitness]`, so the objective is reshaped
purely from config.
"""

from dataclasses import dataclass
from typing import Any, Callable

from ardevo.evolution.genome import Genome
from ardevo.evolution.registry import Registry

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


# --- weight-robustness components (metrics produced by the weight_samples / hybrid evaluate ops) ---
# All default to 0.0 when the metric is absent so a misconfigured combo degrades instead of crashing.


@FITNESS.register("mean_sample_accuracy")
def mean_sample_accuracy(genome: Genome, metrics: dict[str, float]) -> float:
    return float(metrics.get("mean_sample_accuracy", 0.0))


@FITNESS.register("max_sample_accuracy")
def max_sample_accuracy(genome: Genome, metrics: dict[str, float]) -> float:
    return float(metrics.get("max_sample_accuracy", 0.0))


@FITNESS.register("weight_robustness")
def weight_robustness(genome: Genome, metrics: dict[str, float]) -> float:
    # mean minus std over the shared-weight samples: rewards topologies whose function survives
    # weight perturbation, the signal that predicts a module will compose and transfer well.
    return float(metrics.get("weight_robustness", 0.0))


@FITNESS.register("negative_mean_sample_loss")
def negative_mean_sample_loss(genome: Genome, metrics: dict[str, float]) -> float:
    return -float(metrics.get("mean_sample_loss", 0.0))


@dataclass
class FitnessAggregator:
    """Weighted sum of fitness components."""

    components: list[tuple[FitnessComponent, float]]

    def __call__(self, genome: Any, metrics: dict[str, float]) -> float:
        return sum(weight * component(genome, metrics) for component, weight in self.components)
