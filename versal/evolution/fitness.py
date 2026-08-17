"""Fitness components and their weighted aggregator. Convention: higher fitness is better.

Each component is a registered `(genome, metrics) -> float`. `FitnessAggregator` sums
`w_<name> * component(...)` using the weights from `[fitness]`, so the objective is reshaped
purely from config.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from versal.evolution.genome import Genome
from versal.evolution.registry import Registry

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
    value = metrics.get("expanded_complexity")
    return -float(genome.complexity() if value is None else value)


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


@FITNESS.register("support_task_exact")
def support_task_exact(genome: Genome, metrics: dict[str, float]) -> float:
    """Whole-task exactness where a structured evaluator provides it, dense accuracy otherwise."""

    return float(metrics.get("support_task_exact", metrics.get("support_accuracy", 0.0)))


@FITNESS.register("query_task_exact")
def query_task_exact(genome: Genome, metrics: dict[str, float]) -> float:
    return float(metrics.get("query_task_exact", metrics.get("query_accuracy", 0.0)))


@FITNESS.register("support_gain_over_baseline")
def support_gain_over_baseline(genome: Genome, metrics: dict[str, float]) -> float:
    """Reward signal above constant/copy baselines; ordinary tasks retain their dense signal."""

    return float(metrics.get("support_gain_over_baseline", metrics.get("support_accuracy", 0.0)))


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


# weight-robustness components (metrics produced by the weight_samples / hybrid evaluate ops)
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


@FITNESS.register("novelty")
def novelty(genome: Genome, metrics: dict[str, float]) -> float:
    # Population-relative k-NN behavioral novelty, injected by the Evolver's post-assess hook
    # ([evolution.novelty]). Absent key degrades to 0.0 like every evaluate-derived signal; the
    # score never reaches library admission (verification re-scores through evaluate_only).
    return float(metrics.get("novelty", 0.0))


@FITNESS.register("connection_cost")
def connection_cost(genome: Genome, metrics: dict[str, float]) -> float:
    # Squared wiring length (Clune/Mouret/Lipson): the cost under which modularity, then hierarchy,
    # EMERGE rather than being imposed. An edge is measurable only when both endpoint coordinates
    # exist and share a length (the coordinate_distance incomparability rule); unmeasurable edges
    # cost 1.0, so a coordinate-free genome degrades to exactly -edge_count and mixed genomes stay
    # smooth as geometry appears. getattr keeps CompositionGenome (no coordinate field) safe.
    total = 0.0
    for connection in genome.enabled_connections():
        source = getattr(genome.nodes[connection.in_id], "coordinate", None)
        target = getattr(genome.nodes[connection.out_id], "coordinate", None)
        if source is None or target is None or len(source) != len(target):
            total += 1.0
        else:
            total += sum((a - b) ** 2 for a, b in zip(source, target))
    # Geometry prices visible wiring. Referenced inner structure has no coordinates in this shell,
    # so add its recursively expanded cost beyond the already-counted local topology.
    expanded_value = metrics.get("expanded_complexity")
    shell_value = metrics.get("shell_complexity")
    shell_complexity = genome.complexity() if expanded_value is None or shell_value is None else 0
    expanded = float(shell_complexity if expanded_value is None else expanded_value)
    shell = float(shell_complexity if shell_value is None else shell_value)
    return -(total + max(0.0, expanded - shell))


def stamp_complexity_metrics(genome: Any, metrics: dict[str, float], library: Any | None) -> None:
    """Attach local and recursively expanded topology costs before fitness/Pareto scoring."""

    shell = int(genome.complexity())
    expanded = shell
    if library is not None:
        from versal.library import COMPOSITION, MODULE, expanded_payload_complexity

        if isinstance(genome, Genome):
            from versal.evolution.genome import genome_to_dict

            expanded = expanded_payload_complexity(MODULE, genome_to_dict(genome), library)
        else:
            from versal.evolution.composition import comp_to_dict

            expanded = expanded_payload_complexity(COMPOSITION, comp_to_dict(genome), library)
    metrics["shell_complexity"] = float(shell)
    metrics["expanded_complexity"] = float(expanded)


@dataclass
class FitnessAggregator:
    """Weighted sum of fitness components."""

    components: list[tuple[FitnessComponent, float]]
    # Named subset forming the Pareto objective vector (`[fitness] objectives`); empty = scalar-only
    # run, byte-identical to the pre-vector behavior. Kept separate from `components` so the scalar
    # sum (speciation budgets, champion tracking) and the selection geometry are tuned independently.
    objective_components: list[tuple[str, FitnessComponent]] = field(default_factory=list)

    def objectives(self, genome: Any, metrics: dict[str, float]) -> list[float] | None:
        """Raw (unweighted) objective values, maximization sense; None when unconfigured.

        Unweighted because a positive monotone rescale can never change Pareto dominance and
        crowding normalizes per objective, so a weight here would be a knob that cannot matter.
        A non-finite slot floors to -1e9 (per slot, mirroring __call__'s corpse floor) so one
        exploded objective buries the candidate on that axis without poisoning the others."""
        if not self.objective_components:
            return None
        values: list[float] = []
        for _name, component in self.objective_components:
            value = component(genome, metrics)
            values.append(value if math.isfinite(value) else -1e9)
        return values

    def __call__(self, genome: Any, metrics: dict[str, float]) -> float:
        total = sum(weight * component(genome, metrics) for component, weight in self.components)
        # Non-finite phenotypes must not enter speciation's share arithmetic.
        return total if math.isfinite(total) else -1e9
