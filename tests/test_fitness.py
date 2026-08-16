"""Fitness components: the bounded loss variants keep the loss term on a comparable scale so it no
longer swamps accuracy and robustness in the weighted sum (phase-5 normalization)."""

import math
from dataclasses import replace

import pytest

from versal.evolution.fitness import (
    FITNESS,
    FitnessAggregator,
    bounded_negative_support_loss,
    connection_cost,
    negative_query_loss,
    negative_support_loss,
    support_accuracy,
    weight_robustness,
)
from versal.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind


def test_bounded_support_loss_maps_to_unit_range() -> None:
    assert bounded_negative_support_loss(None, {"support_loss": 0.0}) == 1.0
    assert bounded_negative_support_loss(None, {"support_loss": 1.0}) == 0.5
    assert 0.0 < bounded_negative_support_loss(None, {"support_loss": 100.0}) < 0.02
    # A negative/degenerate loss is clamped at 0 so the component never exceeds 1.0.
    assert bounded_negative_support_loss(None, {"support_loss": -5.0}) == 1.0


def test_bounded_loss_does_not_swamp_accuracy_and_robustness() -> None:
    """With a large loss, the RAW component dwarfs accuracy+robustness; the bounded one does not."""
    metrics = {"support_loss": 50.0, "support_accuracy": 1.0, "weight_robustness": 0.8}
    raw = FitnessAggregator([(negative_support_loss, 2.0), (support_accuracy, 1.0), (weight_robustness, 0.5)])
    bounded = FitnessAggregator([(bounded_negative_support_loss, 2.0), (support_accuracy, 1.0), (weight_robustness, 0.5)])
    # Raw: -100 + 1.0 + 0.4 = -98.6 (loss dominates entirely).
    assert raw(None, metrics) < -90
    # Bounded: ~0.039 + 1.0 + 0.4 -> accuracy and robustness are the dominant signal.
    bounded_value = bounded(None, metrics)
    assert 1.2 < bounded_value < 1.5


def test_bounded_components_are_registered() -> None:
    assert FITNESS.get("bounded_negative_support_loss") is bounded_negative_support_loss
    assert FITNESS.get("bounded_negative_query_loss")(None, {"query_loss": 3.0}) == 0.25


def test_non_finite_fitness_floors_instead_of_propagating() -> None:
    """A NaN/inf loss (an exploded recurrent unroll) must score the corpse floor, never poison
    downstream stages (speciation's share arithmetic crashed on NaN: rung 8 ecg, 2026-07-05)."""
    aggregator = FitnessAggregator([(negative_support_loss, 1.0), (support_accuracy, 1.0)])
    assert aggregator(None, {"support_loss": float("nan"), "support_accuracy": 0.5}) == -1e9
    assert aggregator(None, {"support_loss": float("inf"), "support_accuracy": 0.5}) == -1e9
    assert aggregator(None, {"support_loss": 0.5, "support_accuracy": 0.5}) > -1e9


# --- objective vectors (the nsga2 foundation) -------------------------------------------------------


def test_objectives_vector_matches_components_and_scalar_unchanged(linear_genome: Genome) -> None:
    aggregator = FitnessAggregator(
        components=[(support_accuracy, 1.0), (weight_robustness, 0.5)],
        objective_components=[("support_accuracy", support_accuracy), ("connection_cost", connection_cost)],
    )
    metrics = {"support_accuracy": 0.8, "weight_robustness": 0.4}
    assert aggregator(linear_genome, metrics) == pytest.approx(0.8 + 0.2)  # scalar path untouched
    # Raw (unweighted) component values: monotone weights cannot change Pareto dominance.
    assert aggregator.objectives(linear_genome, metrics) == [0.8, -3.0]


def test_objectives_none_when_unconfigured() -> None:
    aggregator = FitnessAggregator([(support_accuracy, 1.0)])
    assert aggregator.objective_components == []
    assert aggregator.objectives(None, {"support_accuracy": 1.0}) is None


def test_objectives_floor_non_finite_slot_only() -> None:
    aggregator = FitnessAggregator(
        components=[(support_accuracy, 1.0)],
        objective_components=[("support_accuracy", support_accuracy), ("negative_query_loss", negative_query_loss)],
    )
    vector = aggregator.objectives(None, {"support_accuracy": 0.7, "query_loss": math.inf})
    assert vector == [0.7, -1e9]  # the exploded slot floors; the healthy slot survives


# --- connection cost (the Clune/Mouret/Lipson wiring pressure) --------------------------------------


def _coordinate_genome() -> Genome:
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity", coordinate=(0.0, 0.0)),
        1: NodeGene(1, NodeKind.INPUT, "identity", coordinate=(3.0, 4.0)),
        2: NodeGene(2, NodeKind.OUTPUT, "identity", coordinate=(0.0, 1.0)),
    }
    connections = [
        ConnectionGene(0, 2, 1.0, True, 0),  # squared length (0-0)^2 + (0-1)^2 = 1
        ConnectionGene(1, 2, 1.0, True, 1),  # squared length (3-0)^2 + (4-1)^2 = 18
    ]
    return Genome(nodes=nodes, connections=connections)


def test_connection_cost_equals_known_wiring_length() -> None:
    assert connection_cost(_coordinate_genome(), {}) == pytest.approx(-19.0)


def test_connection_cost_mixed_coordinates_per_edge_fallback() -> None:
    genome = _coordinate_genome()
    genome.nodes[1] = replace(genome.nodes[1], coordinate=None)  # one endpoint loses geometry
    assert connection_cost(genome, {}) == pytest.approx(-(1.0 + 1.0))  # measured edge + unit fallback


def test_connection_cost_falls_back_to_edge_count(linear_genome: Genome) -> None:
    # The two-spirals case: no coordinates anywhere degrades to exactly -enabled_edge_count.
    assert connection_cost(linear_genome, {}) == -3.0
    disabled = replace(linear_genome.connections[0], enabled=False)
    linear_genome.connections[0] = disabled
    assert connection_cost(linear_genome, {}) == -2.0
