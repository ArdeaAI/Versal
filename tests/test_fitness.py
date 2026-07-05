"""Fitness components: the bounded loss variants keep the loss term on a comparable scale so it no
longer swamps accuracy and robustness in the weighted sum (phase-5 normalization)."""

from ardevo.evolution.fitness import (
    FITNESS,
    FitnessAggregator,
    bounded_negative_support_loss,
    negative_support_loss,
    support_accuracy,
    weight_robustness,
)


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
