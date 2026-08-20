"""Tests for self-adaptive per-genome operator rates.

Each genome carries its own operator probabilities as genes; the adaptive pipeline perturbs them
log-normally (ES perturb-and-inherit) and applies operators at the genome's own rates, so the
search rates adapt per problem instead of being hand-set. Off = the ordinary MutationPipeline,
byte-identical, empty rates never serialized.
"""

import random

from versal.evolution.genome import InnovationTracker, genome_from_dict, genome_to_dict
from versal.evolution.mutation import AdaptiveMutationPipeline, MutationContext
from versal.evolution.registry import build_evolver


def _ctx() -> MutationContext:
    return MutationContext(innovations=InnovationTracker(_next_node_id=100), activations=["tanh"], default_activation="tanh")


def _pipeline(**kwargs) -> AdaptiveMutationPipeline:
    from versal.evolution.mutation import MUTATION

    specs = [
        ("add_connection", MUTATION.get("add_connection"), {"prob": 0.2}),
        ("perturb_weights", MUTATION.get("perturb_weights"), {"prob": 0.5, "sigma": 0.4}),
    ]
    return AdaptiveMutationPipeline(specs, **kwargs)


def test_child_carries_perturbed_rates_seeded_from_base_probs(solving_genome) -> None:
    child = _pipeline()(solving_genome, _ctx(), rng=random.Random(0))
    assert set(child.operator_rates) == {"add_connection", "perturb_weights"}
    # First application seeds from the configured base probs, then perturbs, so rates drift off them.
    assert child.operator_rates["add_connection"] != 0.2 or child.operator_rates["perturb_weights"] != 0.5


def test_rates_inherit_and_drift_across_generations(solving_genome) -> None:
    pipeline = _pipeline(learning_rate=0.2)
    rng = random.Random(1)
    genome = solving_genome
    trajectory = []
    for _ in range(20):
        genome = pipeline(genome, _ctx(), rng=rng)
        trajectory.append(genome.operator_rates["add_connection"])
    assert len({round(value, 6) for value in trajectory}) > 10  # the rate is genuinely drifting, not frozen
    assert all(0.001 <= value <= 1.0 for value in trajectory)  # and stays in bounds


def test_rates_stay_within_clamp_bounds(solving_genome) -> None:
    pipeline = _pipeline(learning_rate=2.0, min_rate=0.01, max_rate=0.5)  # violent perturbation
    rng = random.Random(2)
    genome = solving_genome
    for _ in range(60):
        genome = pipeline(genome, _ctx(), rng=rng)
        for value in genome.operator_rates.values():
            assert 0.01 <= value <= 0.5


def test_rng_determinism(solving_genome) -> None:
    first = _pipeline()(solving_genome, _ctx(), rng=random.Random(7))
    second = _pipeline()(solving_genome, _ctx(), rng=random.Random(7))
    assert first.operator_rates == second.operator_rates
    assert first.connections == second.connections


def test_operator_rates_round_trip_and_absent_when_empty(solving_genome) -> None:
    assert "operator_rates" not in genome_to_dict(solving_genome)  # scalar genomes stay byte-identical
    assert genome_from_dict(genome_to_dict(solving_genome)).operator_rates == {}

    adapted = _pipeline()(solving_genome, _ctx(), rng=random.Random(0))
    payload = genome_to_dict(adapted)
    assert payload["operator_rates"] == adapted.operator_rates
    assert genome_from_dict(payload).operator_rates == adapted.operator_rates


def test_clone_and_crossover_carry_rates(solving_genome, linear_genome) -> None:
    from versal.evolution.crossover import neat

    adapted = _pipeline()(solving_genome, _ctx(), rng=random.Random(0))
    assert adapted.clone().operator_rates == adapted.operator_rates
    child = neat(adapted, linear_genome, rng=random.Random(0))
    assert child.operator_rates == adapted.operator_rates  # inherits from the fitter base (parent_a)


def test_registry_builds_adaptive_pipeline_when_configured() -> None:
    base = {
        "seed": 0,
        "substrate": {"available_activations": ["tanh"], "default_activation": "tanh"},
        "evolution": {
            "pop_size": 8,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament"},
            "crossover": {"kind": "none"},
            "mutation": {"operators": ["add_connection"], "add_connection_prob": 0.2},
            "train": {"kind": "none"},
            "speciation": {"kind": "none"},
        },
        "fitness": {"components": ["support_accuracy"]},
    }
    from versal.evolution.mutation import MutationPipeline

    assert isinstance(build_evolver(base).mutation, MutationPipeline)  # default stays the plain pipeline

    base["evolution"]["mutation"]["self_adaptive"] = True
    base["evolution"]["mutation"]["self_adaptive_learning_rate"] = 0.15
    evolver = build_evolver(base)
    assert isinstance(evolver.mutation, AdaptiveMutationPipeline)
    assert evolver.mutation.learning_rate == 0.15
