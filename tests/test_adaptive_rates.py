"""Self-adaptive operator rates (the ACI "mutation as a meta-parameter"): each genome carries its own
per-operator probabilities, perturbed log-normally and inherited (Evolution-Strategy self-adaptation),
so the search tunes its own operator mix and selection on the genome selects good rates for free."""

import random
from typing import Any

from ardevo.evolution.genome import Genome, InnovationTracker, NodeGene, NodeKind, genome_from_dict, genome_to_dict
from ardevo.evolution.mutation import MUTATION, AdaptiveRates, MutationContext, MutationPipeline
from ardevo.evolution.registry import build_evolver


def _ctx() -> MutationContext:
    return MutationContext(innovations=InnovationTracker(_next_node_id=50), activations=["tanh", "identity"], default_activation="tanh")


def _genome() -> Genome:
    nodes = {0: NodeGene(0, NodeKind.INPUT, "identity"), 1: NodeGene(1, NodeKind.OUTPUT, "identity"), 2: NodeGene(2, NodeKind.HIDDEN, "tanh")}
    from ardevo.evolution.genome import ConnectionGene

    connections = [ConnectionGene(0, 2, 0.5, True, 0), ConnectionGene(2, 1, 0.5, True, 1)]
    return Genome(nodes=nodes, connections=connections)


def _named(names: list[str]) -> list[tuple[str, Any]]:
    from functools import partial

    return [(name, partial(MUTATION.get(name))) for name in names]


# --- gene round-trip ----------------------------------------------------------------------------


def test_operator_rates_round_trip_and_legacy_default() -> None:
    genome = _genome()
    genome.operator_rates = {"add_connection": 0.2, "add_rich_node": 0.05}
    restored = genome_from_dict(genome_to_dict(genome))
    assert restored.operator_rates == {"add_connection": 0.2, "add_rich_node": 0.05}
    legacy = genome_to_dict(genome)
    del legacy["operator_rates"]  # a pre-phase-6 genome dict
    assert genome_from_dict(legacy).operator_rates is None


def test_clone_copies_operator_rates_independently() -> None:
    genome = _genome()
    genome.operator_rates = {"add_connection": 0.2}
    clone = genome.clone()
    assert clone.operator_rates is not None and genome.operator_rates is not None
    clone.operator_rates["add_connection"] = 0.9
    assert genome.operator_rates["add_connection"] == 0.2  # parent unaffected (deep copy of the dict)


# --- pipeline behavior --------------------------------------------------------------------------


def test_disabled_pipeline_never_stamps_rates() -> None:
    pipeline = MutationPipeline(_named(["add_connection", "add_rich_node"]), base_rates={"add_connection": 0.3, "add_rich_node": 0.3}, adaptive=AdaptiveRates(enabled=False))
    child = pipeline(_genome(), _ctx(), rng=random.Random(0))
    assert child.operator_rates is None  # off: the genome stays free of strategy parameters


def test_enabled_pipeline_stamps_and_inherits_rates() -> None:
    base = {"add_connection": 0.3, "add_rich_node": 0.2}
    pipeline = MutationPipeline(_named(["add_connection", "add_rich_node"]), base_rates=base, adaptive=AdaptiveRates(enabled=True, sigma=0.1))
    child = pipeline(_genome(), _ctx(), rng=random.Random(0))
    assert child.operator_rates is not None and set(child.operator_rates) == set(base)
    grandchild = pipeline(child, _ctx(), rng=random.Random(1))
    assert grandchild.operator_rates is not None and set(grandchild.operator_rates) == set(base)  # rates carry forward


def test_rates_stay_within_bounds_over_many_generations() -> None:
    base = {"add_connection": 0.3}
    pipeline = MutationPipeline(_named(["add_connection"]), base_rates=base, adaptive=AdaptiveRates(enabled=True, sigma=0.5, min_rate=0.02, max_rate=0.4))
    genome = _genome()
    for seed in range(200):
        genome = pipeline(genome, _ctx(), rng=random.Random(seed))
        assert genome.operator_rates is not None
        assert 0.02 <= genome.operator_rates["add_connection"] <= 0.4


def test_rates_actually_drift_under_perturbation() -> None:
    base = {"add_connection": 0.3}
    pipeline = MutationPipeline(_named(["add_connection"]), base_rates=base, adaptive=AdaptiveRates(enabled=True, sigma=0.3, min_rate=0.001, max_rate=0.9))
    values = set()
    genome = _genome()
    for seed in range(10):
        genome = pipeline(genome, _ctx(), rng=random.Random(seed))
        assert genome.operator_rates is not None
        values.add(round(genome.operator_rates["add_connection"], 6))
    assert len(values) > 1  # the rate is not frozen at its base value


# --- evolver integration ------------------------------------------------------------------------


def _config(adaptive: bool) -> dict[str, Any]:
    return {
        "seed": 0,
        "substrate": {"available_activations": ["tanh", "identity"], "default_activation": "tanh"},
        "evolution": {
            "pop_size": 8,
            "elitism": 1,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "neat", "rate": 0.2},
            "mutation": {
                "operators": ["add_rich_node", "add_connection"],
                "add_rich_node_prob": 0.3,
                "add_connection_prob": 0.3,
                "adaptive_rates": adaptive,
                "adaptive_rate_sigma": 0.15,
            },
            "train": {"kind": "gradient", "steps": 4, "lr": 0.05, "writeback": True},
            "speciation": {"kind": "neat", "threshold": 1.5, "target_species": 3},
        },
        "fitness": {"components": ["support_accuracy"], "w_support_accuracy": 1.0},
    }


def test_evolver_with_adaptive_rates_runs_and_stamps_offspring(xor_adapter) -> None:
    evolver = build_evolver(_config(adaptive=True))
    state = evolver.seed_state(xor_adapter, random.Random(0))
    for _ in range(3):
        evolver.advance(state, xor_adapter)
    assert any(item.genome.operator_rates is not None for item in state.population)  # offspring carry evolved rates


def test_disabled_adaptive_is_deterministic_and_rate_free(xor_adapter) -> None:
    def run() -> list[float]:
        evolver = build_evolver(_config(adaptive=False))
        state = evolver.seed_state(xor_adapter, random.Random(3))
        for _ in range(3):
            evolver.advance(state, xor_adapter)
        assert all(item.genome.operator_rates is None for item in state.population)
        return sorted(item.fitness for item in state.population)

    assert run() == run()  # off path stays deterministic and rate-free
