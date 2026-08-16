"""Objective vectors through the generational loop: gated fill, Pareto elite ordering, corpse floor.

The scalar path must stay byte-identical when `[fitness] objectives` is absent; these tests pin the
gated behavior when it is present (NoSpeciation = one global species, so ranks are global).
"""

import random
from typing import Any

from versal.evolution.evolver import Evolver, TaskAdapter
from versal.evolution.genome import ConnectionGene, Genome
from versal.evolution.registry import build_evolver
from versal.evolution.selection import pareto_ranks_and_crowding


def _config(objectives: bool) -> dict[str, Any]:
    config: dict[str, Any] = {
        "seed": 0,
        "substrate": {"available_activations": ["tanh"], "default_activation": "tanh"},
        "evolution": {
            "pop_size": 8,
            "elitism": 1,
            "init": {"kind": "minimal"},
            "selection": {"kind": "nsga2"},
            "crossover": {"kind": "none", "rate": 0.0},
            "mutation": {"operators": ["perturb_weights"], "perturb_weights_prob": 0.5},
            "train": {"kind": "none"},
            "speciation": {"kind": "none"},
        },
        "fitness": {"components": ["support_accuracy", "complexity_penalty"], "w_complexity_penalty": 0.01},
    }
    if objectives:
        config["fitness"]["objectives"] = ["support_accuracy", "connection_cost"]
    return config


def _corpse(linear_genome: Genome) -> Genome:
    corpse = linear_genome.clone()
    corpse.connections = [*corpse.connections, ConnectionGene(99, 3, 1.0, True, 77)]  # node 99 does not exist: decode KeyErrors
    return corpse


def test_advance_fills_vectors_and_elite_is_rank_zero(xor_adapter: TaskAdapter) -> None:
    evolver: Evolver = build_evolver(_config(objectives=True))
    state = evolver.seed_state(xor_adapter, random.Random(0))
    previous = list(state.population)
    evolver.advance(state, xor_adapter)

    assert all(item.objectives is not None and len(item.objectives) == 2 for item in previous)
    ranks, _crowding = pareto_ranks_and_crowding([item.objectives or [] for item in previous])
    elites = [item for item in state.population if any(item is old for old in previous)]
    assert elites  # elitism = 1 carries at least one member unchanged
    for elite in elites:
        assert ranks[next(index for index, old in enumerate(previous) if old is elite)] == 0


def test_corpse_floor_vector_never_elite(xor_adapter: TaskAdapter, linear_genome: Genome) -> None:
    evolver: Evolver = build_evolver(_config(objectives=True))
    corpse_genome = _corpse(linear_genome)
    state = evolver.seed_state(xor_adapter, random.Random(0), seeded_front=lambda _innovations: [corpse_genome])
    corpse = next(item for item in state.population if item.module is None)
    assert corpse.fitness == -1e9
    previous = list(state.population)
    evolver.advance(state, xor_adapter)
    assert corpse.objectives == [-1e9, -1e9]  # the tiny-graph wiring advantage must not survive flooring
    elites = [item for item in state.population if any(item is old for old in previous)]
    assert corpse not in elites


def test_scalar_config_leaves_vectors_unset(xor_adapter: TaskAdapter) -> None:
    evolver: Evolver = build_evolver(_config(objectives=False))
    state = evolver.seed_state(xor_adapter, random.Random(0))
    previous = list(state.population)
    evolver.advance(state, xor_adapter)
    assert all(item.objectives is None for item in previous)  # the vector path never ran
