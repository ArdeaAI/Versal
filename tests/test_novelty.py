"""Functional novelty / quality-diversity selection: the deceptive-landscape escape lever.

Pure-fitness tournament collapses to the local optimum on deceptive tasks (two_spirals). Blending a
behavioral-novelty term keeps the search diverging. These tests pin the math, the byte-identical
disabled path, determinism, and the core property: novelty shifts selection toward behaviorally-rare
genomes even when their fitness is lower (the essence of escaping deception)."""

import random
from copy import deepcopy
from typing import Any

from ardevo.evaluation import behavior_descriptor
from ardevo.evolution.novelty import blend, distance, novelty_scores, update_archive
from ardevo.evolution.registry import build_evolver


def _config(novelty: dict[str, Any] | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "seed": 0,
        "substrate": {"available_activations": ["tanh", "identity"], "default_activation": "tanh"},
        "evolution": {
            "pop_size": 8,
            "elitism": 1,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "neat", "rate": 0.2},
            "mutation": {"operators": ["add_rich_node", "add_connection"], "add_rich_node_prob": 0.3, "add_rich_node_fan_in": 2, "add_connection_prob": 0.3},
            "train": {"kind": "gradient", "steps": 4, "lr": 0.05, "writeback": True},
            "speciation": {"kind": "neat", "threshold": 1.5, "target_species": 3},
        },
        "fitness": {"components": ["support_accuracy", "complexity_penalty"], "w_support_accuracy": 1.0, "w_complexity_penalty": 0.01},
    }
    if novelty is not None:
        config["evolution"]["novelty"] = novelty
    return config


# --- novelty math -------------------------------------------------------------------------------


def test_distance_shared_prefix() -> None:
    assert distance((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert distance((), ()) == 0.0


def test_novelty_scores_rank_isolated_behavior_highest() -> None:
    # Three behaviors cluster near the origin, one sits far away: the far one must be the most novel.
    behaviors = [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (10.0, 10.0)]
    scores = novelty_scores(behaviors, archive=[], k=2)
    assert scores.index(max(scores)) == 3


def test_novelty_scores_count_the_archive() -> None:
    behaviors = [(0.0, 0.0)]
    alone = novelty_scores(behaviors, archive=[], k=1)[0]
    crowded = novelty_scores(behaviors, archive=[(0.0, 0.0)], k=1)[0]
    assert alone == 0.0 and crowded == 0.0  # identical neighbour: zero distance either way
    far = novelty_scores([(0.0, 0.0)], archive=[(5.0, 0.0)], k=1)[0]
    assert far == 5.0


def test_blend_normalizes_and_weights() -> None:
    pure_fitness = blend([1.0, 2.0, 3.0], [9.0, 1.0, 1.0], weight=0.0)
    assert pure_fitness == [0.0, 0.5, 1.0]  # min-max of fitness only
    pure_novelty = blend([1.0, 2.0, 3.0], [9.0, 1.0, 1.0], weight=1.0)
    assert pure_novelty[0] == 1.0 and pure_novelty[1] == 0.0  # the high-novelty, low-fitness one wins


def test_update_archive_grows_and_caps() -> None:
    archive: list[tuple[float, ...]] = []
    rng = random.Random(0)
    for step in range(50):
        update_archive(archive, [(float(step), 0.0), (0.0, 0.0)], [1.0, 0.0], rng=rng, archive_max=10, add_prob=0.0)
    assert len(archive) == 10  # bounded
    assert archive[-1] == (49.0, 0.0)  # most-novel of the last generation was kept


def test_novelty_shifts_selection_toward_rare_behavior() -> None:
    # The deceptive essence: a behaviorally-isolated genome with the LOWEST fitness gets the HIGHEST
    # effective score under high novelty weight, so selection explores it instead of the crowd.
    fitnesses = [3.0, 3.0, 3.0, 0.5]  # the rare one (index 3) is the worst by fitness
    behaviors = [(0.0, 0.0), (0.05, 0.0), (0.0, 0.05), (9.0, 9.0)]
    novelties = novelty_scores(behaviors, archive=[], k=2)
    effective = blend(fitnesses, novelties, weight=0.8)
    assert effective.index(max(effective)) == 3


# --- behavior descriptor -------------------------------------------------------------------------


def test_behavior_descriptor_is_bounded_and_functional(xor_adapter, solving_genome, linear_genome) -> None:
    solver = behavior_descriptor(xor_adapter.decode(solving_genome), xor_adapter.encoded, max_dim=8)
    linear = behavior_descriptor(xor_adapter.decode(linear_genome), xor_adapter.encoded, max_dim=8)
    assert len(solver) <= 8 and len(linear) <= 8
    assert solver != linear  # different functions -> different fingerprints
    again = behavior_descriptor(xor_adapter.decode(solving_genome), xor_adapter.encoded, max_dim=8)
    assert solver == again  # same function -> same fingerprint (deterministic)


# --- Evolver integration -------------------------------------------------------------------------


def test_disabled_novelty_is_identical_to_no_novelty(xor_adapter) -> None:
    # An explicit disabled config must behave exactly like having no novelty config at all.
    none_evolver = build_evolver(_config(novelty=None))
    off_evolver = build_evolver(_config(novelty={"enabled": False}))
    none_state = none_evolver.seed_state(xor_adapter, random.Random(0))
    off_state = off_evolver.seed_state(xor_adapter, random.Random(0))
    for _ in range(3):
        none_evolver.advance(none_state, xor_adapter)
        off_evolver.advance(off_state, xor_adapter)
    none_fit = sorted(item.fitness for item in none_state.population)
    off_fit = sorted(item.fitness for item in off_state.population)
    assert none_fit == off_fit
    assert all(item.behavior == () for item in none_state.population)  # no descriptor when off


def test_novelty_on_populates_behavior_and_archive(xor_adapter) -> None:
    evolver = build_evolver(_config(novelty={"enabled": True, "weight": 0.5, "k": 3, "archive_max": 50}))
    state = evolver.seed_state(xor_adapter, random.Random(0))
    assert all(len(item.behavior) > 0 for item in state.population)  # descriptors computed when on
    for _ in range(3):
        evolver.advance(state, xor_adapter)
    assert len(state.novelty_archive) > 0  # the archive accumulated behaviors


def test_novelty_run_is_deterministic(xor_adapter) -> None:
    def run() -> list[float]:
        evolver = build_evolver(_config(novelty={"enabled": True, "weight": 0.6, "k": 3}))
        state = evolver.seed_state(xor_adapter, random.Random(7))
        for _ in range(4):
            evolver.advance(state, xor_adapter)
        return sorted(item.fitness for item in state.population)

    assert run() == run()  # same seed -> identical run


def test_novelty_does_not_corrupt_true_fitness(xor_adapter) -> None:
    # effective_fitness drives selection; the reported fitness stays the true objective value.
    evolver = build_evolver(_config(novelty={"enabled": True, "weight": 1.0, "k": 3}))
    state = evolver.seed_state(xor_adapter, random.Random(0))
    snapshot = deepcopy(state.population)
    evolver.advance(state, xor_adapter)
    for item in snapshot:
        recomputed = evolver.fitness(item.genome, item.metrics)
        assert abs(recomputed - item.fitness) < 1e-9
