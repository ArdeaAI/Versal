"""nsga2: exact non-dominated sort + crowding on objective vectors; scalar ops ignore the kwarg.

Maximization sense throughout (higher is better on every objective), matching the fitness
convention. The crowded binary tournament breaks ties deterministically by (rank, -crowding,
-scalar fitness, index) so seeded runs reproduce exactly.
"""

import math
import random
from typing import Any

import pytest

from ardevo.evolution.selection import nsga2, pareto_ranks_and_crowding, tournament, truncation

# Front 0 = the first three (mutually non-dominating); (2,2) is dominated by (3,3) only; (0,0) by all.
_OBJECTIVES = [[1.0, 5.0], [5.0, 1.0], [3.0, 3.0], [2.0, 2.0], [0.0, 0.0]]


def test_pareto_ranks_hand_built_front() -> None:
    ranks, _crowding = pareto_ranks_and_crowding(_OBJECTIVES)
    assert ranks == [0, 0, 0, 1, 2]


def test_crowding_boundary_infinite_and_interior_normalized() -> None:
    _ranks, crowding = pareto_ranks_and_crowding(_OBJECTIVES)
    # Within front 0, (1,5) and (5,1) are boundary points on both objectives; (3,3) is interior
    # with normalized gap (5-1)/(5-1) = 1.0 per objective.
    assert crowding[0] == math.inf and crowding[1] == math.inf
    assert crowding[2] == pytest.approx(2.0)
    # Singleton fronts are boundaries by construction.
    assert crowding[3] == math.inf and crowding[4] == math.inf


def test_zero_range_objective_contributes_no_crowding() -> None:
    # One shared front (axis 0 rises while axis 1 falls) with a constant axis 2: the degenerate
    # axis must contribute 0 to interior crowding, never a division by zero or NaN.
    ranks, crowding = pareto_ranks_and_crowding([[1.0, 5.0, 7.0], [2.0, 4.0, 7.0], [3.0, 3.0, 7.0]])
    assert ranks == [0, 0, 0]
    assert crowding[1] == pytest.approx(2.0)  # 1.0 from axis 0 plus 1.0 from axis 1, nothing from axis 2
    assert all(not math.isnan(value) for value in crowding)


def test_floored_corpse_vector_ranks_last() -> None:
    # The corpse guard: an undecodable genome's floor vector must never reach the front, however
    # attractive its (tiny) wiring cost would have been.
    ranks, _crowding = pareto_ranks_and_crowding([[0.8, -5.0], [0.5, -3.0], [-1e9, -1e9]])
    assert ranks[2] == max(ranks)


def test_nsga2_selects_front_zero_dominant_parents() -> None:
    population: list[Any] = ["a", "b", "c", "d", "e"]  # selection never reads genome internals, so strings duck-type
    fitnesses = [1.0, 1.0, 1.0, 0.5, 0.1]
    parents: list[Any] = nsga2(population, fitnesses, rng=random.Random(0), count=50, objectives=_OBJECTIVES)
    assert len(parents) == 50
    front_zero = {"a", "b", "c"}
    assert sum(parent in front_zero for parent in parents) >= 40  # rank 0 wins every mixed tournament
    assert "e" not in parents or parents.count("e") <= 3  # the fully dominated point almost never wins


def test_nsga2_deterministic_under_seeded_rng() -> None:
    population = list(range(5))
    fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5]
    first = nsga2(population, fitnesses, rng=random.Random(3), count=20, objectives=_OBJECTIVES)
    second = nsga2(population, fitnesses, rng=random.Random(3), count=20, objectives=_OBJECTIVES)
    assert first == second


def test_nsga2_scalar_degradation_orders_by_fitness() -> None:
    # Without vectors, single-objective fronts are descending fitness equivalence classes, so the
    # op degrades to a deterministic fitness tournament (the hierarchical-loop call sites).
    ranks, _crowding = pareto_ranks_and_crowding([[0.1], [0.9], [0.5], [0.9]])
    assert ranks == [2, 0, 1, 0]
    population: list[Any] = ["low", "high"]
    parents: list[Any] = nsga2(population, [0.0, 1.0], rng=random.Random(0), count=30, objectives=None)
    assert parents.count("high") > parents.count("low")


def test_scalar_ops_ignore_objectives_kwarg() -> None:
    population = list(range(6))
    fitnesses = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2]
    vectors = [[value] for value in fitnesses]
    for op in (tournament, truncation):
        without = op(population, fitnesses, rng=random.Random(7), count=12)
        with_kwarg = op(population, fitnesses, rng=random.Random(7), count=12, objectives=vectors)
        assert without == with_kwarg  # byte-identical: the kwarg is accepted and never read


class _Sized:
    """Duck-typed candidate for the parsimony path (which reads only `complexity()`)."""

    def __init__(self, name: str, complexity: int) -> None:
        self.name = name
        self._complexity = complexity

    def complexity(self) -> int:
        return self._complexity

    def __repr__(self) -> str:
        return self.name


def test_parsimony_epsilon_zero_is_byte_identical_including_rng_stream() -> None:
    population = list(range(6))
    fitnesses = [0.1, 0.9, 0.3, 0.7, 0.5, 0.2]
    for op in (tournament, truncation, nsga2):
        rng_base, rng_gated = random.Random(11), random.Random(11)
        base = op(population, fitnesses, rng=rng_base, count=12)
        gated = op(population, fitnesses, rng=rng_gated, count=12, parsimony_epsilon=0.0)
        assert base == gated
        assert rng_base.random() == rng_gated.random()  # the NEXT draw matches: no hidden rng consumption


def test_tournament_parsimony_prefers_simpler_within_a_band_and_fitter_across_bands() -> None:
    big, small = _Sized("big", 50), _Sized("small", 5)
    population: list[Any] = [big, small]
    within_band = [1.005, 1.000]  # same floor(f / 0.01) band: noise-level difference
    parents: list[Any] = tournament(population, within_band, rng=random.Random(0), count=40, tournament_size=2, parsimony_epsilon=0.01)
    assert parents.count(small) > parents.count(big)
    without: list[Any] = tournament(population, within_band, rng=random.Random(0), count=40, tournament_size=2)
    assert without.count(big) > without.count(small)  # eps off: raw fitness rules
    across_bands = [1.020, 1.000]  # a real fitness step crosses a band and beats parsimony
    banded: list[Any] = tournament(population, across_bands, rng=random.Random(0), count=40, tournament_size=2, parsimony_epsilon=0.01)
    assert banded.count(big) > banded.count(small)


def test_truncation_parsimony_reorders_the_elite_cut_only_when_on() -> None:
    population: list[Any] = [_Sized("small", 5), _Sized("big", 50), _Sized("weak_a", 1), _Sized("weak_b", 1)]
    fitnesses = [1.000, 1.001, 0.5, 0.4]
    scalar_elite: list[Any] = truncation(population, fitnesses, rng=random.Random(2), count=20, fraction=0.25)
    assert set(scalar_elite) == {population[1]}  # keep=1: highest raw fitness
    parsimonious: list[Any] = truncation(population, fitnesses, rng=random.Random(2), count=20, fraction=0.25, parsimony_epsilon=0.01)
    assert set(parsimonious) == {population[0]}  # same band, the smaller genome takes the slot


def test_nsga2_parsimony_sits_behind_rank_and_crowding() -> None:
    big, small = _Sized("big", 50), _Sized("small", 5)
    population: list[Any] = [big, small]
    # Two-candidate front: mutually non-dominating, both boundary (infinite crowding), same fitness
    # band, so parsimony is the deciding tier.
    tied_front = [[1.0, 0.0], [0.0, 1.0]]
    parents: list[Any] = nsga2(population, [1.0, 1.0], rng=random.Random(0), count=40, objectives=tied_front, parsimony_epsilon=0.01)
    assert parents.count(small) > parents.count(big)  # small wins every MIXED tournament (same-index draws split)
    # Dominance first: when big strictly dominates, rank decides regardless of complexity.
    dominant = [[1.0, 1.0], [0.5, 0.5]]
    ranked: list[Any] = nsga2(population, [1.0, 1.0], rng=random.Random(0), count=40, objectives=dominant, parsimony_epsilon=0.01)
    assert ranked.count(big) > ranked.count(small)


def test_nsga2_scalar_path_parsimony_decides_within_a_band() -> None:
    # The module-pool shape: no objective vectors, 1-D fronts from scalar fitness. Within a band
    # the fronts tie and the smaller genome wins every mixed tournament.
    big, small = _Sized("big", 50), _Sized("small", 5)
    parents: list[Any] = nsga2([big, small], [1.005, 1.000], rng=random.Random(0), count=30, objectives=None, parsimony_epsilon=0.01)
    assert parents.count(small) > parents.count(big)


def test_parsimony_epsilon_flows_from_config() -> None:
    from functools import partial

    from ardevo.evolution.registry import build_evolver
    from tests.test_hierarchical_loop import _config as _loop_config

    config = _loop_config()
    config["evolution"]["selection"] = {"kind": "tournament", "tournament_size": 3, "parsimony_epsilon": 0.01}
    evolver = build_evolver(config)
    assert isinstance(evolver.selection_op, partial)
    assert evolver.selection_op.keywords["parsimony_epsilon"] == 0.01
