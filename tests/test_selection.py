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
