"""Selection operators: choose parents from the scored population. Higher fitness is better.

Every op accepts an optional `objectives` keyword (per-candidate objective vectors, maximization
sense, aligned with `population`). The scalar ops ignore it, so any (kind x config) combination is
safe; `nsga2` uses it, and without it degrades to single-objective fronts, i.e. a deterministic
fitness tournament, which is what the hierarchical loop's scalar call sites get.
"""

import math
import random
from typing import Callable

from ardevo.evolution.genome import Genome
from ardevo.evolution.registry import Registry

SelectionOp = Callable[..., list[Genome]]

SELECTION: Registry[SelectionOp] = Registry("selection")


@SELECTION.register("tournament")
def tournament(
    population: list[Genome],
    fitnesses: list[float],
    *,
    rng: random.Random,
    count: int,
    tournament_size: int = 3,
    objectives: list[list[float]] | None = None,
) -> list[Genome]:
    """Pick `count` parents, each the fittest of `tournament_size` random contenders."""
    size = len(population)
    parents: list[Genome] = []
    for _ in range(count):
        contenders = [rng.randrange(size) for _ in range(tournament_size)]
        winner = max(contenders, key=lambda index: fitnesses[index])
        parents.append(population[winner])
    return parents


@SELECTION.register("truncation")
def truncation(
    population: list[Genome],
    fitnesses: list[float],
    *,
    rng: random.Random,
    count: int,
    fraction: float = 0.5,
    objectives: list[list[float]] | None = None,
) -> list[Genome]:
    """Sample `count` parents uniformly from the top `fraction` of the population."""
    size = len(population)
    keep = max(1, int(size * fraction))
    elite = sorted(range(size), key=lambda index: fitnesses[index], reverse=True)[:keep]
    return [population[rng.choice(elite)] for _ in range(count)]


def _dominates(a: list[float], b: list[float]) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def _accumulate_crowding(front: list[int], objectives: list[list[float]], crowding: list[float]) -> None:
    if len(front) <= 2:
        for index in front:
            crowding[index] = math.inf
        return
    n_objectives = len(objectives[front[0]])
    for axis in range(n_objectives):
        ordered = sorted(front, key=lambda index: (objectives[index][axis], index))
        span = objectives[ordered[-1]][axis] - objectives[ordered[0]][axis]
        crowding[ordered[0]] = crowding[ordered[-1]] = math.inf
        if span <= 0.0:
            continue  # a degenerate axis contributes no gap, never a division by zero
        for position in range(1, len(ordered) - 1):
            if crowding[ordered[position]] == math.inf:
                continue
            crowding[ordered[position]] += (objectives[ordered[position + 1]][axis] - objectives[ordered[position - 1]][axis]) / span


def pareto_ranks_and_crowding(objectives: list[list[float]]) -> tuple[list[int], list[float]]:
    """Deb's fast non-dominated sort (O(M * N^2)) plus per-front crowding distance.

    Fully deterministic: pairs iterate in index order, fronts accumulate in index order, and
    boundary members of each front get infinite crowding."""
    size = len(objectives)
    ranks = [0] * size
    crowding = [0.0] * size
    dominator_count = [0] * size
    dominated: list[list[int]] = [[] for _ in range(size)]
    for a in range(size):
        for b in range(a + 1, size):
            if _dominates(objectives[a], objectives[b]):
                dominated[a].append(b)
                dominator_count[b] += 1
            elif _dominates(objectives[b], objectives[a]):
                dominated[b].append(a)
                dominator_count[a] += 1
    front = [index for index in range(size) if dominator_count[index] == 0]
    rank = 0
    while front:
        for index in front:
            ranks[index] = rank
        _accumulate_crowding(front, objectives, crowding)
        successors: list[int] = []
        for index in front:
            for other in dominated[index]:
                dominator_count[other] -= 1
                if dominator_count[other] == 0:
                    successors.append(other)
        front = sorted(successors)
        rank += 1
    return ranks, crowding


def pareto_sort_key(ranks: list[int], crowding: list[float], fitnesses: list[float]) -> Callable[[int], tuple[float, float, float, int]]:
    """Ascending-sort key: rank first, widest crowding next, scalar fitness as the continuity
    tie-break, index last so equal candidates order deterministically."""

    def key(index: int) -> tuple[float, float, float, int]:
        return (ranks[index], -crowding[index], -fitnesses[index], index)

    return key


@SELECTION.register("nsga2")
def nsga2(
    population: list[Genome],
    fitnesses: list[float],
    *,
    rng: random.Random,
    count: int,
    objectives: list[list[float]] | None = None,
) -> list[Genome]:
    """Crowded binary tournament over (Pareto rank, crowding distance): NSGA-II's selection rule.

    Non-dominated sort keeps behaviorally different trade-offs alive through a deceptive middle
    (the divergent-selection lever); crowding spreads parents along each front."""
    size = len(population)
    vectors = objectives if objectives is not None else [[fitness] for fitness in fitnesses]
    ranks, crowding = pareto_ranks_and_crowding(vectors)
    key = pareto_sort_key(ranks, crowding, fitnesses)
    parents: list[Genome] = []
    for _ in range(count):
        first = rng.randrange(size)
        second = rng.randrange(size)
        parents.append(population[first if key(first) <= key(second) else second])
    return parents
