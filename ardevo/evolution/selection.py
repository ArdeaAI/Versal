"""Selection operators: choose parents from the scored population. Higher fitness is better."""

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
) -> list[Genome]:
    """Sample `count` parents uniformly from the top `fraction` of the population."""
    size = len(population)
    keep = max(1, int(size * fraction))
    elite = sorted(range(size), key=lambda index: fitnesses[index], reverse=True)[:keep]
    return [population[rng.choice(elite)] for _ in range(count)]
