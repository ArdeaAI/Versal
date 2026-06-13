"""Functional novelty / quality-diversity pressure for the population.

Pure-fitness selection collapses to the local optimum on DECEPTIVE landscapes (two_spirals: query
accuracy is decoupled from the training fitness the search climbs). Novelty search (Lehman & Stanley)
and MAP-Elites (Mouret & Clune) escape this by rewarding genomes that BEHAVE differently from what
has been seen, not just genomes that score higher: the population diverges through stepping stones
instead of converging on the deceptive optimum.

The behavior must be FUNCTIONAL (what a genome COMPUTES, its output fingerprint), not structural:
structurally diverse genomes can compute the same low-accuracy function, so a structural niche
descriptor would preserve topological variety while every member stays stuck at the same ceiling.

This is one implementation (no registry): a single behavior characterization and a single archive
rule, so per the project's no-premature-abstraction rule it stays plain functions plus one config.
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

Behavior = tuple[float, ...]


@dataclass
class NoveltyConfig:
    """Selectable from `[evolution.novelty]`. Disabled by default so the flat path is unchanged."""

    enabled: bool = False
    weight: float = 0.5  # blend: 0 = pure fitness, 1 = pure novelty
    k: int = 10  # k-nearest neighbours for the sparseness measure
    archive_max: int = 200  # bounded archive of past behaviors (oldest evicts first)
    add_prob: float = 0.1  # chance to also archive a random individual each generation
    descriptor_dim: int = 64  # cap on the functional descriptor length


def distance(a: Behavior, b: Behavior) -> float:
    """Euclidean distance over the shared prefix (descriptors are equal-length within one evolve)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(n)))


def novelty_scores(behaviors: Sequence[Behavior], archive: Sequence[Behavior], k: int) -> list[float]:
    """Sparseness of each behavior: the mean distance to its k nearest neighbours among the OTHER
    current population members plus the archive. Higher = in a sparser, less-explored region."""
    scores: list[float] = []
    for index, behavior in enumerate(behaviors):
        references = [other for other_index, other in enumerate(behaviors) if other_index != index]
        references.extend(archive)
        if not references:
            scores.append(0.0)
            continue
        nearest = sorted(distance(behavior, reference) for reference in references)[: max(1, k)]
        scores.append(sum(nearest) / len(nearest))
    return scores


def update_archive(archive: list[Behavior], behaviors: Sequence[Behavior], scores: Sequence[float], *, rng: random.Random, archive_max: int, add_prob: float) -> None:
    """Grow the archive with the generation's most-novel individual plus an occasional random one
    (the Lehman-Stanley rule), bounded by `archive_max` with oldest-first eviction."""
    if behaviors:
        most_novel = max(range(len(behaviors)), key=lambda index: scores[index])
        archive.append(behaviors[most_novel])
        if rng.random() < add_prob:
            archive.append(behaviors[rng.randrange(len(behaviors))])
    while len(archive) > archive_max:
        archive.pop(0)


def _min_max(values: Sequence[float]) -> list[float]:
    low, high = min(values), max(values)
    span = high - low
    if span <= 0.0:
        return [0.0 for _ in values]
    return [(value - low) / span for value in values]


def blend(fitnesses: Sequence[float], novelties: Sequence[float], weight: float) -> list[float]:
    """Min-max normalize fitness and novelty to [0, 1] across the population, then
    effective = (1 - weight) * fitness_norm + weight * novelty_norm. Normalizing first makes `weight`
    meaningful despite the different, task-dependent scales of fitness and novelty distance."""
    fitness_norm = _min_max(fitnesses)
    novelty_norm = _min_max(novelties)
    return [(1.0 - weight) * f + weight * n for f, n in zip(fitness_norm, novelty_norm, strict=True)]
