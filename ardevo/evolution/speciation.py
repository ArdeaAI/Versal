"""Speciation: an independent stage that protects structural innovation.

A freshly-grown topology is usually worse before its weights are tuned, so under global selection
it is culled before it can prove itself. Speciation (NEAT-style) groups genomes by compatibility
distance and makes them compete mostly within their own niche, with fitness sharing so no single
species dominates. It is a registered, swappable stage: `none` reproduces the current global
behavior, `neat` enables niching.

A speciator is a callable that turns the scored population into a list of `SpeciesPlan`s (how many
champions to keep and offspring to breed per group); the Evolver applies the existing selection /
crossover / mutation operators within each group. NEAT speciation is stateful (it remembers a
representative genome per species across generations), so the registry returns configured instances.
"""

import random
from dataclasses import dataclass, field
from typing import Any

from ardevo.evolution.genome import Genome, genome_from_dict, genome_to_dict
from ardevo.evolution.registry import Registry

SPECIATION: Registry = Registry("speciation")


@dataclass
class SpeciesPlan:
    """A reproduction quota for one group: which members it spans, and how many to keep/breed.

    `species_id` is stable across generations (a persistent niche keeps its id until it dies out),
    so the per-generation sizes can be charted as bands that appear and disappear.
    """

    species_id: int
    members: list[int]  # indices into the scored population
    n_elites: int
    n_offspring: int


def compatibility_distance(a: Genome, b: Genome, *, c_excess: float, c_disjoint: float, c_weight: float) -> float:
    """NEAT compatibility distance: excess/disjoint gene counts plus mean matching-weight difference.

    Genes are aligned by innovation number. Excess genes lie beyond the other genome's last
    innovation; disjoint genes are interior mismatches. Normalized by the larger gene count.
    """
    genes_a = {conn.innovation: conn for conn in a.connections}
    genes_b = {conn.innovation: conn for conn in b.connections}
    if not genes_a and not genes_b:
        return 0.0

    boundary = min(max(genes_a, default=0), max(genes_b, default=0))
    excess = disjoint = matching = 0
    weight_difference = 0.0
    for innovation in set(genes_a) | set(genes_b):
        in_a, in_b = innovation in genes_a, innovation in genes_b
        if in_a and in_b:
            matching += 1
            weight_difference += abs(genes_a[innovation].weight - genes_b[innovation].weight)
        elif innovation > boundary:
            excess += 1
        else:
            disjoint += 1

    normalizer = max(len(genes_a), len(genes_b), 1)
    average_weight = weight_difference / matching if matching else 0.0
    return c_excess * excess / normalizer + c_disjoint * disjoint / normalizer + c_weight * average_weight


@SPECIATION.register("none")
def _build_none(**_params: object) -> "NoSpeciation":
    return NoSpeciation()


@SPECIATION.register("neat")
def _build_neat(
    *,
    threshold: float = 1.5,
    c_excess: float = 1.0,
    c_disjoint: float = 1.0,
    c_weight: float = 0.5,
    target_species: int = 12,
    threshold_adjust: float = 0.3,
    min_threshold: float = 0.3,
    **_params: object,
) -> "NeatSpeciation":
    return NeatSpeciation(
        threshold=threshold,
        c_excess=c_excess,
        c_disjoint=c_disjoint,
        c_weight=c_weight,
        target_species=target_species,
        threshold_adjust=threshold_adjust,
        min_threshold=min_threshold,
    )


class NoSpeciation:
    """One global group: keep `elitism` champions, breed the rest. The pre-speciation behavior."""

    def __call__(self, genomes: list[Genome], fitnesses: list[float], *, rng: random.Random, elitism: int, pop_size: int) -> list[SpeciesPlan]:
        size = len(genomes)
        elites = min(elitism, size)
        return [SpeciesPlan(species_id=0, members=list(range(size)), n_elites=elites, n_offspring=pop_size - elites)]

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, data: dict[str, Any]) -> None:
        return None  # no persistent state


@dataclass
class _Species:
    """A persistent niche: a stable id, a representative genome carried across generations, and this
    generation's member indices."""

    id: int
    representative: Genome
    members: list[int] = field(default_factory=list)


@dataclass
class NeatSpeciation:
    """Compatibility-distance niching with fitness sharing and one champion per species.

    Species persist across generations (each keeps its representative and stable id); a species with
    no members in a generation dies, and a genome compatible with none starts a new species.

    The compatibility `threshold` auto-adjusts toward `target_species` each generation (the standard
    NEAT remedy): a fixed threshold is brittle, and operators like `add_rich_node` make genomes
    diverge fast enough that a too-low threshold fractures the whole population into singletons,
    which starves reproduction. Set `target_species = 0` to disable and keep `threshold` fixed.
    """

    threshold: float
    c_excess: float
    c_disjoint: float
    c_weight: float
    target_species: int = 0
    threshold_adjust: float = 0.3
    min_threshold: float = 0.3
    species: list[_Species] = field(default_factory=list)
    _next_id: int = 0

    def _distance(self, a: Genome, b: Genome) -> float:
        return compatibility_distance(a, b, c_excess=self.c_excess, c_disjoint=self.c_disjoint, c_weight=self.c_weight)

    def _partition(self, genomes: list[Genome], rng: random.Random) -> None:
        """Assign each genome to a persistent species; retire empty species; refresh representatives."""
        for species in self.species:
            species.members = []
        for index, genome in enumerate(genomes):
            for species in self.species:
                if self._distance(genome, species.representative) < self.threshold:
                    species.members.append(index)
                    break
            else:
                self.species.append(_Species(id=self._next_id, representative=genome, members=[index]))
                self._next_id += 1
        self.species = [species for species in self.species if species.members]
        for species in self.species:
            species.representative = genomes[rng.choice(species.members)]

    def __call__(self, genomes: list[Genome], fitnesses: list[float], *, rng: random.Random, elitism: int, pop_size: int) -> list[SpeciesPlan]:
        self._partition(genomes, rng)
        groups = self.species

        if len(groups) >= pop_size:
            # Safety net (should not happen once the threshold is targeting): too many species to give
            # each a champion AND breed, so keep the fittest pop_size champions and breed nothing.
            groups = sorted(groups, key=lambda s: max(fitnesses[i] for i in s.members), reverse=True)[:pop_size]
            self.species = groups
            plans = [SpeciesPlan(species_id=s.id, members=s.members, n_elites=1, n_offspring=0) for s in groups]
        else:
            # Fitness sharing: a species' share is its MEAN fitness (sum of f_i / size), shifted positive.
            means = [sum(fitnesses[i] for i in s.members) / len(s.members) for s in groups]
            shift = min(means)
            shares = [mean - shift + 1e-6 for mean in means]
            total_share = sum(shares)

            offspring_budget = pop_size - len(groups)  # one champion reserved per species
            offspring = [int(round(offspring_budget * share / total_share)) for share in shares]
            self._reconcile(offspring, offspring_budget)
            plans = [SpeciesPlan(species_id=s.id, members=s.members, n_elites=1, n_offspring=count) for s, count in zip(groups, offspring)]

        self._adjust_threshold(len(groups))
        return plans

    def _adjust_threshold(self, n_species: int) -> None:
        """Nudge the compatibility threshold toward keeping the species count near `target_species`."""
        if self.target_species <= 0:
            return
        if n_species > self.target_species:
            self.threshold += self.threshold_adjust
        elif n_species < self.target_species:
            self.threshold = max(self.min_threshold, self.threshold - self.threshold_adjust)

    @staticmethod
    def _reconcile(offspring: list[int], budget: int) -> None:
        """Fix rounding so the offspring counts sum exactly to the budget (adjust the largest group)."""
        difference = budget - sum(offspring)
        if difference and offspring:
            largest = max(range(len(offspring)), key=lambda i: offspring[i])
            offspring[largest] = max(0, offspring[largest] + difference)

    def state_dict(self) -> dict[str, Any]:
        """Serialize the persistent niche state (id counter, threshold, per-species representative).

        Per-generation `members` are not stored: on resume the population is re-assessed and the next
        call re-partitions it against the saved representatives.
        """
        return {
            "threshold": self.threshold,
            "next_id": self._next_id,
            "species": [{"id": species.id, "representative": genome_to_dict(species.representative)} for species in self.species],
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.threshold = float(data["threshold"])
        self._next_id = int(data["next_id"])
        self.species = [_Species(id=int(entry["id"]), representative=genome_from_dict(entry["representative"])) for entry in data["species"]]
