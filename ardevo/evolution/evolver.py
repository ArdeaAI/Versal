"""Evolver: the thin generational loop that runs config-selected operators.

The Evolver owns the algorithm, not the task: genetic operators are injected (config-driven via
`build_evolver`), and a `TaskAdapter` injects the task-specific decode/evaluate. Each generation
runs the stages in order: select -> crossover -> mutate -> train -> evaluate -> fitness -> replace.
"""

import random
from dataclasses import dataclass, field
from typing import Callable

from ardevo.dataset.icarus import EncodedTask, Level0Encoder
from ardevo.evaluation import evaluate
from ardevo.evolution.fitness import FitnessAggregator
from ardevo.evolution.genome import Genome, InnovationTracker, make_acyclic
from ardevo.evolution.mutation import MutationContext, MutationPipeline
from ardevo.evolution.speciation import SpeciesPlan
from ardevo.substrate import GraphNet, decode


@dataclass
class TaskAdapter:
    """Injects task specifics (encoded tensors, widths) so the Evolver stays task-agnostic."""

    encoded: EncodedTask
    encoder: Level0Encoder
    n_inputs: int
    n_outputs: int

    def decode(self, genome: Genome) -> GraphNet:
        return decode(genome, self.n_inputs, self.n_outputs)

    def evaluate(self, module: GraphNet) -> dict[str, float]:
        return evaluate(module, self.encoded, self.encoder)


@dataclass
class Assessed:
    genome: Genome
    metrics: dict[str, float]
    fitness: float
    module: GraphNet  # the exact trained network that produced these metrics (for faithful saving)


GenerationHook = Callable[[int, Assessed, float], None]


@dataclass
class Evolver:
    pop_size: int
    elitism: int
    seed: int
    init_op: Callable[..., Genome]
    selection_op: Callable[..., list[Genome]]
    crossover_op: Callable[..., Genome]
    mutation: MutationPipeline
    train_op: Callable[..., tuple[Genome, GraphNet]]
    fitness: FitnessAggregator
    speciate: Callable[..., list[SpeciesPlan]]
    activations: list[str]
    default_activation: str
    # Per-generation {species_id: size} snapshots, populated during run() for the speciation chart.
    species_history: list[dict[int, int]] = field(default_factory=list)

    def run(
        self,
        adapter: TaskAdapter,
        generations: int,
        on_generation: GenerationHook | None = None,
        stop_at_accuracy: float = 1.0,
    ) -> Assessed:
        rng = random.Random(self.seed)
        self.species_history = []
        population = [self.init_op(adapter.n_inputs, adapter.n_outputs, rng=rng) for _ in range(self.pop_size)]
        ctx = MutationContext(
            innovations=InnovationTracker.from_genomes(population),
            activations=self.activations,
            default_activation=self.default_activation,
        )

        def assess(genome: Genome) -> Assessed:
            try:
                module = adapter.decode(genome)
            except ValueError:
                # Recombination/toggling can produce a cyclic genome; repair to feedforward and retry.
                genome = make_acyclic(genome)
                module = adapter.decode(genome)
            genome, module = self.train_op(genome, module, adapter.encoded, rng=rng)
            metrics = adapter.evaluate(module)
            return Assessed(genome, metrics, self.fitness(genome, metrics), module)

        assessed = [assess(genome) for genome in population]
        best = max(assessed, key=lambda item: item.fitness)

        for generation in range(generations):
            generation_best = max(assessed, key=lambda item: item.fitness)
            if generation_best.fitness > best.fitness:
                best = generation_best
            if on_generation is not None:
                mean_fitness = sum(item.fitness for item in assessed) / len(assessed)
                on_generation(generation, generation_best, mean_fitness)
            if generation_best.metrics.get("query_accuracy", 0.0) >= stop_at_accuracy:
                break

            assessed = self._next_generation(assessed, ctx, rng, assess)

        final_best = max(assessed, key=lambda item: item.fitness)
        return final_best if final_best.fitness > best.fitness else best

    def _next_generation(
        self,
        assessed: list[Assessed],
        ctx: MutationContext,
        rng: random.Random,
        assess: Callable[[Genome], Assessed],
    ) -> list[Assessed]:
        genomes = [item.genome for item in assessed]
        fitnesses = [item.fitness for item in assessed]
        plans = self.speciate(genomes, fitnesses, rng=rng, elitism=self.elitism, pop_size=self.pop_size)
        self.species_history.append({plan.species_id: len(plan.members) for plan in plans})

        next_assessed: list[Assessed] = []
        for plan in plans:
            members = sorted((assessed[index] for index in plan.members), key=lambda item: item.fitness, reverse=True)
            # Champions are carried forward UNCHANGED: re-assessing them would re-run the train
            # step every generation and overfit them on the support set (champion drift).
            next_assessed.extend(members[: plan.n_elites])
            if plan.n_offspring <= 0:
                continue

            species_genomes = [item.genome for item in members]
            species_fitnesses = [item.fitness for item in members]
            parents = self.selection_op(species_genomes, species_fitnesses, rng=rng, count=2 * plan.n_offspring)
            for k in range(plan.n_offspring):
                child = self.crossover_op(parents[2 * k], parents[2 * k + 1], rng=rng)
                child = self.mutation(child, ctx, rng=rng)
                next_assessed.append(assess(child))

        return next_assessed
