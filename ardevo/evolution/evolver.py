"""Evolver: the thin generational loop that runs config-selected operators.

The Evolver owns the algorithm, not the task: genetic operators are injected (config-driven via
`build_evolver`), and a `TaskAdapter` injects the task-specific decode/evaluate. Each generation
runs the stages in order: select -> crossover -> mutate -> train -> evaluate -> fitness -> replace.
"""

import random
from dataclasses import dataclass
from typing import Callable

from ardevo.dataset.icarus import EncodedTask, Level0Encoder
from ardevo.evaluation import evaluate
from ardevo.evolution.fitness import FitnessAggregator
from ardevo.evolution.genome import Genome, InnovationTracker
from ardevo.evolution.mutation import MutationContext, MutationPipeline
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
    activations: list[str]
    default_activation: str

    def run(
        self,
        adapter: TaskAdapter,
        generations: int,
        on_generation: GenerationHook | None = None,
        stop_at_accuracy: float = 1.0,
    ) -> Assessed:
        rng = random.Random(self.seed)
        population = [self.init_op(adapter.n_inputs, adapter.n_outputs, rng=rng) for _ in range(self.pop_size)]
        ctx = MutationContext(
            innovations=InnovationTracker.from_genomes(population),
            activations=self.activations,
            default_activation=self.default_activation,
        )

        def assess(genome: Genome) -> Assessed:
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

        ranked = sorted(range(len(assessed)), key=lambda index: fitnesses[index], reverse=True)
        elites = [genomes[index] for index in ranked[: self.elitism]]

        n_offspring = self.pop_size - len(elites)
        parents = self.selection_op(genomes, fitnesses, rng=rng, count=2 * n_offspring)
        offspring: list[Genome] = []
        for k in range(n_offspring):
            child = self.crossover_op(parents[2 * k], parents[2 * k + 1], rng=rng)
            child = self.mutation(child, ctx, rng=rng)
            offspring.append(child)

        return [assess(genome) for genome in [*elites, *offspring]]
