"""Evolver: the thin generational loop that runs config-selected operators.

The Evolver owns the algorithm, not the task: genetic operators are injected (config-driven via
`build_evolver`), and an adapter injects the task-specific decode/evaluate. Each generation runs the
stages in order: select -> crossover -> mutate -> train -> evaluate -> fitness -> replace.

The loop is exposed two ways. `run()` drives a whole single-task search (phase 1). For the continuous
multi-rung trial, `seed_state` / `advance` step one generation at a time over an explicit,
serializable `EvolverState`, so a driver can swap the task adapter between generations (carrying the
population forward) and checkpoint/resume the run.
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Protocol

from ardevo.dataset.icarus import EncodedTask, Level0Encoder
from ardevo.evaluation import evaluate
from ardevo.evolution.fitness import FitnessAggregator
from ardevo.evolution.genome import Genome, InnovationTracker, make_acyclic
from ardevo.evolution.mutation import MutationContext, MutationPipeline
from ardevo.evolution.speciation import SpeciesPlan
from ardevo.substrate import GraphNet, SubstrateModule, decode


class Adapter(Protocol):
    """What the Evolver needs of a task: encoded tensors, I/O widths, and decode + evaluate.

    `TaskAdapter` (single rung) and `MultiTaskAdapter` (one rung of the continuous run) both satisfy
    it, so the loop is agnostic to which one is active and can swap between them generation to
    generation while keeping the same population.
    """

    encoded: EncodedTask
    n_inputs: int
    n_outputs: int

    def decode(self, genome: Genome) -> SubstrateModule: ...

    def evaluate(self, module: SubstrateModule) -> dict[str, float]: ...


@dataclass
class TaskAdapter:
    """Injects task specifics (encoded tensors, widths) so the Evolver stays task-agnostic."""

    encoded: EncodedTask
    encoder: Level0Encoder
    n_inputs: int
    n_outputs: int

    def decode(self, genome: Genome) -> GraphNet:
        return decode(genome, self.n_inputs, self.n_outputs)

    def evaluate(self, module: SubstrateModule) -> dict[str, float]:
        return evaluate(module, self.encoded, self.encoder)


@dataclass
class Assessed:
    genome: Genome
    metrics: dict[str, float]
    fitness: float
    module: SubstrateModule  # the exact trained network that produced these metrics (for faithful saving)


@dataclass
class EvolverState:
    """The full mutable state of a run, so a driver can step it, swap the task, and checkpoint it."""

    population: list[Assessed]
    innovations: InnovationTracker
    rng: random.Random
    generation: int = 0
    # Per-generation {species_id: size} snapshots, for the speciation chart.
    species_history: list[dict[int, int]] = field(default_factory=list)
    best: Assessed | None = None


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
    train_op: Callable[..., tuple[Genome, SubstrateModule]]
    fitness: FitnessAggregator
    speciate: Callable[..., list[SpeciesPlan]]
    activations: list[str]
    default_activation: str
    # Mirror of the active state's species history, kept so single-task callers can read
    # evolver.species_history after run() (the continuous trial reads it off the state instead).
    species_history: list[dict[int, int]] = field(default_factory=list)

    def _context(self, state: EvolverState) -> MutationContext:
        return MutationContext(innovations=state.innovations, activations=self.activations, default_activation=self.default_activation)

    def assess(self, genome: Genome, adapter: Adapter, state: EvolverState) -> Assessed:
        """Decode (repairing cycles), train the weights, evaluate, and score one genome."""
        module = self._decode(genome, adapter)
        genome, module = self.train_op(genome, module, adapter.encoded, rng=state.rng)
        metrics = adapter.evaluate(module)
        return Assessed(genome, metrics, self.fitness(genome, metrics), module)

    def evaluate_only(self, genome: Genome, adapter: Adapter) -> Assessed:
        """Score a genome WITHOUT training. Used to refresh fitness against a new task on a switch."""
        module = self._decode(genome, adapter)
        metrics = adapter.evaluate(module)
        return Assessed(genome, metrics, self.fitness(genome, metrics), module)

    @staticmethod
    def _decode(genome: Genome, adapter: Adapter) -> SubstrateModule:
        try:
            return adapter.decode(genome)
        except ValueError:
            # Recombination/toggling can produce a cyclic genome; repair to feedforward and retry.
            return adapter.decode(make_acyclic(genome))

    def seed_state(self, adapter: Adapter, rng: random.Random) -> EvolverState:
        """Build the initial population for `adapter`, score it, and return a fresh run state."""
        genomes = [self.init_op(adapter.n_inputs, adapter.n_outputs, rng=rng) for _ in range(self.pop_size)]
        state = EvolverState(population=[], innovations=InnovationTracker.from_genomes(genomes), rng=rng)
        state.population = [self.assess(genome, adapter, state) for genome in genomes]
        state.best = max(state.population, key=lambda item: item.fitness)
        self.species_history = state.species_history
        return state

    def advance(self, state: EvolverState, adapter: Adapter) -> None:
        """Produce the next generation in place (one select -> crossover -> mutate -> ... -> replace)."""
        state.population = self._next_generation(state.population, self._context(state), state, adapter)
        state.generation += 1
        self.species_history = state.species_history

    def run(
        self,
        adapter: TaskAdapter,
        generations: int,
        on_generation: GenerationHook | None = None,
        stop_at_accuracy: float = 1.0,
    ) -> Assessed:
        state = self.seed_state(adapter, random.Random(self.seed))

        for generation in range(generations):
            generation_best = max(state.population, key=lambda item: item.fitness)
            if state.best is None or generation_best.fitness > state.best.fitness:
                state.best = generation_best
            if on_generation is not None:
                mean_fitness = sum(item.fitness for item in state.population) / len(state.population)
                on_generation(generation, generation_best, mean_fitness)
            if generation_best.metrics.get("query_accuracy", 0.0) >= stop_at_accuracy:
                break
            self.advance(state, adapter)

        final_best = max(state.population, key=lambda item: item.fitness)
        best = state.best
        return final_best if best is None or final_best.fitness > best.fitness else best

    def _next_generation(
        self,
        assessed: list[Assessed],
        ctx: MutationContext,
        state: EvolverState,
        adapter: Adapter,
    ) -> list[Assessed]:
        genomes = [item.genome for item in assessed]
        fitnesses = [item.fitness for item in assessed]
        plans = self.speciate(genomes, fitnesses, rng=state.rng, elitism=self.elitism, pop_size=self.pop_size)
        state.species_history.append({plan.species_id: len(plan.members) for plan in plans})

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
            parents = self.selection_op(species_genomes, species_fitnesses, rng=state.rng, count=2 * plan.n_offspring)
            for k in range(plan.n_offspring):
                child = self.crossover_op(parents[2 * k], parents[2 * k + 1], rng=state.rng)
                child = self.mutation(child, ctx, rng=state.rng)
                next_assessed.append(self.assess(child, adapter, state))

        return next_assessed
