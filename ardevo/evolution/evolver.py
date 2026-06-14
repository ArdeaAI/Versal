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
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from ardevo.library import ModuleLibrary

from ardevo.dataset.icarus import EncodedTask, Level0Encoder
from ardevo.evaluation import behavior_descriptor, evaluate, restrict_support, support_fold
from ardevo.evolution.evaluate import standard as standard_evaluate
from ardevo.evolution.fitness import FitnessAggregator
from ardevo.evolution.genome import Genome, InnovationTracker, make_acyclic
from ardevo.evolution.mutation import MutationContext, MutationPipeline
from ardevo.evolution.novelty import NoveltyConfig
from ardevo.evolution.speciation import SpeciesPlan
from ardevo.substrate import GraphNet, SubstrateModule, decode_equilibrium, decode_module


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
    """Injects task specifics (encoded tensors, widths) so the Evolver stays task-agnostic.

    `validation_fraction > 0` carves an INNER fold out of the support set: the trainer fits only
    `train_encoded` (the inner-train rows) and evaluate scores `holdout_encoded` as support_holdout_*,
    a leakage-free generalization signal that never touches the real query/accept set. 0.0 leaves
    `train_encoded == encoded` and `holdout_encoded` None, so the path is byte-identical."""

    encoded: EncodedTask
    encoder: Level0Encoder
    n_inputs: int
    n_outputs: int
    validation_fraction: float = 0.0
    # Phase 7 (Pillar D): equilibrium decode params (tol/damping/max_iters). When set, a genome with
    # recurrent edges decodes to the fixed-point substrate (test-time-tunable depth) instead of the
    # gene-fixed refine substrate; None keeps the decode_module routing (byte-identical).
    equilibrium: dict[str, Any] | None = None
    train_encoded: EncodedTask = field(init=False)
    holdout_encoded: EncodedTask | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        inner_train, holdout = support_fold(self.encoded, self.validation_fraction)
        if holdout:
            self.train_encoded = restrict_support(self.encoded, inner_train)
            self.holdout_encoded = restrict_support(self.encoded, holdout)
        else:
            self.train_encoded = self.encoded

    def decode(self, genome: Genome) -> GraphNet:
        # With equilibrium on, ANY genome carrying a recurrent edge iterates to a fixed point (depth
        # severed from refine_steps). Otherwise: refine_steps > 1 decodes the iterative-refinement
        # substrate and steps == 1 keeps the exact feedforward path, so the flat search is unchanged.
        if self.equilibrium is not None and genome.recurrent_connections():
            return decode_equilibrium(genome, self.n_inputs, self.n_outputs, **self.equilibrium)
        return decode_module(genome, self.n_inputs, self.n_outputs)

    def evaluate(self, module: SubstrateModule) -> dict[str, float]:
        return evaluate(module, self.train_encoded, self.encoder, holdout=self.holdout_encoded)


@dataclass
class Assessed:
    genome: Genome
    metrics: dict[str, float]
    fitness: float
    module: SubstrateModule  # the exact trained network that produced these metrics (for faithful saving)
    # Functional behavior fingerprint + its sparseness, populated only when novelty selection is on.
    # `effective_fitness` is the fitness/novelty blend that drives speciation and parent selection;
    # `fitness` (true) still governs elitism and champion tracking. Default to fitness via the loop.
    behavior: tuple[float, ...] = ()
    novelty: float = 0.0
    effective_fitness: float = 0.0


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
    # Behavior archive for novelty search. Per-evolve (per-task) scope: a fresh seed_state starts it
    # empty, so it is ephemeral and needs no cross-task checkpointing.
    novelty_archive: list[tuple[float, ...]] = field(default_factory=list)


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
    # Metrics production is its own stage so weight-robustness scoring composes with any train op.
    evaluate_op: Callable[..., dict[str, float]] = standard_evaluate
    # Optional population-level trainer (e.g. gradient_batched): assess_many routes a whole
    # generation through one tensor program instead of candidate-by-candidate training.
    train_population_op: Callable[..., list[tuple[Genome, SubstrateModule]]] | None = None
    # N > 1 runs per-candidate assessment on a thread pool (only the sequential branch; the
    # population trainer is already batched). Candidates are independent and assessment never
    # draws from the shared rng, so results are order-preserving and stream-identical.
    parallel_assess: int = 0
    # The LIVE library handle for library-reading mutators (add_library_module / add_macro_node), so
    # they sample the SAME entries the decode-time macro resolver resolves. Left None on the pure
    # flat path; the orchestrator's direct strategy sets it to the attached library. Without this the
    # mutators fall back to a by-path cache that can diverge from the resolver and dangle a macro ref.
    library: "ModuleLibrary | None" = None
    # Functional novelty / quality-diversity selection (config [evolution.novelty]). None or disabled
    # keeps the pure-fitness path byte-identical.
    novelty: NoveltyConfig | None = None
    # Mirror of the species history and of the latest batched-training stats, for trial logging.
    species_history: list[dict[int, int]] = field(default_factory=list)
    assess_stats: dict[str, float] = field(default_factory=dict)

    def _context(self, state: EvolverState) -> MutationContext:
        return MutationContext(innovations=state.innovations, activations=self.activations, default_activation=self.default_activation, library=self.library)

    def _behavior(self, module: SubstrateModule, adapter: Adapter) -> tuple[float, ...]:
        """Functional fingerprint for novelty selection; empty (skipping the extra forward) when off."""
        if not (self.novelty and self.novelty.enabled):
            return ()
        return behavior_descriptor(module, adapter.encoded, max_dim=self.novelty.descriptor_dim)

    @staticmethod
    def _train_encoded(adapter: Adapter) -> EncodedTask:
        """The task view the trainer fits on: the inner-train fold when the adapter carries one (the
        generalization-fold path), else the full encoded task. Non-folding adapters (multitask /
        temporal) lack the attribute and stay byte-identical."""
        return getattr(adapter, "train_encoded", adapter.encoded)

    def assess(self, genome: Genome, adapter: Adapter, state: EvolverState) -> Assessed:
        """Decode (repairing cycles), train the weights, evaluate, and score one genome."""
        module = self._decode(genome, adapter)
        genome, module = self.train_op(genome, module, self._train_encoded(adapter), rng=state.rng)
        metrics = self.evaluate_op(genome, module, adapter)
        return Assessed(genome, metrics, self.fitness(genome, metrics), module, behavior=self._behavior(module, adapter))

    def evaluate_only(self, genome: Genome, adapter: Adapter) -> Assessed:
        """Score a genome WITHOUT training. Used to refresh fitness against a new task on a switch."""
        module = self._decode(genome, adapter)
        metrics = self.evaluate_op(genome, module, adapter)
        return Assessed(genome, metrics, self.fitness(genome, metrics), module)

    def assess_many(self, genomes: list[Genome], adapter: Adapter, state: EvolverState) -> list[Assessed]:
        """Assess a batch of genomes, training them all in one tensor program when a population
        trainer is configured. Order-preserving, and rng-equivalent to the sequential path because
        train ops never draw from the shared rng (the contract documented in train.py)."""
        if self.train_population_op is None:
            if self.parallel_assess <= 1 or len(genomes) <= 1:
                return [self.assess(genome, adapter, state) for genome in genomes]
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.parallel_assess) as pool:
                return list(pool.map(lambda genome: self.assess(genome, adapter, state), genomes))
        modules = [self._decode(genome, adapter) for genome in genomes]
        pairs = self.train_population_op(genomes, modules, self._train_encoded(adapter), rng=state.rng)
        from ardevo.evolution import train as train_stage

        self.assess_stats = dict(train_stage.last_batch_stats)
        assessed = []
        for genome, module in pairs:
            metrics = self.evaluate_op(genome, module, adapter)
            assessed.append(Assessed(genome, metrics, self.fitness(genome, metrics), module, behavior=self._behavior(module, adapter)))
        return assessed

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
        state.population = self.assess_many(genomes, adapter, state)
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

    def _novelty_score(self, assessed: list[Assessed], fitnesses: list[float], state: EvolverState) -> Callable[[Assessed], float]:
        """Return the per-genome score speciation and selection use this generation. When novelty is
        on, compute each genome's behavioral sparseness, blend it with fitness into effective_fitness,
        and grow the per-evolve behavior archive; the returned getter reads effective_fitness. When
        off, the getter reads true fitness, so the caller is numerically unchanged."""
        if not (self.novelty and self.novelty.enabled):
            return lambda item: item.fitness
        from ardevo.evolution import novelty as novelty_mod

        behaviors = [item.behavior for item in assessed]
        novelties = novelty_mod.novelty_scores(behaviors, state.novelty_archive, self.novelty.k)
        effective = novelty_mod.blend(fitnesses, novelties, self.novelty.weight)
        for item, novelty_value, effective_value in zip(assessed, novelties, effective, strict=True):
            item.novelty = novelty_value
            item.effective_fitness = effective_value
        novelty_mod.update_archive(state.novelty_archive, behaviors, novelties, rng=state.rng, archive_max=self.novelty.archive_max, add_prob=self.novelty.add_prob)
        return lambda item: item.effective_fitness

    def _next_generation(
        self,
        assessed: list[Assessed],
        ctx: MutationContext,
        state: EvolverState,
        adapter: Adapter,
    ) -> list[Assessed]:
        genomes = [item.genome for item in assessed]
        fitnesses = [item.fitness for item in assessed]
        # When novelty is on, speciation and parent selection run on an effective fitness that blends
        # true fitness with behavioral novelty (the deceptive-landscape escape); elitism below still
        # uses TRUE fitness so a novel-but-useless genome can never become an elite. When off, `score`
        # is true fitness and this loop is byte-identical to the pure-fitness path.
        score = self._novelty_score(assessed, fitnesses, state)
        plans = self.speciate(genomes, [score(item) for item in assessed], rng=state.rng, elitism=self.elitism, pop_size=self.pop_size)
        state.species_history.append({plan.species_id: len(plan.members) for plan in plans})

        # Elites are carried forward UNCHANGED (re-training champions every generation overfits the
        # support set: champion drift); offspring genomes are collected first so a configured
        # population trainer can assess the whole brood in one batch. Deferring is rng-safe: the
        # select/crossover/mutate draws all happen here, and train ops never touch the rng.
        next_assessed: list[Assessed | None] = []
        children: list[Genome] = []
        child_slots: list[int] = []
        for plan in plans:
            members = sorted((assessed[index] for index in plan.members), key=lambda item: item.fitness, reverse=True)
            next_assessed.extend(members[: plan.n_elites])
            if plan.n_offspring <= 0:
                continue

            species_genomes = [item.genome for item in members]
            species_fitnesses = [score(item) for item in members]
            parents = self.selection_op(species_genomes, species_fitnesses, rng=state.rng, count=2 * plan.n_offspring)
            for k in range(plan.n_offspring):
                child = self.crossover_op(parents[2 * k], parents[2 * k + 1], rng=state.rng)
                child = self.mutation(child, ctx, rng=state.rng)
                child_slots.append(len(next_assessed))
                next_assessed.append(None)
                children.append(child)

        for slot, item in zip(child_slots, self.assess_many(children, adapter, state)):
            next_assessed[slot] = item
        return [item for item in next_assessed if item is not None]
