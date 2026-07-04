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
from functools import partial
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from multiprocessing.pool import Pool

    from ardevo.library import ModuleLibrary

from ardevo.dataset.icarus import EncodedTask, Level0Encoder
from ardevo.evaluation import evaluate
from ardevo.evolution.evaluate import standard as standard_evaluate
from ardevo.evolution.fitness import FitnessAggregator
from ardevo.evolution.genome import Genome, InnovationTracker, make_acyclic
from ardevo.evolution.mutation import MutationContext, MutationPipeline
from ardevo.evolution.speciation import SpeciesPlan
from ardevo.substrate import GraphNet, SubstrateModule, decode_module


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
        # A genome that evolved refine_steps > 1 decodes to the iterative-refinement substrate (the
        # same static input re-applied with state carried across passes); steps == 1 keeps the exact
        # feedforward path, so the flat search is unchanged until refinement is actually evolved.
        return decode_module(genome, self.n_inputs, self.n_outputs)

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
    # Metrics production is its own stage so weight-robustness scoring composes with any train op.
    evaluate_op: Callable[..., dict[str, float]] = standard_evaluate
    # Optional population-level trainer (e.g. gradient_batched): assess_many routes a whole
    # generation through one tensor program instead of candidate-by-candidate training.
    train_population_op: Callable[..., list[tuple[Genome, SubstrateModule]]] | None = None
    # N > 1 runs per-candidate assessment on a thread pool (only the sequential branch; the
    # population trainer is already batched). Candidates are independent and assessment never
    # draws from the shared rng, so results are order-preserving and stream-identical.
    parallel_assess: int = 0
    assess_workers: int = 0
    library_dir: str = "library"
    # The LIVE library handle for library-reading mutators (add_library_module / add_macro_node), so
    # they sample the SAME entries the decode-time macro resolver resolves. Left None on the pure
    # flat path; the orchestrator's direct strategy sets it to the attached library. Without this the
    # mutators fall back to a by-path cache that can diverge from the resolver and dangle a macro ref.
    library: "ModuleLibrary | None" = None
    # Mirror of the species history and of the latest batched-training stats, for trial logging.
    species_history: list[dict[int, int]] = field(default_factory=list)
    assess_stats: dict[str, float] = field(default_factory=dict)

    def _context(self, state: EvolverState) -> MutationContext:
        return MutationContext(innovations=state.innovations, activations=self.activations, default_activation=self.default_activation, library=self.library)

    def assess(self, genome: Genome, adapter: Adapter, state: EvolverState) -> Assessed:
        """Decode (repairing cycles), train the weights, evaluate, and score one genome."""
        module = self._decode(genome, adapter)
        genome, module = self.train_op(genome, module, adapter.encoded, rng=state.rng)
        metrics = self.evaluate_op(genome, module, adapter)
        return Assessed(genome, metrics, self.fitness(genome, metrics), module)

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
            if self.assess_workers > 1 and len(genomes) > 1:
                return self._assess_pooled(genomes, adapter)
            if self.parallel_assess <= 1 or len(genomes) <= 1:
                return [self.assess(genome, adapter, state) for genome in genomes]
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.parallel_assess) as pool:
                return list(pool.map(lambda genome: self.assess(genome, adapter, state), genomes))
        modules = [self._decode(genome, adapter) for genome in genomes]
        pairs = self.train_population_op(genomes, modules, adapter.encoded, rng=state.rng)
        from ardevo.evolution import train as train_stage

        self.assess_stats = dict(train_stage.last_batch_stats)
        assessed = []
        for genome, module in pairs:
            metrics = self.evaluate_op(genome, module, adapter)
            assessed.append(Assessed(genome, metrics, self.fitness(genome, metrics), module))
        return assessed

    def _assess_pooled(self, genomes: list[Genome], adapter: Adapter) -> list[Assessed]:
        """Assess independent genomes across a persistent process pool (true multi-core). Workers
        return (trained genome, metrics, fitness); the module is re-decoded here from the written-back
        genome (faithful and cheap, no retrain), so the returned Assessed matches the sequential path."""
        pool = self._ensure_pool()
        worker = partial(_assess_in_worker, adapter=adapter, train_op=self.train_op, evaluate_op=self.evaluate_op, fitness=self.fitness)
        chunksize = max(1, len(genomes) // (self.assess_workers * 4))
        results = pool.map(worker, genomes, chunksize=chunksize)
        return [Assessed(genome, metrics, fitness, self._decode(genome, adapter)) for genome, metrics, fitness in results]

    def _ensure_pool(self) -> "Pool":
        if _SHARED_POOL is not None:
            return _SHARED_POOL
        pool = getattr(self, "_pool", None)
        if pool is None:
            import atexit

            pool = _spawn_assess_pool(self.assess_workers, self.library_dir)
            self._pool = pool
            atexit.register(self.close_pool)
        return pool

    def close_pool(self) -> None:
        """Close only this evolver's OWN lazy pool; the shared pool is owned by create_assess_pool."""
        pool = getattr(self, "_pool", None)
        if pool is not None:
            self._pool = None
            pool.terminate()
            pool.join()

    @staticmethod
    def _decode(genome: Genome, adapter: Adapter) -> SubstrateModule:
        try:
            return adapter.decode(genome)
        except ValueError:
            # Recombination/toggling can produce a cyclic genome; repair to feedforward and retry.
            return adapter.decode(make_acyclic(genome))

    def seed_state(self, adapter: Adapter, rng: random.Random, *, seeded_front: Callable[[InnovationTracker], list[Genome]] | None = None) -> EvolverState:
        """Build the initial population for `adapter`, score it, and return a fresh run state.

        `seeded_front` (a callback because grafting needs the run's tracker, which is born here)
        replaces the FRONT of the init population with warm-start genomes; they flow through the
        same assess_many as every other member. None is byte-identical to the unseeded path."""
        genomes = [self.init_op(adapter.n_inputs, adapter.n_outputs, rng=rng) for _ in range(self.pop_size)]
        state = EvolverState(population=[], innovations=InnovationTracker.from_genomes(genomes), rng=rng)
        if seeded_front is not None:
            for index, genome in enumerate(seeded_front(state.innovations)[: self.pop_size]):
                genomes[index] = genome
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
            species_fitnesses = [item.fitness for item in members]
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


_WORKER_RNG = random.Random(0)
_SHARED_POOL: "Pool | None" = None


def _spawn_assess_pool(workers: int, library_dir: str) -> "Pool":
    import multiprocessing as mp

    context = mp.get_context("spawn")  # torch/Metal-safe; also avoids fork issues on the Linux queues
    n = min(workers, mp.cpu_count() or workers)
    return context.Pool(processes=n, initializer=_init_worker, initargs=(library_dir,))


def create_assess_pool(workers: int, library_dir: str) -> "Pool":
    """Build the shared assess pool once, at startup. Call this BEFORE clearml.Task.init so the workers
    spawn from an unpatched multiprocessing state and never attach to the ClearML task."""
    global _SHARED_POOL
    if _SHARED_POOL is None:
        import atexit

        _SHARED_POOL = _spawn_assess_pool(workers, library_dir)
        atexit.register(_close_shared_pool)
    return _SHARED_POOL


def set_shared_pool(pool: "Pool | None") -> None:
    global _SHARED_POOL
    _SHARED_POOL = pool


def get_shared_pool() -> "Pool | None":
    """The startup pool, if one was created. The hierarchical loop reuses it to parallelize its own
    composition assessment (same workers as the direct path)."""
    return _SHARED_POOL


def get_worker_library() -> "ModuleLibrary | None":
    """The worker-process ModuleLibrary set by _init_worker, for library: refs during composition
    assembly. On-disk entries resolve by key, so a library grown mid-run needs no reload."""
    return _WORKER_LIBRARY


def _close_shared_pool() -> None:
    global _SHARED_POOL
    if _SHARED_POOL is not None:
        pool, _SHARED_POOL = _SHARED_POOL, None
        pool.terminate()
        pool.join()


# Per-worker ModuleLibrary handle (set in _init_worker), reused by both the direct macro resolver and
# the hierarchical loop's composition assembly.
_WORKER_LIBRARY: "ModuleLibrary | None" = None


def _init_worker(library_dir: str) -> None:
    """Per-worker bootstrap for the process pool. First scrub ClearML's master-task env so a worker can
    never decide it is a ClearML subprocess and stall attaching to the server (workers are pure compute).
    Then pin torch to one intra-op thread (N workers x 1 thread map cleanly to N cores; the kernels are
    too small for intra-op threading to help) and open the on-disk library once (shared by the macro
    resolver and composition assembly)."""
    import os

    os.environ.pop("CLEARML_PROC_MASTER_ID", None)
    os.environ.pop("TRAINS_PROC_MASTER_ID", None)

    import torch

    torch.set_num_threads(1)
    from ardevo.library import ModuleLibrary, macro_resolver
    from ardevo.substrate import set_macro_resolver

    global _WORKER_LIBRARY
    _WORKER_LIBRARY = ModuleLibrary(library_dir)
    set_macro_resolver(macro_resolver(_WORKER_LIBRARY))


def _assess_in_worker(
    genome: Genome,
    *,
    adapter: Adapter,
    train_op: Callable[..., tuple[Genome, SubstrateModule]],
    evaluate_op: Callable[..., dict[str, float]],
    fitness: FitnessAggregator,
) -> tuple[Genome, dict[str, float], float]:
    """Decode, train, and evaluate one genome in a worker process. Returns the plain-data triple
    (trained genome, metrics, fitness); the main process re-decodes the module from the genome."""
    module = Evolver._decode(genome, adapter)
    genome, module = train_op(genome, module, adapter.encoded, rng=_WORKER_RNG)
    metrics = evaluate_op(genome, module, adapter)
    return genome, metrics, fitness(genome, metrics)
