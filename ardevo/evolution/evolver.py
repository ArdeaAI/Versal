"""Evolver: the thin generational loop that runs config-selected operators.

The Evolver owns the algorithm, not the task: genetic operators are injected (config-driven via
`build_evolver`), and an adapter injects the task-specific decode/evaluate. Each generation runs the
stages in order: select -> crossover -> mutate -> train -> evaluate -> fitness -> replace.

`seed_state` / `advance` step one generation at a time over an explicit, serializable
`EvolverState`, so a driver (the direct strategy, the hierarchical loop's module pool) can swap the
task adapter between generations, carry the population forward, and checkpoint/resume the run.
"""

import math
import random
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from multiprocessing.pool import Pool

    from ardevo.library import ModuleLibrary
    from ardevo.topology import TopologyTabuSession

from ardevo.dataset.icarus import Level0Encoder
from ardevo.evaluation import evaluate
from ardevo.evolution.evaluate import standard as standard_evaluate
from ardevo.evolution.fitness import FitnessAggregator, stamp_complexity_metrics
from ardevo.evolution.genome import Genome, InnovationTracker, make_acyclic
from ardevo.evolution.mutation import AdaptiveMutationPipeline, MutationContext, MutationPipeline
from ardevo.evolution.novelty import NoveltyConfig, archive_insert, compute_descriptor, novelty_scores, probe_tensor
from ardevo.evolution.selection import pareto_ranks_and_crowding, pareto_sort_key
from ardevo.evolution.speciation import SpeciesPlan
from ardevo.reference_depth import DEFAULT_MAX_INLINE_DEPTH
from ardevo.substrate import GraphNet, SubstrateModule, decode_module


class Adapter(Protocol):
    """What the Evolver needs of a task: encoded tensors, I/O widths, and decode + evaluate.

    `TaskAdapter` (static tasks) and `TemporalTaskAdapter` (TIME-axis tasks) both satisfy it, so
    the loop is agnostic to which one is active and can swap between them generation to generation
    while keeping the same population.
    """

    encoded: Any
    n_inputs: int
    n_outputs: int

    def decode(self, genome: Genome) -> SubstrateModule: ...

    def evaluate(self, module: SubstrateModule) -> dict[str, float]: ...


@dataclass
class TaskAdapter:
    """Injects task specifics (encoded tensors, widths) so the Evolver stays task-agnostic."""

    encoded: Any
    encoder: Level0Encoder
    n_inputs: int
    n_outputs: int
    # The raw spatial shape of the input Field when it IS a grid (set by the direct strategy from
    # the same probe that stamps geometry coordinates); None elsewhere. Descriptor axes carry no
    # per-axis lengths, so grid-aware evaluate ops (augmented_vote) read the shape from here.
    grid_shape: tuple[int, ...] | None = None
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH

    def decode(self, genome: Genome) -> GraphNet:
        # A genome that evolved refine_steps > 1 decodes to the iterative-refinement substrate (the
        # same static input re-applied with state carried across passes); steps == 1 keeps the exact
        # feedforward path, so the flat search is unchanged until refinement is actually evolved.
        return decode_module(genome, self.n_inputs, self.n_outputs, max_inline_depth=self.max_inline_depth)

    def evaluate(self, module: SubstrateModule) -> dict[str, float]:
        from ardevo.structured import StructuredGridEncoded, evaluate_structured_grid

        if isinstance(self.encoded, StructuredGridEncoded):
            return evaluate_structured_grid(module, self.encoded, self.encoder)
        return evaluate(module, self.encoded, self.encoder)


@dataclass(frozen=True)
class AdapterRef:
    """A by-path handle to a spilled adapter: what pool.map pickles instead of the encoded task
    tensors. Workers resolve it through a one-slot cache (`_resolve_adapter`), so each worker loads
    a task's tensors from disk exactly once, however many chunks it processes."""

    path: str


# A genome that cannot DECODE (macro nesting past the cap, a vanished macro ref) is a nonviable
# phenotype: it scores the floor and selection removes it. It must never kill the run; mutation is
# allowed to propose corpses, assessment just buries them.
_FLOOR_FITNESS = -1e9


def _floored_metrics() -> dict[str, float]:
    return {"support_accuracy": 0.0, "query_accuracy": 0.0, "support_loss": 1e9, "query_loss": 1e9, "weight_robustness": 0.0, "decode_failed": 1.0}


@dataclass
class Assessed:
    genome: Genome
    metrics: dict[str, float]
    fitness: float
    module: SubstrateModule | None  # the exact trained network that produced these metrics; None for a floored (undecodable) genome
    # Pareto objective vector, filled lazily at the top of _next_generation ONLY when
    # `[fitness] objectives` is configured; None on every scalar run (byte-identical path).
    objectives: list[float] | None = None
    # Behavioral descriptor (tanh outputs over the probe set), cached so a carried elite computes
    # it once for its lifetime; only its population-relative SCORE is recomputed each generation.
    descriptor: tuple[float, ...] | None = None


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
    # Rolling novelty archive ([evolution.novelty]): per-task by construction, since this state is
    # born fresh per solve and task attempts are checkpoint-atomic (no serialization needed).
    novelty_archive: list[tuple[float, ...]] = field(default_factory=list)
    # Transient refinement signal: no unseen offspring survived topology deduplication.
    topology_exhausted: bool = False


@dataclass
class Evolver:
    pop_size: int
    elitism: int
    seed: int
    init_op: Callable[..., Genome]
    selection_op: Callable[..., list[Genome]]
    crossover_op: Callable[..., Genome]
    mutation: MutationPipeline | AdaptiveMutationPipeline
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
    execution_mode: str = "serial"
    assess_workers: int = 0
    library_dir: str = "library"
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH
    # The LIVE library handle for library-reading mutators (add_library_module / add_macro_node), so
    # they sample the SAME entries the decode-time macro resolver resolves. Left None on the pure
    # flat path; the orchestrator's direct strategy sets it to the attached library. Without this the
    # mutators fall back to a by-path cache that can diverge from the resolver and dangle a macro ref.
    library: "ModuleLibrary | None" = None
    # Mirror of the species history and of the latest batched-training stats, for trial logging.
    species_history: list[dict[int, int]] = field(default_factory=list)
    assess_stats: dict[str, float] = field(default_factory=dict)
    # Behavioral-novelty config ([evolution.novelty]); None = the hook is a no-op, byte-identical.
    novelty: NoveltyConfig | None = None
    # SUCCESSIVE HALVING ([orchestrator.direct] halving_stages / halving_keep): cumulative step
    # fractions, e.g. [0.25, 1.0] = train everyone a quarter of the budget, keep the best
    # `halving_keep` by support loss for the rest. Losers keep their partial-training assessment
    # (honestly weaker, which IS the culling signal), so long per-candidate budgets (the 2-4k-step
    # flip-to-GO regime) stop paying full price on obvious losers. Stages compose with the assess
    # pool through the writeback/re-decode invariant; Adam state restarts at each stage boundary.
    # Empty (the default) = off, byte-identical.
    halving_stages: list[float] = field(default_factory=list)
    halving_keep: float = 0.5
    # Set only around post-solve refinement. Ordinary unsolved evolution leaves this None.
    topology_tabu: "TopologyTabuSession | None" = None

    def _context(self, state: EvolverState) -> MutationContext:
        return MutationContext(
            innovations=state.innovations,
            activations=self.activations,
            default_activation=self.default_activation,
            library=self.library,
            max_inline_depth=self.max_inline_depth,
        )

    def assess(self, genome: Genome, adapter: Adapter, state: EvolverState) -> Assessed:
        """Decode (repairing cycles), train the weights, evaluate, and score one genome."""
        try:
            module = self._decode(genome, adapter)
        except (ValueError, KeyError):
            return Assessed(genome, _floored_metrics(), _FLOOR_FITNESS, None)
        genome, module = self.train_op(genome, module, adapter.encoded, rng=state.rng)
        metrics = self.evaluate_op(genome, module, adapter)
        return Assessed(genome, metrics, self._score(genome, metrics), module)

    def evaluate_only(self, genome: Genome, adapter: Adapter) -> Assessed:
        """Score a genome WITHOUT training. Used to refresh fitness against a new task on a switch."""
        try:
            module = self._decode(genome, adapter)
        except (ValueError, KeyError):
            return Assessed(genome, _floored_metrics(), _FLOOR_FITNESS, None)
        metrics = self.evaluate_op(genome, module, adapter)
        return Assessed(genome, metrics, self._score(genome, metrics), module)

    def _score(self, genome: Genome, metrics: dict[str, float]) -> float:
        stamp_complexity_metrics(genome, metrics, self.library)
        return self.fitness(genome, metrics)

    def assess_many(self, genomes: list[Genome], adapter: Adapter, state: EvolverState) -> list[Assessed]:
        """Assess a batch of genomes, training them all in one tensor program when a population
        trainer is configured. Order-preserving, and rng-equivalent to the sequential path because
        train ops never draw from the shared rng (the contract documented in train.py)."""
        if self.train_population_op is None:
            if self.halving_stages and len(genomes) > 1:
                return self._assess_staged(genomes, adapter, state)
            if self.assess_workers > 1 and len(genomes) > 1:
                return self._assess_pooled(genomes, adapter)
            return [self.assess(genome, adapter, state) for genome in genomes]
        decoded: list[SubstrateModule | None] = []
        for genome in genomes:
            try:
                decoded.append(self._decode(genome, adapter))
            except (ValueError, KeyError):
                decoded.append(None)  # nonviable phenotype: floored below, never in the batch program
        if self.assess_workers > 1 and sum(module is not None for module in decoded) > 1:
            return self._assess_hybrid(genomes, decoded, adapter, state)
        viable = [(genome, module) for genome, module in zip(genomes, decoded) if module is not None]
        pairs = self.train_population_op([g for g, _m in viable], [m for _g, m in viable], adapter.encoded, rng=state.rng) if viable else []
        from ardevo.evolution import train as train_stage

        self.assess_stats = dict(train_stage.last_batch_stats)
        trained = iter(pairs)
        assessed = []
        for module in decoded:
            if module is None:
                assessed.append(Assessed(genomes[len(assessed)], _floored_metrics(), _FLOOR_FITNESS, None))
                continue
            genome, trained_module = next(trained)
            metrics = self.evaluate_op(genome, trained_module, adapter)
            assessed.append(Assessed(genome, metrics, self._score(genome, metrics), trained_module))
        return assessed

    def _assess_hybrid(self, genomes: list[Genome], decoded: list[SubstrateModule | None], adapter: Adapter, state: EvolverState) -> list[Assessed]:
        """Population trainer + process pool together: the batchable subset trains in ONE tensor
        program on the compute device (main process) WHILE the pool trains the serial subset
        (refine/recurrent/product/macro candidates, through the sequential same-kind op), then the
        batch subset's evaluation is farmed to the pool too. Same results as the inline population
        path: the partition comes from the SAME `partition_batchable` seam the trainer uses, the
        serial subset is exactly `_assess_in_worker` (the pool contract), and the batch subset's
        pooled evaluation re-decodes the written-back genome (exact: float32 round-trips through
        the genome losslessly, the invariant `_assess_pooled` already relies on)."""
        from ardevo.evolution import train as train_stage

        assert self.train_population_op is not None
        keywords = dict(getattr(self.train_population_op, "keywords", {}))
        steps = int(keywords.get("steps", 20))
        max_padded_nodes = int(keywords.get("max_padded_nodes", 1024))
        min_batch_nodes = int(keywords.get("min_batch_nodes", 0))
        writeback = bool(keywords.get("writeback", True))

        viable = [(index, module) for index, module in enumerate(decoded) if module is not None]
        viable_indices = [index for index, _module in viable]
        cores = [module.core() for _index, module in viable]
        batch_local, serial_local = train_stage.partition_batchable(cores, steps=steps, max_padded_nodes=max_padded_nodes, min_batch_nodes=min_batch_nodes)
        batch_indices = [viable_indices[i] for i in batch_local]
        serial_indices = [viable_indices[i] for i in serial_local]

        pool = self._ensure_pool()
        pooled_adapter = self._pooled_adapter(adapter)
        serial_async = None
        if serial_indices:
            worker = partial(_assess_in_worker, adapter=pooled_adapter, train_op=self.train_op, evaluate_op=self.evaluate_op, fitness=self.fitness)
            chunksize = max(1, len(serial_indices) // (self.assess_workers * 4))
            serial_async = pool.map_async(worker, [genomes[index] for index in serial_indices], chunksize=chunksize)

        results: list[Assessed | None] = [None] * len(genomes)
        for index, module in enumerate(decoded):
            if module is None:
                results[index] = Assessed(genomes[index], _floored_metrics(), _FLOOR_FITNESS, None)

        trained_pairs: list[tuple[Genome, SubstrateModule]] = []
        if batch_indices:
            trained_pairs = self.train_population_op([genomes[index] for index in batch_indices], [decoded[index] for index in batch_indices], adapter.encoded, rng=state.rng)
        self.assess_stats = dict(train_stage.last_batch_stats)
        if viable_indices:
            self.assess_stats["fallback"] = len(serial_indices) / len(viable_indices)

        if batch_indices and writeback:
            evaluator = partial(_evaluate_in_worker, adapter=pooled_adapter, evaluate_op=self.evaluate_op, fitness=self.fitness)
            chunksize = max(1, len(batch_indices) // (self.assess_workers * 4))
            eval_async = pool.map_async(evaluator, [genome for genome, _module in trained_pairs], chunksize=chunksize)
            for index, (evaluated, pair) in zip(batch_indices, zip(eval_async.get(), trained_pairs)):
                genome, metrics, fitness = evaluated
                results[index] = Assessed(genome, metrics, fitness, pair[1])
        elif batch_indices:
            # Without writeback the tuned weights exist ONLY on the trained modules, so a pooled
            # re-decode would score untrained weights; evaluate inline exactly like the batched path.
            for index, (genome, module) in zip(batch_indices, trained_pairs):
                metrics = self.evaluate_op(genome, module, adapter)
                results[index] = Assessed(genome, metrics, self._score(genome, metrics), module)

        if serial_async is not None:
            for index, (genome, metrics, fitness) in zip(serial_indices, serial_async.get()):
                module = None if metrics.get("decode_failed") else self._decode(genome, adapter)
                results[index] = Assessed(genome, metrics, fitness, module)
        return [item for item in results if item is not None]

    def _assess_staged(self, genomes: list[Genome], adapter: Adapter, state: EvolverState) -> list[Assessed]:
        """Successive-halving assessment: stage the per-candidate training budget across the
        population, culling the weakest-fitting half (by support loss) at each boundary. Stage k
        CONTINUES training from stage k-1's written-back weights (the same re-decode invariant the
        pool relies on), so survivors receive the full budget in total. rng-free like every train
        path (the staged op inherits the train contract)."""
        keywords = dict(getattr(self.train_op, "keywords", {}))
        total_steps = int(keywords.get("steps", 20))
        fractions = sorted({min(1.0, max(0.0, float(fraction))) for fraction in self.halving_stages} | {1.0})
        boundaries = [max(1, round(total_steps * fraction)) for fraction in fractions]
        deltas: list[int] = []
        previous = 0
        for boundary in boundaries:
            if boundary > previous:
                deltas.append(boundary - previous)
                previous = boundary

        pool = self._ensure_pool() if self.assess_workers > 1 else None
        pooled_adapter = self._pooled_adapter(adapter) if pool is not None else adapter
        results: list[tuple[Genome, dict[str, float], float] | None] = [None] * len(genomes)
        alive = list(range(len(genomes)))
        current = list(genomes)
        for stage_index, delta in enumerate(deltas):
            staged_op = partial(self.train_op, steps=delta)  # call-time kwargs override partial keywords
            worker = partial(_assess_in_worker, adapter=pooled_adapter, train_op=staged_op, evaluate_op=self.evaluate_op, fitness=self.fitness)
            if pool is not None:
                chunksize = max(1, len(alive) // (self.assess_workers * 4))
                triples = pool.map(worker, [current[index] for index in alive], chunksize=chunksize)
            else:
                triples = [worker(current[index]) for index in alive]
            for index, triple in zip(alive, triples):
                current[index] = triple[0]
                results[index] = triple
            if stage_index < len(deltas) - 1:

                def support_loss_of(index: int) -> float:
                    triple = results[index]
                    return float(triple[1].get("support_loss", float("inf"))) if triple is not None else float("inf")

                ranked = sorted(alive, key=lambda index: (support_loss_of(index), index))
                keep = max(1, math.ceil(len(alive) * max(0.0, min(1.0, self.halving_keep))))
                alive = ranked[:keep]
        return [
            Assessed(genome, metrics, fitness, None if metrics.get("decode_failed") else self._decode(genome, adapter))
            for genome, metrics, fitness in (triple for triple in results if triple is not None)
        ]

    def _assess_pooled(self, genomes: list[Genome], adapter: Adapter) -> list[Assessed]:
        """Assess independent genomes across a persistent process pool (true multi-core). Workers
        return (trained genome, metrics, fitness); the module is re-decoded here from the written-back
        genome (faithful and cheap, no retrain), so the returned Assessed matches the sequential path."""
        pool = self._ensure_pool()
        worker = partial(_assess_in_worker, adapter=self._pooled_adapter(adapter), train_op=self.train_op, evaluate_op=self.evaluate_op, fitness=self.fitness)
        chunksize = max(1, len(genomes) // (self.assess_workers * 4))
        results = pool.map(worker, genomes, chunksize=chunksize)
        return [Assessed(genome, metrics, fitness, None if metrics.get("decode_failed") else self._decode(genome, adapter)) for genome, metrics, fitness in results]

    def _pooled_adapter(self, adapter: Adapter) -> "Adapter | AdapterRef":
        """Spill the adapter to disk ONCE per task so pool.map pickles a tiny path per chunk
        instead of the encoded tensors every time (~4MB x chunks x generations for CIFAR).
        Content-addressed under `<library_dir>/encoded_cache/`, so a stale file is impossible;
        the previous task's spill is unlinked on replacement (its map calls have completed: assess
        is synchronous per generation). Any failure falls back to pickling the adapter directly."""
        slot: tuple[Adapter, str] | None = getattr(self, "_adapter_spill", None)
        if slot is not None and slot[0] is adapter:
            return AdapterRef(slot[1])
        try:
            import hashlib
            import io
            import os
            from pathlib import Path

            import torch

            buffer = io.BytesIO()
            torch.save(adapter, buffer)
            payload = buffer.getvalue()
            directory = Path(self.library_dir) / "encoded_cache"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{hashlib.sha256(payload).hexdigest()[:16]}.pt"
            if not path.exists():
                staging = path.with_name(f"{path.name}.tmp.{os.getpid()}")
                staging.write_bytes(payload)
                staging.replace(path)
            if slot is not None and slot[1] != str(path):
                Path(slot[1]).unlink(missing_ok=True)
            self._adapter_spill = (adapter, str(path))
            return AdapterRef(str(path))
        except Exception:  # pragma: no cover - spill is an optimization, never a failure mode
            return adapter

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
        slot: tuple[Adapter, str] | None = getattr(self, "_adapter_spill", None)
        if slot is not None:
            from pathlib import Path

            self._adapter_spill = None
            Path(slot[1]).unlink(missing_ok=True)
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
        if self.topology_tabu is not None:
            from ardevo.evolution.genome import genome_to_dict

            # One warm seed is the assessed anchor selection needs. Every other initial candidate
            # crosses the same pre-evaluation identity gate as offspring; minimal populations often
            # differ only in weights, so assessing all of them was the largest remaining repeat.
            filtered = genomes[:1]
            filtered.extend(genome for genome in genomes[1:] if self.topology_tabu.reserve("module", genome_to_dict(genome)))
            genomes = filtered
        state.population = self.assess_many(genomes, adapter, state)
        if self.topology_tabu is not None:
            for item in state.population:
                self.topology_tabu.observe_evaluated("module", genome_to_dict(item.genome))
            self.topology_tabu.commit()  # durable only after the batch finished assessment
        self._apply_novelty(state.population, state, adapter)
        state.best = max(state.population, key=lambda item: item.fitness)
        self.species_history = state.species_history
        return state

    def _apply_novelty(self, population: list[Assessed], state: EvolverState, adapter: Adapter) -> None:
        """Post-assess novelty pass: score every viable member against population + archive, inject
        `metrics["novelty"]`, and re-aggregate scalar fitness so the score exists BEFORE the next
        generation's speciation/selection reads it (parents and children alike). Runs at the end of
        seed_state and _next_generation; rng-free; a no-op (byte-identical) when unconfigured.

        Floored members (module None) are skipped entirely: no key, no re-aggregation, so the
        -1e9 corpse sentinel survives untouched."""
        if self.novelty is None:
            return
        probe = probe_tensor(adapter.encoded, self.novelty.probe_rows)
        if probe is None:
            return  # TIME-axis or degenerate task: novelty is out of scope, nothing changes
        for item in population:
            if item.descriptor is None and item.module is not None:
                item.descriptor = compute_descriptor(item.module, probe)
        scored = [item for item in population if item.descriptor is not None and item.module is not None]
        if not scored:
            return
        dimension = len(scored[0].descriptor or ())
        if any(len(entry) != dimension for entry in state.novelty_archive):
            state.novelty_archive.clear()  # an adapter swap changed the descriptor space mid-state
        descriptors = [item.descriptor for item in scored if item.descriptor is not None]
        scores = novelty_scores(descriptors, state.novelty_archive, self.novelty.k)
        for item, score in zip(scored, scores):
            item.metrics["novelty"] = score
            item.fitness = self.fitness(item.genome, item.metrics)
        most_novel = max(range(len(scored)), key=lambda index: scores[index])
        archive_insert(state.novelty_archive, descriptors[most_novel], self.novelty.archive_cap)

    def advance(self, state: EvolverState, adapter: Adapter) -> None:
        """Produce the next generation in place (one select -> crossover -> mutate -> ... -> replace)."""
        state.population = self._next_generation(state.population, self._context(state), state, adapter)
        state.generation += 1
        self.species_history = state.species_history

    def _next_generation(
        self,
        assessed: list[Assessed],
        ctx: MutationContext,
        state: EvolverState,
        adapter: Adapter,
    ) -> list[Assessed]:
        from ardevo.evolution.genome import genome_to_dict

        state.topology_exhausted = False
        genomes = [item.genome for item in assessed]
        fitnesses = [item.fitness for item in assessed]
        # Pareto mode: vectors are (re)computed here, not at the Assessed construction sites, so the
        # worker-pool triple contract stays untouched and population-relative metrics injected after
        # the previous assess (novelty) are reflected. Speciation budgets stay on the scalar sum;
        # Pareto replaces ordering only INSIDE each species. A corpse gets the floor vector: its
        # tiny graph would otherwise win the wiring-cost axis and put undecodable genomes on front 0.
        pareto = bool(self.fitness.objective_components)
        if pareto:
            floor_vector = [_FLOOR_FITNESS] * len(self.fitness.objective_components)
            for item in assessed:
                item.objectives = floor_vector.copy() if item.fitness <= _FLOOR_FITNESS else self.fitness.objectives(item.genome, item.metrics)
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
            if pareto:
                pool = [assessed[index] for index in plan.members]
                ranks, crowding = pareto_ranks_and_crowding([item.objectives or [] for item in pool])
                order = pareto_sort_key(ranks, crowding, [item.fitness for item in pool])
                members = [pool[index] for index in sorted(range(len(pool)), key=order)]
            else:
                members = sorted((assessed[index] for index in plan.members), key=lambda item: item.fitness, reverse=True)
            next_assessed.extend(members[: plan.n_elites])
            if plan.n_offspring <= 0:
                continue

            species_genomes = [item.genome for item in members]
            species_fitnesses = [item.fitness for item in members]
            if pareto:
                parents = self.selection_op(species_genomes, species_fitnesses, rng=state.rng, count=2 * plan.n_offspring, objectives=[item.objectives for item in members])
            else:
                parents = self.selection_op(species_genomes, species_fitnesses, rng=state.rng, count=2 * plan.n_offspring)
            for k in range(plan.n_offspring):
                child: Genome | None = None
                attempts = self.topology_tabu.retry_limit + 1 if self.topology_tabu is not None else 1
                for _attempt in range(attempts):
                    candidate = self.crossover_op(parents[2 * k], parents[2 * k + 1], rng=state.rng)
                    candidate = self.mutation(candidate, ctx, rng=state.rng)
                    if self.topology_tabu is None or self.topology_tabu.reserve("module", genome_to_dict(candidate)):
                        child = candidate
                        break
                if child is None:
                    assert self.topology_tabu is not None
                    self.topology_tabu.retry_exhaustions += 1
                    next_assessed.append(members[k % len(members)])
                    continue
                child_slots.append(len(next_assessed))
                next_assessed.append(None)
                children.append(child)

        if self.topology_tabu is not None and child_slots == [] and any(plan.n_offspring > 0 for plan in plans):
            state.topology_exhausted = True
            self.topology_tabu.exhausted = True
        assessed_children = self.assess_many(children, adapter, state) if children else []
        if self.topology_tabu is not None:
            self.topology_tabu.commit()  # a later crash must not cause this generation to repeat
        for slot, item in zip(child_slots, assessed_children):
            next_assessed[slot] = item
        next_population = [item for item in next_assessed if item is not None]
        self._apply_novelty(next_population, state, adapter)
        return next_population


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


# One-slot adapter cache per worker: tasks are processed sequentially, so a worker only ever needs
# the CURRENT task's tensors; a growing cache would hold every visited task's encodings in memory.
_WORKER_ADAPTER: tuple[str, Adapter] | None = None


def _resolve_adapter(adapter: "Adapter | AdapterRef") -> Adapter:
    if not isinstance(adapter, AdapterRef):
        return adapter
    global _WORKER_ADAPTER
    if _WORKER_ADAPTER is None or _WORKER_ADAPTER[0] != adapter.path:
        import torch

        # weights_only=False is required (the spill holds adapter dataclasses, not bare tensors)
        # and safe: the path is only ever produced by this run's own `_pooled_adapter` spill into
        # its library dir, never taken from external input.
        _WORKER_ADAPTER = (adapter.path, torch.load(adapter.path, map_location="cpu", weights_only=False))
    return _WORKER_ADAPTER[1]


def _assess_in_worker(
    genome: Genome,
    *,
    adapter: "Adapter | AdapterRef",
    train_op: Callable[..., tuple[Genome, SubstrateModule]],
    evaluate_op: Callable[..., dict[str, float]],
    fitness: FitnessAggregator,
) -> tuple[Genome, dict[str, float], float]:
    """Decode, train, and evaluate one genome in a worker process. Returns the plain-data triple
    (trained genome, metrics, fitness); the main process re-decodes the module from the genome.
    An undecodable genome floors here instead of raising: a worker exception would kill the WHOLE
    pool.map and with it the run (the two_spirals macro-nesting crash of 2026-07-04)."""
    adapter = _resolve_adapter(adapter)
    try:
        module = Evolver._decode(genome, adapter)
    except (ValueError, KeyError):
        return genome, _floored_metrics(), _FLOOR_FITNESS
    genome, module = train_op(genome, module, adapter.encoded, rng=_WORKER_RNG)
    metrics = evaluate_op(genome, module, adapter)
    stamp_complexity_metrics(genome, metrics, _WORKER_LIBRARY)
    return genome, metrics, fitness(genome, metrics)


def _evaluate_in_worker(
    genome: Genome,
    *,
    adapter: "Adapter | AdapterRef",
    evaluate_op: Callable[..., dict[str, float]],
    fitness: FitnessAggregator,
) -> tuple[Genome, dict[str, float], float]:
    """Evaluate one ALREADY-TRAINED (written-back) genome in a worker process: the hybrid assess
    path's batch subset. Decoding the written-back genome reproduces the trained module exactly
    (float32 weights round-trip through the genome losslessly). Floors instead of raising for the
    same pool-safety reason as `_assess_in_worker`."""
    adapter = _resolve_adapter(adapter)
    try:
        module = Evolver._decode(genome, adapter)
    except (ValueError, KeyError):
        return genome, _floored_metrics(), _FLOOR_FITNESS
    metrics = evaluate_op(genome, module, adapter)
    stamp_complexity_metrics(genome, metrics, _WORKER_LIBRARY)
    return genome, metrics, fitness(genome, metrics)
