"""Loops: the top-level generational strategy, selectable from config like any other stage.

`hierarchical` is the recursive composition loop: a per-task population of `CompositionGenome`s
co-evolves with ONE shared live module population. Compositions are assembled (live refs resolve
to species champions, library refs
inline admitted entries), trained (glue + unfrozen live inner copies), evaluated, and scored;
fitness then flows DOWN as attribution to the module species and library entries each composition
referenced. Modules reproduce with the existing flat-stage operators every `advance_every`
composition generations, using attribution as their fitness.

Ownership is the forgetting fix: compositions and their glue are per task, the module pool is
shared, library entries are frozen. Champion-only module writeback keeps one task's gradient from
silently shredding a module other tasks rely on.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, cast

from ardevo.dataset.icarus import Level0Encoder
from ardevo.evaluation import evaluate
from ardevo.evolution.composition import (
    AssemblyContext,
    CompMutationContext,
    CompMutationPipeline,
    ComposedNet,
    CompositionAssemblyError,
    CompositionGenome,
    RefSpec,
    assemble,
    comp_to_dict,
    minimal_composition,
    writeback_composition,
)
from ardevo.evolution.evolver import AdapterRef, Evolver, _resolve_adapter, get_shared_pool, get_worker_library
from ardevo.evolution.fitness import stamp_complexity_metrics
from ardevo.evolution.genome import Genome, InnovationTracker, genome_from_dict, genome_to_dict, make_acyclic
from ardevo.evolution.mutation import MutationContext
from ardevo.evolution.registry import Registry, _bind_prefixed, build_evolver
from ardevo.evolution.train import _writeback
from ardevo.library import MODULE, ModuleLibrary
from ardevo.reference_depth import configured_max_inline_depth
from ardevo.utils.device import resolve_compute_device
from ardevo.utils.logging import Logger
from ardevo.utils.resources import ResourceEstimate, ResourcePolicy

logger = Logger.get_logger()

LOOP: Registry = Registry("loop")

FLOOR_FITNESS = -1e9


@LOOP.register("hierarchical")
def _build_hierarchical_entry(config: dict[str, Any]) -> "HierarchicalLoop":
    return build_hierarchical(config)


@dataclass
class LiveModule:
    """One member of the shared module population. Fitness is ATTRIBUTED, never measured directly:
    a module is as good as the compositions that use it."""

    genome: Genome
    fitness: float = 0.0


@dataclass
class HierarchicalState:
    """The durable, cross-task state of the hierarchical loop (per-task comp populations are not
    kept: accepted solutions live in the library, which is the point)."""

    modules: list[LiveModule]
    module_innovations: InnovationTracker
    comp_innovations: InnovationTracker
    rng: random.Random
    generation: int = 0
    species_members: dict[int, list[int]] = field(default_factory=dict)
    species_champion_index: dict[int, int] = field(default_factory=dict)
    species_champions: dict[int, Genome] = field(default_factory=dict)
    module_species_history: list[dict[int, int]] = field(default_factory=list)
    repaired_refs: int = 0
    absorbed_keys: list[str] = field(default_factory=list)  # library entries already grafted into the pool
    topology_exhausted: bool = False  # transient; reset for each composition refinement


@dataclass
class AssessedComposition:
    comp: CompositionGenome
    metrics: dict[str, float]
    fitness: float
    net: ComposedNet | None  # None for an assembly failure or a compact non-champion pooled result
    # Process workers must not return ``ComposedNet``: torch's Linux multiprocessing reducer
    # transports every tensor storage through a shared-memory file descriptor.  A composition
    # generation can contain hundreds of storages and exhaust the parent's descriptor limit before
    # the old population is released.  Workers instead return these plain Genome writebacks; the
    # main process applies the winning set and reconstructs only the generation champion's net.
    # None means assembly failed or this assessment ran in the main process; an empty dict is a
    # successfully assembled pooled composition with no live-module references.
    live_writebacks: dict[str, Genome] | None = None


@dataclass(frozen=True)
class CompTaskSpec:
    """Everything the loop needs to evolve compositions against one task."""

    encoded: Any
    encoder: Level0Encoder
    n_inputs: int
    input_specs: list[tuple[str, int]]  # (bank signature, width) per INPUT node
    bank_columns: dict[str, Sequence[int]]
    output_ref: str
    output_width: int
    # The task's full structural I/O contract (library.task_io), computed once per solve so the
    # orchestrator's lookup/wall/admission paths never re-derive it. None-tolerant for callers that
    # build a spec by hand (tests, benches).
    io: dict[str, Any] | None = None


# Throwaway rng for worker-side training. Train ops ignore rng (the train.py contract forbids
# call-order-dependent draws precisely so assessment can be reordered/parallelized), so a fixed
# instance keeps pooled results bit-identical to the sequential path.
_COMP_WORKER_RNG = random.Random(0)


def assess_composition_pure(
    comp: CompositionGenome,
    spec: CompTaskSpec,
    species_champions: dict[int, Genome],
    library: ModuleLibrary | None,
    max_inline_depth: int,
    *,
    train: bool,
    train_op: Callable[..., Any],
    evaluate_op: Callable[..., dict[str, float]],
    fitness: Callable[..., float],
    rng: random.Random,
) -> AssessedComposition:
    """Assemble, optionally train, evaluate, and score ONE composition. Pure w.r.t. shared state: it
    reads only the passed champions + library, decodes its OWN inner-module copies (fresh context), and
    draws no caller rng. This is the single seam the sequential path and the process-pool workers both
    call, so they cannot diverge."""

    def live_resolver(species_id: str) -> Genome:
        genome = species_champions.get(int(species_id))
        if genome is None:
            raise CompositionAssemblyError(f"live species {species_id} no longer exists")
        return genome

    ctx = AssemblyContext(bank_columns=dict(spec.bank_columns), live_resolver=live_resolver, library=library, max_inline_depth=max_inline_depth)
    try:
        net = assemble(comp, ctx, spec.n_inputs)
    except CompositionAssemblyError as error:
        logger.debug("composition floored at assembly: %s", error)
        return AssessedComposition(comp=comp, metrics={}, fitness=FLOOR_FITNESS, net=None)
    if train:
        # writeback=False: the op returns the SAME comp/net; glue lands on the comp genes here and
        # module weights flow via the champion policy (in _module_writeback), not the flat writeback.
        comp, net = cast(tuple[CompositionGenome, ComposedNet], train_op(comp, net, spec.encoded, rng=rng, writeback=False))
        comp = writeback_composition(comp, net)
    metrics = evaluate_op(comp, net, _CompositionEvalAdapter(spec))
    stamp_complexity_metrics(comp, metrics, library)
    if library is not None:
        from ardevo.library import MODULE, expanded_payload_complexity

        live_cost = 0
        for node_id in comp.module_ids:
            reference = comp.nodes[node_id].ref
            if not reference.startswith("live:"):
                continue
            genome = species_champions.get(int(reference.removeprefix("live:")))
            if genome is not None:
                live_cost += expanded_payload_complexity(MODULE, genome_to_dict(genome), library)
        metrics["expanded_complexity"] += float(live_cost)
    return AssessedComposition(comp=comp, metrics=metrics, fitness=fitness(comp, metrics), net=net)


def _assess_comp_in_worker(
    comp: CompositionGenome,
    *,
    spec: CompTaskSpec | AdapterRef,
    species_champions: dict[int, Genome],
    max_inline_depth: int,
    train: bool,
    train_op: Callable[..., Any],
    evaluate_op: Callable[..., dict[str, float]],
    fitness: Callable[..., float],
) -> AssessedComposition:
    """Assess one composition and return an fd-safe, tensor-free result.

    Glue is already copied into ``CompositionGenome`` by ``assess_composition_pure``.  Live inner
    modules need one additional Lamarckian writeback, which is compacted into ordinary Genome
    dataclasses here.  Returning a trained ``ComposedNet`` would make torch send every parameter and
    buffer through ``SCM_RIGHTS`` on Linux and eventually exhaust the parent's open-file limit.
    """
    spec = _resolve_adapter(spec)
    assessed = assess_composition_pure(
        comp,
        spec,
        species_champions,
        get_worker_library(),
        max_inline_depth,
        train=train,
        train_op=train_op,
        evaluate_op=evaluate_op,
        fitness=fitness,
        rng=_COMP_WORKER_RNG,
    )
    if assessed.net is None:
        return assessed
    live_writebacks: dict[str, Genome] = {}
    for ref, inner in assessed.net.inner_modules.items():
        if not ref.startswith("live:"):
            continue
        champion = species_champions.get(int(ref.removeprefix("live:")))
        if champion is not None:
            live_writebacks[ref] = _writeback(champion, inner)
    return AssessedComposition(
        comp=assessed.comp,
        metrics=assessed.metrics,
        fitness=assessed.fitness,
        net=None,
        live_writebacks=live_writebacks,
    )


@dataclass
class HierarchicalLoop:
    evolver: Evolver  # supplies the bound module-stage operators + train/evaluate/fitness stages
    comp_pop_size: int
    comp_elitism: int
    # Selection ops come from the shared SELECTION registry (genome-type agnostic), so the list
    # element type stays loose here.
    comp_selection_op: Callable[..., list[Any]]
    comp_crossover_op: Callable[..., CompositionGenome]
    comp_crossover_rate: float
    comp_mutation: CompMutationPipeline
    max_inline_depth: int
    glue_scale: float | None
    glue_rank: int
    glue_rank_threshold: int
    glue_storage: str
    resource_policy: ResourcePolicy
    resource_device: str
    # -1 exposes every library entry to comp mutations (glue adapts any widths); >= 0 keeps only
    # entries within that many columns of the task's I/O or the module port shape, which focuses
    # the catalog once the library grows large.
    catalog_width_tolerance: int
    module_pop_size: int
    module_elitism: int
    in_ports: int
    out_ports: int
    advance_every: int
    writeback_mode: str  # "champion" | "none"
    attribution_mode: str  # "max" | "mean"
    decay: float
    offspring_discount: float
    seed_fraction: float
    # Hard pre-allocation guard for tuple-backed composition genes. Zero disables it. The estimate
    # uses the exact dense/factored formula and runs before population construction.
    max_initial_glue_values: int = 0
    # At each lookup miss the orchestrator absorbs up to this many NEW exact-port library entries
    # into the module pool (grafted over the worst non-champion members): found structures become
    # building blocks IN the soup, mid-run, not just at fresh_state.
    absorb_top_k: int = 0
    library: ModuleLibrary | None = None
    # Set only around post-solve composition refinement.
    topology_tabu: Any | None = None

    def attach_library(self, library: ModuleLibrary | None) -> None:
        self.library = library

    def assess_glue_resources(
        self,
        glue_values: int,
        *,
        stage: str,
        population_multiplicity: int = 1,
        concurrent_trainers: int = 1,
        storage: str | None = None,
        fixed_limit: int | None = None,
        device: str | None = None,
    ) -> ResourceEstimate:
        return self.resource_policy.assess_glue(
            glue_values,
            stage=stage,
            storage=storage or self.glue_storage,
            device=device or self.resource_device,
            fixed_limit=self.max_initial_glue_values if fixed_limit is None else fixed_limit,
            population_multiplicity=population_multiplicity,
            concurrent_trainers=concurrent_trainers,
        )

    # --- state ------------------------------------------------------------------------------------

    def fresh_state(self, rng: random.Random) -> HierarchicalState:
        # Seed minimal genomes first so the tracker baseline is well-defined, then graft library
        # modules (matching port shape) over the front of the population: cross-run reuse.
        genomes = [self.evolver.init_op(self.in_ports, self.out_ports, rng=rng) for _ in range(self.module_pop_size)]
        tracker = InnovationTracker.from_genomes(genomes)
        if self.library is not None and self.seed_fraction > 0.0:
            from ardevo.library import graft

            wanted = int(self.seed_fraction * self.module_pop_size)
            entries = [
                entry
                for entry in self.library.query(entry_type=MODULE, input_width=self.in_ports, output_width=self.out_ports)
                if self.library.reference_subtree_depth(entry.key) <= self.max_inline_depth
            ][:wanted]
            for index, entry in enumerate(entries):
                genomes[index] = graft(entry, tracker)
        modules = [LiveModule(genome=genome) for genome in genomes]
        state = HierarchicalState(modules=modules, module_innovations=tracker, comp_innovations=InnovationTracker(_next_node_id=0), rng=rng)
        self._speciate_only(state)
        return state

    def _speciate_only(self, state: HierarchicalState) -> None:
        """Refresh species membership/champions WITHOUT reproducing (used at seed and restore)."""
        genomes = [module.genome for module in state.modules]
        fitnesses = [module.fitness for module in state.modules]
        plans = self.evolver.speciate(genomes, fitnesses, rng=state.rng, elitism=self.module_elitism, pop_size=self.module_pop_size)
        state.module_species_history.append({plan.species_id: len(plan.members) for plan in plans})
        state.species_members = {plan.species_id: list(plan.members) for plan in plans}
        state.species_champion_index = {}
        state.species_champions = {}
        for plan in plans:
            champion = max(plan.members, key=lambda index: fitnesses[index])
            state.species_champion_index[plan.species_id] = champion
            state.species_champions[plan.species_id] = state.modules[champion].genome

    # --- ref plumbing -------------------------------------------------------------------------------

    def ref_catalog(self, state: HierarchicalState, spec: CompTaskSpec | None = None) -> list[RefSpec]:
        catalog = [RefSpec(f"live:{species_id}", self.in_ports, self.out_ports) for species_id in sorted(state.species_champions)]
        if self.library is not None:
            tolerance = self.catalog_width_tolerance
            for entry in self.library.query():
                # Adding this entry as a composition MODULE follows one new library edge, leaving
                # max_inline_depth - 1 levels for the entry's own mixed module/composition subtree.
                if self.library.reference_subtree_depth(entry.key) > self.max_inline_depth - 1:
                    continue
                in_width = sum(item["width"] for item in entry.io["inputs"])
                out_width = int(entry.io["output"]["width"])
                if tolerance >= 0 and spec is not None:
                    in_ok = abs(in_width - spec.n_inputs) <= tolerance or abs(in_width - self.in_ports) <= tolerance
                    out_ok = abs(out_width - spec.output_width) <= tolerance or abs(out_width - self.out_ports) <= tolerance
                    if not (in_ok and out_ok):
                        continue
                catalog.append(RefSpec(f"library:{entry.key}", in_width, out_width))
        return catalog

    def _repair_refs(self, comp: CompositionGenome, state: HierarchicalState) -> CompositionGenome:
        """Re-point MODULE nodes whose live species died to a random living species (logged)."""
        from dataclasses import replace

        live = sorted(state.species_champions)
        if not live:
            return comp
        child = None
        for node_id in comp.module_ids:
            node = comp.nodes[node_id]
            if node.ref.startswith("live:") and int(node.ref.removeprefix("live:")) not in state.species_champions:
                if child is None:
                    child = comp.clone()
                replacement = live[state.rng.randrange(len(live))]
                child.nodes[node_id] = replace(node, ref=f"live:{replacement}")
                state.repaired_refs += 1
        return comp if child is None else child

    # --- assessment ---------------------------------------------------------------------------------

    def assess_composition(self, comp: CompositionGenome, spec: CompTaskSpec, state: HierarchicalState, *, train: bool) -> AssessedComposition:
        """Public assessment seam: the orchestrator's strategies verify champions through it
        (fresh assembly against CURRENT champions/library) before threshold checks and admission."""
        return self._assess(comp, spec, state, train=train)

    def _assess(self, comp: CompositionGenome, spec: CompTaskSpec, state: HierarchicalState, *, train: bool) -> AssessedComposition:
        return assess_composition_pure(
            comp,
            spec,
            state.species_champions,
            self.library,
            self.max_inline_depth,
            train=train,
            train_op=self.evolver.train_op,
            evaluate_op=self.evolver.evaluate_op,
            fitness=self.evolver.fitness,
            rng=state.rng,
        )

    def _assess_all(self, comps: list[CompositionGenome], spec: CompTaskSpec, state: HierarchicalState, *, train: bool) -> list[AssessedComposition]:
        """Assess a batch of candidates, order-preserving. Prefer the shared process pool (true
        multi-core, same workers as the direct path) when present; else the sequential list.

        Results are identical to the serial loop: candidates share no trainable state, assessment
        consumes no rng (a throwaway rng in workers is safe), and floored candidates (assembly errors)
        are produced INSIDE the pure assessor. Pooled results carry trained glue and live-module
        writebacks as plain genes; no torch storage crosses the process boundary."""
        pool = get_shared_pool()
        if pool is not None and len(comps) > 1:
            # The encoded task can itself contain large torch tensors. Spill it once and let each
            # worker's one-slot payload cache resolve the path; binding the full spec into every
            # map chunk would recreate the same fragile torch fd transport on the input side.
            pooled_spec = self.evolver._pooled_adapter(spec)
            if not isinstance(pooled_spec, AdapterRef):
                logger.warning("composition task payload could not be spilled; assessing this batch in the main process")
                return [self._assess(comp, spec, state, train=train) for comp in comps]
            worker = partial(
                _assess_comp_in_worker,
                spec=pooled_spec,
                species_champions=state.species_champions,
                max_inline_depth=self.max_inline_depth,
                train=train,
                train_op=self.evolver.train_op,
                evaluate_op=self.evolver.evaluate_op,
                fitness=self.evolver.fitness,
            )
            chunksize = max(1, len(comps) // (4 * (getattr(pool, "_processes", 12) or 12)))
            return pool.map(worker, comps, chunksize=chunksize)
        return [self._assess(comp, spec, state, train=train) for comp in comps]

    # --- attribution and writeback --------------------------------------------------------------------

    def _attribute(self, assessed: list[AssessedComposition], state: HierarchicalState) -> None:
        per_ref: dict[str, list[float]] = {}
        for item in assessed:
            if item.fitness <= FLOOR_FITNESS:
                continue
            for ref in set(item.comp.refs()):
                per_ref.setdefault(ref, []).append(item.fitness)
        attribution = {ref: (max(values) if self.attribution_mode == "max" else sum(values) / len(values)) for ref, values in per_ref.items()}

        referenced_species: set[int] = set()
        for ref, value in attribution.items():
            if ref.startswith("live:"):
                species_id = int(ref.removeprefix("live:"))
                if species_id in state.species_members:
                    referenced_species.add(species_id)
                    champion = state.species_champion_index.get(species_id)
                    for index in state.species_members[species_id]:
                        if index == champion:
                            state.modules[index].fitness = value
                        # A non-champion member of an ACTIVELY REFERENCED species is a live stepping
                        # stone (exploration around a useful champion); leave its fitness intact rather
                        # than decaying it toward neutral as if its species were unused. Only truly
                        # unreferenced species decay (the loop below). Preserves structural diversity.
            elif ref.startswith("library:") and self.library is not None:
                self.library.bump_stats(ref.removeprefix("library:"), value)
        for species_id, members in state.species_members.items():
            if species_id not in referenced_species:
                for index in members:
                    state.modules[index].fitness = self._decayed(state.modules[index].fitness)

    def _decayed(self, fitness: float) -> float:
        """Decay erodes stale POSITIVE credit only. Multiplying a NEGATIVE score by decay would
        move it toward neutral, i.e. REWARD a module for going unreferenced while compositions
        score badly; bad scores must persist until real evidence replaces them."""
        return fitness * self.decay if fitness > 0.0 else fitness

    def _module_writeback(self, assessed: list[AssessedComposition], state: HierarchicalState) -> None:
        """Champion-only Lamarckianism: ONLY the generation's best composition writes its trained
        live-module weights back, so one bad gradient run cannot shred a shared module."""
        if self.writeback_mode != "champion":
            return
        best = max(assessed, key=lambda item: item.fitness)
        if best.fitness <= FLOOR_FITNESS:
            return
        if best.live_writebacks is not None:
            for ref, updated in best.live_writebacks.items():
                species_id = int(ref.removeprefix("live:"))
                if species_id not in state.species_champions:
                    continue
                state.species_champions[species_id] = updated
                champion_index = state.species_champion_index.get(species_id)
                if champion_index is not None and champion_index < len(state.modules):
                    state.modules[champion_index].genome = updated
            return
        if best.net is None:
            return
        for ref, inner in best.net.inner_modules.items():
            if not ref.startswith("live:"):
                continue
            species_id = int(ref.removeprefix("live:"))
            champion_genome = state.species_champions.get(species_id)
            if champion_genome is None:
                continue
            updated = _writeback(champion_genome, inner)
            state.species_champions[species_id] = updated
            champion_index = state.species_champion_index.get(species_id)
            if champion_index is not None and champion_index < len(state.modules):
                state.modules[champion_index].genome = updated

    def _restore_champion_net(self, assessed: AssessedComposition, spec: CompTaskSpec, state: HierarchicalState) -> None:
        """Rebuild only a pooled generation champion in the main process.

        The returned genes contain the exact trained glue and live-module weights that produced the
        worker metrics, so assembly restores the executable network without another optimization or
        evaluation pass.  Keeping one such net preserves the strategy/verification API while the
        rest of the population remains compact and descriptor-free.
        """
        if assessed.net is not None or assessed.live_writebacks is None:
            return
        live_writebacks = assessed.live_writebacks

        def live_resolver(species_id: str) -> Genome:
            ref = f"live:{species_id}"
            genome = live_writebacks.get(ref)
            if genome is None:
                genome = state.species_champions.get(int(species_id))
            if genome is None:
                raise CompositionAssemblyError(f"live species {species_id} no longer exists")
            return genome

        ctx = AssemblyContext(
            bank_columns=dict(spec.bank_columns),
            live_resolver=live_resolver,
            library=self.library,
            max_inline_depth=self.max_inline_depth,
        )
        try:
            assessed.net = assemble(assessed.comp, ctx, spec.n_inputs)
        except CompositionAssemblyError as error:
            logger.debug("pooled champion could not be reconstructed: %s", error)

    # --- reproduction -----------------------------------------------------------------------------------

    def _module_context(self, state: HierarchicalState) -> MutationContext:
        return MutationContext(
            innovations=state.module_innovations,
            activations=self.evolver.activations,
            default_activation=self.evolver.default_activation,
            library=self.library,
            max_inline_depth=self.max_inline_depth,
        )

    def advance_modules(self, state: HierarchicalState) -> None:
        """One module generation: speciate on attributed fitness, then reproduce per species with
        the SAME registered operators the flat loop uses."""
        genomes = [module.genome for module in state.modules]
        fitnesses = [module.fitness for module in state.modules]
        plans = self.evolver.speciate(genomes, fitnesses, rng=state.rng, elitism=self.module_elitism, pop_size=self.module_pop_size)
        state.module_species_history.append({plan.species_id: len(plan.members) for plan in plans})

        ctx = self._module_context(state)
        next_modules: list[LiveModule] = []
        members_map: dict[int, list[int]] = {}
        champion_index: dict[int, int] = {}
        champions: dict[int, Genome] = {}
        for plan in plans:
            members = sorted(plan.members, key=lambda index: fitnesses[index], reverse=True)
            champions[plan.species_id] = state.modules[members[0]].genome
            species_fitness = fitnesses[members[0]]
            new_indices: list[int] = []
            for index in members[: plan.n_elites]:
                new_indices.append(len(next_modules))
                next_modules.append(state.modules[index])
            if new_indices:
                champion_index[plan.species_id] = new_indices[0]
            if plan.n_offspring > 0:
                species_genomes = [genomes[index] for index in members]
                species_fitnesses = [fitnesses[index] for index in members]
                parents = self.evolver.selection_op(species_genomes, species_fitnesses, rng=state.rng, count=2 * plan.n_offspring)
                for k in range(plan.n_offspring):
                    child = self.evolver.crossover_op(parents[2 * k], parents[2 * k + 1], rng=state.rng)
                    child = self.evolver.mutation(child, ctx, rng=state.rng)
                    child = make_acyclic(child)
                    new_indices.append(len(next_modules))
                    # Offspring inherit a discounted species fitness until a composition references them.
                    next_modules.append(LiveModule(genome=child, fitness=species_fitness * self.offspring_discount))
            members_map[plan.species_id] = new_indices
        state.modules = next_modules
        state.species_members = members_map
        state.species_champion_index = champion_index
        state.species_champions = champions

    def absorb_new_entries(self, state: HierarchicalState) -> int:
        """Graft the best not-yet-absorbed exact-port MODULE entries over the worst non-champion
        members (exact ports only: live decode hard-requires matching widths). Newcomers start at
        the pool mean fitness: neutral enough to survive one selection round without dominating."""
        if self.library is None or self.absorb_top_k <= 0 or not state.modules:
            return 0
        from ardevo.library import graft

        ranked = [
            entry
            for entry in self.library.query(entry_type=MODULE, input_width=self.in_ports, output_width=self.out_ports)
            if entry.key not in state.absorbed_keys and self.library.reference_subtree_depth(entry.key) <= self.max_inline_depth
        ]
        # Graft BEHAVIORALLY DIVERSE entries first: one per unseen niche by rank, then fill. A pool of
        # varied building blocks recombines into more than a cluster of near-duplicate top-metric ones.
        seen_niches: set[tuple[str, ...]] = set()
        primary: list[Any] = []
        secondary: list[Any] = []
        for entry in ranked:
            niche = tuple(entry.provenance.get("behavior", []))
            (primary if niche not in seen_niches else secondary).append(entry)
            seen_niches.add(niche)
        candidates = (primary + secondary)[: self.absorb_top_k]
        if not candidates:
            return 0
        protected = set(state.species_champion_index.values())
        replaceable = sorted((index for index in range(len(state.modules)) if index not in protected), key=lambda index: state.modules[index].fitness)
        mean_fitness = sum(module.fitness for module in state.modules) / len(state.modules)
        absorbed = 0
        for entry, index in zip(candidates, replaceable):
            state.modules[index] = LiveModule(genome=graft(entry, state.module_innovations), fitness=mean_fitness)
            state.absorbed_keys.append(entry.key)
            absorbed += 1
        if absorbed:
            logger.info("absorbed %d library entries into the module pool", absorbed)
            self._speciate_only(state)
        return absorbed

    def _reproduce_comps(self, assessed: list[AssessedComposition], spec: CompTaskSpec, state: HierarchicalState) -> list[AssessedComposition]:
        ordered = sorted(assessed, key=lambda item: item.fitness, reverse=True)
        # Elites keep their genes but are RE-EVALUATED (no gradient, no drift): their live refs
        # resolve against modules that moved since last generation, so cached fitness goes stale.
        # Repair dead live refs FIRST: module species die during advance_modules, and an elite
        # pointing at one would otherwise be silently floored (champion lineage lost).
        elite_items = ordered[: self.comp_elitism]
        elites = [self._repair_refs(item.comp, state) for item in elite_items]
        n_offspring = max(self.comp_pop_size - len(elites), 0)
        # ALL rng draws happen up front (selection, crossover, mutation, repair); assessment is
        # rng-free, so deferring it into one (possibly parallel) batch is stream-identical.
        children: list[CompositionGenome] = []
        carried: list[AssessedComposition] = []
        if n_offspring > 0:
            comps = [item.comp for item in ordered]
            fitnesses = [item.fitness for item in ordered]
            parents = self.comp_selection_op(comps, fitnesses, rng=state.rng, count=2 * n_offspring)
            comp_ctx = CompMutationContext(
                innovations=state.comp_innovations,
                ref_catalog=self.ref_catalog(state),
                glue_rank=self.glue_rank,
                glue_rank_threshold=self.glue_rank_threshold,
                glue_storage=self.glue_storage,
            )
            for k in range(n_offspring):
                child: CompositionGenome | None = None
                attempts = self.topology_tabu.retry_limit + 1 if self.topology_tabu is not None else 1
                for _attempt in range(attempts):
                    if state.rng.random() < self.comp_crossover_rate:
                        candidate = self.comp_crossover_op(parents[2 * k], parents[2 * k + 1], rng=state.rng)
                    else:
                        candidate = parents[2 * k].clone()
                    candidate = self._repair_refs(self.comp_mutation(candidate, comp_ctx, rng=state.rng), state)
                    if self.topology_tabu is None or self.topology_tabu.reserve("composition", comp_to_dict(candidate)):
                        child = candidate
                        break
                if child is None:
                    assert self.topology_tabu is not None
                    self.topology_tabu.retry_exhaustions += 1
                    parent = parents[2 * k]
                    carried.append(next((item for item in assessed if item.comp is parent), ordered[0]))
                    continue
                children.append(child)
        if self.topology_tabu is not None and not children and n_offspring > 0:
            state.topology_exhausted = True
            self.topology_tabu.exhausted = True
        assessed_elites = elite_items if self.topology_tabu is not None else self._assess_all(elites, spec, state, train=False)
        assessed_children = self._assess_all(children, spec, state, train=True)
        if self.topology_tabu is not None:
            self.topology_tabu.commit()  # durable only after every selected child was assessed
        return assessed_elites + carried + assessed_children

    # --- the per-task drive ---------------------------------------------------------------------------

    def run_task(
        self,
        spec: CompTaskSpec,
        state: HierarchicalState,
        *,
        budget: int,
        stop: Callable[[int, AssessedComposition], bool] | None = None,
        seed_comps: list[CompositionGenome] | None = None,
        on_generation: Callable[[int, AssessedComposition, float], None] | None = None,
    ) -> AssessedComposition:
        """Evolve a composition population against one task for up to `budget` generations."""
        state.topology_exhausted = False
        population: list[CompositionGenome] = [comp.clone() for comp in (seed_comps or [])][: self.comp_pop_size]
        while len(population) < self.comp_pop_size:
            population.append(
                minimal_composition(
                    spec.input_specs,
                    spec.output_ref,
                    spec.output_width,
                    state.comp_innovations,
                    state.rng,
                    glue_scale=self.glue_scale,
                    glue_rank=self.glue_rank,
                    glue_rank_threshold=self.glue_rank_threshold,
                    glue_storage=self.glue_storage,
                )
            )
        population = [self._repair_refs(comp, state) for comp in population]

        if self.topology_tabu is not None:
            # Keep one warm incumbent as the parent/fitness anchor, then reject repeated initial
            # structures before glue fitting. Minimal compositions otherwise spend a full batch on
            # the same graph with different random glue values.
            filtered = population[:1]
            filtered.extend(comp for comp in population[1:] if self.topology_tabu.reserve("composition", comp_to_dict(comp)))
            population = filtered
        assessed = self._assess_all(population, spec, state, train=True)
        if self.topology_tabu is not None:
            for item in assessed:
                self.topology_tabu.observe_evaluated("composition", comp_to_dict(item.comp))
            self.topology_tabu.commit()
        self._attribute(assessed, state)
        self._module_writeback(assessed, state)
        best = max(assessed, key=lambda item: item.fitness)
        self._restore_champion_net(best, spec, state)

        for generation in range(budget):
            if stop is not None and stop(generation, best):
                break
            if self.advance_every > 0 and generation % self.advance_every == self.advance_every - 1:
                self.advance_modules(state)
            assessed = self._reproduce_comps(assessed, spec, state)
            if state.topology_exhausted:
                break
            self._attribute(assessed, state)
            self._module_writeback(assessed, state)
            generation_best = max(assessed, key=lambda item: item.fitness)
            self._restore_champion_net(generation_best, spec, state)
            if generation_best.fitness > best.fitness:
                best = generation_best
            if on_generation is not None:
                mean = sum(item.fitness for item in assessed) / len(assessed)
                on_generation(generation, generation_best, mean)
            state.generation += 1
        return best


class _CompositionEvalAdapter:
    """The minimal Adapter surface the evaluate stage needs (`evaluate(module)`); decode is owned
    by the loop because it requires per-candidate assembly contexts."""

    def __init__(self, spec: CompTaskSpec) -> None:
        self.encoded = spec.encoded
        self.encoder = spec.encoder
        self.n_inputs = spec.n_inputs
        self.n_outputs = spec.output_width

    def evaluate(self, module: ComposedNet) -> dict[str, float]:
        return evaluate(module, self.encoded, self.encoder)


# --- state serialization ------------------------------------------------------------------------------


def state_to_dict(state: HierarchicalState) -> dict[str, Any]:
    return {
        "generation": state.generation,
        "modules": [{"genome": genome_to_dict(module.genome), "fitness": module.fitness} for module in state.modules],
        "module_innovations": state.module_innovations.to_dict(),
        "comp_innovations": state.comp_innovations.to_dict(),
        "species_members": {str(species_id): members for species_id, members in state.species_members.items()},
        "species_champion_index": {str(species_id): index for species_id, index in state.species_champion_index.items()},
        "species_champions": {str(species_id): genome_to_dict(genome) for species_id, genome in state.species_champions.items()},
        "module_species_history": [{str(species_id): count for species_id, count in snapshot.items()} for snapshot in state.module_species_history],
        "repaired_refs": state.repaired_refs,
        "absorbed_keys": list(state.absorbed_keys),
    }


def state_from_dict(data: dict[str, Any], rng: random.Random) -> HierarchicalState:
    return HierarchicalState(
        modules=[LiveModule(genome=genome_from_dict(item["genome"]), fitness=float(item["fitness"])) for item in data["modules"]],
        module_innovations=InnovationTracker.from_dict(data["module_innovations"]),
        comp_innovations=InnovationTracker.from_dict(data["comp_innovations"]),
        rng=rng,
        generation=int(data["generation"]),
        species_members={int(species_id): [int(index) for index in members] for species_id, members in data["species_members"].items()},
        species_champion_index={int(species_id): int(index) for species_id, index in data["species_champion_index"].items()},
        species_champions={int(species_id): genome_from_dict(genome) for species_id, genome in data["species_champions"].items()},
        module_species_history=[{int(species_id): int(count) for species_id, count in snapshot.items()} for snapshot in data["module_species_history"]],
        repaired_refs=int(data.get("repaired_refs", 0)),
        absorbed_keys=[str(key) for key in data.get("absorbed_keys", [])],
    )


# --- factory ---------------------------------------------------------------------------------------------


def build_hierarchical(config: dict[str, Any]) -> HierarchicalLoop:
    from ardevo.evolution import composition, selection

    evolution = config.get("evolution", {})
    comp_cfg = evolution.get("composition", {})
    modules_cfg = evolution.get("modules", {})
    glue_storage = str(comp_cfg.get("glue_storage", "tuple"))
    if glue_storage not in {"tuple", "f32"}:
        raise ValueError(f"unknown composition glue_storage {glue_storage!r}; expected 'tuple' or 'f32'")

    comp_selection_cfg = comp_cfg.get("selection", {})
    comp_selection_op = partial(
        selection.SELECTION.get(comp_selection_cfg.get("kind", "tournament")),
        **{k: v for k, v in comp_selection_cfg.items() if k != "kind"},
    )
    comp_crossover_cfg = comp_cfg.get("crossover", {})
    comp_crossover_op = partial(
        composition.COMP_CROSSOVER.get(comp_crossover_cfg.get("kind", "none")),
        **{k: v for k, v in comp_crossover_cfg.items() if k not in ("kind", "rate")},
    )
    comp_mutation_cfg = comp_cfg.get("mutation", {})
    mutators: list[Callable[..., CompositionGenome]] = [
        partial(composition.COMP_MUTATION.get(name), **_bind_prefixed(comp_mutation_cfg, name)) for name in comp_mutation_cfg.get("operators", [])
    ]

    return HierarchicalLoop(
        evolver=build_evolver(config),
        comp_pop_size=int(comp_cfg.get("pop_size", 16)),
        comp_elitism=int(comp_cfg.get("elitism", 2)),
        comp_selection_op=comp_selection_op,
        comp_crossover_op=comp_crossover_op,
        comp_crossover_rate=float(comp_crossover_cfg.get("rate", 0.2)),
        comp_mutation=CompMutationPipeline(mutators),
        max_inline_depth=configured_max_inline_depth(config),
        glue_scale=float(comp_cfg["glue_scale"]) if "glue_scale" in comp_cfg else None,
        glue_rank=int(comp_cfg.get("glue_rank", 0)),
        glue_rank_threshold=int(comp_cfg.get("glue_rank_threshold", 0)),
        glue_storage=glue_storage,
        resource_policy=ResourcePolicy.from_config(config.get("resources")),
        resource_device=str(resolve_compute_device(config)),
        catalog_width_tolerance=int(comp_cfg.get("catalog_width_tolerance", -1)),
        module_pop_size=int(modules_cfg.get("pop_size", evolution.get("pop_size", 32))),
        module_elitism=int(modules_cfg.get("elitism", evolution.get("elitism", 1))),
        in_ports=int(modules_cfg.get("in_ports", 4)),
        out_ports=int(modules_cfg.get("out_ports", 2)),
        advance_every=int(modules_cfg.get("advance_every", 3)),
        writeback_mode=str(modules_cfg.get("writeback", "champion")),
        attribution_mode=str(modules_cfg.get("attribution", "max")),
        decay=float(modules_cfg.get("decay", 0.95)),
        offspring_discount=float(modules_cfg.get("offspring_discount", 0.9)),
        seed_fraction=float(modules_cfg.get("seed_fraction", 0.25)),
        max_initial_glue_values=max(0, int(comp_cfg.get("max_initial_glue_values", 0))),
        absorb_top_k=int(modules_cfg.get("absorb_top_k", 0)),
    )
