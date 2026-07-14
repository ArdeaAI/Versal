"""Evolve strategies: HOW the orchestrator's evolve step searches, selectable from config.

The ladder's step 2 used to be hardcoded to the hierarchical composition loop. It is now a
config-ordered list of registered strategies (`[orchestrator] evolve = ["routed", "grammar", "direct", "composition"]`)
sharing one depth budget: execution is config order, first strategy to clear the accept threshold
wins, and a stalled strategy's unspent generations roll into the next allocation.

`composition` wraps the hierarchical loop and owns champion VERIFICATION: `run_task` returns the
best-of-any-generation, whose inner weights can predate later module advances and writebacks. The
champion is re-assessed against CURRENT state before the threshold check, so what gets admitted is
always the fresh net and the fresh metric, never a stale chimera.

`direct` is the proven phase-1 recipe on the task's REAL I/O widths: flat genomes, structural
mutation, gradient-owned weights, and (for TIME-axis tasks) the stepped recurrent substrate, which
is what finally makes recurrent genes execute in the orchestrated path. Its champions are admitted
as task-shaped MODULE entries the composition strategy can immediately reference.
"""

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable

from ardevo.dataset.icarus import Level0Encoder, Task, encode_task, model_output_features, support_loader
from ardevo.evaluation import fit_query_target, input_width, output_features, without_query
from ardevo.evolution.composition import CompositionGenome, edge_storage_value_count, glue_value_count
from ardevo.evolution.evolver import Assessed, Evolver, TaskAdapter, get_shared_pool
from ardevo.evolution.genome import Genome, InnovationTracker, genome_to_dict
from ardevo.evolution.loop import AssessedComposition, CompTaskSpec, HierarchicalLoop, HierarchicalState
from ardevo.evolution.registry import Registry, build_evolver
from ardevo.library import MODULE, LibraryEntry, ModuleLibrary, graft, structural_fingerprint
from ardevo.reference_depth import DEFAULT_MAX_INLINE_DEPTH
from ardevo.temporal import TemporalTaskAdapter, has_time_axis, temporal_adapter
from ardevo.utils.logging import Logger
from ardevo.utils.resources import format_bytes

logger = Logger.get_logger()

EVOLVE_STRATEGY: Registry = Registry("evolve_strategy")


@dataclass
class StrategyRuntime:
    """Everything a strategy needs from the orchestrator, bundled so signatures stay stable."""

    loop: HierarchicalLoop
    library: ModuleLibrary
    state: HierarchicalState
    accept_threshold: float
    metric_of: Callable[[Any], float]  # reads .metrics; works for Assessed and AssessedComposition
    stall_factory: Callable[[int], Callable[[int, Any], bool]]
    on_generation: Callable[[str, int, Any, float], None] | None = None  # (strategy, gen, best, mean)
    accepts: Callable[[Any], bool] | None = None
    deadline_exceeded: Callable[[], bool] | None = None

    def accepted(self, item: Any) -> bool:
        return self.accepts(item) if self.accepts is not None else self.metric_of(item) >= self.accept_threshold

    def should_stop(self) -> bool:
        return self.deadline_exceeded is not None and self.deadline_exceeded()


@dataclass
class StrategyResult:
    strategy: str
    metric: float
    generations_used: int
    champion_comp: AssessedComposition | None = None  # composition-shaped winner (verified fresh)
    champion_genome: Genome | None = None  # module-shaped winner (trained weights written back)
    champion_metrics: dict[str, float] = field(default_factory=dict)
    # Held-out metrics are kept on a separate rail. They are reporting only and must never affect
    # admission, refinement, robustness ranking, or the next task's search state.
    report_metrics: dict[str, float] = field(default_factory=dict)
    # A routed winner is a RECORD (ardevo.routing.RoutedSolution), not an admissible payload: the
    # executable state lives in the persisted router. Typed Any to keep strategy free of a routing import.
    champion_routed: Any | None = None
    # Refine-on-hit fairness: the best metric the grafted seed itself reached under THIS run's
    # training. The refine comparator uses it as the incumbent baseline, so a candidate must beat
    # the incumbent given the same training, not just beat its untrained quick-eval score.
    seed_metric: float | None = None
    # Champion/population size stats: the always-on bloat readout. Task cost tracks genome size,
    # so growth must be visible in run_summary rows, not only through `seconds` (the diag_g2
    # free-growth arm hit hour-scale tasks with zero size signal in any record).
    size_metrics: dict[str, float] = field(default_factory=dict)
    # Pre-allocation estimates are operational evidence, kept separate from task/quality metrics.
    resource_metrics: dict[str, float] = field(default_factory=dict)
    # Cross-strategy diagnostics (especially routed distillation and executable handoff).
    strategy_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def has_admissible_champion(self) -> bool:
        """Whether this result carries an executable payload that can satisfy the solve contract."""

        return self.champion_comp is not None or self.champion_genome is not None or self.champion_routed is not None


def _module_size_metrics(champion: Genome, population: list[Assessed]) -> dict[str, float]:
    """Champion plus final-population genome size. The population medians are what show a
    free-growth run inflating task over task; the champion scalars are what admission shelves."""
    metrics = {
        "champion_nodes": float(len(champion.nodes)),
        "champion_connections": float(len(champion.enabled_connections())),
        "champion_complexity": float(champion.complexity()),
    }
    if population:
        node_counts = sorted(len(member.genome.nodes) for member in population)
        connection_counts = sorted(len(member.genome.enabled_connections()) for member in population)
        metrics["pop_median_nodes"] = float(node_counts[len(node_counts) // 2])
        metrics["pop_max_nodes"] = float(node_counts[-1])
        metrics["pop_median_connections"] = float(connection_counts[len(connection_counts) // 2])
        metrics["pop_max_connections"] = float(connection_counts[-1])
    return metrics


def comp_size_metrics(comp: Any) -> dict[str, float]:
    """Composition-shaped champions: module count plus glue complexity (inner genome cost is
    priced at the module layer, not here). Public because the routed strategy stamps the same
    keys on a distilled win."""
    return {"champion_modules": float(len(comp.module_ids)), "champion_complexity": float(comp.complexity())}


def _restamp_genome(source: Genome, tracker: InnovationTracker) -> Genome:
    """Move a grammar seed into the receiving run's innovation namespace."""

    id_map = {node_id: tracker.new_node_id() for node_id in sorted(source.nodes)}
    nodes = {id_map[node_id]: replace(node, id=id_map[node_id]) for node_id, node in source.nodes.items()}
    groups = {group for conn in source.connections if (group := conn.tie_group) is not None}
    tie_groups = {group: tracker.new_marker() for group in sorted(groups)}
    connections = [
        replace(
            conn,
            in_id=id_map[conn.in_id],
            out_id=id_map[conn.out_id],
            innovation=tracker.innovation(id_map[conn.in_id], id_map[conn.out_id], conn.recurrent),
            tie_group=tie_groups[conn.tie_group] if conn.tie_group is not None else None,
        )
        for conn in source.connections
    ]
    macros = [
        replace(
            macro,
            input_node_ids=tuple(id_map[node_id] for node_id in macro.input_node_ids),
            output_node_ids=tuple(id_map[node_id] for node_id in macro.output_node_ids),
            innovation=tracker.new_marker(),
        )
        for macro in source.macros
    ]
    return Genome(nodes, connections, macros, source.refine_steps, dict(source.operator_rates))


def _restamp_composition(source: CompositionGenome, tracker: InnovationTracker) -> CompositionGenome:
    id_map = {node_id: tracker.new_node_id() for node_id in sorted(source.nodes)}
    nodes = {id_map[node_id]: replace(node, id=id_map[node_id]) for node_id, node in source.nodes.items()}
    edges = [replace(edge, in_id=id_map[edge.in_id], out_id=id_map[edge.out_id], innovation=tracker.innovation(id_map[edge.in_id], id_map[edge.out_id])) for edge in source.edges]
    return CompositionGenome(nodes=nodes, edges=edges)


@EVOLVE_STRATEGY.register("composition")
def _build_composition(config: dict[str, Any]) -> "CompositionStrategy":
    return CompositionStrategy()


@dataclass
class CompositionStrategy:
    name: str = "composition"

    @staticmethod
    def _initial_glue_values(spec: CompTaskSpec, runtime: StrategyRuntime, seed_comps: list[CompositionGenome] | None) -> int:
        """Largest candidate gene allocation needed before the first composition assessment."""

        loop = runtime.loop
        minimal_values = spec.output_width + sum(
            glue_value_count(width, spec.output_width, glue_rank=loop.glue_rank, glue_rank_threshold=loop.glue_rank_threshold) for _signature, width in spec.input_specs
        )
        seeded = [sum(edge_storage_value_count(edge) for edge in comp.edges) for comp in (seed_comps or [])[: loop.comp_pop_size]]
        if len(seeded) < loop.comp_pop_size:
            seeded.append(minimal_values)
        return max(seeded, default=minimal_values)

    def __call__(
        self,
        task: Task,
        spec: CompTaskSpec,
        runtime: StrategyRuntime,
        *,
        budget: int,
        seed_comps: list | None = None,
    ) -> StrategyResult:
        initial_glue_values = self._initial_glue_values(spec, runtime, seed_comps)
        loop = runtime.loop
        pool = get_shared_pool()
        concurrent_trainers = int(getattr(pool, "_processes", 0) or loop.evolver.assess_workers or 1)
        estimate = loop.assess_glue_resources(
            initial_glue_values,
            stage="composition_population",
            population_multiplicity=loop.comp_pop_size,
            concurrent_trainers=min(loop.comp_pop_size, max(1, concurrent_trainers)),
            device="cpu",
        )
        if not estimate.accepted:
            logger.warning(
                "composition declined before allocation: initial candidate needs %s glue values (%s host, %s device at stage multiplicity; limit %s)",
                f"{initial_glue_values:,}",
                format_bytes(estimate.host_required_bytes),
                format_bytes(estimate.device_required_bytes),
                f"{estimate.limit_values:,}",
            )
            return StrategyResult(
                strategy=self.name,
                metric=0.0,
                generations_used=0,
                champion_metrics={
                    "declined_composition_glue_values": float(initial_glue_values),
                    **estimate.metrics("composition_resource"),
                },
                resource_metrics=estimate.metrics("composition_resource"),
            )
        progress = {"generations": 0}

        def hook(generation: int, best: AssessedComposition, mean_fitness: float) -> None:
            progress["generations"] = generation + 1
            if runtime.on_generation is not None:
                runtime.on_generation(self.name, generation, best, mean_fitness)

        best = runtime.loop.run_task(spec, runtime.state, budget=budget, stop=runtime.stall_factory(budget), seed_comps=seed_comps, on_generation=hook)
        verified = self._verify(best, spec, runtime)
        return StrategyResult(
            strategy=self.name,
            metric=runtime.metric_of(verified),
            generations_used=progress["generations"],
            champion_comp=verified,
            champion_metrics=dict(verified.metrics),
            size_metrics=comp_size_metrics(verified.comp),
            resource_metrics=estimate.metrics("composition_resource"),
        )

    def _verify(self, best: AssessedComposition, spec: CompTaskSpec, runtime: StrategyRuntime) -> AssessedComposition:
        """B4: the returned champion may predate later module advances/writebacks. Re-assess it
        against CURRENT champions/library (pure forward); if the stale score passed the bar but the
        fresh one does not, allow ONE bounded re-fit (a single trained candidate) and keep the
        better FRESH assessment. The verified assessment is what the threshold check and admission
        consume, so stale weights or stale metrics can never be persisted."""
        if best.net is None:
            return best  # floored (or white-box-stubbed): nothing fresher exists to assemble
        fresh = runtime.loop.assess_composition(best.comp, spec, runtime.state, train=False)
        if runtime.accepted(fresh):
            return fresh
        if runtime.accepted(best):
            logger.info("champion verification dropped below threshold (stale %.3f -> fresh %.3f); one re-fit", runtime.metric_of(best), runtime.metric_of(fresh))
            refit = runtime.loop.assess_composition(best.comp, spec, runtime.state, train=True)
            return refit if runtime.metric_of(refit) >= runtime.metric_of(fresh) else fresh
        return fresh


@EVOLVE_STRATEGY.register("direct")
def _build_direct(config: dict[str, Any]) -> "DirectStrategy":
    overlay = dict(config)
    table = config.get("orchestrator", {}).get("direct", {})
    evolution = {key: value for key, value in config.get("evolution", {}).items() if key != "loop"}
    evolution["pop_size"] = int(table.get("pop_size", 48))
    evolution["elitism"] = int(table.get("elitism", 2))
    evolution["assess_workers"] = table.get("assess_workers", 0)  # "auto" resolves in build_evolver
    overlay["library_dir"] = config.get("orchestrator", {}).get("library_dir", "library")
    # Single-task structure growth usually wants a stronger inner trainer (and sometimes a
    # different mutation recipe) than the composition loop's glue fitting; both are overridable.
    for overridable in ("mutation", "train", "evaluate", "novelty", "halving_stages", "halving_keep"):
        if overridable in table:
            evolution[overridable] = table[overridable]
    overlay["evolution"] = evolution
    max_flat_outputs = table.get("max_flat_outputs", 0)
    max_init_genes = table.get("max_init_genes", 0)
    for name, value in (("max_flat_outputs", max_flat_outputs), ("max_init_genes", max_init_genes)):
        if value != "adaptive" and int(value) < 0:
            raise ValueError(f"[orchestrator.direct] {name} must be non-negative or 'adaptive'")
    return DirectStrategy(
        evolver=build_evolver(overlay),
        max_flat_outputs=max_flat_outputs if max_flat_outputs == "adaptive" else int(max_flat_outputs),
        max_init_genes=max_init_genes if max_init_genes == "adaptive" else int(max_init_genes),
        structured_grid=bool(table.get("structured_grid", False)),
        blind_query=bool(config.get("orchestrator", {}).get("blind_query", False)),
    )


@dataclass
class DirectStrategy:
    """The flat phase-1 recipe on the task's real I/O; the structure-growing fallback that solves
    tasks the composition indirection stalls on (two-spirals class), and the path that admits
    TASK-SHAPED modules into the library."""

    evolver: Evolver
    name: str = "direct"
    # Decline tasks whose flattened OUTPUT width exceeds this ([orchestrator.direct]
    # max_flat_outputs; 0 = off). The flat substrate's [n, h] weight matrix is DENSE and h spans
    # every output node, so a wide-output task (rungs 12-14, 18 class) would allocate GBs per
    # genome before its first forward. Declining lets the ladder escalate to composition (whose
    # glue is already rank-factored above glue_rank_threshold) and to the decomposers.
    max_flat_outputs: int | str = 0
    # Decline tasks whose dense-init gene count (flat_inputs + 1) * flat_outputs exceeds this
    # ([orchestrator.direct] max_init_genes; 0 = off): the input-side twin of max_flat_outputs.
    # The per-task deadline only fires between ladder positions and between generations, so an
    # oversize minimal init plus its first generation (Python object churn, ~0.8 GB genome pickles
    # to the assess pool) runs for HOURS before any check exists; a 409,600 x 8 task wedged two
    # runs on 2026-07-06 exactly this way. The attempt must be refused from arithmetic alone.
    max_init_genes: int | str = 0
    # Structured grids retain dense cell loss for training while reporting predicted output shape,
    # exact-example accuracy, and trivial baselines. Off keeps the historical flat adapter.
    structured_grid: bool = False
    # In blind mode every candidate sees a query-less EncodedTask. The selected payload is evaluated
    # once on the full task below, after evolution has ended.
    blind_query: bool = False

    def _adapter(self, task: Task, *, include_query: bool = True) -> TaskAdapter | TemporalTaskAdapter:
        max_inline_depth = int(getattr(self.evolver, "max_inline_depth", DEFAULT_MAX_INLINE_DEPTH))
        support_input, _support_output = support_loader(task)
        if has_time_axis(support_input.descriptor):
            adapter = temporal_adapter(task, max_inline_depth=max_inline_depth)  # recurrence goes LIVE: decode_recurrent + BPTT
            if not include_query:
                adapter.encoded = without_query(adapter.encoded)
            return adapter
        width = 1
        for dim in support_input.data.shape[1:]:
            width *= int(dim)
        encoder = Level0Encoder(max_flat_dim=width)
        encoded = None
        if self.structured_grid:
            from ardevo.structured import encode_structured_grid

            encoded = encode_structured_grid(task, encoder, include_query=include_query)
        if encoded is None:
            encoded = fit_query_target(encode_task(task, encoder))
            if not include_query:
                encoded = without_query(encoded)
        elif not include_query:
            encoded = encoded.without_query()
        return TaskAdapter(
            encoded,
            encoder,
            input_width(encoded),
            output_features(encoded),
            grid_shape=self._grid_shape(task),
            max_inline_depth=max_inline_depth,
        )

    @staticmethod
    def _grid_shape(task: Task) -> tuple[int, ...] | None:
        support_input, _support_output = support_loader(task)
        shape = tuple(int(dim) for dim in support_input.data.shape[1:])
        return shape if len(shape) >= 2 else None

    def __call__(
        self,
        task: Task,
        spec: CompTaskSpec,
        runtime: StrategyRuntime,
        *,
        budget: int,
        seed_comps: list | None = None,
        seed_entries: list[LibraryEntry] | None = None,
        seed_genomes: list[Genome] | None = None,
    ) -> StrategyResult:
        resource_metrics: dict[str, float] = {}
        if self.max_flat_outputs == "adaptive" or self.max_init_genes == "adaptive" or int(self.max_flat_outputs) > 0 or int(self.max_init_genes) > 0:
            support_input, support_output = support_loader(task)
            flat_positions = 1
            for dim in support_output.data.shape[1:]:
                flat_positions *= int(dim)
            flat_outputs = model_output_features(support_output.descriptor, flat_positions)
            if self.max_flat_outputs != "adaptive" and 0 < int(self.max_flat_outputs) < flat_outputs:
                return StrategyResult(strategy=self.name, metric=0.0, generations_used=0, champion_metrics={"declined_flat_width": float(flat_outputs)})
            flat_inputs = 1
            for dim in support_input.data.shape[1:]:
                flat_inputs *= int(dim)
            init_genes = (flat_inputs + 1) * flat_outputs
            if self.max_init_genes != "adaptive" and 0 < int(self.max_init_genes) < init_genes:
                return StrategyResult(strategy=self.name, metric=0.0, generations_used=0, champion_metrics={"declined_init_genes": float(init_genes)})
            if self.max_init_genes == "adaptive" or self.max_flat_outputs == "adaptive":
                population_execution = str(getattr(self.evolver, "execution_mode", "serial")).startswith("population_")
                estimate = runtime.loop.assess_glue_resources(
                    init_genes,
                    stage="direct_population",
                    storage="tuple",
                    population_multiplicity=self.evolver.pop_size,
                    concurrent_trainers=self.evolver.pop_size if population_execution else max(1, self.evolver.assess_workers),
                    fixed_limit=0,
                )
                if not estimate.accepted:
                    logger.warning(
                        "direct evolution declined before allocation: dense init needs %s genes (%s host, %s device; adaptive limit %s)",
                        f"{init_genes:,}",
                        format_bytes(estimate.host_required_bytes),
                        format_bytes(estimate.device_required_bytes),
                        f"{estimate.limit_values:,}",
                    )
                    return StrategyResult(
                        strategy=self.name,
                        metric=0.0,
                        generations_used=0,
                        champion_metrics={"declined_init_genes": float(init_genes), **estimate.metrics("direct_resource")},
                        resource_metrics=estimate.metrics("direct_resource"),
                    )
                resource_metrics = estimate.metrics("direct_resource")
        adapter = self._adapter(task, include_query=not self.blind_query)
        # The direct population's library-reading mutators must sample from the SAME library the
        # decode-time macro resolver resolves (the orchestrator's attached one), or add_macro_node
        # can graft a macro ref that decode cannot satisfy -> a hard KeyError mid-search.
        self.evolver.library = runtime.library
        # Grid tasks get coordinates stamped on the seed population so the geometry-biased
        # mutators (add_local_node and friends) can grow local receptive fields.
        grid = self._grid_shape(task) if not isinstance(adapter, TemporalTaskAdapter) else None
        original_init = self.evolver.init_op
        if grid is not None:
            from ardevo.evolution.init import stamp_input_coordinates

            self.evolver.init_op = lambda n_inputs, n_outputs, *, rng: stamp_input_coordinates(original_init(n_inputs, n_outputs, rng=rng), grid)

        def seeded_front(tracker: InnovationTracker) -> list[Genome]:
            # Refine-on-hit warm start: grafted entries take the front of the population and are
            # trained/assessed like every other member. Grid stamping keeps geometry mutators live.
            grafted = [graft(entry, tracker) for entry in (seed_entries or [])]
            grafted.extend(_restamp_genome(genome, tracker) for genome in (seed_genomes or []))
            if grid is not None:
                from ardevo.evolution.init import stamp_input_coordinates

                grafted = [stamp_input_coordinates(genome, grid) for genome in grafted]
            return grafted

        try:
            # Shared rng: keeps the whole solve deterministic per seed and checkpoint-coherent.
            state = self.evolver.seed_state(adapter, runtime.state.rng, seeded_front=seeded_front if seed_entries or seed_genomes else None)
        finally:
            self.evolver.init_op = original_init
        # Refine fairness: the grafted seeds' TRAINED standing is the incumbent baseline. Lineage is
        # tracked by structural fingerprint (selection reshuffles the population, and a same-topology
        # descendant with remixed weights is still the incumbent topology, so it counts).
        seed_fingerprints: set[str] = set()
        seed_metric: float | None = None
        if seed_entries:
            seed_fingerprints = {structural_fingerprint(MODULE, genome_to_dict(member.genome)) for member in state.population[: len(seed_entries)]}

        def refresh_seed_metric() -> None:
            nonlocal seed_metric
            for member in state.population:
                if structural_fingerprint(MODULE, genome_to_dict(member.genome)) in seed_fingerprints:
                    value = runtime.metric_of(member)
                    seed_metric = value if seed_metric is None else max(seed_metric, value)

        stop = runtime.stall_factory(budget)
        best: Assessed = max(state.population, key=lambda item: item.fitness)
        generations = 0
        for generation in range(budget):
            generation_best = max(state.population, key=lambda item: item.fitness)
            if generation_best.fitness > best.fitness:
                best = generation_best
            if seed_fingerprints:
                refresh_seed_metric()
            if runtime.on_generation is not None:
                mean_fitness = sum(item.fitness for item in state.population) / len(state.population)
                runtime.on_generation(self.name, generation, generation_best, mean_fitness)
            runtime.state.generation += 1  # the global clock spans strategies for monotonic logging
            generations = generation + 1
            if runtime.accepted(best) or stop(generation, best):
                break
            self.evolver.advance(state, adapter)

        # Verification: the genome PAYLOAD (not the live module object) must reproduce the metric,
        # because the payload is what admission persists and lookups re-decode.
        verified = self.evolver.evaluate_only(best.genome, adapter)
        reported = self.evolver.evaluate_only(verified.genome, self._adapter(task, include_query=True)) if self.blind_query else None
        search_metric = runtime.metric_of(verified)
        return StrategyResult(
            strategy=self.name,
            # Acceptance remains support-only in blind mode. Query metrics on ``verified`` are a
            # one-shot report attached to the champion and cannot steer search or early stopping.
            metric=search_metric if self.blind_query else runtime.metric_of(verified),
            generations_used=generations,
            champion_genome=verified.genome,
            champion_metrics=dict(verified.metrics),
            report_metrics=dict(reported.metrics) if reported is not None else {},
            seed_metric=seed_metric,
            size_metrics=_module_size_metrics(verified.genome, state.population),
            resource_metrics=resource_metrics,
        )


@EVOLVE_STRATEGY.register("grammar")
def _build_grammar(config: dict[str, Any]) -> "GrammarStrategy":
    table = config.get("orchestrator", {}).get("grammar", {}) or {}
    return GrammarStrategy(
        direct=_build_direct(config),
        max_productions=max(1, int(table.get("max_productions", 12))),
        candidates_per_production=max(1, int(table.get("candidates_per_production", 3))),
        mutation_steps=max(0, int(table.get("mutation_steps", 2))),
        module_sizes=tuple(int(size) for size in table.get("module_sizes", [3, 4])),
        composition_sizes=tuple(int(size) for size in table.get("composition_sizes", [2, 3, 4])),
        min_lineage_support=max(2, int(table.get("min_lineage_support", 2))),
        per_entry_cap=max(1, int(table.get("per_entry_cap", 5000))),
    )


@dataclass
class GrammarStrategy:
    """Search programs assembled only from motifs independently rediscovered by evolution."""

    direct: Callable[..., StrategyResult]
    max_productions: int = 12
    candidates_per_production: int = 3
    mutation_steps: int = 2
    module_sizes: tuple[int, ...] = (3, 4)
    composition_sizes: tuple[int, ...] = (2, 3, 4)
    min_lineage_support: int = 2
    per_entry_cap: int = 5000
    name: str = "grammar"
    _library_keys: tuple[str, ...] = field(default=(), init=False, repr=False)
    _grammar: Any = field(default=None, init=False, repr=False)

    def _programs(self, runtime: StrategyRuntime) -> list[Any]:
        from ardevo.grammar import crossover_program, mutate_program, rebuild_grammar, seed_program

        keys = tuple(runtime.library.keys())
        if self._grammar is None or keys != self._library_keys:
            self._grammar = rebuild_grammar(
                runtime.library,
                module_sizes=self.module_sizes,
                composition_sizes=self.composition_sizes,
                min_lineage_support=self.min_lineage_support,
                per_entry_cap=self.per_entry_cap,
            )
            self._library_keys = keys
        productions = sorted(self._grammar.productions, key=lambda item: (-item.mdl_gain, -item.support, item.key))[: self.max_productions]
        programs: list[Any] = []
        seen: set[str] = set()
        for production in productions:
            seed = seed_program(production)
            candidates = [seed]
            for _ in range(self.candidates_per_production - 1):
                candidate = seed
                for _step in range(self.mutation_steps):
                    candidate = mutate_program(candidate, self._grammar, rng=runtime.state.rng)
                candidates.append(candidate)
            for candidate in candidates:
                key = repr(candidate.to_dict())
                if key not in seen:
                    seen.add(key)
                    programs.append(candidate)
        # Aligned crossover is useful only once mutation has produced multi-production parents.
        parents = list(programs)
        for left, right in zip(parents[::2], parents[1::2]):
            child = crossover_program(left, right, self._grammar, rng=runtime.state.rng)
            key = repr(child.to_dict())
            if key not in seen:
                seen.add(key)
                programs.append(child)
        return programs

    @staticmethod
    def _composition_compatible(comp: CompositionGenome, spec: CompTaskSpec) -> bool:
        if len(comp.output_ids) != 1 or comp.nodes[comp.output_ids[0]].in_width != spec.output_width:
            return False
        for node_id in comp.input_ids:
            node = comp.nodes[node_id]
            columns = spec.bank_columns.get(node.ref)
            if columns is None or len(columns) != node.out_width:
                return False
        return True

    def __call__(
        self,
        task: Task,
        spec: CompTaskSpec,
        runtime: StrategyRuntime,
        *,
        budget: int,
        seed_comps: list | None = None,
    ) -> StrategyResult:
        from ardevo.grammar import GrammarError, compile_program

        module_seeds: list[Genome] = []
        comp_seeds: list[CompositionGenome] = []
        for program in self._programs(runtime):
            try:
                compiled = compile_program(program, self._grammar, library=runtime.library, rng=runtime.state.rng)
            except (GrammarError, KeyError, ValueError):
                continue
            if isinstance(compiled, Genome) and len(compiled.input_ids) == spec.n_inputs and len(compiled.output_ids) == spec.output_width:
                module_seeds.append(compiled)
            elif isinstance(compiled, CompositionGenome) and self._composition_compatible(compiled, spec):
                comp_seeds.append(_restamp_composition(compiled, runtime.state.comp_innovations))
        if not module_seeds and not comp_seeds:
            return StrategyResult(strategy=self.name, metric=0.0, generations_used=0, champion_metrics={"grammar_productions": float(len(self._grammar.productions))})

        results: list[StrategyResult] = []
        used = 0
        if module_seeds:
            allocation = budget if not comp_seeds else max(1, budget // 2)
            result = self.direct(task, spec, runtime, budget=allocation, seed_genomes=module_seeds)
            used += result.generations_used
            results.append(result)
            if runtime.accepted(SimpleNamespace(metrics=result.champion_metrics)):
                result.strategy = self.name
                result.generations_used = used
                return result
        remaining = max(0, budget - used)
        if comp_seeds and remaining > 0:
            result = CompositionStrategy()(task, spec, runtime, budget=remaining, seed_comps=[*(seed_comps or []), *comp_seeds])
            used += result.generations_used
            results.append(result)
        winner = max(results, key=lambda item: item.metric)
        winner.strategy = self.name
        winner.generations_used = used
        winner.champion_metrics["grammar_productions"] = float(len(self._grammar.productions))
        winner.champion_metrics["grammar_programs"] = float(len(module_seeds) + len(comp_seeds))
        return winner


@EVOLVE_STRATEGY.register("routed")
def _build_routed(config: dict[str, Any]) -> Any:
    # Lazy import (the train.py pattern): routing pulls in torch-heavy machinery only when configured.
    from ardevo.routing import build_routed_strategy

    return build_routed_strategy(config)


def build_strategies(config: dict[str, Any]) -> list[tuple[str, Callable[..., StrategyResult]]]:
    """Resolve `[orchestrator] evolve` (default: composition only, the pre-strategy behavior)."""
    names = [str(name) for name in config.get("orchestrator", {}).get("evolve", ["composition"])]
    return [(name, EVOLVE_STRATEGY.get(name)(config)) for name in names]
