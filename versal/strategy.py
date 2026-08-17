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

import math
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from versal.topology import TopologyTabuSession

from versal.dataset.icarus import Level0Encoder, Task, encode_task, model_output_features, support_loader
from versal.evaluation import fit_query_target, input_width, output_features, without_query
from versal.evolution.composition import CompositionGenome, edge_storage_value_count, glue_value_count
from versal.evolution.evolver import Assessed, Evolver, TaskAdapter, get_shared_pool
from versal.evolution.genome import Genome, InnovationTracker, genome_from_dict, genome_to_dict
from versal.evolution.loop import AssessedComposition, CompTaskSpec, HierarchicalLoop, HierarchicalState
from versal.evolution.registry import Registry, build_evolver
from versal.library import MODULE, LibraryEntry, ModuleLibrary, graft, structural_fingerprint
from versal.reference_depth import DEFAULT_MAX_INLINE_DEPTH
from versal.temporal import TemporalTaskAdapter, has_time_axis, temporal_adapter
from versal.utils.logging import Logger
from versal.utils.resources import StageDecision, StageFootprint, format_bytes

logger = Logger.get_logger()

EVOLVE_STRATEGY: Registry = Registry("evolve_strategy")


@dataclass(frozen=True)
class StrategyPreflight:
    eligible: bool
    representation: str
    footprint: StageFootprint | None = None
    decision: StageDecision | None = None
    reason: str | None = None


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
    shutdown_requested: Callable[[], bool] | None = None
    topology_tabu: "TopologyTabuSession | None" = None
    deadline: float | None = None

    def accepted(self, item: Any) -> bool:
        return self.accepts(item) if self.accepts is not None else self.metric_of(item) >= self.accept_threshold

    def should_stop(self) -> bool:
        return self.deadline_exceeded is not None and self.deadline_exceeded()

    def should_shutdown(self) -> bool:
        return self.shutdown_requested is not None and self.shutdown_requested()


@dataclass
class StrategyResult:
    strategy: str
    metric: float
    generations_used: int
    champion_comp: AssessedComposition | None = None  # composition-shaped winner (verified fresh)
    champion_genome: Genome | None = None  # module-shaped winner (trained weights written back)
    # Best support-search payload retained only for one later held-out report. Unlike a champion,
    # these candidates have not cleared the verification/admission contract.
    report_candidate_comp: AssessedComposition | None = None
    report_candidate_genome: Genome | None = None
    # Router-space state can execute the task even when it cannot satisfy the reusable DSL
    # contract. Keep that state on the report-only rail so blind held-out evaluation can still
    # measure progress without making it admissible or persistable as a task solution.
    report_candidate_routed: Any | None = None
    report_candidate_metrics: dict[str, float] = field(default_factory=dict)
    champion_metrics: dict[str, float] = field(default_factory=dict)
    # Held-out metrics are kept on a separate rail. They are reporting only and must never affect
    # admission, refinement, robustness ranking, or the next task's search state.
    report_metrics: dict[str, float] = field(default_factory=dict)
    report_attempted: bool = False
    # Support-only admission evidence. Search strategies stop on the ordinary support gate; the
    # orchestrator fills these fields before deciding whether the payload is a verified solution.
    validation_status: str = "not_run"
    validation_metrics: dict[str, float] = field(default_factory=dict)
    # A routed winner is a RECORD (versal.routing.RoutedSolution), not an admissible payload: the
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
    # Present only for resolution-independent site programs. The genome remains ordinary.
    field_template: dict[str, Any] | None = None
    representation: str | None = None

    @property
    def has_report_candidate(self) -> bool:
        """Whether a support-selected payload is available for held-out reporting."""

        return (
            self.champion_comp is not None
            or self.champion_genome is not None
            or self.report_candidate_comp is not None
            or self.report_candidate_genome is not None
            or self.report_candidate_routed is not None
        )

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
    return CompositionStrategy(blind_query=bool(config.get("orchestrator", {}).get("blind_query", False)))


@dataclass
class CompositionStrategy:
    name: str = "composition"
    blind_query: bool = False

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
        support_rows = int(spec.encoded.support_input[0].shape[0])
        footprint = StageFootprint(
            stage="composition_population",
            representation="composition_mixed_glue",
            candidate_bytes=initial_glue_values * 4 + (len(spec.input_specs) + 2) * 128,
            population_size=loop.comp_pop_size,
            optimizer_bytes=initial_glue_values * 12 * min(loop.comp_pop_size, max(1, concurrent_trainers)),
            activation_bytes=support_rows * (spec.n_inputs + spec.output_width) * 4 * min(loop.comp_pop_size, max(1, concurrent_trainers)),
            transfer_bytes=initial_glue_values * 4 * min(loop.comp_pop_size, max(1, concurrent_trainers)) if concurrent_trainers > 1 else 0,
            work_operations=support_rows * initial_glue_values,
            detail="dense, rank-factored, and fixed-port-map edges priced by stored values",
        )
        stage_decision = loop.resource_policy.assess_stage(footprint, device="cpu")
        declined = not estimate.accepted if estimate.mode == "fixed" else not stage_decision.accepted
        combined_resource_metrics = {**estimate.metrics("composition_resource"), **stage_decision.metrics("composition_stage")}
        if declined:
            logger.debug(
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
                    **combined_resource_metrics,
                },
                resource_metrics=combined_resource_metrics,
            )
        progress = {"generations": 0}
        loop.evolver.deadline = runtime.deadline
        loop.evolver.deadline_exceeded = runtime.deadline_exceeded

        def hook(generation: int, best: AssessedComposition, mean_fitness: float) -> None:
            progress["generations"] = generation + 1
            if runtime.on_generation is not None:
                runtime.on_generation(self.name, generation, best, mean_fitness)

        best = runtime.loop.run_task(spec, runtime.state, budget=budget, stop=runtime.stall_factory(budget), seed_comps=seed_comps, on_generation=hook)
        verification_timed_out = False
        try:
            verified = self._verify(best, spec, runtime)
        except TimeoutError:
            if not self.blind_query:
                raise
            verified = None
            verification_timed_out = True
        if self.blind_query and (runtime.should_stop() or verification_timed_out):
            return StrategyResult(
                strategy=self.name,
                metric=runtime.metric_of(best),
                generations_used=progress["generations"],
                report_candidate_comp=best,
                champion_metrics=dict(best.metrics),
                size_metrics=comp_size_metrics(best.comp),
                resource_metrics=combined_resource_metrics,
                representation="composition",
            )
        assert verified is not None
        return StrategyResult(
            strategy=self.name,
            metric=runtime.metric_of(verified),
            generations_used=progress["generations"],
            champion_comp=verified,
            champion_metrics=dict(verified.metrics),
            size_metrics=comp_size_metrics(verified.comp),
            resource_metrics=combined_resource_metrics,
            representation="composition",
        )

    def _verify(self, best: AssessedComposition, spec: CompTaskSpec, runtime: StrategyRuntime) -> AssessedComposition:
        """B4: the returned champion may predate later module advances/writebacks. Re-assess it
        against CURRENT champions/library (pure forward); if the stale score passed the bar but the
        fresh one does not, allow ONE bounded re-fit (a single trained candidate) and keep the
        better FRESH assessment. The verified assessment is what the threshold check and admission
        consume, so stale weights or stale metrics can never be persisted."""
        if best.net is None or runtime.should_stop():
            return best  # floored (or white-box-stubbed): nothing fresher exists to assemble
        fresh = runtime.loop.assess_composition(best.comp, spec, runtime.state, train=False)
        if runtime.accepted(fresh):
            return fresh
        if runtime.accepted(best):
            logger.debug("champion verification dropped below threshold (stale %.3f -> fresh %.3f); one re-fit", runtime.metric_of(best), runtime.metric_of(fresh))
            refit = runtime.loop.assess_composition(best.comp, spec, runtime.state, train=True)
            return refit if runtime.metric_of(refit) >= runtime.metric_of(fresh) else fresh
        return fresh


@EVOLVE_STRATEGY.register("field")
def _build_field(config: dict[str, Any]) -> "FieldStrategy":
    overlay = dict(config)
    table = config.get("orchestrator", {}).get("field", {}) or {}
    evolution = {key: value for key, value in config.get("evolution", {}).items() if key != "loop"}
    for key in ("pop_size", "elitism", "assess_workers", "mutation", "train", "evaluate", "novelty", "halving_stages", "halving_keep"):
        if key in table:
            evolution[key] = table[key]
    overlay["evolution"] = evolution
    overlay["library_dir"] = config.get("orchestrator", {}).get("library_dir", "library")
    return FieldStrategy(
        evolver=build_evolver(overlay),
        train_sites=max(1, int(table.get("train_sites", 4096))),
        audit_sites=max(1, int(table.get("audit_sites", 16384))),
        verify_top_k=max(1, int(table.get("verify_top_k", 5))),
        verify_chunk_size=max(1, int(table.get("verify_chunk_size", 32768))),
        blind_query=bool(config.get("orchestrator", {}).get("blind_query", False)),
    )


@dataclass
class FieldStrategy:
    """Evolve one compact ordinary graph applied at every valid aligned spatial site."""

    evolver: Evolver
    train_sites: int = 4096
    audit_sites: int = 16384
    verify_top_k: int = 5
    verify_chunk_size: int = 32768
    blind_query: bool = False
    name: str = "field"

    def preflight(self, task: Task, runtime: StrategyRuntime) -> StrategyPreflight:
        from versal.evolution.init import estimate_initialization
        from versal.field import field_contract, field_feature_width

        contract = field_contract(task)
        if contract is None:
            return StrategyPreflight(False, "field_template", reason="support is not an aligned spatial mapping")
        n_inputs = field_feature_width(contract.input_channels)
        output_classes = contract.output_n_classes if contract.output_value_type in {"CATEGORICAL", "ORDINAL"} else 1
        n_outputs = contract.output_channels * int(output_classes or 1)
        try:
            init = estimate_initialization(self.evolver.init_kind, n_inputs, n_outputs, **self.evolver.init_params)
        except KeyError as error:
            if runtime.loop.resource_policy.mode == "adaptive":
                return StrategyPreflight(False, "field_template", reason=str(error))
            return StrategyPreflight(True, "field_template", reason=str(error))
        computed = max(0, init.nodes - n_inputs - 1)
        cells = init.nodes * computed
        audit_bytes = self.audit_sites * (n_inputs + n_outputs) * 4
        population = max(1, self.evolver.pop_size)
        footprint = StageFootprint(
            stage="field_population",
            representation=f"field_template/{self.evolver.init_kind}",
            candidate_bytes=init.nodes * 32 + init.edges * 64 + cells * 5,
            population_size=population,
            optimizer_bytes=cells * 12 * max(1, self.evolver.assess_workers),
            activation_bytes=audit_bytes,
            work_operations=self.audit_sites * max(1, init.edges),
            detail=f"{contract.identity}: {init.nodes} nodes, {init.edges} edges; H/W symbolic",
        )
        decision = runtime.loop.resource_policy.assess_stage(footprint)
        return StrategyPreflight(decision.accepted, footprint.representation, footprint, decision, decision.reason)

    def evaluate_report(self, genome: Genome, task: Task, field_template: dict[str, Any]) -> dict[str, float]:
        """Decode a field payload and evaluate its held-out query without training."""

        if not task.query:
            return {"query_loss": math.inf}
        from versal.field import decode_field_payload, evaluate_field_module

        payload = genome_to_dict(genome)
        payload["field_template"] = field_template
        try:
            module, contract = decode_field_payload(
                payload,
                library=getattr(self.evolver, "library", None),
                max_inline_depth=int(getattr(self.evolver, "max_inline_depth", DEFAULT_MAX_INLINE_DEPTH)),
            )
        except (KeyError, ValueError):
            return {}
        return evaluate_field_module(module, task, contract, split="query", chunk_size=self.verify_chunk_size, deadline=None)

    def __call__(
        self,
        task: Task,
        spec: CompTaskSpec,
        runtime: StrategyRuntime,
        *,
        budget: int,
        seed_comps: list | None = None,
        seed_entries: list[LibraryEntry] | None = None,
    ) -> StrategyResult:
        from versal.field import FieldAdapter, deterministic_sites, encode_sites, evaluate_field_module, field_contract, valid_sites

        contract = field_contract(task)
        if contract is None:
            return StrategyResult(self.name, 0.0, 0, strategy_metrics={"field_ineligible": 1.0})
        preflight = self.preflight(task, runtime)
        if not preflight.eligible:
            metrics = preflight.decision.metrics("field_resource") if preflight.decision is not None else {}
            return StrategyResult(self.name, 0.0, 0, resource_metrics=metrics, strategy_metrics={"field_preflight_ineligible": 1.0})
        all_sites = valid_sites(task.support)
        train_sites = deterministic_sites(all_sites, self.train_sites, salt=f"train:{contract.identity}")
        audit_sites = deterministic_sites(all_sites, self.audit_sites, salt=f"audit:{contract.identity}")
        try:
            training = encode_sites(task, train_sites, contract, chunk_size=self.verify_chunk_size, deadline=runtime.deadline)
            audit = encode_sites(task, audit_sites, contract, chunk_size=self.verify_chunk_size, deadline=runtime.deadline)
        except TimeoutError:
            return StrategyResult(self.name, 0.0, 0, strategy_metrics={"field_deadline_stage_feature_preparation": 1.0})
        adapter = FieldAdapter(training, audit, contract, max_inline_depth=self.evolver.max_inline_depth, library=runtime.library)
        self.evolver.library = runtime.library
        self.evolver.deadline_exceeded = runtime.deadline_exceeded
        self.evolver.deadline = runtime.deadline
        self.evolver.topology_tabu = runtime.topology_tabu

        def seeded_front(tracker: InnovationTracker) -> list[Genome]:
            candidates: list[Genome] = []
            for entry in seed_entries or []:
                try:
                    from versal.field import payload_field_contract

                    if payload_field_contract(entry.payload) == contract:
                        candidates.append(_restamp_genome(genome_from_dict(entry.payload), tracker))
                except ValueError:
                    continue
            return candidates

        state = self.evolver.seed_state(adapter, runtime.state.rng, seeded_front=seeded_front if seed_entries else None)
        best_full: Assessed | None = None
        generations = 0
        stop = runtime.stall_factory(budget)

        def verify_front() -> bool:
            nonlocal best_full
            ranked = sorted(state.population, key=lambda item: item.fitness, reverse=True)[: self.verify_top_k]
            for member in ranked:
                if runtime.should_stop() and best_full is not None:
                    return False
                module = member.module if member.module is not None else adapter.decode(member.genome)
                try:
                    full = evaluate_field_module(
                        module,
                        task,
                        contract,
                        split="support",
                        chunk_size=self.verify_chunk_size,
                        deadline=runtime.deadline,
                    )
                except TimeoutError:
                    return False
                metrics = dict(member.metrics)
                metrics.update(full)
                metrics["full_support_accuracy"] = full["support_accuracy"]
                metrics["verification_gap"] = metrics.get("sampled_support_accuracy", full["support_accuracy"]) - full["support_accuracy"]
                assessed = Assessed(member.genome, metrics, member.fitness, module)
                if best_full is None or runtime.metric_of(assessed) > runtime.metric_of(best_full):
                    best_full = assessed
            return best_full is not None and runtime.accepted(best_full)

        for generation in range(budget):
            generations = generation + 1
            generation_best = max(state.population, key=lambda item: item.fitness)
            if runtime.on_generation is not None:
                runtime.on_generation(self.name, generation, generation_best, sum(item.fitness for item in state.population) / len(state.population))
            runtime.state.generation += 1
            if verify_front() or stop(generation, generation_best) or runtime.should_stop():
                break
            self.evolver.advance(state, adapter)
            if state.topology_exhausted:
                break
        if best_full is None:
            verify_front()
        if self.blind_query and (runtime.should_stop() or best_full is None):
            metrics = preflight.decision.metrics("field_resource") if preflight.decision is not None else {}
            best_sampled = max(state.population, key=lambda item: item.fitness)
            return StrategyResult(
                strategy=self.name,
                metric=runtime.metric_of(best_sampled),
                generations_used=generations,
                report_candidate_genome=best_sampled.genome,
                champion_metrics=dict(best_sampled.metrics),
                size_metrics=_module_size_metrics(best_sampled.genome, state.population),
                resource_metrics=metrics,
                strategy_metrics={
                    "field_deadline_before_full_verification": 1.0,
                    "field_application_sites": float(len(all_sites)),
                    "field_sampled_sites": float(len(audit_sites)),
                },
                field_template=contract.to_dict(),
                representation=f"field/{contract.version}",
            )
        if best_full is None:
            metrics = preflight.decision.metrics("field_resource") if preflight.decision is not None else {}
            return StrategyResult(self.name, 0.0, generations, resource_metrics=metrics, strategy_metrics={"field_deadline_before_full_verification": 1.0})
        resource_metrics = preflight.decision.metrics("field_resource") if preflight.decision is not None else {}
        return StrategyResult(
            strategy=self.name,
            metric=runtime.metric_of(best_full),
            generations_used=generations,
            champion_genome=best_full.genome,
            champion_metrics=dict(best_full.metrics),
            size_metrics=_module_size_metrics(best_full.genome, state.population),
            resource_metrics=resource_metrics,
            strategy_metrics={
                "field_application_sites": float(len(all_sites)),
                "field_sampled_sites": float(len(audit_sites)),
                "field_verification_gap": float(best_full.metrics.get("verification_gap", 0.0)),
            },
            field_template=contract.to_dict(),
            representation=f"field/{contract.version}",
        )


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

    def preflight(self, task: Task, runtime: StrategyRuntime) -> StrategyPreflight:
        """Price the configured initializer and its decoded compact GraphNet exactly enough to
        reject certain OOMs.  This intentionally does not instantiate a Genome."""

        from versal.evolution.init import estimate_initialization

        support_input, support_output = support_loader(task)
        n_inputs = math.prod(int(dim) for dim in support_input.data.shape[1:])
        positions = math.prod(int(dim) for dim in support_output.data.shape[1:])
        n_outputs = model_output_features(support_output.descriptor, positions)
        try:
            init = estimate_initialization(self.evolver.init_kind, n_inputs, n_outputs, **self.evolver.init_params)
        except KeyError as error:
            if runtime.loop.resource_policy.mode == "adaptive":
                return StrategyPreflight(False, "explicit_flat", reason=str(error))
            return StrategyPreflight(True, "explicit_flat", reason=str(error))
        computed = max(0, init.nodes - n_inputs - 1)
        decoded_cells = init.nodes * computed
        # Python genes are a conservative lower-bound proxy (not a claim about exact CPython
        # layout); decoded weights+mask are exact cell counts. Adam/grad state is 12 bytes/cell.
        candidate_bytes = init.nodes * 32 + init.edges * 64 + decoded_cells * 5
        population = max(1, self.evolver.pop_size)
        trainers = population if self.evolver.execution_mode.startswith("population_") else max(1, self.evolver.assess_workers)
        optimizer_bytes = decoded_cells * 12 * trainers
        examples = int(support_input.data.shape[0])
        footprint = StageFootprint(
            stage="direct_population",
            representation=f"explicit_flat/{self.evolver.init_kind}",
            candidate_bytes=candidate_bytes,
            population_size=population,
            optimizer_bytes=optimizer_bytes,
            activation_bytes=examples * init.nodes * 4 * trainers,
            transfer_bytes=(init.nodes * 32 + init.edges * 64) * trainers if self.evolver.assess_workers > 1 else 0,
            work_operations=examples * max(1, init.edges),
            detail=f"{init.nodes} initial nodes, {init.edges} initial edges, {decoded_cells} decoded GraphNet cells",
        )
        decision = runtime.loop.resource_policy.assess_stage(footprint)
        return StrategyPreflight(decision.accepted, footprint.representation, footprint, decision, decision.reason)

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
            from versal.structured import encode_structured_grid

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

    def evaluate_report(self, genome: Genome, task: Task) -> dict[str, float]:
        """Evaluate one support-selected payload on the full task without training."""

        previous_deadline = getattr(self.evolver, "deadline", None)
        previous_callback = getattr(self.evolver, "deadline_exceeded", None)
        self.evolver.deadline = None
        self.evolver.deadline_exceeded = None
        try:
            assessed = self.evolver.evaluate_only(genome, self._adapter(task, include_query=True))
            return {} if assessed.module is None else dict(assessed.metrics)
        finally:
            self.evolver.deadline = previous_deadline
            self.evolver.deadline_exceeded = previous_callback

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
            from versal.evolution.init import estimate_initialization

            init_genes = estimate_initialization(self.evolver.init_kind, flat_inputs, flat_outputs, **self.evolver.init_params).edges
            if self.max_init_genes != "adaptive" and 0 < int(self.max_init_genes) < init_genes:
                return StrategyResult(strategy=self.name, metric=0.0, generations_used=0, champion_metrics={"declined_init_genes": float(init_genes)})
        preflight = self.preflight(task, runtime)
        if not preflight.eligible:
            metrics = preflight.decision.metrics("direct_resource") if preflight.decision is not None else {}
            return StrategyResult(
                strategy=self.name,
                metric=0.0,
                generations_used=0,
                champion_metrics={"declined_preflight": 1.0, **metrics},
                resource_metrics=metrics,
                strategy_metrics={"preflight_ineligible": 1.0},
            )
        if preflight.decision is not None:
            resource_metrics.update(preflight.decision.metrics("direct_resource"))
        adapter = self._adapter(task, include_query=not self.blind_query)
        # The direct population's library-reading mutators must sample from the SAME library the
        # decode-time macro resolver resolves (the orchestrator's attached one), or add_macro_node
        # can graft a macro ref that decode cannot satisfy -> a hard KeyError mid-search.
        self.evolver.library = runtime.library
        # Grid tasks get coordinates stamped on the seed population so the geometry-biased
        # mutators (add_local_node and friends) can grow local receptive fields.
        grid = self._grid_shape(task) if not isinstance(adapter, TemporalTaskAdapter) else None
        original_init = self.evolver.init_op
        self.evolver.deadline_exceeded = runtime.deadline_exceeded
        self.evolver.deadline = runtime.deadline
        if grid is not None:
            from versal.evolution.init import stamp_input_coordinates

            self.evolver.init_op = lambda n_inputs, n_outputs, *, rng: stamp_input_coordinates(original_init(n_inputs, n_outputs, rng=rng), grid)

        def seeded_front(tracker: InnovationTracker) -> list[Genome]:
            # Refine-on-hit warm start: grafted entries take the front of the population and are
            # trained/assessed like every other member. Grid stamping keeps geometry mutators live.
            grafted = [graft(entry, tracker) for entry in (seed_entries or [])]
            grafted.extend(_restamp_genome(genome, tracker) for genome in (seed_genomes or []))
            if grid is not None:
                from versal.evolution.init import stamp_input_coordinates

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
            if runtime.accepted(best) or stop(generation, best) or (self.blind_query and runtime.should_stop()):
                break
            self.evolver.advance(state, adapter)
            if state.topology_exhausted:
                break

        # Verification: the genome PAYLOAD (not the live module object) must reproduce the metric,
        # because the payload is what admission persists and lookups re-decode.
        verification_timed_out = runtime.should_stop() and self.blind_query
        verified: Assessed | None = None
        if not verification_timed_out:
            if best.module is None and runtime.should_stop():
                return StrategyResult(
                    strategy=self.name,
                    metric=0.0,
                    generations_used=generations,
                    champion_metrics=dict(best.metrics),
                    resource_metrics=resource_metrics,
                    strategy_metrics={"deadline_before_fully_evaluated_candidate": 1.0},
                )
            try:
                verified = best if runtime.should_stop() else self.evolver.evaluate_only(best.genome, adapter)
            except TimeoutError:
                if not self.blind_query:
                    raise
                verification_timed_out = True
        if verification_timed_out:
            final_best = max(state.population, key=lambda item: item.fitness)
            if final_best.fitness > best.fitness:
                best = final_best
            return StrategyResult(
                strategy=self.name,
                metric=runtime.metric_of(best),
                generations_used=generations,
                report_candidate_genome=best.genome,
                champion_metrics=dict(best.metrics),
                seed_metric=seed_metric,
                size_metrics=_module_size_metrics(best.genome, state.population),
                resource_metrics=resource_metrics,
                strategy_metrics={"deadline_before_fully_evaluated_candidate": 1.0},
                representation=preflight.representation,
            )
        assert verified is not None
        search_metric = runtime.metric_of(verified)
        return StrategyResult(
            strategy=self.name,
            metric=search_metric,
            generations_used=generations,
            champion_genome=verified.genome,
            champion_metrics=dict(verified.metrics),
            seed_metric=seed_metric,
            size_metrics=_module_size_metrics(verified.genome, state.population),
            resource_metrics=resource_metrics,
            representation=preflight.representation,
        )


@EVOLVE_STRATEGY.register("grammar")
def _build_grammar(config: dict[str, Any]) -> "GrammarStrategy":
    table = config.get("orchestrator", {}).get("grammar", {}) or {}
    return GrammarStrategy(
        direct=_build_direct(config),
        blind_query=bool(config.get("orchestrator", {}).get("blind_query", False)),
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
    blind_query: bool = False
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
        from versal.grammar import crossover_program, mutate_program, rebuild_grammar, seed_program

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
        from versal.grammar import GrammarError, compile_program

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
            result = CompositionStrategy(blind_query=self.blind_query)(task, spec, runtime, budget=remaining, seed_comps=[*(seed_comps or []), *comp_seeds])
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
    from versal.routing import build_routed_strategy

    return build_routed_strategy(config)


def build_strategies(config: dict[str, Any]) -> list[tuple[str, Callable[..., StrategyResult]]]:
    """Resolve `[orchestrator] evolve` (default: composition only, the pre-strategy behavior)."""
    names = [str(name) for name in config.get("orchestrator", {}).get("evolve", ["composition"])]
    return [(name, EVOLVE_STRATEGY.get(name)(config)) for name in names]
