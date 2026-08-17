"""Flat task-shaped network search strategy."""

import math
from dataclasses import dataclass

from versal.dataset.icarus import Level0Encoder, Task, encode_task, model_output_features, support_loader
from versal.evaluation import fit_query_target, input_width, output_features, without_query
from versal.evolution.evolver import Assessed, Evolver, TaskAdapter
from versal.evolution.genome import Genome, InnovationTracker, genome_to_dict
from versal.evolution.loop import CompTaskSpec
from versal.library import MODULE, LibraryEntry, graft, structural_fingerprint
from versal.reference_depth import DEFAULT_MAX_INLINE_DEPTH
from versal.strategy_common import StrategyPreflight, StrategyResult, StrategyRuntime, _module_size_metrics, _restamp_genome
from versal.temporal import TemporalTaskAdapter, has_time_axis, temporal_adapter
from versal.utils.resources import StageFootprint


@dataclass
class DirectStrategy:
    """Evolve task-shaped flat networks on the task's real I/O widths."""

    evolver: Evolver
    name: str = "direct"
    # Arithmetic guards prevent dense initialization from allocating an unsafe population.
    max_flat_outputs: int | str = 0
    max_init_genes: int | str = 0
    structured_grid: bool = False
    blind_query: bool = False

    def preflight(self, task: Task, runtime: StrategyRuntime) -> StrategyPreflight:
        """Price the configured initializer and its decoded compact GraphNet exactly enough to
        reject certain OOMs.  This intentionally does not instantiate a Genome."""

        from versal.evolution.init import estimate_initialization

        support_input, support_output = support_loader(task)
        n_inputs = math.prod(int(dim) for dim in support_input.data.shape[1:])
        positions = math.prod(int(dim) for dim in support_output.data.shape[1:])
        n_outputs = model_output_features(support_output.descriptor, positions)
        if self.max_flat_outputs != "adaptive" and 0 < int(self.max_flat_outputs) < n_outputs:
            limit = int(self.max_flat_outputs)
            return StrategyPreflight(
                False,
                "explicit_flat",
                reason=f"{n_outputs:,} flattened outputs exceed the {limit:,} safety limit",
                metrics={
                    "direct_guard_declined": 1.0,
                    "direct_flat_outputs": float(n_outputs),
                    "direct_max_flat_outputs": float(limit),
                },
            )
        try:
            init = estimate_initialization(self.evolver.init_kind, n_inputs, n_outputs, **self.evolver.init_params)
        except KeyError as error:
            if runtime.loop.resource_policy.mode == "adaptive":
                return StrategyPreflight(False, "explicit_flat", reason=str(error))
            return StrategyPreflight(True, "explicit_flat", reason=str(error))
        if self.max_init_genes != "adaptive" and 0 < int(self.max_init_genes) < init.edges:
            limit = int(self.max_init_genes)
            return StrategyPreflight(
                False,
                "explicit_flat",
                reason=f"{init.edges:,} initialization genes exceed the {limit:,} safety limit",
                metrics={
                    "direct_guard_declined": 1.0,
                    "direct_init_genes": float(init.edges),
                    "direct_max_init_genes": float(limit),
                },
            )
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
        preflight = self.preflight(task, runtime)
        if not preflight.eligible:
            metrics = preflight.decision.metrics("direct_resource") if preflight.decision is not None else {}
            return StrategyResult(
                strategy=self.name,
                metric=0.0,
                generations_used=0,
                champion_metrics={"declined_preflight": 1.0, **metrics},
                resource_metrics=metrics,
                strategy_metrics={"preflight_ineligible": 1.0, **preflight.metrics},
                representation=preflight.representation,
                skip_reason=preflight.reason or "dense representation failed resource preflight",
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
