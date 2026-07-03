"""Evolve strategies: HOW the orchestrator's evolve step searches, selectable from config.

The ladder's step 2 used to be hardcoded to the hierarchical composition loop. It is now a
config-ordered list of registered strategies (`[orchestrator] evolve = ["composition", "direct"]`)
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

from dataclasses import dataclass, field
from typing import Any, Callable

from ardevo.dataset.icarus import Level0Encoder, Task, encode_task, support_loader
from ardevo.evaluation import input_width, output_features
from ardevo.evolution.evolver import Assessed, Evolver, TaskAdapter
from ardevo.evolution.genome import Genome
from ardevo.evolution.loop import AssessedComposition, CompTaskSpec, HierarchicalLoop, HierarchicalState
from ardevo.evolution.registry import Registry, build_evolver
from ardevo.library import ModuleLibrary
from ardevo.temporal import TemporalTaskAdapter, has_time_axis, temporal_adapter
from ardevo.utils.logging import Logger

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


@dataclass
class StrategyResult:
    strategy: str
    metric: float
    generations_used: int
    champion_comp: AssessedComposition | None = None  # composition-shaped winner (verified fresh)
    champion_genome: Genome | None = None  # module-shaped winner (trained weights written back)
    champion_metrics: dict[str, float] = field(default_factory=dict)


@EVOLVE_STRATEGY.register("composition")
def _build_composition(config: dict[str, Any]) -> "CompositionStrategy":
    return CompositionStrategy()


@dataclass
class CompositionStrategy:
    name: str = "composition"

    def __call__(
        self,
        task: Task,
        spec: CompTaskSpec,
        runtime: StrategyRuntime,
        *,
        budget: int,
        seed_comps: list | None = None,
    ) -> StrategyResult:
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
        if runtime.metric_of(fresh) >= runtime.accept_threshold:
            return fresh
        if runtime.metric_of(best) >= runtime.accept_threshold:
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
    evolution["assess_workers"] = int(table.get("assess_workers", 0))
    overlay["library_dir"] = config.get("orchestrator", {}).get("library_dir", "library")
    # Single-task structure growth usually wants a stronger inner trainer (and sometimes a
    # different mutation recipe) than the composition loop's glue fitting; both are overridable.
    for overridable in ("mutation", "train", "evaluate"):
        if overridable in table:
            evolution[overridable] = table[overridable]
    overlay["evolution"] = evolution
    return DirectStrategy(evolver=build_evolver(overlay))


@dataclass
class DirectStrategy:
    """The flat phase-1 recipe on the task's real I/O; the structure-growing fallback that solves
    tasks the composition indirection stalls on (two-spirals class), and the path that admits
    TASK-SHAPED modules into the library."""

    evolver: Evolver
    name: str = "direct"

    def _adapter(self, task: Task) -> TaskAdapter | TemporalTaskAdapter:
        support_input, _support_output = support_loader(task)
        if has_time_axis(support_input.descriptor):
            return temporal_adapter(task)  # recurrence goes LIVE: decode_recurrent + BPTT
        width = 1
        for dim in support_input.data.shape[1:]:
            width *= int(dim)
        encoder = Level0Encoder(max_flat_dim=width)
        encoded = encode_task(task, encoder)
        return TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded))

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
    ) -> StrategyResult:
        adapter = self._adapter(task)
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
        try:
            # Shared rng: keeps the whole solve deterministic per seed and checkpoint-coherent.
            state = self.evolver.seed_state(adapter, runtime.state.rng)
        finally:
            self.evolver.init_op = original_init
        stop = runtime.stall_factory(budget)
        best: Assessed = max(state.population, key=lambda item: item.fitness)
        generations = 0
        for generation in range(budget):
            generation_best = max(state.population, key=lambda item: item.fitness)
            if generation_best.fitness > best.fitness:
                best = generation_best
            if runtime.on_generation is not None:
                mean_fitness = sum(item.fitness for item in state.population) / len(state.population)
                runtime.on_generation(self.name, generation, generation_best, mean_fitness)
            runtime.state.generation += 1  # the global clock spans strategies for monotonic logging
            generations = generation + 1
            if runtime.metric_of(best) >= runtime.accept_threshold or stop(generation, best):
                break
            self.evolver.advance(state, adapter)

        # Verification: the genome PAYLOAD (not the live module object) must reproduce the metric,
        # because the payload is what admission persists and lookups re-decode.
        verified = self.evolver.evaluate_only(best.genome, adapter)
        return StrategyResult(
            strategy=self.name,
            metric=runtime.metric_of(verified),
            generations_used=generations,
            champion_genome=verified.genome,
            champion_metrics=dict(verified.metrics),
        )


def build_strategies(config: dict[str, Any]) -> list[tuple[str, Callable[..., StrategyResult]]]:
    """Resolve `[orchestrator] evolve` (default: composition only, the pre-strategy behavior)."""
    names = [str(name) for name in config.get("orchestrator", {}).get("evolve", ["composition"])]
    return [(name, EVOLVE_STRATEGY.get(name)(config)) for name in names]
