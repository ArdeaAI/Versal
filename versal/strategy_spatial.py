"""
Ground-up graph recipe search on typed tensors of any rank.
"""

from dataclasses import dataclass
from typing import Any

from versal.dataset.icarus import Task
from versal.evolution.evolver import Evolver
from versal.evolution.genome import Genome
from versal.spatial import SpatialContract, SpatialTaskAdapter
from versal.strategy_common import StrategyPreflight, StrategyResult, StrategyRuntime
from versal.utils.resources import StageFootprint


@dataclass
class SpatialStrategy:
    evolver: Evolver
    blind_query: bool = False
    name: str = "spatial"
    chunk_size: int = 32768
    max_expanded_edges: int = 1_000_000
    max_activation_cells: int = 8_000_000

    def preflight(self, task: Task, runtime: StrategyRuntime) -> StrategyPreflight:
        try:
            contract = SpatialContract.from_task(task)
        except ValueError as error:
            return StrategyPreflight(False, self.name, reason=str(error))
        edges = contract.output_width * 2
        cells = (contract.input_width + contract.output_width + 1) * len(task.support)
        if edges > self.max_expanded_edges or cells > self.max_activation_cells:
            return StrategyPreflight(
                False,
                self.name,
                reason=f"initial recipe needs {edges:,} edges and {cells:,} activation cells; configured limits are {self.max_expanded_edges:,}/{self.max_activation_cells:,}",
            )
        workers = max(1, self.evolver.assess_workers)
        footprint = StageFootprint(
            stage="spatial_population",
            representation=self.name,
            candidate_bytes=edges * 4 + 1024,
            population_size=self.evolver.pop_size,
            optimizer_bytes=edges * 12 * workers,
            activation_bytes=cells * 4 * workers,
            work_operations=edges * len(task.support),
            detail=f"raw tensor {contract.input_shape} -> {contract.output_shape}; sparse neuron recipe",
        )
        decision = runtime.loop.resource_policy.assess_stage(footprint)
        return StrategyPreflight(decision.accepted, self.name, footprint, decision, decision.reason)

    def _adapter(self, task: Task, *, include_query: bool = True) -> SpatialTaskAdapter:
        return SpatialTaskAdapter(
            task,
            include_query=include_query,
            chunk_size=self.chunk_size,
            deadline=self.evolver.deadline,
            library_dir=self.evolver.library_dir,
            max_inline_depth=self.evolver.max_inline_depth,
            max_expanded_edges=self.max_expanded_edges,
            max_activation_cells=self.max_activation_cells,
        )

    def evaluate_report(self, genome: Genome, task: Task) -> dict[str, float]:
        previous_deadline, previous_callback = self.evolver.deadline, self.evolver.deadline_exceeded
        self.evolver.deadline, self.evolver.deadline_exceeded = None, None
        try:
            assessed = self.evolver.evaluate_only(genome, self._adapter(task, include_query=True))
            return {} if assessed.module is None else dict(assessed.metrics)
        finally:
            self.evolver.deadline, self.evolver.deadline_exceeded = previous_deadline, previous_callback

    @staticmethod
    def _grid_shape(task: Task) -> None:
        return None

    def __call__(self, task: Task, spec: Any, runtime: StrategyRuntime, *, budget: int, seed_entries: list | None = None, seed_comps: list | None = None) -> StrategyResult:
        from versal.strategy_sessions import SpatialSession

        session = SpatialSession(self, task, spec, seed=runtime.state.rng.randrange(2**32))
        session.pending = [entry.key for entry in (seed_entries or [])]
        best = StrategyResult(self.name, 0.0, 0)
        if not session.ready(runtime):
            return best
        for _ in range(budget):
            if runtime.should_stop() or runtime.should_shutdown():
                break
            result = session.advance(runtime)
            if result.metric >= best.metric:
                best = result
            if runtime.accepted(type("Candidate", (), {"metrics": result.champion_metrics})()):
                break
        best.generations_used = session.generations
        return best
