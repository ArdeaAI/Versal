"""Hierarchical composition search strategy."""

from dataclasses import dataclass

from versal.dataset.icarus import Task
from versal.evolution.composition import CompositionGenome, edge_storage_value_count, glue_value_count
from versal.evolution.evolver import get_shared_pool
from versal.evolution.loop import AssessedComposition, CompTaskSpec
from versal.strategy_common import StrategyResult, StrategyRuntime, comp_size_metrics
from versal.utils.logging import Logger
from versal.utils.resources import StageFootprint, format_bytes

logger = Logger.get_logger()


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
        """Reassess against current modules and allow one refit when a stale winner regresses."""
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
