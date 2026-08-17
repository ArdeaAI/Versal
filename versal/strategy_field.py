"""Resolution-independent spatial field search strategy."""

import math
from dataclasses import dataclass
from typing import Any

from versal.dataset.icarus import Task
from versal.evolution.evolver import Assessed, Evolver
from versal.evolution.genome import Genome, InnovationTracker, genome_from_dict, genome_to_dict
from versal.evolution.loop import CompTaskSpec
from versal.library import LibraryEntry
from versal.reference_depth import DEFAULT_MAX_INLINE_DEPTH
from versal.strategy_common import StrategyPreflight, StrategyResult, StrategyRuntime, _module_size_metrics, _restamp_genome
from versal.utils.resources import StageFootprint


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
            return StrategyResult(
                self.name,
                0.0,
                0,
                strategy_metrics={"field_ineligible": 1.0},
                skip_reason="support is not an aligned spatial mapping",
            )
        preflight = self.preflight(task, runtime)
        if not preflight.eligible:
            metrics = preflight.decision.metrics("field_resource") if preflight.decision is not None else {}
            return StrategyResult(
                self.name,
                0.0,
                0,
                resource_metrics=metrics,
                strategy_metrics={"field_preflight_ineligible": 1.0, **preflight.metrics},
                skip_reason=preflight.reason or "field representation failed resource preflight",
            )
        all_sites = valid_sites(task.support)
        train_sites = deterministic_sites(all_sites, self.train_sites, salt=f"train:{contract.identity}")
        audit_sites = deterministic_sites(all_sites, self.audit_sites, salt=f"audit:{contract.identity}")
        compatible_seeds: list[LibraryEntry] = []
        for entry in seed_entries or []:
            try:
                from versal.field import payload_field_contract

                if payload_field_contract(entry.payload) == contract:
                    compatible_seeds.append(entry)
            except ValueError:
                continue
        try:
            training = encode_sites(task, train_sites, contract, chunk_size=self.verify_chunk_size, deadline=runtime.deadline)
            audit = encode_sites(task, audit_sites, contract, chunk_size=self.verify_chunk_size, deadline=runtime.deadline)
        except TimeoutError:
            return StrategyResult(
                self.name,
                0.0,
                0,
                strategy_metrics={"field_deadline_stage_feature_preparation": 1.0, "field_seed_count": float(len(compatible_seeds))},
            )
        adapter = FieldAdapter(training, audit, contract, max_inline_depth=self.evolver.max_inline_depth, library=runtime.library)
        self.evolver.library = runtime.library
        self.evolver.deadline_exceeded = runtime.deadline_exceeded
        self.evolver.deadline = runtime.deadline
        self.evolver.topology_tabu = runtime.topology_tabu

        def seeded_front(tracker: InnovationTracker) -> list[Genome]:
            return [_restamp_genome(genome_from_dict(entry.payload), tracker) for entry in compatible_seeds]

        state = self.evolver.seed_state(adapter, runtime.state.rng, seeded_front=seeded_front if compatible_seeds else None)
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
                    "field_seed_count": float(len(compatible_seeds)),
                },
                field_template=contract.to_dict(),
                representation=f"field/{contract.version}",
            )
        if best_full is None:
            metrics = preflight.decision.metrics("field_resource") if preflight.decision is not None else {}
            return StrategyResult(
                self.name,
                0.0,
                generations,
                resource_metrics=metrics,
                strategy_metrics={"field_deadline_before_full_verification": 1.0, "field_seed_count": float(len(compatible_seeds))},
            )
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
                "field_seed_count": float(len(compatible_seeds)),
            },
            field_template=contract.to_dict(),
            representation=f"field/{contract.version}",
        )
