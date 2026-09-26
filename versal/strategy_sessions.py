"""
Stateful search processes advanced by the interleaved orchestration policy.

Sessions own task-local populations and random streams. Shared modules and the router
remain shared learning state; exchanged candidates always reference immutable payloads.
"""

from __future__ import annotations

import base64
import copy
import json
import random
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import torch

from versal.checkpoint import deserialize_rng, serialize_rng
from versal.dataset.icarus import Task
from versal.evolution.composition import (
    CompEdgeGene,
    CompNodeGene,
    CompNodeKind,
    CompositionGenome,
    IndexRun,
    PortMap,
    comp_from_dict,
    comp_to_dict,
    minimal_composition,
)
from versal.evolution.evolver import Assessed, EvolverState
from versal.evolution.genome import Genome, InnovationTracker, genome_from_dict, genome_to_dict
from versal.evolution.loop import AssessedComposition, CompTaskSpec
from versal.evolution.registry import Registry
from versal.evolution.train import _writeback
from versal.library import COMPOSITION, MODULE, graft, module_level
from versal.strategy_common import StrategyResult, StrategyRuntime, _module_size_metrics, _restamp_composition, _restamp_genome, comp_size_metrics
from versal.temporal import TemporalTaskAdapter

SESSION_STRATEGY: Registry = Registry("strategy_session")


def tensor_state(tensor: torch.Tensor) -> dict[str, Any]:
    """
    Encode small optimizer tensors without pickle or lossy decimal conversion.
    """
    value = tensor.detach().cpu().contiguous()
    return {"dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape), "bytes": base64.b64encode(value.reshape(-1).view(torch.uint8).numpy().tobytes()).decode()}


def restore_tensor(data: dict[str, Any]) -> torch.Tensor:
    """
    Restore a tensor written by tensor_state.
    """
    dtype = getattr(torch, data["dtype"])
    raw = bytearray(base64.b64decode(data["bytes"]))
    return torch.frombuffer(raw, dtype=dtype).clone().reshape(data["shape"]) if raw else torch.empty(data["shape"], dtype=dtype)


def capture_torch_rng() -> dict[str, Any]:
    """
    Capture random streams for initialized compute backends.
    """
    result: dict[str, Any] = {"cpu": tensor_state(torch.get_rng_state())}
    if torch.cuda.is_initialized():
        result["cuda"] = [tensor_state(state) for state in torch.cuda.get_rng_state_all()]
    if torch.backends.mps.is_available():
        result["mps"] = tensor_state(torch.mps.get_rng_state())
    return result


def restore_torch_rng(data: dict[str, Any]) -> None:
    """
    Restore backend random streams without replacing their devices.
    """
    torch.set_rng_state(restore_tensor(data["cpu"]))
    if "cuda" in data:
        torch.cuda.set_rng_state_all([restore_tensor(item) for item in data["cuda"]])
    if "mps" in data:
        torch.mps.set_rng_state(restore_tensor(data["mps"]))


def replay_state(replay: list[Any]) -> list[dict[str, Any]]:
    """
    Serialize the router's bounded replay buffer without query examples.
    """
    from dataclasses import asdict

    def descriptor(value: Any) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(value), default=lambda item: item.value))

    return [
        {
            "input": tensor_state(encoded.support_input[0]),
            "input_descriptor": descriptor(encoded.support_input[1]),
            "target": tensor_state(encoded.support_target[0]),
            "target_descriptor": descriptor(encoded.support_target[2]),
            "mask": tensor_state(encoded.support_target[1]) if encoded.support_target[1] is not None else None,
            "input_key": input_key,
            "head_key": head_key,
        }
        for encoded, input_key, head_key, _support_input in replay
    ]


def restore_replay(rows: list[dict[str, Any]]) -> list[Any]:
    """
    Restore the support encodings used by router replay training.
    """
    from versal.dataset.icarus import Axis, EncodedTask, FieldDescriptor, ValueType

    def descriptor(value: dict[str, Any]) -> FieldDescriptor:
        return FieldDescriptor(
            tuple(Axis(axis) for axis in value["axes"]),
            ValueType(value["value_type"]),
            value["n_classes"],
            tuple(value["value_range"]) if value["value_range"] is not None else None,
        )

    replay = []
    for row in rows:
        tensor = restore_tensor(row["input"])
        encoded = EncodedTask(
            (tensor, descriptor(row["input_descriptor"])),
            (restore_tensor(row["target"]), restore_tensor(row["mask"]) if row["mask"] is not None else None, descriptor(row["target_descriptor"])),
            None,
            None,
        )
        replay.append((encoded, row["input_key"], row["head_key"], tensor))
    return replay


def assessed_order(item: Any, runtime: StrategyRuntime) -> tuple[float, float, float]:
    """
    Prefer task score, then expanded structure, independently of search fitness.
    """
    return (runtime.metric_of(item), -float(item.metrics.get("expanded_complexity", 1e12)), float(item.metrics.get("weight_robustness", 0.0)))


def freeze_composition(item: AssessedComposition, runtime: StrategyRuntime, task: Task) -> AssessedComposition:
    """
    Detach the exact scored inner weights from mutable live species.
    """
    comp = item.comp.clone()
    inner_modules = item.net.inner_modules if item.net is not None else {}
    frozen: dict[str, str] = {}
    for node_id in comp.module_ids:
        node = comp.nodes[node_id]
        if not node.ref.startswith("live:"):
            continue
        if node.ref not in frozen:
            genome = runtime.state.species_champions[int(node.ref.removeprefix("live:"))]
            inner = inner_modules.get(node.ref)
            trained = _writeback(genome, inner) if inner is not None else (item.live_writebacks or {}).get(node.ref, genome)
            key = runtime.library.add(
                entry_type=MODULE,
                payload=genome_to_dict(trained),
                io={"inputs": [{"signature": "ANY", "width": node.in_width}], "output": {"signature": "ANY", "width": node.out_width}},
                provenance={"task": task.meta.name, "dependency": True, "session_snapshot": True, "search_lineage": runtime.search_lineage},
                level=module_level(trained, runtime.library),
            )
            frozen[node.ref] = f"library:{key}"
        comp.nodes[node_id] = replace(node, ref=frozen[node.ref])
    return AssessedComposition(comp, dict(item.metrics), item.fitness, None)


class StrategySession:
    """
    One resumable strategy process, with an independent random stream.
    """

    def __init__(self, strategy: Any, task: Task, spec: CompTaskSpec, *, seed: int, saved: dict[str, Any] | None = None, role: str | None = None) -> None:
        self.strategy = strategy
        self.task = task
        self.spec = spec
        self.role = role or strategy.name
        self.saved = saved or {}
        self.seed = seed
        self.rng = deserialize_rng(self.saved["rng"]) if "rng" in self.saved else random.Random(seed)
        self.torch_rng = self.saved.get("torch_rng")
        self.generations = int(self.saved.get("generations", 0))
        self.received = set(self.saved.get("received", []))
        self.pending = list(self.saved.get("pending", []))
        self.skip_reason: str | None = None
        self.refining = False
        self.evaluations = 0
        self.optimizer_steps = 0

    def offer(self, key: str) -> None:
        if key not in self.received:
            self.received.add(key)
            self.pending.append(key)
            self.pending = self.pending[-8:]

    def ready(self, runtime: StrategyRuntime) -> bool:
        return self.skip_reason is None

    @contextmanager
    def bound(self, runtime: StrategyRuntime) -> Iterator[None]:
        original_rng = runtime.state.rng
        original_torch = capture_torch_rng()
        runtime.state.rng = self.rng
        if self.torch_rng is None:
            torch.manual_seed(self.seed)
        else:
            restore_torch_rng(self.torch_rng)
        try:
            yield
        finally:
            self.torch_rng = capture_torch_rng()
            restore_torch_rng(original_torch)
            runtime.state.rng = original_rng

    def advance(self, runtime: StrategyRuntime) -> StrategyResult:
        with self.bound(runtime):
            try:
                result = self._advance(runtime)
            except TimeoutError:
                result = StrategyResult(self.strategy.name, 0.0, 1, skip_reason="support search reached its time limit")
        self.generations += result.generations_used
        return result

    def _advance(self, runtime: StrategyRuntime) -> StrategyResult:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {"rng": serialize_rng(self.rng), "torch_rng": self.torch_rng, "generations": self.generations, "received": sorted(self.received), "pending": self.pending}


@SESSION_STRATEGY.register("direct")
class DirectSession(StrategySession):
    """
    An Evolver population that survives scheduler yields and exact task revisits.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.state: EvolverState | None = None
        self.adapter: Any = None
        self.speciation = self.saved.get("speciation")
        self.seed_genomes = [genome_from_dict(payload) for payload in self.saved.get("seed_genomes", [])]
        self.resource_metrics: dict[str, float] = {}
        self.field_template: dict[str, Any] | None = None
        self.checked = False
        protected = self.saved.get("protected")
        self.protected = Assessed(genome_from_dict(protected["genome"]), dict(protected["metrics"]), float(protected["fitness"]), None) if protected is not None else None

    def ready(self, runtime: StrategyRuntime) -> bool:
        if not self.checked:
            checked = self.strategy.preflight(self.task, runtime)
            self.skip_reason = None if checked.eligible else checked.reason or "resource preflight declined"
            self.resource_metrics = checked.decision.metrics(f"{self.strategy.name}_resource") if checked.decision is not None else {}
            self.checked = True
        return self.skip_reason is None

    def _adapter(self, runtime: StrategyRuntime) -> Any:
        return self.strategy._adapter(self.task, include_query=False)

    def _seeds(self, runtime: StrategyRuntime, tracker: InnovationTracker) -> list[Genome]:
        seeds = [_restamp_genome(genome, tracker) for genome in self.seed_genomes]
        self.seed_genomes = []
        for key in self.pending:
            entry = runtime.library.load(key)
            if entry.entry_type == MODULE and "field_template" not in entry.payload and entry.io == self.spec.io:
                seeds.append(graft(entry, tracker))
        self.pending = []
        return seeds

    def _initialize(self, runtime: StrategyRuntime) -> None:
        evolver = self.strategy.evolver
        self.adapter = self._adapter(runtime)
        saved_state = self.saved.get("population")
        if saved_state is not None:
            population = [
                Assessed(
                    genome_from_dict(item["genome"]),
                    dict(item["metrics"]),
                    float(item["fitness"]),
                    None,
                    descriptor=tuple(item["descriptor"]) if item.get("descriptor") is not None else None,
                )
                for item in saved_state
            ]
            self.state = EvolverState(
                population,
                InnovationTracker.from_dict(self.saved["innovations"]),
                self.rng,
                generation=int(self.saved.get("evolver_generation", 0)),
                species_history=[{int(key): value for key, value in row.items()} for row in self.saved.get("species_history", [])],
                novelty_archive=[tuple(row) for row in self.saved.get("novelty_archive", [])],
            )
            return
        original_init = evolver.init_op
        # Recurrent inputs represent features per time step, not cells across the whole sequence.
        # Match DirectStrategy.__call__: spatial coordinates apply only to non-temporal adapters.
        grid = None if isinstance(self.adapter, TemporalTaskAdapter) else getattr(self.strategy, "_grid_shape", lambda _task: None)(self.task)
        if grid is not None:
            from versal.evolution.init import stamp_input_coordinates

            evolver.init_op = lambda n_inputs, n_outputs, *, rng: stamp_input_coordinates(original_init(n_inputs, n_outputs, rng=rng), grid)
        try:
            self.state = evolver.seed_state(self.adapter, self.rng, seeded_front=lambda tracker: self._seeds(runtime, tracker))
        finally:
            evolver.init_op = original_init

    def _verified(self, runtime: StrategyRuntime) -> StrategyResult:
        assert self.state is not None
        members = sorted(
            [*self.state.population, *([self.protected] if self.refining and self.protected is not None else [])], key=lambda item: assessed_order(item, runtime), reverse=True
        )
        best = members[0]
        verified = self.strategy.evolver.evaluate_only(best.genome, self.adapter)
        if runtime.accepted(verified) and (self.protected is None or assessed_order(verified, runtime) > assessed_order(self.protected, runtime)):
            self.protected = Assessed(verified.genome.clone(), dict(verified.metrics), verified.fitness, verified.module)
        self._pin()
        return StrategyResult(
            self.strategy.name,
            runtime.metric_of(verified),
            1,
            champion_genome=verified.genome.clone(),
            champion_metrics=dict(verified.metrics),
            size_metrics=_module_size_metrics(verified.genome, self.state.population),
            resource_metrics=dict(self.resource_metrics),
            representation=f"explicit_flat/{self.strategy.evolver.init_kind}",
        )

    def _pin(self) -> None:
        """
        Keep the best accepted native parent available to reproduction, without refitting it.
        """
        if not self.refining or self.protected is None or self.state is None:
            return
        population = self.state.population
        payload = genome_to_dict(self.protected.genome)
        if any(genome_to_dict(item.genome) == payload for item in population):
            return
        if len(population) < self.strategy.evolver.pop_size:
            population.append(self.protected)
        else:
            worst = min(range(len(population)), key=lambda index: population[index].fitness)
            population[worst] = self.protected

    def _advance(self, runtime: StrategyRuntime) -> StrategyResult:
        evolver = self.strategy.evolver
        evaluations, steps = evolver.work_evaluations, evolver.work_optimizer_steps
        evolver.library = runtime.library
        evolver.deadline = runtime.deadline
        evolver.deadline_exceeded = runtime.deadline_exceeded
        previous_tabu = evolver.topology_tabu
        previous_species = copy.deepcopy(evolver.speciate.state_dict())
        evolver.topology_tabu = runtime.topology_tabu if self.refining else None
        if self.speciation is not None:
            evolver.speciate.load_state_dict(copy.deepcopy(self.speciation))
        try:
            if self.state is None:
                self._initialize(runtime)
            assert self.state is not None
            self._pin()
            if self.generations:
                seeds = self._seeds(runtime, self.state.innovations)
                if seeds:
                    imported = evolver.assess_many(seeds[:2], self.adapter, self.state)
                    self.state.population = sorted([*self.state.population, *imported], key=lambda item: assessed_order(item, runtime), reverse=True)[: evolver.pop_size]
                evolver.advance(self.state, self.adapter)
            if runtime.on_generation is not None:
                best = max(self.state.population, key=lambda item: assessed_order(item, runtime))
                runtime.on_generation(self.role, self.generations, best, sum(item.fitness for item in self.state.population) / len(self.state.population))
            return self._verified(runtime)
        finally:
            self.evaluations += evolver.work_evaluations - evaluations
            self.optimizer_steps += evolver.work_optimizer_steps - steps
            self.speciation = copy.deepcopy(evolver.speciate.state_dict())
            evolver.speciate.load_state_dict(previous_species)
            evolver.topology_tabu = previous_tabu

    def state_dict(self) -> dict[str, Any]:
        data = super().state_dict()
        data["seed_genomes"] = [genome_to_dict(genome) for genome in self.seed_genomes]
        data["protected"] = (
            {"genome": genome_to_dict(self.protected.genome), "metrics": self.protected.metrics, "fitness": self.protected.fitness} if self.protected is not None else None
        )
        if self.state is None:
            return {**self.saved, **data}
        data.update(
            population=[{"genome": genome_to_dict(item.genome), "metrics": item.metrics, "fitness": item.fitness, "descriptor": item.descriptor} for item in self.state.population],
            innovations=self.state.innovations.to_dict(),
            evolver_generation=self.state.generation,
            species_history=self.state.species_history,
            novelty_archive=self.state.novelty_archive,
            speciation=self.speciation,
        )
        return data


@SESSION_STRATEGY.register("field")
class FieldSession(DirectSession):
    """
    A persistent field population, with full-support verification before comparison.
    """

    def _adapter(self, runtime: StrategyRuntime) -> Any:
        from versal.field import FieldAdapter, deterministic_sites, encode_sites, field_contract, valid_sites

        contract = field_contract(self.task)
        assert contract is not None
        self.field_template = contract.to_dict()
        sites = valid_sites(self.task.support)
        train = deterministic_sites(sites, self.strategy.train_sites, salt=f"train:{contract.identity}")
        audit = deterministic_sites(sites, self.strategy.audit_sites, salt=f"audit:{contract.identity}")
        return FieldAdapter(
            encode_sites(self.task, train, contract, chunk_size=self.strategy.verify_chunk_size, deadline=runtime.deadline),
            encode_sites(self.task, audit, contract, chunk_size=self.strategy.verify_chunk_size, deadline=runtime.deadline),
            contract,
            max_inline_depth=self.strategy.evolver.max_inline_depth,
            library=runtime.library,
        )

    def _seeds(self, runtime: StrategyRuntime, tracker: InnovationTracker) -> list[Genome]:
        seeds = []
        for key in self.pending:
            entry = runtime.library.load(key)
            if entry.payload.get("field_template") == self.field_template:
                seeds.append(_restamp_genome(genome_from_dict(entry.payload), tracker))
        self.pending = []
        return seeds

    def _verified(self, runtime: StrategyRuntime) -> StrategyResult:
        from versal.field import FieldContract, evaluate_field_module

        assert self.state is not None and self.field_template is not None
        contract = FieldContract.from_dict(self.field_template)
        ranked = sorted(self.state.population, key=lambda item: assessed_order(item, runtime), reverse=True)
        candidates = ranked[: self.strategy.verify_top_k]
        if self.refining and self.protected is not None:
            candidates = [self.protected, *candidates]
        verified = []
        for item in candidates:
            if verified and runtime.should_stop():
                break
            module = self.adapter.decode(item.genome)
            metrics = dict(item.metrics)
            metrics.update(evaluate_field_module(module, self.task, contract, split="support", chunk_size=self.strategy.verify_chunk_size, deadline=runtime.deadline))
            verified.append(Assessed(item.genome, metrics, item.fitness, module))
            self.evaluations += 1
        best = max(verified, key=lambda item: assessed_order(item, runtime))
        if runtime.accepted(best) and (self.protected is None or assessed_order(best, runtime) > assessed_order(self.protected, runtime)):
            self.protected = Assessed(best.genome.clone(), dict(best.metrics), best.fitness, best.module)
        self._pin()
        return StrategyResult(
            "field",
            runtime.metric_of(best),
            1,
            champion_genome=best.genome.clone(),
            champion_metrics=best.metrics,
            field_template=self.field_template,
            representation="field",
            resource_metrics=dict(self.resource_metrics),
            size_metrics=_module_size_metrics(best.genome, self.state.population),
        )


@SESSION_STRATEGY.register("composition")
class CompositionSession(StrategySession):
    """
    Keep composition genes across yields while sharing the evolving module pool.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.population: list[AssessedComposition] = []
        self.seed_comps = [comp_from_dict(payload) for payload in self.saved.get("seed_comps", [])]
        self.resource_metrics: dict[str, float] = {}
        self.checked = False
        self.innovations = InnovationTracker.from_dict(self.saved["innovations"]) if self.saved.get("innovations") else None
        protected = self.saved.get("protected")
        self.protected = AssessedComposition(comp_from_dict(protected["comp"]), dict(protected["metrics"]), float(protected["fitness"]), None) if protected is not None else None

    @contextmanager
    def bound(self, runtime: StrategyRuntime) -> Iterator[None]:
        with super().bound(runtime):
            original = runtime.state.comp_innovations
            if self.innovations is None:
                self.innovations = InnovationTracker.from_dict(original.to_dict())
            runtime.state.comp_innovations = self.innovations
            try:
                yield
            finally:
                runtime.state.comp_innovations = original

    def _pin(self, runtime: StrategyRuntime) -> None:
        """
        Preserve the compact native parent with immutable module dependencies.
        """
        if not self.refining or self.protected is None:
            return
        payload = comp_to_dict(self.protected.comp)
        if any(comp_to_dict(item.comp) == payload for item in self.population):
            return
        if len(self.population) < runtime.loop.comp_pop_size:
            self.population.append(self.protected)
        else:
            worst = min(range(len(self.population)), key=lambda index: self.population[index].fitness)
            self.population[worst] = self.protected

    def ready(self, runtime: StrategyRuntime) -> bool:
        if not self.checked:
            checked = self.strategy.preflight_population(self.task, self.spec, runtime, budget=1, seed_comps=self.seed_comps)
            if isinstance(checked, StrategyResult):
                self.skip_reason = "composition resource preflight declined"
            else:
                self.resource_metrics = checked
            self.checked = True
        return self.skip_reason is None

    def _seeds(self, runtime: StrategyRuntime) -> list[CompositionGenome]:
        tracker = runtime.state.comp_innovations
        seeds = [_restamp_composition(comp, tracker) for comp in self.seed_comps]
        self.seed_comps = []
        for key in self.pending:
            entry = runtime.library.load(key)
            if entry.io != self.spec.io:
                continue
            if entry.entry_type == COMPOSITION:
                seeds.append(_restamp_composition(comp_from_dict(entry.payload), tracker))
            elif "field_template" not in entry.payload and len(self.spec.input_specs) == 1:
                # A fixed identity wrapper preserves an offered expert's actual function.
                source, module, target = (tracker.new_node_id() for _ in range(3))
                width = self.spec.input_specs[0][1]
                out = self.spec.output_width
                nodes = {
                    source: CompNodeGene(source, CompNodeKind.INPUT, self.spec.input_specs[0][0], 0, width),
                    module: CompNodeGene(module, CompNodeKind.MODULE, f"library:{key}", width, out, trainable=False),
                    target: CompNodeGene(target, CompNodeKind.OUTPUT, self.spec.output_ref, out, 0),
                }
                edges = [
                    CompEdgeGene(a, b, True, tracker.innovation(a, b), (), port_map=PortMap((IndexRun(0, 0, size),)))
                    for a, b, size in [(source, module, width), (module, target, out)]
                ]
                seeds.append(CompositionGenome(nodes, edges))
        self.pending = []
        return seeds

    def _advance(self, runtime: StrategyRuntime) -> StrategyResult:
        loop, state = runtime.loop, runtime.state
        evaluations, steps = loop.work_evaluations, loop.work_optimizer_steps
        loop.evolver.deadline = runtime.deadline
        loop.evolver.deadline_exceeded = runtime.deadline_exceeded
        previous_tabu = loop.topology_tabu
        loop.topology_tabu = runtime.topology_tabu if self.refining else None
        state.topology_exhausted = False
        try:
            if not self.population:
                stored = self.saved.get("population")
                genes = [comp_from_dict(row["comp"]) for row in stored] if stored else self._seeds(runtime)
                while len(genes) < loop.comp_pop_size:
                    genes.append(
                        minimal_composition(
                            self.spec.input_specs,
                            self.spec.output_ref,
                            self.spec.output_width,
                            state.comp_innovations,
                            self.rng,
                            glue_scale=loop.glue_scale,
                            glue_rank=loop.glue_rank,
                            glue_rank_threshold=loop.glue_rank_threshold,
                            glue_storage=loop.glue_storage,
                        )
                    )
                self.population = loop._assess_all([loop._repair_refs(comp, state) for comp in genes[: loop.comp_pop_size]], self.spec, state, train=not bool(stored))
            self._pin(runtime)
            if self.generations:
                if loop.advance_every > 0 and self.generations % loop.advance_every == 0:
                    loop.advance_modules(state)
                # Other sessions can change shared modules between quanta. Never select parents
                # using fitness measured against an earlier set of live weights.
                genes = [loop._repair_refs(item.comp, state) for item in self.population]
                self.population = loop._assess_all(genes, self.spec, state, train=False)
                seeds = self._seeds(runtime)[:2]
                if seeds:
                    imported = loop._assess_all(seeds, self.spec, state, train=True)
                    self.population = sorted([*self.population, *imported], key=lambda item: assessed_order(item, runtime), reverse=True)[: loop.comp_pop_size]
                self.population = loop._reproduce_comps(self.population, self.spec, state)
            best = max([*self.population, *([self.protected] if self.refining and self.protected is not None else [])], key=lambda item: assessed_order(item, runtime))
            loop._restore_champion_net(best, self.spec, state)
            frozen = freeze_composition(best, runtime, self.task)
            verified = loop.assess_composition(frozen.comp, self.spec, state, train=False)
            if runtime.accepted(verified) and (self.protected is None or assessed_order(verified, runtime) > assessed_order(self.protected, runtime)):
                self.protected = AssessedComposition(verified.comp.clone(), dict(verified.metrics), verified.fitness, None)
            loop._attribute(self.population, state)
            loop._module_writeback(self.population, state)
            self._pin(runtime)
            state.generation += 1
            if runtime.on_generation is not None:
                runtime.on_generation(self.role, self.generations, verified, sum(item.fitness for item in self.population) / len(self.population))
            return StrategyResult(
                "composition",
                runtime.metric_of(verified),
                1,
                champion_comp=verified,
                champion_metrics=dict(verified.metrics),
                size_metrics=comp_size_metrics(verified.comp),
                resource_metrics=dict(self.resource_metrics),
                representation="composition",
            )
        finally:
            self.evaluations += loop.work_evaluations - evaluations
            self.optimizer_steps += loop.work_optimizer_steps - steps
            loop.topology_tabu = previous_tabu

    def state_dict(self) -> dict[str, Any]:
        data = super().state_dict()
        data["seed_comps"] = [comp_to_dict(comp) for comp in self.seed_comps]
        data["innovations"] = self.innovations.to_dict() if self.innovations is not None else None
        data["protected"] = (
            {"comp": comp_to_dict(self.protected.comp), "metrics": self.protected.metrics, "fitness": self.protected.fitness} if self.protected is not None else None
        )
        if not self.population:
            return {**self.saved, **data}
        data["population"] = [{"comp": comp_to_dict(item.comp)} for item in self.population]
        return data


@SESSION_STRATEGY.register("grammar")
class GrammarSession(StrategySession):
    """
    Compile independently supported motifs into persistent child populations.

    A child generation consumes the grammar quantum; it never launches a nested run.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from versal.strategy_composition import CompositionStrategy

        super().__init__(*args, **kwargs)
        self.children = {
            "direct": DirectSession(self.strategy.direct, self.task, self.spec, seed=self.seed + 1, saved=self.saved.get("direct"), role="grammar"),
            "composition": CompositionSession(
                CompositionStrategy(blind_query=self.strategy.blind_query), self.task, self.spec, seed=self.seed + 2, saved=self.saved.get("composition"), role="grammar"
            ),
        }
        self.compiled = set(self.saved.get("compiled", []))
        self.active = set(self.saved.get("active", []))
        self.catalog: tuple[str, ...] | None = None

    def ready(self, runtime: StrategyRuntime) -> bool:
        from versal.grammar import GrammarError, compile_program

        catalog = tuple(runtime.library.keys())
        if catalog != self.catalog:
            self.catalog = catalog
            with self.bound(runtime):
                for program in self.strategy._programs(runtime):
                    identity = repr(program.to_dict())
                    if identity in self.compiled:
                        continue
                    self.compiled.add(identity)
                    try:
                        compiled = compile_program(program, self.strategy._grammar, library=runtime.library, rng=self.rng)
                    except (GrammarError, KeyError, ValueError):
                        continue
                    if isinstance(compiled, Genome) and len(compiled.input_ids) == self.spec.n_inputs and len(compiled.output_ids) == self.spec.output_width:
                        self.children["direct"].seed_genomes.append(compiled)
                        self.active.add("direct")
                    elif isinstance(compiled, CompositionGenome) and self.strategy._composition_compatible(compiled, self.spec):
                        self.children["composition"].seed_comps.append(compiled)
                        self.active.add("composition")
        self.skip_reason = None if self.active else "no compatible independently supported grammar productions"
        return bool(self.active)

    def _advance(self, runtime: StrategyRuntime) -> StrategyResult:
        ready = [name for name in sorted(self.active) if self.children[name].ready(runtime)]
        if not ready:
            return StrategyResult("grammar", 0.0, 0, skip_reason="grammar children declined resource allocation")
        child = self.children[ready[self.generations % len(ready)]]
        child.refining = self.refining
        before, steps = child.evaluations, child.optimizer_steps
        result = child.advance(runtime)
        self.evaluations += child.evaluations - before
        self.optimizer_steps += child.optimizer_steps - steps
        result.strategy = "grammar"
        result.champion_metrics["grammar_programs"] = float(len(self.compiled))
        return result

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), **{name: child.state_dict() for name, child in self.children.items()}, "active": sorted(self.active), "compiled": sorted(self.compiled)}


@SESSION_STRATEGY.register("routed")
class RoutedSession(StrategySession):
    """
    Slice router training while retaining Adam moments for this exact task.

    Distillation is a separate charged quantum. Experts remain shared immutable entries.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer: Any = None
        self.optimizer_saved = dict(self.saved.get("optimizer", {}))
        self.distillation = self.saved.get("distillation")
        self.view: Any = None
        self.service: Any = None
        self.catalog: tuple[str, ...] | None = None
        self.used = False
        self.train_steps = int(self.saved.get("train_steps", 0))

    def ready(self, runtime: StrategyRuntime) -> bool:
        # Constructing adapters consumes randomness, so eligibility is bound too.
        with self.bound(runtime):
            loading = self.strategy.service is None and self.strategy.persist and (Path(self.strategy.library_dir) / "router" / "router_meta.json").exists()
            original = capture_torch_rng() if loading else None
            self.service = self.strategy._service(runtime.library)
            if original is not None:
                restore_torch_rng(original)
            catalog = tuple(runtime.library.keys())
            if catalog != self.catalog:
                self.service.sync(include_compositions=self.strategy.include_compositions, exclude_temporal=self.strategy.exclude_temporal, render=False)
                self.catalog = catalog
            net = self.service.net
            available = any(name not in net._retired for name in net._vertex_order)
        self.skip_reason = None if available or not self.strategy.distill else "no eligible library experts"
        return self.skip_reason is None

    def _optimizer(self) -> Any:
        net = self.service.net
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.strategy.lr, weight_decay=self.strategy.weight_decay)
        self.strategy._sync_optimizer_parameters(self.optimizer, net)
        for name, parameter in net.named_parameters():
            if name in self.optimizer_saved:
                saved = self.optimizer_saved.pop(name)
                self.optimizer.state[parameter] = {key: restore_tensor(value).to(parameter.device) if isinstance(value, dict) else value for key, value in saved.items()}
        return self.optimizer

    def _advance(self, runtime: StrategyRuntime) -> StrategyResult:
        from versal.evaluation import evaluate, support_loss
        from versal.routing import RoutedTaskView

        net = self.service.net
        io = self.spec.io
        assert io is not None
        input_key = net.ensure_input_adapter(io["inputs"][0]["signature"], io["inputs"][0]["width"])
        head_key = net.ensure_output_head(io["output"]["signature"], io["output"]["width"])
        support_input = self.spec.encoded.support_input[0]
        self.view = RoutedTaskView(net, input_key=input_key, head_key=head_key, support_input=support_input)
        self.used = True
        if self.generations == 0 and self.strategy.zero_shot_accept and self.distillation is None:
            with torch.no_grad():
                initial = dict(evaluate(self.view, self.spec.encoded, self.spec.encoder))
            self.evaluations += 1
            if self.strategy.distill and runtime.accepted(self.strategy._metrics_view(initial)):
                self.distillation = {"pathway": self.strategy._dominant_pathway(self.view)}
        if self.distillation is not None:
            pending, self.distillation = self.distillation, None
            assessed = self.strategy._verify_distilled(pending["pathway"], self.spec, runtime)
            self.evaluations += 1
            self.optimizer_steps += int(assessed.metrics.get("training_optimizer_steps", 0)) if assessed is not None else 0
            metric = runtime.metric_of(assessed) if assessed is not None else 0.0
            # Capture the current router only after its own support evaluation. Its diagnostic
            # candidate and the distilled artifact have explicit, separate scores.
            with torch.no_grad():
                metrics = dict(evaluate(self.view, self.spec.encoded, self.spec.encoder))
            router_metric = runtime.metric_of(self.strategy._metrics_view(metrics))
            candidate = self.strategy._report_candidate(self.service, self.view, metric=router_metric, zero_shot=False, steps_used=self.optimizer_steps, metrics=metrics)
            return StrategyResult(
                "routed",
                metric,
                1,
                champion_comp=assessed,
                champion_metrics=dict(assessed.metrics) if assessed else {},
                report_candidate_routed=candidate,
                report_candidate_metrics=metrics,
                strategy_metrics={"router_score": router_metric, "distilled_score": metric, "distillation_gap": router_metric - metric},
                size_metrics=comp_size_metrics(assessed.comp) if assessed else {},
                representation="composition",
                resource_metrics=dict(self.strategy._last_distill_resource_metrics),
            )
        for _step in range(self.strategy._step_cap(1)):
            if runtime.should_stop() or runtime.should_shutdown():
                break
            # A lazy forward can materialize trainable vertex adapters.
            net.zero_grad(set_to_none=True)
            loss = support_loss(self.view, self.spec.encoded) + self.strategy.load_balance_weight * net.last_aux_loss
            if not torch.isfinite(loss):
                break
            optimizer = self._optimizer()
            loss.backward()
            optimizer.step()
            self.optimizer_steps += 1
            self.train_steps += 1
            if self.strategy._replay and self.strategy.replay_every > 0 and self.train_steps % self.strategy.replay_every == 0:
                self.optimizer_steps += self.strategy._replay_step(optimizer)
        with torch.no_grad():
            metrics = dict(evaluate(self.view, self.spec.encoded, self.spec.encoder))
        self.evaluations += 1
        metric = runtime.metric_of(self.strategy._metrics_view(metrics))
        candidate = self.strategy._report_candidate(self.service, self.view, metric=metric, zero_shot=False, steps_used=self.optimizer_steps, metrics=metrics)
        if self.strategy.distill and runtime.accepted(self.strategy._metrics_view(metrics)):
            self.distillation = {"pathway": self.strategy._dominant_pathway(self.view)}
        if runtime.on_generation is not None:
            runtime.on_generation("routed", self.generations, self.strategy._metrics_view(metrics), metric)
        return StrategyResult(
            "routed",
            metric if not self.strategy.distill else 0.0,
            1,
            champion_routed=candidate if not self.strategy.distill else None,
            champion_metrics=metrics if not self.strategy.distill else {},
            report_candidate_routed=candidate,
            report_candidate_metrics=metrics,
            strategy_metrics={"router_score": metric},
            representation="routed",
        )

    def finish(self) -> None:
        if self.used:
            self.service.record_traffic()
            self.service.record_task({"task": self.task.meta.name, "rung": self.task.meta.rung, "steps_used": self.optimizer_steps, "interleaved": True})
            self.strategy._remember_for_replay(self.spec, self.view.input_key, self.view.head_key, self.view.support_input)
            self.service.save()

    def state_dict(self) -> dict[str, Any]:
        optimizer = dict(self.optimizer_saved)
        if self.optimizer is not None:
            for name, parameter in self.service.net.named_parameters():
                if parameter in self.optimizer.state:
                    optimizer[name] = {key: tensor_state(value) if isinstance(value, torch.Tensor) else value for key, value in self.optimizer.state[parameter].items()}
        return {**super().state_dict(), "optimizer": optimizer, "distillation": self.distillation, "train_steps": self.train_steps}
