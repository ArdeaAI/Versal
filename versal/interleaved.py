"""
Persistent, support-only competition between cooperating search processes.

An immutable task incumbent is separate from the diversity archive. Population snapshots
are compressed, content-addressed files, so only the active task occupies memory and a
between-task checkpoint can retain its exact index even after later task revisits.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from versal.dataset.icarus import Task
from versal.evolution.composition import comp_from_dict, comp_to_dict
from versal.evolution.genome import Genome, genome_from_dict, genome_to_dict
from versal.evolution.loop import AssessedComposition, CompTaskSpec
from versal.library import COMPOSITION, MODULE, expanded_payload_complexity, module_level, payload_refs
from versal.orchestrator_types import RefinementRank, refinement_improves
from versal.strategy_common import StrategyResult, _module_size_metrics, comp_size_metrics
from versal.strategy_sessions import SESSION_STRATEGY, CompositionSession, RoutedSession, StrategySession, replay_state, restore_replay
from versal.topology import TopologyRecord, TopologyTabuSession, TopologyTabuStore, task_content_fingerprint


@dataclass
class TaskSearchState:
    """
    Durable task-local state; query tensors and metrics never enter this record.
    """

    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    deficits: dict[str, float] = field(default_factory=dict)
    incumbent: dict[str, Any] | None = None
    frontier: dict[str, dict[str, Any]] = field(default_factory=dict)
    tabu: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    encounters: int = 0


class SnapshotTabuStore(TopologyTabuStore):
    """
    Store topology history inside the same atomic task snapshot as its populations.
    """

    def __init__(self, records: dict[str, list[dict[str, Any]]]) -> None:
        self.records = records

    def load_bucket(self, context: str, bucket: str) -> list[TopologyRecord]:
        return [TopologyRecord(bucket, graph) for graph in self.records.get(bucket, [])]

    def append(self, context: str, records: list[TopologyRecord]) -> None:
        for record in records:
            self.records.setdefault(record.bucket, []).append(record.graph)


@dataclass
class ExecutableTabuSession(TopologyTabuSession):
    """
    Deduplicate fitted candidates without banning a topology's future weight states.

    Persistent populations inherit trained weights. A topology-only lifetime ban prevents
    those weights from improving and can block later pruning. Exact payload fingerprints
    deliberately err toward extra evaluations for renamed equivalent graphs. Search-only
    mutation rates and growth hints do not change the executable and are excluded.
    """

    live_genomes: Callable[[], dict[int, Genome]] | None = None

    def _insert_if_new(self, entry_type: str, payload: dict[str, Any]) -> bool:
        if self._past_deadline():
            return False
        parameters = {key: value for key, value in payload.items() if key not in {"operator_rates", "growth_hints"}}
        dependencies = {}
        if entry_type == COMPOSITION and self.live_genomes is not None:
            live = self.live_genomes()
            for node in payload.get("nodes", []):
                reference = node.get("ref", "")
                if reference.startswith("live:") and int(reference.removeprefix("live:")) in live:
                    genome = genome_to_dict(live[int(reference.removeprefix("live:"))])
                    dependencies[reference] = {key: value for key, value in genome.items() if key not in {"operator_rates", "growth_hints"}}
        digest = hashlib.sha256(json.dumps([entry_type, parameters, dependencies], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        bucket = f"executable-v1:{digest}"
        known = self._buckets.setdefault(bucket, self.store.load_bucket(self.context, bucket))
        if known:
            return False
        record = TopologyRecord(bucket, {"executable_sha256": digest})
        known.append(record)
        self.pending.append(record)
        return True

    def metrics(self) -> dict[str, float]:
        return {key.replace("topology_", "dedup_"): value for key, value in super().metrics().items()} | {"deduplication_parameter_aware": 1.0}


def candidate_payload(result: StrategyResult) -> tuple[str, dict[str, Any]] | None:
    """
    Return the reusable executable representation of a candidate.
    """
    if result.champion_genome is not None:
        payload = genome_to_dict(result.champion_genome)
        if result.field_template is not None:
            payload["field_template"] = result.field_template
        return MODULE, payload
    if result.champion_comp is not None:
        return COMPOSITION, comp_to_dict(result.champion_comp.comp)
    return None


def result_record(result: StrategyResult, key: str) -> dict[str, Any]:
    """
    Retain support evidence only; reporting runs after the final selection.
    """
    return {
        "key": key,
        "strategy": result.strategy,
        "metric": result.metric,
        "metrics": {name: value for name, value in result.champion_metrics.items() if not name.startswith("query_")},
        "validation_status": result.validation_status,
        "validation_metrics": dict(result.validation_metrics),
        "representation": result.representation,
    }


class InterleavedSearch:
    """
    Deterministic weighted round-robin selection with a shared post-solve allowance.
    """

    def __init__(self, orchestrator: Any, config: dict[str, Any]) -> None:
        self.owner = orchestrator
        self.root = orchestrator.library.root / "search"
        self.seed = int(config.get("seed", config.get("run", {}).get("seed", 0)))
        contract = copy.deepcopy({key: config.get(key) for key in ("substrate", "evolution", "fitness", "resources", "orchestrator", "library", "machine_env", "tf32")})
        orchestration = contract.get("orchestrator")
        if orchestration:
            orchestration.pop("library_dir", None)
            orchestration.pop("tasks", None)
        contract["seed"] = self.seed
        self.config_identity = hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()
        self.index: dict[str, dict[str, Any]] = {}
        self.restored_replay: list[dict[str, Any]] | None = None
        if (self.root / "index.json").exists():
            self.index = json.loads((self.root / "index.json").read_text())

    def identity(self, task: Task) -> str:
        return hashlib.sha256(f"2:{self.config_identity}:{task_content_fingerprint(task)}".encode()).hexdigest()

    def incumbent_key(self, task: Task) -> str | None:
        return self.index.get(self.identity(task), {}).get("incumbent")

    def state_dict(self) -> dict[str, Any]:
        routed = dict(self.owner.strategies).get("routed")
        replay = self.restored_replay if self.restored_replay is not None else replay_state(routed._replay) if routed is not None else []
        return {"version": 1, "index": copy.deepcopy(self.index), "router_replay": replay}

    def load_state_dict(self, data: dict[str, Any]) -> None:
        if int(data.get("version", 1)) != 1:
            raise ValueError("unsupported interleaved search checkpoint")
        self.index = copy.deepcopy(data.get("index", {}))
        self.restored_replay = data.get("router_replay", [])
        for record in self.index.values():
            if not (self.root / record["file"]).exists():
                raise FileNotFoundError(f"missing search population snapshot: {record['file']}")

    def _load(self, identity: str) -> TaskSearchState:
        record = self.index.get(identity)
        if record is None:
            return TaskSearchState()
        raw = gzip.decompress((self.root / record["file"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("search population snapshot checksum mismatch")
        return TaskSearchState(**json.loads(raw))

    def _save(self, identity: str, state: TaskSearchState) -> None:
        from dataclasses import asdict

        data = asdict(state)
        references: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if "nodes" in value and "connections" in value:
                    references.update(payload_refs(MODULE, value))
                elif "nodes" in value and "edges" in value:
                    references.update(payload_refs(COMPOSITION, value))
                for name, item in value.items():
                    if name == "key" and isinstance(item, str):
                        references.add(item)
                    elif name == "pending" and isinstance(item, list):
                        references.update(item)
                    elif name == "pathway" and isinstance(item, list):
                        references.update(key for step in item for key in step)
                    else:
                        visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(data)
        raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        filename = f"{identity[:16]}-{digest}.json.gz"
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / filename
        if not destination.exists():
            temporary = destination.with_suffix(".tmp")
            temporary.write_bytes(gzip.compress(raw, mtime=0))
            temporary.replace(destination)
        self.index[identity] = {
            "file": filename,
            "sha256": digest,
            "incumbent": state.incumbent["key"] if state.incumbent else None,
            "protect": sorted(references),
            "encounters": state.encounters,
        }
        temporary = self.root / "index.tmp"
        temporary.write_text(json.dumps(self.index, sort_keys=True))
        temporary.replace(self.root / "index.json")

    def from_record(self, record: dict[str, Any]) -> StrategyResult:
        entry = self.owner.library.load(record["key"])
        metrics = dict(record["metrics"])
        result = StrategyResult(
            record["strategy"],
            float(record["metric"]),
            0,
            champion_metrics=metrics,
            validation_status=record["validation_status"],
            validation_metrics=dict(record["validation_metrics"]),
            representation=record.get("representation"),
            candidate_id=entry.key,
        )
        if entry.entry_type == MODULE:
            result.champion_genome = genome_from_dict(entry.payload)
            result.field_template = entry.payload.get("field_template")
            result.size_metrics = _module_size_metrics(result.champion_genome, [])
        else:
            result.champion_comp = AssessedComposition(comp_from_dict(entry.payload), metrics, 0.0, None)
            result.size_metrics = comp_size_metrics(result.champion_comp.comp)
        result.size_metrics["expanded_complexity"] = float(expanded_payload_complexity(entry.entry_type, entry.payload, self.owner.library))
        return result

    def rank(self, result: StrategyResult) -> RefinementRank | None:
        rank = self.owner._candidate_rank(result)
        if rank is None and result.champion_routed is not None:
            net = getattr(result.champion_routed, "snapshot", None)
            if net is not None:
                # Adapter-only wins are allowed when distillation is explicitly off. Their
                # full routing machinery is charged, and they remain solved-but-unshelved.
                cost = sum(parameter.numel() for parameter in net.parameters())
                for name, vertex in net._vertices.items():
                    if name not in net._retired:
                        entry = self.owner.library.load(vertex.original_key)
                        cost += net.max_steps * expanded_payload_complexity(entry.entry_type, entry.payload, self.owner.library)
                rank = RefinementRank(result.metric, float(result.champion_metrics.get("weight_robustness", 0.0)), cost, "routed")
        return replace(rank, metric=self.owner._accept_value(result.champion_metrics)) if rank is not None else None

    def improves(self, candidate: StrategyResult, incumbent: StrategyResult | None) -> bool:
        if incumbent is None:
            return self.rank(candidate) is not None
        left, right = self.rank(candidate), self.rank(incumbent)
        return (
            left is not None
            and right is not None
            and refinement_improves(
                left,
                right,
                metric_epsilon=self.owner.refine_metric_epsilon,
                robustness_epsilon=self.owner.refine_robustness_epsilon,
            )
        )

    def _publish(self, result: StrategyResult, task: Task, spec: CompTaskSpec, identity: str) -> str:
        packed = candidate_payload(result)
        assert packed is not None
        entry_type, payload = packed
        library = self.owner.library
        io = spec.io or self.owner._io_of(task, spec)
        if result.field_template is not None:
            from versal.field import FieldContract

            io = dict(io) | {"field_identity": FieldContract.from_dict(result.field_template).identity}
        level = (
            module_level(result.champion_genome, library)
            if result.champion_genome is not None
            else 1 + max((library.load(key).level for key in payload_refs(entry_type, payload)), default=0)
        )
        key = library.add(
            entry_type=entry_type,
            payload=payload,
            io=io,
            level=level,
            provenance={
                "task": task.meta.name,
                "rung": task.meta.rung,
                "strategy": result.strategy,
                "dependency": True,
                "stepping_stone": True,
                "search_lineage": identity,
                "accepted_metric": self.owner._accept_value(result.champion_metrics),
                "weight_robustness": result.champion_metrics.get("weight_robustness", 0.0),
                "validation_status": result.validation_status,
            },
        )
        result.candidate_id = key
        rank = self.rank(result)
        if rank is not None:
            result.size_metrics["expanded_complexity"] = float(rank.complexity)
        return key

    def run(
        self,
        task: Task,
        spec: CompTaskSpec,
        budget: int,
        *,
        incumbent_key: str | None = None,
        seed_comps: list[Any] | None = None,
        seed_entries: list[Any] | None = None,
    ) -> StrategyResult:
        owner = self.owner
        identity = self.identity(task)
        state = self._load(identity)
        runtime = owner._runtime()
        runtime.search_lineage = identity
        if self.restored_replay is not None:
            routed = dict(owner.strategies).get("routed")
            if routed is not None:
                routed._replay = restore_replay(self.restored_replay)
            self.restored_replay = None
        sessions: dict[str, StrategySession] = {}
        support_task = Task(task.meta, task.support, [])
        for name, strategy in owner.strategies:
            seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{task_content_fingerprint(task)}:{name}".encode()).digest()[:4], "big")
            sessions[name] = SESSION_STRATEGY.get(name)(strategy, support_task, spec, seed=seed, saved=state.sessions.get(name))
        composition = sessions.get("composition")
        if seed_comps and isinstance(composition, CompositionSession):
            composition.seed_comps.extend(seed_comps)
        for entry in seed_entries or []:
            for session in sessions.values():
                session.offer(entry.key)
        incumbent = self.from_record(state.incumbent) if state.incumbent else None
        if incumbent_key is not None and (incumbent is None or incumbent.candidate_id != incumbent_key):
            entry = owner.library.load(incumbent_key)
            assessed = owner._quick_assessment(entry, task, spec)
            if assessed is not None and owner._accepts_item(assessed):
                record = {
                    "key": incumbent_key,
                    "strategy": "lookup",
                    "metric": owner._metric(assessed),
                    "metrics": assessed.metrics,
                    "validation_status": "not_run",
                    "validation_metrics": {},
                    "representation": entry.entry_type,
                }
                candidate = owner._cross_validate_result(self.from_record(record), task)
                if owner._accepts_result(candidate) and self.improves(candidate, incumbent):
                    incumbent = candidate
                    state.incumbent = result_record(candidate, incumbent_key)
        for record in [*state.frontier.values(), *([state.incumbent] if state.incumbent else [])]:
            for session in sessions.values():
                session.offer(record["key"])
        started_with_incumbent = incumbent is not None
        phase = "refine" if started_with_incumbent else "solve"
        depth = getattr(owner, "_display_depth", 0)
        refine_limit = owner.refine_budget_k if depth <= owner.refine_depth_max else 0
        if started_with_incumbent:
            assert incumbent is not None and incumbent.candidate_id is not None
            refine_limit = owner._effective_refine_budget(owner.library.load(incumbent.candidate_id)) if depth <= owner.refine_depth_max else 0
        remaining = refine_limit if started_with_incumbent else budget
        used, refined = 0, 0
        best_report: StrategyResult | None = None
        work = {name: {"generations": 0.0, "evaluations": 0.0, "optimizer_steps": 0.0, "seconds": 0.0, "preparation_steps": 0.0, "preparation_seconds": 0.0} for name in sessions}
        tabu = ExecutableTabuSession(
            SnapshotTabuStore(state.tabu),
            identity,
            owner.library,
            retry_limit=owner.refine_topology_retry_limit,
            deadline_exceeded=owner._deadline_exceeded,
            live_genomes=lambda: owner.state.species_champions,
        )
        runtime.topology_tabu = tabu if owner.refine_deduplicate_topologies else None
        dormant: set[str] = set()

        def eligible(name: str) -> bool:
            before = time.perf_counter()
            ready = sessions[name].ready(runtime)
            elapsed = time.perf_counter() - before
            work[name]["seconds"] += elapsed
            stages = getattr(owner, "_active_stages", None)
            if stages is not None:
                stages[name] = round(stages.get(name, 0.0) + elapsed, 3)
            return ready

        while remaining > 0 and not runtime.should_stop() and not runtime.should_shutdown():
            ready = []
            for name in sessions:
                if runtime.should_stop() or runtime.should_shutdown():
                    break
                if name not in dormant and owner.evolve_shares[name] > 0 and eligible(name):
                    ready.append(name)
            if not ready or runtime.should_stop() or runtime.should_shutdown():
                break
            total_share = sum(owner.evolve_shares[name] for name in ready)
            for name in ready:
                state.deficits[name] = state.deficits.get(name, 0.0) + owner.evolve_shares[name] / total_share
            selected = max(ready, key=lambda name: state.deficits[name])
            state.deficits[selected] -= 1.0
            session = sessions[selected]
            session.refining = phase == "refine"
            owner.display.stage_started(selected, phase=phase, shared_generation=used + 1)
            before = time.perf_counter()
            evaluations, steps = session.evaluations, session.optimizer_steps
            outcome = session.advance(runtime)
            elapsed = time.perf_counter() - before
            work[selected]["seconds"] += elapsed
            stages = getattr(owner, "_active_stages", None)
            if stages is not None:
                stages[selected] = round(stages.get(selected, 0.0) + elapsed, 3)
            if outcome.preparation_steps:
                remaining -= outcome.preparation_steps
                work[selected]["preparation_steps"] += outcome.preparation_steps
                work[selected]["preparation_seconds"] += elapsed
                owner.display.stage_result(selected, "continue", "preparing learned structures", seconds=elapsed, depth=depth)
                continue
            outcome = owner._cross_validate_result(outcome, task)
            owner._consider_parent_report_result(outcome, depth=depth)
            if outcome.has_report_candidate and (best_report is None or owner._report_candidate_value(outcome) > owner._report_candidate_value(best_report)):
                best_report = outcome
            cost = outcome.generations_used
            if cost <= 0:
                dormant.add(selected)
                continue
            if cost != 1:
                raise ValueError(f"strategy session {selected} exceeded its one-generation quantum")
            used += cost
            remaining -= cost
            refined += cost if phase == "refine" else 0
            work[selected]["generations"] += cost
            work[selected]["evaluations"] += session.evaluations - evaluations
            work[selected]["optimizer_steps"] += session.optimizer_steps - steps
            previous = self.from_record(state.frontier[selected]) if selected in state.frontier else None
            accepted = owner._accepts_result(outcome)
            improves_incumbent = accepted and self.improves(outcome, incumbent)
            # Only a bounded frontier is published. Exchanges never masquerade as independently
            # rediscovered grammar evidence: the entire task session keeps one lineage root.
            if candidate_payload(outcome) is not None and (improves_incumbent or self.improves(outcome, previous)):
                key = self._publish(outcome, task, spec, identity)
                old_key = state.frontier.get(selected, {}).get("key")
                state.frontier[selected] = result_record(outcome, key)
                for receiver in sessions.values():
                    receiver.offer(key)
                dormant.clear()
                if old_key and old_key != key and (incumbent is None or old_key != incumbent.candidate_id):
                    owner.library.retire(old_key, reason="superseded search frontier")
            if improves_incumbent:
                previous_key = incumbent.candidate_id if incumbent is not None else None
                incumbent = outcome
                if candidate_payload(outcome) is not None:
                    assert outcome.candidate_id is not None
                    state.incumbent = result_record(outcome, outcome.candidate_id)
                    if previous_key and previous_key != outcome.candidate_id and owner.refine_retire_superseded:
                        owner.library.retire(previous_key, reason="strictly improved task incumbent")
                else:
                    outcome.candidate_id = getattr(outcome.champion_routed, "identity", None)
                    routed_rank = self.rank(outcome)
                    if routed_rank is not None:
                        outcome.size_metrics["champion_expanded_complexity"] = float(routed_rank.complexity)
                if phase == "solve":
                    phase = "refine"
                    remaining = refine_limit
            support, _query, _status, _query_status = owner._quality_of_result(outcome)
            owner.display.stage_result(
                selected, "accepted" if improves_incumbent else "continue", f"{phase} · shared generation {used}", seconds=elapsed, depth=depth, support_accuracy=support
            )
        tabu.commit()
        for session in sessions.values():
            if isinstance(session, RoutedSession):
                session.finish()
        for session in sessions.values():
            saver = getattr(session.strategy, "save_preparation", None)
            if saver is not None:
                saver(runtime)
        state.sessions = {name: session.state_dict() for name, session in sessions.items()}
        state.encounters += 1
        self._save(identity, state)
        result = incumbent or best_report or StrategyResult("interleaved", 0.0, 0)
        result.generations_used = used
        result.refinement_generations = refined
        result.phase = phase
        result.strategy_work = work
        result.strategy_status = {
            name: {
                "status": "ran" if work[name]["generations"] else "preparing" if work[name]["preparation_steps"] else "skipped" if session.skip_reason else "not_reached",
                "reason": session.skip_reason
                or (
                    "generation completed"
                    if work[name]["generations"]
                    else "grammar preparation in progress"
                    if work[name]["preparation_steps"]
                    else "stop requested"
                    if runtime.should_shutdown()
                    else "task deadline reached"
                    if runtime.should_stop()
                    else "shared budget exhausted before allocation"
                ),
            }
            for name, session in sessions.items()
        }
        result.strategy_metrics.update(tabu.metrics())
        for name, session in sessions.items():
            if session.skip_reason is not None:
                result.strategy_metrics[f"{name}_ineligible"] = 1.0
        return result
