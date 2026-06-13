"""The orchestrator: the policy layer that decides HOW each task gets solved.

Escalation ladder per task:

1. LOOKUP: query the library by structural I/O signature, quick-evaluate the top candidates, and
   accept without spending a single generation when an admitted solution already clears the bar.
   This is what makes a solved task STAY solved (the anti-forgetting property).
2. EVOLVE: run the hierarchical loop (compositions over library entries + the live module pool)
   under a depth-scaled generation budget with stall detection.
3. DECOMPOSE: on a stall, split the task into subtasks (registered DECOMPOSE operators yielding
   fully valid Tasks) and RECURSE on each. Accepted sub-solutions become frozen library entries;
   the parent then re-evolves seeded with a PortSpec-wired composition over them.
4. Give up gracefully: record the attempt and move on. Every decision lands in the attempts ledger.

Admission detaches solutions from run-local state: live module refs are snapshotted as frozen
MODULE entries (with the exact trained weights that scored) and the composition is rewritten to
reference them, so library entries never dangle across runs.
"""

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

from ardevo.dataset.icarus import Level0Encoder, Task, encode_task
from ardevo.decompose import Subtask, build_decomposers
from ardevo.evolution.composition import BIAS_REF, CompEdgeGene, CompNodeGene, CompNodeKind, CompositionGenome, comp_from_dict, comp_to_dict
from ardevo.evolution.genome import genome_from_dict, genome_to_dict
from ardevo.evolution.loop import AssessedComposition, CompTaskSpec, HierarchicalLoop, HierarchicalState
from ardevo.evolution.train import _writeback
from ardevo.library import COMPOSITION, LIBRARY_ADMISSION, MODULE, LibraryEntry, ModuleLibrary, module_level, task_io
from ardevo.strategy import StrategyResult, StrategyRuntime, build_strategies
from ardevo.substrate import decode_module, decode_recurrent
from ardevo.temporal import temporal_adapter
from ardevo.utils.logging import Logger

logger = Logger.get_logger()

_SNAPSHOT_SIGNATURE = "ANY"  # module entries are glue-fed, so they match structurally by width only


def comp_task_spec(task: Task) -> CompTaskSpec:
    """Everything the hierarchical loop needs to evolve compositions against `task` (flat encoding;
    stepped/temporal composition assembly is a documented v1 limitation)."""
    io = task_io(task)
    width = io["inputs"][0]["width"]
    signature = io["inputs"][0]["signature"]
    encoder = Level0Encoder(max_flat_dim=width)
    return CompTaskSpec(
        encoded=encode_task(task, encoder),
        encoder=encoder,
        n_inputs=width,
        input_specs=[(signature, width)],
        bank_columns={signature: list(range(width))},
        output_ref=task.meta.name,
        output_width=io["output"]["width"],
    )


@dataclass
class Attempt:
    """One row of the policy ledger: what the orchestrator tried and how it ended."""

    task: str
    depth: int
    outcome: str  # "library_hit" | "evolved" | "decomposed" | "failed"
    metric: float
    generations: int
    library_key: str | None = None
    decompose_op: str | None = None
    strategy: str | None = None  # which evolve strategy produced the winner (or the best loser)
    failure_stage: str | None = None  # for failed decomposed tasks: "subtask:<name>" | "parent_re_evolve"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "depth": self.depth,
            "outcome": self.outcome,
            "metric": self.metric,
            "generations": self.generations,
            "library_key": self.library_key,
            "decompose_op": self.decompose_op,
            "strategy": self.strategy,
            "failure_stage": self.failure_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attempt":
        return cls(
            task=data["task"],
            depth=int(data["depth"]),
            outcome=data["outcome"],
            metric=float(data["metric"]),
            generations=int(data["generations"]),
            library_key=data.get("library_key"),
            decompose_op=data.get("decompose_op"),
            strategy=data.get("strategy"),
            failure_stage=data.get("failure_stage"),
        )


@dataclass(frozen=True)
class Solution:
    key: str | None  # None when the task was SOLVED but the admission gate declined to shelve it
    entry_type: str
    metric: float


@dataclass
class StallDetector:
    """Stops an evolve phase that has flatlined (no best-fitness gain) or is hopeless (below the
    floor metric at half budget). One instance per evolve phase; it is stateful."""

    stall_generations: int
    stall_epsilon: float
    floor: float
    budget: int
    metric_of: Callable[[Any], float]
    best_fitness: float = -math.inf
    since_improvement: int = 0
    stalled: bool = False

    def __call__(self, generation: int, best: Any) -> bool:
        if best.fitness > self.best_fitness + self.stall_epsilon:
            self.best_fitness = best.fitness
            self.since_improvement = 0
        else:
            self.since_improvement += 1
        if self.since_improvement >= self.stall_generations:
            self.stalled = True
        if generation >= self.budget // 2 and self.metric_of(best) < self.floor:
            self.stalled = True
        return self.stalled


class Orchestrator:
    def __init__(self, config: dict[str, Any], loop: HierarchicalLoop, library: ModuleLibrary, state: HierarchicalState, proctor: Any = None) -> None:
        table = config.get("orchestrator", {})
        self.accept_metric = str(table.get("accept_metric", "query_accuracy"))
        self.accept_threshold = float(table.get("accept_threshold", 0.95))
        self.floor = float(table.get("floor", 0.55))
        self.stall_generations = int(table.get("stall_generations", 15))
        self.stall_epsilon = float(table.get("stall_epsilon", 0.005))
        self.max_depth = int(table.get("max_depth", 2))
        self.quick_eval_top_k = int(table.get("quick_eval_top_k", 5))
        # Decompose solvability gate: before committing the depth budget to a decomposition, probe
        # each subtask with a short evolve and require it to fit its support past this floor. A floor
        # of 0.0 disables the gate (legacy behavior). It is what stops I/O-axis slicing from burning
        # the whole budget on unsolvable parts (the two_spirals-class 0-for-N decompose failures).
        self.decompose_solvability_floor = float(table.get("decompose_solvability_floor", 0.0))
        self.decompose_probe_generations = int(table.get("decompose_probe_generations", 6))
        budgets = table.get("budgets", {})
        self.budgets = {int(name.removeprefix("depth")): int(value) for name, value in budgets.items()} or {0: 120, 1: 60, 2: 30}
        self.decomposers = build_decomposers(table)
        library_cfg = config.get("library", {}) or {}
        self.admission = LIBRARY_ADMISSION.get(library_cfg.get("admission", "accept_all"))(**{k: v for k, v in library_cfg.items() if k != "admission"})
        self.strategies = build_strategies(config)
        shares = table.get("evolve_budget", {})
        self.evolve_shares = {name: float(shares.get(name, 1.0)) for name, _strategy in self.strategies}
        self.loop = loop
        self.library = library
        self.state = state
        self.proctor = proctor
        self.attempts: list[Attempt] = []
        self.counters = {
            "library_hits": 0,
            "library_misses": 0,
            "accepts": 0,
            "failures": 0,
            "decompositions": 0,
            "admission_rejected": 0,
            "decompose_subtask_failed": 0,
            "decompose_parent_failed": 0,
        }
        self._failure_stage: str | None = None
        self._failure_op: str | None = None

    # --- public API ---------------------------------------------------------------------------------

    def solve(self, task: Task, depth: int = 0) -> Solution | None:
        spec = comp_task_spec(task)
        name = task.meta.name
        if depth == 0:
            self._failure_stage = None  # forensics for THIS top-level task only
            self._failure_op = None

        hit = self._lookup(task, spec)
        if hit is not None:
            self.counters["library_hits"] += 1
            self._record(Attempt(task=name, depth=depth, outcome="library_hit", metric=hit.metric, generations=0, library_key=hit.key))
            return hit
        self.counters["library_misses"] += 1
        self.loop.absorb_new_entries(self.state)  # fresh library knowledge enters the module pool

        budget = self._budget(depth)
        result = self._evolve(task, spec, budget)
        if result.metric >= self.accept_threshold:
            key = self._admit_result(result, task, depth, decompose_op=None)
            self.counters["accepts"] += 1
            self._record(Attempt(task=name, depth=depth, outcome="evolved", metric=result.metric, generations=result.generations_used, library_key=key, strategy=result.strategy))
            return Solution(key=key, entry_type=COMPOSITION if result.champion_comp is not None else MODULE, metric=result.metric)

        if depth < self.max_depth:
            solution = self._decompose_and_recurse(task, spec, depth, budget, first_metric=result.metric)
            if solution is not None:
                return solution

        self.counters["failures"] += 1
        self._record(
            Attempt(
                task=name,
                depth=depth,
                outcome="failed",
                metric=result.metric,
                generations=result.generations_used,
                strategy=result.strategy,
                decompose_op=self._failure_op if depth == 0 else None,
                failure_stage=self._failure_stage if depth == 0 else None,
            )
        )
        logger.info("orchestrator gave up on %s at depth %d (best %s=%.3f via %s)", name, depth, self.accept_metric, result.metric, result.strategy)
        return None

    # --- the evolve step: a config-ordered strategy ladder with budget carry --------------------------

    def _runtime(self) -> StrategyRuntime:
        return StrategyRuntime(
            loop=self.loop,
            library=self.library,
            state=self.state,
            accept_threshold=self.accept_threshold,
            metric_of=self._metric,
            stall_factory=self._stall_detector,
            on_generation=self._on_generation,
        )

    def _evolve(self, task: Task, spec: CompTaskSpec, budget: int, seed_comps: list[CompositionGenome] | None = None) -> StrategyResult:
        """Run the configured strategies in order under one shared budget. First strategy to clear
        the accept threshold wins (later ones never run); a stalled strategy's UNSPENT generations
        roll into the next allocation; the best loser is returned when nobody clears the bar."""
        runtime = self._runtime()
        total_share = sum(self.evolve_shares.values()) or 1.0
        results: list[StrategyResult] = []
        remaining = budget
        for position, (name, strategy) in enumerate(self.strategies):
            if position == len(self.strategies) - 1:
                allocation = remaining
            else:
                allocation = min(remaining, max(1, round(budget * self.evolve_shares[name] / total_share)))
            if allocation <= 0:
                break
            outcome = strategy(task, spec, runtime, budget=allocation, seed_comps=seed_comps if name == "composition" else None)
            results.append(outcome)
            remaining -= outcome.generations_used
            if outcome.metric >= self.accept_threshold:
                return outcome
            if remaining <= 0:
                break
        return max(results, key=lambda item: item.metric)

    # --- ladder steps -------------------------------------------------------------------------------

    def _lookup(self, task: Task, spec: CompTaskSpec) -> Solution | None:
        io = task_io(task)
        candidates = self.library.query(
            input_signature=io["inputs"][0]["signature"],
            input_width=io["inputs"][0]["width"],
            output_width=io["output"]["width"],
            limit=self.quick_eval_top_k,
        )
        for entry in candidates:
            metric = self._quick_metric(entry, task, spec)
            if metric is not None and metric >= self.accept_threshold:
                return Solution(key=entry.key, entry_type=entry.entry_type, metric=metric)
        return None

    @staticmethod
    def _entry_is_temporal(entry: LibraryEntry) -> bool:
        signature = entry.io["inputs"][0].get("signature", "")
        return "|" in signature and "T" in signature.split("|", 1)[1].split(",")

    def _quick_metric(self, entry: LibraryEntry, task: Task, spec: CompTaskSpec) -> float | None:
        """Evaluate a stored entry against the task with NO training (forward passes only).

        TIME-bearing MODULE entries (direct-strategy temporal winners) are scored through the
        stepped substrate; everything else uses the flat decode."""
        from ardevo.evaluation import evaluate
        from ardevo.evolution.composition import AssemblyContext, CompositionAssemblyError, assemble

        try:
            if entry.entry_type == MODULE and self._entry_is_temporal(entry):
                adapter = temporal_adapter(task)
                module = decode_recurrent(genome_from_dict(entry.payload), adapter.n_inputs, adapter.n_outputs, adapter.mode)
                metrics = adapter.evaluate(module)
                return self._metric(AssessedComposition(comp=CompositionGenome(), metrics=metrics, fitness=0.0, net=None))
            if entry.entry_type == MODULE:
                module = decode_module(genome_from_dict(entry.payload), spec.n_inputs, spec.output_width)
            else:
                comp = comp_from_dict(entry.payload)
                ctx = AssemblyContext(bank_columns=dict(spec.bank_columns), library=self.library, max_inline_depth=self.loop.max_inline_depth)
                module = assemble(comp, ctx, spec.n_inputs)
        except (ValueError, CompositionAssemblyError) as error:
            logger.debug("library candidate %s not evaluable here: %s", entry.key, error)
            return None
        metrics = evaluate(module, spec.encoded, spec.encoder)
        return self._metric(AssessedComposition(comp=CompositionGenome(), metrics=metrics, fitness=0.0, net=None))

    def _decompose_and_recurse(self, task: Task, spec: CompTaskSpec, depth: int, budget: int, first_metric: float) -> Solution | None:
        chosen_name: str | None = None
        subtasks: list[Subtask] = []
        for op_name, op in self.decomposers:
            produced = op(task, rng=self.state.rng)
            if len(produced) < 2:
                continue
            if not self._subtasks_promising(produced):
                logger.info("decompose op %s produced subtasks that fail the solvability probe; skipping", op_name)
                continue
            chosen_name, subtasks = op_name, produced
            break
        if not subtasks:
            return None
        self.counters["decompositions"] += 1
        logger.info("decomposing %s via %s into %d subtasks (depth %d)", task.meta.name, chosen_name, len(subtasks), depth)

        solutions: list[tuple[Subtask, Solution]] = []
        for subtask in subtasks:
            solved = self.solve(subtask.task, depth + 1)
            if solved is None:
                # A missing part means the wired parent cannot be completed; record WHERE it died.
                self.counters["decompose_subtask_failed"] += 1
                self._failure_stage = f"subtask:{subtask.task.meta.name}"
                self._failure_op = chosen_name
                return None
            solutions.append((subtask, solved))

        seed = self._port_wired_skeleton(spec, solutions)
        seeds = [seed] if seed is not None else None
        retry_budget = max(budget // 2, 5)
        result = self._evolve(task, spec, retry_budget, seed_comps=seeds)
        if result.metric >= self.accept_threshold:
            key = self._admit_result(result, task, depth, decompose_op=chosen_name)
            self.counters["accepts"] += 1
            attempt = Attempt(
                task=task.meta.name,
                depth=depth,
                outcome="decomposed",
                metric=result.metric,
                generations=result.generations_used,
                library_key=key,
                decompose_op=chosen_name,
                strategy=result.strategy,
            )
            self._record(attempt)
            return Solution(key=key, entry_type=COMPOSITION if result.champion_comp is not None else MODULE, metric=result.metric)
        # Every part solved but the wired parent still missed the bar.
        self.counters["decompose_parent_failed"] += 1
        self._failure_stage = "parent_re_evolve"
        self._failure_op = chosen_name
        return None

    def _subtasks_promising(self, subtasks: list[Subtask]) -> bool:
        """Probe each subtask with a short evolve and require it to fit its support past the
        solvability floor. Floor 0.0 disables the gate. This is what catches decompositions whose
        parts carry no learnable signal (e.g. classifying an entangled task from one input slice)
        BEFORE the depth budget is spent recursing on them."""
        if self.decompose_solvability_floor <= 0.0:
            return True
        for subtask in subtasks:
            spec = comp_task_spec(subtask.task)
            probe = self._evolve(subtask.task, spec, self.decompose_probe_generations)
            if probe.champion_metrics.get("support_accuracy", 0.0) < self.decompose_solvability_floor:
                return False
        return True

    # --- admission ----------------------------------------------------------------------------------

    def _gated_add(self, *, entry_type: str, payload: dict[str, Any], io: dict[str, Any], provenance: dict[str, Any], level: int, dependency: bool) -> str | None:
        """All library writes flow through here. Dependencies (module snapshots a composition
        needs, and sub-solutions inside a decompose recursion) BYPASS the policy: a parent must
        never dangle. Top-level winners face the configured admission policy; rejection still
        counts the task as solved, it just is not shelved."""
        if dependency:
            return self.library.add(entry_type=entry_type, payload=payload, io=io, provenance={**provenance, "dependency": True}, level=level)
        decision = self.admission(self.library, entry_type=entry_type, io=io, provenance=provenance)
        if not decision.admit:
            self.counters["admission_rejected"] += 1
            logger.info("admission rejected (%s): %s", entry_type, decision.reason)
            return None
        for retired_key in decision.retire:
            self.library.retire(retired_key)
            logger.info("retired library entry %s (%s)", retired_key, decision.reason)
        return self.library.add(entry_type=entry_type, payload=payload, io=io, provenance=provenance, level=level)

    def _admit_result(self, result: StrategyResult, task: Task, depth: int, decompose_op: str | None) -> str | None:
        """Route a strategy winner to the right admission shape."""
        if result.champion_comp is not None:
            return self._admit(result.champion_comp, task, depth, decompose_op)
        if result.champion_genome is not None:
            return self._admit_direct_module(result, task, depth)
        raise ValueError(f"strategy {result.strategy!r} produced no admissible champion")

    def _admit_direct_module(self, result: StrategyResult, task: Task, depth: int) -> str | None:
        """A direct-strategy winner is a TASK-SHAPED mini-model: admitted with its REAL io
        signature/widths (not the ANY-port snapshot shape), so lookups hit it exactly and the
        composition strategy can reference it through the catalog immediately."""
        assert result.champion_genome is not None
        provenance = {
            "task": task.meta.name,
            "rung": task.meta.rung,
            "depth": depth,
            "strategy": result.strategy,
            "accepted_metric": result.metric,
            "weight_robustness": result.champion_metrics.get("weight_robustness", 0.0),
        }
        level = module_level(result.champion_genome, self.library)
        return self._gated_add(entry_type=MODULE, payload=genome_to_dict(result.champion_genome), io=task_io(task), provenance=provenance, level=level, dependency=depth > 0)

    def _admit(self, best: AssessedComposition, task: Task, depth: int, decompose_op: str | None) -> str | None:
        """Detach the champion from run-local state and persist it: live refs become frozen MODULE
        entries carrying the exact trained weights that scored; the composition references those."""
        metric = self._metric(best)
        ref_map: dict[str, str] = {}
        levels: list[int] = []
        inner_by_ref = best.net.inner_modules if best.net is not None else {}
        for ref in sorted(set(best.comp.refs())):
            if ref.startswith("library:"):
                levels.append(self.library.load(ref.removeprefix("library:")).level)
                continue
            if not ref.startswith("live:"):
                continue
            species_id = int(ref.removeprefix("live:"))
            champion = self.state.species_champions.get(species_id)
            if champion is None:
                continue
            inner = inner_by_ref.get(ref)
            # The scored net carries the exact trained weights; without one, snapshot the champion as-is.
            tuned = _writeback(champion, inner) if inner is not None else champion
            module_io = {
                "inputs": [{"signature": _SNAPSHOT_SIGNATURE, "width": self.loop.in_ports}],
                "output": {"signature": _SNAPSHOT_SIGNATURE, "width": self.loop.out_ports},
            }
            provenance = {
                "task": task.meta.name,
                "rung": task.meta.rung,
                "depth": depth,
                "accepted_metric": metric,
                "weight_robustness": best.metrics.get("weight_robustness", 0.0),
            }
            snapshot_level = module_level(tuned, self.library)
            key = self.library.add(entry_type=MODULE, payload=genome_to_dict(tuned), io=module_io, provenance={**provenance, "dependency": True}, level=snapshot_level)
            ref_map[ref] = f"library:{key}"
            levels.append(snapshot_level)

        detached = best.comp.clone()
        for node_id in detached.module_ids:
            node = detached.nodes[node_id]
            if node.ref in ref_map:
                detached.nodes[node_id] = replace(node, ref=ref_map[node.ref])
        level = 1 + max(levels, default=1)
        provenance = {
            "task": task.meta.name,
            "rung": task.meta.rung,
            "depth": depth,
            "decompose_op": decompose_op,
            "accepted_metric": metric,
            "weight_robustness": best.metrics.get("weight_robustness", 0.0),
        }
        return self._gated_add(entry_type=COMPOSITION, payload=comp_to_dict(detached), io=task_io(task), provenance=provenance, level=level, dependency=depth > 0)

    # --- skeleton wiring ------------------------------------------------------------------------------

    def _port_wired_skeleton(self, spec: CompTaskSpec, solutions: list[tuple[Subtask, Solution]]) -> CompositionGenome | None:
        """Seed composition wiring each sub-solution into the parent per its PortSpec. Supported
        roles: output_slice (module output lands in its head slice) and input_subset (module reads
        its input slice, outputs sum into the whole head). Other roles fall back to None: the
        sub-solutions are still in the library and reachable through add_module_node."""
        if any(solution.key is None for _subtask, solution in solutions):
            logger.info("a sub-solution was solved but not shelved; relying on the ref catalog instead of a wired skeleton")
            return None
        roles = {subtask.port.role for subtask, _ in solutions}
        if not roles <= {"output_slice", "input_subset"}:
            logger.info("no skeleton wiring for roles %s; relying on the ref catalog instead", roles)
            return None

        tracker = self.state.comp_innovations
        comp = CompositionGenome()
        signature, width = spec.input_specs[0]
        input_id = tracker.new_node_id()
        comp.nodes[input_id] = CompNodeGene(input_id, CompNodeKind.INPUT, signature, 0, width)
        bias_id = tracker.new_node_id()
        comp.nodes[bias_id] = CompNodeGene(bias_id, CompNodeKind.INPUT, BIAS_REF, 0, 1)
        output_id = tracker.new_node_id()
        comp.nodes[output_id] = CompNodeGene(output_id, CompNodeKind.OUTPUT, spec.output_ref, spec.output_width, 0)
        comp.edges.append(CompEdgeGene(bias_id, output_id, True, tracker.innovation(bias_id, output_id), tuple(0.0 for _ in range(spec.output_width))))

        positions_total = sum(subtask.port.width for subtask, _ in solutions if subtask.port.role == "output_slice")
        per_position = spec.output_width // positions_total if positions_total else 1

        for subtask, solution in solutions:
            if solution.key is None:  # unreachable after the guard above; narrows for the type checker
                return None
            entry = self.library.load(solution.key)
            in_width = sum(item["width"] for item in entry.io["inputs"])
            out_width = entry.io["output"]["width"]
            node_id = tracker.new_node_id()
            comp.nodes[node_id] = CompNodeGene(node_id, CompNodeKind.MODULE, f"library:{solution.key}", in_width, out_width)
            port = subtask.port
            if port.role == "output_slice":
                in_glue = _identity_glue(width, in_width)
                start = port.offsets[0] * per_position
                out_glue = _placement_glue(out_width, spec.output_width, start)
            else:  # input_subset
                in_glue = _selection_glue(width, port.offsets)
                out_glue = _identity_glue(out_width, spec.output_width)
            comp.edges.append(CompEdgeGene(input_id, node_id, True, tracker.innovation(input_id, node_id), in_glue))
            comp.edges.append(CompEdgeGene(node_id, output_id, True, tracker.innovation(node_id, output_id), out_glue))
        return comp

    # --- plumbing -------------------------------------------------------------------------------------

    def _budget(self, depth: int) -> int:
        if depth in self.budgets:
            return self.budgets[depth]
        return max(self.budgets[max(self.budgets)] // (2 ** (depth - max(self.budgets))), 5)

    def _stall_detector(self, budget: int) -> StallDetector:
        return StallDetector(stall_generations=self.stall_generations, stall_epsilon=self.stall_epsilon, floor=self.floor, budget=budget, metric_of=self._metric)

    def _metric(self, item: Any) -> float:
        metrics = item.metrics
        if self.accept_metric == "query_accuracy" and not math.isfinite(metrics.get("query_loss", math.inf)):
            return float(metrics.get("support_accuracy", 0.0))  # degenerate query-less task
        return float(metrics.get(self.accept_metric, 0.0))

    def _record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)
        logger.info("attempt: %s", attempt.to_dict())

    def _on_generation(self, strategy: str, generation: int, best: Any, mean_fitness: float) -> None:
        if self.proctor is not None:
            self.proctor.log_scalar("Fitness", f"{strategy}_best", best.fitness, self.state.generation)
            self.proctor.log_scalar("Fitness", f"{strategy}_mean", mean_fitness, self.state.generation)
            self.proctor.log_scalar("Robustness", "weight_robustness", best.metrics.get("weight_robustness", 0.0), self.state.generation)


def _identity_glue(in_width: int, out_width: int) -> tuple[float, ...]:
    """Row-major identity-ish map: 1.0 on the diagonal, zero elsewhere (rectangular allowed)."""
    return tuple(1.0 if row == column else 0.0 for row in range(in_width) for column in range(out_width))


def _placement_glue(in_width: int, out_width: int, start: int) -> tuple[float, ...]:
    """Maps a module's output block onto its slice of the parent head: out[start + r] = in[r]."""
    return tuple(1.0 if row == start + column else 0.0 for row in range(in_width) for column in range(out_width))


def _selection_glue(in_width: int, offsets: tuple[int, int]) -> tuple[float, ...]:
    """Selects parent input columns [start, end) into a module's input ports."""
    start, end = offsets
    out_width = end - start
    return tuple(1.0 if start + column == row else 0.0 for row in range(in_width) for column in range(out_width))


def attempts_to_dicts(attempts: list[Attempt]) -> list[dict[str, Any]]:
    return [attempt.to_dict() for attempt in attempts]


def attempts_from_dicts(data: list[dict[str, Any]]) -> list[Attempt]:
    return [Attempt.from_dict(item) for item in data]
