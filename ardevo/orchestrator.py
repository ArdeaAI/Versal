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
from ardevo.evolution.genome import Genome, genome_from_dict, genome_to_dict
from ardevo.evolution.loop import AssessedComposition, CompTaskSpec, HierarchicalLoop, HierarchicalState
from ardevo.evolution.train import _writeback
from ardevo.library import COMPOSITION, LIBRARY_ADMISSION, MODULE, LibraryEntry, ModuleLibrary, module_level, structural_fingerprint, task_io
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
    outcome: str  # "library_hit" | "refined" | "evolved" | "decomposed" | "failed"
    metric: float
    generations: int
    library_key: str | None = None
    decompose_op: str | None = None
    strategy: str | None = None  # which evolve strategy produced the winner (or the best loser)
    failure_stage: str | None = None  # for failed decomposed tasks: "subtask:<name>" | "parent_re_evolve"
    refine_generations: int = 0  # learn-mode generations spent refining a library hit (bounded extra compute)

    def to_dict(self) -> dict[str, Any]:
        data = {
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
        if self.refine_generations:  # only when refinement ran, so live-mode summaries stay byte-identical
            data["refine_generations"] = self.refine_generations
        return data

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
            refine_generations=int(data.get("refine_generations", 0)),
        )


@dataclass(frozen=True)
class Solution:
    key: str | None  # None when the task was SOLVED but the admission gate declined to shelve it
    entry_type: str
    metric: float


@dataclass(frozen=True)
class RefinementRank:
    """The lexicographic standing of one topology in a refine-on-hit comparison."""

    metric: float
    robustness: float
    complexity: int
    entry_type: str


def refinement_improves(candidate: RefinementRank, incumbent: RefinementRank, *, metric_epsilon: float, robustness_epsilon: float) -> bool:
    """Lexicographic better-than: accept metric, then weight robustness, then LOWER structural
    complexity, each tier inside an epsilon band so "equal" is a non-event (equal everything never
    admits). Complexity only tie-breaks within one entry type: composition complexity counts glue
    edges + module nodes, not inner genome cost, so cross-type size comparisons are meaningless."""
    if not (math.isfinite(candidate.metric) and math.isfinite(candidate.robustness)):
        return False
    if candidate.metric > incumbent.metric + metric_epsilon:
        return True
    if candidate.metric < incumbent.metric - metric_epsilon:
        return False
    if candidate.robustness > incumbent.robustness + robustness_epsilon:
        return True
    if candidate.robustness < incumbent.robustness - robustness_epsilon:
        return False
    if candidate.entry_type != incumbent.entry_type:
        return False
    return candidate.complexity < incumbent.complexity


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
        # Learn-mode refinement of library hits ([orchestrator.refine]). budget_k = 0 is live mode:
        # hits stay zero-cost and the whole path below is byte-identical to the plain hit branch.
        refine = table.get("refine", {}) or {}
        self.refine_budget_k = int(refine.get("budget_k", 0))
        self.refine_depth_max = int(refine.get("depth_max", 0))  # > 0 also refines sub-solve hits, whose improvements admit UNCONDITIONALLY (dependency rail)
        self.refine_metric_epsilon = float(refine.get("metric_epsilon", 0.005))
        self.refine_robustness_epsilon = float(refine.get("robustness_epsilon", 0.01))
        self.refine_decay = float(refine.get("decay", 0.5))
        self.refine_min_generations = int(refine.get("min_generations", 4))
        self.refine_stall_generations = int(refine.get("stall_generations", 8))
        self.refine_retire_superseded = bool(refine.get("retire_superseded", True))
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
        if self.refine_budget_k > 0:  # registered only in learn mode so live-mode summaries stay byte-identical
            self.counters.update(
                {
                    "refine_attempts": 0,
                    "refine_improvements": 0,
                    "refine_no_gain": 0,
                    "refine_skipped_decayed": 0,
                    "refine_skipped_no_strategy": 0,
                    "refine_generations": 0,
                }
            )
        if any(name == "routed" for name, _strategy in self.strategies):  # registered only when the routed strategy is configured
            # routed_solved counts DISTILLED admissions (the router's win became a library composition);
            # no_experts = short-circuited on an empty vertex set; undistillable = won in router space
            # but the pathway did not survive verification as a composition (reported as a miss).
            self.counters.update({"routed_solved": 0, "routed_zero_shot": 0, "routed_no_experts": 0, "routed_undistillable": 0})
        self._failure_stage: str | None = None
        self._failure_op: str | None = None
        self._refined_from: str | None = None  # lineage provenance for the admission inside a refine

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
            return self._handle_library_hit(hit, task, spec, depth)
        self.counters["library_misses"] += 1
        self.loop.absorb_new_entries(self.state)  # fresh library knowledge enters the module pool

        budget = self._budget(depth)
        result = self._evolve(task, spec, budget)
        if result.metric >= self.accept_threshold:
            key = self._admit_result(result, task, depth, decompose_op=None)
            self.counters["accepts"] += 1
            self._record(Attempt(task=name, depth=depth, outcome="evolved", metric=result.metric, generations=result.generations_used, library_key=key, strategy=result.strategy))
            return Solution(key=key, entry_type=self._entry_type_of(result), metric=result.metric)

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
            if name == "routed":  # the strategy has no counter access; it stamps markers instead
                for marker in ("routed_no_experts", "routed_undistillable"):
                    if outcome.champion_metrics.get(marker):
                        self.counters[marker] += 1
            results.append(outcome)
            remaining -= outcome.generations_used
            if outcome.metric >= self.accept_threshold:
                return outcome
            if remaining <= 0:
                break
        return max(results, key=lambda item: item.metric)

    # --- learn-mode refinement of library hits --------------------------------------------------------

    def _handle_library_hit(self, hit: Solution, task: Task, spec: CompTaskSpec, depth: int) -> Solution:
        """STEP 1b: a hit is free, but learn mode (refine budget_k > 0) spends a bounded, decaying
        budget trying to beat the stored solution before settling for it. The guard runs FIRST with
        zero side effects, so budget_k = 0 (live mode) is byte-identical to the plain hit path. The
        task can never regress: a failed refinement returns the original hit."""
        refine_generations = 0
        if self.refine_budget_k > 0 and depth <= self.refine_depth_max and hit.key is not None:
            improved, refine_generations = self._refine_hit(hit, task, spec, depth)
            if improved is not None:
                return improved  # _refine_hit already recorded the "refined" attempt
        self._record(Attempt(task=task.meta.name, depth=depth, outcome="library_hit", metric=hit.metric, generations=0, library_key=hit.key, refine_generations=refine_generations))
        return hit

    def _refine_hit(self, hit: Solution, task: Task, spec: CompTaskSpec, depth: int) -> tuple[Solution | None, int]:
        """Seed a bounded evolve from the stored solution and admit only a strict lexicographic
        improvement (metric, then robustness, then lower complexity). Runs ONLY the strategy
        matching the entry shape, so K stays focused and `_evolve`'s share arithmetic is untouched."""
        assert hit.key is not None
        entry = self.library.load(hit.key)
        strategy_name = "direct" if entry.entry_type == MODULE else "composition"
        strategy = dict(self.strategies).get(strategy_name)
        if strategy is None:
            self.counters["refine_skipped_no_strategy"] += 1
            return None, 0
        effective_budget = self._effective_refine_budget(entry)
        if effective_budget < self.refine_min_generations:
            self.counters["refine_skipped_decayed"] += 1
            return None, 0
        self.counters["refine_attempts"] += 1
        # The target is deliberately NOT clamped to 1.0: an incumbent at 1.0 makes it unreachable,
        # so the strategy runs to its (refine-local) stall and the tie-breaks decide afterward;
        # a beatable incumbent lets the strategy's early exit stop the moment it wins.
        runtime = self._refine_runtime(hit.metric + self.refine_metric_epsilon)
        if entry.entry_type == MODULE:
            result = strategy(task, spec, runtime, budget=effective_budget, seed_entries=[entry])
        else:
            result = strategy(task, spec, runtime, budget=effective_budget, seed_comps=[comp_from_dict(entry.payload)])
        self.counters["refine_generations"] += result.generations_used

        candidate = self._candidate_rank(result)
        incumbent = self._incumbent_rank(hit, entry, seed_metric=result.seed_metric)
        improves = (
            candidate is not None
            # A robustness/size tie-break can sit epsilon below an at-the-bar incumbent; never shelve below the bar.
            and result.metric >= self.accept_threshold
            # Identity check FIRST: the incumbent topology retrained on this variant is NOT a new
            # solution (entry keys hash weights, so a key comparison can never catch this).
            and self._candidate_fingerprint(result) != structural_fingerprint(entry.entry_type, entry.payload)
            and refinement_improves(candidate, incumbent, metric_epsilon=self.refine_metric_epsilon, robustness_epsilon=self.refine_robustness_epsilon)
        )
        if not improves:
            self.library.record_refinement(hit.key, improved=False)
            self.counters["refine_no_gain"] += 1
            return None, result.generations_used

        # Scalars, not the stats dict: `summary` copies shallowly, and record_refinement below
        # mutates the live row the copy still aliases.
        parent_stats = (self.library.summary(hit.key) or {}).get("stats") or {}
        parent_attempts = int(parent_stats.get("refine_attempts", 0))
        parent_failures = int(parent_stats.get("refine_failures_since_gain", 0))
        self._refined_from = hit.key
        try:
            key = self._admit_result(result, task, depth, decompose_op=None)
        finally:
            self._refined_from = None
        # Decay resets only when the gain was actually shelved; a policy-rejected gain still returns
        # the improved solution this run but counts as a failure so full-K retries do not repeat.
        self.library.record_refinement(hit.key, improved=key is not None)
        if key is not None:
            # The replacement continues the SAME lineage, so its cooldown rides the chain: a
            # capability gain (metric/robustness tier) recharges the family, but a compression-only
            # gain spends it (24 -> 12 -> 6 -> skip), or one capability epoch would fund an endless
            # per-variant polish treadmill of near-identical entries (the 2026-07-04 lesson).
            assert candidate is not None
            capability_gain = candidate.metric > incumbent.metric + self.refine_metric_epsilon or candidate.robustness > incumbent.robustness + self.refine_robustness_epsilon
            self.library.seed_refine_stats(key, attempts=parent_attempts + 1, failures=0 if capability_gain else parent_failures + 1)
        if key is not None and self.refine_retire_superseded and candidate is not None:
            self._retire_if_dominated(hit.key, candidate, incumbent)
        self.counters["refine_improvements"] += 1
        self._record(
            Attempt(
                task=task.meta.name,
                depth=depth,
                outcome="refined",
                metric=result.metric,
                generations=result.generations_used,
                library_key=key,
                strategy=result.strategy,
                refine_generations=result.generations_used,
            )
        )
        return Solution(key=key, entry_type=self._entry_type_of(result), metric=result.metric), result.generations_used

    def _effective_refine_budget(self, entry: LibraryEntry) -> int:
        summary = self.library.summary(entry.key) or {}
        failures = int((summary.get("stats") or {}).get("refine_failures_since_gain", 0))
        return int(self.refine_budget_k * (self.refine_decay**failures))

    def _refine_runtime(self, target_metric: float) -> StrategyRuntime:
        """Like `_runtime`, but the bar is beating the incumbent and the flatline window is the
        refine-local one (K is a cap, not a fixed cost: a saturated refinement stalls out early).
        The detector's half-budget floor check can never fire: the seed already scores above floor."""

        def refine_stall_factory(budget: int) -> StallDetector:
            return StallDetector(stall_generations=self.refine_stall_generations, stall_epsilon=self.stall_epsilon, floor=self.floor, budget=budget, metric_of=self._metric)

        return StrategyRuntime(
            loop=self.loop,
            library=self.library,
            state=self.state,
            accept_threshold=target_metric,
            metric_of=self._metric,
            stall_factory=refine_stall_factory,
            on_generation=self._on_generation,
        )

    def _candidate_rank(self, result: StrategyResult) -> RefinementRank | None:
        robustness = float(result.champion_metrics.get("weight_robustness", 0.0))
        if result.champion_genome is not None:
            return RefinementRank(metric=result.metric, robustness=robustness, complexity=result.champion_genome.complexity(), entry_type=MODULE)
        if result.champion_comp is not None:
            return RefinementRank(metric=result.metric, robustness=robustness, complexity=result.champion_comp.comp.complexity(), entry_type=COMPOSITION)
        return None

    @staticmethod
    def _candidate_fingerprint(result: StrategyResult) -> str | None:
        if result.champion_genome is not None:
            return structural_fingerprint(MODULE, genome_to_dict(result.champion_genome))
        if result.champion_comp is not None:
            return structural_fingerprint(COMPOSITION, comp_to_dict(result.champion_comp.comp))
        return None

    def _incumbent_rank(self, hit: Solution, entry: LibraryEntry, *, seed_metric: float | None = None) -> RefinementRank:
        # The metric baseline is the STRONGER of the quick metric just measured on THIS task and the
        # seed's own trained standing inside the refine run (when the strategy tracked it): a
        # candidate must beat the incumbent GIVEN THE SAME TRAINING, or the comparison rewards
        # retraining instead of topology. Robustness is stored-at-admission (the index max over
        # re-admissions); recomputing it fresh would cost a weight-samples evaluation per hit,
        # against the cheap-hit contract, and the asymmetry only matters inside the metric-tie tier
        # where the epsilon band absorbs it.
        summary = self.library.summary(entry.key) or {}
        robustness = float(summary.get("weight_robustness", entry.provenance.get("weight_robustness", 0.0)))
        if entry.entry_type == MODULE:
            complexity = genome_from_dict(entry.payload).complexity()
        else:
            complexity = comp_from_dict(entry.payload).complexity()
        metric = hit.metric if seed_metric is None else max(hit.metric, seed_metric)
        return RefinementRank(metric=metric, robustness=robustness, complexity=complexity, entry_type=entry.entry_type)

    def _retire_if_dominated(self, old_key: str, candidate: RefinementRank, incumbent: RefinementRank) -> None:
        """Retire the superseded entry only when the shelved improvement weakly dominates its STORED
        ranking fields AND carries a strict margin: better beyond epsilon on metric or robustness,
        or strictly simpler (same entry type) at parity. Weak dominance alone is not enough: entries
        with a degenerate stored robustness of 0.0 (the temporal-module case) would otherwise be
        tombstoned by any same-metric clone. Mixed trade-offs (metric up, robustness down) keep
        distinct value and stay live for the archive policy to manage."""
        summary = self.library.summary(old_key)
        if summary is None:
            return
        stored_metric = float(summary.get("accepted_metric", 0.0))
        stored_robustness = float(summary.get("weight_robustness", 0.0))
        weakly_dominates = candidate.metric >= stored_metric and candidate.robustness >= stored_robustness
        strict_margin = candidate.metric > stored_metric + self.refine_metric_epsilon or candidate.robustness > stored_robustness + self.refine_robustness_epsilon
        strictly_simpler = candidate.entry_type == incumbent.entry_type and candidate.complexity < incumbent.complexity
        if weakly_dominates and (strict_margin or strictly_simpler):
            self.library.retire(old_key)
            logger.info("retired superseded library entry %s (refined replacement dominates)", old_key)

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
            return Solution(key=key, entry_type=self._entry_type_of(result), metric=result.metric)
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

    @staticmethod
    def _entry_type_of(result: StrategyResult) -> str:
        """The entry-type label for a strategy winner's Solution (keeps the ledger truthful)."""
        if result.champion_routed is not None:
            return "routed"
        return COMPOSITION if result.champion_comp is not None else MODULE

    def _admit_result(self, result: StrategyResult, task: Task, depth: int, decompose_op: str | None) -> str | None:
        """Route a strategy winner to the right admission shape."""
        if result.champion_routed is not None:
            # Distillation off ([orchestrator.routed] distill = false): the win is solved-but-not-
            # shelved (Solution.key = None), its executable state being the persisted router. With
            # distillation on, a routed win arrives here as champion_comp instead.
            self.counters["routed_solved"] += 1
            if getattr(result.champion_routed, "zero_shot", False):
                self.counters["routed_zero_shot"] += 1
            return None
        if result.champion_comp is not None:
            if result.strategy == "routed":
                # A distilled routed win: the pathway survived verification as a composition and is
                # admitted through the ordinary rail below, becoming a routable vertex at next sync.
                self.counters["routed_solved"] += 1
                if result.champion_metrics.get("routed_zero_shot"):
                    self.counters["routed_zero_shot"] += 1
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
            "behavior": _genome_behavior(result.champion_genome),
        }
        if self._refined_from is not None:
            provenance["refined_from"] = self._refined_from  # lineage: this entry continues that one
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
            "behavior": _comp_behavior(detached, level),
        }
        if self._refined_from is not None:
            provenance["refined_from"] = self._refined_from  # lineage: this entry continues that one
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


def _genome_behavior(genome: Genome) -> list[str]:
    """A coarse structural fingerprint that niches a flat module in the QD archive: distinct KINDS of
    solution (small vs deep, feedforward vs recurrent vs refining, summed vs gated, with/without
    macros) land in different niches and so coexist as diverse stepping stones instead of the
    top-k-by-metric collapse. Cheap (no re-evaluation); functional fingerprinting is a later refinement."""
    return [
        f"h{min(len(genome.hidden_ids) // 4, 6)}",
        "rec" if genome.recurrent_connections() else "ff",
        "refine" if genome.refine_steps > 1 else "single",
        "prod" if any(node.aggregation == "product" for node in genome.nodes.values()) else "sum",
        "macro" if genome.macros else "flat",
    ]


def _comp_behavior(comp: CompositionGenome, level: int) -> list[str]:
    """Structural niche for a composition: how many modules it wires and at what level."""
    return [f"m{min(len(comp.module_ids), 6)}", f"L{level}"]


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
