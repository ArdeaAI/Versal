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
import time
from array import array
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from ardevo.dataset.icarus import Level0Encoder, Task, encode_task
from ardevo.decompose import Subtask, build_decomposers
from ardevo.evaluation import fit_query_target, without_query
from ardevo.evolution.composition import BIAS_REF, CompEdgeGene, CompNodeGene, CompNodeKind, CompositionGenome, GlueValues, comp_from_dict, comp_to_dict
from ardevo.evolution.genome import Genome, genome_from_dict, genome_to_dict
from ardevo.evolution.loop import AssessedComposition, CompTaskSpec, HierarchicalLoop, HierarchicalState
from ardevo.evolution.train import _writeback
from ardevo.library import COMPOSITION, LIBRARY_ADMISSION, MODULE, LibraryEntry, ModuleLibrary, macro_resolver, module_level, structural_fingerprint, task_io
from ardevo.strategy import StrategyResult, StrategyRuntime, build_strategies
from ardevo.substrate import decode_module, decode_recurrent
from ardevo.temporal import temporal_adapter
from ardevo.utils.logging import Logger
from ardevo.utils.resources import format_bytes
from ardevo.utils.status import BOARD

logger = Logger.get_logger()

_SNAPSHOT_SIGNATURE = "ANY"  # module entries are glue-fed, so they match structurally by width only


def comp_task_spec(task: Task, *, include_query: bool = True, structured_grid: bool = False) -> CompTaskSpec:
    """Everything the hierarchical loop needs to evolve compositions against `task` (flat encoding;
    stepped/temporal composition assembly is a documented v1 limitation)."""
    io = task_io(task)
    width = io["inputs"][0]["width"]
    signature = io["inputs"][0]["signature"]
    encoder = Level0Encoder(max_flat_dim=width)
    encoded: Any = None
    if structured_grid:
        from ardevo.structured import encode_structured_grid

        encoded = encode_structured_grid(task, encoder, include_query=include_query)
    if encoded is None:
        encoded = fit_query_target(encode_task(task, encoder))
    if not include_query:
        from ardevo.structured import StructuredGridEncoded

        encoded = encoded.without_query() if isinstance(encoded, StructuredGridEncoded) else without_query(encoded)
    return CompTaskSpec(
        encoded=encoded,
        encoder=encoder,
        n_inputs=width,
        input_specs=[(signature, width)],
        bank_columns={signature: range(width)},
        output_ref=task.meta.name,
        output_width=io["output"]["width"],
        io=io,
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
    # Wall-clock forensics: where this solve actually SPENT its time, so a wedged stage shows up
    # in run_summary.json instead of requiring a live `sample` of the process (the 2026-07-05
    # 8-hour CIFAR mutation wedge was invisible in every record).
    seconds: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    # Champion weight-sample diagnostics (the G0 structure-vs-weights readout): present only when
    # the evaluate op emitted them (hybrid / weight_samples), so standard-eval summaries and old
    # checkpoints stay byte-identical.
    sample_metrics: dict[str, float] = field(default_factory=dict)
    # Champion/population genome size (the bloat readout): task cost tracks genome size, so growth
    # must show in the record, not only through `seconds` (the diag_g2 free-growth blowup was
    # invisible until the wall-clock had already exploded). Empty on cheap library hits.
    size_metrics: dict[str, float] = field(default_factory=dict)
    resource_metrics: dict[str, float] = field(default_factory=dict)
    # Optional held-out report, emitted only by the blind-query protocol after candidate selection.
    report_metric: float | None = None
    task_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
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
        if self.seconds:
            data["seconds"] = self.seconds
        if self.stage_seconds:
            data["stage_seconds"] = self.stage_seconds
        if self.sample_metrics:
            data["sample_metrics"] = self.sample_metrics
        if self.size_metrics:
            data["size_metrics"] = self.size_metrics
        if self.resource_metrics:
            data["resource_metrics"] = self.resource_metrics
        if self.report_metric is not None:
            data["report_metric"] = self.report_metric
        if self.task_metrics:
            data["task_metrics"] = self.task_metrics
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
            seconds=float(data.get("seconds", 0.0)),
            stage_seconds=dict(data.get("stage_seconds", {})),
            sample_metrics=dict(data.get("sample_metrics", {})),
            size_metrics=dict(data.get("size_metrics", {})),
            resource_metrics=dict(data.get("resource_metrics", {})),
            report_metric=float(data["report_metric"]) if data.get("report_metric") is not None else None,
            task_metrics={str(key): float(value) for key, value in data.get("task_metrics", {}).items()},
        )


_SAMPLE_METRIC_KEYS = ("mean_sample_accuracy", "max_sample_accuracy", "best_sample_weight", "weight_robustness")


def _sample_metrics_of(result: StrategyResult) -> dict[str, float]:
    # mean_sample_accuracy marks a real weight-sample measurement; _floored_metrics carries a bare
    # weight_robustness = 0.0 that must not fabricate a diagnostic row under standard eval.
    if "mean_sample_accuracy" not in result.champion_metrics:
        return {}
    return {key: float(result.champion_metrics[key]) for key in _SAMPLE_METRIC_KEYS if key in result.champion_metrics}


_TASK_METRIC_SUFFIXES = ("_exact", "_task_exact", "_shape_accuracy", "_baseline_accuracy", "_gain_over_baseline", "_coverage", "_covered_accuracy")


def _task_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if key.startswith(("support_", "query_")) and key.endswith(_TASK_METRIC_SUFFIXES) and math.isfinite(float(value))}


def _task_metrics_of(result: StrategyResult) -> dict[str, float]:
    return _task_metrics(dict(result.champion_metrics) | dict(result.report_metrics))


@dataclass(frozen=True)
class Solution:
    key: str | None  # None when the task was SOLVED but the admission gate declined to shelve it
    entry_type: str
    metric: float
    report_metric: float | None = None
    task_metrics: dict[str, float] = field(default_factory=dict)


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
        self.search_metric = str(table.get("search_metric", self.accept_metric))
        self.report_metric = str(table.get("report_metric", self.accept_metric))
        self.blind_query = bool(table.get("blind_query", False))
        self.structured_grid = bool(table.get("direct", {}).get("structured_grid", False))
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
        # DECOMPOSE-FIRST ([orchestrator] decompose_first_above): a task whose dense-init gene
        # estimate ((n_in + 1) x n_out) exceeds this runs the decompose registry BEFORE the evolve
        # ladder, because decompose is otherwise the LAST stage and an init-wall task (rungs 11-14,
        # 15, 18) would wedge in direct evolution before ever reaching it. On decompose failure the
        # ordinary ladder still runs, so this knob must ship WITH a scale-safe init (factored or
        # sparse) in the same config. 0 (the default) is off and byte-identical.
        decompose_first = table.get("decompose_first_above", 0)
        if decompose_first != "adaptive" and int(decompose_first) < 0:
            raise ValueError("[orchestrator] decompose_first_above must be non-negative or 'adaptive'")
        self.decompose_first_above: int | str = decompose_first if decompose_first == "adaptive" else int(decompose_first)
        budgets = table.get("budgets", {})
        self.budgets = {int(name.removeprefix("depth")): int(value) for name, value in budgets.items()} or {0: 120, 1: 60, 2: 30}
        self.decomposers = build_decomposers(table)
        library_cfg = config.get("library", {}) or {}
        # "gc" is the trial's run-end sweep flag, not an admission-policy knob.
        self.admission = LIBRARY_ADMISSION.get(library_cfg.get("admission", "accept_all"))(**{k: v for k, v in library_cfg.items() if k not in ("admission", "gc")})
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
        # WALL LEDGER ([orchestrator.wall]): a depth-0 failure shelves its best champion as a
        # below-bar stepping stone, and later attempts on that signature warm-start from it, so
        # assaults on a hard task family (the two_spirals wall) accumulate instead of restarting.
        # ledger = false (the default) is byte-identical to before.
        wall = table.get("wall", {}) or {}
        self.wall_ledger = bool(wall.get("ledger", False))
        self.wall_min_metric = float(wall.get("min_metric", 0.45))
        self.wall_min_gain_over_baseline = float(wall.get("min_gain_over_baseline", -math.inf))
        self.wall_seed_top_k = int(wall.get("seed_top_k", 1))
        # PER-ATTEMPT WALL-CLOCK BUDGET ([orchestrator] max_task_seconds): with a free-growth config
        # nothing bounds per-generation cost, so an attempt can silently eat hours (task 2 of the
        # 2026-07-05 diag_g2 run: 3532s). Past the deadline the running stage finishes its current
        # generation, later ladder stages are skipped, and the attempt fails with its best champion
        # (the wall ledger still shelves it). 0 (the default) is off and byte-identical.
        self.max_task_seconds = float(table.get("max_task_seconds", 0.0))
        # Cumulative budget for the whole top-level solve, including recursive subtasks.  This is
        # distinct from max_task_seconds, which remains a per-depth attempt budget.
        self.max_total_task_seconds = float(table.get("max_total_task_seconds", 0.0))
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
            self.counters.update({"routed_solved": 0, "routed_zero_shot": 0, "routed_no_experts": 0, "routed_undistillable": 0, "routed_resource_declined": 0})
        if self.wall_ledger:
            self.counters.update({"wall_stones_admitted": 0, "wall_stones_improved": 0, "wall_seeded_attempts": 0})
        if self.max_task_seconds > 0 or self.max_total_task_seconds > 0:  # absent limits keep legacy summaries byte-identical
            self.counters.update({"time_budget_hits": 0})
        if self.max_total_task_seconds > 0:
            self.counters.update({"total_time_budget_hits": 0})
        if self.loop.resource_policy.mode == "adaptive":
            self.counters.update({"resource_declines": 0, "direct_resource_declines": 0, "composition_resource_declines": 0, "decomposition_resource_declines": 0})
        if self.decompose_first_above == "adaptive" or int(self.decompose_first_above) > 0:  # registered only when the policy is on
            self.counters.update({"decompose_first": 0})
        self._failure_stage: str | None = None
        self._failure_op: str | None = None
        self._refined_from: str | None = None  # lineage provenance for the admission inside a refine
        self._stepping_stone = False  # marks the admission inside a wall-ledger shelving

    # --- public API ---------------------------------------------------------------------------------

    def solve(self, task: Task, depth: int = 0) -> Solution | None:
        # Per-solve wall-clock forensics, save/restored so recursive sub-solves time themselves
        # without clobbering the parent's ledger. The deadline rides the same tuple: each depth
        # gets a fresh budget, but a sub-solve only starts while its parent still has time.
        previous_timing = (getattr(self, "_solve_started", None), getattr(self, "_active_stages", None), getattr(self, "_solve_deadline", None))
        previous_total_deadline = getattr(self, "_total_task_deadline", None)
        now = time.perf_counter()
        if depth == 0:
            self._total_task_deadline: float | None = (now + self.max_total_task_seconds) if self.max_total_task_seconds > 0 else None
        self._solve_started: float | None = now
        self._active_stages: dict[str, float] | None = {}
        local_deadline = (now + self.max_task_seconds) if self.max_task_seconds > 0 else None
        total_deadline = getattr(self, "_total_task_deadline", None)
        deadlines = [deadline for deadline in (local_deadline, total_deadline) if deadline is not None]
        self._solve_deadline = min(deadlines) if deadlines else None
        BOARD.clock(max(0.0, self._solve_deadline - now) if self._solve_deadline is not None else None)
        try:
            return self._solve_timed(task, depth)
        finally:
            self._solve_started, self._active_stages, self._solve_deadline = previous_timing
            if depth == 0:
                self._total_task_deadline = previous_total_deadline

    def _solve_timed(self, task: Task, depth: int = 0) -> Solution | None:
        if self._total_deadline_exceeded():
            return self._record_total_timeout(task, depth)
        spec = comp_task_spec(task, include_query=not self.blind_query, structured_grid=self.structured_grid)
        report_spec = comp_task_spec(task, structured_grid=self.structured_grid) if self.blind_query else spec
        name = task.meta.name
        if depth == 0:
            self._failure_stage = None  # forensics for THIS top-level task only
            self._failure_op = None

        hit = self._lookup(task, spec)
        if hit is not None:
            self.counters["library_hits"] += 1
            return self._handle_library_hit(hit, task, spec, depth)
        if self._total_deadline_exceeded():
            return self._record_total_timeout(task, depth)
        self.counters["library_misses"] += 1
        self.loop.absorb_new_entries(self.state)  # fresh library knowledge enters the module pool

        budget = self._budget(depth)
        decomposed_first = False
        if self._wants_decompose_first(task, spec) and depth < self.max_depth and not self._deadline_exceeded():
            decomposed_first = True
            self.counters["decompose_first"] += 1
            decompose_started = time.perf_counter()
            solution = self._decompose_and_recurse(task, spec, depth, budget, first_metric=0.0)
            if self._active_stages is not None:
                self._active_stages["decompose_first"] = round(time.perf_counter() - decompose_started, 3)
            if solution is not None:
                return solution
            # Fall through to the ordinary ladder: a scale-safe init (factored/sparse) rides the
            # same config, so the flat attempt below stays affordable even at init-wall widths.

        stone_modules, stone_comps = self._wall_stone_seeds(task, spec) if self.wall_ledger and depth == 0 else ([], [])
        if stone_modules or stone_comps:
            self.counters["wall_seeded_attempts"] += 1
        result = self._evolve(task, spec, budget, seed_comps=stone_comps or None, seed_entries=stone_modules or None)
        if self.blind_query:
            result = self._attach_report_metrics(result, report_spec)
        if self._accepts_result(result):
            key = self._admit_result(result, task, spec, depth, decompose_op=None)
            self.counters["accepts"] += 1
            self._record(
                Attempt(
                    task=name,
                    depth=depth,
                    outcome="evolved",
                    metric=result.metric,
                    generations=result.generations_used,
                    library_key=key,
                    strategy=result.strategy,
                    sample_metrics=_sample_metrics_of(result),
                    size_metrics=dict(result.size_metrics),
                    resource_metrics=dict(result.resource_metrics),
                    report_metric=self._result_report_value(result),
                    task_metrics=_task_metrics_of(result),
                )
            )
            return Solution(key=key, entry_type=self._entry_type_of(result), metric=result.metric, report_metric=self._result_report_value(result))

        timed_out = self._deadline_exceeded()  # sampled ONCE: the failure forensics below must match the decompose gate
        if depth < self.max_depth and not timed_out and not decomposed_first:  # decompose-first already spent its shot
            decompose_started = time.perf_counter()
            solution = self._decompose_and_recurse(task, spec, depth, budget, first_metric=result.metric)
            # Wall time of the whole decompose phase (probe evolves + sub-solves; sub-solve attempts
            # also carry their own rows). Probe strategy time additionally shows under strategy keys.
            if self._active_stages is not None:
                self._active_stages["decompose"] = round(time.perf_counter() - decompose_started, 3)
            if solution is not None:
                return solution
            timed_out = self._deadline_exceeded()

        self.counters["failures"] += 1
        if timed_out:
            # The counter key only exists when max_task_seconds > 0, and timed_out requires it.
            # Decompose was skipped above, so at depth 0 the forensics slot is free to claim.
            self.counters["time_budget_hits"] += 1
            total_deadline = getattr(self, "_total_task_deadline", None)
            if depth == 0 and total_deadline is not None and time.perf_counter() > total_deadline:
                self.counters["total_time_budget_hits"] += 1
            if depth == 0:
                self._failure_stage = "time_budget"
        stone_key = self._admit_stepping_stone(result, task, spec) if self.wall_ledger and depth == 0 else None
        self._record(
            Attempt(
                task=name,
                depth=depth,
                outcome="failed",
                metric=result.metric,
                generations=result.generations_used,
                library_key=stone_key,  # the wall ledger's trace of this failure, when one was shelved
                strategy=result.strategy,
                decompose_op=self._failure_op if depth == 0 else None,
                failure_stage=self._failure_stage if depth == 0 else None,
                sample_metrics=_sample_metrics_of(result),
                size_metrics=dict(result.size_metrics),
                resource_metrics=dict(result.resource_metrics),
                report_metric=self._result_report_value(result),
                task_metrics=_task_metrics_of(result),
            )
        )
        logger.info("orchestrator gave up on %s at depth %d (best %s=%.3f via %s)", name, depth, self.search_metric, result.metric, result.strategy)
        return None

    def _attach_report_metrics(self, result: StrategyResult, report_spec: CompTaskSpec) -> StrategyResult:
        """Evaluate the selected composition once against held-out query tensors.

        Direct strategy champions already attach their one-shot report through their structured or
        flat adapter. Routed-only state has no immutable payload to re-evaluate here; distilled
        routed winners arrive as compositions and use the same rail below.
        """

        if result.champion_comp is None:
            return result
        reported = self.loop.assess_composition(result.champion_comp.comp, report_spec, self.state, train=False)
        if reported.net is None:
            return result
        result.report_metrics = dict(reported.metrics)
        return result

    # --- the evolve step: a config-ordered strategy ladder with budget carry --------------------------

    def _deadline_exceeded(self) -> bool:
        deadline = getattr(self, "_solve_deadline", None)
        return deadline is not None and time.perf_counter() > deadline

    def _total_deadline_exceeded(self) -> bool:
        deadline = getattr(self, "_total_task_deadline", None)
        return deadline is not None and time.perf_counter() > deadline

    def _record_total_timeout(self, task: Task, depth: int) -> None:
        """Record an expired cumulative budget without starting lookup, probing, or evolution."""

        self.counters["failures"] += 1
        self.counters["time_budget_hits"] += 1
        if depth == 0:
            self.counters["total_time_budget_hits"] += 1
            self._failure_stage = "time_budget"
        self._record(
            Attempt(
                task=task.meta.name,
                depth=depth,
                outcome="failed",
                metric=0.0,
                generations=0,
                strategy="time_budget",
                failure_stage="time_budget" if depth == 0 else None,
            )
        )
        logger.info("orchestrator skipped %s at depth %d: cumulative task budget exhausted", task.meta.name, depth)
        return None

    def _wants_decompose_first(self, task: Task, spec: CompTaskSpec) -> bool:
        """True when the task's dense-init gene estimate marks it too wide to evolve flat first."""
        if self.decompose_first_above != "adaptive" and int(self.decompose_first_above) <= 0:
            return False
        io = self._io_of(task, spec)
        init_genes = (int(io["inputs"][0]["width"]) + 1) * int(io["output"]["width"])
        if self.decompose_first_above != "adaptive":
            return init_genes > int(self.decompose_first_above)
        direct = next((strategy for name, strategy in self.strategies if name == "direct"), None)
        evolver = getattr(direct, "evolver", None)
        population_execution = str(getattr(evolver, "execution_mode", "serial")).startswith("population_")
        estimate = self.loop.assess_glue_resources(
            init_genes,
            stage="decompose_first",
            storage="tuple",
            fixed_limit=0,
            population_multiplicity=max(1, int(getattr(evolver, "pop_size", 1))),
            concurrent_trainers=max(1, int(getattr(evolver, "pop_size", 1))) if population_execution else max(1, int(getattr(evolver, "assess_workers", 1))),
        )
        return not estimate.accepted

    def _with_deadline(self, detector: StallDetector) -> Callable[[int, Any], bool]:
        """Chain the per-solve deadline behind a stall detector. With no deadline the detector is
        returned AS-IS (identical object flow, the byte-identical off path); with one, the detector
        still runs FIRST so its flatline state advances exactly as it would unbudgeted."""
        if getattr(self, "_solve_deadline", None) is None:
            return detector

        def stop(generation: int, best: Any) -> bool:
            return detector(generation, best) or self._deadline_exceeded()

        return stop

    def _bounded_stall_detector(self, budget: int) -> Callable[[int, Any], bool]:
        return self._with_deadline(self._stall_detector(budget))

    def _runtime(self) -> StrategyRuntime:
        return StrategyRuntime(
            loop=self.loop,
            library=self.library,
            state=self.state,
            accept_threshold=self.accept_threshold,
            metric_of=self._metric,
            stall_factory=self._bounded_stall_detector,
            on_generation=self._on_generation,
            accepts=self._accepts_item,
            deadline_exceeded=self._deadline_exceeded,
        )

    def _evolve(
        self, task: Task, spec: CompTaskSpec, budget: int, seed_comps: list[CompositionGenome] | None = None, seed_entries: list[LibraryEntry] | None = None
    ) -> StrategyResult:
        """Run the configured strategies in order under one shared budget. First strategy to clear
        the accept threshold wins (later ones never run); a stalled strategy's UNSPENT generations
        roll into the next allocation; the best loser is returned when nobody clears the bar."""
        runtime = self._runtime()
        total_share = sum(self.evolve_shares.values()) or 1.0
        results: list[StrategyResult] = []
        resource_metrics: dict[str, float] = {}
        remaining = budget
        carry = 0
        if self._total_deadline_exceeded():
            return StrategyResult(strategy="time_budget", metric=0.0, generations_used=0)
        for position, (name, strategy) in enumerate(self.strategies):
            # Past the deadline, later ladder stages never start; position 0 always runs so
            # `max(results)` below is never asked to rank an empty list.
            if position > 0 and self._deadline_exceeded():
                break
            if position == len(self.strategies) - 1:
                allocation = remaining
            else:
                base_allocation = max(1, round(budget * self.evolve_shares[name] / total_share))
                allocation = min(remaining, base_allocation + carry)
            if allocation <= 0:
                break
            stage_started = time.perf_counter()
            BOARD.stage(name, f"starting (budget {allocation} gens)")
            if name == "direct" and seed_entries:
                outcome = strategy(task, spec, runtime, budget=allocation, seed_entries=seed_entries)
            else:
                outcome = strategy(task, spec, runtime, budget=allocation, seed_comps=seed_comps if name == "composition" else None)
            stages = getattr(self, "_active_stages", None)
            if stages is not None:
                stages[name] = round(stages.get(name, 0.0) + (time.perf_counter() - stage_started), 3)
            if name == "routed":  # the strategy has no counter access; it stamps markers instead
                for marker in ("routed_no_experts", "routed_undistillable", "routed_resource_declined"):
                    if outcome.champion_metrics.get(marker):
                        self.counters[marker] += 1
            resource_metrics.update(outcome.resource_metrics)
            declined = any(key.endswith("_declined") and value > 0.0 for key, value in outcome.resource_metrics.items())
            if declined and "resource_declines" in self.counters:
                self.counters["resource_declines"] += 1
                counter = f"{name}_resource_declines"
                if counter in self.counters:
                    self.counters[counter] += 1
            results.append(outcome)
            remaining -= outcome.generations_used
            carry = max(0, allocation - outcome.generations_used)
            if self._accepts_result(outcome):
                outcome.resource_metrics = dict(resource_metrics)
                return outcome
            if remaining <= 0:
                break
        # Metric-only diagnostics (for example an adapter-space routed score whose pathway could
        # not be distilled) must not displace a real below-bar champion that the wall ledger can
        # preserve. When every strategy declined, the metric remains a useful failure diagnostic.
        best = max(results, key=lambda item: (item.has_admissible_champion, item.metric))
        best.resource_metrics = dict(resource_metrics)
        return best

    # --- learn-mode refinement of library hits --------------------------------------------------------

    def _handle_library_hit(self, hit: Solution, task: Task, spec: CompTaskSpec, depth: int) -> Solution:
        """STEP 1b: a hit is free, but learn mode (refine budget_k > 0) spends a bounded, decaying
        budget trying to beat the stored solution before settling for it. The guard runs FIRST with
        zero side effects, so budget_k = 0 (live mode) is byte-identical to the plain hit path. The
        task can never regress: a failed refinement returns the original hit."""
        refine_generations = 0
        if self.refine_budget_k > 0 and depth <= self.refine_depth_max and hit.key is not None and not self._total_deadline_exceeded():
            improved, refine_generations = self._refine_hit(hit, task, spec, depth)
            if improved is not None:
                return improved  # _refine_hit already recorded the "refined" attempt
        self._record(
            Attempt(
                task=task.meta.name,
                depth=depth,
                outcome="library_hit",
                metric=hit.metric,
                generations=0,
                library_key=hit.key,
                refine_generations=refine_generations,
                report_metric=hit.report_metric,
                task_metrics=dict(hit.task_metrics),
            )
        )
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
        if self.blind_query:
            result = self._attach_report_metrics(result, comp_task_spec(task, structured_grid=self.structured_grid))
        self.counters["refine_generations"] += result.generations_used

        candidate = self._candidate_rank(result)
        incumbent = self._incumbent_rank(hit, entry, seed_metric=result.seed_metric)
        improves = (
            candidate is not None
            # A robustness/size tie-break can sit epsilon below an at-the-bar incumbent; never shelve below the bar.
            and self._accepts_result(result)
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
            key = self._admit_result(result, task, spec, depth, decompose_op=None)
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
                sample_metrics=_sample_metrics_of(result),
                size_metrics=dict(result.size_metrics),
                resource_metrics=dict(result.resource_metrics),
                report_metric=self._result_report_value(result),
                task_metrics=_task_metrics_of(result),
            )
        )
        return (
            Solution(
                key=key,
                entry_type=self._entry_type_of(result),
                metric=result.metric,
                report_metric=self._result_report_value(result),
                task_metrics=_task_metrics_of(result),
            ),
            result.generations_used,
        )

    def _effective_refine_budget(self, entry: LibraryEntry) -> int:
        summary = self.library.summary(entry.key) or {}
        failures = int((summary.get("stats") or {}).get("refine_failures_since_gain", 0))
        return int(self.refine_budget_k * (self.refine_decay**failures))

    def _refine_runtime(self, target_metric: float) -> StrategyRuntime:
        """Like `_runtime`, but the bar is beating the incumbent and the flatline window is the
        refine-local one (K is a cap, not a fixed cost: a saturated refinement stalls out early).
        The detector's half-budget floor check can never fire: the seed already scores above floor."""

        def refine_stall_factory(budget: int) -> Callable[[int, Any], bool]:
            detector = StallDetector(stall_generations=self.refine_stall_generations, stall_epsilon=self.stall_epsilon, floor=self.floor, budget=budget, metric_of=self._metric)
            return self._with_deadline(detector)

        return StrategyRuntime(
            loop=self.loop,
            library=self.library,
            state=self.state,
            accept_threshold=target_metric,
            metric_of=self._metric,
            stall_factory=refine_stall_factory,
            on_generation=self._on_generation,
            accepts=lambda item: self._metric(item) >= target_metric,
            deadline_exceeded=self._deadline_exceeded,
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

    # --- the wall ledger: failure leaves a trace ------------------------------------------------------

    @staticmethod
    def _io_of(task: Task, spec: CompTaskSpec) -> dict[str, Any]:
        """The task's structural I/O contract, computed once per solve and carried on the spec."""
        return spec.io if spec.io is not None else task_io(task)

    def _wall_stones(self, task: Task, spec: CompTaskSpec) -> list[LibraryEntry]:
        """Stepping stones matching this task's signature, best-ranked first (query order)."""
        io = self._io_of(task, spec)
        candidates = self.library.query(input_signature=io["inputs"][0]["signature"], input_width=io["inputs"][0]["width"], output_width=io["output"]["width"])
        return [entry for entry in candidates if entry.provenance.get("stepping_stone")]

    def _wall_stone_seeds(self, task: Task, spec: CompTaskSpec) -> tuple[list[LibraryEntry], list[CompositionGenome]]:
        """The warm start for a fresh assault on a known wall, split by shape for the seeding rails
        (MODULE stones graft into the direct population, COMPOSITION stones seed the comp loop)."""
        stones = self._wall_stones(task, spec)[: self.wall_seed_top_k]
        modules = [stone for stone in stones if stone.entry_type == MODULE]
        comps = [comp_from_dict(stone.payload) for stone in stones if stone.entry_type == COMPOSITION]
        return modules, comps

    def _admit_stepping_stone(self, result: StrategyResult, task: Task, spec: CompTaskSpec) -> str | None:
        """Shelve a failed attempt's best champion as a below-bar stepping stone: a dependency
        entry (bypasses the admission policy and signature caps, invisible to `signature_group`)
        that can never be a false lookup hit because quick-eval still gates on the accept bar.
        One stone per signature lineage: replaced only on a strict lexicographic AND structural
        win (the refine comparator, reused), so the wall gets chipped, not wallpapered. Free
        synergies: stones enter module-pool absorption and the comp ref catalog through `query`,
        and become router vertices at sync (immature circuits in the overmind, by design)."""
        candidate = self._candidate_rank(result)
        if candidate is None or not math.isfinite(result.metric) or result.metric < self.wall_min_metric:
            return None
        if float(result.champion_metrics.get("support_gain_over_baseline", math.inf)) < self.wall_min_gain_over_baseline:
            return None
        incumbent_stone = next(iter(self._wall_stones(task, spec)), None)
        if incumbent_stone is not None:
            if self._candidate_fingerprint(result) == structural_fingerprint(incumbent_stone.entry_type, incumbent_stone.payload):
                return None
            stone_solution = Solution(key=incumbent_stone.key, entry_type=incumbent_stone.entry_type, metric=float(incumbent_stone.provenance.get("accepted_metric", 0.0)))
            stone_rank = self._incumbent_rank(stone_solution, incumbent_stone)
            if not refinement_improves(candidate, stone_rank, metric_epsilon=self.refine_metric_epsilon, robustness_epsilon=self.refine_robustness_epsilon):
                return None
        self._stepping_stone = True
        self._refined_from = incumbent_stone.key if incumbent_stone is not None else None
        try:
            key = self._admit_result(result, task, spec, depth=0, decompose_op=None)
        finally:
            self._stepping_stone = False
            self._refined_from = None
        if key is None:
            return None
        if incumbent_stone is not None:
            self.library.retire(incumbent_stone.key)  # the strict win above is the dominance proof
            self.counters["wall_stones_improved"] += 1
        else:
            self.counters["wall_stones_admitted"] += 1
        logger.info("wall ledger shelved stepping stone %s for %s (best %s=%.3f)", key, task.meta.name, self.search_metric, result.metric)
        return key

    # --- ladder steps -------------------------------------------------------------------------------

    def _lookup(self, task: Task, spec: CompTaskSpec) -> Solution | None:
        io = self._io_of(task, spec)
        candidates = self.library.query(
            input_signature=io["inputs"][0]["signature"],
            input_width=io["inputs"][0]["width"],
            output_width=io["output"]["width"],
            limit=self.quick_eval_top_k,
        )
        for entry in candidates:
            if self._total_deadline_exceeded():
                return None
            assessment = self._quick_assessment(entry, task, spec)
            if assessment is None or not self._accepts_item(assessment):
                continue
            metric = self._metric(assessment)
            report_assessment = self._quick_assessment(entry, task, comp_task_spec(task, structured_grid=self.structured_grid)) if self.blind_query else assessment
            report_metric = self._report(report_assessment) if self.blind_query and report_assessment is not None else metric
            combined = dict(assessment.metrics) | (dict(report_assessment.metrics) if report_assessment is not None else {})
            return Solution(key=entry.key, entry_type=entry.entry_type, metric=metric, report_metric=report_metric, task_metrics=_task_metrics(combined))
        return None

    @staticmethod
    def _entry_is_temporal(entry: LibraryEntry) -> bool:
        signature = entry.io["inputs"][0].get("signature", "")
        return "|" in signature and "T" in signature.split("|", 1)[1].split(",")

    def _quick_assessment(self, entry: LibraryEntry, task: Task, spec: CompTaskSpec) -> AssessedComposition | None:
        """Evaluate a stored entry against the task with NO training (forward passes only).

        TIME-bearing MODULE entries (direct-strategy temporal winners) are scored through the
        stepped substrate; everything else uses the flat decode."""
        from ardevo.evaluation import evaluate
        from ardevo.evolution.composition import AssemblyContext, CompositionAssemblyError, assemble

        try:
            if entry.entry_type == MODULE and self._entry_is_temporal(entry):
                adapter = temporal_adapter(task, max_inline_depth=self.loop.max_inline_depth)
                if spec.encoded.query_input is None:
                    adapter.encoded = without_query(adapter.encoded)
                module = decode_recurrent(
                    genome_from_dict(entry.payload),
                    adapter.n_inputs,
                    adapter.n_outputs,
                    adapter.mode,
                    macro_resolver=macro_resolver(self.library),
                    max_inline_depth=self.loop.max_inline_depth,
                    _reference_stack=(entry.key,),
                )
                metrics = adapter.evaluate(module)
                return AssessedComposition(comp=CompositionGenome(), metrics=metrics, fitness=0.0, net=None)
            if entry.entry_type == MODULE:
                module = decode_module(
                    genome_from_dict(entry.payload),
                    spec.n_inputs,
                    spec.output_width,
                    macro_resolver=macro_resolver(self.library),
                    max_inline_depth=self.loop.max_inline_depth,
                    _reference_stack=(entry.key,),
                )
            else:
                comp = comp_from_dict(entry.payload)
                ctx = AssemblyContext(
                    bank_columns=dict(spec.bank_columns),
                    library=self.library,
                    max_inline_depth=self.loop.max_inline_depth,
                    expansion_stack=[entry.key],
                )
                module = assemble(comp, ctx, spec.n_inputs)
        except (ValueError, CompositionAssemblyError) as error:
            logger.debug("library candidate %s not evaluable here: %s", entry.key, error)
            return None
        metrics = evaluate(module, spec.encoded, spec.encoder)
        return AssessedComposition(comp=CompositionGenome(), metrics=metrics, fitness=0.0, net=None)

    def _quick_metric(self, entry: LibraryEntry, task: Task, spec: CompTaskSpec, *, report: bool = False) -> float | None:
        assessment = self._quick_assessment(entry, task, spec)
        if assessment is None:
            return None
        return self._report(assessment) if report else self._metric(assessment)

    def _decompose_and_recurse(self, task: Task, spec: CompTaskSpec, depth: int, budget: int, first_metric: float) -> Solution | None:
        if self._total_deadline_exceeded():
            return None
        chosen_name: str | None = None
        subtasks: list[Subtask] = []
        for op_name, op in self.decomposers:
            if self._total_deadline_exceeded():
                return None
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
            if self._total_deadline_exceeded():
                return None
            solved = self.solve(subtask.task, depth + 1)
            if solved is None:
                # A missing part means the wired parent cannot be completed; record WHERE it died.
                self.counters["decompose_subtask_failed"] += 1
                self._failure_stage = f"subtask:{subtask.task.meta.name}"
                self._failure_op = chosen_name
                return None
            solutions.append((subtask, solved))

        if self._total_deadline_exceeded():
            return None
        seed = self._port_wired_skeleton(spec, solutions)
        seeds = [seed] if seed is not None else None
        retry_budget = max(budget // 2, 5)
        if self._total_deadline_exceeded():
            return None
        result = self._evolve(task, spec, retry_budget, seed_comps=seeds)
        if self.blind_query:
            result = self._attach_report_metrics(result, comp_task_spec(task, structured_grid=self.structured_grid))
        if self._accepts_result(result):
            key = self._admit_result(result, task, spec, depth, decompose_op=chosen_name)
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
                sample_metrics=_sample_metrics_of(result),
                size_metrics=dict(result.size_metrics),
                resource_metrics=dict(result.resource_metrics),
                report_metric=self._result_report_value(result),
                task_metrics=_task_metrics_of(result),
            )
            self._record(attempt)
            return Solution(
                key=key,
                entry_type=self._entry_type_of(result),
                metric=result.metric,
                report_metric=self._result_report_value(result),
                task_metrics=_task_metrics_of(result),
            )
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
            if self._total_deadline_exceeded():
                return False
            spec = comp_task_spec(subtask.task, include_query=not self.blind_query, structured_grid=self.structured_grid)
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

    def _admit_result(self, result: StrategyResult, task: Task, spec: CompTaskSpec, depth: int, decompose_op: str | None) -> str | None:
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
            return self._admit(result.champion_comp, task, spec, depth, decompose_op)
        if result.champion_genome is not None:
            return self._admit_direct_module(result, task, spec, depth)
        raise ValueError(f"strategy {result.strategy!r} produced no admissible champion")

    def _admit_direct_module(self, result: StrategyResult, task: Task, spec: CompTaskSpec, depth: int) -> str | None:
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
        if self._stepping_stone:
            provenance["stepping_stone"] = True  # a below-bar wall-ledger trace, not a solution
        level = module_level(result.champion_genome, self.library)
        return self._gated_add(
            entry_type=MODULE,
            payload=genome_to_dict(result.champion_genome),
            io=self._io_of(task, spec),
            provenance=provenance,
            level=level,
            dependency=depth > 0 or self._stepping_stone,
        )

    def _admit(self, best: AssessedComposition, task: Task, spec: CompTaskSpec, depth: int, decompose_op: str | None) -> str | None:
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
        if self._stepping_stone:
            provenance["stepping_stone"] = True  # a below-bar wall-ledger trace, not a solution
        return self._gated_add(
            entry_type=COMPOSITION, payload=comp_to_dict(detached), io=self._io_of(task, spec), provenance=provenance, level=level, dependency=depth > 0 or self._stepping_stone
        )

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

        signature, parent_width = spec.input_specs[0]
        loaded: list[tuple[Subtask, str, LibraryEntry]] = []
        glue_values = spec.output_width  # bias -> output
        for subtask, solution in solutions:
            key = solution.key
            if key is None:  # unreachable after the guard above; narrows for the type checker
                return None
            entry = self.library.load(key)
            in_width = sum(item["width"] for item in entry.io["inputs"])
            out_width = entry.io["output"]["width"]
            if subtask.port.role == "output_slice":
                glue_values += parent_width * in_width + out_width * spec.output_width
            else:
                subset_width = max(subtask.port.offsets[1] - subtask.port.offsets[0], 0)
                glue_values += parent_width * subset_width + out_width * spec.output_width
            loaded.append((subtask, key, entry))

        estimate = self.loop.assess_glue_resources(glue_values, stage="decomposition_skeleton", device="cpu")
        if not estimate.accepted:
            if "resource_declines" in self.counters:
                self.counters["resource_declines"] += 1
                self.counters["decomposition_resource_declines"] += 1
            logger.warning(
                "decomposition skeleton declined before allocation: candidate needs %s glue values (%s host, %s device; limit %s)",
                f"{glue_values:,}",
                format_bytes(estimate.host_required_bytes),
                format_bytes(estimate.device_required_bytes),
                f"{estimate.limit_values:,}",
            )
            return None

        tracker = self.state.comp_innovations
        comp = CompositionGenome()
        input_id = tracker.new_node_id()
        comp.nodes[input_id] = CompNodeGene(input_id, CompNodeKind.INPUT, signature, 0, parent_width)
        bias_id = tracker.new_node_id()
        comp.nodes[bias_id] = CompNodeGene(bias_id, CompNodeKind.INPUT, BIAS_REF, 0, 1)
        output_id = tracker.new_node_id()
        comp.nodes[output_id] = CompNodeGene(output_id, CompNodeKind.OUTPUT, spec.output_ref, spec.output_width, 0)
        comp.edges.append(
            CompEdgeGene(
                bias_id,
                output_id,
                True,
                tracker.innovation(bias_id, output_id),
                _zero_glue(spec.output_width, self.loop.glue_storage),
            )
        )

        positions_total = sum(subtask.port.width for subtask, _ in solutions if subtask.port.role == "output_slice")
        per_position = spec.output_width // positions_total if positions_total else 1

        for subtask, key, entry in loaded:
            in_width = sum(item["width"] for item in entry.io["inputs"])
            out_width = entry.io["output"]["width"]
            node_id = tracker.new_node_id()
            comp.nodes[node_id] = CompNodeGene(node_id, CompNodeKind.MODULE, f"library:{key}", in_width, out_width)
            port = subtask.port
            if port.role == "output_slice":
                in_glue = _identity_glue(parent_width, in_width, self.loop.glue_storage)
                start = port.offsets[0] * per_position
                out_glue = _placement_glue(out_width, spec.output_width, start, self.loop.glue_storage)
            else:  # input_subset
                in_glue = _selection_glue(parent_width, port.offsets, self.loop.glue_storage)
                out_glue = _identity_glue(out_width, spec.output_width, self.loop.glue_storage)
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
        if self.search_metric == "support_task_appropriate":
            return float(metrics.get("support_task_exact", metrics.get("support_accuracy", 0.0)))
        if self.search_metric == "query_task_appropriate":
            if not math.isfinite(metrics.get("query_loss", math.inf)):
                return float(metrics.get("support_task_exact", metrics.get("support_accuracy", 0.0)))
            return float(metrics.get("query_task_exact", metrics.get("query_accuracy", 0.0)))
        if self.search_metric == "query_accuracy" and not math.isfinite(metrics.get("query_loss", math.inf)):
            return float(metrics.get("support_accuracy", 0.0))  # degenerate query-less task
        return float(metrics.get(self.search_metric, 0.0))

    def _accept_value(self, metrics: dict[str, float]) -> float:
        if self.accept_metric == "support_task_appropriate":
            return float(metrics.get("support_task_exact", metrics.get("support_accuracy", 0.0)))
        if self.accept_metric == "query_task_appropriate":
            if "query_loss" in metrics and not math.isfinite(metrics["query_loss"]):
                return float(metrics.get("support_task_exact", metrics.get("support_accuracy", 0.0)))
            return float(metrics.get("query_task_exact", metrics.get("query_accuracy", 0.0)))
        if self.accept_metric == "query_accuracy" and "query_loss" in metrics and not math.isfinite(metrics["query_loss"]):
            return float(metrics.get("support_accuracy", 0.0))
        return float(metrics.get(self.accept_metric, 0.0))

    def _accepts_metrics(self, metrics: dict[str, float]) -> bool:
        return self._accept_value(metrics) >= self.accept_threshold

    def _accepts_item(self, item: Any) -> bool:
        return self._accepts_metrics(item.metrics)

    def _accepts_result(self, result: StrategyResult) -> bool:
        # A score is not a solution unless the strategy can hand admission an executable payload.
        # This defense is intentionally strategy-agnostic: declined guards, empty grammars, and
        # undistillable router wins all use metric-only StrategyResults to continue the ladder.
        return result.has_admissible_champion and self._accepts_metrics(result.champion_metrics)

    def _report(self, item: Any) -> float:
        return self._report_value(item.metrics) or 0.0

    def _result_report_value(self, result: StrategyResult) -> float | None:
        return self._report_value(result.report_metrics) if result.report_metrics else None

    def _report_value(self, metrics: dict[str, float]) -> float | None:
        if not self.blind_query:
            return None
        if self.report_metric == "query_task_appropriate":
            if not math.isfinite(metrics.get("query_loss", math.inf)):
                return None
            return float(metrics.get("query_task_exact", metrics.get("query_accuracy", 0.0)))
        value = metrics.get(self.report_metric)
        return float(value) if value is not None and math.isfinite(float(value)) else None

    def _record(self, attempt: Attempt) -> None:
        # Stamp wall-clock forensics from the enclosing solve() unless the caller already did.
        started = getattr(self, "_solve_started", None)
        if attempt.seconds == 0.0 and started is not None:
            attempt.seconds = round(time.perf_counter() - started, 3)
            attempt.stage_seconds = dict(getattr(self, "_active_stages", None) or {})
        self.attempts.append(attempt)
        logger.info("attempt: %s", attempt.to_dict())
        BOARD.event(f"{attempt.task} d{attempt.depth}: {attempt.outcome} {self.search_metric}={attempt.metric:.3f}")

    def _on_generation(self, strategy: str, generation: int, best: Any, mean_fitness: float) -> None:
        BOARD.generation(strategy, generation, best.fitness, self._metric(best), mean_fitness)
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


def _stored_glue(values: Any, storage: str) -> GlueValues:
    return array("f", values) if storage == "f32" else tuple(values)


def _zero_glue(count: int, storage: str) -> GlueValues:
    return array("f", [0.0]) * count if storage == "f32" else tuple(0.0 for _ in range(count))


def _identity_glue(in_width: int, out_width: int, storage: str = "tuple") -> GlueValues:
    """Row-major identity-ish map: 1.0 on the diagonal, zero elsewhere (rectangular allowed)."""
    return _stored_glue((1.0 if row == column else 0.0 for row in range(in_width) for column in range(out_width)), storage)


def _placement_glue(in_width: int, out_width: int, start: int, storage: str = "tuple") -> GlueValues:
    """Maps a module's output block onto its slice of the parent head: out[start + r] = in[r]."""
    return _stored_glue((1.0 if row == start + column else 0.0 for row in range(in_width) for column in range(out_width)), storage)


def _selection_glue(in_width: int, offsets: tuple[int, int], storage: str = "tuple") -> GlueValues:
    """Selects parent input columns [start, end) into a module's input ports."""
    start, end = offsets
    out_width = end - start
    return _stored_glue((1.0 if start + column == row else 0.0 for row in range(in_width) for column in range(out_width)), storage)


def attempts_to_dicts(attempts: list[Attempt]) -> list[dict[str, Any]]:
    return [attempt.to_dict() for attempt in attempts]


def attempts_from_dicts(data: list[dict[str, Any]]) -> list[Attempt]:
    return [Attempt.from_dict(item) for item in data]
