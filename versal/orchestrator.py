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
import random
import time
from array import array
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable

from versal.cross_validation import CrossValidationConfig, SupportCrossValidator, fresh_composition_glue, fresh_genome_weights
from versal.dataset.icarus import Level0Encoder, Task, encode_task
from versal.decompose import Subtask, build_decomposers
from versal.evaluation import fit_query_target, without_query
from versal.evolution.composition import BIAS_REF, CompEdgeGene, CompNodeGene, CompNodeKind, CompositionGenome, GlueValues, IndexRun, PortMap, comp_from_dict, comp_to_dict
from versal.evolution.evolver import EvolverState
from versal.evolution.genome import Genome, InnovationTracker, genome_from_dict, genome_to_dict
from versal.evolution.loop import AssessedComposition, CompTaskSpec, HierarchicalLoop, HierarchicalState, assess_composition_pure
from versal.evolution.train import _writeback
from versal.library import (
    COMPOSITION,
    LIBRARY_ADMISSION,
    MODULE,
    LibraryEntry,
    ModuleLibrary,
    expanded_payload_complexity,
    macro_resolver,
    module_level,
    structural_fingerprint,
    task_io,
)
from versal.strategy import StrategyResult, StrategyRuntime, build_strategies
from versal.substrate import decode_module, decode_recurrent
from versal.temporal import temporal_adapter
from versal.topology import TopologyTabuSession, TopologyTabuStore, refinement_context, refinement_lineage_root, same_topology, task_content_fingerprint, topology_record
from versal.utils.logging import Logger
from versal.utils.resources import format_bytes
from versal.utils.runtime_display import NULL_DISPLAY
from versal.utils.status import BOARD

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
        from versal.structured import encode_structured_grid

        encoded = encode_structured_grid(task, encoder, include_query=include_query)
    if encoded is None:
        encoded = fit_query_target(encode_task(task, encoder))
    if not include_query:
        from versal.structured import StructuredGridEncoded

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
    strategy_metrics: dict[str, float] = field(default_factory=dict)
    validation_status: str = "not_run"
    validation_metrics: dict[str, float] = field(default_factory=dict)
    # Best non-parent observation encountered while solving this task. It is explicitly separate
    # from the literal parent accuracy rails: a router score or solved recursive child is evidence
    # about search progress, not an executable answer to the parent task.
    diagnostic_observation: dict[str, Any] = field(default_factory=dict)
    # Literal quality observations. `metric` and `report_metric` remain the configurable policy
    # scores; these fields always mean sample-level accuracy and never silently substitute one for
    # the other. Statuses make a real 0.0 distinguishable from an evaluation that never happened.
    support_accuracy: float | None = None
    query_accuracy: float | None = None
    support_status: str = "legacy_missing"
    query_status: str = "legacy_missing"
    representation: str | None = None

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
        if self.strategy_metrics:
            data["strategy_metrics"] = self.strategy_metrics
        if self.validation_status != "not_run" or self.validation_metrics:
            data["validation_status"] = self.validation_status
            data["validation_metrics"] = self.validation_metrics
        if self.diagnostic_observation:
            data["diagnostic_observation"] = self.diagnostic_observation
        if self.support_status != "legacy_missing" or self.query_status != "legacy_missing" or self.support_accuracy is not None or self.query_accuracy is not None:
            data.update(
                {
                    "support_accuracy": self.support_accuracy,
                    "query_accuracy": self.query_accuracy,
                    "support_status": self.support_status,
                    "query_status": self.query_status,
                }
            )
        if self.representation is not None:
            data["representation"] = self.representation
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
            strategy_metrics={str(key): float(value) for key, value in data.get("strategy_metrics", {}).items()},
            validation_status=str(data.get("validation_status", "not_run")),
            validation_metrics={str(key): float(value) for key, value in data.get("validation_metrics", {}).items()},
            diagnostic_observation=dict(data.get("diagnostic_observation", {})),
            support_accuracy=float(data["support_accuracy"]) if data.get("support_accuracy") is not None else None,
            query_accuracy=float(data["query_accuracy"]) if data.get("query_accuracy") is not None else None,
            support_status=str(data.get("support_status", "legacy_missing")),
            query_status=str(data.get("query_status", "legacy_missing")),
            representation=str(data["representation"]) if data.get("representation") is not None else None,
        )


_SAMPLE_METRIC_KEYS = ("mean_sample_accuracy", "max_sample_accuracy", "best_sample_weight", "weight_robustness")


def _sample_metrics_of(result: StrategyResult) -> dict[str, float]:
    # mean_sample_accuracy marks a real weight-sample measurement; _floored_metrics carries a bare
    # weight_robustness = 0.0 that must not fabricate a diagnostic row under standard eval.
    if "mean_sample_accuracy" not in result.champion_metrics:
        return {}
    return {key: float(result.champion_metrics[key]) for key in _SAMPLE_METRIC_KEYS if key in result.champion_metrics}


_TASK_METRIC_SUFFIXES = (
    "_exact",
    "_task_exact",
    "_shape_accuracy",
    "_baseline_accuracy",
    "_gain_over_baseline",
    "_coverage",
    "_covered_accuracy",
    "_correct_cells",
    "_valid_cells",
    "_exact_examples",
    "_total_examples",
)


def _task_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if key.startswith(("support_", "query_")) and key.endswith(_TASK_METRIC_SUFFIXES) and math.isfinite(float(value))}


def _task_metrics_of(result: StrategyResult) -> dict[str, float]:
    return _task_metrics(dict(result.champion_metrics) | dict(result.report_candidate_metrics) | dict(result.report_metrics))


def _finite_accuracy(metrics: dict[str, float], key: str, *, loss_key: str | None = None) -> float | None:
    """Read a literal accuracy without turning a missing/invalid evaluation into a zero."""

    if loss_key is not None and not math.isfinite(float(metrics.get(loss_key, math.inf))):
        return None
    value = metrics.get(key)
    return float(value) if value is not None and math.isfinite(float(value)) else None


@dataclass(frozen=True)
class Solution:
    key: str | None  # None when the task was SOLVED but the admission gate declined to shelve it
    entry_type: str
    metric: float
    report_metric: float | None = None
    task_metrics: dict[str, float] = field(default_factory=dict)
    support_accuracy: float | None = None
    query_accuracy: float | None = None
    support_status: str = "legacy_missing"
    query_status: str = "legacy_missing"


@dataclass(frozen=True)
class RefinementRank:
    """The lexicographic standing of one topology in a refine-on-hit comparison."""

    metric: float
    robustness: float
    complexity: int
    entry_type: str


def refinement_improves(candidate: RefinementRank, incumbent: RefinementRank, *, metric_epsilon: float, robustness_epsilon: float) -> bool:
    """Prefer simpler executable structure inside a non-regressing accuracy band.

    Expanded complexity makes modules and compositions comparable. Pareto selection still explores
    novelty and robustness; this final gate deterministically chooses whether to replace a known-good
    solution and never treats an equal candidate as an improvement.
    """
    if not (math.isfinite(candidate.metric) and math.isfinite(candidate.robustness)):
        return False
    if candidate.metric > incumbent.metric + metric_epsilon:
        return True
    if candidate.metric < incumbent.metric - metric_epsilon:
        return False
    if candidate.complexity < incumbent.complexity:
        return True
    if candidate.complexity > incumbent.complexity:
        return False
    return candidate.robustness > incumbent.robustness + robustness_epsilon


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
    def __init__(
        self,
        config: dict[str, Any],
        loop: HierarchicalLoop,
        library: ModuleLibrary,
        state: HierarchicalState,
        proctor: Any = None,
        shutdown_requested: Callable[[], bool] | None = None,
    ) -> None:
        table = config.get("orchestrator", {})
        self.accept_metric = str(table.get("accept_metric", "query_accuracy"))
        self.search_metric = str(table.get("search_metric", self.accept_metric))
        self.report_metric = str(table.get("report_metric", self.accept_metric))
        self.blind_query = bool(table.get("blind_query", False))
        self.structured_grid = bool(table.get("direct", {}).get("structured_grid", False))
        self.accept_threshold = float(table.get("accept_threshold", 0.95))
        self.cross_validation_config = CrossValidationConfig.from_table(table.get("cross_validation"), seed=int(config.get("run", {}).get("seed", 0)))
        self.cross_validator = SupportCrossValidator(self.cross_validation_config, accept_threshold=self.accept_threshold)
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
        self.decompose_probe_seconds = min(30.0, max(0.1, float(table.get("decompose_probe_seconds", 30.0))))
        self.decompose_leaf_cap = min(64, max(2, int(table.get("decompose_leaf_cap", 64))))
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
        routed_lifecycle = (table.get("routed", {}) or {}).get("lifecycle", {}) or {}
        self.route_lifecycle_enabled = bool(routed_lifecycle.get("enabled", False))
        self.library_patience_tasks = max(0, int(routed_lifecycle.get("library_patience_tasks", 0)))
        library.configure_lifecycle(library_patience_tasks=self.library_patience_tasks if self.route_lifecycle_enabled else 0)
        shares = table.get("evolve_budget", {})
        self.evolve_shares = {name: float(shares.get(name, 1.0)) for name, _strategy in self.strategies}
        # Learn-mode refinement of library hits ([orchestrator.refine]). budget_k = 0 is live mode:
        # hits stay zero-cost and the whole path below is byte-identical to the plain hit branch.
        refine = table.get("refine", {}) or {}
        self.refine_mode = str(refine.get("mode", "decay"))
        if self.refine_mode not in {"decay", "always"}:
            raise ValueError("[orchestrator.refine] mode must be 'decay' or 'always'")
        self.refine_budget_k = int(refine.get("budget_k", 0))
        self.refine_depth_max = int(refine.get("depth_max", 0))  # > 0 also refines sub-solve hits, whose improvements admit UNCONDITIONALLY (dependency rail)
        self.refine_metric_epsilon = float(refine.get("metric_epsilon", 0.005))
        self.refine_robustness_epsilon = float(refine.get("robustness_epsilon", 0.01))
        self.refine_decay = float(refine.get("decay", 0.5))
        self.refine_min_generations = int(refine.get("min_generations", 4))
        self.refine_stall_generations = int(refine.get("stall_generations", 8))
        self.refine_retire_superseded = bool(refine.get("retire_superseded", True))
        self.refine_deduplicate_topologies = bool(refine.get("deduplicate_topologies", False))
        self.refine_topology_retry_limit = max(0, int(refine.get("topology_retry_limit", 8)))
        self.config_fingerprint = str(config.get("config_effective_sha256", config.get("config_sha256", "unversioned")))
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
        self._shutdown_requested_callback = shutdown_requested
        self.display = getattr(proctor, "display", NULL_DISPLAY)
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
            if self.refine_deduplicate_topologies:
                self.counters.update(
                    {
                        "topology_unique_evaluated": 0,
                        "topology_duplicates_skipped": 0,
                        "topology_retry_exhaustions": 0,
                        "topology_exhausted_attempts": 0,
                    }
                )
        if any(name == "routed" for name, _strategy in self.strategies):  # registered only when the routed strategy is configured
            # routed_solved counts DISTILLED admissions (the router's win became a library composition);
            # no_experts = short-circuited on an empty vertex set; undistillable = won in router space
            # but the pathway did not survive verification as a composition (reported as a miss).
            self.counters.update({"routed_solved": 0, "routed_zero_shot": 0, "routed_no_experts": 0, "routed_undistillable": 0, "routed_resource_declined": 0})
            if self.route_lifecycle_enabled:
                self.counters.update({"router_vertices_expired": 0, "router_vertices_revived": 0, "router_edges_expired": 0, "library_inactivity_retired": 0})
        if self.wall_ledger:
            self.counters.update({"wall_stones_admitted": 0, "wall_stones_improved": 0, "wall_seeded_attempts": 0})
        if self.max_task_seconds > 0 or self.max_total_task_seconds > 0:  # absent limits keep legacy summaries byte-identical
            self.counters.update({"time_budget_hits": 0})
        if self.max_total_task_seconds > 0:
            self.counters.update({"total_time_budget_hits": 0})
        if self.loop.resource_policy.mode == "adaptive":
            self.counters.update({"resource_declines": 0, "direct_resource_declines": 0, "composition_resource_declines": 0, "decomposition_resource_declines": 0})
        direct_table = table.get("direct", {}) or {}
        fixed_direct_guards = (direct_table.get("max_flat_outputs", 0), direct_table.get("max_init_genes", 0))
        if any(value != "adaptive" and int(value) > 0 for value in fixed_direct_guards):
            self.counters["direct_guard_declines"] = 0
        if self.decompose_first_above == "adaptive" or int(self.decompose_first_above) > 0:  # registered only when the policy is on
            self.counters.update({"decompose_first": 0})
        self._failure_stage: str | None = None
        self._failure_op: str | None = None
        self._refined_from: str | None = None  # lineage provenance for the admission inside a refine
        self._stepping_stone = False  # marks the admission inside a wall-ledger shelving
        self._last_refine_strategy_metrics: dict[str, float] = {}
        self._last_decompose_report_overrun = False

    # --- public API ---------------------------------------------------------------------------------

    def solve(self, task: Task, depth: int = 0) -> Solution | None:
        # Per-solve wall-clock forensics, save/restored so recursive sub-solves time themselves
        # without clobbering the parent's ledger. The deadline rides the same tuple: each depth
        # gets a fresh budget, but a sub-solve only starts while its parent still has time.
        previous_timing = (
            getattr(self, "_solve_started", None),
            getattr(self, "_active_stages", None),
            getattr(self, "_solve_deadline", None),
            getattr(self, "_task_deadline", None),
        )
        previous_total_deadline = getattr(self, "_total_task_deadline", None)
        previous_display_depth = getattr(self, "_display_depth", 0)
        now = time.perf_counter()
        if depth == 0:
            self._total_task_deadline: float | None = (now + self.max_total_task_seconds) if self.max_total_task_seconds > 0 else None
        self._solve_started: float | None = now
        self._active_stages: dict[str, float] | None = {}
        self._display_depth = depth
        local_deadline = (now + self.max_task_seconds) if self.max_task_seconds > 0 else None
        total_deadline = getattr(self, "_total_task_deadline", None)
        deadlines = [deadline for deadline in (local_deadline, total_deadline) if deadline is not None]
        self._task_deadline = min(deadlines) if deadlines else None
        available = max(0.0, self._task_deadline - now) if self._task_deadline is not None else None
        reserve = self.cross_validation_config.reserve_seconds(available)
        self._solve_deadline = self._task_deadline - reserve if self._task_deadline is not None else None
        BOARD.clock(available)
        try:
            return self._solve_timed(task, depth)
        finally:
            self._solve_started, self._active_stages, self._solve_deadline, self._task_deadline = previous_timing
            self._display_depth = previous_display_depth
            if depth == 0:
                self._total_task_deadline = previous_total_deadline

    def _solve_timed(self, task: Task, depth: int = 0) -> Solution | None:
        if depth == 0:
            self._failure_stage = None  # forensics for THIS top-level task only
            self._failure_op = None
            self._best_diagnostic_observation: dict[str, Any] | None = None
            self._best_parent_result: StrategyResult | None = None
            self._best_parent_report_result: StrategyResult | None = None
            self._best_parent_field_result: StrategyResult | None = None
            self._decomposition_leaf_count = 1
        if self._shutdown_requested():
            return self._record_shutdown(task, depth)
        if self._total_deadline_exceeded():
            return self._record_total_timeout(task, depth)
        spec = comp_task_spec(task, include_query=not self.blind_query, structured_grid=self.structured_grid)
        report_spec = comp_task_spec(task, structured_grid=self.structured_grid) if self.blind_query else spec
        name = task.meta.name

        self.display.stage_started("lookup")
        lookup_started = time.perf_counter()
        hit = self._lookup(task, spec)
        if hit is not None:
            self.counters["library_hits"] += 1
            self.display.stage_result(
                "lookup",
                "hit",
                "compatible learned solution cleared the support gate",
                seconds=time.perf_counter() - lookup_started,
                depth=depth,
                support_accuracy=hit.support_accuracy,
            )
            return self._handle_library_hit(hit, task, spec, depth)
        self.display.stage_result("lookup", "miss", "no compatible learned solution cleared the support gate", seconds=time.perf_counter() - lookup_started, depth=depth)
        if self._shutdown_requested():
            return self._record_shutdown(task, depth)
        if self._total_deadline_exceeded():
            return self._record_total_timeout(task, depth)
        self.counters["library_misses"] += 1
        self.loop.absorb_new_entries(self.state)  # fresh library knowledge enters the module pool

        budget = self._budget(depth)
        result: StrategyResult | None = None
        support_timed_out: bool | None = None
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
            if self._deadline_exceeded():
                result = self._remember_or_recover_parent_result(StrategyResult("decompose", metric=0.0, generations_used=0), depth)
                support_timed_out = self._support_timed_out(result) and not self._last_decompose_report_overrun
            # Fall through to the ordinary ladder: a scale-safe init (factored/sparse) rides the
            # same config, so the flat attempt below stays affordable even at init-wall widths.

        if result is None:
            stone_modules, stone_comps = self._wall_stone_seeds(task, spec) if self.wall_ledger and depth == 0 else ([], [])
            if stone_modules or stone_comps:
                self.counters["wall_seeded_attempts"] += 1
            result = self._evolve(task, spec, budget, seed_comps=stone_comps or None, seed_entries=stone_modules or None)
            result = self._remember_or_recover_parent_result(result, depth)
            support_timed_out = self._support_timed_out(result)
        assert result is not None and support_timed_out is not None
        report_result = self._report_result_for(result, depth)
        if self.blind_query and not self._shutdown_requested():
            report_result = self._attach_report_metrics(report_result, task, report_spec)
            _support, query, _support_status, query_status = self._quality_of_result(report_result)
            if report_result.has_report_candidate:
                self.display.query_result(query, query_status, depth=depth)
        stopping = self._shutdown_requested()
        if not support_timed_out and not stopping and self._accepts_result(result):
            key = self._admit_result(result, task, spec, depth, decompose_op=None)
            self.display.stage_result(
                "persist",
                "saved" if key is not None else "skipped",
                "solution retained for reuse" if key is not None else "solution accepted but not retained by library policy",
                depth=depth,
            )
            self.counters["accepts"] += 1
            self._record(self._attempt_from_result(result, task=name, depth=depth, outcome="evolved", library_key=key, report_result=report_result))
            return self._solution_from_result(result, key, report_result=report_result)

        # The held-out pass is intentionally allowed to overrun the support deadline. Such an
        # overrun blocks new decomposition work, but it must not relabel completed support search
        # as a timeout. A later decomposition can still establish a genuine support timeout below.
        timed_out = support_timed_out
        deadline_blocks_more_work = self._deadline_exceeded()
        halted = stopping or deadline_blocks_more_work
        if depth < self.max_depth and not halted and not decomposed_first:  # decompose-first already spent its shot
            decompose_started = time.perf_counter()
            solution = self._decompose_and_recurse(task, spec, depth, budget, first_metric=result.metric)
            # Wall time of the whole decompose phase (probe evolves + sub-solves; sub-solve attempts
            # also carry their own rows). Probe strategy time additionally shows under strategy keys.
            if self._active_stages is not None:
                self._active_stages["decompose"] = round(time.perf_counter() - decompose_started, 3)
            if solution is not None:
                return solution
            result = self._remember_or_recover_parent_result(result, depth)
            stopping = self._shutdown_requested()
            timed_out = self._support_timed_out(result) and not self._last_decompose_report_overrun
            halted = stopping or timed_out

        self.counters["failures"] += 1
        if timed_out:
            self.counters["time_budget_hits"] += 1
            total_deadline = getattr(self, "_total_task_deadline", None)
            if depth == 0 and total_deadline is not None and time.perf_counter() > total_deadline:
                self.counters["total_time_budget_hits"] += 1
            if depth == 0:
                self._failure_stage = "time_budget"
        elif stopping and depth == 0:
            self._failure_stage = "shutdown_requested"
        stone_keys = self._admit_failure_stepping_stones(result, task, spec) if self.wall_ledger and depth == 0 else []
        stone_key = stone_keys[0] if stone_keys else None
        if stone_keys:
            detail = "below-threshold stepping stone retained for a later attempt"
            if len(stone_keys) > 1:
                detail = f"{len(stone_keys)} representation-specific stepping stones retained for later attempts"
            self.display.stage_result("persist", "saved", detail, depth=depth)
        report_result = self._report_result_for(result, depth)
        if self.blind_query and not stopping:
            report_result = self._attach_report_metrics(report_result, task, report_spec)
        self._record(
            self._attempt_from_result(
                result,
                task=name,
                depth=depth,
                outcome="failed",
                library_key=stone_key,
                decompose_op=self._failure_op if depth == 0 else None,
                failure_stage=self._failure_stage if depth == 0 else None,
                query_status="time_limit_before_evaluation" if timed_out else ("shutdown_before_evaluation" if stopping else None),
                report_result=report_result,
                suppress_query=stopping,
            )
        )
        logger.debug("orchestrator gave up on %s at depth %d (best %s=%.3f via %s)", name, depth, self.search_metric, result.metric, result.strategy)
        return None

    def _attach_report_metrics(self, result: StrategyResult, task: Task, report_spec: CompTaskSpec) -> StrategyResult:
        """Evaluate the selected support-search payload once on held-out query data."""

        if result.report_attempted or not result.has_report_candidate:
            return result
        if result.report_candidate_routed is not None:
            reporter = dict(self.strategies).get("routed")
            evaluate_report = getattr(reporter, "evaluate_report", None)
            if not callable(evaluate_report):
                return result
            result.report_attempted = True
            try:
                result.report_metrics = dict(evaluate_report(result.report_candidate_routed, task, report_spec, self.library))
            except TimeoutError:
                result.report_metrics = {}
            return result
        composition_candidate = result.champion_comp or result.report_candidate_comp
        if composition_candidate is not None:
            result.report_attempted = True
            previous_deadline = getattr(self.loop.evolver, "deadline", None)
            previous_callback = getattr(self.loop.evolver, "deadline_exceeded", None)
            self.loop.evolver.deadline = None
            self.loop.evolver.deadline_exceeded = None
            try:
                try:
                    reported = self.loop.assess_composition(composition_candidate.comp, report_spec, self.state, train=False)
                except TimeoutError:
                    return result
                if reported.net is not None:
                    result.report_metrics = dict(reported.metrics)
                return result
            finally:
                self.loop.evolver.deadline = previous_deadline
                self.loop.evolver.deadline_exceeded = previous_callback

        candidate = result.champion_genome or result.report_candidate_genome
        assert candidate is not None
        strategies = dict(self.strategies)
        if result.field_template is not None:
            reporter = strategies.get("field")
            args = (candidate, task, result.field_template)
        else:
            reporter = strategies.get("direct")
            if reporter is None and result.strategy == "grammar":
                reporter = getattr(strategies.get("grammar"), "direct", None)
            args = (candidate, task)

        evaluate_report = getattr(reporter, "evaluate_report", None)
        if not callable(evaluate_report):
            return result
        result.report_attempted = True
        evolver = getattr(reporter, "evolver", None)
        previous_deadline = getattr(evolver, "deadline", None) if evolver is not None else None
        previous_callback = getattr(evolver, "deadline_exceeded", None) if evolver is not None else None
        if evolver is not None:
            evolver.deadline = None
            evolver.deadline_exceeded = None
        try:
            try:
                result.report_metrics = dict(evaluate_report(*args))
            except TimeoutError:
                result.report_metrics = {}
            return result
        finally:
            if evolver is not None:
                evolver.deadline = previous_deadline
                evolver.deadline_exceeded = previous_callback

    def _cross_validate_result(self, result: StrategyResult, task: Task) -> StrategyResult:
        """Attach support-fold evidence to a provisional executable solution exactly once."""

        if not self.cross_validation_config.enabled or result.validation_status != "not_run":
            return result
        if not result.has_admissible_champion or not self._accepts_metrics(result.champion_metrics):
            return result
        now = time.perf_counter()
        task_deadline = getattr(self, "_task_deadline", None)
        bounded_deadline = now + self.cross_validation_config.reserve_max_seconds
        deadline = min(task_deadline, bounded_deadline) if task_deadline is not None else bounded_deadline
        self.display.stage_started("cross_validation")
        validation = self.cross_validator.run(task, self._fold_evaluator(result), deadline=deadline)
        result.validation_status = validation.status
        result.validation_metrics = validation.metrics()
        stages = getattr(self, "_active_stages", None)
        if stages is not None:
            stages["cross_validation"] = round(stages.get("cross_validation", 0.0) + validation.seconds, 3)
        if validation.folds_completed:
            detail = f"{validation.folds_passed}/{validation.folds_completed} support folds passed"
        elif validation.status == "not_applicable":
            detail = "fewer than two usable support folds"
        else:
            detail = "no fold completed before validation became unavailable"
        outcome = "accepted" if validation.admits else ("unavailable" if validation.status == "inconclusive" else "continue")
        self.display.stage_result("cross_validation", outcome, f"{validation.status} · {detail}", seconds=validation.seconds, depth=getattr(self, "_display_depth", 0))
        return result

    def _fold_evaluator(self, result: StrategyResult) -> Callable[[Task, int, float | None], dict[str, float]]:
        if result.champion_comp is not None:
            comp = result.champion_comp.comp
            return lambda fold, seed, deadline: self._evaluate_composition_fold(comp, fold, seed, deadline)
        if result.champion_genome is None:
            return lambda _fold, _seed, _deadline: {}
        genome = result.champion_genome
        if result.field_template is not None:
            field_template = result.field_template
            return lambda fold, seed, deadline: self._evaluate_field_fold(genome, field_template, fold, seed, deadline)
        return lambda fold, seed, deadline: self._evaluate_genome_fold(genome, fold, seed, deadline)

    def _evaluate_genome_fold(self, genome: Genome, task: Task, seed: int, deadline: float | None) -> dict[str, float]:
        strategy: Any = dict(self.strategies).get("direct")
        if strategy is None:
            grammar = dict(self.strategies).get("grammar")
            strategy = getattr(grammar, "direct", None)
        if strategy is None:
            return {}
        evolver = strategy.evolver
        fresh = fresh_genome_weights(genome, seed)
        adapter = strategy._adapter(task, include_query=True)
        state = EvolverState([], InnovationTracker.from_genomes([fresh]), random.Random(seed))
        previous_deadline, previous_callback = evolver.deadline, evolver.deadline_exceeded
        evolver.deadline = deadline
        evolver.deadline_exceeded = lambda: deadline is not None and time.perf_counter() >= deadline
        try:
            assessed = evolver.assess(fresh, adapter, state)
            return {} if assessed.module is None else dict(assessed.metrics)
        finally:
            evolver.deadline, evolver.deadline_exceeded = previous_deadline, previous_callback

    def _evaluate_field_fold(self, genome: Genome, field_template: dict[str, Any], task: Task, seed: int, deadline: float | None) -> dict[str, float]:
        strategy: Any = dict(self.strategies).get("field")
        if strategy is None:
            return {}
        from versal.field import FieldAdapter, FieldContract, deterministic_sites, encode_sites, evaluate_field_module, valid_sites

        contract = FieldContract.from_dict(field_template)
        sites = valid_sites(task.support)
        selected = deterministic_sites(sites, strategy.train_sites, salt=f"cv:{seed}:{contract.identity}")
        encoded = encode_sites(task, selected, contract, chunk_size=strategy.verify_chunk_size, deadline=deadline)
        adapter = FieldAdapter(encoded, encoded, contract, max_inline_depth=strategy.evolver.max_inline_depth, library=self.library)
        fresh = fresh_genome_weights(genome, seed)
        state = EvolverState([], InnovationTracker.from_genomes([fresh]), random.Random(seed))
        evolver = strategy.evolver
        previous_deadline, previous_callback = evolver.deadline, evolver.deadline_exceeded
        evolver.deadline = deadline
        evolver.deadline_exceeded = lambda: deadline is not None and time.perf_counter() >= deadline
        try:
            assessed = evolver.assess(fresh, adapter, state)
            if assessed.module is None:
                return {}
            return evaluate_field_module(assessed.module, task, contract, split="query", chunk_size=strategy.verify_chunk_size, deadline=deadline)
        finally:
            evolver.deadline, evolver.deadline_exceeded = previous_deadline, previous_callback

    def _evaluate_composition_fold(self, comp: CompositionGenome, task: Task, seed: int, deadline: float | None) -> dict[str, float]:
        spec = comp_task_spec(task, structured_grid=self.structured_grid)
        fresh = fresh_composition_glue(comp, seed, dense_scale=self.loop.glue_scale)
        champions = {species_id: fresh_genome_weights(genome, seed ^ species_id) for species_id, genome in self.state.species_champions.items()}
        assessed = assess_composition_pure(
            fresh,
            spec,
            champions,
            self.library,
            self.loop.max_inline_depth,
            train=True,
            train_op=self.loop.evolver.train_op,
            evaluate_op=self.loop.evolver.evaluate_op,
            fitness=self.loop.evolver.fitness,
            rng=random.Random(seed),
            deadline=deadline,
        )
        return dict(assessed.metrics)

    # --- the evolve step: a config-ordered strategy ladder with budget carry --------------------------

    def _shutdown_requested(self) -> bool:
        callback = self._shutdown_requested_callback
        return bool(callback is not None and callback())

    def _time_deadline_exceeded(self) -> bool:
        """Whether the support-search cutoff has expired (legacy/public timeout meaning)."""

        deadline = getattr(self, "_solve_deadline", None)
        return deadline is not None and time.perf_counter() > deadline

    def _search_deadline_exceeded(self) -> bool:
        return self._time_deadline_exceeded()

    def _task_allowance_exceeded(self) -> bool:
        deadline = getattr(self, "_task_deadline", None)
        return deadline is not None and time.perf_counter() > deadline

    def _support_timed_out(self, result: StrategyResult | None = None) -> bool:
        """Allow a completed reserved-time validation to cross the earlier search cutoff."""

        if self._task_allowance_exceeded():
            return True
        if not self._time_deadline_exceeded():
            return False
        return result is None or result.validation_status not in {"passed", "not_applicable"}

    def _deadline_exceeded(self) -> bool:
        return self._shutdown_requested() or self._search_deadline_exceeded()

    def _total_deadline_exceeded(self) -> bool:
        deadline = getattr(self, "_total_task_deadline", None)
        return self._shutdown_requested() or (deadline is not None and time.perf_counter() > deadline)

    def _record_shutdown(self, task: Task, depth: int) -> None:
        """Record a cooperative stop without misclassifying it as a configured deadline."""

        self.counters["failures"] += 1
        if depth == 0:
            self._failure_stage = "shutdown_requested"
        recovered = getattr(self, "_best_parent_result", None) if depth == 0 else None
        if recovered is not None:
            report_result = self._report_result_for(recovered, depth)
            attempt = self._attempt_from_result(
                recovered,
                task=task.meta.name,
                depth=depth,
                outcome="failed",
                failure_stage="shutdown_requested" if depth == 0 else None,
                query_status="shutdown_before_evaluation",
                report_result=report_result,
                suppress_query=True,
            )
        else:
            attempt = Attempt(
                task=task.meta.name,
                depth=depth,
                outcome="failed",
                metric=0.0,
                generations=0,
                strategy="shutdown",
                failure_stage="shutdown_requested" if depth == 0 else None,
                support_status="not_reached",
                query_status="shutdown_before_evaluation",
            )
        self._record(attempt)
        logger.debug("orchestrator stopped %s at depth %d after an Escape request", task.meta.name, depth)
        return None

    def _record_total_timeout(self, task: Task, depth: int) -> None:
        """Record an expired cumulative budget without starting lookup, probing, or evolution."""

        self.counters["failures"] += 1
        self.counters["time_budget_hits"] += 1
        if depth == 0:
            self.counters["total_time_budget_hits"] += 1
            self._failure_stage = "time_budget"
        recovered = getattr(self, "_best_parent_result", None) if depth == 0 else None
        if recovered is None and depth == 0:
            recovered = getattr(self, "_best_parent_report_result", None)
        if recovered is not None:
            report_result = self._report_result_for(recovered, depth)
            if self.blind_query and not self._shutdown_requested() and not report_result.report_attempted:
                report_result = self._attach_report_metrics(report_result, task, comp_task_spec(task, structured_grid=self.structured_grid))
            attempt = self._attempt_from_result(
                recovered,
                task=task.meta.name,
                depth=depth,
                outcome="failed",
                failure_stage="time_budget",
                query_status="time_limit_before_evaluation",
                report_result=report_result,
            )
        else:
            attempt = Attempt(
                task=task.meta.name,
                depth=depth,
                outcome="failed",
                metric=0.0,
                generations=0,
                strategy="time_budget",
                failure_stage="time_budget" if depth == 0 else None,
                support_status="not_reached",
                query_status="time_limit_before_evaluation",
            )
        self._record(attempt)
        logger.debug("orchestrator skipped %s at depth %d: cumulative task budget exhausted", task.meta.name, depth)
        return None

    def _wants_decompose_first(self, task: Task, spec: CompTaskSpec) -> bool:
        """Decompose only when every configured native representation refuses pre-allocation."""
        if self.decompose_first_above != "adaptive" and int(self.decompose_first_above) <= 0:
            return False
        if self.decompose_first_above != "adaptive":
            runtime = self._runtime()
            for name, strategy in self.strategies:
                if name != "field":
                    continue
                preflight = getattr(strategy, "preflight", None)
                if preflight is not None and preflight(task, runtime).eligible:
                    return False
            io = self._io_of(task, spec)
            init_genes = (int(io["inputs"][0]["width"]) + 1) * int(io["output"]["width"])
            return init_genes > int(self.decompose_first_above)
        consulted = False
        runtime = self._runtime()
        for _name, strategy in self.strategies:
            preflight = getattr(strategy, "preflight", None)
            if preflight is None:
                continue
            consulted = True
            decision = preflight(task, runtime)
            if decision.eligible:
                return False
        if consulted:
            return True
        # Compatibility for ladders containing only strategies without a native preflight.
        io = self._io_of(task, spec)
        init_genes = (int(io["inputs"][0]["width"]) + 1) * int(io["output"]["width"])
        estimate = self.loop.assess_glue_resources(init_genes, stage="decompose_first", storage="tuple", fixed_limit=0)
        return not estimate.accepted

    def _with_deadline(self, detector: StallDetector) -> Callable[[int, Any], bool]:
        """Chain the per-solve deadline behind a stall detector. With no deadline the detector is
        returned AS-IS (identical object flow, the byte-identical off path); with one, the detector
        still runs FIRST so its flatline state advances exactly as it would unbudgeted."""
        if getattr(self, "_solve_deadline", None) is None and self._shutdown_requested_callback is None:
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
            shutdown_requested=self._shutdown_requested,
            deadline=getattr(self, "_solve_deadline", None),
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
        handoff_seeds = list(seed_comps or [])
        handoff_fingerprints = {structural_fingerprint(COMPOSITION, comp_to_dict(comp)) for comp in handoff_seeds}
        ladder_metrics: dict[str, float] = {}
        resource_metrics: dict[str, float] = {}
        remaining = budget
        carry = 0
        if self._shutdown_requested():
            return StrategyResult(strategy="shutdown", metric=0.0, generations_used=0)
        if self._total_deadline_exceeded():
            return StrategyResult(strategy="time_budget", metric=0.0, generations_used=0)
        for position, (name, strategy) in enumerate(self.strategies):
            if self._shutdown_requested():
                if not results:
                    return StrategyResult(strategy="shutdown", metric=0.0, generations_used=0)
                break
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
            self.display.stage_started(name)
            if name == "field":
                from versal.field import field_contract

                contract = field_contract(task)
                field_seeds = self.library.query_field(contract, limit=self.quick_eval_top_k) if contract is not None else []
                outcome = strategy(task, spec, runtime, budget=allocation, seed_entries=field_seeds or None)
            elif name == "direct" and seed_entries:
                outcome = strategy(task, spec, runtime, budget=allocation, seed_entries=seed_entries)
            else:
                outcome = strategy(task, spec, runtime, budget=allocation, seed_comps=handoff_seeds if name == "composition" else None)
            stage_elapsed = time.perf_counter() - stage_started
            outcome = self._cross_validate_result(outcome, task)
            stages = getattr(self, "_active_stages", None)
            if stages is not None:
                stages[name] = round(stages.get(name, 0.0) + stage_elapsed, 3)
            if name == "routed":  # the strategy has no counter access; it stamps markers instead
                for marker in ("routed_no_experts", "routed_undistillable", "routed_resource_declined"):
                    if outcome.champion_metrics.get(marker):
                        self.counters[marker] += 1
                for metric, counter in (
                    ("router_vertices_expired", "router_vertices_expired"),
                    ("router_vertices_revived", "router_vertices_revived"),
                    ("router_edges_expired", "router_edges_expired"),
                ):
                    if counter in self.counters:
                        self.counters[counter] += int(outcome.strategy_metrics.get(metric, 0.0))
            resource_metrics.update(outcome.resource_metrics)
            if outcome.strategy_metrics.get("direct_guard_declined") and "direct_guard_declines" in self.counters:
                self.counters["direct_guard_declines"] += 1
            resource_declined = any(key.endswith("_declined") and value > 0.0 for key, value in outcome.resource_metrics.items())
            skipped = outcome.skip_reason is not None or resource_declined
            if resource_declined and "resource_declines" in self.counters:
                self.counters["resource_declines"] += 1
                counter = f"{name}_resource_declines"
                if counter in self.counters:
                    self.counters[counter] += 1
            results.append(outcome)
            self._consider_parent_report_result(outcome, depth=getattr(self, "_display_depth", 0))
            self._consider_parent_field_result(outcome, depth=getattr(self, "_display_depth", 0))
            ladder_metrics.update(outcome.strategy_metrics)
            if name == "routed" and outcome.champion_comp is not None and not self._accepts_result(outcome):
                fingerprint = structural_fingerprint(COMPOSITION, comp_to_dict(outcome.champion_comp.comp))
                if fingerprint not in handoff_fingerprints:
                    handoff_fingerprints.add(fingerprint)
                    handoff_seeds.append(outcome.champion_comp.comp)
                    ladder_metrics["handoff_count"] = ladder_metrics.get("handoff_count", 0.0) + 1.0
            if name == "composition" and ladder_metrics.get("handoff_count", 0.0):
                ladder_metrics["recovery_result"] = float(outcome.metric)
            support, _query, support_status, _query_status = self._quality_of_result(outcome)
            notes = [outcome.skip_reason] if outcome.skip_reason is not None else [f"{outcome.generations_used} generations"]
            if support_status == "no_executable_champion" and not skipped:
                notes.append("diagnostic only; no executable champion")
            if name == "routed" and "router_score" in outcome.strategy_metrics:
                notes.append(f"router {outcome.strategy_metrics['router_score']:.3f}")
                if "distilled_score" in outcome.strategy_metrics:
                    notes.append(f"distilled {outcome.strategy_metrics['distilled_score']:.3f}")
                if ladder_metrics.get("handoff_count", 0.0):
                    notes.append("executable handoff prepared")
            if resource_declined and outcome.skip_reason is None:
                notes.append("explicit flat substrate cannot fit; continuing with field search/composition" if name == "direct" else "resource guard skipped allocation")
            self.display.stage_result(
                name,
                "accepted" if self._accepts_result(outcome) else ("skipped" if skipped else "continue"),
                " · ".join(notes),
                seconds=stage_elapsed,
                depth=getattr(self, "_display_depth", 0),
                support_accuracy=support,
            )
            remaining -= outcome.generations_used
            carry = max(0, allocation - outcome.generations_used)
            if self._accepts_result(outcome):
                outcome.resource_metrics = dict(resource_metrics)
                outcome.strategy_metrics = dict(ladder_metrics)
                return outcome
            if remaining <= 0:
                break
        # Metric-only diagnostics (for example an adapter-space routed score whose pathway could
        # not be distilled) must not displace a real support-selected payload that can receive the
        # mandatory held-out report. When every strategy declined, the metric remains useful.
        if self.blind_query:
            best = max(results, key=lambda item: (item.has_report_candidate, self._report_candidate_value(item)))
        else:
            best = max(results, key=lambda item: item.metric)
        best.resource_metrics = dict(resource_metrics)
        best.strategy_metrics = dict(ladder_metrics)
        return best

    # --- learn-mode refinement of library hits --------------------------------------------------------

    def _handle_library_hit(self, hit: Solution, task: Task, spec: CompTaskSpec, depth: int) -> Solution:
        """STEP 1b: a hit is free, but learn mode (refine budget_k > 0) spends a bounded, decaying
        budget trying to beat the stored solution before settling for it. The guard runs FIRST with
        zero side effects, so budget_k = 0 (live mode) is byte-identical to the plain hit path. The
        task can never regress: a failed refinement returns the original hit."""
        refine_generations = 0
        self._last_refine_strategy_metrics = {}
        if self.refine_budget_k > 0 and depth <= self.refine_depth_max and hit.key is not None and not self._total_deadline_exceeded():
            improved, refine_generations = self._refine_hit(hit, task, spec, depth)
            if improved is not None:
                return improved  # _refine_hit already recorded the "refined" attempt
        self.display.query_result(hit.query_accuracy, hit.query_status, depth=depth)
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
                support_accuracy=hit.support_accuracy,
                query_accuracy=hit.query_accuracy,
                support_status=hit.support_status,
                query_status=hit.query_status,
                strategy_metrics=dict(self._last_refine_strategy_metrics),
                representation=("field" if hit.key is not None and "field_template" in self.library.load(hit.key).payload else hit.entry_type),
            )
        )
        return hit

    def _refine_hit(self, hit: Solution, task: Task, spec: CompTaskSpec, depth: int) -> tuple[Solution | None, int]:
        """Seed a bounded evolve from the stored solution and admit only a strict improvement:
        capability beyond the accuracy band, otherwise expanded complexity then robustness. Runs ONLY the strategy
        matching the entry shape, so K stays focused and `_evolve`'s share arithmetic is untouched."""
        assert hit.key is not None
        self.display.stage_started("refine")
        refine_started = time.perf_counter()
        entry = self.library.load(hit.key)
        strategy_name = "field" if entry.entry_type == MODULE and "field_template" in entry.payload else ("direct" if entry.entry_type == MODULE else "composition")
        strategy = dict(self.strategies).get(strategy_name)
        if strategy is None:
            self.counters["refine_skipped_no_strategy"] += 1
            self.display.stage_result("refine", "skipped", "no compatible refinement strategy", seconds=time.perf_counter() - refine_started, depth=depth)
            return None, 0
        effective_budget = self._effective_refine_budget(entry)
        if effective_budget < self.refine_min_generations:
            self.counters["refine_skipped_decayed"] += 1
            self.display.stage_result("refine", "skipped", "the lineage has exhausted its useful refinement allowance", seconds=time.perf_counter() - refine_started, depth=depth)
            return None, 0
        self.counters["refine_attempts"] += 1
        # The target is deliberately NOT clamped to 1.0: an incumbent at 1.0 makes it unreachable,
        # so the strategy runs to its (refine-local) stall and the tie-breaks decide afterward;
        # a beatable incumbent lets the strategy's early exit stop the moment it wins.
        topology_tabu = self._topology_tabu(entry, task) if self.refine_deduplicate_topologies else None
        runtime = self._refine_runtime(hit.metric + self.refine_metric_epsilon, topology_tabu=topology_tabu)
        target = getattr(strategy, "evolver", self.loop.evolver) if entry.entry_type == MODULE else self.loop
        previous_tabu = target.topology_tabu
        target.topology_tabu = topology_tabu
        try:
            if entry.entry_type == MODULE:
                result = strategy(task, spec, runtime, budget=effective_budget, seed_entries=[entry])
            else:
                result = strategy(task, spec, runtime, budget=effective_budget, seed_comps=[comp_from_dict(entry.payload)])
        finally:
            target.topology_tabu = previous_tabu
        result = self._cross_validate_result(result, task)
        support_timed_out = self._support_timed_out(result)
        stopping = self._shutdown_requested()
        if topology_tabu is not None:
            topology_tabu.commit()
            topology_metrics = topology_tabu.metrics()
            result.strategy_metrics.update(topology_metrics)
            self.counters["topology_unique_evaluated"] += int(topology_metrics["topology_unique_evaluated"])
            self.counters["topology_duplicates_skipped"] += int(topology_metrics["topology_duplicates_skipped"])
            self.counters["topology_retry_exhaustions"] += int(topology_metrics["topology_retry_exhaustions"])
            self.counters["topology_exhausted_attempts"] += int(topology_metrics["topology_exhausted"])
        self._last_refine_strategy_metrics = dict(result.strategy_metrics)
        if self.blind_query and not stopping:
            result = self._attach_report_metrics(result, task, comp_task_spec(task, structured_grid=self.structured_grid))
        stopping = stopping or self._shutdown_requested()
        self.counters["refine_generations"] += result.generations_used

        candidate = self._candidate_rank(result)
        incumbent = self._incumbent_rank(hit, entry, seed_metric=result.seed_metric)
        improves = (
            candidate is not None
            and not support_timed_out
            and not stopping
            # A robustness/size tie-break can sit epsilon below an at-the-bar incumbent; never shelve below the bar.
            and self._accepts_result(result)
            # Identity check FIRST: the incumbent topology retrained on this variant is NOT a new
            # solution (entry keys hash weights, so a key comparison can never catch this).
            and not self._candidate_matches_entry(result, entry)
            and refinement_improves(candidate, incumbent, metric_epsilon=self.refine_metric_epsilon, robustness_epsilon=self.refine_robustness_epsilon)
        )
        if not improves:
            self.library.record_refinement(hit.key, improved=False)
            self.counters["refine_no_gain"] += 1
            support, _query, _support_status, _query_status = self._quality_of_result(result)
            detail = "retrieved solution remains stronger"
            if topology_tabu is not None:
                detail += f" · tested {topology_tabu.unique} new architectures · skipped {topology_tabu.duplicates} repeats"
                if topology_tabu.exhausted:
                    detail += " · architecture search exhausted"
            self.display.stage_result("refine", "continue", detail, seconds=time.perf_counter() - refine_started, depth=depth, support_accuracy=support)
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
        self._record(self._attempt_from_result(result, task=task.meta.name, depth=depth, outcome="refined", library_key=key, refine_generations=result.generations_used))
        support, query, _support_status, query_status = self._quality_of_result(result)
        self.display.stage_result(
            "refine", "accepted", "strict non-regressing improvement selected", seconds=time.perf_counter() - refine_started, depth=depth, support_accuracy=support
        )
        self.display.query_result(query, query_status, depth=depth)
        return self._solution_from_result(result, key), result.generations_used

    def _effective_refine_budget(self, entry: LibraryEntry) -> int:
        if self.refine_mode == "always":
            return self.refine_budget_k
        summary = self.library.summary(entry.key) or {}
        failures = int((summary.get("stats") or {}).get("refine_failures_since_gain", 0))
        return int(self.refine_budget_k * (self.refine_decay**failures))

    def _refine_runtime(self, target_metric: float, *, topology_tabu: TopologyTabuSession | None = None) -> StrategyRuntime:
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
            shutdown_requested=self._shutdown_requested,
            topology_tabu=topology_tabu,
            deadline=getattr(self, "_solve_deadline", None),
        )

    def _topology_tabu(self, entry: LibraryEntry, task: Task) -> TopologyTabuSession:
        root = refinement_lineage_root(self.library, entry.key)
        context = refinement_context(lineage_root=root, task_fingerprint=task_content_fingerprint(task), config_fingerprint=self.config_fingerprint)
        session = TopologyTabuSession(
            TopologyTabuStore(self.library.root / "topology_tabu.sqlite3"),
            context,
            self.library,
            retry_limit=self.refine_topology_retry_limit,
            deadline_exceeded=self._deadline_exceeded,
        )
        current = entry
        visited: set[str] = set()
        while current.key not in visited:
            visited.add(current.key)
            session.prime(current.entry_type, current.payload)
            parent = current.provenance.get("refined_from")
            if not parent:
                break
            try:
                current = self.library.load(str(parent))
            except KeyError:
                break
        return session

    def _candidate_rank(self, result: StrategyResult) -> RefinementRank | None:
        robustness = float(result.champion_metrics.get("weight_robustness", 0.0))
        if result.champion_genome is not None:
            complexity = expanded_payload_complexity(MODULE, genome_to_dict(result.champion_genome), self.library)
            return RefinementRank(metric=result.metric, robustness=robustness, complexity=complexity, entry_type=MODULE)
        if result.champion_comp is not None:
            complexity = self._expanded_composition_complexity(result.champion_comp.comp)
            return RefinementRank(metric=result.metric, robustness=robustness, complexity=complexity, entry_type=COMPOSITION)
        return None

    def _expanded_composition_complexity(self, comp: CompositionGenome) -> int:
        payload = comp_to_dict(comp)
        total = expanded_payload_complexity(COMPOSITION, payload, self.library)
        for node in comp.nodes.values():
            if node.kind is not CompNodeKind.MODULE or not node.ref.startswith("live:"):
                continue
            genome = self.state.species_champions.get(int(node.ref.removeprefix("live:")))
            if genome is not None:
                total += expanded_payload_complexity(MODULE, genome_to_dict(genome), self.library)
        return total

    @staticmethod
    def _candidate_fingerprint(result: StrategyResult) -> str | None:
        if result.champion_genome is not None:
            payload = genome_to_dict(result.champion_genome)
            if result.field_template is not None:
                payload["field_template"] = result.field_template
            return structural_fingerprint(MODULE, payload)
        if result.champion_comp is not None:
            return structural_fingerprint(COMPOSITION, comp_to_dict(result.champion_comp.comp))
        return None

    def _candidate_matches_entry(self, result: StrategyResult, entry: LibraryEntry) -> bool:
        """Exact architecture identity for refinement; hashes are only the fast bucket."""

        if result.champion_genome is not None and entry.entry_type == MODULE:
            candidate_type, candidate_payload = MODULE, genome_to_dict(result.champion_genome)
            if result.field_template is not None:
                candidate_payload["field_template"] = result.field_template
        elif result.champion_comp is not None and entry.entry_type == COMPOSITION:
            candidate_type, candidate_payload = COMPOSITION, comp_to_dict(result.champion_comp.comp)
        else:
            return False
        return same_topology(
            topology_record(candidate_type, candidate_payload, library=self.library),
            topology_record(entry.entry_type, entry.payload, library=self.library),
        )

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
        complexity = self.library.expanded_complexity(entry.key)
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
        strictly_simpler = candidate.complexity < incumbent.complexity
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
        # Field programs have a symbolic, cross-resolution seed rail of their own. Filter them
        # before applying top-k so a strong field stone cannot crowd the ordinary direct/comp rail.
        stones = [stone for stone in self._wall_stones(task, spec) if "field_template" not in stone.payload][: self.wall_seed_top_k]
        modules = [stone for stone in stones if stone.entry_type == MODULE]
        comps = [comp_from_dict(stone.payload) for stone in stones if stone.entry_type == COMPOSITION]
        return modules, comps

    def _wall_stones_for_result(self, result: StrategyResult, task: Task, spec: CompTaskSpec) -> list[LibraryEntry]:
        """Return only stones competing for the candidate's representation-specific niche.

        Field programs compete by symbolic field identity, independent of absolute grid size.
        Ordinary modules and compositions each keep their own task-shaped lineage. This preserves
        bounded replacement semantics without letting a minimal composition monopolize every
        representation that could make progress on the same raw I/O contract.
        """

        if result.field_template is not None:
            from versal.field import FieldContract

            try:
                contract = FieldContract.from_dict(result.field_template)
            except ValueError:
                return []
            return [entry for entry in self.library.query_field(contract) if entry.provenance.get("stepping_stone")]
        entry_type = COMPOSITION if result.champion_comp is not None else MODULE
        return [entry for entry in self._wall_stones(task, spec) if entry.entry_type == entry_type and "field_template" not in entry.payload]

    def _admit_stepping_stone(self, result: StrategyResult, task: Task, spec: CompTaskSpec) -> str | None:
        """Shelve a failed attempt's best champion as a below-bar stepping stone: a dependency
        entry (bypasses the admission policy and signature caps, invisible to `signature_group`)
        that can never be a false lookup hit because quick-eval still gates on the accept bar.
        One stone per representation lineage (and per symbolic field identity): replaced only on
        a strict lexicographic AND structural win (the refine comparator, reused), so the wall gets
        chipped, not wallpapered. Free
        synergies: stones enter module-pool absorption and the comp ref catalog through `query`,
        and become router vertices at sync (immature circuits in the overmind, by design)."""
        candidate = self._candidate_rank(result)
        if candidate is None or not math.isfinite(result.metric) or result.metric < self.wall_min_metric:
            return None
        if float(result.champion_metrics.get("support_gain_over_baseline", math.inf)) < self.wall_min_gain_over_baseline:
            return None
        incumbent_stone = next(iter(self._wall_stones_for_result(result, task, spec)), None)
        if incumbent_stone is not None:
            if self._candidate_matches_entry(result, incumbent_stone):
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
        logger.debug("wall ledger shelved stepping stone %s for %s (best %s=%.3f)", key, task.meta.name, self.search_metric, result.metric)
        return key

    def _admit_failure_stepping_stones(self, result: StrategyResult, task: Task, spec: CompTaskSpec) -> list[str]:
        """Retain the overall loser and the best independently observed field program.

        The ladder returns only one overall result, so without this second rail every losing field
        population disappears whenever a routed/direct/composition candidate ranks higher. Exact
        fingerprints are de-duplicated before admission; niche replacement performs the remaining
        archive bounds.
        """

        candidates = [result]
        field_result = getattr(self, "_best_parent_field_result", None)
        if field_result is not None:
            candidates.append(field_result)
        admitted: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            fingerprint = self._candidate_fingerprint(candidate)
            if fingerprint is None or fingerprint in seen:
                continue
            seen.add(fingerprint)
            key = self._admit_stepping_stone(candidate, task, spec)
            if key is not None and key not in admitted:
                admitted.append(key)
        return admitted

    # --- ladder steps -------------------------------------------------------------------------------

    def _lookup(self, task: Task, spec: CompTaskSpec) -> Solution | None:
        io = self._io_of(task, spec)
        from versal.field import field_contract

        contract = field_contract(task)
        candidates = self.library.query_field(contract, limit=self.quick_eval_top_k) if contract is not None else []
        exact = self.library.query(
            input_signature=io["inputs"][0]["signature"],
            input_width=io["inputs"][0]["width"],
            output_width=io["output"]["width"],
            limit=self.quick_eval_top_k,
        )
        seen = {entry.key for entry in candidates}
        candidates.extend(entry for entry in exact if entry.key not in seen)
        candidates = candidates[: self.quick_eval_top_k]
        for entry in candidates:
            # Stones are reusable search material, not verified task solutions.  In particular a
            # support-perfect stone that failed CV must never turn into a zero-cost lookup hit.
            if entry.provenance.get("stepping_stone"):
                continue
            if self._total_deadline_exceeded():
                return None
            assessment = self._quick_assessment(entry, task, spec)
            if assessment is None or not self._accepts_item(assessment):
                continue
            self.library.note_reuse(entry.key, channel="lookup")
            metric = self._metric(assessment)
            shutdown_before_report = self.blind_query and self._shutdown_requested()
            report_assessment = (
                None
                if shutdown_before_report
                else (self._quick_assessment(entry, task, comp_task_spec(task, structured_grid=self.structured_grid)) if self.blind_query else assessment)
            )
            report_metric = self._report(report_assessment) if self.blind_query and report_assessment is not None else (None if self.blind_query else metric)
            combined = dict(assessment.metrics) | (dict(report_assessment.metrics) if report_assessment is not None else {})
            task_metrics = _task_metrics(combined)
            if "field_template" in entry.payload:
                task_metrics["cross_resolution_reuse"] = float(entry.io != io)
                task_metrics["representation_field"] = 1.0
            support_accuracy = _finite_accuracy(assessment.metrics, "support_accuracy")
            query_accuracy = _finite_accuracy(report_assessment.metrics, "query_accuracy", loss_key="query_loss") if report_assessment is not None else None
            return Solution(
                key=entry.key,
                entry_type=entry.entry_type,
                metric=metric,
                report_metric=report_metric,
                task_metrics=task_metrics,
                support_accuracy=support_accuracy,
                query_accuracy=query_accuracy,
                support_status="evaluated" if support_accuracy is not None else "evaluation_unavailable",
                query_status="shutdown_before_evaluation" if shutdown_before_report else ("evaluated" if query_accuracy is not None else "evaluation_unavailable"),
            )
        return None

    def finish_root_task(self, attempt: Attempt | None) -> list[str]:
        """Advance lifecycle state after a durable root task and annotate its operational record."""

        retired = self.library.finish_root_task()
        if retired and "library_inactivity_retired" in self.counters:
            self.counters["library_inactivity_retired"] += len(retired)
        if attempt is not None and retired:
            attempt.strategy_metrics["library_inactivity_retired"] = float(len(retired))
        if retired:
            # Retirement happens after the routed stage has saved. Refresh its in-memory mask and
            # paired portraits now, so a run that stops at this boundary still shows current state.
            for name, strategy in self.strategies:
                service = getattr(strategy, "service", None)
                if name == "routed" and service is not None:
                    service.sync(
                        include_compositions=bool(getattr(strategy, "include_compositions", True)),
                        exclude_temporal=bool(getattr(strategy, "exclude_temporal", True)),
                    )
        return retired

    @staticmethod
    def _entry_is_temporal(entry: LibraryEntry) -> bool:
        signature = entry.io["inputs"][0].get("signature", "")
        return "|" in signature and "T" in signature.split("|", 1)[1].split(",")

    def _quick_assessment(self, entry: LibraryEntry, task: Task, spec: CompTaskSpec) -> AssessedComposition | None:
        """Evaluate a stored entry against the task with NO training (forward passes only).

        TIME-bearing MODULE entries (direct-strategy temporal winners) are scored through the
        stepped substrate; everything else uses the flat decode."""
        from versal.evaluation import evaluate
        from versal.evolution.composition import AssemblyContext, CompositionAssemblyError, assemble

        try:
            if entry.entry_type == MODULE and "field_template" in entry.payload:
                from versal.field import decode_field_payload, evaluate_field_module, field_contract

                wanted = field_contract(task)
                module, contract = decode_field_payload(entry.payload, library=self.library, max_inline_depth=self.loop.max_inline_depth)
                if wanted is None or wanted != contract:
                    return None
                metrics = evaluate_field_module(module, task, contract, split="support", deadline=getattr(self, "_solve_deadline", None))
                metrics.update({"query_accuracy": 0.0, "query_loss": float("inf"), "cross_resolution_reuse": 1.0})
                if spec.encoded.query_input is not None and task.query:
                    metrics.update(evaluate_field_module(module, task, contract, split="query", deadline=getattr(self, "_solve_deadline", None)))
                return AssessedComposition(comp=CompositionGenome(), metrics=metrics, fitness=0.0, net=None)
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
        except (ValueError, CompositionAssemblyError, TimeoutError) as error:
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
        self._last_decompose_report_overrun = False
        if self._total_deadline_exceeded():
            return None
        self.display.stage_started("decompose")
        decompose_started = time.perf_counter()
        chosen_name: str | None = None
        subtasks: list[Subtask] = []
        for op_name, op in self.decomposers:
            if self._total_deadline_exceeded():
                return None
            produced = op(task, rng=self.state.rng)
            if len(produced) < 2:
                continue
            projected_leaves = getattr(self, "_decomposition_leaf_count", 1) + len(produced) - 1
            if projected_leaves > self.decompose_leaf_cap:
                self._failure_stage = "representation_limit"
                self._failure_op = op_name
                continue
            if not self._subtasks_promising(produced, depth):
                logger.debug("decompose op %s produced subtasks that fail the solvability probe; skipping", op_name)
                continue
            chosen_name, subtasks = op_name, produced
            self._decomposition_leaf_count = projected_leaves
            break
        if not subtasks:
            self.display.stage_result("decompose", "miss", "no registered split produced useful independent subtasks", seconds=time.perf_counter() - decompose_started, depth=depth)
            return None
        self.counters["decompositions"] += 1
        logger.debug("decomposing %s via %s into %d subtasks (depth %d)", task.meta.name, chosen_name, len(subtasks), depth)

        solutions: list[tuple[Subtask, Solution]] = []
        for subtask in subtasks:
            if self._total_deadline_exceeded():
                return None
            solved = self.solve(subtask.task, depth + 1)
            self._last_decompose_report_overrun = False
            if solved is None:
                # A missing part means the wired parent cannot be completed; record WHERE it died.
                self.counters["decompose_subtask_failed"] += 1
                self._failure_stage = f"subtask:{subtask.task.meta.name}"
                self._failure_op = chosen_name
                self.display.stage_result(
                    "decompose", "failed", f"required subtask {subtask.task.meta.name} did not solve", seconds=time.perf_counter() - decompose_started, depth=depth
                )
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
        result = self._remember_or_recover_parent_result(result, depth)
        support_timed_out = self._support_timed_out(result)
        report_result = self._report_result_for(result, depth)
        report_attempted_before = report_result.report_attempted
        if self.blind_query and not self._shutdown_requested():
            report_result = self._attach_report_metrics(report_result, task, comp_task_spec(task, structured_grid=self.structured_grid))
            report_started = not report_attempted_before and report_result.report_attempted
            self._last_decompose_report_overrun = report_started and not support_timed_out and self._time_deadline_exceeded()
            _support, query, _support_status, query_status = self._quality_of_result(report_result)
            if report_result.has_report_candidate:
                self.display.query_result(query, query_status, depth=depth)
        if not support_timed_out and not self._shutdown_requested() and self._accepts_result(result):
            key = self._admit_result(result, task, spec, depth, decompose_op=chosen_name)
            support, _query, _support_status, _query_status = self._quality_of_result(result)
            self.display.stage_result(
                "decompose",
                "accepted",
                f"{len(subtasks)} solved parts recomposed successfully",
                seconds=time.perf_counter() - decompose_started,
                depth=depth,
                support_accuracy=support,
            )
            self.display.stage_result(
                "persist",
                "saved" if key is not None else "skipped",
                "recomposed solution retained for reuse" if key is not None else "solution accepted but not retained by library policy",
                depth=depth,
            )
            self.counters["accepts"] += 1
            attempt = self._attempt_from_result(
                result,
                task=task.meta.name,
                depth=depth,
                outcome="decomposed",
                library_key=key,
                decompose_op=chosen_name,
                report_result=report_result,
            )
            self._record(attempt)
            return self._solution_from_result(result, key, report_result=report_result)
        # Every part solved but the wired parent still missed the bar.
        self.counters["decompose_parent_failed"] += 1
        self._failure_stage = "parent_re_evolve"
        self._failure_op = chosen_name
        support, _query, _support_status, _query_status = self._quality_of_result(result)
        self.display.stage_result(
            "decompose",
            "failed",
            "all parts solved, but the recomposed parent stayed below threshold",
            seconds=time.perf_counter() - decompose_started,
            depth=depth,
            support_accuracy=support,
        )
        return None

    def _remember_or_recover_parent_result(self, result: StrategyResult, depth: int) -> StrategyResult:
        """Keep the strongest executable top-level candidate across recursive fallbacks/timeouts."""

        if depth != 0:
            return result
        self._consider_parent_report_result(result, depth=depth)
        incumbent = getattr(self, "_best_parent_result", None)
        result_retainable = result.has_report_candidate if self.blind_query else result.has_admissible_champion
        if incumbent is None:
            if result_retainable:
                self._best_parent_result = result
            return result
        result_accepted = self._accepts_result(result)
        incumbent_accepted = self._accepts_result(incumbent)
        if result_accepted != incumbent_accepted:
            if result_accepted:
                self._best_parent_result = result
                return result
            return incumbent
        incumbent_retainable = incumbent.has_report_candidate if self.blind_query else incumbent.has_admissible_champion
        if result_retainable and (not incumbent_retainable or result.metric > incumbent.metric):
            self._best_parent_result = result
            return result
        if incumbent_retainable:
            return incumbent
        return result

    def _report_candidate_value(self, result: StrategyResult) -> float:
        """Rank the executable reporting rail by its own support-search score."""

        if not result.has_report_candidate:
            return -math.inf
        metrics = result.report_candidate_metrics or result.champion_metrics
        return self._metric(SimpleNamespace(metrics=metrics))

    def _consider_parent_report_result(self, result: StrategyResult, *, depth: int) -> None:
        """Remember the highest-support parent payload independently of admission ranking."""

        if depth != 0 or not result.has_report_candidate:
            return
        incumbent = getattr(self, "_best_parent_report_result", None)
        candidate_rank = (self._report_candidate_value(result), result.has_admissible_champion)
        incumbent_rank = (-math.inf, False) if incumbent is None else (self._report_candidate_value(incumbent), incumbent.has_admissible_champion)
        if incumbent is None or candidate_rank > incumbent_rank:
            self._best_parent_report_result = result

    def _consider_parent_field_result(self, result: StrategyResult, *, depth: int) -> None:
        """Remember the strongest verified field champion independently of ladder ranking."""

        if depth != 0 or result.field_template is None or result.champion_genome is None or not math.isfinite(result.metric):
            return
        incumbent = getattr(self, "_best_parent_field_result", None)
        candidate_rank = (
            result.metric,
            float(result.champion_metrics.get("weight_robustness", 0.0)),
            -result.champion_genome.complexity(),
        )
        incumbent_rank = (
            (-math.inf, -math.inf, -math.inf)
            if incumbent is None or incumbent.champion_genome is None
            else (
                incumbent.metric,
                float(incumbent.champion_metrics.get("weight_robustness", 0.0)),
                -incumbent.champion_genome.complexity(),
            )
        )
        if incumbent is None or candidate_rank > incumbent_rank:
            self._best_parent_field_result = result

    def _report_result_for(self, result: StrategyResult, depth: int) -> StrategyResult:
        if depth == 0:
            self._consider_parent_report_result(result, depth=depth)
            return getattr(self, "_best_parent_report_result", None) or result
        return result

    def _subtasks_promising(self, subtasks: list[Subtask], parent_depth: int = 0) -> bool:
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
            # An oversized child has not failed empirically; it simply needs another planning split.
            if self._wants_decompose_first(subtask.task, spec):
                continue
            probe = self._bounded_decomposition_probe(subtask.task, spec)
            self._consider_diagnostic_result(probe, task=subtask.task.meta.name, depth=parent_depth + 1)
            if probe.champion_metrics.get("support_accuracy", 0.0) < self.decompose_solvability_floor:
                return False
        return True

    def _bounded_decomposition_probe(self, task: Task, spec: CompTaskSpec) -> StrategyResult:
        """Support-only, two-candidate/two-step probe with a hard per-operator wall cap."""

        # White-box/embedding compatibility: callers may provide a purpose-built bounded probe by
        # shadowing `_evolve` on this instance.
        if "_evolve" in self.__dict__:
            return self.__dict__["_evolve"](task, spec, min(2, self.decompose_probe_generations))

        from functools import partial

        from versal.evolution.evaluate import standard

        probe_task = Task(meta=task.meta, support=list(task.support[:2048]), query=[])
        probe_spec = comp_task_spec(probe_task, include_query=False, structured_grid=self.structured_grid)

        runtime = self._runtime()
        probe_deadline = time.perf_counter() + self.decompose_probe_seconds
        runtime.deadline = min(value for value in (runtime.deadline, probe_deadline) if value is not None)
        bounded_deadline = runtime.deadline
        runtime.deadline_exceeded = lambda: bounded_deadline is not None and time.perf_counter() >= bounded_deadline
        for name in ("field", "direct"):
            strategy: Any = dict(self.strategies).get(name)
            if strategy is None:
                continue
            preflight = getattr(strategy, "preflight", None)
            if preflight is not None and not preflight(probe_task, runtime).eligible:
                continue
            evolver = getattr(strategy, "evolver", None)
            if evolver is None:
                continue
            saved = (evolver.pop_size, evolver.assess_workers, evolver.train_op, evolver.evaluate_op)
            field_saved = None
            if name == "field":
                field_saved = (strategy.train_sites, strategy.audit_sites, strategy.verify_top_k)
                strategy.train_sites, strategy.audit_sites, strategy.verify_top_k = 2048, 2048, 2
            try:
                evolver.pop_size = min(2, evolver.pop_size)
                evolver.assess_workers = 0
                evolver.train_op = partial(evolver.train_op, steps=2)
                evolver.evaluate_op = standard
                return strategy(probe_task, probe_spec, runtime, budget=1)
            finally:
                evolver.pop_size, evolver.assess_workers, evolver.train_op, evolver.evaluate_op = saved
                if field_saved is not None:
                    strategy.train_sites, strategy.audit_sites, strategy.verify_top_k = field_saved
        return StrategyResult("decomposition_probe", 0.0, 0, strategy_metrics={"probe_no_feasible_representation": 1.0})

    def _consider_diagnostic_result(self, result: StrategyResult, *, task: str, depth: int) -> None:
        support = _finite_accuracy(result.champion_metrics, "support_accuracy")
        score = support
        metric = "support_accuracy"
        if score is None and "router_score" in result.strategy_metrics:
            score = float(result.strategy_metrics["router_score"])
            metric = "router_score"
        if score is None:
            return
        observation = {
            "score": float(score),
            "metric": metric,
            "task": task,
            "depth": depth,
            "strategy": result.strategy,
            "executable": result.has_admissible_champion,
        }
        current = getattr(self, "_best_diagnostic_observation", None)
        if current is None or float(observation["score"]) > float(current["score"]):
            self._best_diagnostic_observation = observation

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
            logger.debug("admission rejected (%s): %s", entry_type, decision.reason)
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
            return self._admit(result.champion_comp, task, spec, depth, decompose_op, validation_result=result)
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
            **self._validation_provenance(result),
        }
        if self._refined_from is not None:
            provenance["refined_from"] = self._refined_from  # lineage: this entry continues that one
        if self._stepping_stone:
            provenance["stepping_stone"] = True  # a below-bar wall-ledger trace, not a solution
        level = module_level(result.champion_genome, self.library)
        payload = genome_to_dict(result.champion_genome)
        io = self._io_of(task, spec)
        if result.field_template is not None:
            from versal.field import FieldContract

            payload["field_template"] = result.field_template
            provenance["representation"] = "field"
            provenance["field_identity"] = FieldContract.from_dict(result.field_template).identity
            io = dict(io) | {"field_identity": provenance["field_identity"]}
        return self._gated_add(
            entry_type=MODULE,
            payload=payload,
            io=io,
            provenance=provenance,
            level=level,
            dependency=depth > 0 or self._stepping_stone,
        )

    def _admit(
        self,
        best: AssessedComposition,
        task: Task,
        spec: CompTaskSpec,
        depth: int,
        decompose_op: str | None,
        *,
        validation_result: StrategyResult,
    ) -> str | None:
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
            **self._validation_provenance(validation_result),
        }
        if self._refined_from is not None:
            provenance["refined_from"] = self._refined_from  # lineage: this entry continues that one
        if self._stepping_stone:
            provenance["stepping_stone"] = True  # a below-bar wall-ledger trace, not a solution
        return self._gated_add(
            entry_type=COMPOSITION, payload=comp_to_dict(detached), io=self._io_of(task, spec), provenance=provenance, level=level, dependency=depth > 0 or self._stepping_stone
        )

    @staticmethod
    def _validation_provenance(result: StrategyResult) -> dict[str, Any]:
        if result.validation_status == "not_run" and not result.validation_metrics:
            return {}
        return {"validation_status": result.validation_status, **result.validation_metrics}

    # --- skeleton wiring ------------------------------------------------------------------------------

    def _port_wired_skeleton(self, spec: CompTaskSpec, solutions: list[tuple[Subtask, Solution]]) -> CompositionGenome | None:
        """Seed every decomposer role with compact immutable gather/scatter port maps."""
        if any(solution.key is None for _subtask, solution in solutions):
            logger.debug("a sub-solution was solved but not shelved; relying on the ref catalog instead of a wired skeleton")
            return None
        if any(not subtask.port.input_runs or not subtask.port.output_runs for subtask, _ in solutions):
            logger.debug("decomposition produced a legacy port without compact maps; relying on the ref catalog instead")
            return None

        signature, parent_width = spec.input_specs[0]
        loaded: list[tuple[Subtask, str, LibraryEntry]] = []
        # Bias remains a small trainable vector. Fixed maps allocate only two int64 vectors per
        # selected port (four float-equivalent values), never source_width * target_width glue.
        glue_values = spec.output_width
        for subtask, solution in solutions:
            key = solution.key
            if key is None:  # unreachable after the guard above; narrows for the type checker
                return None
            entry = self.library.load(key)
            in_width = sum(item["width"] for item in entry.io["inputs"])
            out_width = entry.io["output"]["width"]
            input_count = sum(run[2] for run in subtask.port.input_runs)
            output_count = sum(run[2] for run in subtask.port.output_runs)
            if input_count != in_width or output_count != out_width:
                logger.debug(
                    "compact port map width drift for %s: map %d->%d, entry %d->%d; relying on the ref catalog",
                    subtask.task.meta.name,
                    input_count,
                    output_count,
                    in_width,
                    out_width,
                )
                return None
            if any(source + length > parent_width or target + length > in_width for source, target, length in subtask.port.input_runs):
                return None
            if any(source + length > out_width or target + length > spec.output_width for source, target, length in subtask.port.output_runs):
                return None
            glue_values += 8 * (input_count + output_count)
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

        for subtask, key, entry in loaded:
            in_width = sum(item["width"] for item in entry.io["inputs"])
            out_width = entry.io["output"]["width"]
            node_id = tracker.new_node_id()
            comp.nodes[node_id] = CompNodeGene(node_id, CompNodeKind.MODULE, f"library:{key}", in_width, out_width)
            input_map = PortMap(tuple(IndexRun(*run) for run in subtask.port.input_runs))
            output_map = PortMap(tuple(IndexRun(*run) for run in subtask.port.output_runs))
            comp.edges.append(CompEdgeGene(input_id, node_id, True, tracker.innovation(input_id, node_id), (), 0, input_map))
            comp.edges.append(CompEdgeGene(node_id, output_id, True, tracker.innovation(node_id, output_id), (), 0, output_map))
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
        if not result.has_admissible_champion or not self._accepts_metrics(result.champion_metrics):
            return False
        if not getattr(self, "cross_validation_config", CrossValidationConfig()).enabled:
            return True
        return result.validation_status in {"passed", "not_applicable"}

    def _report(self, item: Any) -> float:
        return self._report_value(item.metrics) or 0.0

    def _result_report_value(self, result: StrategyResult) -> float | None:
        return self._report_value(result.report_metrics) if result.report_metrics else None

    def _quality_of_result(self, result: StrategyResult) -> tuple[float | None, float | None, str, str]:
        """Return literal support/query accuracy plus explicit availability states.

        A router diagnostic without an executable payload is deliberately not promoted to a task
        result. The configurable policy scores continue to live in `metric`/`report_metric`.
        """

        if not result.has_admissible_champion and not result.has_report_candidate:
            return None, None, "no_executable_champion", "no_executable_champion"
        support_metrics = result.report_candidate_metrics or result.champion_metrics
        support = _finite_accuracy(support_metrics, "support_accuracy")
        if result.report_candidate_routed is not None:
            support_status = "evaluated" if support is not None else "evaluation_unavailable"
        elif result.has_admissible_champion:
            support_status = "evaluated" if support is not None else "evaluation_unavailable"
        else:
            support_status = "support_verification_incomplete"
        query_metrics = result.report_metrics if self.blind_query else support_metrics
        query = _finite_accuracy(query_metrics, "query_accuracy", loss_key="query_loss")
        if query is not None:
            query_status = "evaluated"
        elif "query_loss" in query_metrics and not math.isfinite(float(query_metrics["query_loss"])):
            query_status = "query_split_unavailable"
        else:
            query_status = "evaluation_unavailable"
        return support, query, support_status, query_status

    def _attempt_from_result(
        self,
        result: StrategyResult,
        *,
        task: str,
        depth: int,
        outcome: str,
        library_key: str | None = None,
        decompose_op: str | None = None,
        failure_stage: str | None = None,
        refine_generations: int = 0,
        query_status: str | None = None,
        report_result: StrategyResult | None = None,
        suppress_query: bool = False,
    ) -> Attempt:
        quality_result = report_result or result
        support, query, support_status, observed_query_status = self._quality_of_result(quality_result)
        if suppress_query:
            query = None
            observed_query_status = query_status or "shutdown_before_evaluation"
        elif support_status == "no_executable_champion":
            observed_query_status = "no_executable_champion"
        elif query is None and not quality_result.report_attempted and query_status is not None:
            observed_query_status = query_status
        size_metrics = dict(result.size_metrics)
        rank = self._candidate_rank(result)
        if rank is not None:
            if result.champion_genome is not None:
                shell = result.champion_genome.complexity()
            else:
                assert result.champion_comp is not None
                shell = result.champion_comp.comp.complexity()
            size_metrics.setdefault("champion_complexity", float(shell))
            size_metrics["champion_shell_complexity"] = float(shell)
            size_metrics["champion_expanded_complexity"] = float(rank.complexity)
        return Attempt(
            task=task,
            depth=depth,
            outcome=outcome,
            metric=result.metric,
            generations=result.generations_used,
            library_key=library_key,
            decompose_op=decompose_op,
            strategy=result.strategy,
            failure_stage=failure_stage,
            refine_generations=refine_generations,
            sample_metrics=_sample_metrics_of(result),
            size_metrics=size_metrics,
            resource_metrics=dict(result.resource_metrics),
            strategy_metrics=dict(result.strategy_metrics),
            validation_status=result.validation_status,
            validation_metrics=dict(result.validation_metrics),
            report_metric=None if suppress_query else self._result_report_value(quality_result),
            task_metrics=_task_metrics_of(quality_result),
            support_accuracy=support,
            query_accuracy=query,
            support_status=support_status,
            query_status=observed_query_status,
            representation=quality_result.representation or result.representation,
        )

    def _solution_from_result(self, result: StrategyResult, key: str | None, *, report_result: StrategyResult | None = None) -> Solution:
        quality_result = report_result or result
        support, query, support_status, query_status = self._quality_of_result(quality_result)
        return Solution(
            key=key,
            entry_type=self._entry_type_of(result),
            metric=result.metric,
            report_metric=self._result_report_value(quality_result),
            task_metrics=_task_metrics_of(quality_result),
            support_accuracy=support,
            query_accuracy=query,
            support_status=support_status,
            query_status=query_status,
        )

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
        if attempt.depth > 0:
            diagnostic_score = attempt.support_accuracy
            metric_name = "support_accuracy"
            if diagnostic_score is None and "router_score" in attempt.strategy_metrics:
                diagnostic_score = float(attempt.strategy_metrics["router_score"])
                metric_name = "router_score"
            if diagnostic_score is not None:
                observation = {
                    "score": float(diagnostic_score),
                    "metric": metric_name,
                    "task": attempt.task,
                    "depth": attempt.depth,
                    "strategy": attempt.strategy,
                    "executable": attempt.support_accuracy is not None,
                }
                current = getattr(self, "_best_diagnostic_observation", None)
                if current is None or float(observation["score"]) > float(current["score"]):
                    self._best_diagnostic_observation = observation
        elif not attempt.diagnostic_observation:
            attempt.diagnostic_observation = dict(getattr(self, "_best_diagnostic_observation", None) or {})
        self.attempts.append(attempt)
        logger.debug("attempt: %s", attempt.to_dict())
        BOARD.event(f"{attempt.task} d{attempt.depth}: {attempt.outcome} {self.search_metric}={attempt.metric:.3f}")

    def _on_generation(self, strategy: str, generation: int, best: Any, mean_fitness: float) -> None:
        support = float(best.metrics.get("support_accuracy", self._metric(best)))
        self.display.generation(strategy, generation, best.fitness, support, mean_fitness, depth=getattr(self, "_display_depth", 0))
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
