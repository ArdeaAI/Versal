"""OrchestratedTrial: the recursive hierarchical orchestrated evolver, end to end.

Consumes the same `[schedule]` task stream as the continuous trial, but each task goes through the
orchestrator's escalation ladder (library lookup -> evolve -> decompose/recurse -> admit) instead of
sharing one mutable champion. Durable cross-task state is the LIBRARY (file-persistent), the live
module population, and the scheduler cursors; recursion within a task is synchronous and depth
capped, so checkpoints are written only when a task admits novel library entries.
"""

import datetime
import fnmatch
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import psutil
import torch

from ardevo import checkpoint, rendering, results
from ardevo.evolution.composition import comp_from_dict
from ardevo.evolution.genome import genome_from_dict
from ardevo.evolution.loop import HierarchicalLoop, HierarchicalState, state_from_dict, state_to_dict
from ardevo.evolution.multitask import TaskEntry, build_pool_report
from ardevo.evolution.registry import build_loop
from ardevo.evolution.schedule import build_schedule
from ardevo.external_archive import ArchiveManager, ExperimentLock
from ardevo.library import COMPOSITION, MODULE, ModuleLibrary, macro_resolver
from ardevo.orchestrator import Orchestrator, Solution, attempts_from_dicts, attempts_to_dicts
from ardevo.reporting import write_run_report
from ardevo.utils.device import capture_hardware_profile
from ardevo.utils.logging import Logger
from ardevo.utils.proctor import Proctor
from ardevo.utils.status import BOARD

logger = Logger.get_logger()
console = Logger.get_console()


def pool_from_tasks(tasks: list[Any]) -> list[TaskEntry]:
    """Adapter seam for future task sources (e.g. an ARC harness) that never touch IcarusDataset."""
    from ardevo.evolution.multitask import task_entry

    return [task_entry(task) for task in tasks]


class OrchestratedTrial(Proctor):
    """Throw tasks at the orchestrator; everything it solves becomes reusable library knowledge."""

    def __init__(self, config: dict[str, Any], task: Any = None) -> None:
        super().__init__(config=config, task=task)
        torch.manual_seed(int(config.get("seed", 0)))

        table = config.get("orchestrator", {})
        schedule_cfg = config.get("schedule", {})
        rungs_cfg = schedule_cfg.get("rungs", [1, 2, 3, 4, 5])
        self.rungs = list(range(1, 19)) if rungs_cfg == "all" else [int(rung) for rung in rungs_cfg]
        self.tasks_to_run = int(table.get("tasks", 20))
        # Resume state is persisted every `checkpoint_every` tasks (default 1 = bit-stable resume);
        # the per-task run_summary record is always written regardless of this cadence.
        self.checkpoint_every = max(1, int(table.get("checkpoint_every", 1)))
        self.task_records: list[dict[str, Any]] = []

        report = build_pool_report(
            source=config["dataset"],
            rungs=self.rungs,
            n_samples=int(config["n_samples"]),
            support_fraction=float(config.get("support_fraction", 0.8)),
            tasks_per_rung=int(schedule_cfg.get("tasks_per_rung", 100)),
            shuffle=bool(schedule_cfg.get("shuffle", True)),
            seed=int(config.get("seed", 0)),
        )
        self.pool: list[TaskEntry] = report.entries
        include = schedule_cfg.get("task_include", [])
        exclude = schedule_cfg.get("task_exclude", [])
        include_patterns = [str(pattern) for pattern in ([include] if isinstance(include, str) else include)]
        exclude_patterns = [str(pattern) for pattern in ([exclude] if isinstance(exclude, str) else exclude)]
        if include_patterns:
            self.pool = [entry for entry in self.pool if any(fnmatch.fnmatchcase(entry.name, pattern) for pattern in include_patterns)]
        if exclude_patterns:
            self.pool = [entry for entry in self.pool if not any(fnmatch.fnmatchcase(entry.name, pattern) for pattern in exclude_patterns)]
        self.skipped_rungs = report.skipped
        for skipped in self.skipped_rungs:
            console.print(f"[bold red]rung {skipped.rung} skipped[/bold red]: {skipped.error_type}: {skipped.message}")
        if self.skipped_rungs and bool(schedule_cfg.get("require_all_rungs", False)):
            # NO SKIPPING: the ladder is climbed whole or the run refuses to start. Silent rung
            # tolerance is how a wall stops being attempted without anyone deciding that.
            reasons = "; ".join(f"rung {s.rung}: {s.error_type}: {s.message}" for s in self.skipped_rungs)
            raise RuntimeError(f"require_all_rungs: {len(self.skipped_rungs)} rung(s) failed to load ({reasons}); probe with `uv run rung_doctor`")
        if not self.pool:
            reasons = "; ".join(f"rung {s.rung}: {s.error_type}" for s in self.skipped_rungs) or "no rungs configured"
            raise RuntimeError(f"no tasks found for rungs {self.rungs} in {config['dataset']!r} ({reasons})")

        loop = build_loop(config)
        if not isinstance(loop, HierarchicalLoop):
            raise ValueError('the orchestrated trial requires [evolution] loop = "hierarchical"')
        self.loop = loop
        self.library = ModuleLibrary(table.get("library_dir", "library"))
        self.loop.attach_library(self.library)
        # Macro-bearing genomes resolve their frozen inners through the LIVE library, everywhere
        # decode is reachable in this process (direct strategy, quick evals, evolver internals).
        from ardevo.substrate import set_macro_resolver

        set_macro_resolver(macro_resolver(self.library))
        self.scheduler = build_schedule(schedule_cfg)
        self.resume_dir = config.get("resume")
        self.run_dir = Path(results.DEFAULT_ROOT)
        self.gc_enabled = bool(config.get("library", {}).get("gc", False))
        self.fresh_per_task = bool(config.get("library", {}).get("fresh_per_task", False))
        self.frozen_library_dir: Path | None = None
        self.gc_removed: list[str] | None = None  # set by the run-end sweep, reported in run_summary
        self.archive_manager: ArchiveManager | None = None
        self.experiment_lock: ExperimentLock | None = None
        self.hardware_profile = capture_hardware_profile().to_dict()

    def run(self) -> dict[str, Any]:
        if self.resume_dir:
            self.run_dir = Path(self.resume_dir)
            state, task_cursor, attempts, counters = self._restore()
            self.task_records = self._load_prior_records()
            console.rule(f"[bold]Resuming orchestrated run at task {task_cursor} ({self.run_dir})")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(results.DEFAULT_ROOT) / f"{timestamp}_orchestrated"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._snapshot_config()
            self._prepare_frozen_library()
            state = self.loop.fresh_state(random.Random(int(self.config.get("seed", 0))))
            task_cursor, attempts, counters = 0, [], None
            self.task_records = []
            console.rule(f"[bold]Orchestrated run: rungs {self.rungs}, {len(self.pool)} tasks, library {self.library.root} -> {self.run_dir}")

        self.experiment_lock = ExperimentLock(self.run_dir, self.library.root)
        self.experiment_lock.acquire()
        # A durable record exists before the first task so a crash during setup or task 0 leaves a
        # diagnosable run_summary.json instead of an empty directory (the silent-failure mode we kill).
        orchestrator: Orchestrator | None = None
        try:
            if bool(self.config.get("render_async", False)):
                rendering.enable_async_rendering()
            if bool(self.config.get("live_status", True)):
                BOARD.enable(console)  # quietly refuses off-terminal (pipes, agents, CI)
            orchestrator = Orchestrator(self.config, self.loop, self.library, state, proctor=self)
            orchestrator.attempts = attempts
            if counters:
                orchestrator.counters = {**orchestrator.counters, **counters}  # old checkpoints lack newer counters
            self.archive_manager = ArchiveManager.from_config(self.config, self.run_dir, self.library.root)
            self._write_run_summary(orchestrator, state, task_cursor, status="running")
            while task_cursor < self.tasks_to_run:
                index = self.scheduler.next_index(self.pool, state.rng)
                entry = self.pool[index]
                console.print(f"[cyan]task {task_cursor + 1}/{self.tasks_to_run}[/cyan] rung {entry.rung} {entry.name}")
                BOARD.task(task_cursor + 1, self.tasks_to_run, entry.rung, entry.name)
                library_keys_before = set(self.library.keys())
                task_started = time.perf_counter()
                isolated = self._solve_isolated(entry, task_cursor, orchestrator) if self.fresh_per_task else None
                if isolated is None:
                    solution = orchestrator.solve(entry.task)
                    attempt = orchestrator.attempts[-1] if orchestrator.attempts else None
                    module_pool_sizes = self._module_pool_sizes(state)
                else:
                    solution, attempt, isolated_generations = isolated
                    state.generation += isolated_generations
                    module_pool_sizes = {}
                task_seconds = time.perf_counter() - task_started
                task_cursor += 1
                self.library.flush_stats()  # deferred bump_stats writes land at the task boundary
                new_library_keys = [key for key in self.library.keys() if key not in library_keys_before]
                outcome = attempt.outcome if attempt is not None else "unknown"
                if hasattr(self.scheduler, "observe"):  # feedback-driven schedulers (regret); others untouched
                    self.scheduler.observe(entry.rung, attempt.metric if attempt is not None else 0.0, solution is not None)
                label = f"[green]{outcome}[/green]" if solution is not None else f"[red]{outcome}[/red]"
                stages = dict(getattr(attempt, "stage_seconds", None) or {})
                stage_note = f" [{', '.join(f'{k} {v:.0f}s' for k, v in stages.items())}]" if stages else ""
                console.print(f"  -> {label} (library size {len(self.library)}, {task_seconds:.0f}s{stage_note})")
                self._log_task(orchestrator, state, task_cursor, task_seconds, module_pool_sizes)
                # Durable record EVERY task, regardless of admission: a run is now measurable and
                # resumable even when nothing new is shelved (the empty-run-dir bug is gone).
                self._record_task(entry, attempt, new_library_keys, len(self.library), module_pool_sizes)
                self._write_run_summary(orchestrator, state, task_cursor, status="running")
                if new_library_keys:
                    self._checkpoint(orchestrator, state, task_cursor, new_library_keys, solution)
                if self.archive_manager is not None and self.archive_manager.due(task_cursor):
                    self._archive_boundary(orchestrator, state, task_cursor)
                elif task_cursor % self.checkpoint_every == 0:
                    self._persist_resume_state(orchestrator, state, task_cursor)
        except BaseException as error:  # record the failure, then re-raise: no more silent empty runs
            try:
                BOARD.close()  # release the terminal first so the traceback and summaries print clean
                self.library.flush_stats()  # a crash still leaves the stats it had
                rendering.flush_renders()  # pending async renders finish (their own failures only log)
                self._write_run_summary(orchestrator, state, task_cursor, status=f"crashed: {type(error).__name__}: {error}")
                if orchestrator is not None:
                    self._persist_resume_state(orchestrator, state, task_cursor)
                self._archive_boundary(orchestrator, state, task_cursor, status=f"crashed-{type(error).__name__}", force=True, best_effort=True)
            finally:
                self._release_experiment_lock()
            raise

        try:
            BOARD.close()
            self.library.flush_stats()
            rendering.flush_renders()  # renders land before GC can delete images and before finalize
            self._persist_resume_state(orchestrator, state, task_cursor)
            if self.gc_enabled and not self.fresh_per_task:
                self._run_gc(state)
            self._write_run_summary(orchestrator, state, task_cursor, status="done")
            self._publish_final_archive(orchestrator, state, task_cursor)
            self.results = {
                "tasks_attempted": task_cursor,
                "library_size": len(self.library),
                "counters": dict(orchestrator.counters),
                "module_pool": len(state.modules),
                "generations_run": state.generation,
                "run_dir": str(self.run_dir),
            }
            console.print(f"[bold green]Done[/bold green]: {task_cursor} tasks, library {len(self.library)} entries, counters {orchestrator.counters}")
            self.finalize()
            return self.results
        finally:
            self._release_experiment_lock()

    def _release_experiment_lock(self) -> None:
        lock = getattr(self, "experiment_lock", None)
        if lock is not None:
            lock.release()
            self.experiment_lock = None

    def _prepare_frozen_library(self) -> None:
        if not self.fresh_per_task:
            return
        frozen = self.run_dir / "frozen_library"
        if frozen.exists():
            self.frozen_library_dir = frozen
            return
        self.library.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.library.root, frozen, ignore=shutil.ignore_patterns("encoded_cache", "images"))
        self.frozen_library_dir = frozen

    def _solve_isolated(
        self,
        entry: TaskEntry,
        task_cursor: int,
        aggregate: Orchestrator,
    ) -> tuple[Solution | None, Any, int] | None:
        """Solve one task against a disposable copy of the frozen starting library."""

        if self.frozen_library_dir is None:
            self._prepare_frozen_library()
        if self.frozen_library_dir is None:
            return None
        with tempfile.TemporaryDirectory(prefix="ardevo_task_") as tmp:
            root = Path(tmp) / "library"
            shutil.copytree(self.frozen_library_dir, root)
            library = ModuleLibrary(root)
            loop = build_loop(self.config)
            if not isinstance(loop, HierarchicalLoop):
                raise ValueError('the orchestrated trial requires [evolution] loop = "hierarchical"')
            loop.attach_library(library)
            from ardevo.evolution.evolver import get_shared_pool, set_shared_pool
            from ardevo.substrate import set_macro_resolver

            shared_pool = get_shared_pool()
            set_shared_pool(None)
            set_macro_resolver(macro_resolver(library))
            try:
                seed = int(self.config.get("seed", 0)) * 1_000_003 + task_cursor
                state = loop.fresh_state(random.Random(seed))
                isolated = Orchestrator(self.config, loop, library, state, proctor=self)
                solution = isolated.solve(entry.task)
                library.flush_stats()
                attempt = isolated.attempts[-1] if isolated.attempts else None
                # The task-local directory is deleted on return. Never publish keys that will point
                # at vanished entries in checkpoints, summaries, or the returned Solution.
                for row in isolated.attempts:
                    row.library_key = None
                if solution is not None:
                    solution = replace(solution, key=None)
                aggregate.attempts.extend(isolated.attempts)
                for key, value in isolated.counters.items():
                    aggregate.counters[key] = aggregate.counters.get(key, 0) + value
                return solution, attempt, state.generation
            finally:
                set_macro_resolver(macro_resolver(self.library))
                set_shared_pool(shared_pool)

    def _run_gc(self, state: HierarchicalState) -> None:
        """Run-end sweep of unreferenced tombstones ([library] gc = true). Rooted in the LIVE state:
        macro refs inside the pooled/champion genomes are protected because the final checkpoint
        just serialized exactly those, so resuming this run can never dangle. Dead router vertices
        are pruned (tolerant reload + save) after the sweep."""
        from ardevo.evolution.genome import genome_to_dict
        from ardevo.library import payload_refs

        protect: set[str] = set()
        for module in state.modules:
            protect |= payload_refs(MODULE, genome_to_dict(module.genome))
        for genome in state.species_champions.values():
            protect |= payload_refs(MODULE, genome_to_dict(genome))
        self.gc_removed = self.library.collect_garbage(protect=protect)
        router_dir = Path(self.library.root) / "router"
        if self.gc_removed and (router_dir / "router_meta.json").exists():
            from ardevo.tools.library_gc import prune_router

            pruned = prune_router(self.library, router_dir)
            console.print(f"[dim]gc: removed {len(self.gc_removed)} tombstones, pruned {pruned} router vertices[/dim]")
        elif self.gc_removed:
            console.print(f"[dim]gc: removed {len(self.gc_removed)} tombstones[/dim]")

    def _restore(self) -> tuple[HierarchicalState, int, list[Any], dict[str, int]]:
        # Prefer the rolling run-root checkpoint (written EVERY task, so it is the true latest
        # state); fall back to the newest per-admission task_*/ checkpoint for pre-Phase-5 runs.
        if (self.run_dir / "checkpoint.json").exists():
            directory: Path | None = self.run_dir
        else:
            directory = checkpoint.latest_task_checkpoint_dir(self.run_dir)
        if directory is None:
            raise FileNotFoundError(f"no checkpoint found under {self.run_dir}")
        data = checkpoint.read_checkpoint(directory)
        rng = checkpoint.deserialize_rng(data["rng"])
        state = state_from_dict(data["loop_state"], rng)
        self.scheduler.load_state_dict(data["schedule"])
        cast(Any, self.loop.evolver.speciate).load_state_dict(data["speciation"])
        return state, int(data["task_cursor"]), attempts_from_dicts(data["attempts"]), {k: int(v) for k, v in data["counters"].items()}

    def _load_prior_records(self) -> list[dict[str, Any]]:
        """On resume, recover the per-task records already written so run_summary.json stays
        cumulative across resumes (best-effort: a malformed/absent summary just starts empty)."""
        summary_path = self.run_dir / "run_summary.json"
        if not summary_path.exists():
            return []
        try:
            return list(json.loads(summary_path.read_text()).get("tasks", []))
        except (ValueError, OSError):
            return []

    @staticmethod
    def _module_pool_sizes(state: HierarchicalState) -> dict[str, float]:
        """Median/max genome size across the persistent module pool: the cross-task bloat
        reservoir (it rides checkpoints, so unchecked growth compounds over every later task)."""
        if not state.modules:
            return {}
        node_counts = sorted(len(member.genome.nodes) for member in state.modules)
        connection_counts = sorted(len(member.genome.enabled_connections()) for member in state.modules)
        return {
            "pool_median_nodes": float(node_counts[len(node_counts) // 2]),
            "pool_max_nodes": float(node_counts[-1]),
            "pool_median_connections": float(connection_counts[len(connection_counts) // 2]),
            "pool_max_connections": float(connection_counts[-1]),
        }

    def _record_task(self, entry: TaskEntry, attempt: Any, new_library_keys: list[str], library_size: int, module_pool_sizes: dict[str, float] | None = None) -> None:
        record = {
            "rung": entry.rung,
            "task": entry.name,
            "outcome": attempt.outcome if attempt is not None else "unknown",
            "metric": attempt.metric if attempt is not None else 0.0,
            "strategy": attempt.strategy if attempt is not None else None,
            "generations": attempt.generations if attempt is not None else 0,
            "depth": attempt.depth if attempt is not None else 0,
            "decompose_op": attempt.decompose_op if attempt is not None else None,
            "failure_stage": attempt.failure_stage if attempt is not None else None,
            "new_library_keys": list(new_library_keys),
            "library_size": library_size,
        }
        if attempt is not None and getattr(attempt, "refine_generations", 0):  # only when refinement ran (live mode stays byte-identical)
            record["refine_generations"] = attempt.refine_generations
        if attempt is not None and getattr(attempt, "seconds", 0.0):
            record["seconds"] = attempt.seconds
            if getattr(attempt, "stage_seconds", None):
                record["stage_seconds"] = dict(attempt.stage_seconds)
        if attempt is not None and getattr(attempt, "sample_metrics", None):  # hybrid-eval G0 diagnostic; absent under standard eval
            record["sample_metrics"] = dict(attempt.sample_metrics)
        if attempt is not None and getattr(attempt, "size_metrics", None):  # champion/population genome size: the bloat readout
            record["size_metrics"] = dict(attempt.size_metrics)
        if attempt is not None and getattr(attempt, "report_metric", None) is not None:
            record["report_metric"] = float(attempt.report_metric)
        if attempt is not None and getattr(attempt, "task_metrics", None):
            record["task_metrics"] = dict(attempt.task_metrics)
        if attempt is not None and getattr(attempt, "resource_metrics", None):
            record["resource_metrics"] = dict(attempt.resource_metrics)
        if attempt is not None and getattr(attempt, "strategy_metrics", None):
            record["strategy_metrics"] = dict(attempt.strategy_metrics)
        if module_pool_sizes:
            record["module_pool"] = dict(module_pool_sizes)
        self.task_records.append(record)

    def _write_run_summary(self, orchestrator: Orchestrator | None, state: HierarchicalState, task_cursor: int, *, status: str) -> None:
        """The always-on, cheap, durable record of a run: one row per attempted task plus aggregate
        counters. Written every task and on crash, so a run is never an empty directory again.
        Tolerates `orchestrator=None` so a crash during orchestrator construction is still recorded."""
        outcomes = Counter(record["outcome"] for record in self.task_records)
        summary = {
            "run_dir": str(self.run_dir),
            "status": status,
            "config_path": self.config.get("config_path", ""),
            "config_sha256": self.config.get("config_sha256", ""),
            "config_effective_sha256": self.config.get("config_effective_sha256", ""),
            "config_sources": list(self.config.get("config_sources", [])),
            "seed": int(self.config.get("seed", 0)),
            "library_dir": str(self.library.root),
            "rungs": self.rungs,
            "tasks_attempted": task_cursor,
            "tasks_to_run": self.tasks_to_run,
            "library_size": len(self.library),
            "library_keys": self.library.keys(),
            "counters": dict(orchestrator.counters) if orchestrator is not None else {},
            "outcomes": dict(outcomes),
            "generations_run": state.generation,
            "skipped_rungs": [{"rung": s.rung, "error_type": s.error_type, "message": s.message} for s in self.skipped_rungs],
            "tasks": self.task_records,
        }
        hardware_profile = getattr(self, "hardware_profile", None)
        if hardware_profile is not None:
            summary["hardware_profile"] = hardware_profile
            summary["execution"] = {
                "hierarchical": getattr(self.loop.evolver, "execution_mode", "serial"),
                "assess_workers": int(getattr(self.loop.evolver, "assess_workers", 0)),
            }
            if orchestrator is not None:
                summary["execution"]["strategies"] = {name: getattr(getattr(strategy, "evolver", None), "execution_mode", "n/a") for name, strategy in orchestrator.strategies}
        gc_removed = getattr(self, "gc_removed", None)  # tolerate partially-constructed trials (white-box tests)
        if gc_removed is not None:
            summary["gc_removed"] = len(gc_removed)
        if orchestrator is not None and getattr(orchestrator, "blind_query", False):
            summary["search_metric"] = orchestrator.search_metric
            summary["report_metric"] = orchestrator.report_metric
        path = self.run_dir / "run_summary.json"
        payload = (json.dumps(summary, indent=2) + "\n").encode()
        descriptor, temporary = tempfile.mkstemp(prefix=".run_summary.json.", dir=self.run_dir)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        # Reports share the summary's durability boundary.  A report failure is a run failure:
        # silently stale analysis is more dangerous than a visibly resumable crash.
        write_run_report(self.run_dir, self.library.root)

    def _snapshot_config(self) -> None:
        """Copy the exact run config beside the checkpoint before task zero.

        A hash without bytes is insufficient once a working config evolves. Snapshotting is local
        run state (``results/`` is gitignored) and makes every crash or interrupted campaign
        independently reconstructable.
        """

        source_text = str(self.config.get("config_path", ""))
        if not source_text:
            return
        source = Path(source_text)
        if not source.exists():
            return
        payload = source.read_bytes()
        source_hash = hashlib.sha256(payload).hexdigest()
        self.config["config_sha256"] = source_hash
        (self.run_dir / "config.toml").write_bytes(payload)
        (self.run_dir / "config.toml.sha256").write_text(f"{source_hash}  config.toml\n")

        effective = json.dumps(self.config, indent=2, sort_keys=True, default=str) + "\n"
        effective_payload = effective.encode("utf-8")
        effective_hash = hashlib.sha256(effective_payload).hexdigest()
        (self.run_dir / "config.effective.json").write_bytes(effective_payload)
        (self.run_dir / "config.effective.json.sha256").write_text(f"{effective_hash}  config.effective.json\n")

    def _persist_resume_state(self, orchestrator: Orchestrator, state: HierarchicalState, task_cursor: int) -> None:
        """Rolling resumable checkpoint at the run root, written every task (not gated on admission),
        so a resume restores the EXACT latest state and RNG/scheduler never desync."""
        checkpoint.write_checkpoint(
            self.run_dir,
            checkpoint.build_orchestrated_payload(
                task_cursor=task_cursor,
                rng=state.rng,
                scheduler=self.scheduler,
                speciator=self.loop.evolver.speciate,
                loop_state=state_to_dict(state),
                attempts=attempts_to_dicts(orchestrator.attempts),
                counters=orchestrator.counters,
            ),
        )

    def _archive_boundary(
        self,
        orchestrator: Orchestrator | None,
        state: HierarchicalState,
        task_cursor: int,
        *,
        status: str = "running",
        force: bool = False,
        best_effort: bool = False,
    ) -> dict[str, Any] | None:
        """Publish only a render-complete, checkpointed between-task state."""
        manager = getattr(self, "archive_manager", None)
        if manager is None or (not force and not manager.due(task_cursor)):
            return None
        try:
            rendering.flush_renders()
            if orchestrator is not None:
                self._persist_resume_state(orchestrator, state, task_cursor)
            return manager.snapshot(task_cursor, status=status)
        except BaseException:
            if not best_effort:
                raise
            logger.exception("best-effort external archive snapshot failed at task %d", task_cursor)
            return None

    def _publish_final_archive(self, orchestrator: Orchestrator, state: HierarchicalState, task_cursor: int) -> None:
        """Make a failed authoritative final upload visibly resumable instead of leaving `done`."""

        try:
            self._archive_boundary(orchestrator, state, task_cursor, status="done", force=True)
        except BaseException as error:
            self._write_run_summary(orchestrator, state, task_cursor, status=f"crashed: external archive: {type(error).__name__}: {error}")
            raise

    def _log_task(
        self, orchestrator: Orchestrator, state: HierarchicalState, task_cursor: int, task_seconds: float = 0.0, module_pool_sizes: dict[str, float] | None = None
    ) -> None:
        for series, value in orchestrator.counters.items():
            self.log_scalar("Orchestrator", series, value, task_cursor)
        self.log_scalar("Orchestrator", "library_size", len(self.library), task_cursor)
        self.log_scalar("Orchestrator", "skipped_rungs", len(self.skipped_rungs), task_cursor)
        self.log_scalar("Orchestrator", "repaired_refs", state.repaired_refs, task_cursor)
        self.log_scalar("Modules", "pool_size", len(state.modules), task_cursor)
        if state.modules:
            self.log_scalar("Modules", "mean_fitness", sum(m.fitness for m in state.modules) / len(state.modules), task_cursor)
        self.log_scalar("Modules", "species", len(state.species_champions), task_cursor)
        # Genome-size series: the bloat curves that made the diag_g2 wall-clock explosion diagnosable.
        for series, value in (module_pool_sizes or {}).items():
            self.log_scalar("Modules", series, value, task_cursor)
        attempt = orchestrator.attempts[-1] if orchestrator.attempts else None
        for series, value in (getattr(attempt, "size_metrics", None) or {}).items():
            self.log_scalar("Size", series, value, task_cursor)
        for series, value in (getattr(attempt, "resource_metrics", None) or {}).items():
            self.log_scalar("Resources", series, value, task_cursor)
        # Wall-clock + memory per task: a wedged stage or a leaking process must be visible in
        # the run record, not only via sampling a live process.
        if task_seconds:
            self.log_scalar("Throughput", "task_seconds", task_seconds, task_cursor)
        self.log_scalar("Resources", "main_rss_gb", psutil.Process().memory_info().rss / 1e9, task_cursor)
        self.log_hardware_stats(task_cursor)

    def _checkpoint(self, orchestrator: Orchestrator, state: HierarchicalState, task_cursor: int, new_library_keys: list[str], solution: Solution | None) -> None:
        directory = self.run_dir / f"task_{task_cursor:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        net_key = self._net_artifact_key(new_library_keys, solution.key if solution is not None else None)
        results.write_stats(directory, self._stats(orchestrator, state, task_cursor, new_library_keys, net_key))
        self._render_net_artifact(directory, net_key, task_cursor)
        checkpoint.write_checkpoint(
            directory,
            checkpoint.build_orchestrated_payload(
                task_cursor=task_cursor,
                rng=state.rng,
                scheduler=self.scheduler,
                speciator=self.loop.evolver.speciate,
                loop_state=state_to_dict(state),
                attempts=attempts_to_dicts(orchestrator.attempts),
                counters=orchestrator.counters,
            ),
        )
        # Snapshot the history: the async render thread must never read a list the next task mutates.
        rendering.submit_render(results.render_speciation, directory, [dict(row) for row in state.module_species_history], title=f"module species through task {task_cursor}")
        if self.task:
            rendering.flush_renders()  # artifacts upload only finished files
            for name in ("stats.json", "checkpoint.json", "speciation.png", "net.png"):
                path = directory / name
                if path.exists():
                    self.save_artifact(f"task{task_cursor:04d}_{name}", str(path))

    def _net_artifact_key(self, new_library_keys: list[str], solution_key: str | None) -> str:
        if solution_key in new_library_keys:
            return str(solution_key)
        entries = [self.library.load(key) for key in new_library_keys]
        compositions = [entry for entry in entries if entry.entry_type == COMPOSITION]
        chosen = max(compositions or entries, key=lambda entry: entry.level)
        return chosen.key

    def _render_net_artifact(self, directory: Path, key: str, task_cursor: int) -> None:
        entry = self.library.load(key)
        title = f"orchestrated task {task_cursor}: {entry.entry_type} {entry.key}"
        if entry.entry_type == MODULE:
            rendering.submit_render(
                rendering.render_network,
                directory,
                genome_from_dict(entry.payload),
                title=title,
                library=self.library,
                max_inline_depth=self.loop.max_inline_depth,
            )
        elif entry.entry_type == COMPOSITION:
            rendering.submit_render(
                rendering.render_composition_network,
                directory,
                comp_from_dict(entry.payload),
                title=title,
                library=self.library,
                max_inline_depth=self.loop.max_inline_depth,
            )
        else:
            raise ValueError(f"unknown library entry type {entry.entry_type!r}")

    def _stats(self, orchestrator: Orchestrator, state: HierarchicalState, task_cursor: int, new_library_keys: list[str], net_key: str) -> dict[str, Any]:
        return {
            "task_cursor": task_cursor,
            "generations_run": state.generation,
            "orchestrator": {
                "counters": dict(orchestrator.counters),
                "attempts": attempts_to_dicts(orchestrator.attempts),
            },
            "library": {"size": len(self.library), "keys": self.library.keys(), "path": str(self.library.root), "new_keys": new_library_keys, "net_key": net_key},
            "schedule_coverage": {"rungs": self.rungs, "skipped": [{"rung": s.rung, "error_type": s.error_type, "message": s.message} for s in self.skipped_rungs]},
            "modules": {
                "pool_size": len(state.modules),
                "species": len(state.species_champions),
                "mean_fitness": (sum(m.fitness for m in state.modules) / len(state.modules)) if state.modules else 0.0,
                "repaired_refs": state.repaired_refs,
            },
            "config": {
                "orchestrator": self.config.get("orchestrator", {}),
                "schedule": self.config.get("schedule", {}),
                "evolution": self.config.get("evolution", {}),
                "fitness": self.config.get("fitness", {}),
                "dataset": {"source": self.config.get("dataset"), "n_samples": self.config.get("n_samples"), "support_fraction": self.config.get("support_fraction")},
            },
        }
