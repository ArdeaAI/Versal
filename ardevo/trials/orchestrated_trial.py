"""OrchestratedTrial: the recursive hierarchical orchestrated evolver, end to end.

Consumes the same `[schedule]` task stream as the continuous trial, but each task goes through the
orchestrator's escalation ladder (library lookup -> evolve -> decompose/recurse -> admit) instead of
sharing one mutable champion. Durable cross-task state is the LIBRARY (file-persistent), the live
module population, and the scheduler cursors; recursion within a task is synchronous and depth
capped, so checkpoints are written only when a task admits novel library entries.
"""

import datetime
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, cast

import torch

from ardevo import checkpoint, rendering, results
from ardevo.evolution.composition import comp_from_dict
from ardevo.evolution.genome import genome_from_dict
from ardevo.evolution.loop import HierarchicalLoop, HierarchicalState, state_from_dict, state_to_dict
from ardevo.evolution.multitask import TaskEntry, build_pool_report
from ardevo.evolution.registry import build_loop
from ardevo.evolution.schedule import build_schedule
from ardevo.library import COMPOSITION, MODULE, ModuleLibrary, macro_resolver
from ardevo.orchestrator import Orchestrator, Solution, attempts_from_dicts, attempts_to_dicts
from ardevo.utils.logging import Logger
from ardevo.utils.proctor import Proctor

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
        self.gc_removed: list[str] | None = None  # set by the run-end sweep, reported in run_summary

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
            state = self.loop.fresh_state(random.Random(int(self.config.get("seed", 0))))
            task_cursor, attempts, counters = 0, [], None
            self.task_records = []
            console.rule(f"[bold]Orchestrated run: rungs {self.rungs}, {len(self.pool)} tasks, library {self.library.root} -> {self.run_dir}")

        # A durable record exists before the first task so a crash during setup or task 0 leaves a
        # diagnosable run_summary.json instead of an empty directory (the silent-failure mode we kill).
        orchestrator: Orchestrator | None = None
        try:
            orchestrator = Orchestrator(self.config, self.loop, self.library, state, proctor=self)
            orchestrator.attempts = attempts
            if counters:
                orchestrator.counters = {**orchestrator.counters, **counters}  # old checkpoints lack newer counters
            self._write_run_summary(orchestrator, state, task_cursor, status="running")
            while task_cursor < self.tasks_to_run:
                index = self.scheduler.next_index(self.pool, state.rng)
                entry = self.pool[index]
                console.print(f"[cyan]task {task_cursor + 1}/{self.tasks_to_run}[/cyan] rung {entry.rung} {entry.name}")
                library_keys_before = set(self.library.keys())
                solution = orchestrator.solve(entry.task)
                task_cursor += 1
                self.library.flush_stats()  # deferred bump_stats writes land at the task boundary
                new_library_keys = [key for key in self.library.keys() if key not in library_keys_before]
                attempt = orchestrator.attempts[-1] if orchestrator.attempts else None
                outcome = attempt.outcome if attempt is not None else "unknown"
                label = f"[green]{outcome}[/green]" if solution is not None else f"[red]{outcome}[/red]"
                console.print(f"  -> {label} (library size {len(self.library)})")
                self._log_task(orchestrator, state, task_cursor)
                # Durable record EVERY task, regardless of admission: a run is now measurable and
                # resumable even when nothing new is shelved (the empty-run-dir bug is gone).
                self._record_task(entry, attempt, new_library_keys, len(self.library))
                self._write_run_summary(orchestrator, state, task_cursor, status="running")
                if task_cursor % self.checkpoint_every == 0:
                    self._persist_resume_state(orchestrator, state, task_cursor)
                if new_library_keys:
                    self._checkpoint(orchestrator, state, task_cursor, new_library_keys, solution)
        except BaseException as error:  # record the failure, then re-raise: no more silent empty runs
            self.library.flush_stats()  # a crash still leaves the stats it had
            self._write_run_summary(orchestrator, state, task_cursor, status=f"crashed: {type(error).__name__}: {error}")
            if orchestrator is not None:
                self._persist_resume_state(orchestrator, state, task_cursor)
            raise

        self.library.flush_stats()
        self._persist_resume_state(orchestrator, state, task_cursor)
        if self.gc_enabled:
            self._run_gc(state)
        self._write_run_summary(orchestrator, state, task_cursor, status="done")
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

    def _record_task(self, entry: TaskEntry, attempt: Any, new_library_keys: list[str], library_size: int) -> None:
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
        self.task_records.append(record)

    def _write_run_summary(self, orchestrator: Orchestrator | None, state: HierarchicalState, task_cursor: int, *, status: str) -> None:
        """The always-on, cheap, durable record of a run: one row per attempted task plus aggregate
        counters. Written every task and on crash, so a run is never an empty directory again.
        Tolerates `orchestrator=None` so a crash during orchestrator construction is still recorded."""
        outcomes = Counter(record["outcome"] for record in self.task_records)
        summary = {
            "run_dir": str(self.run_dir),
            "status": status,
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
        gc_removed = getattr(self, "gc_removed", None)  # tolerate partially-constructed trials (white-box tests)
        if gc_removed is not None:
            summary["gc_removed"] = len(gc_removed)
        (self.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

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

    def _log_task(self, orchestrator: Orchestrator, state: HierarchicalState, task_cursor: int) -> None:
        for series, value in orchestrator.counters.items():
            self.log_scalar("Orchestrator", series, value, task_cursor)
        self.log_scalar("Orchestrator", "library_size", len(self.library), task_cursor)
        self.log_scalar("Orchestrator", "skipped_rungs", len(self.skipped_rungs), task_cursor)
        self.log_scalar("Orchestrator", "repaired_refs", state.repaired_refs, task_cursor)
        self.log_scalar("Modules", "pool_size", len(state.modules), task_cursor)
        if state.modules:
            self.log_scalar("Modules", "mean_fitness", sum(m.fitness for m in state.modules) / len(state.modules), task_cursor)
        self.log_scalar("Modules", "species", len(state.species_champions), task_cursor)
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
        results.render_speciation(directory, state.module_species_history, title=f"module species through task {task_cursor}")
        if self.task:
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
            rendering.render_network(directory, genome_from_dict(entry.payload), title=title, library=self.library)
        elif entry.entry_type == COMPOSITION:
            rendering.render_composition_network(directory, comp_from_dict(entry.payload), title=title, library=self.library)
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
