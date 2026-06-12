"""OrchestratedTrial: the recursive hierarchical orchestrated evolver, end to end.

Consumes the same `[schedule]` task stream as the continuous trial, but each task goes through the
orchestrator's escalation ladder (library lookup -> evolve -> decompose/recurse -> admit) instead of
sharing one mutable champion. Durable cross-task state is the LIBRARY (file-persistent), the live
module population, and the scheduler cursors; recursion within a task is synchronous and depth
capped, so checkpoints are written only when a task admits novel library entries.
"""

import datetime
import random
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

    def run(self) -> dict[str, Any]:
        if self.resume_dir:
            self.run_dir = Path(self.resume_dir)
            state, task_cursor, attempts, counters = self._restore()
            console.rule(f"[bold]Resuming orchestrated run at task {task_cursor} ({self.run_dir})")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(results.DEFAULT_ROOT) / f"{timestamp}_orchestrated"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            state = self.loop.fresh_state(random.Random(int(self.config.get("seed", 0))))
            task_cursor, attempts, counters = 0, [], None
            console.rule(f"[bold]Orchestrated run: rungs {self.rungs}, {len(self.pool)} tasks, library {self.library.root} -> {self.run_dir}")

        orchestrator = Orchestrator(self.config, self.loop, self.library, state, proctor=self)
        orchestrator.attempts = attempts
        if counters:
            orchestrator.counters = {**orchestrator.counters, **counters}  # old checkpoints lack newer counters

        while task_cursor < self.tasks_to_run:
            index = self.scheduler.next_index(self.pool, state.rng)
            entry = self.pool[index]
            console.print(f"[cyan]task {task_cursor + 1}/{self.tasks_to_run}[/cyan] rung {entry.rung} {entry.name}")
            library_keys_before = set(self.library.keys())
            solution = orchestrator.solve(entry.task)
            task_cursor += 1
            new_library_keys = [key for key in self.library.keys() if key not in library_keys_before]
            outcome = orchestrator.attempts[-1].outcome if orchestrator.attempts else "unknown"
            label = f"[green]{outcome}[/green]" if solution is not None else f"[red]{outcome}[/red]"
            console.print(f"  -> {label} (library size {len(self.library)})")
            self._log_task(orchestrator, state, task_cursor)
            if new_library_keys:
                self._checkpoint(orchestrator, state, task_cursor, new_library_keys, solution)

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

    def _restore(self) -> tuple[HierarchicalState, int, list[Any], dict[str, int]]:
        directory = checkpoint.latest_task_checkpoint_dir(self.run_dir)
        if directory is None:
            raise FileNotFoundError(f"no task checkpoint found under {self.run_dir}")
        data = checkpoint.read_checkpoint(directory)
        rng = checkpoint.deserialize_rng(data["rng"])
        state = state_from_dict(data["loop_state"], rng)
        self.scheduler.load_state_dict(data["schedule"])
        cast(Any, self.loop.evolver.speciate).load_state_dict(data["speciation"])
        return state, int(data["task_cursor"]), attempts_from_dicts(data["attempts"]), {k: int(v) for k, v in data["counters"].items()}

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
