"""ContinuousTrial: grow ONE topology across the interleaved Icarus ladder, with checkpoint/resume.

Drives the steppable `Evolver` over an explicit `EvolverState`: it builds a pool of tasks across the
configured rungs, seeds the population for the first task, then runs `generations_per_task` at a time
before the scheduler picks the next task. On a switch it grows the shared interface for the new task
(`MultiTaskSubstrate.expand`) and re-assesses the carried population against it. Every
`checkpoint_every` generations it writes a `gen_<NNNNNN>/` directory (model, stats, net, speciation,
checkpoint) under one run dir, and `--resume <run_dir>` continues from the latest one.
"""

import datetime
import random
from collections import Counter
from pathlib import Path
from typing import Any, cast

import torch

from ardevo import checkpoint, results
from ardevo.evolution.evolver import Assessed, Evolver, EvolverState
from ardevo.evolution.genome import InnovationTracker, NodeKind, genome_from_dict, genome_to_dict
from ardevo.evolution.multitask import MultiTaskAdapter, MultiTaskSubstrate, TaskEntry, build_pool
from ardevo.evolution.registry import build_evolver
from ardevo.evolution.schedule import build_schedule
from ardevo.utils.logging import Logger
from ardevo.utils.proctor import Proctor

logger = Logger.get_logger()
console = Logger.get_console()


class ContinuousTrial(Proctor):
    """Evolve a single growing topology challenged by randomly-interleaved tasks across rungs."""

    def __init__(self, config: dict[str, Any], task: Any = None) -> None:
        super().__init__(config=config, task=task)
        torch.manual_seed(int(config.get("seed", 0)))

        schedule_cfg = config.get("schedule", {})
        rungs_cfg = schedule_cfg.get("rungs", [1, 2, 3, 4, 5])
        self.rungs = list(range(1, 19)) if rungs_cfg == "all" else [int(rung) for rung in rungs_cfg]
        self.generations_per_task = int(schedule_cfg.get("generations_per_task", 40))
        self.checkpoint_every = int(schedule_cfg.get("checkpoint_every", 100))
        self.total_generations = int(config.get("generations", 800))
        self.weight_scale = float(config.get("evolution", {}).get("init", {}).get("weight_scale", 1.0))

        if not any(component.endswith("_penalty") for component in config.get("fitness", {}).get("components", [])):
            raise ValueError("continuous run requires a complexity/hidden penalty fitness component so the shared topology cannot grow unbounded")

        self.pool: list[TaskEntry] = build_pool(
            source=config["dataset"],
            rungs=self.rungs,
            n_samples=int(config["n_samples"]),
            support_fraction=float(config.get("support_fraction", 0.8)),
            tasks_per_rung=int(schedule_cfg.get("tasks_per_rung", 100)),
            shuffle=bool(schedule_cfg.get("shuffle", True)),
            seed=int(config.get("seed", 0)),
        )
        if not self.pool:
            raise RuntimeError(f"no tasks found for rungs {self.rungs} in {config['dataset']!r}")

        self.evolver: Evolver = build_evolver(config)
        self.scheduler = build_schedule(schedule_cfg)
        self.substrate = MultiTaskSubstrate(default_activation=config.get("substrate", {}).get("default_activation", "tanh"))
        self.resume_dir = config.get("resume")
        self.run_dir = Path(results.DEFAULT_ROOT)
        self.history: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        if self.resume_dir:
            self.run_dir = Path(self.resume_dir)
            state, active_index, adapter = self._restore()
            console.rule(f"[bold]Resuming continuous run at gen {state.generation} ({self.run_dir})")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path(results.DEFAULT_ROOT) / f"{timestamp}_continuous"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            state, active_index, adapter = self._fresh_start()
            console.rule(f"[bold]Continuous run: rungs {self.rungs}, {len(self.pool)} tasks -> {self.run_dir}")

        active_entry = self.pool[active_index]
        while state.generation < self.total_generations:
            block_end = min(state.generation + self.generations_per_task, self.total_generations)
            while state.generation < block_end:
                self._log_generation(state, active_entry)
                self.evolver.advance(state, adapter)
                if state.generation % self.checkpoint_every == 0:
                    self._checkpoint(state, active_index)
            if state.generation >= self.total_generations:
                break
            active_index = self.scheduler.next_index(self.pool, state.rng)
            active_entry = self.pool[active_index]
            adapter = self._switch(active_entry, state)

        if state.generation % self.checkpoint_every != 0:  # avoid a duplicate when the last gen already checkpointed
            self._checkpoint(state, active_index)
        champion = max(state.population, key=lambda item: item.fitness)
        self.results = {
            "rungs": self.rungs,
            "pool_size": len(self.pool),
            "generations_run": state.generation,
            "n_inputs": self.substrate.n_inputs,
            "n_outputs": self.substrate.n_outputs,
            "champion_complexity": champion.genome.complexity(),
            "run_dir": str(self.run_dir),
        }
        complexity = champion.genome.complexity()
        console.print(f"[bold green]Done[/bold green]: {state.generation} gens, io {self.substrate.n_inputs}->{self.substrate.n_outputs}, complexity {complexity}")
        self.finalize()
        return self.results

    def _fresh_start(self) -> tuple[EvolverState, int, MultiTaskAdapter]:
        rng = random.Random(int(self.config.get("seed", 0)))
        tracker = InnovationTracker(_next_node_id=0)
        active_index = self.scheduler.next_index(self.pool, rng)
        entry = self.pool[active_index]
        genomes = self.substrate.seed(entry, tracker, rng, self.evolver.pop_size, self.weight_scale)
        state = EvolverState(population=[], innovations=tracker, rng=rng)
        adapter = self.substrate.adapter(entry)
        state.population = [self.evolver.assess(genome, adapter, state) for genome in genomes]
        return state, active_index, adapter

    def _restore(self) -> tuple[EvolverState, int, MultiTaskAdapter]:
        checkpoint_dir = checkpoint.latest_checkpoint_dir(self.run_dir)
        if checkpoint_dir is None:
            raise FileNotFoundError(f"no checkpoint found under {self.run_dir}")
        data = checkpoint.read_checkpoint(checkpoint_dir)
        default_activation = self.config.get("substrate", {}).get("default_activation", "tanh")
        self.substrate = MultiTaskSubstrate.from_dict(data["substrate"], default_activation)
        self.scheduler.load_state_dict(data["schedule"])
        cast(Any, self.evolver.speciate).load_state_dict(data["speciation"])
        tracker = InnovationTracker.from_dict(data["innovations"])
        rng = checkpoint.deserialize_rng(data["rng"])
        state = EvolverState(
            population=[],
            innovations=tracker,
            rng=rng,
            generation=int(data["generation"]),
            species_history=checkpoint.restored_species_history(data),
        )
        active_index = int(data["active_index"])
        adapter = self.substrate.adapter(self.pool[active_index])
        state.population = [self.evolver.evaluate_only(genome_from_dict(genome), adapter) for genome in data["population"]]
        return state, active_index, adapter

    def _switch(self, entry: TaskEntry, state: EvolverState) -> MultiTaskAdapter:
        """Grow the interface for `entry` and re-assess (re-train) the carried population on it."""
        grown = self.substrate.expand(entry, [item.genome for item in state.population], state.innovations, state.rng)
        adapter = self.substrate.adapter(entry)
        state.population = [self.evolver.assess(genome, adapter, state) for genome in grown]
        return adapter

    def _log_generation(self, state: EvolverState, entry: TaskEntry) -> None:
        best = max(state.population, key=lambda item: item.fitness)
        mean = sum(item.fitness for item in state.population) / len(state.population)
        hidden = len(best.genome.hidden_ids)
        edges = len(best.genome.enabled_connections())
        self.history.append(
            {
                "generation": state.generation,
                "rung": entry.rung,
                "task": entry.name,
                "best_fitness": best.fitness,
                "mean_fitness": mean,
                "query_accuracy": best.metrics["query_accuracy"],
                "support_accuracy": best.metrics.get("support_accuracy", 0.0),
                "support_loss": best.metrics.get("support_loss", 0.0),
                "hidden_nodes": hidden,
                "enabled_edges": edges,
            }
        )
        self.log_scalar("Fitness", "best", best.fitness, state.generation)
        self.log_scalar("Accuracy", "support", best.metrics.get("support_accuracy", 0.0), state.generation)
        self.log_scalar("Loss", "support", best.metrics.get("support_loss", 0.0), state.generation)
        self.log_scalar("Complexity", "hidden_nodes", hidden, state.generation)
        self.log_scalar("Complexity", "enabled_edges", edges, state.generation)
        if state.generation % 10 == 0:
            self.log_hardware_stats(state.generation)
            console.print(
                f"[cyan]gen {state.generation:5d}[/cyan] rung {entry.rung} {entry.name[:16]:<16} "
                f"fit={best.fitness:6.3f} s_acc={best.metrics.get('support_accuracy', 0.0):4.2f} s_loss={best.metrics.get('support_loss', 0.0):6.3f} "
                f"hidden={hidden} edges={edges} io={self.substrate.n_inputs}->{self.substrate.n_outputs}"
            )

    def _checkpoint(self, state: EvolverState, active_index: int) -> None:
        champion = max(state.population, key=lambda item: item.fitness)
        state.best = champion
        directory = self.run_dir / f"gen_{state.generation:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        per_rung = self._champion_across_rungs(champion)

        results.write_stats(directory, self._stats(state, champion, per_rung))
        results.write_model(directory, self._champion_model(champion))
        hidden = len(champion.genome.hidden_ids)
        edges = len(champion.genome.enabled_connections())
        title = f"continuous gen {state.generation}: {hidden} hidden / {edges} edges, io {self.substrate.n_inputs}->{self.substrate.n_outputs}"
        results.render_network(directory, champion.genome, title=title)
        results.render_speciation(directory, state.species_history, title=f"species over {len(state.species_history)} generations")
        checkpoint.write_checkpoint(
            directory, checkpoint.build_payload(state=state, speciator=self.evolver.speciate, scheduler=self.scheduler, substrate=self.substrate, active_index=active_index)
        )

        if self.task:
            for name in ("stats.json", "model.json", "net.png", "speciation.png", "checkpoint.json"):
                self.save_artifact(f"gen{state.generation:06d}_{name}", str(directory / name))
        console.print(f"[green]checkpoint[/green] {directory}  ({self._summary(per_rung)})")

    def _champion_across_rungs(self, champion: Assessed) -> dict[str, dict[str, float]]:
        """Score the current champion on EVERY task whose head has been grown (the cross-rung signal)."""
        per_rung: dict[str, dict[str, float]] = {}
        for entry in self.pool:
            if entry.name not in self.substrate.heads:
                continue
            metrics = self.evolver.evaluate_only(champion.genome, self.substrate.adapter(entry)).metrics
            per_rung[entry.name] = {
                "rung": float(entry.rung),
                "query_accuracy": metrics["query_accuracy"],
                "support_accuracy": metrics["support_accuracy"],
                "support_loss": metrics["support_loss"],
            }
        return per_rung

    @staticmethod
    def _summary(per_rung: dict[str, dict[str, float]]) -> str:
        if not per_rung:
            return "no rungs scored yet"
        accuracies = [entry["query_accuracy"] for entry in per_rung.values()]
        return f"champion mean query_acc {sum(accuracies) / len(accuracies):.2f} over {len(accuracies)} seen tasks"

    def _champion_model(self, champion: Assessed) -> dict[str, Any]:
        tuned = champion.module.export_weights()
        genome = genome_to_dict(champion.genome)
        for connection in genome["connections"]:
            edge = (connection["in"], connection["out"])
            if connection["enabled"] and edge in tuned:
                connection["weight"] = tuned[edge]
        return {
            "n_inputs": self.substrate.n_inputs,
            "n_outputs": self.substrate.n_outputs,
            "heads": {name: head.node_ids for name, head in self.substrate.heads.items()},
            "genome": genome,
        }

    def _stats(self, state: EvolverState, champion: Assessed, per_rung: dict[str, dict[str, float]]) -> dict[str, Any]:
        genome = champion.genome
        return {
            "generation": state.generation,
            "rungs": self.rungs,
            "source": self.config.get("dataset"),
            "seed": int(self.config.get("seed", 0)),
            "pool_size": len(self.pool),
            "champion": {
                "fitness": champion.fitness,
                "total_nodes": len(genome.nodes),
                "hidden_nodes": len(genome.hidden_ids),
                "enabled_edges": len(genome.enabled_connections()),
                "complexity": genome.complexity(),
                "activations": dict(Counter(node.activation for node in genome.nodes.values() if node.kind in (NodeKind.HIDDEN, NodeKind.OUTPUT))),
            },
            "per_rung_champion": per_rung,
            "interface": {
                "n_inputs": self.substrate.n_inputs,
                "n_outputs": self.substrate.n_outputs,
                "banks": {signature: bank.width for signature, bank in self.substrate.banks.items()},
                "heads": list(self.substrate.heads),
            },
            "history": self.history,
            "speciation": {
                "generations": len(state.species_history),
                "max_concurrent_species": max((len(snapshot) for snapshot in state.species_history), default=0),
                "total_species_seen": len({species_id for snapshot in state.species_history for species_id in snapshot}),
            },
            "config": {
                "schedule": self.config.get("schedule", {}),
                "evolution": self.config.get("evolution", {}),
                "fitness": self.config.get("fitness", {}),
                "dataset": {"source": self.config.get("dataset"), "n_samples": self.config.get("n_samples"), "support_fraction": self.config.get("support_fraction")},
            },
        }
