"""EvolutionTrial: run topology evolution on one Icarus rung and report through Proctor.

The trial owns the task (it loads its own Icarus data), builds the encoder + task adapter, assembles
the evolver from config, and drives the generational loop. Per-generation metrics go to ClearML (when
enabled) and the Rich console; every run also writes a durable local record (stats, model, network
image) under ./results, which is additionally uploaded to ClearML when a task is active.
"""

import datetime
from collections import Counter
from typing import Any

import torch

from ardevo import results
from ardevo.dataset.icarus import Level0Encoder, support_loader
from ardevo.dataset.loader import load_rung_task
from ardevo.evaluation import encode, input_width, output_features
from ardevo.evolution.evolver import Assessed, Evolver, TaskAdapter
from ardevo.evolution.genome import NodeKind, genome_to_dict
from ardevo.evolution.registry import build_evolver
from ardevo.utils.logging import Logger
from ardevo.utils.proctor import Proctor

logger = Logger.get_logger()
console = Logger.get_console()


class EvolutionTrial(Proctor):
    """Evolve a topology to solve a single rung's task."""

    def __init__(self, config: dict[str, Any], task: Any = None) -> None:
        super().__init__(config=config, task=task)
        torch.manual_seed(int(config.get("seed", 0)))

        self.icarus_task = load_rung_task(
            source=config["dataset"],
            rung=int(config["rung"]),
            n_samples=int(config["n_samples"]),
            seed=int(config.get("seed", 0)),
            support_fraction=float(config.get("support_fraction", 0.8)),
        )
        # Size the encoder to the task's natural input width (no padding/truncation).
        support_input, _support_output = support_loader(self.icarus_task)
        natural_width = int(support_input.data.reshape(support_input.data.shape[0], -1).shape[1])
        self.encoder = Level0Encoder(max_flat_dim=natural_width)
        self.encoded = encode(self.icarus_task, self.encoder)
        self.adapter = TaskAdapter(
            encoded=self.encoded,
            encoder=self.encoder,
            n_inputs=input_width(self.encoded),
            n_outputs=output_features(self.encoded),
        )
        self.evolver: Evolver = build_evolver(config)
        self.generations = int(config.get("generations", 100))
        self.history: list[dict[str, float]] = []

    def _on_generation(self, generation: int, best: Assessed, mean_fitness: float) -> None:
        accuracy = best.metrics["query_accuracy"]
        support = best.metrics.get("support_accuracy", 0.0)
        edges = len(best.genome.enabled_connections())
        hidden = len(best.genome.hidden_ids)

        self.history.append(
            {
                "generation": generation,
                "best_fitness": best.fitness,
                "mean_fitness": mean_fitness,
                "support_accuracy": support,
                "query_accuracy": accuracy,
                "query_loss": best.metrics["query_loss"],
                "hidden_nodes": hidden,
                "enabled_edges": edges,
            }
        )

        self.log_scalar("Fitness", "best", best.fitness, generation)
        self.log_scalar("Fitness", "mean", mean_fitness, generation)
        self.log_scalar("Accuracy", "query", accuracy, generation)
        self.log_scalar("Loss", "query", best.metrics["query_loss"], generation)
        self.log_scalar("Complexity", "enabled_edges", edges, generation)
        self.log_scalar("Complexity", "hidden_nodes", hidden, generation)
        if generation % 10 == 0:
            self.log_hardware_stats(generation)
            console.print(f"[cyan]gen {generation:4d}[/cyan]  fit={best.fitness:6.3f}  support_acc={support:4.2f}  query_acc={accuracy:4.2f}  hidden={hidden}  edges={edges}")

    def _champion_model(self, best: Assessed) -> dict[str, Any]:
        """The champion genome dict with enabled weights taken from the exact scored network."""
        tuned = best.module.export_weights()
        genome = genome_to_dict(best.genome)
        for connection in genome["connections"]:
            edge = (connection["in"], connection["out"])
            if connection["enabled"] and edge in tuned:
                connection["weight"] = tuned[edge]
        return {
            "task": self.icarus_task.meta.name,
            "rung": self.icarus_task.meta.rung,
            "n_inputs": self.adapter.n_inputs,
            "n_outputs": self.adapter.n_outputs,
            "genome": genome,
        }

    def _stats(self, best: Assessed, timestamp: str) -> dict[str, Any]:
        genome = best.genome
        return {
            "timestamp": timestamp,
            "task": self.icarus_task.meta.name,
            "rung": self.icarus_task.meta.rung,
            "source": self.config.get("dataset"),
            "seed": int(self.config.get("seed", 0)),
            "generations_run": len(self.history),
            "champion": {
                "fitness": best.fitness,
                "support_accuracy": best.metrics.get("support_accuracy", 0.0),
                "support_loss": best.metrics.get("support_loss", 0.0),
                "query_accuracy": best.metrics["query_accuracy"],
                "query_loss": best.metrics["query_loss"],
                "total_nodes": len(genome.nodes),
                "hidden_nodes": len(genome.hidden_ids),
                "enabled_edges": len(genome.enabled_connections()),
                "complexity": genome.complexity(),
                "nodes_by_kind": dict(Counter(node.kind.value for node in genome.nodes.values())),
                "activations": dict(Counter(node.activation for node in genome.nodes.values() if node.kind in (NodeKind.HIDDEN, NodeKind.OUTPUT))),
            },
            "history": self.history,
            "speciation": {
                "generations": len(self.evolver.species_history),
                "max_concurrent_species": max((len(snapshot) for snapshot in self.evolver.species_history), default=0),
                "total_species_seen": len({species_id for snapshot in self.evolver.species_history for species_id in snapshot}),
                "history": self.evolver.species_history,
            },
            "config": {
                "seed": int(self.config.get("seed", 0)),
                "generations": self.generations,
                "dataset": {"source": self.config.get("dataset"), "rung": self.config.get("rung"), "n_samples": self.config.get("n_samples")},
                "substrate": self.config.get("substrate", {}),
                "evolution": self.config.get("evolution", {}),
                "fitness": self.config.get("fitness", {}),
            },
        }

    def _save_results(self, best: Assessed, timestamp: str) -> str:
        accuracy = best.metrics["query_accuracy"]
        loss = best.metrics["query_loss"]
        directory = results.run_directory(timestamp, best.fitness, accuracy, loss)

        stats_path = results.write_stats(directory, self._stats(best, timestamp))
        model_path = results.write_model(directory, self._champion_model(best))
        title = (
            f"{self.icarus_task.meta.name} (rung {self.icarus_task.meta.rung})  "
            f"acc={accuracy:.2f} fit={best.fitness:.3f}  {len(best.genome.hidden_ids)} hidden / {len(best.genome.enabled_connections())} edges"
        )
        net_path = results.render_network(directory, best.genome, title=title)
        species_title = f"{self.icarus_task.meta.name} (rung {self.icarus_task.meta.rung}) species over {len(self.evolver.species_history)} generations"
        species_path = results.render_speciation(directory, self.evolver.species_history, title=species_title)

        if self.task:
            self.save_artifact("run_stats", str(stats_path))
            self.save_artifact("best_genome", str(model_path))
            self.save_artifact("network_image", str(net_path))
            self.save_artifact("speciation_image", str(species_path))

        console.print(f"[green]Results saved to[/green] {directory}")
        return str(directory)

    def run(self) -> dict[str, Any]:
        console.rule(f"[bold]Evolving {self.icarus_task.meta.name!r} (rung {self.icarus_task.meta.rung})")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best = self.evolver.run(self.adapter, generations=self.generations, on_generation=self._on_generation)

        results_dir = self._save_results(best, timestamp)
        self.results = {
            "task": self.icarus_task.meta.name,
            "rung": self.icarus_task.meta.rung,
            "best_fitness": best.fitness,
            "query_accuracy": best.metrics["query_accuracy"],
            "query_loss": best.metrics["query_loss"],
            "hidden_nodes": len(best.genome.hidden_ids),
            "enabled_edges": len(best.genome.enabled_connections()),
            "complexity": best.genome.complexity(),
            "results_dir": results_dir,
        }
        console.print(f"[bold green]Done[/bold green]: acc={self.results['query_accuracy']:.2f}  hidden={self.results['hidden_nodes']}  complexity={self.results['complexity']}")
        self.finalize()
        return self.results
