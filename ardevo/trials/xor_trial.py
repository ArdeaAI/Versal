"""EvolutionTrial: run topology evolution on one Icarus rung and report through Proctor.

The trial owns the task (it loads its own Icarus data, so `require_dataloaders` is moot), builds
the encoder + task adapter, assembles the evolver from config, and drives the generational loop.
Per-generation metrics go to ClearML (when enabled) and to the Rich console; the best genome is
saved as an artifact.
"""

from typing import Any

import torch

from ardevo.dataset.icarus import Level0Encoder, support_loader
from ardevo.dataset.loader import load_rung_task
from ardevo.evaluation import encode, input_width, output_features
from ardevo.evolution.evolver import Assessed, Evolver, TaskAdapter
from ardevo.evolution.genome import Genome
from ardevo.evolution.registry import build_evolver
from ardevo.utils.logging import Logger
from ardevo.utils.proctor import Proctor

logger = Logger.get_logger()
console = Logger.get_console()


def _genome_to_dict(genome: Genome) -> dict[str, Any]:
    return {
        "nodes": [{"id": n.id, "kind": n.kind.value, "activation": n.activation} for n in genome.nodes.values()],
        "connections": [{"in": c.in_id, "out": c.out_id, "weight": c.weight, "enabled": c.enabled, "innovation": c.innovation} for c in genome.connections],
    }


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

    def _on_generation(self, generation: int, best: Assessed, mean_fitness: float) -> None:
        accuracy = best.metrics["query_accuracy"]
        edges = len(best.genome.enabled_connections())
        hidden = len(best.genome.hidden_ids)

        self.log_scalar("Fitness", "best", best.fitness, generation)
        self.log_scalar("Fitness", "mean", mean_fitness, generation)
        self.log_scalar("Accuracy", "query", accuracy, generation)
        self.log_scalar("Loss", "query", best.metrics["query_loss"], generation)
        self.log_scalar("Complexity", "enabled_edges", edges, generation)
        self.log_scalar("Complexity", "hidden_nodes", hidden, generation)
        if generation % 10 == 0:
            self.log_hardware_stats(generation)
            console.print(
                f"[cyan]gen {generation:4d}[/cyan]  fitness={best.fitness:6.3f}  acc={accuracy:4.2f}  loss={best.metrics['query_loss']:6.3f}  hidden={hidden}  edges={edges}"
            )

    def run(self) -> dict[str, Any]:
        console.rule(f"[bold]Evolving {self.icarus_task.meta.name!r} (rung {self.icarus_task.meta.rung})")
        best = self.evolver.run(
            self.adapter,
            generations=self.generations,
            on_generation=self._on_generation,
        )

        self.results = {
            "task": self.icarus_task.meta.name,
            "rung": self.icarus_task.meta.rung,
            "best_fitness": best.fitness,
            "query_accuracy": best.metrics["query_accuracy"],
            "query_loss": best.metrics["query_loss"],
            "hidden_nodes": len(best.genome.hidden_ids),
            "enabled_edges": len(best.genome.enabled_connections()),
            "complexity": best.genome.complexity(),
            "genome": _genome_to_dict(best.genome),
        }
        console.print(f"[bold green]Done[/bold green]: acc={self.results['query_accuracy']:.2f}  hidden={self.results['hidden_nodes']}  complexity={self.results['complexity']}")
        self.save_artifact("best_genome", self.results["genome"])
        self.finalize()
        return self.results
