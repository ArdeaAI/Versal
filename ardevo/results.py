"""Per-run local results: write a durable record of a run to `./results/<name>/`.

Every run leaves a directory `results/<YYYYMMDD_HHMMSS>_fit-<f>_acc-<a>_loss-<l>/` holding:
- `stats.json`: run metadata, champion metrics, per-generation history, config snapshot
- `model.json`: the champion genome (topology + scored weights), reloadable via `genome_from_dict`
- `net.png`: a networkx render of the champion topology

These functions are pure IO/visualization (no trial or ClearML coupling) so they are easy to test
and reuse. matplotlib/networkx are imported inside `render_network` to keep this module light and to
set the headless `Agg` backend before pyplot loads.
"""

import json
from pathlib import Path
from typing import Any

from ardevo.evolution.genome import Genome, topological_order

DEFAULT_ROOT = "results"


def run_directory(timestamp: str, fitness: float, accuracy: float, loss: float, root: str = DEFAULT_ROOT) -> Path:
    """Create and return `<root>/<timestamp>_fit-<f>_acc-<a>_loss-<l>/`."""
    name = f"{timestamp}_fit-{fitness:.3f}_acc-{accuracy:.3f}_loss-{loss:.3f}"
    path = Path(root) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_stats(directory: Path, stats: dict[str, Any]) -> Path:
    path = directory / "stats.json"
    path.write_text(json.dumps(stats, indent=2))
    return path


def write_model(directory: Path, model: dict[str, Any]) -> Path:
    path = directory / "model.json"
    path.write_text(json.dumps(model, indent=2))
    return path


def _node_layers(genome: Genome) -> dict[int, int]:
    """Longest-path topological depth per node (inputs/bias at 0). Used for layout + color."""
    incoming: dict[int, list[int]] = {}
    for conn in genome.enabled_connections():
        incoming.setdefault(conn.out_id, []).append(conn.in_id)
    layer: dict[int, int] = {}
    for node_id in topological_order(genome):
        predecessors = incoming.get(node_id, [])
        layer[node_id] = 0 if not predecessors else 1 + max(layer[pred] for pred in predecessors)
    return layer


def render_network(directory: Path, genome: Genome, *, title: str) -> Path:
    """Render the champion topology to `net.png` (viridis nodes by depth, size by degree)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    layers = _node_layers(genome)
    graph: nx.DiGraph = nx.DiGraph()
    for node_id in genome.nodes:
        graph.add_node(node_id, layer=layers.get(node_id, 0))
    for conn in genome.enabled_connections():
        graph.add_edge(conn.in_id, conn.out_id, weight=conn.weight)

    position = nx.multipartite_layout(graph, subset_key="layer")
    degrees = dict(graph.degree())
    node_sizes = [140 + 220 * degrees.get(node_id, 0) for node_id in graph.nodes()]
    node_colors = [layers.get(node_id, 0) for node_id in graph.nodes()]
    max_layer = max(layers.values(), default=1) or 1
    edge_widths = [0.6 + 2.4 * min(abs(data["weight"]), 3.0) / 3.0 for _u, _v, data in graph.edges(data=True)]

    figure, axis = plt.subplots(figsize=(10, 7))
    figure.patch.set_facecolor("#e8e8e8")
    axis.set_facecolor("#e8e8e8")
    nx.draw_networkx_edges(graph, position, ax=axis, edge_color="black", width=edge_widths, alpha=0.5, arrows=False)
    nx.draw_networkx_nodes(graph, position, ax=axis, node_size=node_sizes, node_color=node_colors, cmap=plt.get_cmap("viridis"), vmin=0, vmax=max_layer, linewidths=0.0)
    axis.set_title(title, fontsize=11)
    axis.axis("off")
    figure.tight_layout()

    path = directory / "net.png"
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return path
