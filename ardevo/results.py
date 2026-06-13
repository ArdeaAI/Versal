"""Per-run local results: write a durable record of a run to `./results/<name>/`.

Every run leaves a directory `results/<YYYYMMDD_HHMMSS>_fit-<f>_acc-<a>_loss-<l>/` holding:
- `stats.json`: run metadata, champion metrics, per-generation history, config snapshot
- `model.json`: the champion genome (topology + scored weights), reloadable via `genome_from_dict`
- `net.png`: the champion topology, rendered by `ardevo.rendering` (recursive, dark)

These functions are pure IO/visualization (no trial or ClearML coupling) so they are easy to test
and reuse. matplotlib is imported inside `render_speciation` to keep this module light and to set
the headless `Agg` backend before pyplot loads.
"""

import json
from pathlib import Path
from typing import Any

from ardevo.rendering import THEME

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


def render_speciation(directory: Path, species_history: list[dict[int, int]], *, title: str) -> Path:
    """Stacked-area chart of each species' population over generations (births and deaths over time)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = directory / "speciation.png"
    figure, axis = plt.subplots(figsize=(11, 6))
    figure.patch.set_facecolor(THEME["background"])
    axis.set_facecolor(THEME["background"])

    if species_history:
        generations = list(range(len(species_history)))
        species_ids = sorted({species_id for snapshot in species_history for species_id in snapshot})
        # One band per species, in birth order, zero where the species is absent (before birth / after death).
        bands = [[snapshot.get(species_id, 0) for snapshot in species_history] for species_id in species_ids]
        cmap = plt.get_cmap("viridis")
        colors = [cmap(index / max(len(species_ids) - 1, 1)) for index in range(len(species_ids))]
        axis.stackplot(generations, *bands, colors=colors, edgecolor=THEME["background"], linewidth=0.2)
        axis.set_xlim((0, max(generations)) if max(generations) > 0 else (-0.5, 0.5))
        axis.set_xlabel("generation", color=THEME["label"])
        axis.set_ylabel("population by species", color=THEME["label"])
        axis.tick_params(colors=THEME["label"])
        for spine in axis.spines.values():
            spine.set_color(THEME["container_edge"])
    else:
        axis.text(0.5, 0.5, "no speciation history", ha="center", va="center", color=THEME["label"])
        axis.axis("off")

    axis.set_title(title, fontsize=11, color=THEME["title"])
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return path
