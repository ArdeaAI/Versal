"""Per-run local results: durable JSON/PNG records under `./results/<run>/`.

Pure IO/visualization helpers (no trial or ClearML coupling): `write_stats` dumps a JSON stats
record, `render_speciation` charts species populations over generations. matplotlib is imported
inside `render_speciation` to keep this module light and to set the headless `Agg` backend before
pyplot loads.
"""

import json
from pathlib import Path
from typing import Any

from versal.rendering import THEME

DEFAULT_ROOT = "results"


def write_stats(directory: Path, stats: dict[str, Any]) -> Path:
    path = directory / "stats.json"
    path.write_text(json.dumps(stats, indent=2))
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
