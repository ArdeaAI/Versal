# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.9"]
# ///
"""Build the Apart paper's method diagram and cold overmind portrait.

Run from the repository root:

    uv run python ai/for_apart/make_system_figures.py
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ai" / "for_apart" / "figures"
DEFAULT_LIBRARY = REPO_ROOT / "library_canary_clean_seed0"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "canary.toml"

INK = "#172033"
MUTED = "#5D687A"
LINE = "#465268"
BLUE = "#2563A7"
BLUE_LIGHT = "#EAF2FA"
ORANGE = "#C65D13"
ORANGE_LIGHT = "#FFF3E8"
GREEN = "#2D7A4D"
GREEN_LIGHT = "#EAF6EF"
STONE = "#89511D"
PANEL = "#F7F8FA"
WHITE = "#FFFFFF"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.linewidth": 0.7,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    fill: str = WHITE,
    edge: str = LINE,
    size: float = 6.7,
    weight: str = "normal",
    linewidth: float = 0.9,
    radius: float = 0.6,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.20,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=linewidth,
        )
    )
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        linespacing=1.18,
    )


def arrow(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    linewidth: float = 1.0,
    curve: float = 0.0,
    dashed: bool = False,
    head: bool = True,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={curve}",
            arrowstyle="-|>" if head else "-",
            mutation_scale=8,
            color=color,
            linewidth=linewidth,
            linestyle=(0, (3, 2)) if dashed else "solid",
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def lane(axis: Any, y: float, height: float, label: str, *, fill: str, color: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (0.5, y),
            99.0,
            height,
            boxstyle="round,pad=0.0,rounding_size=0.8",
            facecolor=fill,
            edgecolor="none",
            linewidth=0,
        )
    )
    axis.text(1.4, y + height - 1.2, label, ha="left", va="top", fontsize=6.2, fontweight="bold", color=color)


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(png_path, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"wrote {display_path(pdf_path)}")
    print(f"wrote {display_path(png_path)}")


def make_lifecycle(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.15))
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 60)
    axis.axis("off")

    lane(axis, 25.0, 34.0, "SUPPORT-ONLY SEARCH AND ADMISSION", fill=PANEL, color=MUTED)
    lane(axis, 11.2, 11.0, "BLIND HELD-OUT REPORTING", fill=ORANGE_LIGHT, color=ORANGE)
    lane(axis, 0.5, 7.8, "PERSISTENT MEMORY", fill=BLUE_LIGHT, color=BLUE)

    box(axis, 2.4, 37.2, 9.7, 10.0, "task\ncontract", weight="bold", size=7.1)
    axis.text(13.0, 44.3, "support", fontsize=6.1, color=MUTED, va="center")
    arrow(axis, (12.1, 42.2), (16.0, 42.2))

    box(axis, 16.0, 37.2, 13.4, 10.0, "library lookup\n+ bounded refinement", fill=BLUE_LIGHT, edge=BLUE)
    arrow(axis, (29.4, 42.2), (33.0, 42.2))
    axis.text(31.2, 43.2, "miss", fontsize=5.8, color=MUTED, ha="center")

    box(axis, 33.0, 29.0, 25.0, 26.0, "", fill=WHITE, edge=LINE, linewidth=1.0)
    axis.text(45.5, 52.8, "structural search ladder", ha="center", va="center", fontsize=7.0, fontweight="bold")
    strategy_rows = [
        "route frozen experts",
        "induce grammar programs",
        "evolve spatial field",
        "evolve dense network",
        "compose library modules",
    ]
    for index, text in enumerate(strategy_rows):
        y = 47.5 - index * 4.0
        box(axis, 35.0, y, 21.0, 3.1, text, fill=PANEL, edge="#7B8494", size=6.1, radius=0.4)
        if index < len(strategy_rows) - 1:
            arrow(axis, (45.5, y - 0.1), (45.5, y - 0.9), linewidth=0.7)

    box(axis, 60.8, 29.3, 9.7, 8.7, "decompose\n+ recurse", fill=PANEL, edge=LINE, size=6.4)
    arrow(axis, (58.0, 33.5), (60.8, 33.5))
    axis.text(59.4, 34.5, "stall", fontsize=5.7, color=MUTED, ha="center")
    axis.plot((65.7, 65.7, 29.4), (29.3, 26.6, 26.6), color=LINE, linewidth=1.0, linestyle=(0, (3, 2)))
    arrow(axis, (29.4, 26.6), (29.4, 37.2), dashed=True)
    axis.text(49.5, 25.6, "typed subtasks re-enter the same loop", fontsize=5.8, color=MUTED, ha="center")

    box(axis, 72.0, 37.2, 11.6, 10.0, "best executable\nchampion\nsurvives deadline", fill=ORANGE_LIGHT, edge=ORANGE, weight="bold", size=6.3, linewidth=1.1)
    axis.plot((22.7, 22.7, 77.8), (47.2, 56.2, 56.2), color=BLUE, linewidth=1.0)
    arrow(axis, (77.8, 56.2), (77.8, 47.2), color=BLUE)
    axis.text(62.5, 57.1, "verified hit or refinement", fontsize=5.8, color=BLUE, ha="center")
    arrow(axis, (58.0, 42.2), (72.0, 42.2))

    box(axis, 87.0, 37.2, 10.6, 10.0, "support-fold\nadmission gate", fill=GREEN_LIGHT, edge=GREEN, size=6.5)
    arrow(axis, (83.6, 42.2), (87.0, 42.2))

    box(axis, 16.0, 13.6, 13.4, 5.9, "sealed held-out query", fill=WHITE, edge=ORANGE, size=6.3)
    arrow(axis, (7.3, 37.2), (16.0, 16.5), color=ORANGE, curve=0.18)
    axis.text(8.5, 23.2, "query", fontsize=5.8, color=ORANGE, ha="center", rotation=-62)

    box(axis, 72.0, 13.6, 11.6, 5.9, "evaluate once\nif executable", fill=WHITE, edge=ORANGE, size=6.3)
    arrow(axis, (29.4, 16.5), (72.0, 16.5), color=ORANGE)
    arrow(axis, (77.8, 37.2), (77.8, 19.5), color=ORANGE)
    box(axis, 87.0, 13.6, 10.6, 5.9, "reported result", fill=WHITE, edge=ORANGE, weight="bold", size=6.4)
    arrow(axis, (83.6, 16.5), (87.0, 16.5), color=ORANGE)
    axis.text(
        50.5,
        14.0,
        "never feeds search or admission",
        fontsize=5.8,
        color=ORANGE,
        ha="center",
        bbox={"facecolor": ORANGE_LIGHT, "edgecolor": "none", "pad": 0.5},
    )

    box(
        axis,
        16.0,
        2.0,
        81.6,
        4.8,
        "immutable modules and compositions  ·  structural I/O index  ·  provenance and lifecycle",
        fill=BLUE_LIGHT,
        edge=BLUE,
        size=6.4,
        linewidth=1.1,
    )
    axis.plot((31.2, 31.2), (6.8, 34.1), color=BLUE, linewidth=1.0)
    arrow(axis, (31.2, 34.1), (27.8, 37.2), color=BLUE)
    axis.text(30.2, 23.4, "candidates\nand seeds", fontsize=5.8, color=BLUE, ha="right", va="center")
    arrow(axis, (52.0, 6.8), (52.0, 29.0), color=BLUE)
    axis.text(52.0, 23.6, "experts · macros · modules", fontsize=5.8, color=BLUE, ha="center", va="center")
    axis.plot((92.3, 98.4, 98.4), (37.2, 37.2, 8.3), color=GREEN, linewidth=1.0)
    arrow(axis, (98.4, 8.3), (96.5, 6.8), color=GREEN)
    axis.text(97.8, 27.0, "accepted", fontsize=5.8, color=GREEN, ha="right", rotation=90, va="center")
    axis.plot((82.2, 84.7, 84.7), (37.2, 34.5, 8.4), color=STONE, linewidth=1.0, linestyle=(0, (3, 2)))
    arrow(axis, (84.7, 8.4), (82.5, 6.8), color=STONE, dashed=True)
    axis.text(85.4, 27.0, "eligible stepping stone", fontsize=5.7, color=STONE, ha="left", rotation=90, va="center")

    figure.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002)
    save_figure(figure, output_dir, "system_lifecycle")


def make_cold_overmind(library_root: Path, config_path: Path, output_dir: Path, *, seed: int) -> None:
    import numpy as np
    import torch

    from versal.library import ModuleLibrary
    from versal.reference_depth import configured_max_inline_depth
    from versal.routing import RouterService
    from versal.utils.config import Config

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    runtime = Config(conf_path=str(config_path)).current
    routed = runtime.get("orchestrator", {}).get("routed", {})
    library = ModuleLibrary(library_root)
    service = RouterService(
        library,
        d_model=int(routed.get("d_model", 64)),
        top_k=int(routed.get("top_k", 2)),
        max_steps=int(routed.get("max_steps", 4)),
        adapter_rank=int(routed.get("adapter_rank", 0)),
        halting=bool(routed.get("halting", False)),
        ponder_epsilon=float(routed.get("ponder_epsilon", 0.01)),
        ponder_cost=float(routed.get("ponder_cost", 0.001)),
        edge_bias=bool(routed.get("edge_bias", False)),
        persist_dir=None,
        image_dir=None,
        max_inline_depth=configured_max_inline_depth(runtime),
    )
    service.sync(
        include_compositions=bool(routed.get("include_compositions", True)),
        exclude_temporal=bool(routed.get("exclude_temporal", True)),
    )
    if not service.net._vertex_order:
        raise RuntimeError("the final library has no routable entries")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "overmind_structural.png"
    with tempfile.TemporaryDirectory(prefix="versal-paper-overmind-") as temporary:
        service.image_dir = Path(temporary)
        service.render_overmind()
        source = Path(temporary) / "overmind_pruned.png"
        if not source.exists():
            raise RuntimeError("cold overmind render did not produce a pruned portrait")
        shutil.copyfile(source, destination)

    live = sum(not bool(row.get("retired", False)) for row in library.summaries(include_retired=True))
    total = len(library)
    print(f"wrote {display_path(destination)} ({len(service.net._vertex_order)} routable experts; {live}/{total} live library records)")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def display_path(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0, help="fixed seed for the untrained router portrait")
    parser.add_argument("--skip-overmind", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library_root = resolve_repo_path(args.library).resolve()
    config_path = resolve_repo_path(args.config).resolve()
    output_dir = resolve_repo_path(args.output_dir).resolve()
    if not library_root.exists():
        raise FileNotFoundError(f"library not found: {library_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if output_dir.is_relative_to(library_root):
        raise ValueError("paper figures must be written outside the experiment library")

    configure_matplotlib()
    make_lifecycle(output_dir)
    if not args.skip_overmind:
        make_cold_overmind(library_root, config_path, output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
