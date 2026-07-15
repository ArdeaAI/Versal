# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.9", "numpy==2.4.6", "pillow==12.2.0"]
# ///
"""Generate figures for the conference paper and technical report from frozen archives.

Deterministic: fixed jitter seed, no timestamps in any output (PDF CreationDate stripped).
Run from the repo root:  uv run python paper/figures/make_figures.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "paper" / "figures"
ARCHIVE_DIR = REPO_ROOT / "ai" / "archive"

FLAGSHIP_SUMMARY = ARCHIVE_DIR / "20260706_flagship" / "results" / "run_summary.json"
G0_SUMMARY = ARCHIVE_DIR / "20260705_g0" / "results" / "20260705_175445_orchestrated" / "run_summary.json"
G1_COLD_SUMMARY = ARCHIVE_DIR / "20260705_g1" / "results" / "20260705_183030_orchestrated" / "run_summary.json"
G1_WARM_SUMMARY = ARCHIVE_DIR / "20260705_g1" / "results" / "20260705_185704_orchestrated" / "run_summary.json"
PROBE_SUMMARY = ARCHIVE_DIR / "20260705_g2" / "results" / "20260705_234350_orchestrated" / "run_summary.json"
G2_SNAPSHOT_SUMMARY = ARCHIVE_DIR / "20260705_g2" / "results" / "20260706_015530_orchestrated" / "run_summary.json"
G2_TAIL_SUMMARY = ARCHIVE_DIR / "20260705_g2" / "results" / "20260706_024019_orchestrated" / "run_summary.json"
CANARY_20260715_SUMMARY = ARCHIVE_DIR / "20260715_canary" / "results" / "20260715_014520_orchestrated" / "run_summary.json"

# Okabe-Ito colorblind-safe palette, fixed assignment per rung reused across all figures.
RUNG_COLOR = {
    1: "#0072B2",  # blue: xor
    2: "#56B4E9",  # sky blue: parity
    3: "#D55E00",  # vermillion: two_spirals (the unsolved family, also used for all two-spirals arms)
    4: "#009E73",  # bluish green: pole
    5: "#CC79A7",  # reddish purple: double_pole
    6: "#E69F00",  # orange: mnist family
}
RUNG_LABEL = {
    1: "rung 1: xor",
    2: "rung 2: parity",
    3: "rung 3: two_spirals",
    4: "rung 4: pole",
    5: "rung 5: double_pole",
    6: "rung 6: mnist family",
}
ARM_COLOR = {
    "G0": "#000000",
    "G1": "#0072B2",
    "G1 warm": "#56B4E9",
    "probe": "#D55E00",
}
ARM_MARKER = {"G0": "o", "G1": "s", "G1 warm": "D", "probe": "^"}
NEUTRAL_GRAY = "#8A8A8A"
GRID_GRAY = "#E4E4E4"
INK = "#1A1A1A"

MEMORY_OUTCOMES = {"library_hit", "refined"}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": GRID_GRAY,
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def save_figure(figure: plt.Figure, stem: str) -> None:
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    figure.savefig(pdf_path, metadata={"CreationDate": None})
    figure.savefig(png_path, dpi=200)
    plt.close(figure)
    print(f"wrote {pdf_path.name} + {png_path.name}")


def load_tasks(summary_path: Path) -> list[dict]:
    return json.loads(summary_path.read_text())["tasks"]


# ---------------------------------------------------------------- figure 1: the solve ladder


def draw_box(ax, x, y, w, h, text, *, fill="#F4F4F4", edge="#666666", fontsize=7.0, bold=False, lw=0.9, text_color=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=0.9",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            mutation_aspect=1.0,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold" if bold else "normal",
        linespacing=1.25,
    )


def draw_arrow(ax, start, end, *, color="#444444", lw=1.1, rad=0.0, style="-|>", ls="solid"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style,
            mutation_scale=9,
            color=color,
            linewidth=lw,
            linestyle=ls,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def arrow_label(ax, x, y, text, *, color="#444444", fontsize=6.2, ha="center", rotation=0):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color, rotation=rotation, linespacing=1.2)


def make_fig1() -> None:
    library_blue = "#0072B2"
    stone_red = "#D55E00"
    figure, ax = plt.subplots(figsize=(7.05, 4.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 69.5)
    ax.axis("off")
    ax.grid(False)

    # Tier labels.
    ax.text(0.6, 40, "per-task solve", rotation=90, va="center", ha="center", fontsize=7.0, color="#777777", style="italic")
    ax.text(0.6, 8.5, "persistent memory", rotation=90, va="center", ha="center", fontsize=7.0, color=library_blue, style="italic")

    # Persistent tier: the library.
    draw_box(
        ax,
        8,
        3,
        84,
        10.5,
        "library: persistent, immutable solution store\nmodules (level 1) + compositions (level 2+), keyed by structural I/O signature;"
        "\nquality-diversity admission; route decay, retirement, and reference-safe garbage collection",
        fill="#E3EFF8",
        edge=library_blue,
        fontsize=7.0,
        bold=False,
        lw=1.4,
    )
    ax.text(9.5, 12.2, "", fontsize=6)

    # Per-task tier boxes.
    draw_box(ax, 3, 38, 10, 8, "task", fontsize=7.6, bold=True)
    draw_box(ax, 17, 38, 15, 8, "lookup\ntop-5 quick eval\nby I/O signature", fontsize=6.4)
    draw_box(ax, 33, 53, 27, 8, "refine on hit\nbudgeted, decaying; seeded\nfrom the stored solution", fontsize=6.4)

    # Evolve ladder container + strategies.
    draw_box(ax, 36, 20.5, 24, 27.5, "", fill="#FBFBFB", edge="#999999", lw=0.8)
    ax.text(48, 44.8, "evolve ladder (one budget)", ha="center", va="center", fontsize=6.6, color=INK, fontweight="bold")
    draw_box(ax, 38, 38.0, 20, 4.5, "routed: learned reuse", fontsize=6.0, fill="#EFEFEF")
    draw_box(ax, 38, 32.2, 20, 4.5, "grammar: induced programs", fontsize=6.0, fill="#EFEFEF")
    draw_box(ax, 38, 26.4, 20, 4.5, "direct: structure growth", fontsize=6.0, fill="#EFEFEF")
    draw_box(ax, 38, 20.8, 20, 4.5, "composition: hierarchical reuse", fontsize=6.0, fill="#EFEFEF")
    draw_arrow(ax, (48, 37.8), (48, 36.9), lw=0.8)
    draw_arrow(ax, (48, 32.0), (48, 31.1), lw=0.8)
    draw_arrow(ax, (48, 26.2), (48, 25.3), lw=0.8)

    draw_box(ax, 64, 20.5, 15, 10, "decompose\n+ recurse\n(solvability-gated)", fontsize=6.4)
    draw_box(ax, 82, 39, 15.5, 9, "support-only\nadmission gate", fontsize=6.6)
    draw_box(ax, 62, 31.5, 17, 8, "one-shot held-out report\n(never feeds search)", fontsize=6.2, fill="#FFF4E6", edge="#D55E00")

    # Forward flow.
    draw_arrow(ax, (13, 42), (17, 42))
    draw_arrow(ax, (24.5, 46), (37, 53), rad=-0.25)
    arrow_label(ax, 27.2, 51.3, "hit", color="#444444")
    draw_arrow(ax, (32, 42), (36, 41.5))
    arrow_label(ax, 34, 44.0, "miss", color="#444444")
    draw_arrow(ax, (60, 56.5), (90, 48.3), rad=-0.22)
    arrow_label(ax, 77, 57.4, "strict lexicographic win", color="#444444")
    draw_arrow(ax, (60, 40), (82, 43.5), rad=0.12)
    arrow_label(ax, 70.5, 43.9, "executable + clears support bar", color="#444444")
    draw_arrow(ax, (60, 35.5), (62, 35.5), color="#D55E00", lw=0.9, ls=(0, (3, 2)))
    draw_arrow(ax, (60, 24.0), (64, 25.5))
    arrow_label(ax, 62.0, 22.2, "stall", color="#444444")
    # Recursion loop back to the task, routed around the right and top edges.
    ax.plot([79.6, 99.0, 99.0, 8.0], [25.5, 25.5, 66.0, 66.0], color="#444444", linewidth=1.0, linestyle=(0, (4, 2)), zorder=1)
    draw_arrow(ax, (8.0, 66.0), (8.0, 46.6), color="#444444", lw=1.0, ls=(0, (4, 2)))
    arrow_label(ax, 53.5, 67.9, "sub-tasks recurse (depth + 1); parent re-evolves over the sub-solutions", color="#444444")

    # Admission writes into the library.
    draw_arrow(ax, (89.5, 38.5), (89.5, 14), color=library_blue, lw=1.2)
    arrow_label(ax, 91.1, 25.5, "admit (provenance, levels)", color=library_blue, rotation=90)

    # Library feedback into the per-task flow.
    draw_arrow(ax, (24, 14), (24, 37.5), color=library_blue, lw=1.0)
    arrow_label(ax, 22.6, 25.5, "top-5 candidates", color=library_blue, rotation=90)
    draw_arrow(ax, (34, 14), (34, 52.5), color=library_blue, lw=1.0)
    arrow_label(ax, 32.6, 33.0, "refinement seeds", color=library_blue, rotation=90)
    draw_arrow(ax, (50, 14), (50, 20.2), color=library_blue, lw=1.0)
    arrow_label(
        ax,
        48.8,
        17.6,
        "frozen experts (routed)\nmacros + modules\nstone warm-starts",
        color=library_blue,
        ha="right",
        fontsize=5.8,
    )
    # Failure shelves a stepping stone.
    draw_arrow(ax, (57, 20.2), (57, 14), color=stone_red, lw=1.2)
    arrow_label(ax, 58.4, 17.6, "fail: shelve best champion\nas a stepping stone (wall ledger)", color=stone_red, ha="left", fontsize=5.8)

    figure.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save_figure(figure, "fig1_ladder")


# ---------------------------------------------------------------- figure 2: marginal cost per encounter


def make_fig2() -> None:
    tasks = load_tasks(FLAGSHIP_SUMMARY)
    per_rung: dict[int, list[dict]] = {r: [] for r in range(1, 7)}
    for row in tasks:
        per_rung[row["rung"]].append(row)

    figure, axes = plt.subplots(2, 3, figsize=(7.05, 4.3), sharex=True, sharey=True)
    for idx, rung in enumerate(range(1, 7)):
        ax = axes[idx // 3][idx % 3]
        color = RUNG_COLOR[rung]
        rows = per_rung[rung]
        xs = np.arange(1, len(rows) + 1)
        seconds = np.array([row["seconds"] for row in rows])
        outcome = [row["outcome"] for row in rows]
        memory_mask = np.array([o in MEMORY_OUTCOMES for o in outcome])
        evolved_mask = np.array([o == "evolved" for o in outcome])
        failed_mask = np.array([o == "failed" for o in outcome])

        ax.scatter(xs[memory_mask], seconds[memory_mask], marker="o", s=8, color=color, linewidths=0, alpha=0.9, zorder=3)
        ax.scatter(xs[evolved_mask], seconds[evolved_mask], marker="^", s=16, facecolors="none", edgecolors=color, linewidths=0.8, zorder=3)
        ax.scatter(xs[failed_mask], seconds[failed_mask], marker="x", s=13, color=color, linewidths=0.8, alpha=0.75, zorder=3)

        ax.set_yscale("log")
        ax.set_ylim(6e-4, 4e2)
        ax.set_xlim(0, 69)
        ax.set_xticks([1, 20, 40, 60])
        ax.set_title(RUNG_LABEL[rung], loc="left", fontsize=7.2, color=color, fontweight="bold", pad=3)
        if idx % 3 == 0:
            ax.set_ylabel("task wall-clock (s)")
        if idx // 3 == 1:
            ax.set_xlabel("encounter index within rung")

    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.4, color=NEUTRAL_GRAY, label="solved from memory (hit / refined)"),
        Line2D([], [], marker="^", linestyle="none", markersize=4.6, markerfacecolor="none", markeredgecolor=NEUTRAL_GRAY, label="evolved (fresh solve)"),
        Line2D([], [], marker="x", linestyle="none", markersize=4.2, color=NEUTRAL_GRAY, label="failed"),
    ]
    figure.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005), handletextpad=0.4, columnspacing=1.6)
    figure.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.105, wspace=0.08, hspace=0.30)
    save_figure(figure, "fig2_cost")


# ---------------------------------------------------------------- figure 3: two-spirals arms


def make_fig3() -> None:
    arms = {
        "G0": [row["metric"] for row in load_tasks(G0_SUMMARY)],
        "G1": [row["metric"] for row in load_tasks(G1_COLD_SUMMARY)],
        "G1 warm": [row["metric"] for row in load_tasks(G1_WARM_SUMMARY)],
        "probe": [row["metric"] for row in load_tasks(PROBE_SUMMARY)],
    }

    figure, ax = plt.subplots(figsize=(3.4, 2.55))
    ax.axhline(0.95, color=NEUTRAL_GRAY, linewidth=0.9, linestyle=(0, (4, 2)), zorder=2)
    ax.text(24.6, 0.956, "accept bar (0.95)", ha="right", va="bottom", fontsize=6.0, color="#666666")

    for arm, metrics in arms.items():
        xs = np.arange(1, len(metrics) + 1)
        ax.plot(
            xs,
            metrics,
            marker=ARM_MARKER[arm],
            markersize=2.6,
            linewidth=1.0,
            color=ARM_COLOR[arm],
            markerfacecolor=ARM_COLOR[arm],
            markeredgewidth=0,
            zorder=3,
            label=arm,
        )
        ax.annotate(
            arm,
            xy=(xs[-1], metrics[-1]),
            xytext=(3.5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=6.6,
            color=ARM_COLOR[arm],
            fontweight="bold",
        )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 1.03),
        ncol=2,
        frameon=False,
        fontsize=5.8,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.9,
        labelspacing=0.25,
    )
    ax.set_xlim(0.4, 25.0)
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("attempt index (accumulated assaults)")
    ax.set_ylabel("trained query accuracy")
    figure.subplots_adjust(left=0.145, right=0.985, top=0.975, bottom=0.165)
    save_figure(figure, "fig3_spirals")


# ---------------------------------------------------------------- figure 4: structure vs weights


def make_fig4() -> None:
    sources = [
        FLAGSHIP_SUMMARY,
        G0_SUMMARY,
        G1_COLD_SUMMARY,
        G1_WARM_SUMMARY,
        PROBE_SUMMARY,
        G2_SNAPSHOT_SUMMARY,
        G2_TAIL_SUMMARY,
    ]
    points: list[tuple[float, float, int, bool]] = []
    for source in sources:
        for row in load_tasks(source):
            samples = row.get("sample_metrics")
            if not samples or row.get("metric") is None:
                continue
            is_spirals = str(row["task"]).startswith("two_spirals")
            points.append((samples["max_sample_accuracy"], row["metric"], row["rung"], is_spirals))

    rng = np.random.default_rng(20260706)
    figure, ax = plt.subplots(figsize=(3.4, 2.9))
    for wall in (0.70, 0.80):
        ax.axvline(wall, color=NEUTRAL_GRAY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=2)
        ax.text(wall + 0.008, 0.575, f"wall {wall:.2f}", rotation=90, va="bottom", ha="left", fontsize=5.8, color="#666666")

    def jittered(values: list[float], spread: float) -> np.ndarray:
        return np.clip(np.array(values) + rng.uniform(-spread, spread, len(values)), 0.0, 1.0)

    for rung in range(1, 7):
        rung_points = [p for p in points if p[2] == rung and not p[3]]
        if rung_points:
            xs = jittered([p[0] for p in rung_points], 0.010)
            ys = jittered([p[1] for p in rung_points], 0.006)
            ax.scatter(xs, ys, marker="o", s=9, color=RUNG_COLOR[rung], alpha=0.5, linewidths=0, zorder=3, label=RUNG_LABEL[rung])
    spiral_points = [p for p in points if p[3]]
    xs = jittered([p[0] for p in spiral_points], 0.010)
    ys = jittered([p[1] for p in spiral_points], 0.006)
    ax.scatter(xs, ys, marker="D", s=11, color=RUNG_COLOR[3], alpha=0.6, linewidths=0, zorder=4, label="two_spirals (all arms)")

    ax.set_xlim(-0.04, 1.05)
    ax.set_ylim(0.35, 1.04)
    ax.set_xlabel("max sample accuracy (shared-weight structure test)")
    ax.set_ylabel("trained query metric")
    handles, labels = ax.get_legend_handles_labels()
    order = ["rung 1: xor", "rung 2: parity", "two_spirals (all arms)", "rung 4: pole", "rung 5: double_pole", "rung 6: mnist family"]
    ordered = sorted(zip(labels, handles), key=lambda pair: order.index(pair[0]))
    ax.legend(
        [h for _, h in ordered],
        [lab for lab, _ in ordered],
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
        fontsize=5.8,
        handletextpad=0.25,
        borderaxespad=0.2,
        labelspacing=0.35,
    )
    figure.subplots_adjust(left=0.145, right=0.985, top=0.975, bottom=0.145)
    save_figure(figure, "fig4_wall")


# ---------------------------------------------------------------- figure 7: post-change full-method canary


def make_fig7() -> None:
    """Show the July 15 canary's literal support and held-out rails without imputing N/A as zero."""

    rows = load_tasks(CANARY_20260715_SUMMARY)
    rungs = np.array([int(row["rung"]) for row in rows])
    support = np.array([float(row["support_accuracy"]) if row.get("support_accuracy") is not None else np.nan for row in rows])
    query = np.array([float(row["query_accuracy"]) if row.get("query_accuracy") is not None else np.nan for row in rows])

    support_blue = "#0072B2"
    query_orange = "#D55E00"
    figure, ax = plt.subplots(figsize=(7.05, 3.25))
    ax.axhspan(-0.13, 0.0, color="#F2F2F2", zorder=0)
    ax.axhline(0.95, color=NEUTRAL_GRAY, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
    ax.text(18.35, 0.956, "support acceptance bar", ha="right", va="bottom", fontsize=5.8, color="#666666")

    for rung, support_value, query_value in zip(rungs, support, query):
        if np.isfinite(support_value) and np.isfinite(query_value):
            ax.plot([rung, rung], [support_value, query_value], color="#C8C8C8", linewidth=0.8, zorder=2)

    support_mask = np.isfinite(support)
    query_mask = np.isfinite(query)
    ax.scatter(rungs[support_mask], support[support_mask], marker="o", s=24, color=support_blue, edgecolor="white", linewidth=0.45, zorder=4, label="best support accuracy")
    ax.scatter(rungs[query_mask], query[query_mask], marker="s", s=23, color=query_orange, edgecolor="white", linewidth=0.45, zorder=5, label="held-out query accuracy")

    missing = ~(support_mask | query_mask)
    ax.scatter(rungs[missing], np.full(int(missing.sum()), -0.065), marker="x", s=28, color="#333333", linewidth=1.0, zorder=5, label="N/A: no executable parent before deadline")
    for rung in rungs[missing]:
        ax.text(rung, -0.105, "N/A", ha="center", va="center", fontsize=5.5, color="#444444")

    ax.set_xlim(0.5, 18.5)
    ax.set_ylim(-0.13, 1.04)
    ax.set_xticks(rungs)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel("Icarus rung (one task per rung, seed 0)")
    ax.set_ylabel("literal accuracy")
    handles, labels = ax.get_legend_handles_labels()
    figure.suptitle(
        "Post-change full-method canary: fitting remains distinct from held-out generalization",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=8.2,
        fontweight="bold",
    )
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.4)
    ax.text(0.6, -0.065, "not evaluated", ha="left", va="center", fontsize=5.8, color="#777777", style="italic")
    figure.subplots_adjust(left=0.075, right=0.99, top=0.76, bottom=0.18)
    save_figure(figure, "fig7_canary")


# ---------------------------------------------------------------- figures 5 and 6: curated renders


def make_fig5e() -> None:
    """Arrange the three differently shaped historical network renders on one readable plate."""

    panels = {
        "(a) level-3 reuse chain": FIGURE_DIR / "fig5c_level3_chain.png",
        "(b) repeated-macro assembly": FIGURE_DIR / "fig5b_seven_macro_artifact.png",
        "(c) interrupted probe champion": FIGURE_DIR / "fig5d_probe_champion.png",
    }
    figure = plt.figure(figsize=(7.05, 4.6), facecolor="white")
    grid = figure.add_gridspec(2, 2, width_ratios=(0.36, 0.64), height_ratios=(1, 1), wspace=0.025, hspace=0.12)
    axes = (figure.add_subplot(grid[:, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, 1]))
    for ax, (label, path) in zip(axes, panels.items()):
        ax.imshow(plt.imread(path))
        ax.set_title(label, loc="left", fontsize=7.0, pad=3, fontweight="bold")
        ax.axis("off")
    figure.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005)
    save_figure(figure, "fig5e_artifact_triptych")


def copy_renders() -> None:
    copies = {
        "fig5a_overmind.png": ARCHIVE_DIR / "20260706_flagship" / "library" / "images" / "overmind.png",
        "fig5b_seven_macro_artifact.png": ARCHIVE_DIR / "20260705_g1" / "results" / "20260705_185704_orchestrated" / "task_0009" / "net.png",
        "fig5c_level3_chain.png": ARCHIVE_DIR / "20260705_g2" / "results" / "20260706_015530_orchestrated" / "task_0027" / "net.png",
        "fig5d_probe_champion.png": ARCHIVE_DIR / "20260705_g2" / "results" / "20260705_234350_orchestrated" / "task_0001" / "net.png",
        "fig6_motifs.png": ARCHIVE_DIR / "20260706_flagship" / "library" / "images" / "motifs.png",
    }
    for name, source in copies.items():
        shutil.copyfile(source, FIGURE_DIR / name)
        print(f"copied {name} <- {source.relative_to(REPO_ROOT)}")


def main() -> None:
    configure_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig7()
    copy_renders()
    make_fig5e()


if __name__ == "__main__":
    main()
