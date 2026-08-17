# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.9", "numpy==2.4.6"]
# ///
"""Build the Apart paper's result figures from pinned run summaries.

Run from the repository root:

    uv run python ai/for_apart/make_evidence_figures.py
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "ai" / "for_apart" / "figures"

CURRENT_SUMMARY = REPO_ROOT / "results" / "20260817_031349_orchestrated" / "run_summary.json"
HISTORICAL_SUMMARY = REPO_ROOT / "ai" / "archive" / "20260706_flagship" / "results" / "run_summary.json"
XOR_SUMMARY = REPO_ROOT / "ai" / "archive" / "20260816_02_hackathon" / "results" / "20260816_185653_orchestrated" / "run_summary.json"

SOURCE_HASHES = {
    CURRENT_SUMMARY: "deeafeb90e790fbc8f72fc1bfadc10d07eebf95e5f2cb41d9647b5d3dd155e43",
    HISTORICAL_SUMMARY: "04838c39f752fa423cac70f796faf349511eed3682c45bdf71349c0bb8198288",
    XOR_SUMMARY: "3f4071a5e649b1488781bd61643f3f5f03e320ab718b1d7a13aaed0c4564b0e7",
}

FAMILY_LABELS = {
    1: "XOR",
    2: "parity",
    3: "two spirals",
    4: "pole",
    5: "double pole",
    6: "MNIST",
    7: "CIFAR-100",
    8: "ECG",
    9: "satellite",
    10: "NinaPro",
    11: "spherical",
    12: "Cosmic",
    13: "Darcy flow",
    14: "PSiCov",
    15: "FSD50K",
    16: "DeepSEA",
    17: "PGM",
    18: "ARC",
}

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#777777"
LIGHT_GRAY = "#D8D8D8"
GRID_GRAY = "#E7E7E7"
INK = "#1A1A1A"
MEMORY_OUTCOMES = {"library_hit", "refined"}


@dataclass(frozen=True)
class RungResult:
    rung: int
    support: float
    held_out: float
    literal_query: float
    seconds: float
    deadline_count: int


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_GRAY,
            "grid.linewidth": 0.5,
            "grid.alpha": 1.0,
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


def load_summary(path: Path) -> dict:
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_hash = SOURCE_HASHES[path]
    if actual_hash != expected_hash:
        raise RuntimeError(f"Pinned source changed: {path.relative_to(REPO_ROOT)} ({actual_hash})")
    return json.loads(path.read_text())


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    figure.savefig(pdf_path, metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(png_path, dpi=240, metadata={"Software": "Versal evidence figure generator"})
    plt.close(figure)
    print(f"wrote {pdf_path.relative_to(REPO_ROOT)} and {png_path.relative_to(REPO_ROOT)}")


def assert_close(actual: float, expected: float, *, tolerance: float = 5e-5) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0.0):
        raise AssertionError(f"expected {expected}, got {actual}")


def current_rung_results(summary: dict) -> list[RungResult]:
    if summary["status"] != "done" or summary["tasks_attempted"] != 36:
        raise AssertionError("the clean canary must be complete before plotting")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for task in summary["tasks"]:
        grouped[int(task["rung"])].append(task)

    results = []
    for rung in range(1, 19):
        tasks = grouped[rung]
        if len(tasks) != 2 or any(task["query_status"] != "evaluated" for task in tasks):
            raise AssertionError(f"rung {rung} does not contain two evaluated queries")
        results.append(
            RungResult(
                rung=rung,
                support=mean(float(task["support_accuracy"]) for task in tasks),
                held_out=mean(float(task["report_metric"]) for task in tasks),
                literal_query=mean(float(task["query_accuracy"]) for task in tasks),
                seconds=sum(float(task["seconds"]) for task in tasks),
                deadline_count=sum(task.get("failure_stage") == "time_budget" for task in tasks),
            )
        )

    assert_close(mean(row.support for row in results), 0.8870098472)
    assert_close(mean(row.held_out for row in results), 0.6099488826)
    assert_close(mean(row.literal_query for row in results), 0.6471437472)
    assert sum(row.deadline_count for row in results) == 15
    assert_close(results[-1].held_out, 0.0)
    assert_close(results[-1].literal_query, 0.6695075631)
    return results


def make_clean_canary(summary: dict) -> None:
    rows = current_rung_results(summary)
    support = np.array([row.support for row in rows])
    held_out = np.array([row.held_out for row in rows])
    positions = np.arange(len(rows))[::-1]

    figure, (quality_ax, time_ax) = plt.subplots(
        1,
        2,
        figsize=(7.05, 4.65),
        sharey=True,
        gridspec_kw={"width_ratios": [3.1, 1.5]},
    )

    for y, row in zip(positions, rows, strict=True):
        quality_ax.plot([row.held_out, row.support], [y, y], color=LIGHT_GRAY, linewidth=1.1, zorder=1)
    quality_ax.scatter(support, positions, s=25, color=BLUE, marker="o", linewidths=0, zorder=3)
    quality_ax.scatter(held_out, positions, s=24, color=ORANGE, marker="s", linewidths=0, zorder=3)

    arc = rows[-1]
    arc_y = positions[-1]
    quality_ax.scatter(
        [arc.literal_query],
        [arc_y],
        s=29,
        marker="D",
        facecolors="white",
        edgecolors=GRAY,
        linewidths=0.9,
        zorder=4,
    )
    quality_ax.annotate(
        "cell .670",
        (arc.literal_query, arc_y),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=6.2,
        color=GRAY,
    )
    quality_ax.annotate(
        "exact 0/2",
        (arc.held_out, arc_y),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=6.2,
        color=ORANGE,
    )

    held_out_mean = mean(row.held_out for row in rows)
    quality_ax.axvline(held_out_mean, color=ORANGE, linewidth=0.8, linestyle=(0, (3, 2)), alpha=0.8)
    quality_ax.text(
        held_out_mean + 0.012,
        positions[0] + 0.52,
        "cross-rung held-out mean .610",
        ha="left",
        va="bottom",
        fontsize=6.4,
        color=ORANGE,
    )

    quality_ax.set_xlim(-0.025, 1.035)
    quality_ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    quality_ax.set_xlabel("task-appropriate score")
    quality_ax.set_yticks(positions, [f"{row.rung:>2}  {FAMILY_LABELS[row.rung]}" for row in rows])
    quality_ax.set_ylim(-0.8, len(rows) - 0.2)
    quality_ax.set_title("a  Generalization across 18 rungs", loc="left", fontweight="bold", pad=20)
    quality_ax.grid(axis="x")
    quality_ax.grid(axis="y", visible=False)

    score_legend = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.2, color=BLUE, label="support mean"),
        Line2D([], [], marker="s", linestyle="none", markersize=4.0, color=ORANGE, label="held-out mean"),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=GRAY,
            label="ARC cell accuracy",
        ),
    ]
    quality_ax.legend(
        handles=score_legend,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.002),
        frameon=False,
        ncol=3,
        borderaxespad=0,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    seconds = np.array([row.seconds for row in rows])
    deadline_counts = np.array([row.deadline_count for row in rows])
    no_deadline = deadline_counts == 0
    deadline = ~no_deadline
    time_ax.hlines(positions, 5.0, seconds, color=LIGHT_GRAY, linewidth=1.0, zorder=1)
    time_ax.scatter(seconds[no_deadline], positions[no_deadline], s=20, color=GRAY, marker="o", linewidths=0, zorder=3)
    time_ax.scatter(seconds[deadline], positions[deadline], s=31, color=ORANGE, marker="^", linewidths=0, zorder=3)
    for x, y, count in zip(seconds[deadline], positions[deadline], deadline_counts[deadline], strict=True):
        time_ax.annotate(
            f"{count}/2",
            (x, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.1,
            color=ORANGE,
        )
    time_ax.set_xscale("log")
    time_ax.set_xlim(5.0, 2_000.0)
    time_ax.set_xticks([10, 100, 1_000], ["10", "100", "1k"])
    time_ax.set_xlabel("total task-seconds (log scale)")
    time_ax.set_title("b  Compute", loc="left", fontweight="bold", pad=20)
    time_ax.grid(axis="x")
    time_ax.grid(axis="y", visible=False)
    time_ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=3.8, color=GRAY, label="no deadline"),
            Line2D([], [], marker="^", linestyle="none", markersize=4.2, color=ORANGE, label="deadline; label = tasks"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.002),
        frameon=False,
        ncol=1,
        borderaxespad=0,
        handletextpad=0.35,
        labelspacing=0.25,
    )

    figure.text(
        0.01,
        0.005,
        "Two tasks per rung; 36/36 held-out queries evaluated. ARC held-out uses exact task success; the gray diamond is literal cell accuracy.",
        fontsize=6.3,
        color=GRAY,
    )
    figure.subplots_adjust(left=0.182, right=0.986, top=0.865, bottom=0.105, wspace=0.18)
    save_figure(figure, "clean_canary")


def admitted_level(key: str) -> int | None:
    match = re.match(r"^[mc](\d+)_", key)
    return int(match.group(1)) if match else None


def historical_trace(summary: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    tasks = summary["tasks"]
    if summary["status"] != "done" or len(tasks) != 400:
        raise AssertionError("historical persistence source must contain 400 completed encounters")

    total_memory = []
    non_xor_memory = []
    running_total = 0
    running_non_xor = 0
    highest_level = 0
    first_level_encounter: dict[int, int] = {}
    for encounter, task in enumerate(tasks, start=1):
        from_memory = task["outcome"] in MEMORY_OUTCOMES
        running_total += int(from_memory)
        running_non_xor += int(from_memory and int(task["rung"]) != 1)
        total_memory.append(running_total)
        non_xor_memory.append(running_non_xor)
        levels = [level for key in task.get("new_library_keys", []) if (level := admitted_level(key)) is not None]
        if levels and max(levels) > highest_level:
            for level in range(highest_level + 1, max(levels) + 1):
                first_level_encounter[level] = encounter
            highest_level = max(levels)

    non_xor_encounters = sum(int(task["rung"]) != 1 for task in tasks)
    if (running_total, running_non_xor, non_xor_encounters, highest_level) != (234, 168, 333, 5):
        raise AssertionError("historical headline counts no longer match the frozen evidence")
    return np.arange(1, 401), np.array(total_memory), np.array(non_xor_memory), first_level_encounter


def xor_trace(summary: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    tasks = summary["tasks"]
    if summary["status"] != "done" or len(tasks) != 200:
        raise AssertionError("XOR source must contain 200 completed encounters")
    outcomes = defaultdict(int)
    for task in tasks:
        outcomes[task["outcome"]] += 1
        assert_close(float(task["support_accuracy"]), 1.0)
        assert_close(float(task["query_accuracy"]), 1.0)
    if dict(outcomes) != {"evolved": 1, "refined": 7, "library_hit": 192}:
        raise AssertionError(f"unexpected XOR outcomes: {dict(outcomes)}")

    accepted = [task for task in tasks if task.get("new_library_keys")]
    encounters = np.array([index for index, task in enumerate(tasks, start=1) if task.get("new_library_keys")])
    complexity = np.array([float(task["size_metrics"]["champion_complexity"]) for task in accepted])
    if encounters.tolist() != [1, 2, 4, 5, 6, 10, 13, 25]:
        raise AssertionError("XOR admission encounters changed")
    if complexity.tolist() != [18.0, 16.0, 10.0, 12.0, 10.0, 7.0, 6.0, 5.0]:
        raise AssertionError("XOR complexity trajectory changed")
    return encounters, complexity, accepted


def make_persistence_xor(historical_summary: dict, xor_summary: dict) -> None:
    encounters, total_memory, non_xor_memory, level_milestones = historical_trace(historical_summary)
    xor_encounters, xor_complexity, xor_accepted = xor_trace(xor_summary)

    figure, (memory_ax, xor_ax) = plt.subplots(
        1,
        2,
        figsize=(7.05, 3.15),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )

    memory_ax.plot(encounters, total_memory, color=BLUE, linewidth=1.7, label="all rungs")
    memory_ax.plot(encounters, non_xor_memory, color=SKY, linewidth=1.4, linestyle=(0, (4, 2)), label="excluding XOR")
    memory_ax.fill_between(encounters, 0, total_memory, color=BLUE, alpha=0.06, linewidth=0)
    memory_ax.scatter([400], [234], s=25, color=BLUE, zorder=4)
    memory_ax.scatter([400], [168], s=23, color=SKY, zorder=4)
    memory_ax.annotate("234 / 400", (400, 234), xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=7.0, color=BLUE)
    memory_ax.annotate(
        "168 / 333 non-XOR",
        (400, 168),
        xytext=(-4, -10),
        textcoords="offset points",
        ha="right",
        fontsize=7.0,
        color="#3887A8",
    )

    milestone_specs = [
        (2, 31, "L2/L3 at encounters 6/9"),
        (4, 61, "L4 at encounter 26"),
        (5, 132, "L5 at encounter 194"),
    ]
    for level, label_y, label in milestone_specs:
        x = level_milestones[level]
        if level == 2 and level_milestones[3] != 9:
            raise AssertionError("unexpected early L3 milestone")
        memory_ax.vlines(x, 0, label_y - 4, color=GRAY, linewidth=0.65, linestyle=(0, (2, 2)), alpha=0.75)
        memory_ax.text(x + 4, label_y, label, ha="left", va="bottom", fontsize=6.1, color=GRAY)

    memory_ax.set_xlim(0, 410)
    memory_ax.set_ylim(0, 260)
    memory_ax.set_xticks([0, 100, 200, 300, 400])
    memory_ax.set_yticks([0, 50, 100, 150, 200, 250])
    memory_ax.set_xlabel("encounter")
    memory_ax.set_ylabel("cumulative hits + refinements")
    memory_ax.set_title("a  Historical persistence", loc="left", fontweight="bold")
    memory_ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.2, columnspacing=1.0)

    step_x = np.append(xor_encounters, 200)
    step_y = np.append(xor_complexity, xor_complexity[-1])
    xor_ax.step(step_x, step_y, where="post", color=GREEN, linewidth=1.6)
    xor_ax.scatter(xor_encounters, xor_complexity, s=25, color=GREEN, edgecolors="white", linewidths=0.45, zorder=4)
    first = xor_accepted[0]["size_metrics"]
    last = xor_accepted[-1]["size_metrics"]
    if (first["champion_nodes"], first["champion_connections"], last["champion_nodes"], last["champion_connections"]) != (
        9.0,
        13.0,
        5.0,
        4.0,
    ):
        raise AssertionError("XOR endpoint topology counts changed")
    xor_ax.annotate(
        "final: 5 nodes, 4 edges",
        (200, 5),
        xytext=(-3, 8),
        textcoords="offset points",
        ha="right",
        fontsize=6.8,
        color=GREEN,
    )
    xor_ax.annotate("18", (xor_encounters[0], xor_complexity[0]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.2, color=GREEN)
    xor_ax.text(
        0.98,
        0.47,
        "accepted sequence\n18 -> 16 -> 10 -> 12 -> 10 -> 7 -> 6 -> 5",
        transform=xor_ax.transAxes,
        va="center",
        ha="right",
        fontsize=6.4,
        color=GREEN,
        linespacing=1.25,
    )
    xor_ax.text(
        0.02,
        0.97,
        "1 evolution · 7 refinements · 192 hits\nsupport = held-out = 1.0 throughout",
        transform=xor_ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.6,
        color=GRAY,
        linespacing=1.25,
    )
    xor_ax.set_xlim(0, 205)
    xor_ax.set_ylim(3.5, 20.5)
    xor_ax.set_xticks([0, 50, 100, 150, 200])
    xor_ax.set_yticks([5, 10, 15, 20])
    xor_ax.set_xlabel("repeated XOR encounter")
    xor_ax.set_ylabel("accepted solution complexity")
    xor_ax.set_title("b  Fixed-task refinement", loc="left", fontweight="bold")

    figure.text(
        0.01,
        0.006,
        "Observational traces from earlier software states: persistence is not a matched cost comparison; XOR is one fixed task, not 200 independent tasks.",
        fontsize=6.3,
        color=GRAY,
    )
    figure.subplots_adjust(left=0.083, right=0.985, top=0.93, bottom=0.175, wspace=0.29)
    save_figure(figure, "persistence_xor")


def main() -> None:
    configure_matplotlib()
    current = load_summary(CURRENT_SUMMARY)
    historical = load_summary(HISTORICAL_SUMMARY)
    xor = load_summary(XOR_SUMMARY)
    make_clean_canary(current)
    make_persistence_xor(historical, xor)


if __name__ == "__main__":
    main()
