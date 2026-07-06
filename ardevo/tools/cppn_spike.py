"""The lever-E spike runner (ai/cppn_spike_plan.md Phase 2): evolve a CPPN generator on two-spirals.

Arms: `cppn` (the generative encoding, sin or tanh detector bank) and `direct` (the in-harness
control: the ordinary flat encoding under the IDENTICAL config, operators, palette, and gradient
budget, so a stalled CPPN arm reads as generator-space hardness rather than a weak harness).
The inline config is pinned here, per the plan: sin/gaussian in the palette (build_evolver's
fallback excludes them), the flat recipe with explicit probs, sequential assessment (no pool
side effects), and the rung3-recipe trainer (200 steps + weight decay) so a stall means search.

    uv run cppn_spike --arm cppn --phenotype-activation sin --seed 0
    uv run cppn_spike --arm direct --seed 0
    uv run cppn_spike --offline --generations 20        # synthetic smoke, no network
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from ardevo.cppn import CppnTaskAdapter, fourier_cppn_genome, synthetic_two_spirals_task
from ardevo.dataset.icarus import Level0Encoder, Task, encode_task, support_loader
from ardevo.evaluation import input_width, output_features
from ardevo.evolution.evolver import Evolver, TaskAdapter
from ardevo.evolution.genome import genome_to_dict
from ardevo.evolution.registry import build_evolver
from ardevo.evolution.train import gradient

console = Console()

_TRANSFER_BANKS = (16, 32, 64, 128, 256)


def spike_config(pop_size: int, seed: int, train_steps: int, pareto: bool, train_kind: str = "gradient") -> dict[str, Any]:
    config: dict[str, Any] = {
        "seed": seed,
        "substrate": {"available_activations": ["tanh", "relu", "sigmoid", "identity", "sin", "gaussian"], "default_activation": "tanh"},
        "evolution": {
            "pop_size": pop_size,
            "elitism": 2,
            "assess_workers": 0,
            "init": {"kind": "minimal", "weight_scale": 1.0},
            "selection": {"kind": "tournament", "tournament_size": 3},
            "crossover": {"kind": "neat", "rate": 0.2},
            "mutation": {
                "operators": ["add_rich_node", "add_deep_node", "add_connection", "toggle_connection", "mutate_activation", "mutate_aggregation"],
                "add_rich_node_prob": 0.12,
                "add_rich_node_fan_in": 4,
                "add_deep_node_prob": 0.12,
                "add_deep_node_fan_in": 4,
                "add_deep_node_fan_out": 2,
                "add_connection_prob": 0.12,
                "toggle_connection_prob": 0.05,
                "mutate_activation_prob": 0.06,
                "mutate_aggregation_prob": 0.05,
                "mutate_aggregation_max_fan_in": 4,
            },
            # gradient = the hand-tuned rung3 standalone recipe (the gate-E baseline arms);
            # gradient_scheduled = remedy 1 (warmup + cosine at a gentler peak, probe_6's regime).
            "train": {"kind": train_kind, "steps": train_steps, "lr": 0.01 if train_kind == "gradient_scheduled" else 0.03, "writeback": True, "weight_decay": 0.0002},
            "evaluate": {"kind": "hybrid", "samples": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], "batched_samples": False},
            "speciation": {"kind": "neat", "threshold": 1.5, "target_species": 6, "threshold_adjust": 0.3, "min_threshold": 0.3},
        },
        "fitness": {
            "components": ["bounded_negative_support_loss", "support_accuracy", "weight_robustness", "hidden_penalty", "complexity_penalty"],
            "w_bounded_negative_support_loss": 2.0,
            "w_support_accuracy": 1.0,
            "w_weight_robustness": 0.5,
            "w_hidden_penalty": 0.05,
            "w_complexity_penalty": 0.01,
        },
    }
    if pareto:
        config["evolution"]["selection"] = {"kind": "nsga2"}
        config["evolution"]["novelty"] = {"k": 15, "archive_cap": 256, "probe_rows": 64}
        config["fitness"]["objectives"] = ["support_accuracy", "novelty", "connection_cost"]
    return config


def load_two_spirals(offline: bool, seed: int) -> Task:
    if offline:
        return synthetic_two_spirals_task()
    from ardevo.evolution.multitask import build_pool_report

    report = build_pool_report(source="Ardea/Icarus-dataset", rungs=[3], n_samples=400, support_fraction=0.8, tasks_per_rung=1, shuffle=False, seed=seed)
    if not report.entries:
        raise RuntimeError(f"rung 3 failed to load: {[skipped.message for skipped in report.skipped]}")
    return report.entries[0].task


def build_adapter(arm: str, task: Task, h: int, phenotype_activation: str) -> CppnTaskAdapter | TaskAdapter:
    support_input, _support_output = support_loader(task)
    width = 1
    for dim in support_input.data.shape[1:]:
        width *= int(dim)
    encoder = Level0Encoder(max_flat_dim=width)
    encoded = encode_task(task, encoder)
    if arm == "cppn":
        return CppnTaskAdapter(encoded, encoder, h=h, phenotype_activation=phenotype_activation)
    return TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded))


def transfer_curve(champion_genome: Any, adapter: CppnTaskAdapter, refit_steps: int) -> list[dict[str, float]]:
    """Re-expand the SAME generator at other bank sizes. The no-refit column gates (a refit can
    mask a generator that memorized its evolved resolution); the refit column is secondary."""
    curve = []
    for h in _TRANSFER_BANKS:
        resized = CppnTaskAdapter(adapter.encoded, adapter.encoder, h=h, phenotype_activation=adapter.phenotype_activation)
        no_refit = resized.evaluate(resized.decode(champion_genome))["query_accuracy"]
        module = resized.decode(champion_genome)
        _genome, refit_module = gradient(champion_genome, module, resized.encoded, rng=random.Random(0), steps=refit_steps, lr=0.03, writeback=False)
        curve.append({"h": h, "query_accuracy": no_refit, "query_accuracy_refit": resized.evaluate(refit_module)["query_accuracy"]})
    return curve


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    task = load_two_spirals(args.offline, args.seed)
    adapter = build_adapter(args.arm, task, args.hidden, args.phenotype_activation)
    evolver: Evolver = build_evolver(spike_config(args.pop, args.seed, args.steps, args.pareto, args.train_kind))
    rng = random.Random(args.seed)
    seeded_front = None
    if args.seed_fixture and isinstance(adapter, CppnTaskAdapter):
        # The motif-seeding probe (gate-E fork, remedy a): graft the E0 Fourier topology into the
        # initial population. Its hand-numbered innovations may misalign NEAT crossover against the
        # init population's numbering; acceptable for the probe (mutation + elitism carry the lineage).
        fixture = fourier_cppn_genome(m=4, seed=args.seed, n_logits=adapter.n_logits, coefficient_scale=10.0)
        seeded_front = lambda _tracker: [fixture]  # noqa: E731
    state = evolver.seed_state(adapter, rng, seeded_front=seeded_front)

    def metric_of(item: Any) -> float:
        return float(item.metrics.get("query_accuracy", 0.0))

    history: list[float] = []
    best = max(state.population, key=metric_of)
    last_improved = 0
    started = time.perf_counter()
    for generation in range(args.generations):
        generation_best = max(state.population, key=metric_of)
        if metric_of(generation_best) > metric_of(best) + 0.0049:  # the orchestrated stall epsilon
            best = generation_best
            last_improved = generation
        elif metric_of(generation_best) > metric_of(best):
            best = generation_best
        history.append(metric_of(generation_best))
        if metric_of(best) >= args.accept:
            console.print(f"[green]accept bar cleared at generation {generation}: query {metric_of(best):.4f}[/green]")
            break
        if args.stall and generation - last_improved >= args.stall:
            console.print(f"[yellow]stalled: no {0.005} improvement in {args.stall} generations (gen {generation})[/yellow]")
            break
        if generation % 20 == 0:
            console.print(f"gen {generation:>3}: best query {metric_of(best):.4f} | complexity {best.genome.complexity()} | {time.perf_counter() - started:.0f}s")
        evolver.advance(state, adapter)

    champion = best
    report: dict[str, Any] = {
        "arm": args.arm,
        "phenotype_activation": args.phenotype_activation if args.arm == "cppn" else None,
        "pareto": args.pareto,
        "offline": args.offline,
        "seed": args.seed,
        "hidden": args.hidden if args.arm == "cppn" else None,
        "pop": args.pop,
        "generations_run": len(history),
        "train_steps": args.steps,
        "seconds": round(time.perf_counter() - started, 1),
        "champion": {
            "metrics": champion.metrics,
            "complexity": champion.genome.complexity(),
            "hidden_nodes": len(champion.genome.hidden_ids),
            "activations": sorted(champion.genome.nodes[node_id].activation for node_id in champion.genome.hidden_ids),
            "genome": genome_to_dict(champion.genome),
        },
        "history": history,
    }
    if args.arm == "cppn" and isinstance(adapter, CppnTaskAdapter):
        report["transfer"] = transfer_curve(champion.genome, adapter, args.refit_steps)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Lever-E CPPN spike runner (ai/cppn_spike_plan.md)")
    parser.add_argument("--arm", choices=("cppn", "direct"), default="cppn")
    parser.add_argument("--phenotype-activation", choices=("sin", "tanh"), default="sin")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--pop", type=int, default=64)
    parser.add_argument("--generations", type=int, default=240)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train-kind", choices=("gradient", "gradient_scheduled"), default="gradient")
    parser.add_argument("--refit-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accept", type=float, default=0.95)
    parser.add_argument("--stall", type=int, default=40, help="flatline generations before early exit (0 = run the full budget)")
    parser.add_argument("--pareto", action="store_true", help="the G1 escalation tables (nsga2 + novelty + connection_cost)")
    parser.add_argument("--seed-fixture", action="store_true", help="graft the E0 Fourier topology into the initial population (the motif-seeding probe)")
    parser.add_argument("--offline", action="store_true", help="pinned synthetic spiral instead of the HF rung-3 task")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    report = run_arm(args)
    variant = f"{'_pareto' if args.pareto else ''}{'_seeded' if args.seed_fixture else ''}{'_scheduled' if args.train_kind == 'gradient_scheduled' else ''}"
    label = f"{args.arm}{'_' + args.phenotype_activation if args.arm == 'cppn' else ''}{variant}_seed{args.seed}{'_offline' if args.offline else ''}"
    out_path = args.report or Path("results") / "cppn_spike" / f"{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    champion = report["champion"]
    console.print(f"[bold]{label}[/bold]: query {champion['metrics'].get('query_accuracy', 0.0):.4f} | complexity {champion['complexity']} | report {out_path}")


if __name__ == "__main__":
    main()
