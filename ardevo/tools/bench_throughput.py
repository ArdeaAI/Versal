"""Throughput micro-benchmarks for the phase-4 compute work. Run: uv run benchmark

T1: thread-parallel candidate assessment in the hierarchical loop (workers x wall-clock, with the
    determinism claim MEASURED: fitness lists must be identical across worker counts).
T2: serial fill/restore weight-sample evaluation vs the stacked BatchedGraphNet fast path.

Fully offline (synthetic binary tasks); numbers go into the PR description, not committed artifacts.

MEASURED RESULTS (M4 Max, torch 2.12, 2026-06-11): both fancy paths LOSE at current scales.
  T2 stacked sample-eval: 0.44x/0.33x/0.21x at widths 16/64/256 (full-width level math costs D times
  the serial path's column-sliced FLOPs; construction per call adds more). Default stays OFF.
  T1 thread-parallel assess: 0.51x at 2 workers down to 0.12x at 12, at widths 16 AND 256 (tiny
  kernels are GIL/dispatch-bound; torch only releases the GIL inside kernels that take microseconds
  here). Default stays 0. Re-run this bench before flipping either knob for wider rungs.
The throughput lever that DOES pay is the partitioned gradient_batched trainer on flat/direct
populations: 1.5x CPU / 2.1x MPS at pop 48 (measured in phase 3).
"""

import random
import statistics
import time

import torch

from ardevo.dataset.icarus import Axis, Field, Level0Encoder, Task, TaskKind, TaskMeta, ValueType, encode_task
from ardevo.evolution.evaluate import weight_samples
from ardevo.evolution.evolver import TaskAdapter
from ardevo.evolution.genome import InnovationTracker
from ardevo.evolution.loop import CompTaskSpec, HierarchicalLoop
from ardevo.evolution.mutation import MutationContext, add_deep_node, add_rich_node
from ardevo.evolution.registry import build_loop
from ardevo.library import task_io


def synthetic_task(width: int, rows: int = 200, seed: int = 0) -> Task:
    rng = random.Random(seed)
    pairs = []
    for _ in range(rows):
        bits = [float(rng.getrandbits(1)) for _ in range(width)]
        x = Field(torch.tensor(bits), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
        y = Field(torch.tensor([float(int(sum(bits)) % 2)]), (Axis.EXTRA,), ValueType.BINARY, None, None, None)
        pairs.append((x, y))
    meta = TaskMeta(rung=0, kind=TaskKind.MAP, name=f"bench_w{width}")
    split = int(rows * 0.8)
    return Task(meta=meta, support=pairs[:split], query=pairs[split:])


def loop_config(workers: int) -> dict:
    return {
        "seed": 0,
        "evolution": {
            "loop": "hierarchical",
            "parallel_assess": workers,
            "pop_size": 32,
            "elitism": 1,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 3},
            "crossover": {"kind": "neat", "rate": 0.2},
            "mutation": {"operators": ["add_rich_node", "add_connection"], "add_rich_node_prob": 0.3, "add_connection_prob": 0.2},
            "train": {"kind": "gradient", "steps": 40, "lr": 0.03},
            "evaluate": {"kind": "hybrid"},
            "speciation": {"kind": "neat", "target_species": 4},
            "composition": {
                "pop_size": 12,
                "elitism": 2,
                "selection": {"kind": "tournament", "tournament_size": 2},
                "crossover": {"kind": "comp_neat", "rate": 0.3},
                "mutation": {"operators": ["add_module_node", "add_comp_edge", "perturb_glue"], "add_module_node_prob": 0.4, "add_comp_edge_prob": 0.2, "perturb_glue_prob": 0.6},
            },
            "modules": {"pop_size": 32, "elitism": 1, "in_ports": 4, "out_ports": 2, "advance_every": 3},
        },
        "fitness": {
            "components": ["negative_support_loss", "support_accuracy", "hidden_penalty"],
            "w_negative_support_loss": 2.0,
            "w_support_accuracy": 1.0,
            "w_hidden_penalty": 0.05,
        },
    }


def spec_for(task: Task) -> CompTaskSpec:
    io = task_io(task)
    width = io["inputs"][0]["width"]
    signature = io["inputs"][0]["signature"]
    encoder = Level0Encoder(max_flat_dim=width)
    return CompTaskSpec(
        encoded=encode_task(task, encoder),
        encoder=encoder,
        n_inputs=width,
        input_specs=[(signature, width)],
        bank_columns={signature: list(range(width))},
        output_ref=task.meta.name,
        output_width=io["output"]["width"],
    )


def bench_t1(width: int = 16, reps: int = 3) -> None:
    print(f"\nT1: hierarchical brood assessment, input width {width}, comp pop 12, train steps 40")
    print(f"{'workers':>8} {'median s':>10} {'speedup':>8}")
    task = synthetic_task(width)
    spec = spec_for(task)
    baseline = None
    reference_fitness: list[float] | None = None
    for workers in (1, 2, 4, 8, 10, 12):
        loop = build_loop(loop_config(workers))
        assert isinstance(loop, HierarchicalLoop)
        state = loop.fresh_state(random.Random(0))
        assessed = loop._assess_all(
            [comp for comp in _brood(loop, spec, state)],
            spec,
            state,
            train=True,
        )
        fitness = [item.fitness for item in assessed]
        if reference_fitness is None:
            reference_fitness = fitness
        assert fitness == reference_fitness, "parallel assessment diverged from serial"
        timings = []
        for _ in range(reps):
            state = loop.fresh_state(random.Random(0))
            brood = _brood(loop, spec, state)
            start = time.perf_counter()
            loop._assess_all(brood, spec, state, train=True)
            timings.append(time.perf_counter() - start)
        median = statistics.median(timings)
        baseline = baseline or median
        print(f"{workers:>8} {median:>10.3f} {baseline / median:>7.2f}x")


def _brood(loop: HierarchicalLoop, spec: CompTaskSpec, state) -> list:
    from ardevo.evolution.composition import minimal_composition

    rng = random.Random(7)
    return [minimal_composition(spec.input_specs, spec.output_ref, spec.output_width, state.comp_innovations, rng) for _ in range(12)]


def bench_t2(widths: tuple[int, ...] = (16, 64, 256), calls: int = 50) -> None:
    print("\nT2: weight-sample evaluation, serial fill/restore vs stacked batch (6 samples)")
    print(f"{'width':>6} {'nodes':>6} {'serial ms':>10} {'stacked ms':>11} {'speedup':>8}")
    for width in widths:
        task = synthetic_task(width, rows=200, seed=1)
        encoder = Level0Encoder(max_flat_dim=width)
        encoded = encode_task(task, encoder)
        from ardevo.evaluation import input_width, output_features
        from ardevo.evolution.init import minimal

        adapter = TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded))
        rng = random.Random(0)
        genome = minimal(adapter.n_inputs, adapter.n_outputs, rng=rng, default_activation="tanh", weight_scale=1.0)
        ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh", "relu"], "tanh")
        for _ in range(10):
            genome = add_rich_node(genome, ctx, rng=rng, prob=1.0, fan_in=6)
            genome = add_deep_node(genome, ctx, rng=rng, prob=1.0, fan_in=6, fan_out=3)
        module = adapter.decode(genome)

        rows = {}
        for label, batched in (("serial", False), ("stacked", True)):
            start = time.perf_counter()
            for _ in range(calls):
                weight_samples(genome, module, adapter, batched_samples=batched)
            rows[label] = (time.perf_counter() - start) / calls * 1000.0
        print(f"{width:>6} {len(genome.nodes):>6} {rows['serial']:>10.2f} {rows['stacked']:>11.2f} {rows['serial'] / rows['stacked']:>7.2f}x")


def main() -> None:
    torch.set_num_threads(1)
    print(f"torch {torch.__version__}, intra-op threads pinned to 1")
    bench_t2()
    bench_t1()


if __name__ == "__main__":
    main()
