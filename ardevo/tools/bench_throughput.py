"""Throughput micro-benchmarks. Run: uv run benchmark

T2: serial fill/restore weight-sample evaluation vs the stacked BatchedGraphNet fast path.
T3: per-genome decode / gradient-train / hybrid-eval cost at image-rung widths (784/3072) under
    worker conditions (1 thread, support batch 320). Tracks the compact-column substrate.
T4: population-batched training vs the serial loop per device (cpu/mps/cuda) + numeric drift.
T5: assess_many end-to-end, pooled per-genome path vs the hybrid (device batch + pool) path.
    The go/no-go for `[orchestrator.direct.train] batched` on the current machine.
L1/L2/L3: library I/O against a temp-dir synthetic library at N entries: per-call add() and
    bump_stats() cost (each rewrites index.json in full), repeated load() of the same keys
    (the entry-cache metric), and query() latency.

Fully offline (synthetic tasks/payloads); numbers go into commit messages, not committed artifacts.

MEASURED RESULTS (M4 Max, torch 2.12, 2026-07-04; T7 2026-07-05):
  THE SECOND PARITY WEDGE (same day, self-inflicted, fixed): wrapping `would_create_cycle` around
  a per-call ForwardReachability turned `make_acyclic` (per module child per generation,
  loop.py) into a constructor/GC storm: run forensics showed parity/two_spirals spending
  849s/831s of ~860s tasks in the COMPOSITION stage. Fix: restored the early-exit BFS body, plus
  make_acyclic got a topological_order fast path + an incremental-adjacency repair. Measured at a
  mature 525-gene module: 0.06 ms/call (~4 ms per 64-child generation); the ORIGINAL per-edge
  rebuild would have been ~3-6 s/generation at that size, so the fast path also removes a latent
  scaling wall. Decisions are reference-pinned in tests/test_mutation_equivalence.py.
  T7 mutation pipeline (the image-rung WEDGE fix): pre-fix ONE add_local_connection call was
  0.3s at width 784 / 4.9s at 3072 (per-pair has_connection scan + adjacency-rebuild BFS across
  the full source x target sweep, on the MAIN thread), ~6 firing calls/generation = minutes per
  CIFAR generation and an 8h wedged task. Post-fix (per-call existing-edge set +
  ForwardReachability memo, bitwise-identical children): full 64-child pipeline = 23.6 ms/gen at
  784, 144 ms/gen at 3072 (~200x at 3072). tests/test_mutation_equivalence.py pins equivalence
  and a 1s speed guard. Numpy-vectorizing the pair sweep was CONSIDERED AND DROPPED by
  measurement (2026-07-05, rung-11 scale, 108300 inputs x 100 outputs): minimal init is a 10.8M
  bipartite gene list (7s to build, GBs per genome), so post-fix mutation cost there (~4s/call)
  is O(E) preprocessing over those genes, not the sweep, and the rung is unrunnable at population
  scale regardless: it belongs to the inputs-x-outputs init regime wall (rungs 11-14), an
  algorithm-level design problem, not a code-speed one.
  T3 compact-column substrate (dense [n,n] at 9116b08 -> slim [n,h]; grown genomes, h=13, 20 train
  steps, batch 320, 1 thread): width 784 train 399.6 -> 30.0 ms (13.3x), hybrid eval 7.6 -> 5.9 ms;
  width 3072 train 464.4 -> 81.4 ms (5.7x), hybrid eval 45.8 -> 17.3 ms (2.6x). Forward and
  feedforward training are BITWISE equal to the dense layout; recurrent-substrate training matches
  to 1 ulp (backward reduction grouping; see tests/test_substrate_slim.py).
  T4/T5 hybrid population training (slim substrate era): drift <= 1.5e-4 everywhere. T4 vs a
  1-thread serial loop at P=48/20 steps: cpu 1.09x (784) / 0.86x (3072); mps 1.14x (784) / 2.59x
  (3072). T5 END-TO-END vs the 12-worker pool (P=48, 120 steps): width 784 pooled 1481-1556 ms vs
  hybrid-mps 1800-1838 ms; width 3072 pooled 4468 ms vs hybrid-mps 6069 ms (re-measured
  2026-07-05 post-mutation-fix). The pool WINS at both image widths on M4 Max (h~13 keeps the GPU
  dispatch-bound), so the overmind config ships `batched = false`. The `min_batch_nodes` width
  floor exists so a machine where the device DOES win (LatticeCUDA, or once structures grow wide)
  can batch only above the measured break-even; re-run T4/T5 there before flipping.
  T2 stacked sample-eval on the compact layout: 0.56x/0.68x/0.83x at widths 16/64/256, breaks even
  at image widths (1.04x @784, 1.14x @3072). `batched_samples = "auto"` engages it at >= 768 nodes
  (STACKED_AUTO_MIN_NODES); dense-era truth was 0.44x/0.33x/0.20x (full-width level math).
  Library I/O BEFORE the entry cache + deferred stats: add/bump per call 1.2/1.2 ms at 100 entries,
  2.7/2.8 at 300, 8.4/8.6 at 1000 (the O(entries) full index rewrite dominates); load 0.05 ms/call
  flat (uncached re-read), query(limit=8) 0.5/0.6/0.9 ms at 100/300/1000.
  AFTER: load 0.003 ms/call (cached), query(limit=8) 0.05/0.15/0.49 ms, bump_stats ~0.00 ms/call
  (deferred; flush_stats pays one index write per task). add() keeps the immediate O(entries)
  rewrite by design: structural admissions are rare and must be durable.
Removed: the T1 thread-parallel assess bench. The path itself was removed 2026-07-04; it measured
0.51x at 2 workers down to 0.12x at 12 (2026-06-11, tiny dispatch-bound kernels). See git history.
The throughput lever that DOES pay is the partitioned gradient_batched trainer on direct
populations: 1.5x CPU / 2.1x MPS at pop 48 (measured in phase 3, dense layout).
"""

import random
import statistics
import tempfile
import time

import torch

from ardevo.dataset.icarus import Axis, Field, Level0Encoder, Task, TaskKind, TaskMeta, ValueType, encode_task
from ardevo.evolution.evaluate import weight_samples
from ardevo.evolution.evolver import TaskAdapter
from ardevo.evolution.genome import Genome, InnovationTracker
from ardevo.evolution.mutation import MutationContext, add_deep_node, add_rich_node
from ardevo.library import MODULE, ModuleLibrary


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


def bench_t3(widths: tuple[int, ...] = (784, 3072)) -> None:
    """Per-genome cost at image-rung widths under worker conditions (the direct strategy's unit of
    pooled work): decode, 20 gradient steps, one hybrid evaluation."""
    from ardevo.evolution.evaluate import hybrid
    from ardevo.evolution.train import gradient

    print("\nT3: per-genome decode / train(20 steps) / hybrid-eval at image widths (batch 320, 1 thread)")
    print(f"{'width':>6} {'n':>6} {'h':>4} {'decode ms':>10} {'train ms':>9} {'eval ms':>8}")
    warmed_up = False
    for width in widths:
        task = synthetic_task(width, rows=400, seed=width)
        encoder = Level0Encoder(max_flat_dim=width)
        encoded = encode_task(task, encoder)
        from ardevo.evaluation import input_width, output_features

        adapter = TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded))
        rng = random.Random(width)
        genome = minimal_grown(adapter, rng)

        start = time.perf_counter()
        for _ in range(3):
            module = adapter.decode(genome)
        decode_ms = (time.perf_counter() - start) / 3 * 1000.0
        module = adapter.decode(genome)
        if not warmed_up:  # the process's first backward pays ~300ms of lazy torch init; not the metric
            gradient(genome.clone(), adapter.decode(genome), encoded, rng=random.Random(0), steps=2, lr=0.03, weight_decay=0.0002)
            warmed_up = True
        start = time.perf_counter()
        gradient(genome.clone(), module, encoded, rng=random.Random(0), steps=20, lr=0.03, weight_decay=0.0002)
        train_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        hybrid(genome, module, adapter)
        eval_ms = (time.perf_counter() - start) * 1000.0
        print(f"{width:>6} {module.n:>6} {module.h:>4} {decode_ms:>10.1f} {train_ms:>9.1f} {eval_ms:>8.1f}")


def minimal_grown(adapter: TaskAdapter, rng: random.Random, rounds: int = 6) -> Genome:
    from ardevo.evolution.init import minimal
    from ardevo.evolution.mutation import add_connection

    genome = minimal(adapter.n_inputs, adapter.n_outputs, rng=rng, default_activation="tanh", weight_scale=1.0)
    ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh", "relu"], "tanh")
    for _ in range(rounds):
        genome = add_rich_node(genome, ctx, rng=rng, prob=1.0, fan_in=5)
        genome = add_deep_node(genome, ctx, rng=rng, prob=1.0, fan_in=5, fan_out=3)
        genome = add_connection(genome, ctx, rng=rng, prob=1.0)
    return genome


def bench_t4(widths: tuple[int, ...] = (784, 3072), population: int = 48, steps: int = 20) -> None:
    """Population-batched training vs the per-genome serial loop at image widths, per device.
    Also prints the CPU-vs-GPU trained-weight drift (the numeric canary; must stay <= 1e-3)."""
    import torch as _torch

    from ardevo.evolution.train import gradient, gradient_refine_population

    devices = ["cpu"]
    if _torch.backends.mps.is_available():
        devices.append("mps")
    if _torch.cuda.is_available():
        devices.append("cuda")

    print(f"\nT4: population train (P={population}, {steps} steps, batch 320) vs serial loop, per device")
    print(f"{'width':>6} {'device':>7} {'batched ms':>11} {'serial ms':>10} {'speedup':>8} {'drift':>10}")
    for width in widths:
        task = synthetic_task(width, rows=400, seed=width)
        encoder = Level0Encoder(max_flat_dim=width)
        encoded = encode_task(task, encoder)
        from ardevo.evaluation import input_width, output_features

        adapter = TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded))
        genomes = [minimal_grown(adapter, random.Random(width * 1000 + i), rounds=3) for i in range(population)]

        start = time.perf_counter()
        serial_results = [gradient(g.clone(), adapter.decode(g), encoded, rng=random.Random(0), steps=steps, lr=0.03, weight_decay=0.0002) for g in genomes]
        serial_ms = (time.perf_counter() - start) * 1000.0

        reference_weights: list[dict] | None = None
        for device in devices:
            modules = [adapter.decode(g) for g in genomes]
            start = time.perf_counter()
            pairs = gradient_refine_population(
                [g.clone() for g in genomes], modules, encoded, rng=random.Random(0), steps=steps, lr=0.03, weight_decay=0.0002, device=device, max_padded_nodes=4096
            )
            batched_ms = (time.perf_counter() - start) * 1000.0
            exported = [module.export_weights() for _genome, module in pairs]
            if device == "cpu":
                reference_weights = exported
                drift = max(abs(sm.export_weights()[key] - b[key]) for (_g, sm), b in zip(serial_results, exported) for key in b)
            else:
                assert reference_weights is not None
                drift = max(abs(r[key] - b[key]) for r, b in zip(reference_weights, exported) for key in b)
            print(f"{width:>6} {device:>7} {batched_ms:>11.0f} {serial_ms:>10.0f} {serial_ms / batched_ms:>7.2f}x {drift:>10.2e}")


def bench_t5(width: int = 784, population: int = 48, steps: int = 120, workers: int = 12) -> None:
    """End-to-end assess_many: the pooled per-genome path (batched=false) vs the hybrid path
    (population program on the resolved device + pool for serial subset and evaluation).
    The go/no-go gate for shipping `batched = true` in the overmind config."""
    import tempfile

    from ardevo.evolution import evolver as ev_mod
    from ardevo.evolution.evolver import EvolverState
    from ardevo.evolution.genome import InnovationTracker
    from ardevo.evolution.registry import build_evolver

    task = synthetic_task(width, rows=400, seed=width)
    encoder = Level0Encoder(max_flat_dim=width)
    encoded = encode_task(task, encoder)
    from ardevo.evaluation import input_width, output_features

    adapter = TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded))
    genomes = [minimal_grown(adapter, random.Random(width * 77 + i), rounds=3) for i in range(population)]

    print(f"\nT5: assess_many end-to-end (P={population}, {steps} steps, width {width}, {workers} workers)")
    print(f"{'mode':>8} {'device':>7} {'total ms':>9}")
    with tempfile.TemporaryDirectory() as tmp:
        ev_mod.create_assess_pool(workers, tmp)
        try:
            for label, extras in (("pooled", {"batched": False}), ("hybrid", {})):
                config = {
                    "seed": 0,
                    "library_dir": tmp,
                    "machine_env": "MonadMetal",
                    "evolution": {
                        "pop_size": population,
                        "assess_workers": workers,
                        "init": {"kind": "minimal"},
                        "selection": {"kind": "tournament", "tournament_size": 2},
                        "crossover": {"kind": "none"},
                        "mutation": {"operators": []},
                        "train": {"kind": "gradient_refine", "steps": steps, "lr": 0.03, "weight_decay": 0.0002, **extras},
                        "evaluate": {"kind": "hybrid"},
                    },
                    "fitness": {"components": ["query_accuracy"]},
                }
                evolver = build_evolver(config)
                device = "n/a"
                if evolver.train_population_op is not None:
                    device = getattr(evolver.train_population_op, "keywords", {}).get("device", "auto")
                state = EvolverState(population=[], innovations=InnovationTracker.from_genomes(genomes), rng=random.Random(0))
                evolver.assess_many(list(genomes[:2]), adapter, state)  # warm the pool + device
                start = time.perf_counter()
                evolver.assess_many(list(genomes), adapter, state)
                total_ms = (time.perf_counter() - start) * 1000.0
                print(f"{label:>8} {device:>7} {total_ms:>9.0f}")
        finally:
            ev_mod._close_shared_pool()


def bench_t7(children: int = 64) -> None:
    """Full mutation pipeline per generation at image widths: the geometry-mutator wedge tracker.
    Mutation runs SERIALLY on the main thread between pooled assessments, so ms/generation here is
    pure added wall-clock per generation."""
    import tomllib
    from functools import partial

    from ardevo.evolution.init import minimal, stamp_input_coordinates
    from ardevo.evolution.mutation import MUTATION, MutationContext, MutationPipeline, add_local_node, add_rich_node
    from ardevo.evolution.registry import _bind_prefixed

    with open("configs/orchestrated_overmind.toml", "rb") as handle:
        mutation_cfg = tomllib.load(handle)["evolution"]["mutation"]
    pipeline = MutationPipeline([partial(MUTATION.get(name), **_bind_prefixed(mutation_cfg, name)) for name in mutation_cfg["operators"]])

    print(f"\nT7: mutation pipeline per generation ({children} children, overmind operator list)")
    print(f"{'width':>6} {'conns':>6} {'ms/gen':>9} {'ms/child':>9}")
    for width, shape in ((784, (28, 28)), (3072, (3, 32, 32))):
        rng = random.Random(width)
        genome = minimal(width, 1, rng=rng, default_activation="tanh", weight_scale=1.0)
        genome = stamp_input_coordinates(genome, shape)
        ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh", "relu", "sigmoid"], "tanh")
        for _ in range(4):  # a mid-run genome: coordinated hidden nodes so every geometry op is live
            genome = add_rich_node(genome, ctx, rng=rng, prob=1.0, fan_in=4)
            genome = add_local_node(genome, ctx, rng=rng, prob=1.0, fan_in=4)
        start = time.perf_counter()
        for _ in range(children):
            pipeline(genome, ctx, rng=rng)
        total_ms = (time.perf_counter() - start) * 1000.0
        print(f"{width:>6} {len(genome.connections):>6} {total_ms:>9.1f} {total_ms / children:>9.2f}")


def synthetic_payload(rng: random.Random, nodes: int = 24, connections: int = 48) -> dict:
    """A genome-shaped dict of realistic size; rng-varied weights make every payload hash unique."""
    node_rows = [
        {
            "id": i,
            "kind": "input" if i < 3 else ("output" if i < 4 else "hidden"),
            "activation": rng.choice(["tanh", "relu", "sigmoid"]),
            "coordinate": None,
            "aggregation": "sum",
        }
        for i in range(nodes)
    ]
    conn_rows = [
        {"in": rng.randrange(nodes), "out": rng.randrange(nodes), "weight": rng.random(), "enabled": True, "innovation": j, "recurrent": False} for j in range(connections)
    ]
    return {"nodes": node_rows, "connections": conn_rows, "refine_steps": 1, "macros": []}


def bench_library(sizes: tuple[int, ...] = (100, 300, 1000)) -> None:
    print("\nL1/L2/L3: library I/O per call at N entries (add/bump rewrite index.json in full)")
    print(f"{'entries':>8} {'add ms':>8} {'bump ms':>8} {'load ms':>8} {'query ms':>9}")
    widths = (8, 16, 32, 64)
    for n in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            library = ModuleLibrary(tmp)
            rng = random.Random(0)
            add_timings = []
            for i in range(n):
                payload = synthetic_payload(rng)
                io = {"inputs": [{"signature": "binary|E", "width": widths[i % 4]}], "output": {"signature": "binary|E", "width": 1}}
                start = time.perf_counter()
                library.add(entry_type=MODULE, payload=payload, io=io, provenance={"accepted_metric": rng.random(), "weight_robustness": rng.random()})
                add_timings.append(time.perf_counter() - start)
            add_ms = statistics.median(add_timings[-20:]) * 1000.0
            all_keys = library.keys()
            hot_keys = all_keys[:20]
            start = time.perf_counter()
            for _ in range(25):
                for key in hot_keys:
                    library.load(key)
            load_ms = (time.perf_counter() - start) / (25 * len(hot_keys)) * 1000.0
            start = time.perf_counter()
            for i in range(50):
                library.bump_stats(all_keys[i % n], 0.5)
            bump_ms = (time.perf_counter() - start) / 50 * 1000.0
            start = time.perf_counter()
            for _ in range(50):
                library.query(entry_type=MODULE, input_signature="binary|E", input_width=16, output_signature="binary|E", output_width=1, limit=8)
            query_ms = (time.perf_counter() - start) / 50 * 1000.0
            print(f"{n:>8} {add_ms:>8.2f} {bump_ms:>8.2f} {load_ms:>8.3f} {query_ms:>9.2f}")


def main() -> None:
    torch.set_num_threads(1)
    print(f"torch {torch.__version__}, intra-op threads pinned to 1")
    bench_t2()
    bench_t3()
    bench_t4()
    bench_t5()
    bench_t7()
    bench_library()


if __name__ == "__main__":
    main()
