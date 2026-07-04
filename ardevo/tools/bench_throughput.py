"""Throughput micro-benchmarks. Run: uv run benchmark

T2: serial fill/restore weight-sample evaluation vs the stacked BatchedGraphNet fast path.
L1/L2/L3: library I/O against a temp-dir synthetic library at N entries: per-call add() and
    bump_stats() cost (each rewrites index.json in full), repeated load() of the same keys
    (the entry-cache metric), and query() latency.

Fully offline (synthetic tasks/payloads); numbers go into commit messages, not committed artifacts.

MEASURED RESULTS (M4 Max, torch 2.12, 2026-07-04):
  T2 stacked sample-eval: LOSES at current scales (0.44x/0.33x/0.20x at widths 16/64/256; full-width
  level math costs D times the serial path's column-sliced FLOPs). Default stays OFF.
  Library I/O BEFORE the entry cache + deferred stats: add/bump per call 1.2/1.2 ms at 100 entries,
  2.7/2.8 at 300, 8.4/8.6 at 1000 (the O(entries) full index rewrite dominates); load 0.05 ms/call
  flat (uncached re-read), query(limit=8) 0.5/0.6/0.9 ms at 100/300/1000.
  AFTER: load 0.003 ms/call (cached), query(limit=8) 0.05/0.15/0.49 ms, bump_stats ~0.00 ms/call
  (deferred; flush_stats pays one index write per task). add() keeps the immediate O(entries)
  rewrite by design: structural admissions are rare and must be durable.
Removed: the T1 thread-parallel assess bench. The path itself was removed 2026-07-04; it measured
0.51x at 2 workers down to 0.12x at 12 (2026-06-11, tiny dispatch-bound kernels). See git history.
The throughput lever that DOES pay is the partitioned gradient_batched trainer on direct
populations: 1.5x CPU / 2.1x MPS at pop 48 (measured in phase 3).
"""

import random
import statistics
import tempfile
import time

import torch

from ardevo.dataset.icarus import Axis, Field, Level0Encoder, Task, TaskKind, TaskMeta, ValueType, encode_task
from ardevo.evolution.evaluate import weight_samples
from ardevo.evolution.evolver import TaskAdapter
from ardevo.evolution.genome import InnovationTracker
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
    bench_library()


if __name__ == "__main__":
    main()
