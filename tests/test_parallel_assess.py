"""T1/T4: thread-parallel assessment is stream-identical to serial; mixed generations partition."""

import random

import torch

from ardevo.dataset.icarus import Task
from ardevo.evolution.genome import Genome, genome_to_dict
from ardevo.evolution.registry import build_evolver, build_loop
from ardevo.evolution.train import gradient, gradient_batched, last_batch_stats
from tests.test_hierarchical_loop import _config as _loop_config
from tests.test_hierarchical_loop import _spec


def _pinned_threads() -> int:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    return previous


def test_hierarchical_parallel_assess_matches_serial(decomposable_task: Task) -> None:
    previous = _pinned_threads()
    try:
        outcomes = []
        for workers in (0, 4):
            config = _loop_config()
            config["evolution"]["parallel_assess"] = workers
            loop = build_loop(config)
            state = loop.fresh_state(random.Random(0))
            best = loop.run_task(_spec(decomposable_task), state, budget=2)
            outcomes.append((best.fitness, best.metrics, [genome_to_dict(m.genome) for m in state.modules]))
        assert outcomes[0][0] == outcomes[1][0]
        assert outcomes[0][1] == outcomes[1][1]
        assert outcomes[0][2] == outcomes[1][2]  # module pool evolved identically: rng streams match
    finally:
        torch.set_num_threads(previous)


def test_flat_parallel_assess_matches_serial(xor_adapter, linear_genome: Genome, solving_genome: Genome) -> None:
    previous = _pinned_threads()
    try:
        results = []
        for workers in (0, 4):
            config = {
                "seed": 0,
                "evolution": {
                    "pop_size": 4,
                    "parallel_assess": workers,
                    "init": {"kind": "minimal"},
                    "selection": {"kind": "tournament", "tournament_size": 2},
                    "crossover": {"kind": "none"},
                    "mutation": {"operators": ["add_rich_node"], "add_rich_node_prob": 0.5},
                    "train": {"kind": "gradient", "steps": 10, "lr": 0.05},
                },
                "fitness": {"components": ["support_accuracy"]},
            }
            evolver = build_evolver(config)
            state = evolver.seed_state(xor_adapter, random.Random(0))
            evolver.advance(state, xor_adapter)
            results.append([(item.fitness, genome_to_dict(item.genome)) for item in state.population])
        assert results[0] == results[1]
    finally:
        torch.set_num_threads(previous)


def test_hierarchical_process_pool_matches_serial(decomposable_task: Task, tmp_path) -> None:
    """Composition assessment across the shared process pool is bit-identical to serial: champion
    fitness/metrics AND the evolved module pool match (rng streams match; training is rng-free)."""
    from ardevo.evolution import evolver as ev_mod

    outcomes = []
    try:
        for workers in (0, 2):
            ev_mod.set_shared_pool(None)
            if workers > 1:
                ev_mod.create_assess_pool(workers, str(tmp_path))
            loop = build_loop(_loop_config())
            state = loop.fresh_state(random.Random(0))
            best = loop.run_task(_spec(decomposable_task), state, budget=2)
            outcomes.append((best.fitness, best.metrics, [genome_to_dict(m.genome) for m in state.modules]))
            ev_mod._close_shared_pool()
            ev_mod.set_shared_pool(None)
    finally:
        ev_mod.set_shared_pool(None)
    assert outcomes[0][0] == outcomes[1][0]
    assert outcomes[0][1] == outcomes[1][1]
    assert outcomes[0][2] == outcomes[1][2]  # module pool evolved identically


def test_flat_process_pool_matches_serial(xor_adapter, linear_genome: Genome, solving_genome: Genome, tmp_path) -> None:
    """Process-pool assessment (assess_workers) is bit-identical to serial: independent candidates,
    rng-free training, and re-decoded modules reproduce the exact population."""
    results = []
    for workers in (0, 2):
        config = {
            "seed": 0,
            "library_dir": str(tmp_path),
            "evolution": {
                "pop_size": 6,
                "assess_workers": workers,
                "init": {"kind": "minimal"},
                "selection": {"kind": "tournament", "tournament_size": 2},
                "crossover": {"kind": "none"},
                "mutation": {"operators": ["add_rich_node"], "add_rich_node_prob": 0.5},
                "train": {"kind": "gradient", "steps": 10, "lr": 0.05},
            },
            "fitness": {"components": ["support_accuracy"]},
        }
        evolver = build_evolver(config)
        try:
            state = evolver.seed_state(xor_adapter, random.Random(0))
            evolver.advance(state, xor_adapter)
            results.append([(item.fitness, genome_to_dict(item.genome)) for item in state.population])
        finally:
            evolver.close_pool()
    assert results[0] == results[1]


def test_partitioned_batched_training_handles_mixed_generations(xor_adapter, linear_genome: Genome, solving_genome: Genome) -> None:
    """T4: one non-batchable candidate must not force the whole generation onto the serial path."""
    from tests.test_aggregation import _product_xor_genome

    genomes = [linear_genome, solving_genome, _product_xor_genome()]
    rng = random.Random(0)
    sequential = []
    for genome in genomes:
        module = xor_adapter.decode(genome)
        sequential.append(gradient(genome, module, xor_adapter.encoded, rng=rng, steps=30, lr=0.05, writeback=True))

    batch_modules = [xor_adapter.decode(genome) for genome in genomes]
    mixed = gradient_batched(list(genomes), batch_modules, xor_adapter.encoded, rng=rng, steps=30, lr=0.05, writeback=True, device="cpu")
    assert abs(last_batch_stats["fallback"] - 1.0 / 3.0) < 1e-9  # exactly the product candidate fell back
    assert len(mixed) == 3
    for (seq_genome, seq_module), (mix_genome, mix_module) in zip(sequential, mixed):
        seq_weights = seq_module.export_weights()
        mix_weights = mix_module.export_weights()
        assert seq_weights.keys() == mix_weights.keys()
        assert all(abs(seq_weights[key] - mix_weights[key]) < 1e-4 for key in seq_weights)
