"""The hybrid assess path: population-batched training on the compute device + the process pool
for serial candidates and batch-subset evaluation. Must be BIT-IDENTICAL to the inline population
path (same partition seam, same serial op, writeback float32 round-trips exactly), which is itself
within the 1e-4 batch contract of the fully sequential path (tests/test_train_batched.py)."""

import random

from tests.test_aggregation import _product_xor_genome
from versal.evolution.evolver import EvolverState
from versal.evolution.genome import Genome, InnovationTracker, genome_to_dict
from versal.evolution.mutation import MutationContext, add_recurrent_connection, add_rich_node
from versal.evolution.registry import build_evolver
from versal.evolution.train import TRAIN, TRAIN_POPULATION, gradient_refine


def _config(workers: int, tmp_path, **train_extras) -> dict:
    return {
        "seed": 0,
        "library_dir": str(tmp_path),
        "evolution": {
            "pop_size": 6,
            "assess_workers": workers,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "none"},
            "mutation": {"operators": []},
            "train": {"kind": "gradient_refine", "steps": 10, "lr": 0.05, **train_extras},
        },
        "fitness": {"components": ["support_accuracy"]},
    }


def _refine_genome(base: Genome, seed: int) -> Genome:
    genome = base.clone()
    genome.refine_steps = 3
    rng = random.Random(seed)
    ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh"], "tanh")
    genome = add_rich_node(genome, ctx, rng=rng, prob=1.0, fan_in=3)
    return add_recurrent_connection(genome, ctx, rng=rng, prob=1.0)


def _mixed_generation(linear_genome: Genome, solving_genome: Genome) -> list[Genome]:
    # Two batchable plain cores + a refine candidate and a product candidate (both serial-only).
    return [linear_genome, solving_genome, _refine_genome(linear_genome, seed=3), _product_xor_genome()]


def _assess(workers: int, tmp_path, genomes: list[Genome], xor_adapter, **train_extras):
    evolver = build_evolver(_config(workers, tmp_path, **train_extras))
    assert evolver.train_population_op is not None
    state = EvolverState(population=[], innovations=InnovationTracker.from_genomes(genomes), rng=random.Random(0))
    try:
        return evolver.assess_many(list(genomes), xor_adapter, state), evolver
    finally:
        evolver.close_pool()


def test_hybrid_assess_bit_identical_to_inline_population_path(xor_adapter, linear_genome: Genome, solving_genome: Genome, tmp_path) -> None:
    genomes = _mixed_generation(linear_genome, solving_genome)
    outcomes = []
    for workers in (0, 2):
        results, evolver = _assess(workers, tmp_path, genomes, xor_adapter)
        assert len(results) == len(genomes)
        assert all(item.module is not None for item in results)  # every candidate decoded
        # The refine + product candidates fell to the pool/serial side in BOTH modes.
        assert abs(evolver.assess_stats["fallback"] - 2.0 / 4.0) < 1e-9
        outcomes.append([(item.fitness, item.metrics, genome_to_dict(item.genome)) for item in results])
    assert outcomes[0] == outcomes[1]


def test_hybrid_writeback_false_evaluates_inline_and_matches(xor_adapter, linear_genome: Genome, solving_genome: Genome, tmp_path) -> None:
    genomes = _mixed_generation(linear_genome, solving_genome)
    outcomes = []
    for workers in (0, 2):
        results, _evolver = _assess(workers, tmp_path, genomes, xor_adapter, writeback=False)
        # Without writeback the genome must stay untouched; tuned weights live only on the module.
        assert all(genome_to_dict(item.genome) == genome_to_dict(original) for item, original in zip(results, genomes))
        outcomes.append([(item.fitness, item.metrics) for item in results])
    assert outcomes[0] == outcomes[1]


def test_gradient_refine_registers_in_both_registries() -> None:
    assert "gradient_refine" in TRAIN.names()
    assert "gradient_refine" in TRAIN_POPULATION.names()


def test_min_batch_nodes_floor_sends_small_candidates_to_the_pool(xor_adapter, linear_genome: Genome, solving_genome: Genome, tmp_path) -> None:
    """The width floor: below it the 12-worker pool measured FASTER than the device batch, so
    small candidates must partition serial even when batching is on."""
    from versal.evolution.train import partition_batchable

    modules = [xor_adapter.decode(linear_genome), xor_adapter.decode(solving_genome)]
    cores = [module.core() for module in modules]
    batch, serial = partition_batchable(cores, steps=10, max_padded_nodes=1024)
    assert batch == [0, 1] and serial == []  # no floor: both tiny nets batch
    batch, serial = partition_batchable(cores, steps=10, max_padded_nodes=1024, min_batch_nodes=1000)
    assert batch == [] and serial == [0, 1]  # floored: the pool keeps them

    genomes = _mixed_generation(linear_genome, solving_genome)
    results, evolver = _assess(2, tmp_path, genomes, xor_adapter, min_batch_nodes=1000)
    assert len(results) == len(genomes) and all(item.module is not None for item in results)
    assert evolver.assess_stats["fallback"] == 1.0  # everything went through the pool


def test_sequential_op_keeps_the_same_kind(tmp_path) -> None:
    evolver = build_evolver(_config(0, tmp_path))
    assert evolver.train_population_op is not None
    assert getattr(evolver.train_op, "func", None) is gradient_refine  # deep supervision survives everywhere


def test_batched_kill_switch_restores_pool_only_path(tmp_path) -> None:
    evolver = build_evolver(_config(0, tmp_path, batched=False))
    assert evolver.train_population_op is None
    assert getattr(evolver.train_op, "func", None) is gradient_refine
