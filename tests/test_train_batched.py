"""gradient_batched: numerical parity with the sequential path, fallback semantics, evolver wiring."""

import random

from tests.test_substrate_batched import _mixed_activation_genome
from versal.evaluation import support_loss
from versal.evolution.evolver import TaskAdapter
from versal.evolution.genome import Genome
from versal.evolution.registry import build_evolver
from versal.evolution.train import gradient, gradient_batched, gradient_scheduled, gradient_scheduled_population, last_batch_stats


def _clones(adapter: TaskAdapter, genomes: list[Genome]) -> tuple[list[Genome], list]:
    return list(genomes), [adapter.decode(genome) for genome in genomes]


def test_batched_training_matches_sequential(xor_adapter: TaskAdapter, linear_genome: Genome, solving_genome: Genome) -> None:
    genomes = [linear_genome, solving_genome, _mixed_activation_genome()]
    rng = random.Random(0)

    sequential = []
    for genome in genomes:
        module = xor_adapter.decode(genome)
        tuned_genome, tuned_module = gradient(genome, module, xor_adapter.encoded, rng=rng, steps=50, lr=0.05, writeback=True)
        sequential.append((tuned_genome, tuned_module))

    batch_genomes, batch_modules = _clones(xor_adapter, genomes)
    batched = gradient_batched(batch_genomes, batch_modules, xor_adapter.encoded, rng=rng, steps=50, lr=0.05, writeback=True, device="cpu")

    assert last_batch_stats["fallback"] == 0.0
    for (seq_genome, seq_module), (bat_genome, bat_module) in zip(sequential, batched):
        seq_loss = float(support_loss(seq_module, xor_adapter.encoded).detach())
        bat_loss = float(support_loss(bat_module, xor_adapter.encoded).detach())
        assert abs(seq_loss - bat_loss) < 1e-4
        seq_weights = seq_module.export_weights()
        bat_weights = bat_module.export_weights()
        assert seq_weights.keys() == bat_weights.keys()
        assert all(abs(seq_weights[key] - bat_weights[key]) < 1e-4 for key in seq_weights)
        # Lamarckian writeback parity on the genome genes themselves.
        seq_by_key = {(c.in_id, c.out_id): c.weight for c in seq_genome.connections if c.enabled}
        bat_by_key = {(c.in_id, c.out_id): c.weight for c in bat_genome.connections if c.enabled}
        assert all(abs(seq_by_key[key] - bat_by_key[key]) < 1e-4 for key in seq_by_key)


def test_scheduled_microbatches_match_sequential_schedule(xor_adapter: TaskAdapter, linear_genome: Genome, solving_genome: Genome) -> None:
    genomes = [linear_genome, solving_genome, _mixed_activation_genome()]
    schedule = {"steps": 31, "lr": 0.05, "warmup_fraction": 0.2, "final_lr_fraction": 0.17, "writeback": True}
    sequential = [gradient_scheduled(genome, xor_adapter.decode(genome), xor_adapter.encoded, rng=random.Random(0), **schedule) for genome in genomes]
    batched = gradient_scheduled_population(
        list(genomes),
        [xor_adapter.decode(genome) for genome in genomes],
        xor_adapter.encoded,
        rng=random.Random(0),
        device="cpu",
        microbatch_size=2,
        **schedule,
    )

    assert last_batch_stats["microbatches"] == 2.0
    assert last_batch_stats["largest_microbatch"] == 2.0
    for (_seq_genome, seq_module), (_bat_genome, bat_module) in zip(sequential, batched):
        seq_weights = seq_module.export_weights()
        bat_weights = bat_module.export_weights()
        assert seq_weights.keys() == bat_weights.keys()
        assert max(abs(seq_weights[key] - bat_weights[key]) for key in seq_weights) < 1e-4


def test_adaptive_microbatch_halves_after_oom(monkeypatch, xor_adapter: TaskAdapter, linear_genome: Genome, solving_genome: Genome) -> None:
    from versal import substrate_batched

    genomes = [linear_genome, solving_genome, _mixed_activation_genome()]
    original_init = substrate_batched.BatchedGraphNet.__init__

    def limited_init(self, nets, device=None) -> None:
        if len(nets) > 1:
            raise RuntimeError("CUDA out of memory")
        original_init(self, nets, device=device)

    monkeypatch.setattr(substrate_batched.BatchedGraphNet, "__init__", limited_init)
    results = gradient_batched(
        list(genomes),
        [xor_adapter.decode(genome) for genome in genomes],
        xor_adapter.encoded,
        rng=random.Random(0),
        steps=3,
        lr=0.05,
        writeback=False,
        device="cpu",
    )

    assert len(results) == len(genomes)
    assert last_batch_stats["oom_retries"] == 2.0
    assert last_batch_stats["oom_fallbacks"] == 0.0
    assert last_batch_stats["largest_microbatch"] == 1.0


def test_padded_size_guard_falls_back_to_sequential(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    genomes = [solving_genome, solving_genome.clone()]
    _, modules = _clones(xor_adapter, genomes)
    before = float(support_loss(modules[0], xor_adapter.encoded).detach())
    results = gradient_batched(genomes, modules, xor_adapter.encoded, rng=random.Random(0), steps=30, lr=0.05, writeback=False, device="cpu", max_padded_nodes=2)
    assert last_batch_stats["fallback"] == 1.0
    after = float(support_loss(results[0][1], xor_adapter.encoded).detach())
    assert after < before  # the sequential path still trained


def test_product_genomes_fall_back(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    from tests.test_aggregation import _product_xor_genome

    genomes = [solving_genome, _product_xor_genome()]
    _, modules = _clones(xor_adapter, genomes)
    gradient_batched(genomes, modules, xor_adapter.encoded, rng=random.Random(0), steps=5, lr=0.05, writeback=False, device="cpu")
    assert last_batch_stats["fallback"] == 1.0  # product nodes change the math: sequential path owns them


def _config() -> dict:
    return {
        "seed": 0,
        "evolution": {
            "pop_size": 6,
            "elitism": 1,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "none"},
            "mutation": {"operators": ["add_rich_node", "add_connection"], "add_rich_node_prob": 0.3, "add_connection_prob": 0.2},
            "train": {"kind": "gradient_batched", "steps": 15, "lr": 0.05, "writeback": True, "device": "cpu"},
        },
        "fitness": {"components": ["support_accuracy"]},
    }


def test_build_evolver_resolves_population_trainer(xor_adapter: TaskAdapter) -> None:
    evolver = build_evolver(_config())
    assert evolver.train_population_op is not None
    state = evolver.seed_state(xor_adapter, random.Random(0))
    assert len(state.population) == 6
    assert evolver.assess_stats["fallback"] == 0.0
    evolver.advance(state, xor_adapter)
    assert len(state.population) == 6
    assert all(item.metrics["support_loss"] >= 0.0 for item in state.population)
    assert max(item.fitness for item in state.population) > 0.0  # gradient training is doing work


def test_scheduled_population_trainer_is_opt_in_for_existing_configs() -> None:
    config = _config()
    config["evolution"]["train"] = {"kind": "gradient_scheduled", "steps": 15, "lr": 0.05, "warmup_fraction": 0.2}
    assert build_evolver(config).train_population_op is None

    config["evolution"]["train"]["batched"] = True
    evolver = build_evolver(config)
    assert evolver.train_population_op is not None
    assert getattr(evolver.train_population_op, "keywords", {})["warmup_fraction"] == 0.2
    assert getattr(evolver.train_op, "keywords", {})["warmup_fraction"] == 0.2


def test_assess_many_equals_sequential_assess(xor_adapter: TaskAdapter, linear_genome: Genome, solving_genome: Genome) -> None:
    batched_evolver = build_evolver(_config())
    sequential_config = _config()
    sequential_config["evolution"]["train"] = {"kind": "gradient", "steps": 15, "lr": 0.05, "writeback": True}
    sequential_evolver = build_evolver(sequential_config)

    genomes = [linear_genome, solving_genome]
    batched_state = batched_evolver.seed_state(xor_adapter, random.Random(0))
    sequential_state = sequential_evolver.seed_state(xor_adapter, random.Random(0))
    batched = batched_evolver.assess_many(genomes, xor_adapter, batched_state)
    sequential = [sequential_evolver.assess(genome, xor_adapter, sequential_state) for genome in genomes]
    for item_b, item_s in zip(batched, sequential):
        assert abs(item_b.fitness - item_s.fitness) < 1e-4
        assert abs(item_b.metrics["support_loss"] - item_s.metrics["support_loss"]) < 1e-4
