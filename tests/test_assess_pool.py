"""Process-pool assessment is stream-identical to serial; mixed generations partition."""

import random
from typing import Any, cast

from ardevo.dataset.icarus import Task
from ardevo.evolution.genome import Genome, genome_to_dict
from ardevo.evolution.registry import build_evolver, build_loop
from ardevo.evolution.train import gradient, gradient_batched, last_batch_stats
from tests.test_hierarchical_loop import _config as _loop_config
from tests.test_hierarchical_loop import _spec


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


def test_adapter_spill_is_cached_resolves_exactly_and_cleans_up(xor_adapter, tmp_path) -> None:
    """The pool pickles an AdapterRef (a path) instead of the encoded tensors; workers resolve it
    through a one-slot cache; close_pool removes the spill file."""
    import torch

    from ardevo.evolution import evolver as ev_mod
    from ardevo.evolution.evolver import AdapterRef

    config = {
        "seed": 0,
        "library_dir": str(tmp_path),
        "evolution": {
            "pop_size": 2,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "none"},
            "mutation": {"operators": []},
            "train": {"kind": "gradient", "steps": 2, "lr": 0.05},
        },
        "fitness": {"components": ["support_accuracy"]},
    }
    evolver = build_evolver(config)
    ref = evolver._pooled_adapter(xor_adapter)
    assert isinstance(ref, AdapterRef)
    assert ref.path.startswith(str(tmp_path)) and "encoded_cache" in ref.path

    again = evolver._pooled_adapter(xor_adapter)
    assert isinstance(again, AdapterRef) and again.path == ref.path  # identity slot: spilled once

    loaded = ev_mod._resolve_adapter(ref)
    assert torch.equal(loaded.encoded.support_input[0], xor_adapter.encoded.support_input[0])
    assert loaded.n_inputs == xor_adapter.n_inputs and loaded.n_outputs == xor_adapter.n_outputs
    assert ev_mod._resolve_adapter(ref) is loaded  # one-slot worker cache: no second disk read

    from pathlib import Path

    evolver.close_pool()
    assert not Path(ref.path).exists()


def test_task_switch_releases_main_and_worker_adapter_before_loading_next(xor_adapter, tmp_path, monkeypatch) -> None:
    """A different task never overlaps the previous encoded payload in either process slot."""
    from pathlib import Path

    import torch

    from ardevo.evolution import evolver as ev_mod
    from ardevo.evolution.evolver import AdapterRef

    config = {
        "seed": 0,
        "library_dir": str(tmp_path),
        "evolution": {
            "pop_size": 2,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "none"},
            "mutation": {"operators": []},
            "train": {"kind": "none"},
        },
        "fitness": {"components": ["support_accuracy"]},
    }
    evolver = build_evolver(config)
    ref = evolver._pooled_adapter(xor_adapter)
    assert isinstance(ref, AdapterRef)
    spill = ref.path
    assert Path(spill).exists()
    evolver.release_task_adapter()
    assert not Path(spill).exists() and getattr(evolver, "_adapter_spill", None) is None

    ev_mod._WORKER_ADAPTER = ("old-task.pt", xor_adapter)

    def load_after_release(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202 - torch.load seam
        assert ev_mod._WORKER_ADAPTER is None
        return xor_adapter

    monkeypatch.setattr(torch, "load", load_after_release)
    try:
        assert ev_mod._resolve_adapter(AdapterRef("new-task.pt")) is xor_adapter
    finally:
        ev_mod._WORKER_ADAPTER = None


def test_shared_task_release_collects_an_acknowledgement_from_every_worker() -> None:
    from types import SimpleNamespace

    from ardevo.evolution import evolver as ev_mod

    class FakePool:
        _pool = [SimpleNamespace(pid=10, is_alive=lambda: True), SimpleNamespace(pid=11, is_alive=lambda: True)]

        def map(self, function, values, *, chunksize):  # noqa: ANN001, ANN201 - multiprocessing Pool seam
            assert function is ev_mod._release_worker_task_adapter and list(values) == [0, 1, 2, 3] and chunksize == 1
            return [10, 11, 10, 11]

    previous = ev_mod.get_shared_pool()
    ev_mod.set_shared_pool(cast(Any, FakePool()))
    try:
        ev_mod.release_shared_task_adapters()
    finally:
        ev_mod.set_shared_pool(previous)


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
