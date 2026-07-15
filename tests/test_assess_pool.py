"""Process-pool assessment is stream-identical to serial; mixed generations partition."""

import random
from dataclasses import fields, is_dataclass
from typing import Any, cast

import torch

from ardevo.dataset.icarus import Task
from ardevo.evolution.genome import Genome, genome_to_dict
from ardevo.evolution.loop import _assess_comp_in_worker
from ardevo.evolution.registry import build_evolver, build_loop
from ardevo.evolution.train import gradient, gradient_batched, last_batch_stats
from tests.test_hierarchical_loop import _config as _loop_config
from tests.test_hierarchical_loop import _live_comp, _spec


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
            config = _loop_config()
            config["library_dir"] = str(tmp_path / f"run_{workers}")
            loop = build_loop(config)
            state = loop.fresh_state(random.Random(0))
            spec = _spec(decomposable_task)
            best = loop.run_task(spec, state, budget=2)
            assert best.net is not None
            with torch.no_grad():
                prediction = best.net(spec.encoded.support_input[0]).tolist()
            outcomes.append((best.fitness, best.metrics, [genome_to_dict(m.genome) for m in state.modules], prediction))
            ev_mod._close_shared_pool()
            ev_mod.set_shared_pool(None)
    finally:
        ev_mod.set_shared_pool(None)
    assert outcomes[0][0] == outcomes[1][0]
    assert outcomes[0][1] == outcomes[1][1]
    assert outcomes[0][2] == outcomes[1][2]  # module pool evolved identically
    assert outcomes[0][3] == outcomes[1][3]  # the parent-rebuilt champion is exactly executable


def _assert_tensor_free(value: Any, seen: set[int] | None = None) -> None:
    """Recursively pin the composition worker's process-boundary contract."""
    if isinstance(value, (torch.Tensor, torch.nn.Module, torch.storage.UntypedStorage)):
        raise AssertionError(f"worker result contains torch-owned state: {type(value).__name__}")
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if is_dataclass(value):
        for field in fields(value):
            _assert_tensor_free(getattr(value, field.name), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_tensor_free(key, seen)
            _assert_tensor_free(item, seen)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _assert_tensor_free(item, seen)


def test_composition_worker_result_contains_no_torch_storage(xor_task: Task) -> None:
    """A full composition population must not consume one Linux shared-memory fd per tensor."""
    loop = build_loop(_loop_config())
    state = loop.fresh_state(random.Random(0))
    species_id = sorted(state.species_champions)[0]
    spec = _spec(xor_task)

    result = _assess_comp_in_worker(
        _live_comp(species_id, 2, 1),
        spec=spec,
        species_champions=state.species_champions,
        max_inline_depth=loop.max_inline_depth,
        train=True,
        train_op=loop.evolver.train_op,
        evaluate_op=loop.evolver.evaluate_op,
        fitness=loop.evolver.fitness,
    )

    assert result.net is None and result.live_writebacks is not None
    _assert_tensor_free(result)


def test_composition_pool_sends_encoded_task_by_cached_path(xor_task: Task, tmp_path) -> None:
    """The task tensors travel through the worker's one-slot disk cache, never the job queue."""
    from ardevo.evolution import evolver as ev_mod
    from ardevo.evolution.evolver import AdapterRef

    class InlinePool:
        _processes = 2

        def map(self, worker, comps, *, chunksize):  # noqa: ANN001, ANN201 - multiprocessing Pool seam
            assert chunksize == 1
            assert isinstance(worker.keywords["spec"], AdapterRef)
            return [worker(comp) for comp in comps]

    config = _loop_config()
    config["library_dir"] = str(tmp_path)
    loop = build_loop(config)
    state = loop.fresh_state(random.Random(0))
    species_id = sorted(state.species_champions)[0]
    comp = _live_comp(species_id, 2, 1)
    previous = ev_mod.get_shared_pool()
    ev_mod.set_shared_pool(cast(Any, InlinePool()))
    try:
        results = loop._assess_all([comp, comp.clone()], _spec(xor_task), state, train=True)
        assert all(item.net is None and item.live_writebacks is not None for item in results)
        for item in results:
            _assert_tensor_free(item)
    finally:
        loop.evolver.release_task_adapter()
        ev_mod._WORKER_ADAPTER = None
        ev_mod.set_shared_pool(previous)


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


def test_owned_pool_acknowledges_release_before_spill_is_unlinked(xor_adapter, tmp_path) -> None:
    """Queued async work keeps access to its task payload until the worker release barrier."""
    from pathlib import Path
    from types import SimpleNamespace

    from ardevo.evolution import evolver as ev_mod
    from ardevo.evolution.evolver import AdapterRef

    evolver = build_evolver(
        {
            "library_dir": str(tmp_path),
            "evolution": {"pop_size": 2, "train": {"kind": "none"}},
            "fitness": {"components": ["support_accuracy"]},
        }
    )
    ref = evolver._pooled_adapter(xor_adapter)
    assert isinstance(ref, AdapterRef)

    class FakePool:
        _pool = [SimpleNamespace(pid=10, is_alive=lambda: True)]

        def map(self, function, values, *, chunksize):  # noqa: ANN001, ANN201 - multiprocessing Pool seam
            assert function is ev_mod._release_worker_task_adapter
            assert list(values) == [0, 1] and chunksize == 1
            assert Path(ref.path).exists()
            return [10, 10]

    evolver._pool = cast(Any, FakePool())
    evolver.release_task_adapter()
    assert not Path(ref.path).exists()
    evolver._pool = None


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
