"""Behavioral novelty: descriptor determinism, k-NN scoring, the archive, and the evolver hook.

The critical properties: a duplicate behavior scores ~0 while a distant one scores high (the
anti-deception lever), the whole feature is rng-free, and the off switch is byte-identical.
"""

import random
from typing import Any

import pytest
import torch

from versal.evolution.evolver import Evolver, TaskAdapter
from versal.evolution.genome import ConnectionGene, Genome
from versal.evolution.novelty import archive_insert, compute_descriptor, novelty_scores, probe_indices, probe_tensor
from versal.evolution.registry import build_evolver


def _config(novelty_table: dict[str, Any] | None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "seed": 0,
        "substrate": {"available_activations": ["tanh"], "default_activation": "tanh"},
        "evolution": {
            "pop_size": 8,
            "elitism": 1,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 3},
            "crossover": {"kind": "none", "rate": 0.0},
            "mutation": {"operators": ["perturb_weights"], "perturb_weights_prob": 0.5},
            "train": {"kind": "none"},
            "speciation": {"kind": "none"},
        },
        "fitness": {"components": ["support_accuracy", "novelty"], "w_novelty": 0.5},
    }
    if novelty_table is not None:
        config["evolution"]["novelty"] = novelty_table
    return config


def test_novelty_rises_for_distant_falls_for_duplicate() -> None:
    archived = (0.5, 0.5, 0.5, 0.5)
    duplicate = archived
    distant = (-0.9, 0.9, -0.9, 0.9)
    scores = novelty_scores([duplicate, distant], archive=[archived] * 5, k=3)
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] > 0.3
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_knn_excludes_self_and_clamps_k() -> None:
    twin = (0.1, 0.2)
    assert novelty_scores([twin, twin], archive=[], k=15) == [pytest.approx(0.0)] * 2  # duplicates are not novel
    assert novelty_scores([twin], archive=[], k=15) == [0.0]  # a lone member has no neighbors, not a crash


def test_probe_indices_deterministic_and_capped() -> None:
    indices = probe_indices(320, 64)
    assert len(indices) == 64
    assert indices == sorted(indices) and len(set(indices)) == 64
    assert probe_indices(320, 64) == indices
    assert probe_indices(4, 64) == [0, 1, 2, 3]  # small tasks use every row


def test_descriptor_deterministic(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    probe = probe_tensor(xor_adapter.encoded, probe_rows=64)
    assert probe is not None and probe.shape[0] == 4  # XOR support has 4 rows
    first = compute_descriptor(xor_adapter.decode(solving_genome), probe)
    second = compute_descriptor(xor_adapter.decode(solving_genome), probe)
    assert first is not None and first == second
    assert len(first) == 4 * xor_adapter.n_outputs
    assert all(-1.0 <= value <= 1.0 for value in first)  # tanh-bounded


def test_archive_insert_fifo() -> None:
    archive: list[tuple[float, ...]] = []
    for index in range(10):
        archive_insert(archive, (float(index),), cap=3)
    assert archive == [(7.0,), (8.0,), (9.0,)]
    untouched: list[tuple[float, ...]] = []
    archive_insert(untouched, (1.0,), cap=0)
    assert untouched == []  # cap 0 = the archive-free (population-only) ablation


def test_hook_injects_scores_and_rescores_fitness(xor_adapter: TaskAdapter) -> None:
    evolver: Evolver = build_evolver(_config({"k": 3, "archive_cap": 8, "probe_rows": 16}))
    assert evolver.novelty is not None
    state = evolver.seed_state(xor_adapter, random.Random(0))
    viable = [item for item in state.population if item.module is not None]
    assert viable
    for item in viable:
        assert 0.0 <= item.metrics["novelty"] <= 1.0
        assert item.fitness == pytest.approx(evolver.fitness(item.genome, item.metrics))  # w_novelty applied
    assert 1 <= len(state.novelty_archive) <= 8
    evolver.advance(state, xor_adapter)
    assert all("novelty" in item.metrics for item in state.population if item.module is not None)


def test_hook_is_rng_free(xor_adapter: TaskAdapter) -> None:
    evolver: Evolver = build_evolver(_config({"k": 3}))
    state = evolver.seed_state(xor_adapter, random.Random(0))
    before = state.rng.getstate()
    evolver._apply_novelty(state.population, state, xor_adapter)
    assert state.rng.getstate() == before


def test_floored_candidate_keeps_floor_and_no_key(xor_adapter: TaskAdapter, linear_genome: Genome) -> None:
    evolver: Evolver = build_evolver(_config({"k": 3}))
    corpse_genome = linear_genome.clone()
    corpse_genome.connections = [*corpse_genome.connections, ConnectionGene(99, 3, 1.0, True, 77)]  # decode KeyErrors
    state = evolver.seed_state(xor_adapter, random.Random(0), seeded_front=lambda _innovations: [corpse_genome])
    corpse = next(item for item in state.population if item.module is None)
    assert corpse.fitness == -1e9
    assert "novelty" not in corpse.metrics  # floored candidates pass through unscored, floor intact


def test_off_switch_leaves_everything_untouched(xor_adapter: TaskAdapter) -> None:
    evolver: Evolver = build_evolver(_config(None))
    assert evolver.novelty is None
    state = evolver.seed_state(xor_adapter, random.Random(0))
    assert all("novelty" not in item.metrics for item in state.population)
    assert state.novelty_archive == []
    disabled = build_evolver(_config({"enabled": False, "k": 3}))
    assert disabled.novelty is None  # enabled = false parses to the same byte-identical off state


def test_bare_table_enables_with_defaults() -> None:
    # A bare `[evolution.novelty]` header parses to {}: present means ON with the dataclass
    # defaults (the documented off states are ONLY an absent table or enabled = false).
    evolver: Evolver = build_evolver(_config({}))
    assert evolver.novelty is not None
    assert (evolver.novelty.k, evolver.novelty.archive_cap, evolver.novelty.probe_rows) == (15, 256, 64)


def test_non_finite_descriptor_is_rejected() -> None:
    class _NaNModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 1), float("nan"))

    # A NaN forward must go unscored, never feed NaN rows into cdist or the archive.
    assert compute_descriptor(_NaNModule(), torch.zeros((4, 2))) is None


def test_probe_tensor_skips_non_flat_tasks() -> None:
    from typing import cast

    from versal.dataset.icarus import EncodedTask

    class _FakeEncoded:
        support_input = (torch.zeros((3, 4, 2)), None)  # a TIME-axis shape

    assert probe_tensor(cast(EncodedTask, _FakeEncoded()), probe_rows=8) is None
