import random
from array import array
from types import SimpleNamespace

import psutil
import torch

import ardevo.utils.resources as resources
from ardevo.evolution.composition import CompEdgeGene, CompositionGenome, _glue_for, comp_from_dict, comp_to_dict
from ardevo.utils.resources import ResourcePolicy, StageFootprint, format_bytes


class _Memory:
    total = 16 * 1024**3
    available = total


def test_fixed_limit_preserves_value_count_semantics(monkeypatch) -> None:
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Memory())
    estimate = ResourcePolicy(mode="adaptive").assess_glue(
        100,
        stage="composition_population",
        storage="tuple",
        device=torch.device("cpu"),
        fixed_limit=101,
        population_multiplicity=1_000,
        concurrent_trainers=1_000,
    )
    assert estimate.accepted
    assert estimate.mode == "fixed"
    assert estimate.limit_values == 101


def test_adaptive_limit_accounts_for_stage_multiplicity(monkeypatch) -> None:
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Memory())
    monkeypatch.setattr(resources, "_cgroup_memory", lambda: None)
    policy = ResourcePolicy(mode="adaptive", host_fraction=0.5, device_fraction=0.5, host_reserve_bytes=0, device_reserve_bytes=0)
    routed = policy.assess_glue(50_000_000, stage="routed_distillation", storage="f32", device="cpu")
    population = policy.assess_glue(
        50_000_000,
        stage="composition_population",
        storage="f32",
        device="cpu",
        population_multiplicity=48,
        concurrent_trainers=4,
    )
    assert routed.accepted
    assert not population.accepted
    assert population.host_required_bytes == routed.host_required_bytes * 48


def test_adaptive_host_budget_respects_available_and_cgroup_memory(monkeypatch) -> None:
    memory = _Memory()
    memory.available = 12 * 1024**3
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(resources, "_cgroup_memory", lambda: (8 * 1024**3, 6 * 1024**3))
    policy = ResourcePolicy(mode="adaptive", host_fraction=1.0, host_reserve_bytes=1024**3)

    estimate = policy.assess_glue(1, stage="fixture", storage="f32", device="cpu")

    assert estimate.host_budget_bytes == 5 * 1024**3


def test_compact_glue_round_trip_uses_binary_payload() -> None:
    values, rank = _glue_for(64, 32, random.Random(7), glue_rank=4, glue_rank_threshold=1, glue_storage="f32")
    assert isinstance(values, array)
    comp = CompositionGenome(edges=[CompEdgeGene(1, 2, True, 3, values, rank)])
    encoded = comp_to_dict(comp)
    assert "glue_f32_b64" in encoded["edges"][0]
    assert "glue" not in encoded["edges"][0]
    restored = comp_from_dict(encoded)
    assert isinstance(restored.edges[0].glue, array)
    assert restored.edges[0].glue == values


def test_legacy_glue_round_trip_stays_tuple() -> None:
    comp = CompositionGenome(edges=[CompEdgeGene(1, 2, True, 3, (0.25, -0.5))])
    encoded = comp_to_dict(comp)
    assert encoded["edges"][0]["glue"] == [0.25, -0.5]
    assert isinstance(comp_from_dict(encoded).edges[0].glue, tuple)


def test_psicov_decoded_cell_lower_bound_is_refused_without_allocation(monkeypatch) -> None:
    monkeypatch.setattr(resources, "_cgroup_memory", lambda: (64 * 1024**3, 60 * 1024**3))
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: SimpleNamespace(total=64 * 1024**3, available=60 * 1024**3))
    cells = 745_000_000_000
    footprint = StageFootprint(
        stage="psicov.b22.in0",
        representation="explicit_flat/minimal",
        candidate_bytes=cells * 5,
    )
    decision = ResourcePolicy(mode="adaptive", host_reserve_bytes=0).assess_stage(footprint)
    assert decision.accepted is False
    assert decision.footprint.candidate_bytes > 3 * 1024**4
    assert format_bytes(decision.footprint.candidate_bytes).endswith("TiB")


def test_byte_format_extends_through_exbibytes() -> None:
    assert format_bytes(2 * 1024**5) == "2.0 PiB"
    assert format_bytes(3 * 1024**6) == "3.0 EiB"
