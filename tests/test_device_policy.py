"""The central device policy: one mapping from run config to compute devices, consumed by the
population-batched trainer (via build_evolver injection) and Proctor. Per-genome pool work never
touches it (workers are CPU by construction)."""

from functools import partial
from typing import Any, cast

import torch

from ardevo.evolution.registry import build_evolver
from ardevo.utils.device import auto_device, available_device, resolve_compute_device


def _force(monkeypatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_machine_env_mapping(monkeypatch) -> None:
    _force(monkeypatch, cuda=True, mps=True)
    assert resolve_compute_device({"machine_env": "MonadMetal"}).type == "mps"
    assert resolve_compute_device({"machine_env": "LatticeCUDA"}).type == "cuda"
    assert resolve_compute_device({"machine_env": "local"}).type == "cpu"
    assert resolve_compute_device({}).type == "cpu"


def test_unavailable_backends_fall_back_to_cpu(monkeypatch) -> None:
    _force(monkeypatch, cuda=False, mps=False)
    assert resolve_compute_device({"machine_env": "MonadMetal"}).type == "cpu"
    assert resolve_compute_device({"machine_env": "LatticeCUDA"}).type == "cpu"
    assert available_device("cuda").type == "cpu"
    assert available_device("mps").type == "cpu"


def test_explicit_knob_beats_compute_beats_machine(monkeypatch) -> None:
    _force(monkeypatch, cuda=True, mps=True)
    config = {"machine_env": "MonadMetal", "compute": "cuda"}
    assert resolve_compute_device(config).type == "cuda"  # [run] compute overrides the machine map
    assert resolve_compute_device(config, explicit="cpu").type == "cpu"  # per-op knob wins over both
    assert resolve_compute_device(config, explicit="auto").type == "cuda"  # "auto" defers downward


def test_auto_device_prefers_cuda_over_mps(monkeypatch) -> None:
    _force(monkeypatch, cuda=True, mps=True)
    assert auto_device().type == "cuda"
    _force(monkeypatch, cuda=False, mps=True)
    assert auto_device().type == "mps"
    _force(monkeypatch, cuda=False, mps=False)
    assert auto_device().type == "cpu"


def _population_config(machine_env: str, **train_extras) -> dict:
    return {
        "machine_env": machine_env,
        "evolution": {
            "pop_size": 4,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "none"},
            "mutation": {"operators": []},
            "train": {"kind": "gradient_batched", "steps": 2, **train_extras},
        },
        "fitness": {"components": ["query_accuracy"]},
    }


def _bound_device(evolver) -> str:
    assert evolver.train_population_op is not None
    return cast(partial, evolver.train_population_op).keywords["device"]


def test_build_evolver_injects_resolved_device(monkeypatch) -> None:
    _force(monkeypatch, cuda=False, mps=False)
    assert _bound_device(build_evolver(_population_config("LatticeCUDA"))) == "cpu"  # unavailable cuda degrades

    _force(monkeypatch, cuda=True, mps=False)
    assert _bound_device(build_evolver(_population_config("LatticeCUDA"))) == "cuda"


def test_build_evolver_honors_pinned_device(monkeypatch) -> None:
    _force(monkeypatch, cuda=True, mps=True)
    assert _bound_device(build_evolver(_population_config("LatticeCUDA", device="cpu"))) == "cpu"


def test_resolve_worker_count(monkeypatch) -> None:
    import os

    from ardevo.utils.device import resolve_worker_count

    assert resolve_worker_count(12) == 12
    assert resolve_worker_count(0) == 0
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert resolve_worker_count("auto") == 12  # cpu_count - 4
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert resolve_worker_count("auto") == 1  # floors at one worker


def test_proctor_delegates_to_resolver(monkeypatch) -> None:
    from ardevo.utils.proctor import Proctor

    class DummyTrial(Proctor):
        def run(self) -> dict[str, Any]:  # pragma: no cover - never called
            return {}

    _force(monkeypatch, cuda=False, mps=False)
    assert DummyTrial({"machine_env": "MonadMetal"}).device.type == "cpu"
    _force(monkeypatch, cuda=False, mps=True)
    assert DummyTrial({"machine_env": "MonadMetal"}).device.type == "mps"
