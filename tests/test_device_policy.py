"""The central device policy: one mapping from run config to compute devices, consumed by the
population-batched trainer (via build_evolver injection) and Proctor. Per-genome pool work never
touches it (workers are CPU by construction)."""

import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from ardevo.evolution.registry import build_evolver
from ardevo.utils.device import (
    POPULATION_CPU_MODE,
    POPULATION_CUDA_MODE,
    SERIAL_MODE,
    ComputePolicy,
    auto_device,
    available_device,
    calibrate_compute_policy,
    capture_hardware_profile,
    load_compute_policy,
    resolve_compute_device,
    resolve_execution_mode,
    save_compute_policy,
)


def _force(monkeypatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_machine_env_mapping(monkeypatch) -> None:
    _force(monkeypatch, cuda=True, mps=True)
    assert resolve_compute_device({"machine_env": "MonadMetal"}).type == "mps"
    assert resolve_compute_device({"machine_env": "LatticeCUDA"}).type == "cuda"
    assert resolve_compute_device({"machine_env": "ClusterCUDA"}).type == "cuda"
    assert resolve_compute_device({"machine_env": "local"}).type == "cpu"
    assert resolve_compute_device({}).type == "cpu"


def test_local_lattice_machine_env_mapping(monkeypatch) -> None:
    _force(monkeypatch, cuda=True, mps=True)
    assert resolve_compute_device({"machine_env": "LocalLatticeCPU"}).type == "cpu"
    assert resolve_compute_device({"machine_env": "LocalLatticeCUDA"}).type == "cuda"


def test_unavailable_backends_fall_back_to_cpu(monkeypatch) -> None:
    _force(monkeypatch, cuda=False, mps=False)
    assert resolve_compute_device({"machine_env": "MonadMetal"}).type == "cpu"
    assert resolve_compute_device({"machine_env": "LatticeCUDA"}).type == "cpu"
    assert resolve_compute_device({"machine_env": "ClusterCUDA"}).type == "cpu"
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


def test_calibration_selects_valid_speedup_and_round_trips(monkeypatch, tmp_path) -> None:
    from ardevo.utils import device as device_module

    hardware = capture_hardware_profile()

    def serial_runner() -> dict[str, float]:
        return {"weight": 1.0}

    def population_runner() -> dict[str, float]:
        return {"weight": 1.0}

    timings = {serial_runner: 0.03, population_runner: 0.01}
    monkeypatch.setattr(device_module, "_timed_runner", lambda runner, **_kwargs: (timings[runner], runner()))
    destination = tmp_path / "compute_policy.json"
    policy = calibrate_compute_policy(
        {SERIAL_MODE: serial_runner, POPULATION_CPU_MODE: population_runner},
        default_mode=SERIAL_MODE,
        profile_path=destination,
        minimum_speedup=1.15,
        validate=lambda _name, reference, result: reference == result,
        hardware=hardware,
    )

    assert policy.selected_mode == POPULATION_CPU_MODE
    assert load_compute_policy(destination, hardware=hardware) == policy
    assert resolve_execution_mode(destination, default_mode=SERIAL_MODE, supported_modes=(SERIAL_MODE, POPULATION_CPU_MODE), hardware=hardware) == POPULATION_CPU_MODE


def test_policy_defaults_without_path_and_rejects_other_hardware(tmp_path) -> None:
    hardware = capture_hardware_profile()
    assert resolve_execution_mode(None, default_mode=SERIAL_MODE, supported_modes=(SERIAL_MODE, POPULATION_CPU_MODE), hardware=hardware) == SERIAL_MODE
    assert list(tmp_path.iterdir()) == []

    policy = ComputePolicy(
        hardware=hardware,
        default_mode=SERIAL_MODE,
        selected_mode=POPULATION_CPU_MODE,
        timings_seconds={SERIAL_MODE: 2.0, POPULATION_CPU_MODE: 1.0},
        valid_modes=(SERIAL_MODE, POPULATION_CPU_MODE),
        errors={},
        minimum_speedup=1.15,
    )
    destination = tmp_path / "compute_policy.json"
    save_compute_policy(policy, destination)
    other_hardware = replace(hardware, memory_bytes=hardware.memory_bytes + 1)
    assert load_compute_policy(destination, hardware=other_hardware) is None


def test_policy_refuses_tracked_repository_path() -> None:
    hardware = capture_hardware_profile()
    policy = ComputePolicy(
        hardware=hardware,
        default_mode=SERIAL_MODE,
        selected_mode=SERIAL_MODE,
        timings_seconds={SERIAL_MODE: 1.0},
        valid_modes=(SERIAL_MODE,),
        errors={},
        minimum_speedup=1.15,
    )
    destination = Path(__file__).resolve().parents[1] / "compute_policy_not_ignored.json"
    with pytest.raises(ValueError, match="gitignored"):
        save_compute_policy(policy, destination)
    assert not destination.exists()


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


def test_build_evolver_can_opt_in_scheduled_batching_from_profile(tmp_path) -> None:
    hardware = capture_hardware_profile()
    policy = ComputePolicy(
        hardware=hardware,
        default_mode=SERIAL_MODE,
        selected_mode=POPULATION_CPU_MODE,
        timings_seconds={SERIAL_MODE: 2.0, POPULATION_CPU_MODE: 1.0},
        valid_modes=(SERIAL_MODE, POPULATION_CPU_MODE),
        errors={},
        minimum_speedup=1.15,
    )
    destination = tmp_path / "compute_policy.json"
    save_compute_policy(policy, destination)
    config = _population_config("local", compute_policy_path=str(destination))
    config["evolution"]["train"]["kind"] = "gradient_scheduled"

    evolver = build_evolver(config)
    assert evolver.train_population_op is not None
    assert _bound_device(evolver) == "cpu"


def test_explicit_scheduled_batching_overrides_missing_profile(monkeypatch) -> None:
    _force(monkeypatch, cuda=True, mps=False)
    config = _population_config("ClusterCUDA", batched=True)
    config["evolution"]["train"]["kind"] = "gradient_scheduled"
    evolver = build_evolver(config)
    assert evolver.execution_mode == POPULATION_CUDA_MODE
    assert _bound_device(evolver) == "cuda"


def test_resolve_worker_count(monkeypatch) -> None:
    import os

    from ardevo.utils.device import resolve_worker_count

    assert resolve_worker_count(12) == 12
    assert resolve_worker_count(0) == 0
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        monkeypatch.delenv(name, raising=False)
    if hasattr(os, "sched_getaffinity"):
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(64)))
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert resolve_worker_count("auto") == 12  # cpu_count - 4
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "10")
    assert resolve_worker_count("auto") == 6
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
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


def test_cluster_cuda_uses_local_launcher_with_clearml_telemetry(monkeypatch) -> None:
    from ardevo.utils import pipelines

    monkeypatch.setattr(pipelines, "HAS_CLEARML", True)
    monkeypatch.setattr(pipelines.Pipeline, "_create_task", lambda self: None)
    pipe = pipelines.Pipeline({"machine_env": "ClusterCUDA", "clearml_run": True})
    assert pipe.queue == "local"
    assert pipe.clearml_run is True


@pytest.mark.parametrize("machine_env", ["LocalLatticeCPU", "LocalLatticeCUDA"])
def test_local_lattice_runs_inline_once_with_clearml_telemetry(monkeypatch, machine_env: str) -> None:
    from ardevo.utils import pipelines

    calls = {"task_init": 0, "trial_run": 0, "execute_remotely": 0, "close": 0}

    class FakeRun:
        def connect(self, _values) -> None:
            pass

        def set_repo(self, **_values) -> None:
            pass

        def execute_remotely(self, **_values) -> None:
            calls["execute_remotely"] += 1

        def close(self) -> None:
            calls["close"] += 1

    class FakeTask:
        TaskTypes = SimpleNamespace(custom="custom")

        @staticmethod
        def init(**_values):
            calls["task_init"] += 1
            return FakeRun()

    class FakeTrial:
        def __init__(self, *, config, task) -> None:
            assert config["machine_env"] == machine_env
            assert isinstance(task, FakeRun)

        def run(self) -> dict[str, str]:
            calls["trial_run"] += 1
            return {"machine_env": machine_env}

    monkeypatch.setattr(pipelines, "HAS_CLEARML", True)
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeTask))
    monkeypatch.setattr(pipelines, "get_current_branch", lambda: "")

    pipeline = pipelines.Pipeline({"machine_env": machine_env, "clearml_run": True})
    pipeline.add_trial(FakeTrial)

    assert pipeline.queue == "local"
    assert pipeline.run_task() == [{"machine_env": machine_env}]
    assert calls == {"task_init": 1, "trial_run": 1, "execute_remotely": 0, "close": 1}


def test_clearml_disables_pytorch_model_interception_but_keeps_explicit_telemetry(monkeypatch) -> None:
    from ardevo.utils import pipelines

    captured: dict[str, Any] = {}

    class FakeRun:
        def connect(self, values) -> None:
            captured["connected"] = values

        def set_repo(self, **values) -> None:
            captured["repo"] = values

    class FakeTask:
        TaskTypes = SimpleNamespace(custom="custom")

        @staticmethod
        def init(**values):
            captured["init"] = values
            return FakeRun()

    monkeypatch.setattr(pipelines, "HAS_CLEARML", True)
    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeTask))
    monkeypatch.setattr(pipelines, "get_current_branch", lambda: None)
    pipeline = pipelines.Pipeline(
        {
            "machine_env": "MonadMetal",
            "clearml_run": True,
            "project_name": "ardevo",
            "experiment_name": "test",
            "hyperparameters": {"seed": 1},
        }
    )

    assert pipeline.task is not None
    assert captured["init"]["auto_connect_frameworks"] == {"pytorch": False}
    assert captured["connected"] == {"seed": 1}
