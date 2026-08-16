"""Central compute-device and calibrated execution policy.

Per-genome work (the spawn pool's decode+train+evaluate) always stays on CPU: those kernels are
tiny and dispatch-bound, and the workers are pinned to one intra-op thread each. The GPU pays off
only for population-batched tensor programs (the `gradient_batched` family), so this resolver is
consumed at those sites, threaded in by `build_evolver`, plus `Proctor`'s informational device.
Resolution order: an explicit per-op knob (`device = ...` on the train table) wins, then the
`[run] compute` override, then the `[run] machine` mapping (MonadMetal -> mps, either Lattice
CUDA mode -> cuda, ClusterCUDA -> cuda). Every rung of the ladder falls back to CPU with a warning
rather than crashing a run whose accelerator is missing or unconfigured.

Calibration is deliberately separate from device resolution. A caller supplies named execution
mode runners and, optionally, an explicit path under gitignored run state. Merely importing this
module or resolving a device never benchmarks hardware and never writes a profile.
"""

import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import torch

from versal.utils.logging import Logger

logger = Logger.get_logger()

_MACHINE_DEVICE = {
    "MonadMetal": "mps",
    "LatticeCUDA": "cuda",
    "LocalLatticeCUDA": "cuda",
    "LatticeCPU": "cpu",
    "LocalLatticeCPU": "cpu",
    "ClusterCUDA": "cuda",
}

SERIAL_MODE = "serial"
POPULATION_CPU_MODE = "population_cpu"
POPULATION_MPS_MODE = "population_mps"
POPULATION_CUDA_MODE = "population_cuda"
POPULATION_MODES = (POPULATION_CPU_MODE, POPULATION_MPS_MODE, POPULATION_CUDA_MODE)


@dataclass(frozen=True)
class HardwareProfile:
    """Stable hardware/runtime identity used to reject calibration from another machine."""

    system: str
    release: str
    machine: str
    processor: str
    cpu_logical: int
    cpu_physical: int
    memory_bytes: int
    torch_version: str
    torch_threads: int
    torch_interop_threads: int
    mps_built: bool
    mps_available: bool
    cuda_runtime: str | None
    cuda_devices: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "release": self.release,
            "machine": self.machine,
            "processor": self.processor,
            "cpu_logical": self.cpu_logical,
            "cpu_physical": self.cpu_physical,
            "memory_bytes": self.memory_bytes,
            "torch_version": self.torch_version,
            "torch_threads": self.torch_threads,
            "torch_interop_threads": self.torch_interop_threads,
            "mps_built": self.mps_built,
            "mps_available": self.mps_available,
            "cuda_runtime": self.cuda_runtime,
            "cuda_devices": list(self.cuda_devices),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HardwareProfile":
        return cls(
            system=str(value["system"]),
            release=str(value["release"]),
            machine=str(value["machine"]),
            processor=str(value["processor"]),
            cpu_logical=int(value["cpu_logical"]),
            cpu_physical=int(value["cpu_physical"]),
            memory_bytes=int(value["memory_bytes"]),
            torch_version=str(value["torch_version"]),
            torch_threads=int(value["torch_threads"]),
            torch_interop_threads=int(value["torch_interop_threads"]),
            mps_built=bool(value["mps_built"]),
            mps_available=bool(value["mps_available"]),
            cuda_runtime=None if value.get("cuda_runtime") is None else str(value["cuda_runtime"]),
            cuda_devices=tuple(dict(item) for item in value.get("cuda_devices", [])),
        )


@dataclass(frozen=True)
class ComputePolicy:
    """Result of benchmarking named modes on one exact hardware/runtime profile."""

    hardware: HardwareProfile
    default_mode: str
    selected_mode: str
    timings_seconds: dict[str, float]
    valid_modes: tuple[str, ...]
    errors: dict[str, str]
    minimum_speedup: float
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hardware": self.hardware.to_dict(),
            "hardware_fingerprint": self.hardware.fingerprint,
            "default_mode": self.default_mode,
            "selected_mode": self.selected_mode,
            "timings_seconds": self.timings_seconds,
            "valid_modes": list(self.valid_modes),
            "errors": self.errors,
            "minimum_speedup": self.minimum_speedup,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComputePolicy":
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError(f"unsupported compute policy schema {value.get('schema_version')!r}")
        hardware = HardwareProfile.from_dict(value["hardware"])
        if value.get("hardware_fingerprint") != hardware.fingerprint:
            raise ValueError("compute policy hardware fingerprint is corrupt")
        valid_modes = tuple(str(name) for name in value.get("valid_modes", []))
        selected_mode = str(value["selected_mode"])
        default_mode = str(value["default_mode"])
        timings = {str(name): float(seconds) for name, seconds in dict(value.get("timings_seconds", {})).items()}
        if default_mode not in timings or selected_mode not in valid_modes or selected_mode not in timings:
            raise ValueError("compute policy mode selection is inconsistent with its measurements")
        return cls(
            hardware=hardware,
            default_mode=default_mode,
            selected_mode=selected_mode,
            timings_seconds=timings,
            valid_modes=valid_modes,
            errors={str(name): str(error) for name, error in dict(value.get("errors", {})).items()},
            minimum_speedup=float(value.get("minimum_speedup", 1.0)),
        )


def resolve_compute_device(config: dict[str, Any], *, explicit: str | None = None) -> torch.device:
    """The device population-batched compute should land on for this run configuration."""
    requested = explicit if explicit not in (None, "auto") else None
    if requested is None:
        compute = str(config.get("compute", "auto"))
        requested = compute if compute != "auto" else None
    if requested is None:
        requested = _MACHINE_DEVICE.get(str(config.get("machine_env", "local")), "cpu")
    return available_device(requested)


def auto_device() -> torch.device:
    """Best available device with no configuration signal: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _processor_identity() -> str:
    names = [platform.processor()]
    if platform.system() == "Darwin":
        import subprocess

        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                completed = subprocess.run(["sysctl", "-n", key], check=False, capture_output=True, text=True, timeout=1.0)
                names.append(completed.stdout.strip())
            except (OSError, subprocess.TimeoutExpired):
                pass
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith(("model name", "hardware")) and ":" in line:
                    names.append(line.split(":", 1)[1].strip())
                    break
        except OSError:
            pass
    return " | ".join(dict.fromkeys(name for name in names if name)) or platform.machine()


def capture_hardware_profile() -> HardwareProfile:
    """Fingerprint the CPU, unified-memory MPS backend, CUDA devices, and PyTorch runtime."""
    cuda_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        try:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                cuda_devices.append(
                    {
                        "index": index,
                        "name": properties.name,
                        "total_memory": int(properties.total_memory),
                        "capability": [int(properties.major), int(properties.minor)],
                        "multiprocessors": int(properties.multi_processor_count),
                    }
                )
        except (RuntimeError, AssertionError) as exc:
            # A driver can disappear between is_available() and enumeration. Record that fact in
            # the identity rather than making CPU-only startup depend on a healthy CUDA driver.
            cuda_devices.append({"enumeration_error": type(exc).__name__})
    mps_backend = torch.backends.mps
    mps_built = bool(mps_backend.is_built()) if hasattr(mps_backend, "is_built") else False
    return HardwareProfile(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        processor=_processor_identity(),
        cpu_logical=int(psutil.cpu_count(logical=True) or os.cpu_count() or 1),
        cpu_physical=int(psutil.cpu_count(logical=False) or 0),
        memory_bytes=int(psutil.virtual_memory().total),
        torch_version=str(torch.__version__),
        torch_threads=int(torch.get_num_threads()),
        torch_interop_threads=int(torch.get_num_interop_threads()),
        mps_built=mps_built,
        mps_available=bool(mps_backend.is_available()),
        cuda_runtime=None if torch.version.cuda is None else str(torch.version.cuda),
        cuda_devices=tuple(cuda_devices),
    )


def population_mode(device: str | torch.device) -> str:
    """Canonical execution-mode name for a population program on `device`."""
    device_type = torch.device(device).type
    if device_type == "cpu":
        return POPULATION_CPU_MODE
    if device_type == "mps":
        return POPULATION_MPS_MODE
    if device_type == "cuda":
        return POPULATION_CUDA_MODE
    raise ValueError(f"no population execution mode for device {device!s}")


def device_for_population_mode(mode: str) -> torch.device:
    """Inverse of :func:`population_mode`, availability-checked with the normal fallback policy."""
    devices = {POPULATION_CPU_MODE: "cpu", POPULATION_MPS_MODE: "mps", POPULATION_CUDA_MODE: "cuda"}
    if mode not in devices:
        raise ValueError(f"not a population execution mode: {mode!r}")
    return available_device(devices[mode])


def _synchronize_accelerators() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except (RuntimeError, AssertionError):
            pass
    if torch.backends.mps.is_available() and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except RuntimeError:
            pass


def _timed_runner(runner: Callable[[], object], *, repeats: int, warmups: int) -> tuple[float, object]:
    result: object = None
    for _ in range(warmups):
        result = runner()
        _synchronize_accelerators()
    samples: list[float] = []
    for _ in range(repeats):
        _synchronize_accelerators()
        started = time.perf_counter()
        result = runner()
        _synchronize_accelerators()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), result


def calibrate_compute_policy(
    runners: Mapping[str, Callable[[], object]],
    *,
    default_mode: str,
    profile_path: str | Path | None = None,
    repeats: int = 3,
    warmups: int = 1,
    minimum_speedup: float = 1.15,
    validate: Callable[[str, object, object], bool] | None = None,
    hardware: HardwareProfile | None = None,
) -> ComputePolicy:
    """Benchmark and select among named execution modes.

    `default_mode` remains selected unless another numerically valid mode is at least
    `minimum_speedup` faster. No file is written unless `profile_path` is explicitly provided;
    callers should place it under `results/` or another gitignored state directory. A runner must
    construct fresh mutable state on every call so warmups cannot influence measured trials.
    """
    if default_mode not in runners:
        raise ValueError(f"default mode {default_mode!r} has no runner")
    if repeats < 1 or warmups < 0:
        raise ValueError("calibration needs repeats >= 1 and warmups >= 0")
    if minimum_speedup < 1.0:
        raise ValueError("minimum_speedup must be >= 1.0")

    reference = runners[default_mode]()
    timings: dict[str, float] = {}
    valid_modes: list[str] = []
    errors: dict[str, str] = {}
    for name, runner in runners.items():
        try:
            seconds, result = _timed_runner(runner, repeats=repeats, warmups=warmups)
            valid = name == default_mode or validate is None or validate(name, reference, result)
            if not valid:
                errors[name] = "numeric validation failed"
                continue
            timings[name] = seconds
            valid_modes.append(name)
        except (RuntimeError, ValueError) as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    if default_mode not in timings:
        raise RuntimeError(f"default compute mode {default_mode!r} could not be benchmarked: {errors.get(default_mode, 'unknown error')}")
    fastest = min(valid_modes, key=timings.__getitem__)
    speedup = timings[default_mode] / timings[fastest]
    selected = fastest if speedup >= minimum_speedup else default_mode
    policy = ComputePolicy(
        hardware=hardware or capture_hardware_profile(),
        default_mode=default_mode,
        selected_mode=selected,
        timings_seconds=timings,
        valid_modes=tuple(valid_modes),
        errors=errors,
        minimum_speedup=minimum_speedup,
    )
    if profile_path is not None:
        save_compute_policy(policy, profile_path)
    return policy


def save_compute_policy(policy: ComputePolicy, path: str | Path) -> None:
    """Atomically persist a policy only at the caller-provided run-state path."""
    destination = Path(path)
    project_root = Path(__file__).resolve().parents[2]
    try:
        relative = destination.resolve().relative_to(project_root)
    except ValueError:
        relative = None
    if relative is not None and (project_root / ".git").exists():
        import subprocess

        ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(relative)], cwd=project_root, check=False).returncode == 0
        if not ignored:
            raise ValueError(f"compute policies inside the repository must use a gitignored path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_compute_policy(path: str | Path, *, hardware: HardwareProfile | None = None) -> ComputePolicy | None:
    """Load a policy only when its schema and hardware fingerprint match this process."""
    source = Path(path)
    if not source.is_file():
        return None
    try:
        policy = ComputePolicy.from_dict(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid compute policy %s: %s", source, exc)
        return None
    current = hardware or capture_hardware_profile()
    if policy.hardware.fingerprint != current.fingerprint:
        logger.warning("Ignoring compute policy %s from different hardware/runtime", source)
        return None
    return policy


def resolve_execution_mode(
    profile_path: str | Path | None,
    *,
    default_mode: str,
    supported_modes: Sequence[str],
    hardware: HardwareProfile | None = None,
) -> str:
    """Resolve a saved named mode, or preserve `default_mode` when no valid profile exists."""
    if profile_path is None:
        return default_mode
    policy = load_compute_policy(profile_path, hardware=hardware)
    supported = set(supported_modes)
    if policy is None or policy.default_mode != default_mode or policy.selected_mode not in supported:
        return default_mode
    return policy.selected_mode


def resolve_worker_count(value: int | str) -> int:
    """Resolve ``assess_workers``, keeping CPU and memory headroom for the parent process.

    Auto sizing respects allocation/affinity, leaves four logical CPUs for the parent and OS, and
    caps at physical cores minus two. CPU-bound Torch workers do not benefit enough from sibling
    hyperthreads to justify duplicating a large task payload into all of them. Explicit integers
    remain exact overrides.
    """
    if value == "auto":
        import os

        counts = [os.cpu_count() or 8]
        if hasattr(os, "sched_getaffinity"):
            try:
                counts.append(len(os.sched_getaffinity(0)))
            except OSError:
                pass
        for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
            try:
                allocated = int(os.environ.get(name, "0"))
            except ValueError:
                allocated = 0
            if allocated > 0:
                counts.append(allocated)
        limits = [min(counts) - 4]
        physical = psutil.cpu_count(logical=False)
        if physical is not None and physical > 0:
            limits.append(physical - 2)
        return max(1, min(limits))
    return int(value)


def available_device(requested: str) -> torch.device:
    """The requested device if its backend is actually present, else CPU with a warning."""
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        logger.warning("CUDA requested but unavailable; compute falls back to CPU")
        return torch.device("cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but unavailable; compute falls back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def is_out_of_memory_error(exc: BaseException) -> bool:
    """Recognize PyTorch's common CUDA/MPS/CPU allocator failure forms."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message or "can't allocate memory" in message


def clear_device_cache(device: str | torch.device) -> None:
    """Release allocator caches after an OOM without changing tensors or numeric settings."""
    device_type = torch.device(device).type
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "mps" and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
