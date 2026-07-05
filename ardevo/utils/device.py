"""Central compute-device policy: ONE mapping from run configuration to torch devices.

Per-genome work (the spawn pool's decode+train+evaluate) always stays on CPU: those kernels are
tiny and dispatch-bound, and the workers are pinned to one intra-op thread each. The GPU pays off
only for population-batched tensor programs (the `gradient_batched` family), so this resolver is
consumed at those sites, threaded in by `build_evolver`, plus `Proctor`'s informational device.
Resolution order: an explicit per-op knob (`device = ...` on the train table) wins, then the
`[run] compute` override, then the `[run] machine` mapping (MonadMetal -> mps, LatticeCUDA ->
cuda). Every rung of the ladder falls back to CPU with a warning rather than crashing a queue job
on an agent whose GPU is missing or unconfigured.
"""

from typing import Any

import torch

from ardevo.utils.logging import Logger

logger = Logger.get_logger()

_MACHINE_DEVICE = {"MonadMetal": "mps", "LatticeCUDA": "cuda"}


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


def resolve_worker_count(value: int | str) -> int:
    """The `assess_workers` knob: `"auto"` sizes to the machine (cpu_count - 4 leaves headroom for
    the main process, the OS, and a GPU feeder); anything else is an explicit integer count."""
    if value == "auto":
        import os

        return max(1, (os.cpu_count() or 8) - 4)
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
