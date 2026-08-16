"""Proctor: base trial class. Ported and slimmed from the sibling NEXUS infra.

Provides ClearML logging, hardware monitoring, device selection, and artifact saving.
Trials subclass it and implement `run()`. The DataLoader coupling is removed: Versal
trials load their own Icarus data.
"""

from __future__ import annotations

import importlib.util
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil
import torch

from versal.utils.logging import Logger

if TYPE_CHECKING:
    from clearml import Task

# ClearML is optional; the trial receives a task object (or None) from the Pipeline, so Proctor
# never imports it at runtime. Detect availability only to gate the clearml_run flag.
HAS_CLEARML = importlib.util.find_spec("clearml") is not None

logger = Logger.get_logger()


class Proctor(ABC):
    """Base trial: owns logging, monitoring, device, and artifacts. Subclasses implement `run`."""

    def __init__(self, config: dict[str, Any], task: "Task | None" = None) -> None:
        self.config = config
        self.clearml_run = bool(config.get("clearml_run", False)) and HAS_CLEARML
        self.task = task if self.clearml_run else None
        self.clearml_logger = self.task.get_logger() if self.task else None
        self.hp = config.get("hyperparameters", {})
        self.device = self._get_device()
        self.results: dict[str, Any] = {}
        self.start_time = time.time()

    def _get_device(self) -> torch.device:
        """Map the run configuration to a torch device (the shared resolver owns the mapping)."""
        from versal.utils.device import resolve_compute_device

        device = resolve_compute_device(self.config)
        if device.type == "cuda":
            logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            logger.info("Using %s device", device.type.upper() if device.type == "mps" else device.type)
        return device

    def log_scalar(self, category: str, series: str, value: float, iteration: int) -> None:
        """Report a scalar to ClearML (no-op when offline)."""
        if self.clearml_logger:
            self.clearml_logger.report_scalar(category, series, value, iteration)

    def log_hardware_stats(self, iteration: int) -> None:
        """Report CPU and memory usage to ClearML (no-op when offline)."""
        if not self.clearml_logger:
            return
        self.clearml_logger.report_scalar("Resources", "CPU %", psutil.cpu_percent(), iteration)
        self.clearml_logger.report_scalar("Resources", "Memory %", psutil.virtual_memory().percent, iteration)

    def save_artifact(self, name: str, obj: Any) -> None:
        """Upload an arbitrary (JSON-able) artifact to ClearML; otherwise log locally."""
        if not self.task:
            logger.info("Artifact %r kept in results (offline run).", name)
            return
        self.task.upload_artifact(name=name, artifact_object=obj)
        logger.info("Uploaded artifact: %s", name)

    def save_model(self, model: torch.nn.Module, name: str = "model") -> Path:
        """Save a torch module's state dict locally and register it as a ClearML artifact."""
        path = Path("models/checkpoints") / f"{name}.pth"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        if self.task:
            self.task.upload_artifact(name=f"{name}_checkpoint", artifact_object=str(path))
        logger.info("Saved model checkpoint: %s", path)
        return path

    def finalize(self) -> None:
        """Log total wall-clock time."""
        total = time.time() - self.start_time
        logger.info("Trial complete. Total time: %.2fs", total)

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the trial. Returns a results dict."""
