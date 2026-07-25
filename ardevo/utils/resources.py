"""Deterministic pre-allocation estimates for wide evolutionary candidates.

The policy never changes a search method.  It only answers whether the current host/device can
materialize a candidate at a particular stage, before Python objects or optimizer tensors are
allocated.  Fixed limits preserve the historical value-count guard; adaptive limits account for
the representation, population residency, and concurrent training working set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import psutil
import torch


def _read_memory_counter(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if 0 < parsed < 1 << 60 else None


def _cgroup_memory() -> tuple[int, int] | None:
    """Return (limit, remaining) for cgroup v2/v1 when this process is constrained."""

    pairs: list[tuple[Path, Path]] = []
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        memberships = []
    for membership in memberships:
        parts = membership.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, relative = parts
        relative = relative.lstrip("/")
        if hierarchy == "0":
            root = Path("/sys/fs/cgroup") / relative
            pairs.append((root / "memory.max", root / "memory.current"))
        elif "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory") / relative
            pairs.append((root / "memory.limit_in_bytes", root / "memory.usage_in_bytes"))
    pairs.extend(
        (
            (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
            (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
        )
    )
    for limit_path, usage_path in pairs:
        limit = _read_memory_counter(limit_path)
        usage = _read_memory_counter(usage_path)
        if limit is not None:
            return limit, max(0, limit - (usage or 0))
    return None


@dataclass(frozen=True)
class ResourceEstimate:
    stage: str
    glue_values: int
    storage: str
    population_multiplicity: int
    concurrent_trainers: int
    host_required_bytes: int
    device_required_bytes: int
    host_budget_bytes: int
    device_budget_bytes: int
    limit_values: int
    accepted: bool
    mode: str

    def metrics(self, prefix: str = "resource") -> dict[str, float]:
        """Numeric form suitable for task records and ClearML scalars."""

        return {
            f"{prefix}_glue_values": float(self.glue_values),
            f"{prefix}_host_required_bytes": float(self.host_required_bytes),
            f"{prefix}_device_required_bytes": float(self.device_required_bytes),
            f"{prefix}_host_budget_bytes": float(self.host_budget_bytes),
            f"{prefix}_device_budget_bytes": float(self.device_budget_bytes),
            f"{prefix}_limit_values": float(self.limit_values),
            f"{prefix}_declined": 0.0 if self.accepted else 1.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageFootprint:
    """Representation-neutral, pre-allocation description of one search stage.

    The old guard counted abstract ``glue_values`` and consequently compared Python genes,
    decoded tensors, and optimizer state as if they had identical storage.  New callers describe
    the bytes they will actually retain.  ``work_operations`` is deliberately informational until
    a calibrated throughput is available; certain memory lower bounds remain sufficient to reject.
    """

    stage: str
    representation: str
    candidate_bytes: int
    population_size: int = 1
    optimizer_bytes: int = 0
    activation_bytes: int = 0
    transfer_bytes: int = 0
    work_operations: int = 0
    detail: str = ""

    @property
    def population_bytes(self) -> int:
        return max(0, self.candidate_bytes) * max(1, self.population_size)

    @property
    def total_bytes(self) -> int:
        return self.population_bytes + max(0, self.optimizer_bytes) + max(0, self.activation_bytes) + max(0, self.transfer_bytes)


@dataclass(frozen=True)
class StageDecision:
    footprint: StageFootprint
    host_budget_bytes: int
    device_budget_bytes: int
    accepted: bool
    reason: str | None

    def metrics(self, prefix: str = "resource") -> dict[str, float]:
        fp = self.footprint
        return {
            f"{prefix}_candidate_bytes": float(fp.candidate_bytes),
            f"{prefix}_population_bytes": float(fp.population_bytes),
            f"{prefix}_optimizer_bytes": float(fp.optimizer_bytes),
            f"{prefix}_activation_bytes": float(fp.activation_bytes),
            f"{prefix}_transfer_bytes": float(fp.transfer_bytes),
            f"{prefix}_total_bytes": float(fp.total_bytes),
            f"{prefix}_work_operations": float(fp.work_operations),
            f"{prefix}_declined": 0.0 if self.accepted else 1.0,
        }


@dataclass(frozen=True)
class ResourcePolicy:
    """Memory envelope used by the adaptive guards.

    ``device_bytes_per_value`` covers a float32 parameter, gradient, two Adam moments, and a small
    assembly margin.  The estimate is intentionally conservative and stable across platforms.
    """

    mode: str = "fixed"
    host_fraction: float = 0.55
    device_fraction: float = 0.70
    host_reserve_bytes: int = 4 * 1024**3
    device_reserve_bytes: int = 2 * 1024**3
    tuple_bytes_per_value: int = 32
    f32_bytes_per_value: int = 4
    device_bytes_per_value: int = 20

    @classmethod
    def from_config(cls, value: Mapping[str, Any] | None) -> "ResourcePolicy":
        table = dict(value or {})
        mode = str(table.get("mode", "fixed"))
        if mode not in {"fixed", "adaptive"}:
            raise ValueError(f"unknown resource policy mode {mode!r}; expected 'fixed' or 'adaptive'")
        host_fraction = float(table.get("host_fraction", 0.55))
        device_fraction = float(table.get("device_fraction", 0.70))
        if not 0.0 < host_fraction <= 1.0 or not 0.0 < device_fraction <= 1.0:
            raise ValueError("resource fractions must be in (0, 1]")
        return cls(
            mode=mode,
            host_fraction=host_fraction,
            device_fraction=device_fraction,
            host_reserve_bytes=max(0, int(float(table.get("host_reserve_gb", 4.0)) * 1024**3)),
            device_reserve_bytes=max(0, int(float(table.get("device_reserve_gb", 2.0)) * 1024**3)),
            tuple_bytes_per_value=max(4, int(table.get("tuple_bytes_per_value", 32))),
            f32_bytes_per_value=max(4, int(table.get("f32_bytes_per_value", 4))),
            device_bytes_per_value=max(4, int(table.get("device_bytes_per_value", 20))),
        )

    def _budgets(self, device: torch.device | str) -> tuple[int, int]:
        device = torch.device(device)
        virtual_memory = psutil.virtual_memory()
        host_total = int(virtual_memory.total)
        host_available = int(getattr(virtual_memory, "available", host_total))
        cgroup = _cgroup_memory()
        if cgroup is not None:
            cgroup_limit, cgroup_remaining = cgroup
            host_total = min(host_total, cgroup_limit)
            host_available = min(host_available, cgroup_remaining)
        host_budget = max(0, min(int(host_total * self.host_fraction), host_available) - self.host_reserve_bytes)
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                total = int(torch.cuda.get_device_properties(device).total_memory)
            except (RuntimeError, AssertionError):
                total = 0
            try:
                free, runtime_total = torch.cuda.mem_get_info(device)
                total = min(total or int(runtime_total), int(runtime_total))
                free = int(free)
            except (RuntimeError, AssertionError):
                free = total
            device_budget = max(0, min(int(total * self.device_fraction), free) - self.device_reserve_bytes)
        elif device.type == "mps":
            # Apple unified memory is already represented by the host envelope.  Using the same
            # budget on both rails prevents double-counting it while retaining a device check.
            device_budget = host_budget
        else:
            device_budget = host_budget
        return host_budget, device_budget

    def assess_glue(
        self,
        glue_values: int,
        *,
        stage: str,
        storage: str,
        device: torch.device | str,
        fixed_limit: int = 0,
        population_multiplicity: int = 1,
        concurrent_trainers: int = 1,
    ) -> ResourceEstimate:
        values = max(0, int(glue_values))
        resolved_device = torch.device(device)
        population = max(1, int(population_multiplicity))
        trainers = max(1, int(concurrent_trainers))
        storage_bytes = self.f32_bytes_per_value if storage == "f32" else self.tuple_bytes_per_value
        host_required = values * storage_bytes * population
        device_required = values * self.device_bytes_per_value * trainers
        host_budget, device_budget = self._budgets(resolved_device)

        if fixed_limit > 0:
            limit = int(fixed_limit)
            accepted = values <= limit
            mode = "fixed"
        elif self.mode == "adaptive":
            if resolved_device.type == "cuda":
                host_limit = host_budget // max(storage_bytes * population, 1)
                device_limit = device_budget // max(self.device_bytes_per_value * trainers, 1)
                limit = max(0, min(host_limit, device_limit))
            else:
                # CPU and Apple Metal consume the same physical memory pool.
                limit = max(0, host_budget // max(storage_bytes * population + self.device_bytes_per_value * trainers, 1))
            accepted = values <= limit
            mode = "adaptive"
        else:
            limit = 0
            accepted = True
            mode = "disabled"

        return ResourceEstimate(
            stage=stage,
            glue_values=values,
            storage=storage,
            population_multiplicity=population,
            concurrent_trainers=trainers,
            host_required_bytes=host_required,
            device_required_bytes=device_required,
            host_budget_bytes=host_budget,
            device_budget_bytes=device_budget,
            limit_values=limit,
            accepted=accepted,
            mode=mode,
        )

    def assess_stage(self, footprint: StageFootprint, *, device: torch.device | str = "cpu") -> StageDecision:
        """Assess actual stage residency without assigning meaning to a generic value count."""

        host_budget, device_budget = self._budgets(device)
        required = footprint.total_bytes
        if self.mode != "adaptive":
            accepted = True
        elif torch.device(device).type in {"cuda", "mps"}:
            accepted = required <= host_budget and required <= device_budget
        else:
            accepted = required <= host_budget
        reason = None if accepted else f"{footprint.representation} stage requires {format_bytes(required)}; budget is {format_bytes(min(host_budget, device_budget))}"
        return StageDecision(footprint, host_budget, device_budget, accepted, reason)


def format_bytes(value: int) -> str:
    """Compact binary byte count for resource-decline log messages."""

    size = float(max(0, value))
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if size < 1024.0 or suffix == "EiB":
            return f"{size:.1f} {suffix}"
        size /= 1024.0
    return f"{size:.1f} EiB"
