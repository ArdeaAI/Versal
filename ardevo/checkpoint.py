"""Checkpoint / resume for the orchestrated run.

The trial writes a rolling `checkpoint.json` at the run root after EVERY task (plus per-admission
`task_<NNNN>/` artifact dirs), holding everything needed to resume between tasks bit-for-bit: the
task cursor, the RNG state, the scheduler cursors, the species niches, and the hierarchical loop
state. The library is file-persistent and append-only, so it checkpoints itself.
"""

import json
import random
from pathlib import Path
from typing import Any


def serialize_rng(rng: random.Random) -> dict[str, Any]:
    version, internal, gauss_next = rng.getstate()
    return {"version": version, "internal": list(internal), "gauss_next": gauss_next}


def deserialize_rng(data: dict[str, Any]) -> random.Random:
    rng = random.Random()
    rng.setstate((int(data["version"]), tuple(int(value) for value in data["internal"]), data["gauss_next"]))
    return rng


def write_checkpoint(directory: Path, payload: dict[str, Any]) -> Path:
    path = directory / "checkpoint.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_checkpoint(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "checkpoint.json").read_text())


def build_orchestrated_payload(
    *, task_cursor: int, rng: random.Random, scheduler: Any, speciator: Any, loop_state: dict[str, Any], attempts: list[dict[str, Any]], counters: dict[str, int]
) -> dict[str, Any]:
    """The orchestrated run's between-task resumable state. The library is file-persistent and
    append-only, so it checkpoints itself; an in-flight task simply restarts from its lookup step."""
    return {
        "task_cursor": task_cursor,
        "rng": serialize_rng(rng),
        "schedule": scheduler.state_dict(),
        "speciation": speciator.state_dict(),
        "loop_state": loop_state,
        "attempts": attempts,
        "counters": counters,
    }


def latest_task_checkpoint_dir(run_directory: Path) -> Path | None:
    """The most recent `task_*/` under an orchestrated run dir holding a checkpoint, or None."""
    for candidate in sorted(run_directory.glob("task_*"), reverse=True):
        if (candidate / "checkpoint.json").exists():
            return candidate
    return None
