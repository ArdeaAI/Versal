"""Checkpoint / resume for the continuous run.

Every `checkpoint_every` generations the trial drops a `gen_<NNNNNN>/` directory holding the human
artifacts (`model.json`, `stats.json`, `net.png`, `speciation.png`, via `results.py`) plus a
`checkpoint.json` with everything needed to resume bit-for-bit: the population genomes (weights ride
along via Lamarckian writeback), the innovation tracker, the RNG state, the species niches, the
scheduler cursors, and the growing-interface layout. On resume the population is re-assessed against
the active task, so per-task metrics are recomputed rather than stored.
"""

import json
import random
from pathlib import Path
from typing import Any

from ardevo.evolution.evolver import EvolverState
from ardevo.evolution.genome import genome_to_dict


def serialize_rng(rng: random.Random) -> dict[str, Any]:
    version, internal, gauss_next = rng.getstate()
    return {"version": version, "internal": list(internal), "gauss_next": gauss_next}


def deserialize_rng(data: dict[str, Any]) -> random.Random:
    rng = random.Random()
    rng.setstate((int(data["version"]), tuple(int(value) for value in data["internal"]), data["gauss_next"]))
    return rng


def build_payload(*, state: EvolverState, speciator: Any, scheduler: Any, substrate: Any, active_index: int) -> dict[str, Any]:
    """Gather the full resumable state into one JSON-able dict."""
    best = None
    if state.best is not None:
        best = {"genome": genome_to_dict(state.best.genome), "metrics": state.best.metrics, "fitness": state.best.fitness}
    return {
        "generation": state.generation,
        "active_index": active_index,
        "rng": serialize_rng(state.rng),
        "innovations": state.innovations.to_dict(),
        "species_history": state.species_history,
        "population": [genome_to_dict(item.genome) for item in state.population],
        "best": best,
        "speciation": speciator.state_dict(),
        "schedule": scheduler.state_dict(),
        "substrate": substrate.to_dict(),
    }


def write_checkpoint(directory: Path, payload: dict[str, Any]) -> Path:
    path = directory / "checkpoint.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_checkpoint(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "checkpoint.json").read_text())


def latest_checkpoint_dir(run_directory: Path) -> Path | None:
    """The most recent `gen_*/` under a run dir that actually holds a checkpoint, or None."""
    for candidate in sorted(run_directory.glob("gen_*"), reverse=True):
        if (candidate / "checkpoint.json").exists():
            return candidate
    return None


def restored_species_history(data: dict[str, Any]) -> list[dict[int, int]]:
    """JSON turns the species-id keys into strings; turn them back into ints."""
    return [{int(species_id): size for species_id, size in snapshot.items()} for snapshot in data["species_history"]]
