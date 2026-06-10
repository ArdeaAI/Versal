"""Load a single Icarus task for a rung via the map-style `IcarusDataset`."""

from ardevo.dataset.icarus import IcarusDataset, Task


def load_rung_task(source: str, rung: int, n_samples: int, seed: int = 0) -> Task:
    """Return the first task of `rung` from `source`.

    `source` is passed as `hf_repo` explicitly: the `IcarusDataset` default carries the
    underscore-spelling bug, while the live Hub id is the hyphen form `Ardea/Icarus-dataset`.
    """
    dataset = IcarusDataset(rungs=(rung,), n_tasks=1, n_samples=n_samples, hf_repo=source, seed=seed)
    if len(dataset) == 0:
        raise RuntimeError(f"no tasks found for rung {rung} in {source!r}")
    return dataset[0]
