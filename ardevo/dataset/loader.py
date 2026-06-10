"""Load a single Icarus task for a rung via the map-style `IcarusDataset`."""

from ardevo.dataset.icarus import IcarusDataset, Task


def load_rung_task(source: str, rung: int, n_samples: int, seed: int = 0, support_fraction: float = 0.8) -> Task:
    """Return the first task of `rung` from `source`.

    `source` is passed as `hf_repo` explicitly: the `IcarusDataset` default carries the
    underscore-spelling bug, while the live Hub id is the hyphen form `Ardea/Icarus-dataset`.

    `support_fraction` controls the support/query split for bucketed (non-fixed-split) tasks; a
    lower value yields more query points (a smoother fitness signal) at the cost of training data.
    It is ignored for fixed-split tasks like XOR.
    """
    dataset = IcarusDataset(rungs=(rung,), n_tasks=1, n_samples=n_samples, support_fraction=support_fraction, hf_repo=source, seed=seed)
    if len(dataset) == 0:
        raise RuntimeError(f"no tasks found for rung {rung} in {source!r}")
    return dataset[0]
