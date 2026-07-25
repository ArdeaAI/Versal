"""Spatial structure for the image/grid wall: the spatial_patches decompose op (grid->grid bands),
plus verification that the geometry operators and refine gene are wired into the orchestrated config
so the stamped grid coordinates actually drive local-receptive-field growth."""

import random

import torch

from ardevo.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from ardevo.decompose import spatial_patches


def _grid_to_grid_task(height: int = 4, width: int = 3, n: int = 6) -> Task:
    """A deterministic grid->grid task (invert each cell): both sides carry HEIGHT and WIDTH axes."""
    pairs = []
    for index in range(n):
        grid = ((torch.arange(height * width) + index).reshape(height, width) % 2).float()
        x = Field(grid, (Axis.HEIGHT, Axis.WIDTH), ValueType.BINARY, None, None, None)
        y = Field(1.0 - grid, (Axis.HEIGHT, Axis.WIDTH), ValueType.BINARY, None, None, None)
        pairs.append((x, y))
    return Task(meta=TaskMeta(rung=0, kind=TaskKind.MAP, name="grid_invert", fixed_split=True), support=pairs[:4], query=pairs[4:])


def test_spatial_patches_tiles_the_height_axis() -> None:
    task = _grid_to_grid_task(height=4, width=3)
    subtasks = spatial_patches(task, rng=random.Random(0), n_patches=2)
    assert len(subtasks) == 2
    assert all(subtask.port.role == "spatial_patch" for subtask in subtasks)
    assert [subtask.port.offsets for subtask in subtasks] == [(0, 2), (2, 4)]
    for subtask in subtasks:
        inp, out = subtask.task.support[0]
        assert inp.data.shape == (2, 3) and out.data.shape == (2, 3)
    # The bands tile back losslessly along the height axis.
    reconstructed = torch.cat([subtasks[0].task.support[0][0].data, subtasks[1].task.support[0][0].data], dim=0)
    assert torch.equal(reconstructed, task.support[0][0].data)


def test_spatial_patches_empty_for_classification(xor_task: Task) -> None:
    # xor input/output carry only the EXTRA axis: no shared spatial axis to tile.
    assert spatial_patches(xor_task, rng=random.Random(0)) == []


def test_spatial_patches_uses_other_reducible_axis_when_height_is_short() -> None:
    task = _grid_to_grid_task(height=1, width=3)
    subtasks = spatial_patches(task, rng=random.Random(0), n_patches=2)
    assert len(subtasks) == 2
    assert [child.task.support[0][0].data.shape for child in subtasks] == [(1, 2), (1, 1)]


def test_orchestrated_config_wires_geometry_refine_and_spatial() -> None:
    """The image-wall levers must actually be reachable from the production config."""
    from ardevo.evolution.registry import build_loop
    from ardevo.utils.config import Config

    config = Config(conf_path="configs/canary.toml").current
    operators = set(config["evolution"]["mutation"]["operators"])
    assert {"add_local_node", "add_local_connection", "add_shared_motif", "tweak_refine_steps"} <= operators
    assert "spatial_patches" in config["orchestrator"]["decompose"]
    build_loop(config)  # every named operator resolves without error
