"""Train operators: an independent stage that optimizes a decoded candidate's weights.

Runs between mutation and evaluation. `none` leaves the co-evolved weights alone (phase-1
default). `gradient` backprops on the support set for `steps` using the differentiable Icarus
`loss_fn`. `writeback` controls Lamarckian (tuned weights copied into the genome) vs Baldwinian
(tuned only for this scoring) behavior. A future `cmaes` operator (evosax) drops in here.
"""

import random
from dataclasses import replace
from typing import Callable

import torch

from ardevo.dataset.icarus import EncodedTask
from ardevo.evaluation import support_loss
from ardevo.evolution.genome import Genome
from ardevo.evolution.registry import Registry
from ardevo.substrate import GraphNet

TrainOp = Callable[..., tuple[Genome, GraphNet]]

TRAIN: Registry[TrainOp] = Registry("train")


@TRAIN.register("none")
def no_train(genome: Genome, module: GraphNet, encoded: EncodedTask, *, rng: random.Random, **_params: object) -> tuple[Genome, GraphNet]:
    return genome, module


@TRAIN.register("gradient")
def gradient(
    genome: Genome,
    module: GraphNet,
    encoded: EncodedTask,
    *,
    rng: random.Random,
    steps: int = 20,
    lr: float = 0.01,
    writeback: bool = True,
) -> tuple[Genome, GraphNet]:
    if steps <= 0 or module.weights.numel() == 0:
        return genome, module
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = support_loss(module, encoded)
        loss.backward()
        optimizer.step()
    if writeback:
        genome = _writeback(genome, module)
    return genome, module


def _writeback(genome: Genome, module: GraphNet) -> Genome:
    """Copy the module's tuned weights back onto the matching enabled connection genes."""
    tuned = module.export_weights()
    child = genome.clone()
    child.connections = [replace(conn, weight=tuned[(conn.in_id, conn.out_id)]) if conn.enabled and (conn.in_id, conn.out_id) in tuned else conn for conn in child.connections]
    return child
