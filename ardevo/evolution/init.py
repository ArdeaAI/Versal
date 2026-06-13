"""Initialization operators: how the starting population is seeded.

`minimal` is the NEAT growth-from-nothing seed: inputs and a bias wired straight to the
outputs, no hidden nodes. On a non-linearly-separable task like XOR this seed cannot win, so
selection pressure forces structural growth.
"""

import random
from typing import Callable

from ardevo.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind
from ardevo.evolution.registry import Registry

InitOp = Callable[..., Genome]

INIT: Registry[InitOp] = Registry("init")


@INIT.register("minimal")
def minimal(
    n_inputs: int,
    n_outputs: int,
    *,
    rng: random.Random,
    default_activation: str = "tanh",
    weight_scale: float = 1.0,
) -> Genome:
    """Inputs + bias fully connected to linear-readout outputs; no hidden nodes."""
    nodes: dict[int, NodeGene] = {}
    next_id = 0

    input_ids: list[int] = []
    for _ in range(n_inputs):
        nodes[next_id] = NodeGene(next_id, NodeKind.INPUT, "identity")
        input_ids.append(next_id)
        next_id += 1

    bias_id = next_id
    nodes[bias_id] = NodeGene(bias_id, NodeKind.BIAS, "identity")
    next_id += 1

    output_ids: list[int] = []
    for _ in range(n_outputs):
        # Outputs are linear readouts (raw logits): loss_fn / decode apply the squashing.
        nodes[next_id] = NodeGene(next_id, NodeKind.OUTPUT, "identity")
        output_ids.append(next_id)
        next_id += 1

    connections: list[ConnectionGene] = []
    innovation = 0
    for source in [*input_ids, bias_id]:
        for target in output_ids:
            connections.append(ConnectionGene(source, target, rng.gauss(0.0, weight_scale), True, innovation))
            innovation += 1

    return Genome(nodes=nodes, connections=connections)


def stamp_input_coordinates(genome: Genome, input_shape: tuple[int, ...]) -> Genome:
    """Stamp each INPUT node with its raw unraveled axis-index coordinate (id order = raveled
    row-major order, the exact convention `MultiTaskSubstrate._grow_bank` uses).

    This is what lets the geometry-biased mutators (add_local_node, add_local_connection,
    add_shared_motif) grow LOCAL receptive fields on grid tasks in the DIRECT path, where genomes
    are seeded by `minimal` instead of the multitask substrate."""
    from dataclasses import replace

    total = 1
    for dim in input_shape:
        total *= int(dim)
    input_ids = genome.input_ids
    if total != len(input_ids):
        raise ValueError(f"input shape {input_shape} has {total} cells but the genome has {len(input_ids)} inputs")
    child = genome.clone()
    for flat, node_id in enumerate(input_ids):
        index: list[float] = []
        remainder = flat
        for dim in reversed(input_shape):
            index.append(float(remainder % int(dim)))
            remainder //= int(dim)
        child.nodes[node_id] = replace(child.nodes[node_id], coordinate=tuple(reversed(index)))
    return child
