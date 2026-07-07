"""Initialization operators: how the starting population is seeded.

`minimal` is the NEAT growth-from-nothing seed: inputs and a bias wired straight to the
outputs, no hidden nodes. On a non-linearly-separable task like XOR this seed cannot win, so
selection pressure forces structural growth.

`cppn` is the generative seed (gate-E fork, ai/gate_e.md): each member samples its own tiny
random pair-query generator f(source_coord, target_coord) -> (weight, expression) and compiles
it into an ordinary explicit genome; the expression gate yields SPARSE, spatially patterned
connectivity instead of the dense bipartite (the rungs 11-14 init-wall lever), and the search
then proceeds on the flat genome exactly as with `minimal`. The generator is a process-level
regularity prior (weight patterns tend to be spatially coherent), never an architecture.
"""

import math
import random
from typing import Callable

import torch

from ardevo.evolution.genome import ConnectionGene, Genome, NodeGene, NodeKind
from ardevo.evolution.registry import Registry
from ardevo.substrate import _ACTIVATIONS

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


@INIT.register("factored")
def factored(
    n_inputs: int,
    n_outputs: int,
    *,
    rng: random.Random,
    default_activation: str = "tanh",
    weight_scale: float = 1.0,
    rank: int = 8,
    threshold: int = 4096,
) -> Genome:
    """Low-rank seed for wide I/O: inputs + bias densely wired to `rank` identity latent nodes,
    latents densely to outputs (plus bias -> output offsets). The rank-r factorization U x V is
    represented with ORDINARY genes (the latents are plain HIDDEN sum/identity nodes), so decode,
    training, crossover, pruning, and complexity() all see it natively: (n_in + 1) x n_out genes
    become (n_in + 1) x r + r x n_out + n_out (rung 11: 10.8M -> ~433k at r = 4). This is the
    composition layer's glue_rank auto-factorize pattern ported down to flat genomes. At or below
    `threshold` dense genes this IS `minimal` (same rng draws, byte-identical), so one configured
    init serves xor and spherical in the same ladder. Fan-in-scaled weights (Xavier-style) keep a
    100k-wide sum from saturating the latents at birth; innovations are pair-derived so identical
    edges align for NEAT crossover across independently seeded members."""
    if (n_inputs + 1) * n_outputs <= threshold:
        return minimal(n_inputs, n_outputs, rng=rng, default_activation=default_activation, weight_scale=weight_scale)

    nodes: dict[int, NodeGene] = {}
    input_ids = list(range(n_inputs))
    for node_id in input_ids:
        nodes[node_id] = NodeGene(node_id, NodeKind.INPUT, "identity")
    bias_id = n_inputs
    nodes[bias_id] = NodeGene(bias_id, NodeKind.BIAS, "identity")
    output_ids = list(range(n_inputs + 1, n_inputs + 1 + n_outputs))
    for node_id in output_ids:
        nodes[node_id] = NodeGene(node_id, NodeKind.OUTPUT, "identity")
    latent_ids = list(range(n_inputs + 1 + n_outputs, n_inputs + 1 + n_outputs + rank))
    for node_id, coordinate in zip(latent_ids, _index_continuum(rank)):
        nodes[node_id] = NodeGene(node_id, NodeKind.HIDDEN, "identity", coordinate=(coordinate,))

    total = n_inputs + 1 + n_outputs + rank
    u_scale = weight_scale / math.sqrt(n_inputs + 1)
    v_scale = weight_scale / math.sqrt(rank)
    connections: list[ConnectionGene] = []
    for source in [*input_ids, bias_id]:
        for latent in latent_ids:
            connections.append(ConnectionGene(source, latent, rng.gauss(0.0, u_scale), True, source * total + latent))
    for latent in latent_ids:
        for target in output_ids:
            connections.append(ConnectionGene(latent, target, rng.gauss(0.0, v_scale), True, latent * total + target))
    for target in output_ids:
        connections.append(ConnectionGene(bias_id, target, rng.gauss(0.0, weight_scale), True, bias_id * total + target))
    return Genome(nodes=nodes, connections=connections)


@INIT.register("sparse")
def sparse(
    n_inputs: int,
    n_outputs: int,
    *,
    rng: random.Random,
    default_activation: str = "tanh",
    weight_scale: float = 1.0,
    density: float = 0.01,
    threshold: int = 4096,
) -> Genome:
    """SET-style seed (Mocanu et al. 2018): an Erdos-Renyi sparse input -> output wiring instead of
    the dense bipartite, so a rung-11-class task starts at `density x n_in x n_out` genes and
    `prune_and_regrow` plus the gradient discover which direct paths matter. Keeps direct
    input -> output edges (the prior `factored` trades away), so it suits grid -> grid tasks where
    output j mostly depends on inputs near j. bias -> output floor edges are always present (no
    dead outputs at any density). At or below `threshold` dense genes this IS `minimal` (same rng
    draws, byte-identical). Each member samples its own edge subset; pair-derived innovations keep
    identical edges aligned for NEAT crossover."""
    if (n_inputs + 1) * n_outputs <= threshold:
        return minimal(n_inputs, n_outputs, rng=rng, default_activation=default_activation, weight_scale=weight_scale)

    nodes: dict[int, NodeGene] = {}
    input_ids = list(range(n_inputs))
    for node_id in input_ids:
        nodes[node_id] = NodeGene(node_id, NodeKind.INPUT, "identity")
    bias_id = n_inputs
    nodes[bias_id] = NodeGene(bias_id, NodeKind.BIAS, "identity")
    output_ids = list(range(n_inputs + 1, n_inputs + 1 + n_outputs))
    for node_id in output_ids:
        nodes[node_id] = NodeGene(node_id, NodeKind.OUTPUT, "identity")

    total = n_inputs + 1 + n_outputs
    pairs = n_inputs * n_outputs
    edge_count = min(pairs, max(n_outputs, round(pairs * density)))
    edge_scale = weight_scale / math.sqrt(max(1.0, edge_count / n_outputs))  # expected fan-in per output
    connections: list[ConnectionGene] = []
    for flat in sorted(rng.sample(range(pairs), edge_count)):
        source = input_ids[flat // n_outputs]
        target = output_ids[flat % n_outputs]
        connections.append(ConnectionGene(source, target, rng.gauss(0.0, edge_scale), True, source * total + target))
    for target in output_ids:
        connections.append(ConnectionGene(bias_id, target, rng.gauss(0.0, weight_scale), True, bias_id * total + target))
    return Genome(nodes=nodes, connections=connections)


def _index_continuum(count: int) -> list[float]:
    if count <= 1:
        return [0.0] * count
    return [2.0 * index / (count - 1) - 1.0 for index in range(count)]


class _PairGenerator:
    """A tiny hand-rolled MLP f(source, target) -> (weight, expression), sampled from `rng`.

    Hand-rolled (not a Genome/GraphNet) because it lives for one init call and must stay
    deterministic per member from the shared rng alone."""

    def __init__(self, rng: random.Random, generator_hidden: int, generator_activations: tuple[str, ...]) -> None:
        gauss = lambda: rng.gauss(0.0, 1.0)  # noqa: E731
        self.w_hidden = torch.tensor([[gauss() * math.pi, gauss() * math.pi, gauss()] for _ in range(generator_hidden)])
        self.activations = [_ACTIVATIONS[rng.choice(list(generator_activations))] for _ in range(generator_hidden)]
        self.w_out = torch.tensor([[gauss() for _ in range(generator_hidden + 3)] for _ in range(2)])

    def __call__(self, source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([source, target, torch.ones_like(source)], dim=1)  # [N, 3]
        pre = stacked @ self.w_hidden.T  # [N, generator_hidden]
        hidden = torch.stack([activation(pre[:, index]) for index, activation in enumerate(self.activations)], dim=1)
        out = torch.cat([hidden, stacked], dim=1) @ self.w_out.T  # [N, 2]
        return out[:, 0], out[:, 1]


@INIT.register("cppn")
def cppn_seed(
    n_inputs: int,
    n_outputs: int,
    *,
    rng: random.Random,
    default_activation: str = "tanh",
    weight_scale: float = 1.0,
    hidden: int = 16,
    density: float = 0.3,
    generator_hidden: int = 3,
    generator_activations: tuple[str, ...] = ("sin", "gaussian", "tanh"),
    hidden_activations: list[str] | None = None,
) -> Genome:
    """One structured random seed: generator-patterned input->hidden->output wiring.

    Node ids follow the `minimal` layout (inputs, bias, outputs, then hidden). Innovation numbers
    are PAIR-DERIVED (source * total + target) rather than sequential: every member draws a
    different generator, so identical (in, out) edges across members must still align for NEAT
    crossover, which the run's InnovationTracker only guarantees for post-init mutations. Hidden
    nodes carry their 1-D index-continuum coordinate, so the geometry mutators go live on the
    seeded structure. The expression gate keeps the top `density` fraction of pairs per block
    (a quantile, so the dial is scale-free across generators: density * pairs edges, exactly).
    bias->output edges are always present (the `minimal` floor: no dead outputs at any density)."""
    generator = _PairGenerator(rng, generator_hidden, generator_activations)
    hidden_palette = hidden_activations or [default_activation]

    nodes: dict[int, NodeGene] = {}
    input_ids = list(range(n_inputs))
    for node_id in input_ids:
        nodes[node_id] = NodeGene(node_id, NodeKind.INPUT, "identity")
    bias_id = n_inputs
    nodes[bias_id] = NodeGene(bias_id, NodeKind.BIAS, "identity")
    output_ids = list(range(n_inputs + 1, n_inputs + 1 + n_outputs))
    for node_id in output_ids:
        nodes[node_id] = NodeGene(node_id, NodeKind.OUTPUT, "identity")
    hidden_ids = list(range(n_inputs + 1 + n_outputs, n_inputs + 1 + n_outputs + hidden))
    hidden_coordinates = _index_continuum(hidden)
    for node_id, coordinate in zip(hidden_ids, hidden_coordinates):
        nodes[node_id] = NodeGene(node_id, NodeKind.HIDDEN, rng.choice(hidden_palette), coordinate=(coordinate,))

    total = n_inputs + 1 + n_outputs + hidden
    connections: list[ConnectionGene] = []

    def emit(source_ids: list[int], source_coords: list[float], target_ids: list[int], target_coords: list[float]) -> None:
        source_tensor = torch.tensor([s for s in source_coords for _ in target_coords])
        target_tensor = torch.tensor(target_coords * len(source_coords))
        weights, expressions = generator(source_tensor, target_tensor)
        keep = max(1, round(len(source_tensor) * density))
        cutoff = expressions.abs().sort(descending=True).values[keep - 1]
        gate = expressions.abs() >= cutoff
        for flat_index in gate.nonzero(as_tuple=True)[0].tolist():
            source_id = source_ids[flat_index // len(target_ids)]
            target_id = target_ids[flat_index % len(target_ids)]
            connections.append(ConnectionGene(source_id, target_id, float(weights[flat_index]) * weight_scale, True, source_id * total + target_id))

    emit(input_ids, _index_continuum(n_inputs), hidden_ids, hidden_coordinates)
    emit(hidden_ids, hidden_coordinates, output_ids, _index_continuum(n_outputs))
    for hidden_id in hidden_ids:  # free bias genes; the generator patterns receptive fields, not offsets
        connections.append(ConnectionGene(bias_id, hidden_id, rng.gauss(0.0, weight_scale), True, bias_id * total + hidden_id))
    for output_id in output_ids:
        connections.append(ConnectionGene(bias_id, output_id, rng.gauss(0.0, weight_scale), True, bias_id * total + output_id))
    return Genome(nodes=nodes, connections=connections)


def stamp_input_coordinates(genome: Genome, input_shape: tuple[int, ...]) -> Genome:
    """Stamp each INPUT node with its raw unraveled axis-index coordinate (id order = raveled
    row-major order).

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
