"""Substrate: decode a genome into an executable torch graph network.

`GraphNet` runs an arbitrary feedforward DAG as a sequence of dense matmuls, one per topological
depth level (far faster than a per-edge Python loop, which matters once tasks have hundreds of
examples and the search runs for many generations). Connection weights live in a single weight
matrix `W` masked to the enabled edges, so they are real `nn.Parameter`s the `gradient` train
operator can backprop into. A constant-1 bias node is injected so a single hidden node suffices for
XOR; outputs are linear readouts (loss_fn / decode apply any squashing).
"""

from collections import defaultdict
from collections.abc import Callable

import torch
from torch import nn

from ardevo.evolution.genome import Genome, topological_order

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "identity": lambda value: value,
}


def activation_names() -> list[str]:
    return list(_ACTIVATIONS)


class GraphNet(nn.Module):
    """Executable form of a genome: a topologically-layered weighted DAG evaluated level by level."""

    def __init__(self, genome: Genome, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        input_ids, bias_ids, output_ids = genome.input_ids, genome.bias_ids, genome.output_ids
        if len(input_ids) != n_inputs:
            raise ValueError(f"genome has {len(input_ids)} inputs, expected {n_inputs}")
        if len(output_ids) != n_outputs:
            raise ValueError(f"genome has {len(output_ids)} outputs, expected {n_outputs}")

        order = topological_order(genome)
        position = {node_id: index for index, node_id in enumerate(order)}
        self.n = len(order)

        weight_matrix = torch.zeros(self.n, self.n)
        mask = torch.zeros(self.n, self.n, dtype=torch.bool)
        incoming: dict[int, list[int]] = defaultdict(list)
        self._edge_positions: list[tuple[int, int, int, int]] = []
        for conn in genome.enabled_connections():
            source, target = position[conn.in_id], position[conn.out_id]
            weight_matrix[source, target] = conn.weight
            mask[source, target] = True
            incoming[target].append(source)
            self._edge_positions.append((conn.in_id, conn.out_id, source, target))

        self.weights = nn.Parameter(weight_matrix)
        # Plain typed attributes (not buffers): the module is never moved off CPU here, and buffers
        # confuse the type checker about these tensors.
        self.mask: torch.Tensor = mask
        self.input_pos: torch.Tensor = torch.tensor([position[i] for i in input_ids], dtype=torch.long)
        self.bias_pos: torch.Tensor = torch.tensor([position[i] for i in bias_ids], dtype=torch.long)
        self.output_pos: torch.Tensor = torch.tensor([position[i] for i in output_ids], dtype=torch.long)

        # Longest-path depth per node: inputs/bias are depth 0 (set directly); every other node is one
        # past its deepest enabled predecessor (no intra-level edges can exist, so a level only reads
        # already-computed values). Each level is one matmul; activations are applied per group.
        sources = {position[i] for i in input_ids} | {position[i] for i in bias_ids}
        depth: dict[int, int] = {}
        for node_id in order:
            node_position = position[node_id]
            if node_position in sources:
                depth[node_position] = 0
            else:
                depth[node_position] = 1 + max((depth[pred] for pred in incoming.get(node_position, [])), default=0)

        activation_of = {position[node.id]: node.activation for node in genome.nodes.values()}
        by_depth: dict[int, list[int]] = defaultdict(list)
        for node_position, level in depth.items():
            if level >= 1:
                by_depth[level].append(node_position)

        self._levels: list[tuple[torch.Tensor, list[tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]]]] = []
        for level in sorted(by_depth):
            level_positions = sorted(by_depth[level])
            groups: dict[str, list[int]] = defaultdict(list)
            for local_index, node_position in enumerate(level_positions):
                groups[activation_of[node_position]].append(local_index)
            activation_groups = [
                (_ACTIVATIONS[name], torch.tensor(local_indices, dtype=torch.long))
                for name, local_indices in groups.items()
                if name != "identity"  # identity is a no-op; leave those columns as the raw pre-activation
            ]
            self._levels.append((torch.tensor(level_positions, dtype=torch.long), activation_groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        values = torch.zeros(batch, self.n, dtype=x.dtype, device=x.device)
        if self.input_pos.numel():
            values = values.index_copy(1, self.input_pos, x)
        if self.bias_pos.numel():
            values = values.index_copy(1, self.bias_pos, torch.ones(batch, self.bias_pos.numel(), dtype=x.dtype, device=x.device))

        masked = self.weights * self.mask
        for level_positions, activation_groups in self._levels:
            pre_activation = values @ masked[:, level_positions]
            activated = pre_activation
            for activation, local_indices in activation_groups:
                activated = activated.index_copy(1, local_indices, activation(activated.index_select(1, local_indices)))
            values = values.index_copy(1, level_positions, activated)

        return values.index_select(1, self.output_pos)

    @property
    def has_edges(self) -> bool:
        return bool(self.mask.any())

    def export_weights(self) -> dict[tuple[int, int], float]:
        """Current edge weights keyed by (in_id, out_id), for Lamarckian writeback."""
        detached = self.weights.detach()
        return {(in_id, out_id): float(detached[source, target]) for in_id, out_id, source, target in self._edge_positions}


def decode(genome: Genome, n_inputs: int, n_outputs: int) -> GraphNet:
    """Build the torch module for a genome."""
    return GraphNet(genome, n_inputs, n_outputs)
