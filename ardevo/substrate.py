"""Substrate: decode a genome into an executable torch graph network.

`GraphNet` runs an arbitrary feedforward DAG. Connection weights are real `nn.Parameter`s
initialized from the genome, so the `gradient` train operator can backprop into them; the
`none` operator simply reads them and leaves them untouched. A constant-1 bias node is
injected so a single hidden node suffices to solve XOR.
"""

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
    """Executable form of a genome: a topologically-ordered weighted DAG."""

    def __init__(self, genome: Genome, n_inputs: int, n_outputs: int) -> None:
        super().__init__()
        input_ids = genome.input_ids
        bias_ids = genome.bias_ids
        output_ids = genome.output_ids
        if len(input_ids) != n_inputs:
            raise ValueError(f"genome has {len(input_ids)} inputs, expected {n_inputs}")
        if len(output_ids) != n_outputs:
            raise ValueError(f"genome has {len(output_ids)} outputs, expected {n_outputs}")

        self._input_ids = input_ids
        self._bias_ids = set(bias_ids)
        self._output_ids = output_ids
        self._order = topological_order(genome)
        self._kind = {node.id: node.kind for node in genome.nodes.values()}
        self._activation = {node.id: node.activation for node in genome.nodes.values()}

        # Enabled connections become parameters in a fixed order; incoming[out_id] lists the
        # (source node, weight index) pairs feeding each node.
        enabled = genome.enabled_connections()
        self.weights = nn.Parameter(torch.tensor([conn.weight for conn in enabled], dtype=torch.float32))
        self._edges = [(conn.in_id, conn.out_id) for conn in enabled]
        self._incoming: dict[int, list[tuple[int, int]]] = {}
        for index, (in_id, out_id) in enumerate(self._edges):
            self._incoming.setdefault(out_id, []).append((in_id, index))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        values: dict[int, torch.Tensor] = {}
        for column, node_id in enumerate(self._input_ids):
            values[node_id] = x[:, column]
        for node_id in self._bias_ids:
            values[node_id] = torch.ones(batch, dtype=x.dtype, device=x.device)

        for node_id in self._order:
            if node_id in values:  # inputs and bias are already populated
                continue
            total = torch.zeros(batch, dtype=x.dtype, device=x.device)
            for in_id, index in self._incoming.get(node_id, []):
                total = total + self.weights[index] * values[in_id]
            values[node_id] = _ACTIVATIONS[self._activation[node_id]](total)

        return torch.stack([values[node_id] for node_id in self._output_ids], dim=1)

    def export_weights(self) -> dict[tuple[int, int], float]:
        """Current edge weights keyed by (in_id, out_id), for Lamarckian writeback."""
        detached = self.weights.detach().cpu().tolist()
        return {edge: detached[index] for index, edge in enumerate(self._edges)}


def decode(genome: Genome, n_inputs: int, n_outputs: int) -> GraphNet:
    """Build the torch module for a genome."""
    return GraphNet(genome, n_inputs, n_outputs)
