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

from ardevo.evolution.genome import Genome, macro_implied_edges, topological_order

_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "identity": lambda value: value,
}


def activation_names() -> list[str]:
    return list(_ACTIVATIONS)


# Macro nodes resolve their inner network at decode time. The resolver maps a library key to the
# inner Genome. A module-level default keeps the six decode call sites signature-stable; explicit
# `macro_resolver=` arguments override it (the hierarchical/orchestrated paths pass theirs).
MacroResolver = Callable[[str], Genome]
_DEFAULT_MACRO_RESOLVER: MacroResolver | None = None
_MAX_MACRO_DEPTH = 4


def set_macro_resolver(resolver: MacroResolver | None) -> None:
    global _DEFAULT_MACRO_RESOLVER
    _DEFAULT_MACRO_RESOLVER = resolver


class SubstrateModule(nn.Module):
    """Executable substrate: an nn.Module that also reports edge presence and exports its weights.

    `GraphNet` is the direct form; the multi-task trial wraps it to slice one task's output head.
    Both subclass this so the evolver / train / eval stages can treat any substrate uniformly.
    Exported weights are keyed `(in_id, out_id, recurrent)` so a forward and a recurrent edge
    between the same node pair never collide.
    """

    @property
    def has_edges(self) -> bool:
        raise NotImplementedError

    def export_weights(self) -> dict[tuple[int, int, bool], float]:
        raise NotImplementedError

    def core(self) -> tuple["GraphNet | None", "torch.Tensor | None"]:
        """The plain `GraphNet` behind this substrate plus optional output-column selection, for
        population-batched training. (None, None) means "not batchable" (the caller falls back)."""
        return None, None


class GraphNet(SubstrateModule):
    """Executable form of a genome: a topologically-layered weighted DAG evaluated level by level."""

    def __init__(self, genome: Genome, n_inputs: int, n_outputs: int, *, macro_resolver: MacroResolver | None = None, _macro_depth: int = 0) -> None:
        super().__init__()
        input_ids, bias_ids, output_ids = genome.input_ids, genome.bias_ids, genome.output_ids
        if len(input_ids) != n_inputs:
            raise ValueError(f"genome has {len(input_ids)} inputs, expected {n_inputs}")
        if len(output_ids) != n_outputs:
            raise ValueError(f"genome has {len(output_ids)} outputs, expected {n_outputs}")

        order = topological_order(genome)
        position = {node_id: index for index, node_id in enumerate(order)}
        self.n = len(order)
        self._position = position

        # Only FORWARD edges run here: recurrent genes are time-delayed and inert without a time
        # axis (RecurrentGraphNet gives them semantics).
        weight_matrix = torch.zeros(self.n, self.n)
        mask = torch.zeros(self.n, self.n, dtype=torch.bool)
        incoming: dict[int, list[int]] = defaultdict(list)
        self._edge_positions: list[tuple[int, int, int, int]] = []
        for conn in genome.forward_connections():
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

        # Macro-implied edges (input_i -> output_j) join the depth computation so all of a macro's
        # outputs land at 1 + max(input depths); they carry no weight-matrix entries.
        for source_id, target_id in macro_implied_edges(genome):
            incoming[position[target_id]].append(position[source_id])

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
        aggregation_of = {position[node.id]: node.aggregation for node in genome.nodes.values()}
        by_depth: dict[int, list[int]] = defaultdict(list)
        for node_position, level in depth.items():
            if level >= 1:
                by_depth[level].append(node_position)

        self._levels: list[tuple[torch.Tensor, list[tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]]]] = []
        # Product nodes per level: (local column in the level, node position, incoming source positions).
        # The level matmul computes the SUM for every column; these columns are then overwritten with
        # prod(w_ij * x_i). Kept as a sparse per-node list because product nodes are mutation-gated rare.
        self._product_entries: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        # Per-node bookkeeping the population-batched trainer reads: which level computes each node
        # position, and each computed node's non-identity activation name.
        self.level_of: dict[int, int] = {}
        self.activation_name_of: dict[int, str] = {}
        for level in sorted(by_depth):
            level_positions = sorted(by_depth[level])
            groups: dict[str, list[int]] = defaultdict(list)
            for local_index, node_position in enumerate(level_positions):
                groups[activation_of[node_position]].append(local_index)
                self.level_of[node_position] = len(self._levels)
                if activation_of[node_position] != "identity":
                    self.activation_name_of[node_position] = activation_of[node_position]
            activation_groups = [
                (_ACTIVATIONS[name], torch.tensor(local_indices, dtype=torch.long))
                for name, local_indices in groups.items()
                if name != "identity"  # identity is a no-op; leave those columns as the raw pre-activation
            ]
            products = [
                (
                    torch.tensor([local_index], dtype=torch.long),
                    torch.tensor([node_position], dtype=torch.long),
                    torch.tensor(sorted(incoming[node_position]), dtype=torch.long),
                )
                for local_index, node_position in enumerate(level_positions)
                if aggregation_of[node_position] == "product" and incoming.get(node_position)
            ]
            self._levels.append((torch.tensor(level_positions, dtype=torch.long), activation_groups))
            self._product_entries.append(products)

        # Macro entries per level: (local output columns in macro-gene order, ordered input
        # positions, frozen inner module). Inner output order convention: the inner genome's sorted
        # output ids correspond positionally to macro.output_node_ids.
        self._macro_inner = nn.ModuleList()
        self._macro_entries: list[list[tuple[torch.Tensor, torch.Tensor, "GraphNet"]]] = [[] for _ in self._levels]
        if genome.macros:
            resolver = macro_resolver or _DEFAULT_MACRO_RESOLVER
            if resolver is None:
                raise ValueError("genome has macro nodes but no macro resolver is configured (set_macro_resolver or pass macro_resolver=)")
            if _macro_depth >= _MAX_MACRO_DEPTH:
                raise ValueError(f"macro nesting exceeds depth {_MAX_MACRO_DEPTH}")
            level_index_of = {int(level_positions[i]): (level, i) for level, (level_positions, _groups) in enumerate(self._levels) for i in range(len(level_positions))}
            for macro in genome.macros:
                inner_genome = resolver(macro.ref.removeprefix("library:"))
                if len(inner_genome.input_ids) != len(macro.input_node_ids) or len(inner_genome.output_ids) != len(macro.output_node_ids):
                    raise ValueError(
                        f"macro {macro.ref} shape mismatch: inner {len(inner_genome.input_ids)}->{len(inner_genome.output_ids)}, "
                        f"placement {len(macro.input_node_ids)}->{len(macro.output_node_ids)}"
                    )
                inner = GraphNet(inner_genome, len(inner_genome.input_ids), len(inner_genome.output_ids), macro_resolver=resolver, _macro_depth=_macro_depth + 1)
                if not macro.trainable:
                    for parameter in inner.parameters():
                        parameter.requires_grad_(False)
                self._macro_inner.append(inner)
                levels_hit = {level_index_of[position[node_id]][0] for node_id in macro.output_node_ids}
                if len(levels_hit) != 1:  # pragma: no cover - the implied edges force one level
                    raise ValueError(f"macro {macro.ref} outputs span levels {levels_hit}")
                level = levels_hit.pop()
                local_indices = torch.tensor([level_index_of[position[node_id]][1] for node_id in macro.output_node_ids], dtype=torch.long)
                input_positions = torch.tensor([position[node_id] for node_id in macro.input_node_ids], dtype=torch.long)
                self._macro_entries[level].append((local_indices, input_positions, inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        values = torch.zeros(batch, self.n, dtype=x.dtype, device=x.device)
        if self.input_pos.numel():
            values = values.index_copy(1, self.input_pos, x)
        if self.bias_pos.numel():
            values = values.index_copy(1, self.bias_pos, torch.ones(batch, self.bias_pos.numel(), dtype=x.dtype, device=x.device))

        masked = self.weights * self.mask
        for (level_positions, activation_groups), products, macros in zip(self._levels, self._product_entries, self._macro_entries):
            pre_activation = values @ masked[:, level_positions]
            for local_index, node_position, source_positions in products:
                edge_weights = masked.index_select(0, source_positions).index_select(1, node_position).squeeze(1)
                factors = values.index_select(1, source_positions) * edge_weights
                pre_activation = pre_activation.index_copy(1, local_index, factors.prod(dim=1, keepdim=True))
            for local_indices, input_positions, inner in macros:
                pre_activation = pre_activation.index_copy(1, local_indices, inner(values.index_select(1, input_positions)))
            activated = pre_activation
            for activation, local_indices in activation_groups:
                activated = activated.index_copy(1, local_indices, activation(activated.index_select(1, local_indices)))
            values = values.index_copy(1, level_positions, activated)

        return values.index_select(1, self.output_pos)

    @property
    def has_edges(self) -> bool:
        return bool(self.mask.any())

    def export_weights(self) -> dict[tuple[int, int, bool], float]:
        """Current edge weights keyed by (in_id, out_id, recurrent), for Lamarckian writeback."""
        detached = self.weights.detach()
        return {(in_id, out_id, False): float(detached[source, target]) for in_id, out_id, source, target in self._edge_positions}

    def core(self) -> tuple["GraphNet | None", "torch.Tensor | None"]:
        # Only the EXACT GraphNet form is batchable (RecurrentGraphNet steps over time; product and
        # macro entries change the math; all fall back to the sequential path).
        if type(self) is GraphNet and not any(self._product_entries) and not any(self._macro_entries):
            return self, None
        return None, None


class RecurrentGraphNet(GraphNet):
    """Stepped form of a genome for tasks with a TIME axis.

    Forward edges run within a step exactly as in `GraphNet`; recurrent edges read the PREVIOUS
    step's node values, so cycles through them are legal and state persists across the sequence.
    Input is `[batch, time, features]`; output is the last step's readout (`mode="last"`, seq-to-one)
    or every step's readout flattened t-major (`mode="all"`, seq-to-seq, matching the Level0 flat
    target layout). Recurrent edges into input/bias nodes are inert: those values are overwritten
    from the sequence each step.
    """

    def __init__(self, genome: Genome, n_inputs: int, n_outputs: int, mode: str = "last", *, macro_resolver: MacroResolver | None = None) -> None:
        super().__init__(genome, n_inputs, n_outputs, macro_resolver=macro_resolver)
        if mode not in ("last", "all"):
            raise ValueError(f"unknown recurrent output mode {mode!r}; expected 'last' or 'all'")
        self.mode = mode

        recurrent_matrix = torch.zeros(self.n, self.n)
        recurrent_mask = torch.zeros(self.n, self.n, dtype=torch.bool)
        recurrent_incoming: dict[int, list[int]] = defaultdict(list)
        self._recurrent_edge_positions: list[tuple[int, int, int, int]] = []
        for conn in genome.recurrent_connections():
            source, target = self._position[conn.in_id], self._position[conn.out_id]
            recurrent_matrix[source, target] = conn.weight
            recurrent_mask[source, target] = True
            recurrent_incoming[target].append(source)
            self._recurrent_edge_positions.append((conn.in_id, conn.out_id, source, target))
        self.recurrent_weights = nn.Parameter(recurrent_matrix)
        self.recurrent_mask: torch.Tensor = recurrent_mask

        # Product nodes here must fold the recurrent term in as extra FACTORS (not a summed bias), so
        # rebuild the per-level product entries with both forward and recurrent source positions.
        forward_incoming: dict[int, list[int]] = defaultdict(list)
        for conn in genome.forward_connections():
            forward_incoming[self._position[conn.out_id]].append(self._position[conn.in_id])
        aggregation_of = {self._position[node.id]: node.aggregation for node in genome.nodes.values()}
        self._recurrent_products: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        for level_positions, _activation_groups in self._levels:
            entries = []
            for local_index, node_position in enumerate(level_positions.tolist()):
                if aggregation_of[node_position] != "product":
                    continue
                if not forward_incoming.get(node_position) and not recurrent_incoming.get(node_position):
                    continue
                entries.append(
                    (
                        torch.tensor([local_index], dtype=torch.long),
                        torch.tensor([node_position], dtype=torch.long),
                        torch.tensor(sorted(forward_incoming.get(node_position, [])), dtype=torch.long),
                        torch.tensor(sorted(recurrent_incoming.get(node_position, [])), dtype=torch.long),
                    )
                )
            self._recurrent_products.append(entries)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"RecurrentGraphNet expects [batch, time, features], got shape {tuple(x.shape)}")
        batch, steps, _features = x.shape
        masked = self.weights * self.mask
        recurrent_masked = self.recurrent_weights * self.recurrent_mask
        previous = torch.zeros(batch, self.n, dtype=x.dtype, device=x.device)
        outputs: list[torch.Tensor] = []
        for step in range(steps):
            values = torch.zeros(batch, self.n, dtype=x.dtype, device=x.device)
            if self.input_pos.numel():
                values = values.index_copy(1, self.input_pos, x[:, step])
            if self.bias_pos.numel():
                values = values.index_copy(1, self.bias_pos, torch.ones(batch, self.bias_pos.numel(), dtype=x.dtype, device=x.device))
            recurrent_in = previous @ recurrent_masked
            for (level_positions, activation_groups), products, macros in zip(self._levels, self._recurrent_products, self._macro_entries):
                pre_activation = values @ masked[:, level_positions] + recurrent_in.index_select(1, level_positions)
                for local_index, node_position, forward_sources, recurrent_sources in products:
                    factors = []
                    if forward_sources.numel():
                        forward_weights = masked.index_select(0, forward_sources).index_select(1, node_position).squeeze(1)
                        factors.append(values.index_select(1, forward_sources) * forward_weights)
                    if recurrent_sources.numel():
                        recurrent_edge_weights = recurrent_masked.index_select(0, recurrent_sources).index_select(1, node_position).squeeze(1)
                        factors.append(previous.index_select(1, recurrent_sources) * recurrent_edge_weights)
                    combined = torch.cat(factors, dim=1).prod(dim=1, keepdim=True)
                    pre_activation = pre_activation.index_copy(1, local_index, combined)
                for local_indices, input_positions, inner in macros:
                    pre_activation = pre_activation.index_copy(1, local_indices, inner(values.index_select(1, input_positions)))
                activated = pre_activation
                for activation, local_indices in activation_groups:
                    activated = activated.index_copy(1, local_indices, activation(activated.index_select(1, local_indices)))
                values = values.index_copy(1, level_positions, activated)
            previous = values
            outputs.append(values.index_select(1, self.output_pos))
        if self.mode == "last":
            return outputs[-1]
        return torch.stack(outputs, dim=1).reshape(batch, steps * self.output_pos.numel())

    @property
    def has_edges(self) -> bool:
        return bool(self.mask.any()) or bool(self.recurrent_mask.any())

    def export_weights(self) -> dict[tuple[int, int, bool], float]:
        exported = super().export_weights()
        detached = self.recurrent_weights.detach()
        exported.update({(in_id, out_id, True): float(detached[source, target]) for in_id, out_id, source, target in self._recurrent_edge_positions})
        return exported


def decode(genome: Genome, n_inputs: int, n_outputs: int, *, macro_resolver: MacroResolver | None = None) -> GraphNet:
    """Build the torch module for a genome."""
    return GraphNet(genome, n_inputs, n_outputs, macro_resolver=macro_resolver)


def decode_recurrent(genome: Genome, n_inputs: int, n_outputs: int, mode: str = "last", *, macro_resolver: MacroResolver | None = None) -> RecurrentGraphNet:
    """Build the stepped (time-axis) torch module for a genome."""
    return RecurrentGraphNet(genome, n_inputs, n_outputs, mode, macro_resolver=macro_resolver)
