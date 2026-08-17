"""Substrate: decode a genome into an executable torch graph network.

`GraphNet` runs an arbitrary feedforward DAG as a sequence of dense matmuls, one per topological
depth level (far faster than a per-edge Python loop, which matters once tasks have hundreds of
examples and the search runs for many generations). Connection weights live in a single weight
matrix masked to the enabled edges, so they are real `nn.Parameter`s the `gradient` train
operator can backprop into. The matrix is COMPACT-COLUMN `[n, h]`: rows span all `n` nodes but
columns exist only for the `h` COMPUTED nodes (hidden + outputs), because the level loop only ever
reads columns at computed positions. On wide-input tasks (MNIST 784, CIFAR 3072 inputs) h stays
tiny while n is dominated by input pixels, so this drops the per-forward mask multiply, the
backward grad buffer, and the Adam state from O(n^2) to O(n*h) with bitwise-identical outputs
(the sliced GEMM operands are element-for-element the same as the dense layout's). A constant-1
bias node is injected so a single hidden node suffices for XOR; outputs are linear readouts
(loss_fn / decode apply any squashing).
"""

from collections import defaultdict
from collections.abc import Callable

import torch
from torch import nn

from versal.evolution.genome import Genome, macro_implied_edges, topological_order
from versal.reference_depth import DEFAULT_MAX_INLINE_DEPTH


def _gaussian(value: torch.Tensor) -> torch.Tensor:
    # Worker processes pickle activation callables by reference, so this cannot be a lambda.
    return torch.exp(-value * value)


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "identity": lambda value: value,  # special-cased at decode: never stored on a net, so never pickled
    # Periodic + radial primitives (the WANN/CPPN palette): a single sin neuron can carve a spiral
    # winding, a gaussian is a radial bump. Mutation-only; nothing seeds them (default stays tanh).
    "sin": torch.sin,
    "gaussian": _gaussian,
}


def activation_names() -> list[str]:
    return list(_ACTIVATIONS)


# Macro nodes resolve their inner network at decode time. The resolver maps a library key to the
# inner Genome. A module-level default keeps the six decode call sites signature-stable; explicit
# `macro_resolver=` arguments override it (the hierarchical/orchestrated paths pass theirs).
MacroResolver = Callable[[str], Genome]
_DEFAULT_MACRO_RESOLVER: MacroResolver | None = None
# Compatibility alias for code that imported the historical hard-coded cap. New runtime code must
# pass ``max_inline_depth`` from [evolution.composition] instead of consulting this constant.
_MAX_MACRO_DEPTH = DEFAULT_MAX_INLINE_DEPTH


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

    def __init__(
        self,
        genome: Genome,
        n_inputs: int,
        n_outputs: int,
        *,
        macro_resolver: MacroResolver | None = None,
        max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
        _reference_depth: int = 0,
        _reference_stack: tuple[str, ...] = (),
    ) -> None:
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
        # axis (RecurrentGraphNet gives them semantics). Edges are collected first because the
        # compact-column map depends on the depth computation below.
        forward_edges: list[tuple[int, int, int, int, float]] = []
        incoming: dict[int, list[int]] = defaultdict(list)
        for conn in genome.forward_connections():
            source, target = position[conn.in_id], position[conn.out_id]
            forward_edges.append((conn.in_id, conn.out_id, source, target, conn.weight))
            incoming[target].append(source)

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

        # Compact columns: the level loop only ever reads weight columns at COMPUTED node positions
        # (depth >= 1), so the weight matrix is [n, h] with one column per computed node.
        computed_positions = sorted(node_position for node_position, level in depth.items() if level >= 1)
        self.h = len(computed_positions)
        self._col_of: dict[int, int] = {node_position: col for col, node_position in enumerate(computed_positions)}
        self.col_index: torch.Tensor = torch.tensor(computed_positions, dtype=torch.long)

        weight_matrix = torch.zeros(self.n, self.h)
        mask = torch.zeros(self.n, self.h, dtype=torch.bool)
        self._edge_positions: list[tuple[int, int, int, int]] = []
        # Edges whose target has no compact column (forward or recurrent genes into input/bias
        # nodes) are inert: never read by any level, so they carry no parameter. Their genome
        # weights are kept verbatim for export/writeback completeness. (The dense layout let Adam
        # weight_decay drift these never-read weights as a side effect; freezing them matches
        # their documented inert semantics. Mutators never create them; only legacy genomes can.)
        self._inert_edges: list[tuple[int, int, bool, float]] = []
        # HARD WEIGHT SHARING: tied edges (ConnectionGene.tie_group) draw their value from ONE
        # shared parameter per group instead of a weight-matrix cell; forward overlays them onto
        # the masked matrix with index_put, so the gradient of a group's parameter accumulates
        # across every member edge (the convolution mechanic). The matrix cell stays 0 with mask
        # True: it is overwritten before any level reads it, receives zero gradient, and keeps
        # has_edges truthful. Untied genomes allocate nothing and skip the overlay entirely.
        tie_of_edge = {(conn.in_id, conn.out_id): conn.tie_group for conn in genome.forward_connections() if conn.tie_group is not None}
        group_slot: dict[int, int] = {}
        group_values: list[float] = []
        tie_rows: list[int] = []
        tie_cols: list[int] = []
        tie_slots: list[int] = []
        self._tied_edge_exports: list[tuple[int, int, int]] = []  # (in_id, out_id, slot)
        for in_id, out_id, source, target, weight in forward_edges:
            col = self._col_of.get(target)
            if col is None:
                self._inert_edges.append((in_id, out_id, False, weight))
                continue
            group = tie_of_edge.get((in_id, out_id))
            if group is not None:
                if group not in group_slot:
                    group_slot[group] = len(group_values)
                    group_values.append(weight)  # first member (gene order) seeds the shared value
                slot = group_slot[group]
                mask[source, col] = True
                tie_rows.append(source)
                tie_cols.append(col)
                tie_slots.append(slot)
                self._tied_edge_exports.append((in_id, out_id, slot))
                continue
            weight_matrix[source, col] = weight
            mask[source, col] = True
            self._edge_positions.append((in_id, out_id, source, col))

        self.weights = nn.Parameter(weight_matrix)
        self.tie_values = nn.Parameter(torch.tensor(group_values)) if group_values else None
        self._tie_rows: torch.Tensor = torch.tensor(tie_rows, dtype=torch.long)
        self._tie_cols: torch.Tensor = torch.tensor(tie_cols, dtype=torch.long)
        self._tie_slots: torch.Tensor = torch.tensor(tie_slots, dtype=torch.long)
        # Plain typed attributes (not buffers): the module is never moved off CPU here, and buffers
        # confuse the type checker about these tensors.
        self.mask: torch.Tensor = mask

        activation_of = {position[node.id]: node.activation for node in genome.nodes.values()}
        aggregation_of = {position[node.id]: node.aggregation for node in genome.nodes.values()}
        by_depth: dict[int, list[int]] = defaultdict(list)
        for node_position, level in depth.items():
            if level >= 1:
                by_depth[level].append(node_position)

        self._levels: list[tuple[torch.Tensor, torch.Tensor, list[tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]]]] = []
        # Product nodes per level: (local column in the level, compact weight column, incoming source
        # positions). The level matmul computes the SUM for every column; these columns are then
        # overwritten with prod(w_ij * x_i). Kept as a sparse per-node list because product nodes are
        # mutation-gated rare.
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
                    torch.tensor([self._col_of[node_position]], dtype=torch.long),
                    torch.tensor(sorted(incoming[node_position]), dtype=torch.long),
                )
                for local_index, node_position in enumerate(level_positions)
                if aggregation_of[node_position] == "product" and incoming.get(node_position)
            ]
            level_cols = torch.tensor([self._col_of[node_position] for node_position in level_positions], dtype=torch.long)
            self._levels.append((torch.tensor(level_positions, dtype=torch.long), level_cols, activation_groups))
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
            level_index_of = {int(level_positions[i]): (level, i) for level, (level_positions, _cols, _groups) in enumerate(self._levels) for i in range(len(level_positions))}
            for macro in genome.macros:
                key = macro.ref.removeprefix("library:")
                if key in _reference_stack:
                    raise ValueError(f"macro reference cycle through library entry {key!r}")
                if _reference_depth >= max_inline_depth:
                    raise ValueError(f"macro reference nesting exceeds max_inline_depth={max_inline_depth}")
                inner_genome = resolver(key)
                if len(inner_genome.input_ids) != len(macro.input_node_ids) or len(inner_genome.output_ids) != len(macro.output_node_ids):
                    raise ValueError(
                        f"macro {macro.ref} shape mismatch: inner {len(inner_genome.input_ids)}->{len(inner_genome.output_ids)}, "
                        f"placement {len(macro.input_node_ids)}->{len(macro.output_node_ids)}"
                    )
                inner = GraphNet(
                    inner_genome,
                    len(inner_genome.input_ids),
                    len(inner_genome.output_ids),
                    macro_resolver=resolver,
                    max_inline_depth=max_inline_depth,
                    _reference_depth=_reference_depth + 1,
                    _reference_stack=(*_reference_stack, key),
                )
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

    def _masked_weights(self) -> torch.Tensor:
        masked = self.weights * self.mask
        if self.tie_values is not None:
            # Overlay the shared parameters onto their member positions; index_select's backward
            # scatter-adds, so each group's gradient sums over every stamped copy.
            masked = masked.index_put((self._tie_rows, self._tie_cols), self.tie_values.index_select(0, self._tie_slots))
        return masked

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        values = torch.zeros(batch, self.n, dtype=x.dtype, device=x.device)
        if self.input_pos.numel():
            values = values.index_copy(1, self.input_pos, x)
        if self.bias_pos.numel():
            values = values.index_copy(1, self.bias_pos, torch.ones(batch, self.bias_pos.numel(), dtype=x.dtype, device=x.device))

        masked = self._masked_weights()
        for (level_positions, level_cols, activation_groups), products, macros in zip(self._levels, self._product_entries, self._macro_entries):
            pre_activation = values @ masked[:, level_cols]
            for local_index, node_col, source_positions in products:
                edge_weights = masked.index_select(0, source_positions).index_select(1, node_col).squeeze(1)
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
        return bool(self.mask.any()) or bool(self._inert_edges)

    def export_weights(self) -> dict[tuple[int, int, bool], float]:
        """Current edge weights keyed by (in_id, out_id, recurrent), for Lamarckian writeback.
        Inert edges (no compact column) report their genome weights so `_writeback` keys stay
        complete; `RecurrentGraphNet` appends its recurrent-inert entries to the same list.
        Tied edges all report their group's shared value, so writeback keeps members in sync."""
        detached = self.weights.detach()
        exported = {(in_id, out_id, False): float(detached[source, col]) for in_id, out_id, source, col in self._edge_positions}
        exported.update({(in_id, out_id, recurrent): weight for in_id, out_id, recurrent, weight in self._inert_edges})
        if self.tie_values is not None:
            tied = self.tie_values.detach()
            exported.update({(in_id, out_id, False): float(tied[slot]) for in_id, out_id, slot in self._tied_edge_exports})
        return exported

    def core(self) -> tuple["GraphNet | None", "torch.Tensor | None"]:
        # Only the EXACT GraphNet form is batchable (RecurrentGraphNet steps over time; product,
        # macro, and tied-weight entries change the math; all fall back to the sequential path).
        if type(self) is GraphNet and not any(self._product_entries) and not any(self._macro_entries) and self.tie_values is None:
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

    def __init__(
        self,
        genome: Genome,
        n_inputs: int,
        n_outputs: int,
        mode: str = "last",
        *,
        macro_resolver: MacroResolver | None = None,
        max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
        _reference_depth: int = 0,
        _reference_stack: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            genome,
            n_inputs,
            n_outputs,
            macro_resolver=macro_resolver,
            max_inline_depth=max_inline_depth,
            _reference_depth=_reference_depth,
            _reference_stack=_reference_stack,
        )
        if mode not in ("last", "all"):
            raise ValueError(f"unknown recurrent output mode {mode!r}; expected 'last' or 'all'")
        self.mode = mode

        recurrent_matrix = torch.zeros(self.n, self.h)
        recurrent_mask = torch.zeros(self.n, self.h, dtype=torch.bool)
        recurrent_incoming: dict[int, list[int]] = defaultdict(list)
        self._recurrent_edge_positions: list[tuple[int, int, int, int]] = []
        for conn in genome.recurrent_connections():
            source, target = self._position[conn.in_id], self._position[conn.out_id]
            col = self._col_of.get(target)
            if col is None:  # recurrent edge into an input/bias node: inert (overwritten each step)
                self._inert_edges.append((conn.in_id, conn.out_id, True, conn.weight))
                continue
            recurrent_matrix[source, col] = conn.weight
            recurrent_mask[source, col] = True
            recurrent_incoming[target].append(source)
            self._recurrent_edge_positions.append((conn.in_id, conn.out_id, source, col))
        self.recurrent_weights = nn.Parameter(recurrent_matrix)
        self.recurrent_mask: torch.Tensor = recurrent_mask

        # Product nodes here must fold the recurrent term in as extra FACTORS (not a summed bias), so
        # rebuild the per-level product entries with both forward and recurrent source positions.
        forward_incoming: dict[int, list[int]] = defaultdict(list)
        for conn in genome.forward_connections():
            forward_incoming[self._position[conn.out_id]].append(self._position[conn.in_id])
        aggregation_of = {self._position[node.id]: node.aggregation for node in genome.nodes.values()}
        self._recurrent_products: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        for level_positions, _level_cols, _activation_groups in self._levels:
            entries = []
            for local_index, node_position in enumerate(level_positions.tolist()):
                if aggregation_of[node_position] != "product":
                    continue
                if not forward_incoming.get(node_position) and not recurrent_incoming.get(node_position):
                    continue
                entries.append(
                    (
                        torch.tensor([local_index], dtype=torch.long),
                        torch.tensor([self._col_of[node_position]], dtype=torch.long),
                        torch.tensor(sorted(forward_incoming.get(node_position, [])), dtype=torch.long),
                        torch.tensor(sorted(recurrent_incoming.get(node_position, [])), dtype=torch.long),
                    )
                )
            self._recurrent_products.append(entries)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"RecurrentGraphNet expects [batch, time, features], got shape {tuple(x.shape)}")
        batch, steps, _features = x.shape
        masked = self._masked_weights()  # tied forward edges share parameters here too
        recurrent_masked = self.recurrent_weights * self.recurrent_mask
        previous = torch.zeros(batch, self.n, dtype=x.dtype, device=x.device)
        outputs: list[torch.Tensor] = []
        for step in range(steps):
            values = torch.zeros(batch, self.n, dtype=x.dtype, device=x.device)
            if self.input_pos.numel():
                values = values.index_copy(1, self.input_pos, x[:, step])
            if self.bias_pos.numel():
                values = values.index_copy(1, self.bias_pos, torch.ones(batch, self.bias_pos.numel(), dtype=x.dtype, device=x.device))
            recurrent_in = previous @ recurrent_masked  # [batch, h]: one column per computed node
            for (level_positions, level_cols, activation_groups), products, macros in zip(self._levels, self._recurrent_products, self._macro_entries):
                pre_activation = values @ masked[:, level_cols] + recurrent_in.index_select(1, level_cols)
                for local_index, node_col, forward_sources, recurrent_sources in products:
                    factors = []
                    if forward_sources.numel():
                        forward_weights = masked.index_select(0, forward_sources).index_select(1, node_col).squeeze(1)
                        factors.append(values.index_select(1, forward_sources) * forward_weights)
                    if recurrent_sources.numel():
                        recurrent_edge_weights = recurrent_masked.index_select(0, recurrent_sources).index_select(1, node_col).squeeze(1)
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
        return bool(self.mask.any()) or bool(self.recurrent_mask.any()) or bool(self._inert_edges)

    def export_weights(self) -> dict[tuple[int, int, bool], float]:
        exported = super().export_weights()  # includes recurrent-inert entries via _inert_edges
        detached = self.recurrent_weights.detach()
        exported.update({(in_id, out_id, True): float(detached[source, col]) for in_id, out_id, source, col in self._recurrent_edge_positions})
        return exported


class RefineGraphNet(RecurrentGraphNet):
    """Iterative-refinement decode (the TRM idea: recursion is effective depth without parameters).

    Re-applies the network to a STATIC input `steps` times, threading node state across passes via
    the genome's recurrent edges: hidden->hidden recurrent edges carry the latent reasoning `z`, and
    output->hidden recurrent edges feed the current answer back in (TRM's `y`). The readout after the
    last pass is the refined answer. `steps == 1` is exactly one feedforward pass (the previous state
    is zero, so recurrent edges are inert), which is why the adapter keeps plain `decode` at steps 1
    and the flat path stays byte-identical. Refinement only does work once the genome evolves
    recurrent edges to thread state between passes (`add_recurrent_connection`)."""

    def __init__(
        self,
        genome: Genome,
        n_inputs: int,
        n_outputs: int,
        steps: int,
        *,
        macro_resolver: MacroResolver | None = None,
        max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
        _reference_depth: int = 0,
        _reference_stack: tuple[str, ...] = (),
    ) -> None:
        super().__init__(
            genome,
            n_inputs,
            n_outputs,
            "last",
            macro_resolver=macro_resolver,
            max_inline_depth=max_inline_depth,
            _reference_depth=_reference_depth,
            _reference_stack=_reference_stack,
        )
        if steps < 1:
            raise ValueError(f"refine steps must be >= 1, got {steps}")
        self.steps = steps

    def _repeat(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"RefineGraphNet expects a static [batch, features] input, got shape {tuple(x.shape)}")
        return x.unsqueeze(1).expand(x.shape[0], self.steps, x.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self._repeat(x))  # mode="last": the final refined readout [batch, n_outputs]

    def refine_trace(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pass readouts `[batch, steps, n_outputs]` for DEEP SUPERVISION (a loss at every pass,
        the signal that makes deep recursion train well in the small-data regime)."""
        saved_mode = self.mode
        self.mode = "all"
        try:
            flat = super().forward(self._repeat(x))  # [batch, steps * n_outputs], t-major
        finally:
            self.mode = saved_mode
        return flat.reshape(x.shape[0], self.steps, self.output_pos.numel())


def decode(
    genome: Genome,
    n_inputs: int,
    n_outputs: int,
    *,
    macro_resolver: MacroResolver | None = None,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
    _reference_depth: int = 0,
    _reference_stack: tuple[str, ...] = (),
) -> GraphNet:
    """Build the torch module for a genome."""
    return GraphNet(
        genome,
        n_inputs,
        n_outputs,
        macro_resolver=macro_resolver,
        max_inline_depth=max_inline_depth,
        _reference_depth=_reference_depth,
        _reference_stack=_reference_stack,
    )


def decode_recurrent(
    genome: Genome,
    n_inputs: int,
    n_outputs: int,
    mode: str = "last",
    *,
    macro_resolver: MacroResolver | None = None,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
    _reference_depth: int = 0,
    _reference_stack: tuple[str, ...] = (),
) -> RecurrentGraphNet:
    """Build the stepped (time-axis) torch module for a genome."""
    return RecurrentGraphNet(
        genome,
        n_inputs,
        n_outputs,
        mode,
        macro_resolver=macro_resolver,
        max_inline_depth=max_inline_depth,
        _reference_depth=_reference_depth,
        _reference_stack=_reference_stack,
    )


def decode_refine(
    genome: Genome,
    n_inputs: int,
    n_outputs: int,
    *,
    steps: int | None = None,
    macro_resolver: MacroResolver | None = None,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
    _reference_depth: int = 0,
    _reference_stack: tuple[str, ...] = (),
) -> RefineGraphNet:
    """Build the iterative-refinement module for a genome (static input, re-applied `steps` times)."""
    resolved = genome.refine_steps if steps is None else steps
    return RefineGraphNet(
        genome,
        n_inputs,
        n_outputs,
        resolved,
        macro_resolver=macro_resolver,
        max_inline_depth=max_inline_depth,
        _reference_depth=_reference_depth,
        _reference_stack=_reference_stack,
    )


def decode_module(
    genome: Genome,
    n_inputs: int,
    n_outputs: int,
    *,
    macro_resolver: MacroResolver | None = None,
    max_inline_depth: int = DEFAULT_MAX_INLINE_DEPTH,
    _reference_depth: int = 0,
    _reference_stack: tuple[str, ...] = (),
) -> GraphNet:
    """Decode honoring the genome's evolved refinement depth: refine substrate when refine_steps > 1,
    plain feedforward otherwise. Use this at every STATIC-task decode site that evaluates or reuses a
    genome (adapter, library lookup re-eval, composition inner), so a module that needs its refine
    passes keeps working wherever it is reused, never silently collapsing to a single pass."""
    if genome.refine_steps > 1:
        return decode_refine(
            genome,
            n_inputs,
            n_outputs,
            macro_resolver=macro_resolver,
            max_inline_depth=max_inline_depth,
            _reference_depth=_reference_depth,
            _reference_stack=_reference_stack,
        )
    return decode(
        genome,
        n_inputs,
        n_outputs,
        macro_resolver=macro_resolver,
        max_inline_depth=max_inline_depth,
        _reference_depth=_reference_depth,
        _reference_stack=_reference_stack,
    )
