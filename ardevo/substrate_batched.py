"""Population-batched substrate: train EVERY candidate of a generation in one tensor program.

Inspired by tensorneat's padded-tensor genome encoding, built natively in torch (MPS friendly: all
ops are float32 bmm/scatter/gather/where). Per-genome `GraphNet`s are padded to the generation's max
node count and stacked into `[P, n, n]` weight/mask tensors; a UNIFIED level schedule over the max
depth runs one bmm per level, with per-node level masks discarding the columns a given candidate
does not compute at that level and per-activation masks applying nonlinearities population-wide.

Correctness: longest-path leveling guarantees every edge into a level-l node reads values already
written at levels < l; columns outside a candidate's level mask hold either zeros (never computed,
feeding nothing at this level) or already-final values, and the full-width bmm results outside the
mask are discarded by `torch.where`. So the unified schedule computes EXACTLY what each per-genome
forward computes, just stacked.

This changes wall-clock, never search semantics: `unstack_into` copies trained weights back into the
per-genome nets, so evaluation, writeback, robustness sampling, and artifact saving all run on real
per-candidate modules, unchanged.
"""

import torch
from torch import nn

from ardevo.substrate import _ACTIVATIONS, GraphNet


class BatchedGraphNet(nn.Module):
    """A whole generation of `GraphNet`s as one padded, masked tensor program."""

    def __init__(self, nets: list[GraphNet], device: torch.device | None = None) -> None:
        super().__init__()
        if not nets:
            raise ValueError("BatchedGraphNet needs at least one net")
        input_counts = {int(net.input_pos.numel()) for net in nets}
        output_counts = {int(net.output_pos.numel()) for net in nets}
        if len(input_counts) != 1 or len(output_counts) != 1:
            raise ValueError(f"all nets in one batch must share I/O widths, got inputs {input_counts} outputs {output_counts}")
        self.device = device or torch.device("cpu")
        self.population = len(nets)
        self.n_max = max(net.n for net in nets)
        self.depth = max(len(net._levels) for net in nets)
        self._sizes = [net.n for net in nets]

        weights = torch.zeros(self.population, self.n_max, self.n_max)
        mask = torch.zeros(self.population, self.n_max, self.n_max, dtype=torch.bool)
        level_update = torch.zeros(self.depth, self.population, self.n_max, dtype=torch.bool)
        activation_masks: dict[str, torch.Tensor] = {}
        for index, net in enumerate(nets):
            weights[index, : net.n, : net.n] = net.weights.detach()
            mask[index, : net.n, : net.n] = net.mask
            for node_position, level in net.level_of.items():
                level_update[level, index, node_position] = True
            for node_position, name in net.activation_name_of.items():
                if name not in activation_masks:
                    activation_masks[name] = torch.zeros(self.depth, self.population, self.n_max, dtype=torch.bool)
                activation_masks[name][net.level_of[node_position], index, node_position] = True

        self.weights = nn.Parameter(weights.to(self.device))
        self.mask: torch.Tensor = mask.to(self.device)
        self.level_update: torch.Tensor = level_update.to(self.device)
        self.activation_masks: dict[str, torch.Tensor] = {name: tensor.to(self.device) for name, tensor in activation_masks.items()}
        self.input_pos: torch.Tensor = torch.stack([net.input_pos for net in nets]).to(self.device)
        self.bias_pos: torch.Tensor = torch.stack([net.bias_pos for net in nets]).to(self.device)
        self.output_pos: torch.Tensor = torch.stack([net.output_pos for net in nets]).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is the SHARED `[batch, n_inputs]` task input; returns `[P, batch, n_outputs]`."""
        batch = x.shape[0]
        values = torch.zeros(self.population, batch, self.n_max, dtype=x.dtype, device=x.device)
        if self.input_pos.shape[1]:
            index = self.input_pos[:, None, :].expand(self.population, batch, self.input_pos.shape[1])
            values = values.scatter(2, index, x.unsqueeze(0).expand(self.population, batch, x.shape[1]))
        if self.bias_pos.shape[1]:
            index = self.bias_pos[:, None, :].expand(self.population, batch, self.bias_pos.shape[1])
            values = values.scatter(2, index, torch.ones(self.population, batch, self.bias_pos.shape[1], dtype=x.dtype, device=x.device))

        masked = self.weights * self.mask
        for level in range(self.depth):
            pre_activation = torch.bmm(values, masked)
            activated = pre_activation
            for name, masks in self.activation_masks.items():
                activated = torch.where(masks[level][:, None, :], _ACTIVATIONS[name](pre_activation), activated)
            values = torch.where(self.level_update[level][:, None, :], activated, values)
        return values.gather(2, self.output_pos[:, None, :].expand(self.population, batch, self.output_pos.shape[1]))

    def unstack_into(self, nets: list[GraphNet]) -> None:
        """Copy the trained per-candidate weight slices back into the per-genome nets (on CPU)."""
        trained = self.weights.detach().cpu()
        for index, net in enumerate(nets):
            net.weights.data.copy_(trained[index, : net.n, : net.n])

    def pad_efficiency(self) -> float:
        """Fraction of the padded tensor budget the real graphs actually use (a logging signal)."""
        used = sum(size * size for size in self._sizes)
        return used / (self.population * self.n_max * self.n_max)
