"""Gradient-proposed growth signals (NeST / GradMax idea ports): where the loss says capacity is
missing.

The single trick: in `GraphNet.forward` every weight column is consumed by exactly one level
matmul, so replaying the level loop with the masked weight matrix detached-but-requiring-grad
makes `masked.grad[i, c]` the batch-summed `activation_i x delta_c` for EVERY (source, computed)
pair, dormant pairs included: exactly NeST's dormant-connection score, one backward pass for the
whole matrix (the mask normally zeroes those gradients before they exist). The rank-1 marginals
(per-source activation mass, per-target delta mass) ride the genome as `growth_hints`, and the
hinted mutation operators sample structure proportional to them: gradient proposes, evolution
disposes."""

import torch

from versal.dataset.icarus import EncodedTask, as_logits, loss_fn, target_positions
from versal.evolution.genome import Genome
from versal.substrate import GraphNet, SubstrateModule


def node_scores(module: SubstrateModule, encoded: EncodedTask) -> tuple[dict[int, float], dict[int, float]] | None:
    """(source_scores, target_scores) keyed by node id, or None for non-plain substrates.

    Only the exact `GraphNet` form qualifies (`core()`'s own criterion): recurrent, refine,
    product, and macro substrates change the math the replay below mirrors."""
    if type(module) is not GraphNet:
        return None
    net, _columns = module.core()
    if net is None:
        return None
    x, _input_descriptor = encoded.support_input
    target, mask, descriptor = encoded.support_target

    masked = (net.weights * net.mask).detach().requires_grad_(True)
    batch = x.shape[0]
    values = torch.zeros(batch, net.n, dtype=x.dtype)
    if net.input_pos.numel():
        values = values.index_copy(1, net.input_pos, x)
    if net.bias_pos.numel():
        values = values.index_copy(1, net.bias_pos, torch.ones(batch, net.bias_pos.numel(), dtype=x.dtype))
    for level_positions, level_cols, activation_groups in net._levels:
        activated = values @ masked[:, level_cols]
        for activation, local_indices in activation_groups:
            activated = activated.index_copy(1, local_indices, activation(activated.index_select(1, local_indices)))
        values = values.index_copy(1, level_positions, activated)
    out = values.index_select(1, net.output_pos)

    raw = as_logits(out, descriptor, target_positions(target))
    loss = loss_fn(raw, target, descriptor, mask)
    if not loss.requires_grad or not torch.isfinite(loss):
        return None
    loss.backward()
    if masked.grad is None:
        return None

    magnitude = masked.grad.abs()
    node_of_position = {position: node_id for node_id, position in net._position.items()}
    source_scores = {node_of_position[row]: float(value) for row, value in enumerate(magnitude.sum(dim=1).tolist())}
    target_scores = {node_of_position[int(position)]: float(value) for position, value in zip(net.col_index.tolist(), magnitude.sum(dim=0).tolist())}
    return source_scores, target_scores


def attach_growth_hints(genome: Genome, module: SubstrateModule, encoded: EncodedTask, *, cloned: bool) -> Genome:
    """Stamp `growth_hints` onto a (copy of the) genome after training. In-memory only: hints
    never serialize, so admitted payloads and checkpoints are unchanged even with scoring on."""
    scores = node_scores(module, encoded)
    if scores is None:
        return genome
    if not cloned:
        genome = genome.clone()
    genome.growth_hints = {"source": scores[0], "target": scores[1]}
    return genome
