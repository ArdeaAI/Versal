"""Train operators: an independent stage that optimizes a decoded candidate's weights.

Runs between mutation and evaluation. `none` leaves the co-evolved weights alone (phase-1
default). `gradient` backprops on the support set for `steps` using the differentiable Icarus
`loss_fn`. `writeback` controls Lamarckian (tuned weights copied into the genome) vs Baldwinian
(tuned only for this scoring) behavior.

`TRAIN_POPULATION` ops train a WHOLE generation in one tensor program (`gradient_batched`); the
Evolver routes through them via `assess_many` when configured. CONTRACT: train ops must not draw
from the shared `rng` in a call-order-dependent way, because batched assessment defers and reorders
candidate training relative to the sequential path.

Per-candidate gradient independence in the batched op: the folded loss is multiplied by P so it
equals the SUM of per-candidate mean losses; gradients never cross the population axis and Adam is
elementwise, so one optimizer over the stack updates each candidate exactly as its own optimizer
would. Padded and non-edge entries start at zero with zero gradient and stay zero.
"""

import random
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

import torch

from ardevo.dataset.icarus import EncodedTask, as_logits, loss_fn, target_positions
from ardevo.evaluation import support_loss, support_loss_deep
from ardevo.evolution.genome import Genome
from ardevo.evolution.registry import Registry
from ardevo.substrate import SubstrateModule

if TYPE_CHECKING:
    from ardevo.substrate import GraphNet

TrainOp = Callable[..., tuple[Genome, SubstrateModule]]
PopulationTrainOp = Callable[..., list[tuple[Genome, SubstrateModule]]]

TRAIN: Registry[TrainOp] = Registry("train")
TRAIN_POPULATION: Registry[PopulationTrainOp] = Registry("train_population")

# Timing/shape record of the most recent batched call, mirrored by the Evolver for trial logging.
last_batch_stats: dict[str, float] = {}


def _trainable_parameters(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


@TRAIN.register("none")
def no_train(genome: Genome, module: SubstrateModule, encoded: EncodedTask, *, rng: random.Random, **_params: object) -> tuple[Genome, SubstrateModule]:
    return genome, module


@TRAIN.register("gradient")
def gradient(
    genome: Genome,
    module: SubstrateModule,
    encoded: EncodedTask,
    *,
    rng: random.Random,
    steps: int = 20,
    lr: float = 0.01,
    writeback: bool = True,
    weight_decay: float = 0.0,
) -> tuple[Genome, SubstrateModule]:
    # weight_decay (L2) regularizes the fit: it shrinks weights, which narrows the support->query
    # generalization gap on tasks that can generalize (and is harmless when set to 0).
    parameters = _trainable_parameters(module)
    if steps <= 0 or not module.has_edges or not parameters:
        return genome, module
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = support_loss(module, encoded)
        # Stop on no-grad (frozen params) OR a non-finite loss: Adam does not filter NaN/Inf grads,
        # so stepping on them would silently corrupt every weight in the candidate.
        if not loss.requires_grad or not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
    if writeback:
        genome = _writeback(genome, module)
    return genome, module


@TRAIN.register("gradient_refine")
def gradient_refine(
    genome: Genome,
    module: SubstrateModule,
    encoded: EncodedTask,
    *,
    rng: random.Random,
    steps: int = 20,
    lr: float = 0.01,
    writeback: bool = True,
    weight_decay: float = 0.0,
) -> tuple[Genome, SubstrateModule]:
    """Deep-supervised gradient training for the refine substrate (TRM): backprop a loss summed over
    every refinement pass, through the full recursion. Falls back to plain `gradient` for modules
    that do not refine (steps==1 genomes decode to a GraphNet), so it is a safe drop-in trainer."""
    if not hasattr(module, "refine_trace"):
        return gradient(genome, module, encoded, rng=rng, steps=steps, lr=lr, writeback=writeback, weight_decay=weight_decay)
    parameters = _trainable_parameters(module)
    if steps <= 0 or not module.has_edges or not parameters:
        return genome, module
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = support_loss_deep(module, encoded)
        if not loss.requires_grad or not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
    if writeback:
        genome = _writeback(genome, module)
    return genome, module


def partition_batchable(
    cores: "list[tuple[GraphNet | None, torch.Tensor | None]]",
    *,
    steps: int,
    max_padded_nodes: int,
    min_batch_nodes: int = 0,
) -> tuple[list[int], list[int]]:
    """Split candidate indices into (batchable, serial). The single partition seam shared by the
    population trainers and the Evolver's hybrid assess router, so the two can never disagree
    about which candidates the batch program will take. Batchable = a plain GraphNet core within
    the padding budget whose I/O signature + head columns match the FIRST batchable candidate.
    `min_batch_nodes` is the WIDTH FLOOR: below it the pool wins (measured: 12 workers beat the
    MPS batch program at grown-from-minimal sizes), so small candidates go serial and the device
    engages only where it measured a win. 0 = no floor (the pre-knob behavior)."""
    batch_indices: list[int] = []
    serial_indices: list[int] = []
    reference: tuple[tuple[int, int], torch.Tensor | None] | None = None
    for index, (net, columns) in enumerate(cores):
        if net is None or steps <= 0 or net.n > max_padded_nodes or net.n < min_batch_nodes:
            serial_indices.append(index)
            continue
        signature = (int(net.input_pos.numel()), int(net.output_pos.numel()))
        if reference is None:
            reference = (signature, columns)
        ref_signature, ref_columns = reference
        same_columns = (columns is None and ref_columns is None) or (columns is not None and ref_columns is not None and bool(torch.equal(columns, ref_columns)))
        if signature == ref_signature and same_columns:
            batch_indices.append(index)
        else:
            serial_indices.append(index)
    if len(batch_indices) < 2:  # padding a single candidate buys nothing over the plain path
        serial_indices = sorted(serial_indices + batch_indices)
        batch_indices = []
    return batch_indices, serial_indices


@TRAIN_POPULATION.register("gradient_batched")
def gradient_batched(
    genomes: list[Genome],
    modules: list[SubstrateModule],
    encoded: EncodedTask,
    *,
    rng: random.Random,
    steps: int = 20,
    lr: float = 0.01,
    writeback: bool = True,
    weight_decay: float = 0.0,
    device: str = "auto",
    max_padded_nodes: int = 1024,
    min_batch_nodes: int = 0,
    adam_fused: bool = False,
    torch_compile: bool = False,
) -> list[tuple[Genome, SubstrateModule]]:
    """Train every BATCHABLE candidate in one tensor program and the rest sequentially (identical
    params, identical numerics), stitched back in input order. Non-batchable candidates (recurrent/
    product/macro/composed substrates, oversized padding, mismatched heads) no longer force the
    WHOLE generation onto the sequential path; `fallback` reports the serial fraction."""
    return _gradient_batched_impl(
        genomes,
        modules,
        encoded,
        rng=rng,
        steps=steps,
        lr=lr,
        writeback=writeback,
        weight_decay=weight_decay,
        device=device,
        max_padded_nodes=max_padded_nodes,
        min_batch_nodes=min_batch_nodes,
        adam_fused=adam_fused,
        torch_compile=torch_compile,
        serial_op=gradient,
    )


@TRAIN_POPULATION.register("gradient_refine")
def gradient_refine_population(
    genomes: list[Genome],
    modules: list[SubstrateModule],
    encoded: EncodedTask,
    *,
    rng: random.Random,
    steps: int = 20,
    lr: float = 0.01,
    writeback: bool = True,
    weight_decay: float = 0.0,
    device: str = "auto",
    max_padded_nodes: int = 1024,
    min_batch_nodes: int = 0,
    adam_fused: bool = False,
    torch_compile: bool = False,
) -> list[tuple[Genome, SubstrateModule]]:
    """Population form of `gradient_refine`. Correct by exclusion: `core()` keeps every refine/
    recurrent/product/macro module OUT of the batch (they train through sequential
    `gradient_refine`, deep supervision intact), and for the plain-GraphNet remainder
    `gradient_refine` IS plain `gradient` (no `refine_trace`), which is exactly the batch program's
    math. Registered under the same name so `[orchestrator.direct.train] kind = "gradient_refine"`
    batches with zero config edits; `batched = false` on the table restores the pool-only path."""
    return _gradient_batched_impl(
        genomes,
        modules,
        encoded,
        rng=rng,
        steps=steps,
        lr=lr,
        writeback=writeback,
        weight_decay=weight_decay,
        device=device,
        max_padded_nodes=max_padded_nodes,
        min_batch_nodes=min_batch_nodes,
        adam_fused=adam_fused,
        torch_compile=torch_compile,
        serial_op=gradient_refine,
    )


def _gradient_batched_impl(
    genomes: list[Genome],
    modules: list[SubstrateModule],
    encoded: EncodedTask,
    *,
    rng: random.Random,
    steps: int,
    lr: float,
    writeback: bool,
    weight_decay: float,
    device: str,
    max_padded_nodes: int,
    serial_op: TrainOp,
    min_batch_nodes: int = 0,
    adam_fused: bool = False,
    torch_compile: bool = False,
) -> list[tuple[Genome, SubstrateModule]]:
    from ardevo.substrate_batched import BatchedGraphNet

    started = time.perf_counter()
    cores = [module.core() for module in modules]
    batch_indices, serial_indices = partition_batchable(cores, steps=steps, max_padded_nodes=max_padded_nodes, min_batch_nodes=min_batch_nodes)

    results: list[tuple[Genome, SubstrateModule] | None] = [None] * len(modules)
    for index in serial_indices:
        results[index] = serial_op(genomes[index], modules[index], encoded, rng=rng, steps=steps, lr=lr, writeback=writeback, weight_decay=weight_decay)

    n_max = 0.0
    pad_efficiency = 0.0
    if batch_indices:
        from ardevo.utils.device import auto_device

        nets = [net for index in batch_indices if (net := cores[index][0]) is not None]
        resolved = auto_device() if device == "auto" else torch.device(device)
        batched = BatchedGraphNet(nets, device=resolved)
        x, _descriptor = encoded.support_input
        target, mask, descriptor = encoded.support_target
        x_device = x.to(resolved)
        target_device, mask_device = target.to(resolved), (mask.to(resolved) if mask is not None else None)
        ref_head = cores[batch_indices[0]][1]
        columns = ref_head.to(resolved) if ref_head is not None else None
        population = len(nets)

        if batched.mask.any():
            # Both knobs default OFF: fused Adam is cuda-only and not bit-equal to the unfused
            # step; torch.compile recompiles as the population shape churns generation to
            # generation, so it must prove itself on `uv run benchmark` before use.
            fused = bool(adam_fused and resolved.type == "cuda")
            optimizer = torch.optim.Adam(batched.parameters(), lr=lr, weight_decay=weight_decay, fused=fused)
            program = torch.compile(batched, dynamic=True) if torch_compile else batched
            for _ in range(steps):
                optimizer.zero_grad()
                out = program(x_device)  # [P, B, n_out]
                if columns is not None:
                    out = out.index_select(2, columns)
                folded = out.reshape(population * x_device.shape[0], -1)
                raw = as_logits(folded, descriptor, target_positions(target_device))
                target_repeated = target_device.repeat(population, *([1] * (target_device.dim() - 1)))
                mask_repeated = mask_device.repeat(population, *([1] * (mask_device.dim() - 1))) if mask_device is not None else None
                # The P multiplier turns the folded MEAN into the SUM of per-candidate losses: each
                # candidate's gradient (and Adam update) is exactly what the sequential path computes.
                loss = population * loss_fn(raw, target_repeated, descriptor, mask_repeated)
                if not loss.requires_grad or not torch.isfinite(loss):
                    break
                loss.backward()
                optimizer.step()
            batched.unstack_into(nets)
        n_max = float(batched.n_max)
        pad_efficiency = batched.pad_efficiency()
        for index in batch_indices:
            tuned = _writeback(genomes[index], modules[index]) if writeback else genomes[index]
            results[index] = (tuned, modules[index])

    last_batch_stats.update(
        {
            "fallback": len(serial_indices) / max(len(modules), 1),
            "train_seconds": time.perf_counter() - started,
            "n_max": n_max,
            "pad_efficiency": pad_efficiency,
        }
    )
    return [item for item in results if item is not None]


def _writeback(genome: Genome, module: SubstrateModule) -> Genome:
    """Copy the module's tuned weights back onto the matching enabled connection genes."""
    tuned = module.export_weights()
    child = genome.clone()
    child.connections = [
        replace(conn, weight=tuned[(conn.in_id, conn.out_id, conn.recurrent)]) if conn.enabled and (conn.in_id, conn.out_id, conn.recurrent) in tuned else conn
        for conn in child.connections
    ]
    return child
