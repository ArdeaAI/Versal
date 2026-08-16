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

import math
import random
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

import torch

from versal.dataset.icarus import EncodedTask, as_logits, loss_fn, target_positions
from versal.evaluation import support_loss, support_loss_deep
from versal.evolution.genome import Genome
from versal.evolution.registry import Registry
from versal.substrate import SubstrateModule
from versal.utils.deadline import expired

if TYPE_CHECKING:
    from versal.substrate import GraphNet

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
    score_candidates: bool = False,
    deadline: float | None = None,
) -> tuple[Genome, SubstrateModule]:
    # weight_decay (L2) regularizes the fit: it shrinks weights, which narrows the support->query
    # generalization gap on tasks that can generalize (and is harmless when set to 0).
    parameters = _trainable_parameters(module)
    if steps <= 0 or not module.has_edges or not parameters:
        return genome, module
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        if expired(deadline):
            break
        optimizer.zero_grad()
        loss = support_loss(module, encoded)
        # Stop on no-grad (frozen params) OR a non-finite loss: Adam does not filter NaN/Inf grads,
        # so stepping on them would silently corrupt every weight in the candidate.
        if expired(deadline) or not loss.requires_grad or not torch.isfinite(loss):
            break
        loss.backward()
        if expired(deadline):
            break
        optimizer.step()
    if writeback:
        genome = _writeback(genome, module)
    if score_candidates:  # NeST/GradMax growth hints from the trained weights (one extra backward)
        from versal.evolution.growth import attach_growth_hints

        genome = attach_growth_hints(genome, module, encoded, cloned=writeback)
    return genome, module


def _scheduled_learning_rates(steps: int, lr: float, *, warmup_fraction: float = 0.05, final_lr_fraction: float = 0.05) -> list[float]:
    """Linear warmup to `lr`, then cosine decay to `lr * final_lr_fraction`."""
    warmup_steps = max(1, int(steps * warmup_fraction))
    floor = lr * final_lr_fraction
    rates: list[float] = []
    for step in range(steps):
        if step < warmup_steps:
            rates.append(lr * (step + 1) / warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, steps - warmup_steps)
            rates.append(floor + (lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return rates


@TRAIN.register("gradient_scheduled")
def gradient_scheduled(
    genome: Genome,
    module: SubstrateModule,
    encoded: EncodedTask,
    *,
    rng: random.Random,
    steps: int = 200,
    lr: float = 0.01,
    warmup_fraction: float = 0.05,
    final_lr_fraction: float = 0.05,
    writeback: bool = True,
    weight_decay: float = 0.0,
    score_candidates: bool = False,
    deadline: float | None = None,
) -> tuple[Genome, SubstrateModule]:
    # Warmup + cosine decay: fixed-lr Adam stalls in oscillatory loss regions that a decaying rate
    # anneals through. Measured on the CPPN-generator landscape (ai/trial/probe_6): the same
    # topology goes 0.55 (fixed lr) -> 0.92+ (scheduled) query, which made trainability, not
    # search, gate E's binding constraint. rng-free like every train op (the batching contract).
    parameters = _trainable_parameters(module)
    if steps <= 0 or not module.has_edges or not parameters:
        return genome, module
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    for step_lr in _scheduled_learning_rates(steps, lr, warmup_fraction=warmup_fraction, final_lr_fraction=final_lr_fraction):
        if expired(deadline):
            break
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        optimizer.zero_grad()
        loss = support_loss(module, encoded)
        if expired(deadline) or not loss.requires_grad or not torch.isfinite(loss):
            break
        loss.backward()
        if expired(deadline):
            break
        optimizer.step()
    if writeback:
        genome = _writeback(genome, module)
    if score_candidates:
        from versal.evolution.growth import attach_growth_hints

        genome = attach_growth_hints(genome, module, encoded, cloned=writeback)
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
    score_candidates: bool = False,
    contractivity_weight: float = 0.0,
    deadline: float | None = None,
) -> tuple[Genome, SubstrateModule]:
    """Deep-supervised gradient training for the refine substrate (TRM): backprop a loss summed over
    every refinement pass, through the full recursion. Falls back to plain `gradient` for modules
    that do not refine (steps==1 genomes decode to a GraphNet), so it is a safe drop-in trainer.

    `contractivity_weight` > 0 adds a Lipschitz-style penalty on the recurrent weight block
    (relu(frobenius - 1), the cheap upper bound on the spectral norm), keeping the iterated update
    map contractive: the DT-L finding that makes deep unrolls train stably and gives fixed-point
    convergence a meaning. 0.0 (the default) is off, byte-identical."""
    if not hasattr(module, "refine_trace"):
        return gradient(
            genome, module, encoded, rng=rng, steps=steps, lr=lr, writeback=writeback, weight_decay=weight_decay, score_candidates=score_candidates, deadline=deadline
        )
    parameters = _trainable_parameters(module)
    if steps <= 0 or not module.has_edges or not parameters:
        return genome, module
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        if expired(deadline):
            break
        optimizer.zero_grad()
        loss = support_loss_deep(module, encoded)
        if contractivity_weight > 0.0 and hasattr(module, "recurrent_weights"):
            from typing import cast

            from versal.substrate import RecurrentGraphNet

            recurrent_net = cast(RecurrentGraphNet, module)
            recurrent_masked = recurrent_net.recurrent_weights * recurrent_net.recurrent_mask
            loss = loss + contractivity_weight * torch.relu(recurrent_masked.norm() - 1.0)
        if expired(deadline) or not loss.requires_grad or not torch.isfinite(loss):
            break
        loss.backward()
        if expired(deadline):
            break
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
    score_candidates: bool = False,
    microbatch_size: int = 0,
    adaptive_microbatch: bool = True,
    deadline: float | None = None,
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
        score_candidates=score_candidates,
        microbatch_size=microbatch_size,
        adaptive_microbatch=adaptive_microbatch,
        serial_op=gradient,
        deadline=deadline,
    )


@TRAIN_POPULATION.register("gradient_scheduled")
def gradient_scheduled_population(
    genomes: list[Genome],
    modules: list[SubstrateModule],
    encoded: EncodedTask,
    *,
    rng: random.Random,
    steps: int = 200,
    lr: float = 0.01,
    warmup_fraction: float = 0.05,
    final_lr_fraction: float = 0.05,
    writeback: bool = True,
    weight_decay: float = 0.0,
    device: str = "auto",
    max_padded_nodes: int = 1024,
    min_batch_nodes: int = 0,
    adam_fused: bool = False,
    torch_compile: bool = False,
    score_candidates: bool = False,
    microbatch_size: int = 0,
    adaptive_microbatch: bool = True,
    deadline: float | None = None,
) -> list[tuple[Genome, SubstrateModule]]:
    """Population form of `gradient_scheduled`, with the identical warmup/cosine rates.

    Non-batchable and OOM-fallback candidates call the original sequential operator with the same
    schedule parameters. Existing `gradient_scheduled` configs remain serial unless they opt in
    with `batched = true` or a calibrated compute policy selects a population mode.
    """
    rates = _scheduled_learning_rates(steps, lr, warmup_fraction=warmup_fraction, final_lr_fraction=final_lr_fraction)
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
        score_candidates=score_candidates,
        microbatch_size=microbatch_size,
        adaptive_microbatch=adaptive_microbatch,
        step_learning_rates=rates,
        serial_op=gradient_scheduled,
        serial_params={"warmup_fraction": warmup_fraction, "final_lr_fraction": final_lr_fraction},
        deadline=deadline,
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
    score_candidates: bool = False,
    microbatch_size: int = 0,
    adaptive_microbatch: bool = True,
    deadline: float | None = None,
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
        score_candidates=score_candidates,
        microbatch_size=microbatch_size,
        adaptive_microbatch=adaptive_microbatch,
        serial_op=gradient_refine,
        deadline=deadline,
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
    score_candidates: bool = False,
    microbatch_size: int = 0,
    adaptive_microbatch: bool = True,
    step_learning_rates: list[float] | None = None,
    serial_params: dict[str, object] | None = None,
    deadline: float | None = None,
) -> list[tuple[Genome, SubstrateModule]]:
    from versal.substrate_batched import BatchedGraphNet

    if microbatch_size < 0:
        raise ValueError("microbatch_size must be >= 0")
    if step_learning_rates is not None and len(step_learning_rates) != max(steps, 0):
        raise ValueError("step_learning_rates must contain exactly one rate per training step")

    started = time.perf_counter()
    cores = [module.core() for module in modules]
    batch_indices, serial_indices = partition_batchable(cores, steps=steps, max_padded_nodes=max_padded_nodes, min_batch_nodes=min_batch_nodes)

    results: list[tuple[Genome, SubstrateModule] | None] = [None] * len(modules)
    serial_extras = serial_params or {}

    def train_serial(index: int) -> None:
        results[index] = serial_op(
            genomes[index],
            modules[index],
            encoded,
            rng=rng,
            steps=steps,
            lr=lr,
            writeback=writeback,
            weight_decay=weight_decay,
            score_candidates=score_candidates,
            deadline=deadline,
            **serial_extras,
        )

    for index in serial_indices:
        train_serial(index)

    n_max = 0.0
    weighted_pad_efficiency = 0.0
    trained_in_batches = 0
    microbatch_count = 0
    largest_microbatch = 0
    oom_retries = 0
    oom_fallbacks = 0
    if batch_indices:
        from versal.utils.device import auto_device, clear_device_cache, is_out_of_memory_error

        resolved = auto_device() if device == "auto" else torch.device(device)
        x, _descriptor = encoded.support_input
        target, mask, descriptor = encoded.support_target
        try:
            x_device = x.to(resolved)
            target_device, mask_device = target.to(resolved), (mask.to(resolved) if mask is not None else None)
            ref_head = cores[batch_indices[0]][1]
            columns = ref_head.to(resolved) if ref_head is not None else None
        except RuntimeError as exc:
            if not adaptive_microbatch or not is_out_of_memory_error(exc):
                raise
            # Inputs are shared by every population slice, so splitting cannot reduce this part of
            # the footprint. Preserve the method by running the affected candidates serially on CPU.
            oom_retries += 1
            exc.__traceback__ = None
            clear_device_cache(resolved)
            for index in batch_indices:
                train_serial(index)
            oom_fallbacks += len(batch_indices)
            batch_indices = []

        def train_group(indices: list[int]) -> tuple[float, float]:
            nets = [net for index in indices if (net := cores[index][0]) is not None]
            batched = BatchedGraphNet(nets, device=resolved)
            population = len(nets)
            if batched.mask.any():
                # Both knobs default OFF: fused Adam is cuda-only and not bit-equal to the unfused
                # step; torch.compile recompiles as the population shape churns generation to
                # generation, so it must prove itself on `uv run benchmark` before use.
                fused = bool(adam_fused and resolved.type == "cuda")
                optimizer = torch.optim.Adam(batched.parameters(), lr=lr, weight_decay=weight_decay, fused=fused)
                program = torch.compile(batched, dynamic=True) if torch_compile else batched
                target_repeated = target_device.repeat(population, *([1] * (target_device.dim() - 1)))
                mask_repeated = mask_device.repeat(population, *([1] * (mask_device.dim() - 1))) if mask_device is not None else None
                for step in range(steps):
                    if expired(deadline):
                        break
                    if step_learning_rates is not None:
                        for group in optimizer.param_groups:
                            group["lr"] = step_learning_rates[step]
                    optimizer.zero_grad()
                    out = program(x_device)  # [P, B, n_out]
                    if columns is not None:
                        out = out.index_select(2, columns)
                    folded = out.reshape(population * x_device.shape[0], -1)
                    raw = as_logits(folded, descriptor, target_positions(target_device))
                    # The P multiplier turns the folded MEAN into the SUM of per-candidate losses:
                    # each candidate's gradient and Adam update match its sequential computation.
                    loss = population * loss_fn(raw, target_repeated, descriptor, mask_repeated)
                    if expired(deadline) or not loss.requires_grad or not torch.isfinite(loss):
                        break
                    loss.backward()
                    if expired(deadline):
                        break
                    optimizer.step()
                batched.unstack_into(nets)
            return float(batched.n_max), batched.pad_efficiency()

        size = microbatch_size or max(len(batch_indices), 1)
        pending = [batch_indices[start : start + size] for start in range(0, len(batch_indices), size)]
        while pending:
            indices = pending.pop(0)
            try:
                group_n_max, group_pad_efficiency = train_group(indices)
            except RuntimeError as exc:
                if not adaptive_microbatch or not is_out_of_memory_error(exc):
                    raise
                oom_retries += 1
                # Drop the traceback's references to the failed tensor program before asking the
                # allocator to release cached blocks; otherwise the retry can inherit its peak.
                exc.__traceback__ = None
                clear_device_cache(resolved)
                if len(indices) > 1:
                    midpoint = len(indices) // 2
                    pending[0:0] = [indices[:midpoint], indices[midpoint:]]
                    continue
                train_serial(indices[0])
                oom_fallbacks += 1
                continue

            microbatch_count += 1
            largest_microbatch = max(largest_microbatch, len(indices))
            n_max = max(n_max, group_n_max)
            weighted_pad_efficiency += group_pad_efficiency * len(indices)
            trained_in_batches += len(indices)
            for index in indices:
                tuned = _writeback(genomes[index], modules[index]) if writeback else genomes[index]
                if score_candidates:
                    from versal.evolution.growth import attach_growth_hints

                    tuned = attach_growth_hints(tuned, modules[index], encoded, cloned=writeback)
                results[index] = (tuned, modules[index])

    last_batch_stats.update(
        {
            "fallback": (len(serial_indices) + oom_fallbacks) / max(len(modules), 1),
            "train_seconds": time.perf_counter() - started,
            "n_max": n_max,
            "pad_efficiency": weighted_pad_efficiency / max(trained_in_batches, 1),
            "microbatches": float(microbatch_count),
            "largest_microbatch": float(largest_microbatch),
            "oom_retries": float(oom_retries),
            "oom_fallbacks": float(oom_fallbacks),
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
