"""Phase-1 init-wall levers: `factored` / `sparse` init ops (scale-safe seeds for wide I/O),
`prune_and_regrow` (SET-style constant-density rewiring), and the orchestrator's decompose-first
policy. The contract everywhere: below threshold / knob off is byte-identical to the old behavior."""

import random
from pathlib import Path

import torch

from tests.test_orchestrator import _fake_run_task, _orchestrator, _patch_run_task
from versal.dataset.icarus import Task
from versal.evolution.genome import InnovationTracker, NodeKind, genome_to_dict, topological_order
from versal.evolution.init import factored, minimal, sparse
from versal.evolution.mutation import MutationContext, prune_and_regrow
from versal.substrate import decode


def _ctx(*genomes) -> MutationContext:
    return MutationContext(innovations=InnovationTracker.from_genomes(list(genomes)), activations=["tanh"], default_activation="tanh")


# --- factored ---------------------------------------------------------------------------------------


def test_factored_below_threshold_is_minimal_byte_identical() -> None:
    seeded = factored(4, 2, rng=random.Random(3), rank=8, threshold=4096)
    baseline = minimal(4, 2, rng=random.Random(3))
    assert genome_to_dict(seeded) == genome_to_dict(baseline)


def test_factored_builds_the_rank_bottleneck_above_threshold() -> None:
    genome = factored(100, 10, rng=random.Random(0), rank=4, threshold=64)
    assert len(genome.hidden_ids) == 4
    assert all(genome.nodes[node_id].activation == "identity" and genome.nodes[node_id].coordinate is not None for node_id in genome.hidden_ids)
    assert len(genome.connections) == 101 * 4 + 4 * 10 + 10  # U + V + bias offsets
    net = decode(genome, 100, 10)
    out = net(torch.randn(5, 100))
    assert out.shape == (5, 10)
    loss = out.square().mean()
    loss.backward()
    assert net.weights.grad is not None and float(net.weights.grad.abs().sum()) > 0.0


def test_factored_members_align_for_neat_crossover() -> None:
    rng = random.Random(1)
    first = factored(50, 6, rng=rng, rank=3, threshold=64)
    second = factored(50, 6, rng=rng, rank=3, threshold=64)
    innovations_first = {(conn.in_id, conn.out_id): conn.innovation for conn in first.connections}
    innovations_second = {(conn.in_id, conn.out_id): conn.innovation for conn in second.connections}
    assert innovations_first == innovations_second  # pair-derived: same edge, same number, any member


def test_factored_fan_in_scaling_keeps_latents_sane() -> None:
    genome = factored(10000, 5, rng=random.Random(2), rank=2, threshold=64)
    net = decode(genome, 10000, 5)
    with torch.no_grad():
        out = net(torch.rand(8, 10000))  # encoder-normalized inputs live in [0, 1]
    assert bool(out.isfinite().all())
    assert float(out.abs().max()) < 1000.0  # unscaled gauss weights at 10k fan-in would be ~sqrt(10k) larger


# --- sparse -----------------------------------------------------------------------------------------


def test_sparse_below_threshold_is_minimal_byte_identical() -> None:
    seeded = sparse(4, 2, rng=random.Random(3), density=0.01, threshold=4096)
    baseline = minimal(4, 2, rng=random.Random(3))
    assert genome_to_dict(seeded) == genome_to_dict(baseline)


def test_sparse_edge_count_floor_and_shape() -> None:
    genome = sparse(100, 10, rng=random.Random(0), density=0.05, threshold=64)
    input_edges = [conn for conn in genome.connections if genome.nodes[conn.in_id].kind is NodeKind.INPUT]
    bias_edges = [conn for conn in genome.connections if genome.nodes[conn.in_id].kind is NodeKind.BIAS]
    assert len(input_edges) == round(100 * 10 * 0.05)
    assert len(bias_edges) == 10  # every output floored, no dead readouts at any density
    assert len({(conn.in_id, conn.out_id) for conn in genome.connections}) == len(genome.connections)  # sampled without replacement
    assert all(genome.nodes[conn.out_id].kind is NodeKind.OUTPUT for conn in genome.connections)
    out = decode(genome, 100, 10)(torch.rand(3, 100))
    assert out.shape == (3, 10)


def test_sparse_members_use_pair_derived_innovations() -> None:
    genome = sparse(100, 10, rng=random.Random(5), density=0.02, threshold=64)
    total = 100 + 1 + 10
    assert all(conn.innovation == conn.in_id * total + conn.out_id for conn in genome.connections)


# --- prune_and_regrow -------------------------------------------------------------------------------


def test_prune_and_regrow_holds_density_and_stays_acyclic() -> None:
    genome = sparse(100, 10, rng=random.Random(0), density=0.02, threshold=64)
    child = prune_and_regrow(genome, _ctx(genome), rng=random.Random(7), prob=1.0, fraction=0.2)
    assert child is not genome
    parent_forward = [conn for conn in genome.connections if conn.enabled and not conn.recurrent]
    child_forward = [conn for conn in child.connections if conn.enabled and not conn.recurrent]
    assert len(child_forward) == len(parent_forward)  # k pruned, k regrown
    doomed_count = max(1, round(len(parent_forward) * 0.2))
    doomed = sorted(parent_forward, key=lambda conn: abs(conn.weight))[:doomed_count]
    child_triples = {(conn.in_id, conn.out_id, conn.weight) for conn in child.connections}
    assert all((conn.in_id, conn.out_id, conn.weight) not in child_triples for conn in doomed)
    topological_order(child)  # raises on a cycle
    assert decode(child, 100, 10)(torch.rand(2, 100)).shape == (2, 10)


def test_prune_and_regrow_gate_returns_same_object() -> None:
    genome = sparse(100, 10, rng=random.Random(0), density=0.02, threshold=64)
    assert prune_and_regrow(genome, _ctx(genome), rng=random.Random(1), prob=0.0) is genome


def test_prune_and_regrow_regrown_edges_share_innovations_across_members() -> None:
    ctx = _ctx()
    first = ctx.innovations.innovation(3, 9)
    second = ctx.innovations.innovation(3, 9)
    assert first == second  # the memo keeps regrown edges crossover-aligned within a run


def test_prune_and_regrow_passes_cyclic_genomes_through() -> None:
    # NEAT crossover can hand the mutation pipeline a cyclic child; the pipeline repairs cycles
    # AFTER mutation (loop.advance_modules mutates then make_acyclic-s), so every operator must
    # tolerate one. The 2026-07-12 overnight run died here: topological_order raised mid-pipeline.
    from dataclasses import replace

    from versal.evolution.genome import ConnectionGene

    genome = sparse(20, 4, rng=random.Random(0), density=0.1, threshold=64)
    hidden_a, hidden_b = genome.max_node_id() + 1, genome.max_node_id() + 2
    genome.nodes[hidden_a] = replace(genome.nodes[genome.output_ids[0]], id=hidden_a, kind=NodeKind.HIDDEN)
    genome.nodes[hidden_b] = replace(genome.nodes[genome.output_ids[0]], id=hidden_b, kind=NodeKind.HIDDEN)
    genome.connections.append(ConnectionGene(hidden_a, hidden_b, 1.0, True, 900))
    genome.connections.append(ConnectionGene(hidden_b, hidden_a, 1.0, True, 901))  # the cycle
    child = prune_and_regrow(genome, _ctx(genome), rng=random.Random(3), prob=1.0, fraction=0.2)
    assert child is genome  # untouched pass-through; the downstream make_acyclic repair owns it


def test_prune_and_regrow_never_targets_macro_stubs() -> None:
    from dataclasses import replace

    from versal.evolution.genome import MacroGene

    genome = sparse(50, 5, rng=random.Random(0), density=0.05, threshold=64)
    stub_id = genome.max_node_id() + 1
    genome.nodes[stub_id] = replace(genome.nodes[genome.output_ids[0]], id=stub_id, kind=NodeKind.HIDDEN)
    genome.macros.append(MacroGene(ref="library:fake", input_node_ids=(genome.input_ids[0],), output_node_ids=(stub_id,), innovation=999))
    child = prune_and_regrow(genome, _ctx(genome), rng=random.Random(11), prob=1.0, fraction=0.3)
    assert all(conn.out_id != stub_id for conn in child.connections)


# --- direct wide-output guard -----------------------------------------------------------------------


def test_direct_strategy_declines_wide_outputs(decomposable_task: Task) -> None:
    from tests.test_hierarchical_loop import _config as _loop_config
    from versal.strategy import EVOLVE_STRATEGY

    config = _loop_config()
    config["orchestrator"] = {"direct": {"max_flat_outputs": 1}}
    strategy = EVOLVE_STRATEGY.get("direct")(config)
    result = strategy(decomposable_task, None, None, budget=3)  # declined before spec/runtime are touched
    assert result.metric == 0.0 and result.generations_used == 0
    assert result.champion_genome is None and result.champion_comp is None
    assert result.champion_metrics["declined_flat_width"] == 2.0


def test_direct_strategy_guard_defaults_off(decomposable_task: Task) -> None:
    from tests.test_hierarchical_loop import _config as _loop_config
    from versal.strategy import EVOLVE_STRATEGY

    config = _loop_config()
    config["orchestrator"] = {"direct": {}}
    strategy = EVOLVE_STRATEGY.get("direct")(config)
    assert strategy.max_flat_outputs == 0
    assert strategy.max_init_genes == 0


def test_direct_strategy_declines_oversize_init_genes(decomposable_task: Task) -> None:
    # The 2026-07-06 recon wedge: a 409,600 x 8 task built a 3.3M-gene population whose init plus
    # first generation ran for hours BEFORE any deadline check exists. The guard must refuse the
    # attempt from the dense-init arithmetic alone, before the adapter or population is built.
    from tests.test_hierarchical_loop import _config as _loop_config
    from versal.dataset.icarus import support_loader
    from versal.strategy import EVOLVE_STRATEGY

    support_input, support_output = support_loader(decomposable_task)
    flat_in = 1
    for dim in support_input.data.shape[1:]:
        flat_in *= int(dim)
    flat_out = 1
    for dim in support_output.data.shape[1:]:
        flat_out *= int(dim)

    config = _loop_config()
    config["orchestrator"] = {"direct": {"max_init_genes": 1}}
    strategy = EVOLVE_STRATEGY.get("direct")(config)
    result = strategy(decomposable_task, None, None, budget=3)  # declined before spec/runtime are touched
    assert result.metric == 0.0 and result.generations_used == 0
    assert result.champion_genome is None and result.champion_comp is None
    assert result.champion_metrics["declined_init_genes"] == float((flat_in + 1) * flat_out)


# --- decompose-first --------------------------------------------------------------------------------


def test_decompose_first_runs_before_any_flat_parent_attempt(tmp_path: Path, decomposable_task: Task) -> None:
    # half_parity is 8 -> 2, estimate (8 + 1) x 2 = 18 > 10; its output_slices halves are 9 <= 10.
    orchestrator = _orchestrator(tmp_path, table={"decompose_first_above": 10, "decompose": ["output_slices"], "output_slices_n_groups": 2})
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({"half_parity.out0": 1.0, "half_parity.out1": 1.0, "half_parity": 1.0}, calls))
    solution = orchestrator.solve(decomposable_task)
    assert solution is not None
    assert calls == ["half_parity.out0", "half_parity.out1", "half_parity"]  # NO flat parent evolve first
    assert orchestrator.counters["decompose_first"] == 1
    assert orchestrator.attempts[-1].outcome == "decomposed"


def test_decompose_first_failure_falls_through_to_the_ladder_once(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path, table={"decompose_first_above": 10, "decompose": ["output_slices"], "output_slices_n_groups": 2})
    calls: list[str] = []
    _patch_run_task(orchestrator, _fake_run_task({}, calls))  # everything fails
    solution = orchestrator.solve(decomposable_task)
    assert solution is None
    assert calls == ["half_parity.out0", "half_parity"]  # failed subtask, then ONE flat attempt, never a second decompose
    assert orchestrator.counters["decompositions"] == 1
    assert orchestrator.counters["decompose_subtask_failed"] == 1
    assert orchestrator.attempts[-1].outcome == "failed"


def test_decompose_first_off_registers_nothing(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(tmp_path)
    assert "decompose_first" not in orchestrator.counters
    from versal.orchestrator import comp_task_spec

    assert orchestrator._wants_decompose_first(decomposable_task, comp_task_spec(decomposable_task)) is False


def test_adaptive_decompose_first_uses_hardware_envelope(tmp_path: Path, decomposable_task: Task) -> None:
    orchestrator = _orchestrator(
        tmp_path,
        table={"decompose_first_above": "adaptive"},
        config_extra={"resources": {"mode": "adaptive", "host_reserve_gb": 1_000_000, "device_reserve_gb": 1_000_000}},
    )
    from versal.orchestrator import comp_task_spec

    assert orchestrator._wants_decompose_first(decomposable_task, comp_task_spec(decomposable_task)) is True
    assert "resource_declines" in orchestrator.counters
