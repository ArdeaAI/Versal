"""Routed substrate: shapes across signatures, top-k sparsity, bounded steps, persistence,
strategy/orchestrator integration, and the expert-ablation diagnostic (the Stage A go/no-go gate)."""

import math
import random
from pathlib import Path

import pytest
import torch

from ardevo.dataset.icarus import Task
from ardevo.evolution.composition import comp_to_dict, minimal_composition
from ardevo.evolution.genome import Genome, InnovationTracker, genome_to_dict
from ardevo.library import COMPOSITION, MODULE, ModuleLibrary, task_io
from ardevo.orchestrator import comp_task_spec
from ardevo.routing import RoutedNet, RoutedTaskView, RouterService, build_vertex, sanitize_key
from tests.test_orchestrator import _orchestrator


def _seed_library(tmp_path: Path, xor_task: Task, solving_genome: Genome, *, with_composition: bool = False) -> ModuleLibrary:
    library = ModuleLibrary(tmp_path / "lib")
    library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0, "weight_robustness": 0.5})
    if with_composition:
        comp = minimal_composition([("BINARY|K", 2)], "xor", 1, InnovationTracker(_next_node_id=0), random.Random(1))
        library.add(entry_type=COMPOSITION, payload=comp_to_dict(comp), io=task_io(xor_task), provenance={"accepted_metric": 0.9}, level=2)
    return library


def _task_view(net: RoutedNet, task: Task) -> tuple[RoutedTaskView, torch.Tensor, int]:
    io = task_io(task)
    input_key = net.ensure_input_adapter(io["inputs"][0]["signature"], io["inputs"][0]["width"])
    head_key = net.ensure_output_head(io["output"]["signature"], io["output"]["width"])
    spec = comp_task_spec(task)
    x, _descriptor = spec.encoded.support_input
    return RoutedTaskView(net, input_key=input_key, head_key=head_key, support_input=x), x, io["output"]["width"]


def test_routed_forward_shape_across_signatures(tmp_path: Path, xor_task: Task, decomposable_task: Task, solving_genome: Genome) -> None:
    torch.manual_seed(0)
    library = _seed_library(tmp_path, xor_task, solving_genome, with_composition=True)
    net = RoutedNet(d_model=16, top_k=2, max_steps=3)
    assert net.sync_with_library(library) == 2  # one module vertex, one composition vertex
    for task in (xor_task, decomposable_task):
        view, x, output_width = _task_view(net, task)
        out = view(x)
        assert out.shape == (x.shape[0], output_width)
        assert torch.isfinite(out).all()
    # Signature-keyed lazy surfaces: widths 2 and 8 in, widths 1 and 2 out, disjoint parameters.
    assert len(net.input_adapters) == 2 and len(net.output_heads) == 2


def test_top_k_sparsity_and_gate_probs(tmp_path: Path, xor_task: Task, solving_genome: Genome, linear_genome: Genome) -> None:
    torch.manual_seed(0)
    library = _seed_library(tmp_path, xor_task, solving_genome, with_composition=True)
    library.add(entry_type=MODULE, payload=genome_to_dict(linear_genome), io=task_io(xor_task), provenance={"accepted_metric": 0.5})
    net = RoutedNet(d_model=16, top_k=2, max_steps=2)
    assert net.sync_with_library(library) == 3
    view, x, _width = _task_view(net, xor_task)
    view(x)
    assert len(net.last_selections) == 2  # one gate decision per routing step
    for selections, probs in zip(net.last_selections, net.last_probs):
        assert selections.shape == (x.shape[0], 2)  # exactly top_k selections PER SAMPLE
        assert torch.allclose(probs.sum(dim=1), torch.ones(x.shape[0]))  # a distribution over the selection
    assert net.last_aux_loss.item() >= 0.0  # load-balance term is live


def test_bounded_steps_and_state_threading(tmp_path: Path, xor_task: Task, solving_genome: Genome) -> None:
    """Execution is exactly max_steps bus updates (the no-stuck-loop guarantee is structural), and
    the bus threads state: the same weights run for 1 vs 4 steps give different outputs, which is
    what makes multi-step (cyclic) pathways real rather than decorative."""

    def build(max_steps: int) -> RoutedNet:
        torch.manual_seed(7)  # identical init: the two nets differ ONLY in step count
        net = RoutedNet(d_model=16, top_k=1, max_steps=max_steps)
        net.sync_with_library(_seed_library(tmp_path / f"steps{max_steps}", xor_task, solving_genome))
        return net

    one_step = build(1)
    four_step = build(4)
    view_one, x, _w = _task_view(one_step, xor_task)
    view_four, _x, _w2 = _task_view(four_step, xor_task)
    four_step.load_state_dict(one_step.state_dict())  # lazy surfaces exist on both; byte-identical weights
    out_one, out_four = view_one(x), view_four(x)
    assert len(one_step.last_selections) == 1 and len(four_step.last_selections) == 4
    assert not torch.allclose(out_one, out_four)


def test_persistence_round_trip_and_growth(tmp_path: Path, xor_task: Task, solving_genome: Genome, linear_genome: Genome) -> None:
    torch.manual_seed(0)
    library = _seed_library(tmp_path, xor_task, solving_genome)
    router_dir = tmp_path / "lib" / "router"
    service = RouterService(library, d_model=16, top_k=2, max_steps=2, persist_dir=router_dir)
    service.sync()
    view, x, _w = _task_view(service.net, xor_task)
    view(x)  # touch the lazy surfaces so they carry trained-state semantics
    service.record_task({"task": "xor", "metric": 1.0})
    service.save()
    assert (router_dir / "router_state.pt").exists() and (router_dir / "router_meta.json").exists()

    reloaded = RouterService(library, d_model=16, top_k=2, max_steps=2, persist_dir=router_dir)
    original_state = service.net.state_dict()
    reloaded_state = reloaded.net.state_dict()
    assert sorted(original_state) == sorted(reloaded_state)
    assert all(torch.equal(original_state[key], reloaded_state[key]) for key in original_state)
    assert reloaded.version == 1 and reloaded.train_history[-1]["task"] == "xor"

    # Growth: a new library entry appends a vertex; every persisted row stays byte-identical.
    library.add(entry_type=MODULE, payload=genome_to_dict(linear_genome), io=task_io(xor_task), provenance={"accepted_metric": 0.6})
    grown = RouterService(library, d_model=16, top_k=2, max_steps=2, persist_dir=router_dir)
    grown_state = grown.net.state_dict()
    assert all(torch.equal(original_state[key], grown_state[key]) for key in original_state)  # old rows untouched
    assert len(grown.net._vertex_order) == len(service.net._vertex_order) + 1


def test_persistence_mismatch_starts_fresh_or_raises(tmp_path: Path, xor_task: Task, solving_genome: Genome) -> None:
    library = _seed_library(tmp_path, xor_task, solving_genome)
    router_dir = tmp_path / "lib" / "router"
    service = RouterService(library, d_model=16, top_k=2, max_steps=2, persist_dir=router_dir)
    service.sync()
    service.save()
    mismatched = RouterService(library, d_model=32, top_k=2, max_steps=2, persist_dir=router_dir)  # d_model changed
    assert len(mismatched.net._vertex_order) == 0  # started fresh, old state renamed aside
    assert not (router_dir / "router_meta.json").exists()
    service.persist_dir = router_dir
    service.save()
    with pytest.raises(ValueError, match="does not match config"):
        RouterService(library, d_model=32, top_k=2, max_steps=2, persist_dir=router_dir, persist_strict=True)


def test_retired_vertices_are_masked(tmp_path: Path, xor_task: Task, solving_genome: Genome, linear_genome: Genome) -> None:
    torch.manual_seed(0)
    library = _seed_library(tmp_path, xor_task, solving_genome)
    retired_key = library.add(entry_type=MODULE, payload=genome_to_dict(linear_genome), io=task_io(xor_task), provenance={"accepted_metric": 0.6})
    net = RoutedNet(d_model=16, top_k=2, max_steps=2)
    net.sync_with_library(library)
    library.retire(retired_key)
    net.sync_with_library(library)  # refreshes the retired mask; the row itself stays
    view, x, _w = _task_view(net, xor_task)
    view(x)
    assert sanitize_key(retired_key) in net._vertex_order  # rows never dangle
    assert sanitize_key(retired_key) not in net.last_gate_stats  # but the gate can never pick them


def test_routed_win_distills_into_admitted_composition(tmp_path: Path, xor_task: Task, solving_genome: Genome) -> None:
    """The DSL contract end-to-end: a routed win over a planted expert must come back as a VERIFIED
    composition referencing that expert, admitted into the library (a new routable vertex)."""
    from ardevo.evolution.composition import comp_from_dict

    torch.manual_seed(0)
    table = {
        "evolve": ["routed"],
        "accept_threshold": 0.2,
        "decompose": [],
        "budgets": {"depth0": 3},
        "routed": {"d_model": 16, "top_k": 2, "max_steps": 2, "train_steps": 30, "persist": False},
    }
    orchestrator = _orchestrator(tmp_path, table=table)
    # A foreign signature makes the LOOKUP miss (so the ladder reaches the routed strategy) while
    # the entry stays live and routable: the router's vertex set is signature-agnostic.
    foreign_io = {"inputs": [{"signature": "BINARY|C", "width": 2}], "output": {"signature": "BINARY|C", "width": 1}}
    planted = orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=foreign_io, provenance={"accepted_metric": 1.0})
    keys_before = set(orchestrator.library.keys())
    solution = orchestrator.solve(xor_task)
    assert solution is not None, [attempt.to_dict() for attempt in orchestrator.attempts]
    assert solution.key is not None and solution.entry_type == COMPOSITION  # solved AND shelved
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "evolved" and attempt.strategy == "routed" and attempt.library_key == solution.key
    new_keys = set(orchestrator.library.keys()) - keys_before
    assert new_keys == {solution.key}
    entry = orchestrator.library.load(solution.key)
    assert entry.entry_type == COMPOSITION and entry.level == 2
    assert f"library:{planted}" in comp_from_dict(entry.payload).refs()  # the pathway names the expert
    assert orchestrator.counters["routed_solved"] == 1
    assert orchestrator.counters["routed_zero_shot"] in (0, 1)
    assert orchestrator.counters["routed_undistillable"] == 0


def test_routed_no_experts_short_circuits_and_escalates(tmp_path: Path, xor_task: Task) -> None:
    """A cold library means nothing to route over: the strategy must charge NOTHING and fall
    through, so evolution populates the vertex set first (no adapter-memorization solves)."""
    torch.manual_seed(0)
    table = {
        "evolve": ["routed"],
        "accept_threshold": 0.2,
        "decompose": [],
        "budgets": {"depth0": 3},
        "routed": {"d_model": 16, "top_k": 2, "max_steps": 2, "train_steps": 30, "persist": False},
    }
    orchestrator = _orchestrator(tmp_path, table=table)
    solution = orchestrator.solve(xor_task)
    assert solution is None  # routed was the only strategy; it declined at zero cost
    attempt = orchestrator.attempts[-1]
    assert attempt.outcome == "failed" and attempt.generations == 0
    assert orchestrator.counters["routed_no_experts"] == 1
    assert orchestrator.counters["routed_solved"] == 0
    assert len(orchestrator.library) == 0


def test_routed_undistillable_win_reports_miss(tmp_path: Path, xor_task: Task, solving_genome: Genome) -> None:
    """A router-space win whose pathway cannot be kept (here: an impossible usage floor, the
    adapter-bypass proxy) must report as a miss with the marker, never as a solve."""
    from ardevo.routing import RoutedStrategy

    torch.manual_seed(0)
    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.2, "decompose": []})
    orchestrator.library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    strategy = RoutedStrategy(
        library_dir=str(tmp_path / "lib"),
        d_model=16,
        top_k=1,
        max_steps=2,
        train_steps=30,
        persist=False,
        distill_usage_floor=2.0,  # no expert can clear a floor above 1.0: the pathway is always empty
    )
    result = strategy(xor_task, comp_task_spec(xor_task), orchestrator._runtime(), budget=1)
    assert result.metric == 0.0 and result.champion_comp is None and result.champion_routed is None
    assert result.champion_metrics["routed_undistillable"] == 1.0
    assert result.champion_metrics["routed_metric"] >= 0.2  # the router-space win is kept as a diagnostic


def test_pending_embedding_places_new_vertex(tmp_path: Path, xor_task: Task, solving_genome: Genome) -> None:
    """A distilled entry's vertex is born AT the task embedding that produced it (fingerprint-matched
    at sync), not at the mean-of-peers default."""
    from ardevo.library import structural_fingerprint

    torch.manual_seed(0)
    library = ModuleLibrary(tmp_path / "lib")
    service = RouterService(library, d_model=16, top_k=1, max_steps=2)
    payload = genome_to_dict(solving_genome)
    placement = torch.linspace(-1.0, 1.0, 16)
    service.note_pending_embedding(structural_fingerprint(MODULE, payload), placement)
    library.add(entry_type=MODULE, payload=payload, io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    assert service.sync() == 1
    key = sanitize_key(library.keys()[0])
    assert torch.equal(service.net.vertex_embeddings[key].detach(), placement)
    assert service.pending_embeddings == {}  # consumed on use


def test_build_vertex_skips_temporal_and_undecodable(tmp_path: Path, temporal_task: Task, xor_task: Task, solving_genome: Genome) -> None:
    library = ModuleLibrary(tmp_path / "lib")
    temporal_key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(temporal_task), provenance={})
    assert build_vertex(library.load(temporal_key), library) is None  # TIME-bearing entries are out of scope
    broken_key = library.add(entry_type=COMPOSITION, payload={"nodes": [], "edges": []}, io=task_io(xor_task), provenance={}, level=2)
    assert build_vertex(library.load(broken_key), library) is None  # undecodable entries skip, never raise


def test_halting_weights_readouts_and_can_exit_early(tmp_path: Path, xor_task: Task, solving_genome: Genome) -> None:
    torch.manual_seed(0)
    library = _seed_library(tmp_path, xor_task, solving_genome)
    net = RoutedNet(d_model=16, top_k=1, max_steps=6, halting=True, ponder_epsilon=0.01)
    net.sync_with_library(library)
    view, x, _w = _task_view(net, xor_task)
    out = view(x)
    assert out.shape[0] == x.shape[0] and torch.isfinite(out).all()
    assert 0.0 < net.last_expected_steps <= 6.0
    assert net.last_aux_loss.item() > 0.0  # the ponder cost rides the aux loss
    # Bias the halt head hard toward stopping: the unroll must genuinely shorten under the cap.
    assert net.halt_head is not None
    with torch.no_grad():
        net.halt_head.bias.fill_(12.0)  # sigmoid ~= 1: all halting mass spent at step 1
    view(x)
    assert len(net.last_selections) < 6
    assert net.last_expected_steps < 2.0


def test_routing_trace_shape_for_deep_supervision(tmp_path: Path, xor_task: Task, solving_genome: Genome) -> None:
    torch.manual_seed(0)
    net = RoutedNet(d_model=16, top_k=1, max_steps=3)
    net.sync_with_library(_seed_library(tmp_path, xor_task, solving_genome))
    io = task_io(xor_task)
    input_key = net.ensure_input_adapter(io["inputs"][0]["signature"], io["inputs"][0]["width"])
    head_key = net.ensure_output_head(io["output"]["signature"], io["output"]["width"])
    spec = comp_task_spec(xor_task)
    x, _descriptor = spec.encoded.support_input
    task_embed = net.task_embedding(x, input_key, head_key)
    trace = net.routing_trace(x, input_key=input_key, head_key=head_key, task_embed=task_embed)
    assert trace.shape == (x.shape[0], 3, io["output"]["width"])  # the refine_trace analogue


def test_edge_bias_params_are_per_vertex_and_growth_stays_append_only(tmp_path: Path, xor_task: Task, solving_genome: Genome, linear_genome: Genome) -> None:
    torch.manual_seed(0)
    library = _seed_library(tmp_path, xor_task, solving_genome)
    net = RoutedNet(d_model=16, top_k=1, max_steps=3, edge_bias=True)
    net.sync_with_library(library)
    view, x, _w = _task_view(net, xor_task)
    out = view(x)
    assert torch.isfinite(out).all()
    assert len(net.vertex_edge_out) == 1 and len(net.vertex_edge_in) == 1  # factorized: per-vertex vectors
    before = {key: tensor.clone() for key, tensor in net.state_dict().items()}
    library.add(entry_type=MODULE, payload=genome_to_dict(linear_genome), io=task_io(xor_task), provenance={"accepted_metric": 0.6})
    net.sync_with_library(library)
    after = net.state_dict()
    assert len(net.vertex_edge_out) == 2
    assert all(torch.equal(before[key], after[key]) for key in before)  # a new vertex never resizes old rows


def test_overmind_render_written_on_growth(tmp_path: Path, xor_task: Task, solving_genome: Genome, linear_genome: Genome) -> None:
    torch.manual_seed(0)
    library = _seed_library(tmp_path, xor_task, solving_genome)
    image_dir = tmp_path / "lib" / "images"
    service = RouterService(library, d_model=16, top_k=2, max_steps=2, persist_dir=tmp_path / "lib" / "router", image_dir=image_dir)
    service.sync()  # one vertex -> first render
    overmind = image_dir / "overmind.png"
    assert overmind.exists()
    first_mtime = overmind.stat().st_mtime_ns
    # A new library entry is a "significant addition": the portrait refreshes.
    library.add(entry_type=MODULE, payload=genome_to_dict(linear_genome), io=task_io(xor_task), provenance={"accepted_metric": 0.6})
    assert service.sync() == 1
    assert overmind.stat().st_mtime_ns >= first_mtime  # rewritten on growth


def test_build_overmind_spec_embeds_experts_and_degrades(tmp_path: Path, xor_task: Task, solving_genome: Genome) -> None:
    from ardevo.rendering import OvermindVertex, OvermindView, build_overmind_spec, library_resolver

    library = _seed_library(tmp_path, xor_task, solving_genome, with_composition=True)
    keys = library.keys()
    view = OvermindView(
        vertices=[OvermindVertex(key=keys[0], label=keys[0]), OvermindVertex(key="does_not_exist", label="ghost"), OvermindVertex(key=keys[1], label=keys[1], retired=True)],
        input_signatures=["BINARY|K:2"],
        output_signatures=["BINARY|K:1"],
        d_model=16,
        top_k=2,
        max_steps=3,
    )
    spec = build_overmind_spec(view, resolve=library_resolver(library))
    # Bus host contributes input + gate + output nodes; the real expert embeds its own network too.
    assert spec.node_count > 3
    labels = [box.label for box in spec.containers]
    assert any("ghost" in label for label in labels)  # a missing key degrades to a labeled opaque box, never raises


def test_expert_ablation_diagnostic(tmp_path: Path, xor_task: Task, decomposable_task: Task, solving_genome: Genome) -> None:
    """The Stage A risk experiment: are frozen experts contributing signal the adapters alone
    cannot ("adapter bypass")? Runs the identical router with real vs zeroed experts, same seed and
    steps, on the planted-solution XOR fixture AND on half-parity (not linearly separable, so the
    adapter path cannot trivially absorb it). The comparison is REPORTED (research signal); the
    instrumentation is asserted."""
    from ardevo.routing import RoutedStrategy

    results: dict[tuple[str, str, int], float] = {}
    variants = [("xor", xor_task, 0), ("half_parity", decomposable_task, 0), ("half_parity", decomposable_task, 2)]
    for task_name, task, rank in variants:
        for ablation in ("none", "zero"):
            torch.manual_seed(3)
            library = _seed_library(tmp_path / f"{task_name}_{rank}_{ablation}", xor_task, solving_genome)
            strategy = RoutedStrategy(
                library_dir=str(tmp_path / f"{task_name}_{rank}_{ablation}" / "lib"),
                d_model=16,
                top_k=1,
                max_steps=2,
                train_steps=60,
                lr=0.01,
                adapter_rank=rank,
                persist=False,
                zero_shot_accept=False,
                expert_ablation=ablation,
                distill=False,  # the diagnostic instruments ROUTER-SPACE behavior, pre-contract
            )
            orchestrator = _orchestrator(tmp_path / f"orc_{task_name}_{rank}_{ablation}", table={"accept_threshold": 0.999, "decompose": []})
            runtime = orchestrator._runtime()
            runtime.library = library  # score against the seeded library, not the orchestrator's empty one
            result = strategy(task, comp_task_spec(task), runtime, budget=1)
            assert result.champion_routed is not None and math.isfinite(result.metric)
            assert result.champion_metrics["routed_steps_used"] == 60.0
            results[(task_name, ablation, rank)] = result.champion_metrics["support_accuracy"]
    for task_name, _task, rank in variants:
        real, zeroed = results[(task_name, "none", rank)], results[(task_name, "zero", rank)]
        print(f"expert-ablation diagnostic on {task_name} (adapter_rank={rank}): real={real:.3f} zeroed={zeroed:.3f}")
