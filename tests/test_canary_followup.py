"""Regressions for canary-driven reporting, compact ports, handoff, and persistence."""

import json
import random
import shutil
from pathlib import Path

import torch

from ardevo.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from ardevo.decompose import input_subsets, output_slices, spatial_patches, time_windows
from ardevo.evolution.composition import (
    AssemblyContext,
    CompEdgeGene,
    CompNodeGene,
    CompNodeKind,
    CompositionGenome,
    IndexRun,
    PortMap,
    assemble,
    comp_from_dict,
    comp_to_dict,
    minimal_composition,
)
from ardevo.evolution.genome import InnovationTracker, genome_to_dict
from ardevo.evolution.loop import AssessedComposition
from ardevo.external_archive import ArchiveManager, restore_snapshot
from ardevo.library import MODULE, ModuleLibrary, task_io
from ardevo.motif_discovery import _rewire_degree_preserving, classify_counterfactuals
from ardevo.orchestrator import comp_task_spec
from ardevo.reporting import write_run_report
from ardevo.routing import RouterService, migrate_router_library
from ardevo.strategy import StrategyResult


def test_report_preserves_missing_vs_zero_and_future_wrapper(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 99,
                "report": {
                    "records": [
                        {"rung": 1, "task": "missing", "metric": 1.0, "outcome": "failed"},
                        {"rung": 1, "task": "zero", "metric": 0.0, "report_metric": 0.0, "outcome": "evolved", "new_library_keys": []},
                    ]
                },
                "rungs": [1],
            }
        )
    )
    report = write_run_report(run)
    assert report["quality"]["held_out_query_count"] == 1
    assert report["quality"]["held_out_accuracy_mean"] == 0.0
    assert report["rungs"][0]["query_count"] == 1
    assert "N/A" not in (run / "rung_summary.csv").read_text().splitlines()[1]  # rung aggregate has a valid zero


def test_fixed_port_map_gathers_scatters_and_round_trips() -> None:
    comp = CompositionGenome(
        nodes={
            0: CompNodeGene(0, CompNodeKind.INPUT, "x", 0, 6),
            1: CompNodeGene(1, CompNodeKind.OUTPUT, "y", 5, 0),
        },
        edges=[CompEdgeGene(0, 1, True, 0, (), 0, PortMap((IndexRun(1, 3, 2), IndexRun(5, 0, 1))))],
    )
    restored = comp_from_dict(comp_to_dict(comp))
    assert restored.edges[0].port_map == comp.edges[0].port_map and restored.edges[0].glue == ()
    net = assemble(restored, AssemblyContext(bank_columns={"x": range(6)}), 6)
    output = net(torch.tensor([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]]))
    assert torch.equal(output, torch.tensor([[15.0, 0.0, 0.0, 11.0, 12.0]]))
    assert list(net.parameters()) == []  # fixed maps never become trainable dense glue


def _expanded_runs(runs: tuple[tuple[int, int, int], ...]) -> tuple[list[int], list[int]]:
    sources: list[int] = []
    targets: list[int] = []
    for source, target, length in runs:
        sources.extend(range(source, source + length))
        targets.extend(range(target, target + length))
    return sources, targets


def _axis_task(axes: tuple[Axis, ...], shape: tuple[int, ...], name: str) -> Task:
    values = (torch.arange(torch.tensor(shape).prod().item()).reshape(shape) % 2).float()
    pair = (
        Field(values, axes, ValueType.BINARY, None, None, None),
        Field(1.0 - values, axes, ValueType.BINARY, None, None, None),
    )
    return Task(meta=TaskMeta(rung=0, kind=TaskKind.MAP, name=name, fixed_split=True), support=[pair], query=[pair])


def test_decomposition_port_runs_select_and_place_all_roles_across_axes(decomposable_task, temporal_seq_task) -> None:
    output = output_slices(decomposable_task, rng=random.Random(0), n_groups=2)[1].port
    assert _expanded_runs(output.input_runs) == (list(range(8)), list(range(8)))
    assert _expanded_runs(output.output_runs) == ([0], [1])

    subset = input_subsets(decomposable_task, rng=random.Random(0), n_subsets=2)[1].port
    assert _expanded_runs(subset.input_runs) == (list(range(4, 8)), list(range(4)))
    assert _expanded_runs(subset.output_runs) == ([0, 1], [0, 1])

    temporal = _axis_task((Axis.EXTRA, Axis.TIME), (2, 8), "axis_time")
    window = time_windows(temporal, rng=random.Random(0), n_windows=2)[1].port
    assert _expanded_runs(window.input_runs) == ([4, 5, 6, 7, 12, 13, 14, 15], list(range(8)))
    assert _expanded_runs(window.output_runs) == (list(range(8)), [4, 5, 6, 7, 12, 13, 14, 15])
    # The existing one-dimensional fixture exercises the merged-run fast path too.
    assert time_windows(temporal_seq_task, rng=random.Random(0), n_windows=2)[1].port.input_runs == ((4, 0, 4),)

    spatial = _axis_task((Axis.EXTRA, Axis.HEIGHT, Axis.WIDTH), (2, 4, 3), "axis_spatial")
    patch = spatial_patches(spatial, rng=random.Random(0), n_patches=2)[1].port
    selected = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]
    assert _expanded_runs(patch.input_runs) == (selected, list(range(12)))
    assert _expanded_runs(patch.output_runs) == (list(range(12)), selected)


def test_motif_null_preserves_degrees_and_counterfactual_thresholds() -> None:
    edges = {(0, 2): 1, (0, 3): 1, (1, 2): 1, (1, 3): 1, (2, 4): 2, (3, 5): 2}
    rewired = _rewire_degree_preserving(edges, random.Random(7))
    for node in range(6):
        assert sum(source == node for source, _ in edges) == sum(source == node for source, _ in rewired)
        assert sum(target == node for _, target in edges) == sum(target == node for _, target in rewired)
    assert sorted(edges.values()) == sorted(rewired.values())

    controls = [0.0] * 16
    one_root = classify_counterfactuals([{"accuracy_drop": 0.02, "control_drops": controls, "lineage_root": "a"}], observed=True)
    assert one_root["evidence"] == "functional"
    replicated = classify_counterfactuals(
        [
            {"accuracy_drop": 0.03, "control_drops": controls, "lineage_root": "a"},
            {"accuracy_drop": 0.04, "control_drops": controls, "lineage_root": "b"},
        ],
        observed=True,
    )
    assert replicated["evidence"] == "replicated"


def test_routed_below_bar_composition_is_handed_to_composition(tmp_path: Path, xor_task) -> None:
    from tests.test_orchestrator import _orchestrator

    orchestrator = _orchestrator(tmp_path, table={"accept_threshold": 0.95, "decompose": []})
    spec = comp_task_spec(xor_task)
    comp = minimal_composition(spec.input_specs, spec.output_ref, spec.output_width, InnovationTracker(_next_node_id=0), random.Random(0))
    assessed = AssessedComposition(comp=comp, metrics={"support_accuracy": 0.6}, fitness=0.6, net=None)
    seen: list[CompositionGenome] = []

    def routed(*_args, **_kwargs):
        return StrategyResult(
            strategy="routed",
            metric=0.6,
            generations_used=1,
            champion_comp=assessed,
            champion_metrics={"support_accuracy": 0.6},
            strategy_metrics={"router_score": 1.0, "distilled_score": 0.6, "distillation_gap": 0.4},
        )

    def composition(*_args, **kwargs):
        seen.extend(kwargs["seed_comps"])
        return StrategyResult(strategy="composition", metric=0.7, generations_used=1, champion_comp=assessed, champion_metrics={"support_accuracy": 0.7})

    orchestrator.strategies = [("routed", routed), ("composition", composition)]
    orchestrator.evolve_shares = {"routed": 1.0, "composition": 1.0}
    result = orchestrator._evolve(xor_task, spec, 4)
    assert len(seen) == 1 and comp_to_dict(seen[0]) == comp_to_dict(comp)
    assert result.strategy_metrics["handoff_count"] == 1.0
    assert result.strategy_metrics["recovery_result"] == 0.7


def _seed_router_library(root: Path, xor_task, solving_genome) -> ModuleLibrary:
    library = ModuleLibrary(root)
    library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=task_io(xor_task), provenance={"accepted_metric": 1.0})
    return library


def test_router_v2_eager_lazy_parity_and_v1_migration(tmp_path: Path, xor_task, solving_genome) -> None:
    from tests.test_routing import _task_view

    torch.manual_seed(4)
    library = _seed_router_library(tmp_path / "library", xor_task, solving_genome)
    router_dir = library.root / "router"
    service = RouterService(library, d_model=16, top_k=1, max_steps=2, persist_dir=router_dir)
    service.sync()
    view, x, _width = _task_view(service.net, xor_task)
    expected = view(x).detach()
    service.save()
    meta = json.loads((router_dir / "router_meta.json").read_text())
    assert meta["format_version"] == 2
    assert (router_dir / "shards" / "vertices").is_dir()

    eager = RouterService(library, d_model=16, top_k=1, max_steps=2, persist_dir=router_dir)
    lazy = RouterService(library, d_model=16, top_k=1, max_steps=2, persist_dir=router_dir, lazy_residency=True)
    assert len(lazy.net.vertex_in_adapters) == 0
    eager_view, _x, _width = _task_view(eager.net, xor_task)
    lazy_view, _x, _width = _task_view(lazy.net, xor_task)
    assert torch.equal(eager_view(x), expected)
    assert torch.equal(lazy_view(x), expected)
    assert len(lazy.net.vertex_in_adapters) == 1
    assert all(vertex.module is None for vertex in lazy.net._vertices.values() if vertex.sanitized_key not in lazy.net.last_gate_stats)
    lazy.checkpoint_and_evict()
    assert len(lazy.net.vertex_in_adapters) == len(lazy.net.input_adapters) == len(lazy.net.output_heads) == 0
    assert all(vertex.module is None for vertex in lazy.net._vertices.values())
    assert torch.equal(lazy_view(x), expected)  # evicted surfaces reload exactly on demand

    # Reconstruct a v1 source from the eager state, then migrate a separate copy.
    v1 = tmp_path / "library_v1"
    shutil.copytree(library.root, v1)
    v1_router = v1 / "router"
    v1_meta = dict(meta)
    v1_meta["format_version"] = 1
    (v1_router / "router_meta.json").write_text(json.dumps(v1_meta))
    torch.save(eager.net.state_dict(), v1_router / "router_state.pt")
    source_hash = (v1_router / "router_state.pt").read_bytes()
    migrated_root = tmp_path / "library_v2"
    migration = migrate_router_library(v1, migrated_root)
    assert migration["format_version"] == 2
    assert (v1_router / "router_state.pt").read_bytes() == source_hash
    migrated = RouterService(ModuleLibrary(migrated_root), d_model=16, top_k=1, max_steps=2, persist_dir=migrated_root / "router")
    migrated_view, _x, _width = _task_view(migrated.net, xor_task)
    assert torch.equal(migrated_view(x), expected)


def test_content_archive_deduplicates_unchanged_files(tmp_path: Path) -> None:
    run, library, remote = tmp_path / "run", tmp_path / "library", tmp_path / "remote"
    run.mkdir()
    library.mkdir()
    (run / "run_summary.json").write_text('{"status":"running"}')
    (library / "index.json").write_text("[]")
    manager = ArchiveManager.from_config({"archive": {"enabled": True, "backend": "file", "uri": remote.resolve().as_uri(), "run_key": "dedup"}}, run, library)
    assert manager is not None
    first = manager.snapshot(1)
    second = manager.snapshot(2)
    assert first["schema_version"] == 2 and first["objects_uploaded"] > 0
    assert second["objects_uploaded"] == 0 and second["objects_reused"] == len(second["files"])
    restored = tmp_path / "restored"
    restore_snapshot(remote.resolve().as_uri(), restored, run_key="dedup")
    assert (restored / "run" / "run_summary.json").exists()
