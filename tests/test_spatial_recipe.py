"""
Executable recipes, independent parameter sharing, and reuse across search strategies.
"""

import copy
import math
import random
from dataclasses import replace

import pytest
import torch

from tests.test_interleaved import _policy
from tests.test_orchestrator import _orchestrator
from versal.dataset.icarus import Axis, Field, Task, TaskKind, TaskMeta, ValueType
from versal.evaluation import support_loss
from versal.evolution.genome import ConnectionGene, Genome, InnovationTracker, MacroGene, NodeGene, NodeKind, genome_from_dict, genome_to_dict
from versal.evolution.mutation import MutationContext
from versal.evolution.spatial_ops import add_node, embed_entry, spatial_minimal, split_group
from versal.library import COMPOSITION, MODULE, ModuleLibrary, expanded_payload_complexity, graft, macro_resolver, structural_fingerprint, task_io
from versal.orchestrator import comp_task_spec
from versal.routing import build_vertex
from versal.spatial import Binding, Placement, SpatialContract, SpatialGenome, SpatialNet, SpatialTaskAdapter, encode_field, expand_spatial
from versal.strategy_sessions import SpatialSession
from versal.substrate import GraphNet, decode_module
from versal.topology import same_topology, topology_record


def task(shape=(4,), axes=(Axis.EXTRA,), *, classification=False):
    data = torch.linspace(-1, 1, math.prod(shape)).reshape(shape)
    source = Field(data, axes, ValueType.CONTINUOUS, None, None, None)
    target = Field(torch.tensor(1), (), ValueType.CATEGORICAL, 3, None, None) if classification else Field(data.clone(), axes, ValueType.CONTINUOUS, None, None, None)
    return Task(TaskMeta(999, TaskKind.MAP, "arbitrary"), [(source, target)], [(source, target)])


def recipe(task_value=None, *, shared=True):
    contract = SpatialContract.from_task(task_value or task())
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.BIAS, "identity"),
        2: NodeGene(2, NodeKind.HIDDEN, "sin"),
        3: NodeGene(3, NodeKind.OUTPUT, "identity"),
    }
    return SpatialGenome(
        nodes=nodes,
        connections=[ConnectionGene(0, 2, 0.7, True, 0), ConnectionGene(2, 3, 1.2, True, 1)],
        contract=contract,
        placements={2: Placement(bank="input")},
        bindings={0: Binding(scale=1, shared=shared), 1: Binding(scale=1, shared=shared)},
        groups={2: [2]},
    )


@pytest.mark.parametrize("shared", [False, True])
def test_compact_matches_expanded_forward_gradients_and_roundtrip(shared):
    genome = recipe(shared=shared)
    compact = SpatialNet(genome, chunk_size=2)
    explicit = decode_module(expand_spatial(genome), 4, 4)
    assert isinstance(explicit, GraphNet)
    x = torch.tensor([[0.1, 0.4, -0.3, 0.8]], requires_grad=True)
    y = x.detach().clone().requires_grad_()
    torch.testing.assert_close(compact(x), explicit(y))
    compact(x).square().sum().backward()
    explicit(y).square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    if shared:
        assert explicit.tie_values is not None and explicit.tie_values.grad is not None
        torch.testing.assert_close(compact.weights["0"].grad[0], explicit.tie_values.grad[0])
    else:
        assert explicit.weights.grad is not None
        source = explicit._position[0]
        target = explicit._col_of[explicit._position[5]]
        torch.testing.assert_close(compact.weights["0"].grad[0], explicit.weights.grad[source, target])
    torch.optim.SGD(compact.parameters(), lr=0.1).step()
    payload = genome_to_dict(compact.writeback(genome))
    restored = genome_from_dict(payload)
    assert isinstance(restored, SpatialGenome)
    torch.testing.assert_close(SpatialNet(restored)(x), compact(x))
    assert restored.complexity() == 12


def test_split_preserves_function_then_one_copy_can_specialize():
    genome = recipe()
    separated = split_group(genome, [2], InnovationTracker.from_genomes([genome]), 2)
    x = torch.rand(3, 4)
    torch.testing.assert_close(SpatialNet(genome)(x), SpatialNet(separated)(x))
    separate_node = max(separated.nodes)
    separated.nodes[separate_node] = replace(separated.nodes[separate_node], activation="tanh")
    old, new = SpatialNet(genome)(x), SpatialNet(separated)(x)
    torch.testing.assert_close(old[:, :2], new[:, :2])
    assert not torch.allclose(old[:, 2:], new[:, 2:])


def test_circuit_split_preserves_nonlocal_connections_and_expanded_cost():
    genome = recipe(shared=False)
    genome.nodes[4] = NodeGene(4, NodeKind.HIDDEN, "tanh")
    genome.placements[4] = genome.placements[2]
    genome.groups = {2: [2, 4]}
    genome.connections[1] = replace(genome.connections[1], in_id=4)
    genome.connections.append(ConnectionGene(2, 4, 0.8, True, 2))
    genome.bindings[2] = Binding(scale=1, shift=-1)
    separated = split_group(genome, [2, 4], InnovationTracker.from_genomes([genome]), 2)
    x = torch.rand(3, 4)
    torch.testing.assert_close(SpatialNet(genome)(x), SpatialNet(separated)(x))
    torch.testing.assert_close(SpatialNet(separated)(x), decode_module(expand_spatial(separated), 4, 4)(x))
    assert separated.complexity() == genome.complexity()


def test_untie_keeps_function_and_allows_independent_weights():
    from versal.evolution.spatial_ops import share

    genome = recipe()
    ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["sin"], "sin")
    untied = share(genome, ctx, rng=random.Random(0), prob=1)
    assert isinstance(untied, SpatialGenome)
    changed = next(key for key in genome.bindings if not untied.bindings[key].shared)
    data = torch.rand(2, 4)
    torch.testing.assert_close(SpatialNet(genome)(data), SpatialNet(untied)(data))
    untied.parameters[changed][2] += 1
    old, new = SpatialNet(genome)(data), SpatialNet(untied)(data)
    torch.testing.assert_close(old[:, [0, 1, 3]], new[:, [0, 1, 3]])
    assert not torch.allclose(old[:, 2], new[:, 2])


@pytest.mark.parametrize(
    "shape,axes,classification",
    [
        ((2, 8), (Axis.CHANNEL, Axis.TIME), True),
        ((3, 1, 2, 2), (Axis.EXAMPLE, Axis.CHANNEL, Axis.HEIGHT, Axis.WIDTH), True),
        ((2, 3), (Axis.HEIGHT, Axis.WIDTH), False),
        ((2, 3), (Axis.TIME, Axis.EXTRA), False),
        ((4,), (Axis.EXTRA,), False),
    ],
)
def test_no_axis_or_rung_gate(shape, axes, classification):
    value = task(shape, axes, classification=classification)
    contract = SpatialContract.from_task(value)
    renamed = Task(TaskMeta(1, value.meta.kind, "xor"), value.support, value.query)
    assert SpatialContract.from_task(renamed) == contract
    genome = spatial_minimal(1, 1, rng=random.Random(1), contract=contract)
    adapter = SpatialTaskAdapter(value, include_query=True)
    module = adapter.decode(genome)
    assert module(adapter.encoded.support_input[0]).shape == (1, contract.output_width)
    assert torch.isfinite(support_loss(module, adapter.encoded))
    assert 0 <= adapter.evaluate(module)["support_accuracy"] <= 1


def test_variable_shapes_masks_and_support_only_binding():
    small, large = task((3, 4), (Axis.HEIGHT, Axis.WIDTH)), task((5, 7), (Axis.HEIGHT, Axis.WIDTH))
    value = Task(small.meta, small.support + large.support, task((8, 9), (Axis.HEIGHT, Axis.WIDTH)).query)
    contract = SpatialContract.from_task(value)
    changed_query = Task(value.meta, value.support, task((99, 4), (Axis.HEIGHT, Axis.WIDTH), classification=True).query)
    assert SpatialContract.from_task(changed_query) == contract
    genome = recipe(value)
    genome.nodes[2] = replace(genome.nodes[2], activation="identity")
    genome.connections = [replace(edge, weight=1.0) for edge in genome.connections]
    adapter = SpatialTaskAdapter(value, include_query=True)
    module = adapter.decode(genome)
    assert support_loss(module, adapter.encoded).item() == 0
    assert adapter.evaluate(module)["query_accuracy"] == 1.0
    source = value.support[0][0]
    mask = torch.zeros_like(source.data, dtype=torch.bool)
    mask[0, 0] = True
    poisoned = source.data.clone()
    poisoned[0, 0] = 1e30
    left = replace(source, mask=mask)
    right = replace(source, data=poisoned, mask=mask)
    torch.testing.assert_close(encode_field(left, (3, 4))[0], encode_field(right, (3, 4))[0])
    assert adapter.evaluate(module)["support_accuracy"] == 1.0
    assert SpatialTaskAdapter(changed_query, include_query=True).evaluate(module)["query_accuracy"] == 0.0


def test_inconsistent_output_shapes_decline_before_initialization(tmp_path):
    from versal.strategy_spatial import SpatialStrategy

    left, right = task((3,)), task((3,))
    source, _ = right.support[0]
    _, target = task((5,)).support[0]
    value = Task(left.meta, [left.support[0], (source, target)], [])
    orchestrator = _orchestrator(tmp_path, table=_policy(evolve=["spatial"]))
    strategy = dict(orchestrator.strategies)["spatial"]
    assert isinstance(strategy, SpatialStrategy)
    decision = strategy.preflight(value, orchestrator._runtime())
    assert not decision.eligible and decision.reason is not None and "dimension binding" in decision.reason


def test_resource_limits_and_unknown_payload_version_are_explicit(tmp_path):
    value = task()
    adapter = SpatialTaskAdapter(value, max_expanded_edges=1)
    with pytest.raises(ValueError, match="execution budget"):
        adapter.decode(recipe(value))
    payload = genome_to_dict(recipe(value))
    payload["spatial"]["version"] = 999
    with pytest.raises(ValueError, match="version"):
        genome_from_dict(payload)


def test_connection_counts_match_enumeration_without_expanding_cartesian_fans(monkeypatch):
    rng = random.Random(4)
    source_shape, target_shape = (3, 4), (2, 5)
    for _ in range(100):
        mapped = rng.choice([False, True])
        binding = Binding(
            scale=rng.randrange(-2, 3),
            shift=rng.randrange(-5, 6),
            fan_in=rng.randrange(5),
            fan_stride=rng.randrange(-2, 3),
            target_start=rng.randrange(5),
            target_stride=rng.randrange(1, 4),
            target_count=rng.randrange(5),
            matrix=tuple(tuple(rng.randrange(-1, 2) for _ in target_shape) for _ in source_shape) if mapped else (),
            offsets=(1, 2) if mapped else (),
        )
        low, high = sorted(rng.sample(range(13), 2))
        first, stop = sorted(rng.sample(range(11), 2))
        expected = sum(int(((source >= low) & (source < high) & (target >= first) & (target < stop)).sum()) for source, target, _ in binding.indices(source_shape, target_shape))
        assert binding.count(source_shape, target_shape, source_interval=(low, high), target_interval=(first, stop)) == expected
    monkeypatch.setattr(Binding, "indices", lambda *_args, **_kwargs: pytest.fail("size counting must not construct edges"))
    assert Binding(fan_in=0).count((1_000_000,), (1_000_000,)) == 1_000_000_000_000


@pytest.mark.parametrize("product", [False, True])
def test_recurrent_recipe_matches_expanded_execution_and_input_gradients(product):
    genome = recipe()
    genome.nodes[2] = replace(genome.nodes[2], aggregation="product" if product else "sum")
    genome.connections.append(ConnectionGene(2, 2, 0.4, True, 2, recurrent=True))
    genome.bindings[2] = Binding(scale=1, shared=True)
    genome.refine_steps = 3
    compact = SpatialNet(genome)
    explicit = decode_module(expand_spatial(genome), 4, 4)
    x = torch.rand(2, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, expected = compact(x), explicit(y)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, y.grad)


def test_growth_does_not_replace_an_existing_circuit_group():
    genome = recipe()
    genome.groups = {4: [2]}  # the next neuron ID and an existing group ID may coincide
    tracker = InnovationTracker.from_genomes([genome])
    ctx = MutationContext(tracker, ["tanh"], "tanh")
    grown = add_node(genome, ctx, rng=random.Random(1), prob=1)
    assert isinstance(grown, SpatialGenome)
    assert [2] in grown.groups.values() and [4] in grown.groups.values()


def test_rebound_recipe_is_reassessed_as_a_seed_with_current_execution_cost(tmp_path):
    small, large = task((4,)), task((8,))
    orchestrator = _orchestrator(tmp_path, table=_policy(evolve=["spatial"], spatial={"pop_size": 4}))
    genome = recipe(small, shared=False)
    payload = genome_to_dict(genome)
    key = orchestrator.library.add(entry_type=MODULE, payload=payload, io=task_io(small), provenance={})
    stored = copy.deepcopy(orchestrator.library.load(key).payload)
    spec = comp_task_spec(large, include_query=False)
    assert orchestrator._quick_assessment(orchestrator.library.load(key), large, spec) is None
    session = SpatialSession(dict(orchestrator.strategies)["spatial"], large, spec, seed=1)
    session.adapter = SpatialTaskAdapter(large)
    seeds = session._seeds(orchestrator._runtime(), InnovationTracker(0))
    assert len(seeds) == 1 and isinstance(seeds[0], SpatialGenome)
    assert seeds[0].contract == SpatialContract.from_task(large)
    assert seeds[0].complexity() == 2 * genome.complexity()
    module = session.adapter.decode(genome)
    assert len(module.weights["0"]) == 8
    assert module.writeback(genome).contract == SpatialContract.from_task(large)
    assert orchestrator.library.load(key).payload == stored


def test_invalid_import_placement_is_rejected_before_training(tmp_path):
    library = ModuleLibrary(tmp_path)
    value = task()
    key = library.add(entry_type=MODULE, payload=genome_to_dict(recipe()), io=task_io(value), provenance={})
    base = spatial_minimal(1, 1, rng=random.Random(0), contract=SpatialContract.from_task(value))
    genome = embed_entry(base, library.load(key), InnovationTracker.from_genomes([base]), random.Random(0))
    genome.placements[genome.macros[0].input_node_ids[0]] = Placement(count=2)
    with pytest.raises(ValueError, match="placement domain"):
        SpatialTaskAdapter(value, library_dir=str(tmp_path)).decode(genome)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mutated_imported_circuits_remain_executable(seed, tmp_path):
    from functools import partial

    from versal.evolution.mutation import MUTATION, MutationPipeline

    library = ModuleLibrary(tmp_path)
    library.add(entry_type=MODULE, payload=genome_to_dict(recipe()), io=task_io(task()), provenance={})
    genome = spatial_minimal(1, 1, rng=random.Random(seed), contract=SpatialContract.from_task(task()))
    operators = [
        "spatial_add_node",
        "spatial_connect",
        "spatial_bind",
        "spatial_group",
        "spatial_repeat",
        "spatial_split",
        "spatial_share",
        "spatial_reuse",
        "spatial_prune",
    ]
    mutation = MutationPipeline([partial(MUTATION.get(name), prob=0.4) for name in operators])
    ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh", "sin"], "tanh", library=library)
    rng = random.Random(seed)
    adapter = SpatialTaskAdapter(task(), library_dir=str(tmp_path), max_expanded_edges=1000)
    for _ in range(60):
        candidate = mutation(genome, ctx, rng=rng)
        assert isinstance(candidate, SpatialGenome)
        candidate = candidate.repair()
        try:
            module = adapter.decode(candidate)
        except ValueError:
            continue  # resource/depth limits are checked at decode, before training
        output = module(torch.rand(2, 4))
        assert output.shape == (2, 4)
        if output.requires_grad:
            output.square().mean().backward()
        genome = candidate


def test_bindings_and_sharing_are_topology_but_parameters_are_not():
    genome = recipe()
    changed = genome.clone()
    changed.parameters = {0: [9.0]}
    assert structural_fingerprint(MODULE, genome_to_dict(genome)) == structural_fingerprint(MODULE, genome_to_dict(changed))
    original = topology_record(MODULE, genome_to_dict(genome))
    changed.bindings[0] = replace(changed.bindings[0], shift=1)
    assert not same_topology(original, topology_record(MODULE, genome_to_dict(changed)))
    changed.bindings[0] = replace(genome.bindings[0], shared=False)
    assert not same_topology(original, topology_record(MODULE, genome_to_dict(changed)))


def test_spatial_reused_in_direct_macros_compositions_and_routes(tmp_path):
    from versal.evolution.composition import AssemblyContext, CompEdgeGene, CompNodeGene, CompNodeKind, CompositionGenome, IndexRun, PortMap, assemble, comp_to_dict

    library = ModuleLibrary(tmp_path / "lib")
    value, genome = task(), recipe()
    key = library.add(entry_type=MODULE, payload=genome_to_dict(genome), io=task_io(value), provenance={})
    entry = library.load(key)
    summary = library.summary(key)
    assert summary is not None and summary["representation"] == "spatial"
    data = torch.rand(3, 4)
    expected = SpatialNet(genome)(data)
    torch.testing.assert_close(decode_module(graft(entry, InnovationTracker(0)), 4, 4)(data), expected)
    nodes = {i: NodeGene(i, NodeKind.INPUT if i < 4 else NodeKind.OUTPUT, "identity") for i in range(8)}
    macro = Genome(nodes, macros=[MacroGene(f"library:{key}", tuple(range(4)), tuple(range(4, 8)), 0)])
    torch.testing.assert_close(decode_module(macro, 4, 4, macro_resolver=macro_resolver(library))(data), expected)
    comp = CompositionGenome(
        {
            0: CompNodeGene(0, CompNodeKind.INPUT, "input", 0, 4),
            1: CompNodeGene(1, CompNodeKind.MODULE, f"library:{key}", 4, 4, trainable=False),
            2: CompNodeGene(2, CompNodeKind.OUTPUT, "output", 4, 0),
        },
        [CompEdgeGene(i, i + 1, True, i, (), port_map=PortMap((IndexRun(0, 0, 4),))) for i in range(2)],
    )
    composed = assemble(comp, AssemblyContext({"input": range(4)}, library=library), 4)
    torch.testing.assert_close(composed(data), expected)
    vertex = build_vertex(entry, library)
    assert vertex is not None and vertex.module is not None
    assert not any(p.requires_grad for p in vertex.module.parameters())
    torch.testing.assert_close(vertex.module(data), expected)
    composition_key = library.add(entry_type=COMPOSITION, payload=comp_to_dict(comp), io=entry.io, provenance={})
    for imported in [entry, library.load(composition_key)]:
        shell = spatial_minimal(1, 1, rng=random.Random(0), contract=genome.contract)
        shell.connections, shell.bindings = [], {}
        embedded = embed_entry(shell, imported, InnovationTracker.from_genomes([shell]), random.Random(0), exact=True)
        torch.testing.assert_close(SpatialNet(embedded, library_dir=str(library.root))(data), expected)
        assert expanded_payload_complexity(MODULE, genome_to_dict(embedded), library) > genome.complexity()


def test_spatial_population_resume_and_growth(tmp_path, xor_task):
    orchestrator = _orchestrator(tmp_path, table=_policy(evolve=["spatial"], spatial={"pop_size": 4, "train": {"kind": "gradient", "steps": 2}}))
    strategy = dict(orchestrator.strategies)["spatial"]
    runtime, spec = orchestrator._runtime(), comp_task_spec(xor_task, include_query=False)
    session = SpatialSession(strategy, xor_task, spec, seed=7)
    assert session.ready(runtime)
    session.advance(runtime)
    saved = copy.deepcopy(session.state_dict())
    session.advance(runtime)
    restored = SpatialSession(strategy, xor_task, spec, seed=7, saved=saved)
    restored.advance(runtime)
    assert restored.state_dict() == session.state_dict()
    genome = session.state.population[0].genome
    ctx = MutationContext(InnovationTracker.from_genomes([genome]), ["tanh", "sin"], "tanh")
    grown = add_node(genome, ctx, rng=random.Random(1), prob=1)
    assert len(grown.hidden_ids) == len(genome.hidden_ids) + 1
