"""Phase-3 hard weight sharing: ConnectionGene.tie_group, the shared-parameter overlay in the
substrate (one trainable value per group, gradient summed across every member edge), the
add_shared_motif(tied=true) stamping, and untie_motif_weights function preservation. Untied is
byte-identical everywhere (the whole existing suite pins that side)."""

import random

import torch

from ardevo.dataset.icarus import Level0Encoder, Task
from ardevo.evaluation import encode, input_width, output_features
from ardevo.evolution.evolver import TaskAdapter
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind, genome_from_dict, genome_to_dict
from ardevo.evolution.init import stamp_input_coordinates
from ardevo.evolution.mutation import MutationContext, add_shared_motif, untie_motif_weights
from ardevo.evolution.train import gradient
from ardevo.substrate import decode


def _tied_pair_genome(weight: float = 0.7) -> Genome:
    """Two inputs, one tanh hidden fed by BOTH inputs through ONE shared weight, one output."""
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.HIDDEN, "tanh"),
        4: NodeGene(4, NodeKind.OUTPUT, "identity"),
    }
    connections = [
        ConnectionGene(0, 3, weight, True, 0, tie_group=7),
        ConnectionGene(1, 3, weight, True, 1, tie_group=7),
        ConnectionGene(3, 4, 1.0, True, 2),
    ]
    return Genome(nodes=nodes, connections=connections)


def _ctx(*genomes: Genome) -> MutationContext:
    return MutationContext(innovations=InnovationTracker.from_genomes(list(genomes)), activations=["tanh"], default_activation="tanh")


def test_tied_edges_share_one_parameter_forward() -> None:
    genome = _tied_pair_genome(weight=0.7)
    net = decode(genome, 2, 1)
    assert net.tie_values is not None and net.tie_values.numel() == 1
    x = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    with torch.no_grad():
        out = net(x)
    expected = torch.tanh(0.7 * (x[:, 0] + x[:, 1])).unsqueeze(1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_tied_gradient_accumulates_and_members_stay_in_sync(xor_task: Task) -> None:
    encoder = Level0Encoder(max_flat_dim=2)
    encoded = encode(xor_task, encoder)
    adapter = TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded))
    genome = _tied_pair_genome(weight=0.3)
    trained, module = gradient(genome, decode(genome, 2, 1), adapter.encoded, rng=random.Random(0), steps=5, lr=0.05)
    exported = module.export_weights()
    assert exported[(0, 3, False)] == exported[(1, 3, False)]  # one parameter, one value
    weights = {(conn.in_id, conn.out_id): conn.weight for conn in trained.connections}
    assert weights[(0, 3)] == weights[(1, 3)]  # writeback keeps every member gene in sync
    assert weights[(0, 3)] != 0.3  # and it actually trained


def test_untied_twin_diverges_where_tied_twin_cannot() -> None:
    # y = x0 (input-ASYMMETRIC on purpose: XOR is symmetric under input swap, so equal starting
    # weights would receive equal gradients even untied and the divergence assertion would be void).
    from ardevo.dataset.icarus import Axis, Field, TaskKind, TaskMeta, ValueType

    def field(values: list[float]) -> Field:
        return Field(torch.tensor(values, dtype=torch.float32), (Axis.EXTRA,), ValueType.BINARY, None, None, None)

    pairs = [(field([float(x0), float(x1)]), field([float(x0)])) for x0 in (0, 1) for x1 in (0, 1)]
    task = Task(meta=TaskMeta(rung=0, kind=TaskKind.MAP, name="copy_x0", fixed_split=True), support=pairs, query=pairs)
    encoder = Level0Encoder(max_flat_dim=2)
    encoded = encode(task, encoder)
    adapter = TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded))
    tied = _tied_pair_genome(weight=0.3)
    untied = _tied_pair_genome(weight=0.3)
    untied.connections = [ConnectionGene(conn.in_id, conn.out_id, conn.weight, conn.enabled, conn.innovation) for conn in untied.connections]
    tied_trained, _module = gradient(tied, decode(tied, 2, 1), adapter.encoded, rng=random.Random(0), steps=10, lr=0.05)
    untied_trained, _module = gradient(untied, decode(untied, 2, 1), adapter.encoded, rng=random.Random(0), steps=10, lr=0.05)
    tied_weights = {(conn.in_id, conn.out_id): conn.weight for conn in tied_trained.connections}
    untied_weights = {(conn.in_id, conn.out_id): conn.weight for conn in untied_trained.connections}
    assert tied_weights[(0, 3)] == tied_weights[(1, 3)]
    assert untied_weights[(0, 3)] != untied_weights[(1, 3)]  # XOR rows are asymmetric per input


def test_tie_group_serde_round_trip_and_untied_dicts_unchanged() -> None:
    genome = _tied_pair_genome()
    payload = genome_to_dict(genome)
    assert payload["connections"][0]["tie"] == 7
    assert "tie" not in payload["connections"][2]  # untied gene: key absent, old payloads identical
    restored = genome_from_dict(payload)
    assert restored.connections[0].tie_group == 7 and restored.connections[2].tie_group is None


def test_tied_nets_never_enter_the_batch_program() -> None:
    genome = _tied_pair_genome()
    net, _columns = decode(genome, 2, 1).core()
    assert net is None


def test_add_shared_motif_tied_stamps_shared_groups() -> None:
    rng = random.Random(4)
    from ardevo.evolution.init import minimal

    genome = stamp_input_coordinates(minimal(9, 1, rng=rng), (3, 3))
    ctx = _ctx(genome)
    seeded = genome.clone()
    template_id = ctx.innovations.new_node_id()
    seeded.nodes[template_id] = NodeGene(template_id, NodeKind.HIDDEN, "tanh", coordinate=(1.0, 1.0))
    for source in (0, 1, 3):
        seeded.connections.append(ConnectionGene(source, template_id, 0.5, True, ctx.innovations.innovation(source, template_id)))
    seeded.connections.append(ConnectionGene(template_id, seeded.output_ids[0], 0.8, True, ctx.innovations.innovation(template_id, seeded.output_ids[0])))

    child = add_shared_motif(seeded, ctx, rng=random.Random(1), prob=1.0, copies=2, tied=True)
    groups = {conn.tie_group for conn in child.connections if conn.tie_group is not None}
    assert len(groups) == 4  # 3 fan-in template edges + 1 readout edge, each one shared group
    members_per_group = {group: sum(1 for conn in child.connections if conn.tie_group == group) for group in groups}
    assert all(count >= 2 for count in members_per_group.values())  # template + at least one copy share
    decode(child, 9, 1)(torch.rand(4, 9))  # decodes and runs


def test_untie_is_function_preserving() -> None:
    genome = _tied_pair_genome(weight=0.9)
    child = untie_motif_weights(genome, _ctx(genome), rng=random.Random(0), prob=1.0)
    assert all(conn.tie_group is None for conn in child.connections)
    x = torch.rand(8, 2)
    with torch.no_grad():
        before = decode(genome, 2, 1)(x)
        after = decode(child, 2, 1)(x)
    assert torch.allclose(before, after, atol=1e-6)


def test_add_shared_motif_untied_path_is_byte_identical_to_the_historical_operator() -> None:
    from ardevo.evolution.init import minimal

    genome = stamp_input_coordinates(minimal(9, 1, rng=random.Random(2)), (3, 3))
    ctx_a = _ctx(genome)
    seeded = genome.clone()
    template_id = ctx_a.innovations.new_node_id()
    seeded.nodes[template_id] = NodeGene(template_id, NodeKind.HIDDEN, "tanh", coordinate=(1.0, 1.0))
    for source in (0, 1, 3):
        seeded.connections.append(ConnectionGene(source, template_id, 0.5, True, ctx_a.innovations.innovation(source, template_id)))
    seeded.connections.append(ConnectionGene(template_id, seeded.output_ids[0], 0.8, True, ctx_a.innovations.innovation(template_id, seeded.output_ids[0])))

    default_child = add_shared_motif(seeded, ctx_a, rng=random.Random(9), prob=1.0, copies=2)
    explicit_child = add_shared_motif(seeded, _restamped_ctx(seeded), rng=random.Random(9), prob=1.0, copies=2, tied=False)
    assert genome_to_dict(default_child) == genome_to_dict(explicit_child)
    assert all(conn.tie_group is None for conn in default_child.connections)


def _restamped_ctx(genome: Genome) -> MutationContext:
    return MutationContext(innovations=InnovationTracker.from_genomes([genome]), activations=["tanh"], default_activation="tanh")


def test_census_flags_tied_motifs() -> None:
    from ardevo.motifs import TIED_EDGE, diversity_class, module_motif_graph

    genome = _tied_pair_genome()
    labels, edges = module_motif_graph(genome_to_dict(genome))
    assert edges[(0, 3)] & TIED_EDGE and edges[(1, 3)] & TIED_EDGE
    assert not edges[(3, 4)] & TIED_EDGE
    from ardevo.motifs import canonical_form

    graph = canonical_form([labels[0], labels[3]], [(0, 1, edges[(0, 3)])])
    assert "tied" in diversity_class(graph)
