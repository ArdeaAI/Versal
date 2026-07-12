"""Phase-2 gradient-proposed growth: the NeST dormant-edge scores (growth.py + the train ops'
score_candidates flag), the hinted mutators, split_node/add_hinted_node function preservation
(children compute the parent's function at birth, so selection never culls fresh structure), and
successive-halving assessment. Off means byte-identical: hints never serialize, flags default off."""

import random

import torch

from ardevo.evolution.evolver import TaskAdapter
from ardevo.evolution.genome import ConnectionGene, Genome, InnovationTracker, NodeGene, NodeKind, genome_from_dict, genome_to_dict
from ardevo.evolution.growth import node_scores
from ardevo.evolution.init import minimal
from ardevo.evolution.mutation import MutationContext, add_hinted_connection, add_hinted_node, split_node
from ardevo.evolution.registry import build_evolver
from ardevo.evolution.train import gradient
from ardevo.substrate import decode, decode_refine


def _ctx(*genomes: Genome) -> MutationContext:
    return MutationContext(innovations=InnovationTracker.from_genomes(list(genomes)), activations=["tanh"], default_activation="tanh")


def test_node_scores_cover_every_source_and_computed_node(xor_adapter: TaskAdapter) -> None:
    genome = minimal(2, 1, rng=random.Random(0))
    module = decode(genome, 2, 1)
    scores = node_scores(module, xor_adapter.encoded)
    assert scores is not None
    source_scores, target_scores = scores
    assert set(source_scores) == {0, 1, 2, 3}  # every node row (the mutators filter to legal sources)
    assert set(target_scores) == {3}  # the single computed (output) column
    assert all(value >= 0.0 for value in source_scores.values())
    assert sum(source_scores.values()) > 0.0


def test_node_scores_none_for_non_plain_substrates(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    refined = solving_genome.clone()
    refined.refine_steps = 2
    assert node_scores(decode_refine(refined, 2, 1), xor_adapter.encoded) is None


def test_score_candidates_attaches_hints_without_touching_serde(xor_adapter: TaskAdapter) -> None:
    genome = minimal(2, 1, rng=random.Random(0))
    hinted, _module = gradient(genome.clone(), decode(genome, 2, 1), xor_adapter.encoded, rng=random.Random(0), steps=3, score_candidates=True)
    plain, _module = gradient(genome.clone(), decode(genome, 2, 1), xor_adapter.encoded, rng=random.Random(0), steps=3, score_candidates=False)
    assert hinted.growth_hints and set(hinted.growth_hints) == {"source", "target"}
    assert plain.growth_hints is None
    assert genome_to_dict(hinted) == genome_to_dict(plain)  # identical training, hints never serialize
    assert genome_from_dict(genome_to_dict(hinted)).growth_hints is None
    assert hinted.clone().growth_hints is hinted.growth_hints  # clones carry hints in memory


def test_add_hinted_connection_lands_where_the_gradient_points() -> None:
    nodes = {
        0: NodeGene(0, NodeKind.INPUT, "identity"),
        1: NodeGene(1, NodeKind.INPUT, "identity"),
        2: NodeGene(2, NodeKind.BIAS, "identity"),
        3: NodeGene(3, NodeKind.OUTPUT, "identity"),
    }
    genome = Genome(nodes=nodes, connections=[ConnectionGene(1, 3, 0.5, True, 0)])
    genome.growth_hints = {"source": {0: 100.0, 1: 0.0, 2: 0.0}, "target": {3: 1.0}}
    child = add_hinted_connection(genome, _ctx(genome), rng=random.Random(0), prob=1.0)
    added = [conn for conn in child.connections if (conn.in_id, conn.out_id) != (1, 3)]
    assert len(added) == 1 and added[0].in_id == 0 and added[0].out_id == 3


def test_add_hinted_connection_degrades_to_uniform_without_hints() -> None:
    genome = minimal(2, 1, rng=random.Random(0))
    hidden_free = genome.clone()
    del hidden_free.connections[0]  # open one legal pair
    child = add_hinted_connection(hidden_free, _ctx(hidden_free), rng=random.Random(3), prob=1.0)
    assert len(child.connections) == len(hidden_free.connections) + 1


def test_add_hinted_node_is_function_preserving(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    child = add_hinted_node(solving_genome, _ctx(solving_genome), rng=random.Random(1), prob=1.0, fan_in=2)
    assert len(child.hidden_ids) == len(solving_genome.hidden_ids) + 1
    x = torch.rand(16, 2)
    with torch.no_grad():
        before = decode(solving_genome, 2, 1)(x)
        after = decode(child, 2, 1)(x)
    assert torch.allclose(before, after, atol=1e-6)  # zero fan-out: same function at birth


def test_split_node_is_function_preserving(solving_genome: Genome) -> None:
    child = split_node(solving_genome, _ctx(solving_genome), rng=random.Random(2), prob=1.0)
    assert len(child.hidden_ids) == len(solving_genome.hidden_ids) + 1
    x = torch.rand(16, 2)
    with torch.no_grad():
        before = decode(solving_genome, 2, 1)(x)
        after = decode(child, 2, 1)(x)
    assert torch.allclose(before, after, atol=1e-6)  # Net2Wider: verbatim in-edges, halved out-edges


def test_split_node_skips_recurrent_and_product_entangled_nodes(solving_genome: Genome) -> None:
    from dataclasses import replace

    recurrent = solving_genome.clone()
    recurrent.connections.append(ConnectionGene(3, 3, 0.5, True, 99, recurrent=True))
    product = solving_genome.clone()
    product.nodes[5] = replace(product.nodes[5], aggregation="product")
    for genome in (recurrent, product):
        child = split_node(genome, _ctx(genome), rng=random.Random(0), prob=1.0)
        splittable_before = set(genome.hidden_ids)
        if genome is recurrent:
            assert 3 not in {node_id for node_id in child.hidden_ids if node_id not in splittable_before} or len(child.hidden_ids) == len(genome.hidden_ids) + 1
        # the recurrent-tied node 3 must never be the split source; the product-fed hidden nodes never split at all
    only_recurrent = solving_genome.clone()
    only_recurrent.connections.append(ConnectionGene(3, 3, 0.5, True, 99, recurrent=True))
    only_recurrent.connections.append(ConnectionGene(4, 4, 0.5, True, 100, recurrent=True))
    assert split_node(only_recurrent, _ctx(only_recurrent), rng=random.Random(0), prob=1.0) is only_recurrent


def test_halving_stages_cull_half_between_stages(xor_adapter: TaskAdapter) -> None:
    config = {
        "evolution": {
            "pop_size": 8,
            "elitism": 1,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "none"},
            "mutation": {"operators": []},
            "train": {"kind": "gradient", "steps": 8, "lr": 0.01},
            "evaluate": {"kind": "standard"},
            "speciation": {"kind": "none"},
            "halving_stages": [0.5],
            "halving_keep": 0.5,
        },
        "fitness": {"components": ["support_accuracy"]},
        "substrate": {},
    }
    evolver = build_evolver(config)
    assert evolver.halving_stages == [0.5]
    seen_steps: list[int] = []
    from functools import partial
    from typing import cast

    original = cast(partial, evolver.train_op)

    def counting(genome, module, encoded, *, rng, steps=8, **kwargs):
        seen_steps.append(steps)
        return original.func(genome, module, encoded, rng=rng, steps=steps, **{key: value for key, value in original.keywords.items() if key != "steps"})

    setattr(counting, "keywords", dict(original.keywords))  # _assess_staged reads the configured total here
    evolver.train_op = counting
    state = evolver.seed_state(xor_adapter, random.Random(0))
    assert len(state.population) == 8
    assert seen_steps == [4] * 8 + [4] * 4  # everyone trains half the budget; the best half finishes it
    assert all(item.fitness == item.fitness for item in state.population)  # finite, no NaN
