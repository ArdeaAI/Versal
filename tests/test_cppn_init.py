"""cppn init: generative starting points (gate-E fork, compile-time use of the CPPN).

Each population member gets its own random pair-query generator f(source_coord, target_coord) ->
(weight, expression); the expression gate yields sparse, spatially-patterned connectivity instead
of `minimal`'s dense bipartite (the rungs 11-14 wall lever). The generator is compiled away at
init: the search proceeds on ordinary explicit genomes.
"""

import random

from ardevo.evolution.init import cppn_seed, minimal
from ardevo.evolution.registry import build_evolver


def _init(n_inputs: int = 2, n_outputs: int = 1, seed: int = 0, **params):
    return cppn_seed(n_inputs, n_outputs, rng=random.Random(seed), **params)


def test_generates_valid_decodable_genome(xor_adapter) -> None:
    genome = _init(hidden=8)
    assert len(genome.input_ids) == 2 and len(genome.output_ids) == 1
    assert len(genome.hidden_ids) == 8
    module = xor_adapter.decode(genome)
    assert module(xor_adapter.encoded.support_input[0]).shape == (4, 1)


def test_deterministic_per_seed_and_diverse_across_members() -> None:
    first, second = _init(seed=3), _init(seed=3)
    assert first.connections == second.connections
    other = _init(seed=4)
    assert first.connections != other.connections  # per-member diversity


def test_innovation_ids_align_across_members() -> None:
    # NEAT crossover aligns genes by innovation, so the SAME (in, out) edge in two independently
    # generated members must carry the SAME innovation number (pair-derived, not sequential).
    first, second = _init(seed=0, hidden=8), _init(seed=1, hidden=8)
    by_pair_first = {(c.in_id, c.out_id): c.innovation for c in first.connections}
    by_pair_second = {(c.in_id, c.out_id): c.innovation for c in second.connections}
    shared = set(by_pair_first) & set(by_pair_second)
    assert shared  # random generators overlap somewhere
    assert all(by_pair_first[pair] == by_pair_second[pair] for pair in shared)


def test_density_is_an_exact_sparsity_dial() -> None:
    n_inputs, hidden = 50, 16
    dense_bipartite = (n_inputs + 1) * 4  # what `minimal` would write for 4 outputs
    open_gate = _init(n_inputs, 4, seed=0, hidden=hidden, density=1.0)
    tight_gate = _init(n_inputs, 4, seed=0, hidden=hidden, density=0.1)
    assert len(tight_gate.connections) < len(open_gate.connections)
    assert len(tight_gate.connections) < dense_bipartite  # the rungs 11-14 point: sparser than minimal
    generated = len(tight_gate.connections) - hidden - 4  # minus the free bias floor edges
    assert generated == round(n_inputs * hidden * 0.1) + round(hidden * 4 * 0.1)  # the quantile keeps EXACTLY density * pairs


def test_outputs_are_never_dead() -> None:
    genome = _init(6, 3, seed=5, density=0.01)  # gate nearly closed
    bias_id = genome.bias_ids[0]
    for output_id in genome.output_ids:
        assert any(c.out_id == output_id and c.in_id == bias_id for c in genome.connections)  # the bias floor


def test_hidden_nodes_carry_coordinates_and_default_activation() -> None:
    genome = _init(hidden=4)
    for node_id in genome.hidden_ids:
        node = genome.nodes[node_id]
        assert node.coordinate is not None and len(node.coordinate) == 1  # the index continuum: geometry ops go live
        assert node.activation == "tanh"  # default_activation when hidden_activations is unset
    diverse = _init(hidden=8, seed=2, hidden_activations=["sin", "gaussian"])
    assert {diverse.nodes[node_id].activation for node_id in diverse.hidden_ids} <= {"sin", "gaussian"}


def test_registry_resolves_and_seed_state_runs(xor_adapter) -> None:
    config = {
        "seed": 0,
        "substrate": {"available_activations": ["tanh", "sin"], "default_activation": "tanh"},
        "evolution": {
            "pop_size": 6,
            "elitism": 1,
            "init": {"kind": "cppn", "hidden": 4, "density": 0.4},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "neat", "rate": 0.2},
            "mutation": {"operators": ["add_connection"], "add_connection_prob": 0.2},
            "train": {"kind": "gradient", "steps": 5, "lr": 0.03, "writeback": True},
            "speciation": {"kind": "none"},
        },
        "fitness": {"components": ["support_accuracy"]},
    }
    evolver = build_evolver(config)
    state = evolver.seed_state(xor_adapter, random.Random(0))
    evolver.advance(state, xor_adapter)
    assert len(state.population) == 6
    assert all(item.fitness > -1e9 for item in state.population)
    # Structural diversity from generation zero: members differ, unlike minimal's identical seeds.
    edge_sets = {tuple(sorted((c.in_id, c.out_id) for c in item.genome.connections)) for item in state.population}
    assert len(edge_sets) > 1


def test_single_input_task_does_not_divide_by_zero() -> None:
    genome = _init(1, 4, seed=0, hidden=4)
    assert len(genome.input_ids) == 1
    minimal_reference = minimal(1, 4, rng=random.Random(0))
    assert len(genome.output_ids) == len(minimal_reference.output_ids)
