import copy
import functools
from typing import Any

import pytest

from ardevo.evolution.registry import build_evolver

_CONFIG: dict[str, Any] = {
    "seed": 0,
    "substrate": {"available_activations": ["tanh", "relu"], "default_activation": "tanh"},
    "evolution": {
        "pop_size": 16,
        "elitism": 1,
        "init": {"kind": "minimal"},
        "selection": {"kind": "tournament", "tournament_size": 3},
        "crossover": {"kind": "none", "rate": 0.0},
        "mutation": {"operators": ["perturb_weights", "add_node"], "perturb_weights_prob": 0.5, "add_node_prob": 0.1},
        "train": {"kind": "gradient", "steps": 5, "lr": 0.01, "writeback": True},
    },
    "fitness": {"components": ["query_accuracy", "complexity_penalty"], "w_query_accuracy": 1.0, "w_complexity_penalty": 0.01},
}


def _op_name(op: object) -> str:
    target = op.func if isinstance(op, functools.partial) else op
    return getattr(target, "__name__", repr(target))


def test_build_evolver_resolves_all_stages():
    evolver = build_evolver(_CONFIG)
    assert evolver.pop_size == 16
    assert _op_name(evolver.selection_op) == "tournament"
    assert _op_name(evolver.crossover_op) == "asexual"
    assert _op_name(evolver.train_op) == "gradient"
    assert [name for name, _op in evolver.mutation.operators] == ["perturb_weights", "add_node"]
    assert [_op_name(component) for component, _weight in evolver.fitness.components] == ["query_accuracy", "complexity_penalty"]


def test_mutation_params_bound_by_prefix():
    evolver = build_evolver(_CONFIG)
    name, perturb = evolver.mutation.operators[0]
    assert name == "perturb_weights"
    assert isinstance(perturb, functools.partial)
    assert perturb.keywords == {"prob": 0.5}
    assert evolver.mutation.base_rates == {"perturb_weights": 0.5, "add_node": 0.1}


def test_unknown_operator_raises():
    bad = copy.deepcopy(_CONFIG)
    bad["evolution"]["selection"] = {"kind": "does_not_exist"}
    with pytest.raises(KeyError, match="selection"):
        build_evolver(bad)
