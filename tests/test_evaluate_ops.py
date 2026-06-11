"""Evaluate-stage operators: standard delegation, weight sampling, hybrid merge, registry wiring."""

from ardevo.evolution.evaluate import DEFAULT_WEIGHT_SAMPLES, hybrid, standard, weight_samples
from ardevo.evolution.evolver import TaskAdapter
from ardevo.evolution.fitness import FITNESS
from ardevo.evolution.genome import Genome
from ardevo.evolution.registry import build_evolver

_ROBUSTNESS_KEYS = {
    "mean_sample_accuracy",
    "max_sample_accuracy",
    "min_sample_accuracy",
    "mean_sample_loss",
    "best_sample_weight",
    "weight_robustness",
}
_STANDARD_KEYS = {"support_accuracy", "support_loss", "query_accuracy", "query_loss"}


def test_standard_matches_adapter_evaluate(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    module = xor_adapter.decode(solving_genome)
    assert standard(solving_genome, module, xor_adapter) == xor_adapter.evaluate(module)


def test_weight_samples_emits_all_keys_and_restores_weights(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    module = xor_adapter.decode(solving_genome)
    before = module.export_weights()
    metrics = weight_samples(solving_genome, module, xor_adapter)
    assert _ROBUSTNESS_KEYS <= set(metrics)
    assert _STANDARD_KEYS <= set(metrics)
    assert metrics["best_sample_weight"] in DEFAULT_WEIGHT_SAMPLES
    assert 0.0 <= metrics["min_sample_accuracy"] <= metrics["mean_sample_accuracy"] <= metrics["max_sample_accuracy"] <= 1.0
    assert module.export_weights() == before


def test_weight_samples_standard_keys_come_from_best_sample(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    module = xor_adapter.decode(solving_genome)
    metrics = weight_samples(solving_genome, module, xor_adapter)
    # The reported query accuracy must equal the max over samples (best-sample reporting).
    assert metrics["query_accuracy"] == metrics["max_sample_accuracy"]


def test_hybrid_is_superset_of_standard(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    module = xor_adapter.decode(solving_genome)
    trained = xor_adapter.evaluate(module)
    metrics = hybrid(solving_genome, module, xor_adapter)
    assert _ROBUSTNESS_KEYS <= set(metrics)
    for key in _STANDARD_KEYS:
        assert metrics[key] == trained[key]


def _config(evaluate_kind: str | None) -> dict:
    from typing import Any

    evolution: dict[str, Any] = {
        "pop_size": 4,
        "init": {"kind": "minimal"},
        "selection": {"kind": "tournament", "tournament_size": 2},
        "crossover": {"kind": "none"},
        "mutation": {"operators": []},
        "train": {"kind": "none"},
    }
    if evaluate_kind is not None:
        evolution["evaluate"] = {"kind": evaluate_kind, "samples": [-1.0, 1.0]}
    return {"evolution": evolution, "fitness": {"components": ["query_accuracy"]}}


def test_build_evolver_defaults_to_standard_evaluate(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    evolver = build_evolver(_config(None))
    module = xor_adapter.decode(solving_genome)
    assert evolver.evaluate_op(solving_genome, module, xor_adapter) == xor_adapter.evaluate(module)


def test_build_evolver_resolves_weight_samples(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    evolver = build_evolver(_config("weight_samples"))
    module = xor_adapter.decode(solving_genome)
    metrics = evolver.evaluate_op(solving_genome, module, xor_adapter)
    assert metrics["best_sample_weight"] in (-1.0, 1.0)


def test_robustness_fitness_components_read_metrics(solving_genome: Genome) -> None:
    metrics = {"mean_sample_accuracy": 0.7, "max_sample_accuracy": 0.9, "weight_robustness": 0.55, "mean_sample_loss": 0.4}
    assert FITNESS.get("mean_sample_accuracy")(solving_genome, metrics) == 0.7
    assert FITNESS.get("max_sample_accuracy")(solving_genome, metrics) == 0.9
    assert FITNESS.get("weight_robustness")(solving_genome, metrics) == 0.55
    assert FITNESS.get("negative_mean_sample_loss")(solving_genome, metrics) == -0.4
    assert FITNESS.get("weight_robustness")(solving_genome, {}) == 0.0
