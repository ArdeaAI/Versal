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


def test_batched_samples_parity_with_serial(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    """T2: the stacked fast path must reproduce the serial fill/restore numbers."""
    module = xor_adapter.decode(solving_genome)
    fast = weight_samples(solving_genome, module, xor_adapter, batched_samples=True)
    slow = weight_samples(solving_genome, xor_adapter.decode(solving_genome), xor_adapter, batched_samples=False)
    for key in ("support_accuracy", "query_accuracy", "mean_sample_accuracy", "max_sample_accuracy", "min_sample_accuracy", "best_sample_weight"):
        assert fast[key] == slow[key], key
    for key in ("support_loss", "query_loss", "mean_sample_loss", "weight_robustness"):
        assert abs(fast[key] - slow[key]) < 1e-6, key


def test_batched_samples_auto_gates_on_node_count(xor_adapter: TaskAdapter, solving_genome: Genome, monkeypatch) -> None:
    """`"auto"` uses the stacked path only at the measured break-even node count and above."""
    from ardevo.evolution import evaluate as evaluate_module

    stacked_calls: list[int] = []
    original = evaluate_module._stacked_sample_metrics
    monkeypatch.setattr(evaluate_module, "_stacked_sample_metrics", lambda *args, **kwargs: stacked_calls.append(1) or original(*args, **kwargs))

    module = xor_adapter.decode(solving_genome)
    serial = weight_samples(solving_genome, module, xor_adapter, batched_samples="auto")
    assert not stacked_calls  # tiny net: below the threshold, exact serial path
    assert serial == weight_samples(solving_genome, xor_adapter.decode(solving_genome), xor_adapter, batched_samples=False)

    monkeypatch.setattr(evaluate_module, "STACKED_AUTO_MIN_NODES", 1)
    weight_samples(solving_genome, module, xor_adapter, batched_samples="auto")
    assert stacked_calls  # threshold crossed: stacked path engaged


def test_batched_samples_leave_module_weights_untouched(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    module = xor_adapter.decode(solving_genome)
    before = module.export_weights()
    weight_samples(solving_genome, module, xor_adapter, batched_samples=True)
    assert module.export_weights() == before


def test_batched_samples_head_columns_path(xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    """The column-sliced-head path (`module.core()` returning columns, evaluate.py/train.py's
    generic seam): column selection must survive the stacked forward."""
    from dataclasses import dataclass

    import torch

    from ardevo.dataset.icarus import EncodedTask, Level0Encoder
    from ardevo.evaluation import evaluate
    from ardevo.evolution.genome import ConnectionGene, NodeGene, NodeKind
    from ardevo.substrate import GraphNet, SubstrateModule, decode

    class _ColumnSlicedNet(SubstrateModule):
        """Minimal head wrapper: the inner net is a normal trainable submodule, this only selects
        output columns, so `core()` reports (inner, columns) to the stacked fast path."""

        def __init__(self, inner: GraphNet, columns: torch.Tensor) -> None:
            super().__init__()
            self.inner = inner
            self.columns: torch.Tensor = columns

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x).index_select(1, self.columns)

        @property
        def has_edges(self) -> bool:
            return self.inner.has_edges

        def export_weights(self) -> dict[tuple[int, int, bool], float]:
            return self.inner.export_weights()

        def core(self) -> tuple[GraphNet | None, torch.Tensor | None]:
            inner, _columns = self.inner.core()
            return (inner, self.columns) if inner is not None else (None, None)

    two_headed = solving_genome.clone()
    two_headed.nodes[6] = NodeGene(6, NodeKind.OUTPUT, "identity")
    two_headed.connections = list(two_headed.connections) + [ConnectionGene(2, 6, 0.3, True, 99)]

    @dataclass
    class _HeadAdapter:
        encoded: EncodedTask
        encoder: Level0Encoder

        def evaluate(self, module) -> dict[str, float]:
            return evaluate(module, self.encoded, self.encoder)

    adapter = _HeadAdapter(xor_adapter.encoded, xor_adapter.encoder)
    fast_module = _ColumnSlicedNet(decode(two_headed, 2, 2), torch.tensor([0]))
    slow_module = _ColumnSlicedNet(decode(two_headed, 2, 2), torch.tensor([0]))
    fast = weight_samples(two_headed, fast_module, adapter, batched_samples=True)
    slow = weight_samples(two_headed, slow_module, adapter, batched_samples=False)
    for key in ("mean_sample_accuracy", "max_sample_accuracy", "best_sample_weight"):
        assert fast[key] == slow[key], key
    assert abs(fast["weight_robustness"] - slow["weight_robustness"]) < 1e-6


def test_non_batchable_modules_fall_back_to_serial(xor_adapter: TaskAdapter) -> None:
    from tests.test_aggregation import _product_xor_genome

    genome = _product_xor_genome()
    module = xor_adapter.decode(genome)  # product node: core() is (None, None)
    metrics = weight_samples(genome, module, xor_adapter, batched_samples=True)
    assert _ROBUSTNESS_KEYS <= set(metrics)


def test_frozen_parameters_are_never_filled_during_sampling(tmp_path, xor_adapter: TaskAdapter, solving_genome: Genome) -> None:
    """Robustness measures what evolution/training CONTROLS: frozen library inners stay intact."""
    from ardevo.evolution.composition import AssemblyContext, CompEdgeGene, CompNodeGene, CompNodeKind, CompositionGenome, assemble
    from ardevo.evolution.genome import genome_to_dict
    from ardevo.library import MODULE, ModuleLibrary

    library = ModuleLibrary(tmp_path / "lib")
    io = {"inputs": [{"signature": "BINARY|K", "width": 2}], "output": {"signature": "BINARY|K", "width": 1}}
    key = library.add(entry_type=MODULE, payload=genome_to_dict(solving_genome), io=io, provenance={})
    nodes = {
        0: CompNodeGene(0, CompNodeKind.INPUT, "BINARY|K", 0, 2),
        1: CompNodeGene(1, CompNodeKind.MODULE, f"library:{key}", 2, 1),
        2: CompNodeGene(2, CompNodeKind.OUTPUT, "head", 1, 0),
    }
    comp = CompositionGenome(nodes=nodes, edges=[CompEdgeGene(0, 1, True, 0, (1.0, 0.0, 0.0, 1.0)), CompEdgeGene(1, 2, True, 1, (1.0,))])
    net = assemble(comp, AssemblyContext(bank_columns={"BINARY|K": [0, 1]}, library=library), n_inputs=2)
    inner = net.inner_modules[f"library:{key}"]

    observed: list[float] = []
    first_conn = solving_genome.connections[0]
    inner.register_forward_pre_hook(lambda module, args: observed.append(module.export_weights()[(first_conn.in_id, first_conn.out_id, False)]))
    weight_samples(comp, net, xor_adapter, batched_samples=True)  # ComposedNet falls back to serial
    original = float(solving_genome.connections[0].weight)
    assert observed and all(value == original for value in observed)  # inner never filled mid-sampling


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


def test_hybrid_stamps_weight_robustness_through_temporal_adapter(temporal_task) -> None:
    """The refine comparator and retire guard consume weight_robustness for temporal modules too;
    this pins that the stepped path really stamps it. NOTE the field CAN be exactly 0.0 in the
    wild (a constant-weight pole controller fails identically at every sample: mean 0, variance 0),
    which is why the retire guard demands a strict margin instead of weak dominance."""
    import math

    from ardevo.temporal import temporal_adapter
    from tests.test_recurrence import _running_parity_genome

    adapter = temporal_adapter(temporal_task)
    genome = _running_parity_genome()
    metrics = hybrid(genome, adapter.decode(genome), adapter, samples=[-1.0, 1.0])
    assert _ROBUSTNESS_KEYS <= set(metrics)
    assert math.isfinite(metrics["weight_robustness"])
