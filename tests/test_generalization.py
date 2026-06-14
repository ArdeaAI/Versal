"""Phase 7 Pillar A: the inner support fold + generalization fitness + identity-preserving growth.

The orchestrated/library path may NOT select on the real query (it is the accept metric and the
library admission currency), so it generalizes against an INNER fold carved out of training. These
tests cover the deterministic fold partition, the holdout-aware evaluate contract (byte-identical when
no fold), the generalization fitness components, and that zero-initialized node growth is identity-
preserving (does not perturb the trained function)."""

import random

from ardevo.dataset.icarus import Level0Encoder
from ardevo.evaluation import encode, evaluate, input_width, output_features, restrict_support, support_fold
from ardevo.evolution.evolver import TaskAdapter
from ardevo.evolution.fitness import bounded_negative_holdout_loss, generalization_gap, holdout_accuracy
from ardevo.evolution.genome import Genome, InnovationTracker
from ardevo.evolution.init import minimal
from ardevo.evolution.mutation import MutationContext, add_deep_node, add_rich_node


def _encoded(task):
    encoder = Level0Encoder(max_flat_dim=8)
    return encode(task, encoder), encoder


def _adapter(task, fraction=0.0):
    encoded, encoder = _encoded(task)
    return TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded), validation_fraction=fraction)


def _context(genome: Genome) -> MutationContext:
    return MutationContext(innovations=InnovationTracker.from_genomes([genome]), activations=["tanh"], default_activation="tanh")


def test_support_fold_is_deterministic_and_a_partition(decomposable_task):
    encoded, _ = _encoded(decomposable_task)
    n = int(encoded.support_input[0].shape[0])
    train_a, holdout_a = support_fold(encoded, 0.25)
    train_b, holdout_b = support_fold(encoded, 0.25)
    assert (train_a, holdout_a) == (train_b, holdout_b)  # deterministic by io shape, stable across calls
    assert set(train_a).isdisjoint(holdout_a)
    assert sorted(train_a + holdout_a) == list(range(n))  # exactly partitions the support rows
    assert len(holdout_a) >= 4 and len(train_a) >= 4


def test_support_fold_off_at_zero_and_for_tiny_support(decomposable_task, xor_task, xor_encoder):
    encoded, _ = _encoded(decomposable_task)
    n = int(encoded.support_input[0].shape[0])
    assert support_fold(encoded, 0.0) == (list(range(n)), [])  # fraction 0 = no fold
    xor_encoded = encode(xor_task, xor_encoder)  # only 4 support rows: too small to keep a holdout
    _train, holdout = support_fold(xor_encoded, 0.25)
    assert holdout == []


def test_restrict_support_slices_support_and_leaves_query(decomposable_task):
    encoded, _ = _encoded(decomposable_task)
    view = restrict_support(encoded, [0, 1, 2])
    assert int(view.support_input[0].shape[0]) == 3
    assert int(view.support_target[0].shape[0]) == 3
    assert view.query_input is encoded.query_input  # query untouched


def test_evaluate_emits_holdout_keys_only_with_a_fold(decomposable_task):
    encoded, encoder = _encoded(decomposable_task)
    module = TaskAdapter(encoded, encoder, input_width(encoded), output_features(encoded)).decode(
        minimal(input_width(encoded), output_features(encoded), rng=random.Random(0))
    )
    plain = evaluate(module, encoded, encoder)
    assert "support_holdout_accuracy" not in plain  # no fold -> exact pre-fold contract
    train, holdout = support_fold(encoded, 0.25)
    folded = evaluate(module, restrict_support(encoded, train), encoder, holdout=restrict_support(encoded, holdout))
    assert "support_holdout_accuracy" in folded and "support_holdout_loss" in folded


def test_task_adapter_fold_splits_training_off_from_holdout(decomposable_task):
    full = _adapter(decomposable_task, fraction=0.0)
    folded = _adapter(decomposable_task, fraction=0.25)
    assert folded.holdout_encoded is not None
    full_rows = int(full.encoded.support_input[0].shape[0])
    train_rows = int(folded.train_encoded.support_input[0].shape[0])
    holdout_rows = int(folded.holdout_encoded.support_input[0].shape[0])
    assert train_rows < full_rows  # the trainer fits strictly fewer rows than the full support
    assert train_rows + holdout_rows == full_rows  # and the holdout is the rest


def test_task_adapter_is_byte_identical_with_no_fold(xor_adapter):
    # The default adapter (validation_fraction 0.0) leaves train == encoded and emits no holdout keys.
    assert xor_adapter.holdout_encoded is None
    assert xor_adapter.train_encoded is xor_adapter.encoded
    module = xor_adapter.decode(minimal(xor_adapter.n_inputs, xor_adapter.n_outputs, rng=random.Random(0)))
    assert set(xor_adapter.evaluate(module)) == {"support_accuracy", "support_loss", "query_accuracy", "query_loss"}


def test_generalization_components_are_neutral_without_a_fold():
    # No holdout keys -> gap is exactly 0 and holdout metrics fall back to support, so adding the
    # components to a config with validation_fraction 0 changes nothing.
    metrics = {"support_accuracy": 0.8, "support_loss": 0.3}
    assert generalization_gap(None, metrics) == 0.0
    assert holdout_accuracy(None, metrics) == 0.8
    assert bounded_negative_holdout_loss(None, metrics) == 1.0 / 1.3


def test_generalization_gap_penalizes_overfit():
    overfit = {"support_accuracy": 1.0, "support_holdout_accuracy": 0.6}
    generalizing = {"support_accuracy": 0.9, "support_holdout_accuracy": 0.9}
    assert generalization_gap(None, overfit) == -0.4  # a train/holdout gap is penalized
    assert generalization_gap(None, generalizing) == 0.0
    assert holdout_accuracy(None, overfit) == 0.6  # reads the held-out fold, not the memorized train


def test_init_zero_edges_makes_node_growth_identity_preserving(linear_genome):
    ctx = _context(linear_genome)
    grown = add_rich_node(linear_genome, ctx, rng=random.Random(0), prob=1.0, init_zero_edges=True)
    new_id = max(grown.nodes)  # the freshly added hidden node
    readouts = [conn for conn in grown.connections if conn.in_id == new_id]
    assert readouts and all(conn.weight == 0.0 for conn in readouts)  # readout zeroed -> no perturbation
    incoming = [conn for conn in grown.connections if conn.out_id == new_id]
    assert any(conn.weight != 0.0 for conn in incoming)  # incoming stay live so gradients can grow it


def test_add_deep_node_zero_edges_zeroes_all_outgoing(linear_genome):
    # Seed a hidden node so add_deep_node has a hidden->hidden target to (zero-)wire.
    seeded = add_rich_node(linear_genome, _context(linear_genome), rng=random.Random(1), prob=1.0)
    ctx = _context(seeded)
    grown = add_deep_node(seeded, ctx, rng=random.Random(2), prob=1.0, fan_out=4, init_zero_edges=True)
    new_id = max(grown.nodes)
    outgoing = [conn for conn in grown.connections if conn.in_id == new_id]
    assert outgoing and all(conn.weight == 0.0 for conn in outgoing)


def test_init_zero_edges_default_false_keeps_random_readouts(linear_genome):
    grown = add_rich_node(linear_genome, _context(linear_genome), rng=random.Random(0), prob=1.0)
    new_id = max(grown.nodes)
    readouts = [conn for conn in grown.connections if conn.in_id == new_id]
    assert any(conn.weight != 0.0 for conn in readouts)  # default path unchanged (random readouts)
