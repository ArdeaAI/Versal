"""Critical-path tests for the connective-CPPN spike (ai/cppn_spike_plan.md Phase 0/1).

The existence test IS gate E0: a hand-built Fourier-in-c generator (complexity ~43, vs the m3
stone's 903), weights trained by the ordinary gradient op at the pinned (seed, lr, steps), must
carve the pinned synthetic spiral. The expansion-in-forward tests pin the load-bearing contract:
a decode-time cache would break the second Adam step and fake the weight-sample diagnostic.
"""

import random

import pytest
import torch

from versal.cppn import CPPN_INPUTS, CppnPhenotypeNet, CppnTaskAdapter, cppn_output_count, fourier_cppn_genome, synthetic_two_spirals_task
from versal.dataset.icarus import Level0Encoder
from versal.evaluation import encode, support_loss
from versal.evolution.evaluate import weight_samples
from versal.evolution.registry import build_evolver
from versal.evolution.train import gradient
from versal.substrate import decode as decode_graphnet

# The pinned existence fixture (established best-of-8 in the Phase-0 harness; see the plan).
_FIXTURE = {"m": 3, "seed": 6}
_FIXTURE_TRAIN = {"steps": 3000, "lr": 0.03}


@pytest.fixture(scope="module")
def spiral_adapter() -> CppnTaskAdapter:
    encoder = Level0Encoder(max_flat_dim=2)
    encoded = encode(synthetic_two_spirals_task(), encoder)
    return CppnTaskAdapter(encoded, encoder, h=64, phenotype_activation="sin")


def test_synthetic_task_is_pinned(spiral_adapter: CppnTaskAdapter) -> None:
    tensor, _descriptor = spiral_adapter.encoded.support_input
    assert tensor.shape == (320, 2)  # n = 400, support_fraction = 0.8, deterministic stride split
    again, _ = encode(synthetic_two_spirals_task(), spiral_adapter.encoder).support_input
    assert torch.equal(tensor, again)  # no rng anywhere in the generator


def test_existence_fixture_solves_synthetic_spiral(spiral_adapter: CppnTaskAdapter) -> None:
    """Gate E0: the generative encoding can express the spiral compactly, trained through the
    real pipeline (GraphNet CPPN -> detector bank -> Icarus BCE loss -> Adam)."""
    genome = fourier_cppn_genome(**_FIXTURE)
    assert genome.complexity() <= 60 and len(genome.hidden_ids) <= 10  # the gate-E compactness bars
    module = spiral_adapter.decode(genome)
    _genome, trained = gradient(genome, module, spiral_adapter.encoded, rng=random.Random(0), writeback=False, **_FIXTURE_TRAIN)
    metrics = spiral_adapter.evaluate(trained)
    assert metrics["query_accuracy"] >= 0.9  # pinned run reaches 1.0; the margin absorbs torch drift


def test_deterministic_expansion(spiral_adapter: CppnTaskAdapter) -> None:
    genome = fourier_cppn_genome(**_FIXTURE)
    x = spiral_adapter.encoded.support_input[0][:8]
    first, second = spiral_adapter.decode(genome), spiral_adapter.decode(genome)
    assert torch.equal(first(x), second(x))


def test_expansion_runs_inside_forward_and_writes_back(spiral_adapter: CppnTaskAdapter) -> None:
    """Multi-step training through the wrapper must work (a decode-time expansion cache raises on
    the second backward or silently freezes learning), and Lamarckian writeback must land on the
    CPPN genome such that a fresh decode reproduces the trained function."""
    genome = fourier_cppn_genome(**_FIXTURE)
    module = spiral_adapter.decode(genome)
    before = float(support_loss(module, spiral_adapter.encoded).detach())
    tuned_genome, trained = gradient(genome, module, spiral_adapter.encoded, rng=random.Random(0), steps=50, lr=0.03, writeback=True)
    after = float(support_loss(trained, spiral_adapter.encoded).detach())
    assert after < before  # gradient actually flowed, twice-plus, through fresh expansions
    weights_before = [conn.weight for conn in genome.connections]
    weights_after = [conn.weight for conn in tuned_genome.connections]
    assert weights_before != weights_after  # writeback landed on the GENERATOR's genes
    x = spiral_adapter.encoded.support_input[0][:8]
    assert torch.allclose(spiral_adapter.decode(tuned_genome)(x), trained(x), atol=1e-5)


def test_weight_sample_diagnostic_interrogates_the_generator(spiral_adapter: CppnTaskAdapter) -> None:
    """The sweep fills the CPPN's weights; per-sample accuracies must VARY (a cached expansion
    would leave the phenotype constant and fake the diagnostic), and the weights must restore."""
    genome = fourier_cppn_genome(**_FIXTURE)
    module = spiral_adapter.decode(genome)
    original = [parameter.detach().clone() for parameter in module.parameters()]
    metrics = weight_samples(genome, module, spiral_adapter)
    assert metrics["max_sample_accuracy"] > metrics["min_sample_accuracy"]
    for parameter, saved in zip(module.parameters(), original):
        assert torch.equal(parameter.detach(), saved)


def test_re_expansion_changes_resolution_not_function_family(spiral_adapter: CppnTaskAdapter) -> None:
    genome = fourier_cppn_genome(**_FIXTURE)
    module = spiral_adapter.decode(genome)
    x = spiral_adapter.encoded.support_input[0][:8]
    for h in (16, 32, 128):
        re_expanded = module.re_expanded(h)
        assert re_expanded.h == h
        assert re_expanded(x).shape == (8, 1)  # the BINARY loss/decode shape contract
        assert re_expanded.cppn is module.cppn  # the SAME generator, only the bank resized


def test_evolver_end_to_end_on_the_cppn_adapter(spiral_adapter: CppnTaskAdapter) -> None:
    config = {
        "seed": 0,
        "substrate": {"available_activations": ["tanh", "relu", "sigmoid", "identity", "sin", "gaussian"], "default_activation": "tanh"},
        "evolution": {
            "pop_size": 8,
            "elitism": 1,
            "init": {"kind": "minimal", "weight_scale": 1.0},
            "selection": {"kind": "tournament", "tournament_size": 3},
            "crossover": {"kind": "neat", "rate": 0.2},
            "mutation": {
                "operators": ["add_rich_node", "add_connection", "mutate_activation"],
                "add_rich_node_prob": 0.3,
                "add_connection_prob": 0.2,
                "mutate_activation_prob": 0.2,
            },
            "train": {"kind": "gradient", "steps": 10, "lr": 0.03, "writeback": True},
            "speciation": {"kind": "none"},
        },
        "fitness": {"components": ["bounded_negative_support_loss", "support_accuracy"], "w_bounded_negative_support_loss": 2.0},
    }
    evolver = build_evolver(config)
    state = evolver.seed_state(spiral_adapter, random.Random(0))
    for _ in range(2):
        evolver.advance(state, spiral_adapter)
    assert len(state.population) == 8
    for item in state.population:
        assert len(item.genome.input_ids) == CPPN_INPUTS and len(item.genome.output_ids) == spiral_adapter.n_outputs  # mutators grow the GENERATOR, never its I/O
        assert item.fitness > -1e9  # every CPPN decodes and scores


def test_task_shaped_readout_head() -> None:
    """The real rung-3 target is CATEGORICAL(2): the bank must emit the task's logit width, with
    the CPPN growing one readout curve per logit (the Phase-2 crash of 2026-07-05)."""
    genome = fourier_cppn_genome(m=1, seed=0, n_logits=2)
    assert len(genome.output_ids) == cppn_output_count(2) == 5
    module = CppnPhenotypeNet(decode_graphnet(genome, CPPN_INPUTS, 5), h=16, phenotype_activation="sin", n_logits=2)
    assert module(torch.zeros((7, 2))).shape == (7, 2)
    assert module.re_expanded(32)(torch.zeros((7, 2))).shape == (7, 2)
