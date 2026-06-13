"""The Entity level: a meta-GA evolving the search's own knobs across task batches (the macro analogue
of the per-genome self-adaptive operator rates). The MetaEvolver is pure (scorer injected), so its
convergence is tested against a synthetic objective; the production scorer is smoke-tested on XOR."""

import random
from typing import Any

from ardevo.dataset.icarus import Level0Encoder, Task
from ardevo.evaluation import encode, input_width, output_features
from ardevo.evolution.entity import ENTITY_GENES, EntityGenome, MetaEvolver, build_orchestrator_batch_evaluator


def _base_config() -> dict[str, Any]:
    return {
        "evolution": {"novelty": {"weight": 0.5}, "mutation": {"enable_refinement_prob": 0.03}},
        "orchestrator": {"stall_generations": 28, "budgets": {"depth0": 240}, "decompose_solvability_floor": 0.6},
    }


# --- EntityGenome ---------------------------------------------------------------------------------


def test_seed_reads_current_config_values() -> None:
    genome = EntityGenome.seed(_base_config())
    assert genome.values["evolution/novelty/weight"] == 0.5
    assert genome.values["orchestrator/stall_generations"] == 28.0
    assert genome.values["orchestrator/budgets/depth0"] == 240.0


def test_apply_writes_values_into_nested_paths() -> None:
    genome = EntityGenome.seed(_base_config())
    genome.values["evolution/novelty/weight"] = 0.8
    genome.values["orchestrator/stall_generations"] = 40.0
    applied = genome.apply(_base_config())
    assert applied["evolution"]["novelty"]["weight"] == 0.8
    assert applied["orchestrator"]["stall_generations"] == 40 and isinstance(applied["orchestrator"]["stall_generations"], int)
    # The base config is untouched (apply deep-copies).
    assert _base_config()["evolution"]["novelty"]["weight"] == 0.5


def test_mutation_stays_within_gene_bounds() -> None:
    genome = EntityGenome.seed(_base_config())
    rng = random.Random(0)
    for _ in range(200):
        genome = genome.mutate(rng, sigma=0.5)
        for gene in ENTITY_GENES:
            value = genome.values[gene.key]
            assert gene.low <= value <= gene.high
            if gene.integer:
                assert value == round(value)


# --- MetaEvolver ----------------------------------------------------------------------------------


def test_meta_evolver_converges_toward_a_synthetic_optimum() -> None:
    # A scorer peaked at novelty.weight = 0.8: the champion must move the seed (0.5) toward it.
    def scorer(genome: EntityGenome) -> float:
        return -abs(genome.values["evolution/novelty/weight"] - 0.8)

    meta = MetaEvolver(_base_config(), scorer, pop_size=8, generations=8, sigma=0.2)
    champion = meta.run(random.Random(0))
    assert abs(champion.values["evolution/novelty/weight"] - 0.8) < abs(0.5 - 0.8)  # closer than the seed
    assert meta.history[-1]["best_score"] >= meta.history[0]["best_score"]  # monotone non-decreasing best


def test_meta_evolver_is_deterministic() -> None:
    def scorer(genome: EntityGenome) -> float:
        return genome.values["orchestrator/budgets/depth0"]

    def run() -> dict[str, float]:
        return MetaEvolver(_base_config(), scorer, pop_size=5, generations=4).run(random.Random(3)).to_dict()

    assert run() == run()


# --- production scorer (smoke) --------------------------------------------------------------------


def test_orchestrator_batch_evaluator_scores_without_crashing(xor_task: Task, xor_encoder: Level0Encoder) -> None:
    # A minimal orchestrated config + one solvable XOR task: the evaluator must run the real
    # orchestrator under an Entity's config and return a finite score.
    encoded = encode(xor_task, xor_encoder)
    _ = (input_width(encoded), output_features(encoded))  # XOR encodes cleanly
    config: dict[str, Any] = {
        "seed": 0,
        "dataset": "unused",
        "n_samples": 4,
        "substrate": {"available_activations": ["tanh", "identity"], "default_activation": "tanh"},
        "orchestrator": {
            "tasks": 1,
            "accept_threshold": 0.95,
            "evolve": ["direct"],
            "decompose": [],
            "max_depth": 0,
            "budgets": {"depth0": 6},
            "evolve_budget": {"direct": 1.0},
            "stall_generations": 28,
            "decompose_solvability_floor": 0.6,
        },
        "evolution": {
            "loop": "hierarchical",
            "pop_size": 8,
            "elitism": 1,
            "init": {"kind": "minimal"},
            "selection": {"kind": "tournament", "tournament_size": 2},
            "crossover": {"kind": "neat", "rate": 0.2},
            "mutation": {"operators": ["add_rich_node", "add_connection"], "add_rich_node_prob": 0.3, "add_connection_prob": 0.3, "enable_refinement_prob": 0.03},
            "train": {"kind": "gradient", "steps": 4, "lr": 0.05, "writeback": True},
            "evaluate": {"kind": "standard"},
            "novelty": {"weight": 0.5},
            "speciation": {"kind": "neat", "threshold": 1.5, "target_species": 2},
            "composition": {"pop_size": 4},
            "modules": {"pop_size": 8, "in_ports": 2, "out_ports": 1},
        },
        "fitness": {"components": ["support_accuracy"], "w_support_accuracy": 1.0},
    }
    evaluator = build_orchestrator_batch_evaluator(config, [xor_task], seed=0)
    score = evaluator(EntityGenome.seed(config))
    assert 0.0 <= score <= 1.0  # a finite mean accept metric in [0,1]
