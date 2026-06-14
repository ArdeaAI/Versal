"""The Entity level: a meta-GA evolving the search's own knobs across task batches (the macro analogue
of the per-genome self-adaptive operator rates). The MetaEvolver is pure (scorer injected), so its
convergence is tested against a synthetic objective; the production scorer is smoke-tested on XOR."""

import random
from pathlib import Path
from typing import Any

from ardevo.dataset.icarus import Level0Encoder, Task
from ardevo.evaluation import encode, input_width, output_features
from ardevo.evolution.entity import (
    ENTITY_GENES,
    EntityGenome,
    MetaEvolver,
    _genome_from_saved,
    _should_resave,
    build_orchestrator_batch_evaluator,
    load_entity_policy,
    load_entity_score,
    run_entity_layer,
    save_entity_policy,
)


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


# --- policy persistence (phase 7) -----------------------------------------------------------------


def test_save_and_load_entity_policy_round_trip(tmp_path: Path) -> None:
    champion = EntityGenome.seed(_base_config())
    champion.values["evolution/novelty/weight"] = 0.77
    path = tmp_path / "entity_policy.json"
    save_entity_policy(path, champion, score=0.85, context={"rungs": [1, 2]})
    loaded = load_entity_policy(path)
    assert loaded is not None and loaded["evolution/novelty/weight"] == 0.77
    assert load_entity_score(path) == 0.85  # the bar a later run must beat to re-save
    assert load_entity_policy(tmp_path / "missing.json") is None  # no policy saved yet
    assert load_entity_score(tmp_path / "missing.json") is None


def test_should_resave_only_when_strictly_beating_the_stored_score() -> None:
    assert _should_resave(None, 0.5)  # no policy yet -> save
    assert _should_resave(0.5, 0.6)  # a strict improvement -> re-save
    assert not _should_resave(0.6, 0.5)  # worse -> keep the stored best
    assert not _should_resave(0.5, 0.5)  # a tie does not overwrite (no regression, no churn)


def test_genome_from_saved_clamps_to_current_bounds_and_ignores_unknown_keys() -> None:
    saved = {"evolution/novelty/weight": 0.6, "orchestrator/budgets/depth0": 999999.0, "bogus/key": 1.0}
    genome = _genome_from_saved(_base_config(), saved)
    assert genome.values["evolution/novelty/weight"] == 0.6
    assert genome.values["orchestrator/budgets/depth0"] == 320.0  # clamped to the gene's high
    assert "bogus/key" not in genome.values  # a key that no longer maps to a gene is dropped


def test_run_entity_layer_applies_saved_policy_when_evolve_off(tmp_path: Path) -> None:
    # evolve=false returns BEFORE any dataset load, so the saved policy is applied offline and the
    # meta-GA is skipped entirely (dataset "unused" would crash build_pool_report if reached).
    policy_path = tmp_path / "entity_policy.json"
    champion = EntityGenome.seed(_base_config())
    champion.values["orchestrator/stall_generations"] = 33.0
    save_entity_policy(policy_path, champion, score=0.9, context={})
    config = _base_config()
    config["dataset"] = "unused"
    config["n_samples"] = 4
    config["entity"] = {"evolve": False, "reuse": True, "policy_path": str(policy_path)}
    applied = run_entity_layer(config)
    assert applied["orchestrator"]["stall_generations"] == 33  # saved policy applied, no meta-GA run


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
