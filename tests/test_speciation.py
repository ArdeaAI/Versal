import random

from ardevo.evolution.speciation import NeatSpeciation, NoSpeciation, compatibility_distance


def test_compatibility_distance_zero_for_identical(solving_genome):
    assert compatibility_distance(solving_genome, solving_genome, c_excess=1.0, c_disjoint=1.0, c_weight=0.5) == 0.0


def test_compatibility_distance_grows_with_structural_difference(linear_genome, solving_genome):
    coeffs = {"c_excess": 1.0, "c_disjoint": 1.0, "c_weight": 0.5}
    structural = compatibility_distance(linear_genome, solving_genome, **coeffs)
    weight_only = compatibility_distance(linear_genome, linear_genome, **coeffs)
    assert structural > weight_only == 0.0


def test_node_gene_term_separates_aggregation_variants(solving_genome):
    """B6 regression: sum-vs-product (or activation) flips must be visible to speciation."""
    from dataclasses import replace

    variant = solving_genome.clone()
    variant.nodes[3] = replace(variant.nodes[3], aggregation="product")
    coeffs = {"c_excess": 1.0, "c_disjoint": 1.0, "c_weight": 0.5}
    assert compatibility_distance(solving_genome, variant, **coeffs) == 0.0  # default keeps old behavior
    with_node_term = compatibility_distance(solving_genome, variant, **coeffs, c_node=0.25)
    assert with_node_term > 0.0
    assert compatibility_distance(solving_genome, solving_genome, **coeffs, c_node=0.25) == 0.0


def test_no_speciation_is_one_global_group(linear_genome):
    genomes = [linear_genome.clone() for _ in range(10)]
    fitnesses = [float(i) for i in range(10)]
    plans = NoSpeciation()(genomes, fitnesses, rng=random.Random(0), elitism=2, pop_size=10)
    assert len(plans) == 1
    assert plans[0].members == list(range(10))
    assert plans[0].n_elites == 2
    assert plans[0].n_offspring == 8


def test_neat_speciation_splits_distinct_structures(linear_genome, solving_genome):
    speciator = NeatSpeciation(threshold=0.5, c_excess=1.0, c_disjoint=1.0, c_weight=0.5)
    genomes = [linear_genome.clone() for _ in range(5)] + [solving_genome.clone() for _ in range(5)]
    fitnesses = [1.0] * 10
    plans = speciator(genomes, fitnesses, rng=random.Random(0), elitism=2, pop_size=10)

    assert len(plans) >= 2, "linear and solving topologies should land in separate species"
    assert sum(plan.n_elites + plan.n_offspring for plan in plans) == 10
    assert all(plan.n_elites == 1 for plan in plans)


def test_neat_species_ids_persist_and_birth(linear_genome, solving_genome):
    speciator = NeatSpeciation(threshold=0.5, c_excess=1.0, c_disjoint=1.0, c_weight=0.5)
    first = speciator([linear_genome.clone() for _ in range(6)], [1.0] * 6, rng=random.Random(0), elitism=1, pop_size=6)
    assert {plan.species_id for plan in first} == {0}

    mixed = [linear_genome.clone() for _ in range(3)] + [solving_genome.clone() for _ in range(3)]
    second = speciator(mixed, [1.0] * 6, rng=random.Random(0), elitism=1, pop_size=6)
    ids = {plan.species_id for plan in second}
    assert 0 in ids, "the original species persists with its stable id"
    assert len(ids) >= 2, "the distinct topology is born as a new species"


def test_neat_threshold_targets_species_count():
    speciator = NeatSpeciation(threshold=1.0, c_excess=1.0, c_disjoint=1.0, c_weight=0.5, target_species=5, threshold_adjust=0.3, min_threshold=0.3)
    speciator._adjust_threshold(20)  # too many species -> loosen (raise threshold)
    assert speciator.threshold == 1.3
    speciator._adjust_threshold(2)  # too few -> tighten (lower threshold)
    assert abs(speciator.threshold - 1.0) < 1e-9
    speciator.threshold = 0.4
    speciator._adjust_threshold(1)  # would drop below the floor
    assert speciator.threshold == 0.3

    fixed = NeatSpeciation(threshold=1.0, c_excess=1.0, c_disjoint=1.0, c_weight=0.5, target_species=0)
    fixed._adjust_threshold(100)  # disabled -> unchanged
    assert fixed.threshold == 1.0


def test_neat_speciation_plans_fill_population(solving_genome):
    speciator = NeatSpeciation(threshold=1.5, c_excess=1.0, c_disjoint=1.0, c_weight=0.5)
    genomes = [solving_genome.clone() for _ in range(20)]
    fitnesses = [0.5] * 20
    plans = speciator(genomes, fitnesses, rng=random.Random(1), elitism=2, pop_size=20)
    assert sum(plan.n_elites + plan.n_offspring for plan in plans) == 20


def test_neat_survives_non_finite_fitness(linear_genome, solving_genome):
    """Degenerate shares (NaN fitness that slipped past upstream floors) even-split the offspring
    budget instead of crashing the run at `int(round(NaN))`."""
    speciator = NeatSpeciation(threshold=0.1, c_excess=1.0, c_disjoint=1.0, c_weight=0.5)
    genomes = [linear_genome, solving_genome, linear_genome.clone(), solving_genome.clone()]
    plans = speciator(genomes, [float("nan"), 1.0, 0.5, float("nan")], rng=random.Random(0), elitism=1, pop_size=8)
    assert plans and sum(plan.n_offspring for plan in plans) == 8 - len(plans)
    assert all(plan.n_offspring >= 0 for plan in plans)
