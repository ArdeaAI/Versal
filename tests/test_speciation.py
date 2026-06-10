import random

from ardevo.evolution.speciation import NeatSpeciation, NoSpeciation, compatibility_distance


def test_compatibility_distance_zero_for_identical(solving_genome):
    assert compatibility_distance(solving_genome, solving_genome, c_excess=1.0, c_disjoint=1.0, c_weight=0.5) == 0.0


def test_compatibility_distance_grows_with_structural_difference(linear_genome, solving_genome):
    coeffs = {"c_excess": 1.0, "c_disjoint": 1.0, "c_weight": 0.5}
    structural = compatibility_distance(linear_genome, solving_genome, **coeffs)
    weight_only = compatibility_distance(linear_genome, linear_genome, **coeffs)
    assert structural > weight_only == 0.0


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


def test_neat_speciation_plans_fill_population(solving_genome):
    speciator = NeatSpeciation(threshold=1.5, c_excess=1.0, c_disjoint=1.0, c_weight=0.5)
    genomes = [solving_genome.clone() for _ in range(20)]
    fitnesses = [0.5] * 20
    plans = speciator(genomes, fitnesses, rng=random.Random(1), elitism=2, pop_size=20)
    assert sum(plan.n_elites + plan.n_offspring for plan in plans) == 20
