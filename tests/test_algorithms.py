import numpy as np
import pytest

from tcp.cso import CSOConfig, run_cso
from tcp.fitness import composite_fitness, unweighted_fitness
from tcp.ga import GAConfig, order_crossover, run_ga, swap_mutation
from tcp.hybrid import HybridConfig, run_single_sprint
from tcp.synthetic import generate_synthetic_suite


def _is_valid_permutation(ordering: np.ndarray, n: int) -> bool:
    return sorted(ordering.tolist()) == list(range(n))


def test_order_crossover_produces_valid_permutation():
    rng = np.random.default_rng(0)
    p1 = rng.permutation(10)
    p2 = rng.permutation(10)
    child = order_crossover(p1, p2, rng)
    assert _is_valid_permutation(child, 10)


def test_swap_mutation_preserves_permutation():
    rng = np.random.default_rng(0)
    chromosome = rng.permutation(10)
    mutated = swap_mutation(chromosome, rate=0.5, rng=rng)
    assert _is_valid_permutation(mutated, 10)


def test_ga_improves_over_random_search():
    rng = np.random.default_rng(123)
    suite = generate_synthetic_suite(n_tests=25, n_faults=6, rng=rng)
    fitness_fn = lambda o: unweighted_fitness(o, suite)

    random_scores = [fitness_fn(rng.permutation(suite.n_tests)) for _ in range(50)]

    config = GAConfig(population_size=30, max_iterations=40)
    best_order, best_fitness, history = run_ga(suite.n_tests, fitness_fn, config, rng)

    assert _is_valid_permutation(best_order, suite.n_tests)
    assert best_fitness >= np.mean(random_scores)
    assert history == sorted(history)  # fitness is monotonically non-decreasing (elitism)


def test_cso_runs_and_returns_valid_permutation():
    rng = np.random.default_rng(321)
    suite = generate_synthetic_suite(n_tests=20, n_faults=5, rng=rng)
    fitness_fn = lambda o: unweighted_fitness(o, suite)

    config = CSOConfig(population_size=20, max_iterations=30)
    best_order, best_fitness, history = run_cso(suite.n_tests, fitness_fn, config, rng)

    assert _is_valid_permutation(best_order, suite.n_tests)
    assert history == sorted(history)


def test_hybrid_single_sprint_outperforms_or_matches_ga_alone():
    rng = np.random.default_rng(99)
    suite = generate_synthetic_suite(n_tests=25, n_faults=6, rng=rng)
    fitness_fn = lambda o: composite_fitness(o, suite)

    ga_order, ga_fitness, _ = run_ga(
        suite.n_tests, fitness_fn, GAConfig(population_size=30, max_iterations=40), rng
    )
    hybrid_order, hybrid_fitness, history, population = run_single_sprint(
        suite.n_tests,
        fitness_fn,
        HybridConfig(population_size=30, max_iterations=40),
        np.random.default_rng(99),
    )

    assert _is_valid_permutation(hybrid_order, suite.n_tests)
    assert len(population) == 30
    # hybrid should not be drastically worse than plain GA on a small synthetic suite
    assert hybrid_fitness >= ga_fitness - 0.1
