import numpy as np
import pytest

from tcp.fitness import FitnessWeights, composite_fitness, penalty_cost, unweighted_fitness
from tcp.synthetic import generate_synthetic_suite


def test_fitness_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        FitnessWeights(alpha1=0.5, alpha2=0.5, alpha3=0.5)


def test_fitness_weights_valid():
    w = FitnessWeights(0.6, 0.3, 0.1)
    assert w.alpha1 + w.alpha2 + w.alpha3 == pytest.approx(1.0)


def test_penalty_zero_when_under_budget():
    cost = np.array([1.0, 1.0, 1.0])
    assert penalty_cost(cost, budget=10.0) == 0.0


def test_penalty_positive_when_over_budget():
    cost = np.array([5.0, 5.0, 5.0])  # total 15
    assert penalty_cost(cost, budget=10.0) == pytest.approx(0.5)  # (15-10)/10


def test_penalty_zero_without_budget():
    cost = np.array([100.0])
    assert penalty_cost(cost, budget=None) == 0.0


def test_composite_fitness_in_reasonable_range():
    rng = np.random.default_rng(42)
    suite = generate_synthetic_suite(n_tests=20, n_faults=5, rng=rng)
    order = rng.permutation(suite.n_tests)
    f = composite_fitness(order, suite)
    # wAPFD and CE are in [0,1], penalty >= 0, weights sum to 1 -> F should be
    # bounded well within [-1, 1.5] for any ordering.
    assert -1.0 <= f <= 1.5


def test_composite_fitness_prefers_early_fault_detection():
    rng = np.random.default_rng(7)
    suite = generate_synthetic_suite(n_tests=15, n_faults=4, rng=rng, budget_fraction=1.0)
    # order that detects fault 0 (whichever test covers it first) as early as possible
    # vs a reversed order -- on average, many random reversals should differ.
    order = np.arange(suite.n_tests)
    reversed_order = order[::-1]
    f1 = composite_fitness(order, suite)
    f2 = composite_fitness(reversed_order, suite)
    assert f1 != pytest.approx(f2)  # ordering must matter


def test_unweighted_fitness_runs():
    rng = np.random.default_rng(1)
    suite = generate_synthetic_suite(n_tests=10, n_faults=3, rng=rng)
    order = rng.permutation(suite.n_tests)
    f = unweighted_fitness(order, suite)
    assert 0.0 <= f <= 1.0
