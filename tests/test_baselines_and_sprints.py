import numpy as np
import pytest

from tcp.baselines import (
    greedy_coverage,
    hybrid_ga_cso_no_agile,
    original_order,
    random_order,
    standalone_ga,
)
from tcp.ga import GAConfig
from tcp.hybrid import HybridConfig, priority_biased_permutation, run_across_sprints
from tcp.metrics import apfd
from tcp.method import proposed_method, proposed_method_across_sprints
from tcp.synthetic import generate_sprint_sequence, generate_synthetic_suite


def _is_valid_permutation(ordering: np.ndarray, n: int) -> bool:
    return sorted(ordering.tolist()) == list(range(n))


@pytest.fixture
def suite():
    rng = np.random.default_rng(55)
    return generate_synthetic_suite(n_tests=20, n_faults=5, rng=rng)


def test_random_and_original_order_are_valid(suite):
    rng = np.random.default_rng(1)
    assert _is_valid_permutation(random_order(suite, rng), suite.n_tests)
    assert np.array_equal(original_order(suite), np.arange(suite.n_tests))


def test_greedy_coverage_achieves_full_coverage_eventually(suite):
    order = greedy_coverage(suite)
    assert _is_valid_permutation(order, suite.n_tests)
    covered = suite.coverage_matrix[order].any(axis=0)
    assert covered.all()  # greedy must eventually cover everything (suite guarantees full coverage)


def test_greedy_coverage_front_loads_high_gain_tests(suite):
    order = greedy_coverage(suite)
    # the first test chosen should cover at least as much as a random test on average
    first_test_gain = suite.coverage_matrix[order[0]].sum()
    avg_gain = suite.coverage_matrix.sum(axis=1).mean()
    assert first_test_gain >= avg_gain


def test_standalone_ga_beats_random_baseline_on_average(suite):
    rng = np.random.default_rng(2)
    ga_order = standalone_ga(suite, GAConfig(population_size=30, max_iterations=40), rng)
    random_scores = [apfd(random_order(suite, rng), suite.fault_matrix) for _ in range(30)]
    assert apfd(ga_order, suite.fault_matrix) >= np.mean(random_scores)


def test_hybrid_no_agile_baseline_valid(suite):
    rng = np.random.default_rng(3)
    order = hybrid_ga_cso_no_agile(
        suite, HybridConfig(population_size=20, max_iterations=20), rng
    )
    assert _is_valid_permutation(order, suite.n_tests)


def test_proposed_method_valid_and_reasonable(suite):
    rng = np.random.default_rng(4)
    order = proposed_method(suite, HybridConfig(population_size=20, max_iterations=20), rng=rng)
    assert _is_valid_permutation(order, suite.n_tests)
    assert apfd(order, suite.fault_matrix) > 0.0


def test_priority_biased_permutation_favours_marked_tests():
    rng = np.random.default_rng(0)
    n = 50
    mask = np.zeros(n, dtype=bool)
    mask[:10] = True  # first 10 tests are "new/modified"

    avg_positions = []
    for _ in range(200):
        order = priority_biased_permutation(n, mask, rng, boost=2.0)
        positions = np.argsort(order)  # positions[test] = where it landed
        avg_positions.append(positions[mask].mean())

    marked_avg_position = np.mean(avg_positions)
    # with a strong boost, marked tests should land well before the midpoint (n/2 = 25)
    assert marked_avg_position < n / 2


def test_run_across_sprints_handles_growing_suite():
    rng = np.random.default_rng(10)
    suites = generate_sprint_sequence(
        n_sprints=3, n_tests_per_sprint=15, growth_per_sprint=5, rng=rng
    )
    results = proposed_method_across_sprints(
        suites, HybridConfig(population_size=15, max_iterations=15), rng=rng
    )
    assert len(results) == 3
    for sprint_idx, (res, s) in enumerate(zip(results, suites)):
        assert res["sprint"] == sprint_idx
        assert _is_valid_permutation(res["ordering"], s.n_tests)
