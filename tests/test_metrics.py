import numpy as np
import pytest

from tcp.metrics import apfd, apfdc, coverage_effectiveness, weighted_apfd


@pytest.fixture
def small_fault_matrix():
    # 4 tests, 2 faults. fault 0 detected only by test 0, fault 1 only by test 3.
    fm = np.zeros((2, 4), dtype=bool)
    fm[0, 0] = True
    fm[1, 3] = True
    return fm


def test_apfd_early_detection_scores_higher(small_fault_matrix):
    good_order = np.array([0, 3, 1, 2])  # detects both faults within first 2 tests
    bad_order = np.array([1, 2, 0, 3])  # detects both faults only by position 3 and 4
    assert apfd(good_order, small_fault_matrix) > apfd(bad_order, small_fault_matrix)


def test_apfd_known_value(small_fault_matrix):
    order = np.array([0, 3, 1, 2])
    # TF0=1, TF1=2, n=4, m=2 -> 1 - (1+2)/(4*2) + 1/(2*4) = 1 - 0.375 + 0.125 = 0.75
    assert apfd(order, small_fault_matrix) == pytest.approx(0.75)


def test_apfd_perfect_order_near_one():
    fm = np.eye(5, dtype=bool)  # test i uniquely detects fault i
    order = np.arange(5)
    score = apfd(order, fm)
    # 1 - (1+2+3+4+5)/(5*5) + 1/(2*5) = 1 - 15/25 + 0.1 = 0.5
    assert score == pytest.approx(0.5)


def test_weighted_apfd_reduces_to_apfd_with_uniform_weights(small_fault_matrix):
    order = np.array([0, 3, 1, 2])
    weights = np.ones(2)
    assert weighted_apfd(order, small_fault_matrix, weights) == pytest.approx(
        apfd(order, small_fault_matrix)
    )


def test_weighted_apfd_prioritises_heavier_fault(small_fault_matrix):
    order_detects_fault0_first = np.array([0, 1, 2, 3])
    order_detects_fault1_first = np.array([3, 1, 2, 0])
    weights = np.array([10.0, 1.0])  # fault 0 much more important
    assert weighted_apfd(
        order_detects_fault0_first, small_fault_matrix, weights
    ) > weighted_apfd(order_detects_fault1_first, small_fault_matrix, weights)


def test_apfdc_matches_apfd_under_uniform_cost(small_fault_matrix):
    order = np.array([0, 3, 1, 2])
    cost = np.full(4, 2.5)
    assert apfdc(order, small_fault_matrix, cost) == pytest.approx(
        apfd(order, small_fault_matrix)
    )


def test_apfdc_penalises_expensive_early_detection(small_fault_matrix):
    order = np.array([0, 3, 1, 2])
    cheap_first = np.array([1.0, 1.0, 1.0, 1.0])
    expensive_first = np.array([10.0, 1.0, 1.0, 1.0])  # test 0 (first in order) now costly
    assert apfdc(order, small_fault_matrix, cheap_first) > apfdc(
        order, small_fault_matrix, expensive_first
    )


def test_coverage_effectiveness_full_suite_covers_everything():
    coverage = np.array([[True, False], [False, True]])
    order = np.array([0, 1])
    assert coverage_effectiveness(order, coverage) == pytest.approx(1.0)


def test_coverage_effectiveness_respects_budget_cutoff():
    coverage = np.array([[True, False, False], [False, True, False], [False, False, True]])
    cost = np.array([1.0, 1.0, 1.0])
    order = np.array([0, 1, 2])
    # only the first test (cost 1) fits within budget 1 -> covers 1/3 elements
    assert coverage_effectiveness(order, coverage, cost, budget=1.0) == pytest.approx(1 / 3)
    # budget covering all three tests -> full coverage
    assert coverage_effectiveness(order, coverage, cost, budget=3.0) == pytest.approx(1.0)
