"""Evaluation metrics for TCP orderings: APFD, APFDc, wAPFD, and Coverage
Effectiveness (CE). Formulas follow Chapter 2 (Section 2.4) and Chapter 3
(Section 3.3.2.2) of the thesis proposal.
"""

from __future__ import annotations

import numpy as np

Ordering = np.ndarray  # permutation of test indices, shape (n_tests,)


def _first_detection_positions(ordering: Ordering, fault_matrix: np.ndarray) -> np.ndarray:
    """TFi for every fault i: the 1-indexed position in `ordering` of the first
    test case that detects fault i.

    fault_matrix has shape (m_faults, n_tests) and is indexed by the ORIGINAL
    test indices (not ordering positions), matching TestSuite.fault_matrix.
    """
    n = len(ordering)
    m = fault_matrix.shape[0]
    # detects[i, p] = True if the test placed at position p detects fault i
    detects = fault_matrix[:, ordering]  # (m, n), reordered columns
    tf = np.full(m, n, dtype=float)
    for i in range(m):
        hits = np.flatnonzero(detects[i])
        if hits.size == 0:
            raise ValueError(f"fault {i} is not detected by any test in the ordering")
        tf[i] = hits[0] + 1  # 1-indexed position
    return tf


def apfd(ordering: Ordering, fault_matrix: np.ndarray) -> float:
    """Standard Average Percentage of Faults Detected (Rothermel et al., 2002),
    Eq. 2.1, p.35:

        APFD = 1 - (TF1 + TF2 + ... + TFm) / (n * m) + 1 / (2n)
    """
    n = len(ordering)
    m = fault_matrix.shape[0]
    tf = _first_detection_positions(ordering, fault_matrix)
    return 1.0 - tf.sum() / (n * m) + 1.0 / (2 * n)


def weighted_apfd(ordering: Ordering, fault_matrix: np.ndarray, weights: np.ndarray) -> float:
    """wAPFD: APFD generalised with per-fault weights wi (Eq. 3.2, p.58):

        wAPFD = 1 - sum(TFi * wi) / (n * sum(wi)) + 1 / (2n)

    When all wi == 1 this reduces exactly to the standard APFD, as noted in
    the proposal (p.58).
    """
    n = len(ordering)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (fault_matrix.shape[0],):
        raise ValueError("weights must have one entry per fault")
    if np.any(weights < 0):
        raise ValueError("fault weights must be non-negative")
    total_weight = weights.sum()
    if total_weight <= 0:
        raise ValueError("sum of fault weights must be positive")
    tf = _first_detection_positions(ordering, fault_matrix)
    return 1.0 - float((tf * weights).sum()) / (n * total_weight) + 1.0 / (2 * n)


def apfdc(
    ordering: Ordering,
    fault_matrix: np.ndarray,
    cost: np.ndarray,
    severity: np.ndarray | None = None,
) -> float:
    """Cost-cognizant APFDc (Elbaum et al., 2001, as cited pp.36, 58):

        APFDc = sum_i( severity_i * (TotalCost - cost_to_detect_i) )
                / (TotalCost * sum_i(severity_i))

    where cost_to_detect_i is the cumulative cost of all tests strictly
    before the one that first detects fault i, plus half the cost of that
    detecting test itself (the test is "in progress" when the fault is
    revealed). If `severity` is omitted, all faults are weighted equally
    (severity_i = 1), which is the conventional APFDc definition.
    """
    n = len(ordering)
    m = fault_matrix.shape[0]
    cost = np.asarray(cost, dtype=float)
    if cost.shape != (fault_matrix.shape[1],):
        raise ValueError("cost must have one entry per test case")
    if severity is None:
        severity = np.ones(m, dtype=float)
    else:
        severity = np.asarray(severity, dtype=float)
        if severity.shape != (m,):
            raise ValueError("severity must have one entry per fault")

    ordered_cost = cost[ordering]
    cumulative = np.concatenate(([0.0], np.cumsum(ordered_cost)))  # cumulative[p] = cost of first p tests
    total_cost = cumulative[-1]
    if total_cost <= 0:
        raise ValueError("total execution cost must be positive")

    tf = _first_detection_positions(ordering, fault_matrix).astype(int)  # 1-indexed positions
    cost_to_detect = cumulative[tf - 1] + 0.5 * ordered_cost[tf - 1]

    total_severity = severity.sum()
    if total_severity <= 0:
        raise ValueError("sum of fault severities must be positive")

    return float((severity * (total_cost - cost_to_detect)).sum() / (total_cost * total_severity))


def coverage_effectiveness(
    ordering: Ordering,
    coverage_matrix: np.ndarray,
    cost: np.ndarray | None = None,
    budget: float | None = None,
) -> float:
    """Coverage Effectiveness (CE): fraction of coverable elements
    (statements/branches) covered by the prefix of `ordering` whose
    cumulative cost fits within `budget` (Section 2.4.3, p.36).

    If `budget` is None (or `cost` is None), CE is computed over the whole
    ordering, i.e. the suite's total achievable coverage.
    """
    coverage_matrix = np.asarray(coverage_matrix, dtype=bool)
    total_elements = coverage_matrix.shape[1]
    if total_elements == 0:
        return 0.0

    if budget is None or cost is None:
        included = ordering
    else:
        cost = np.asarray(cost, dtype=float)
        ordered_cost = cost[ordering]
        cumulative = np.cumsum(ordered_cost)
        cutoff = np.searchsorted(cumulative, budget, side="right")
        included = ordering[:cutoff]

    if len(included) == 0:
        return 0.0

    covered = coverage_matrix[included].any(axis=0)
    return float(covered.sum()) / float(total_elements)
