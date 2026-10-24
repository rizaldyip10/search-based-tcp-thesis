"""Composite multi-objective fitness function (Section 3.3.2.2, Eq. 3.1-3.3).

    F = alpha1 * wAPFD + alpha2 * CE - alpha3 * Penalty(cost)

NOTE ON Penalty(cost): the proposal defines it as

    Penalty(cost) = max(0, sum_j(cost(TCj)) - B) / B                 (Eq. 3.3)

summed over the WHOLE test suite T, not over a prefix of the candidate
ordering. Read literally, this term is constant across every permutation of
a fixed test suite (reordering doesn't change the total cost), so it cannot
by itself steer the search away from budget-violating orderings -- only the
CE cutoff (which truncates at the budget) is order-sensitive. We implement
the formula exactly as written so results match the thesis's stated model,
but this is worth confirming with your supervisor: a prefix-cost penalty
(penalising orderings whose cumulative cost to detect all/most faults
exceeds B) would make Penalty ordering-sensitive too, which may be closer to
the intended effect. See `penalty_cost_prefix` below for that alternative,
provided but not used by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tcp.metrics import coverage_effectiveness, weighted_apfd
from tcp.suite import TestSuite


@dataclass(frozen=True)
class FitnessWeights:
    """alpha1, alpha2, alpha3 in Eq. 3.1 (p.57); must sum to 1 and be positive."""

    alpha1: float = 0.6
    alpha2: float = 0.3
    alpha3: float = 0.1

    def __post_init__(self) -> None:
        total = self.alpha1 + self.alpha2 + self.alpha3
        if not np.isclose(total, 1.0):
            raise ValueError(f"alpha1+alpha2+alpha3 must equal 1, got {total}")
        if min(self.alpha1, self.alpha2, self.alpha3) <= 0:
            raise ValueError("alpha1, alpha2, alpha3 must all be > 0")


def penalty_cost(cost: np.ndarray, budget: float | None) -> float:
    """Penalty(cost) exactly as defined in Eq. 3.3 (suite-total, order-independent)."""
    if budget is None or budget <= 0:
        return 0.0
    total_cost = float(np.sum(cost))
    return max(0.0, total_cost - budget) / budget


def penalty_cost_prefix(ordering: np.ndarray, cost: np.ndarray, budget: float | None) -> float:
    """Alternative, order-sensitive penalty: how far the ordering's cumulative
    cost overshoots the budget once accounting for position (not used by
    default; see module docstring). Returns 0 when the whole ordering fits.
    """
    if budget is None or budget <= 0:
        return 0.0
    ordered_cost = cost[ordering]
    total_cost = float(np.sum(ordered_cost))
    return max(0.0, total_cost - budget) / budget


def composite_fitness(
    ordering: np.ndarray,
    suite: TestSuite,
    weights: FitnessWeights = FitnessWeights(),
    alpha_blend: float = 0.5,
) -> float:
    """F = alpha1 * wAPFD + alpha2 * CE - alpha3 * Penalty(cost) (Eq. 3.1, p.57)."""
    fault_weights = suite.fault_weights(alpha_blend)
    wapfd = weighted_apfd(ordering, suite.fault_matrix, fault_weights)
    ce = coverage_effectiveness(ordering, suite.coverage_matrix, suite.cost, suite.budget)
    pen = penalty_cost(suite.cost, suite.budget)
    return weights.alpha1 * wapfd + weights.alpha2 * ce - weights.alpha3 * pen


def unweighted_fitness(ordering: np.ndarray, suite: TestSuite) -> float:
    """Fitness for agile-unaware baselines (standalone GA, predecessor hybrid
    GA-CSO): standard APFD combined with CE, no severity/change-frequency
    weighting and no sprint-budget penalty -- i.e. wi = 1 for every fault.
    """
    uniform_weights = np.ones(suite.n_faults)
    wapfd = weighted_apfd(ordering, suite.fault_matrix, uniform_weights)
    ce = coverage_effectiveness(ordering, suite.coverage_matrix, suite.cost, suite.budget)
    return 0.7 * wapfd + 0.3 * ce
