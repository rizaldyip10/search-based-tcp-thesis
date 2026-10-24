"""Baseline TCP methods for comparison (Table 3.6, p.60-61)."""

from __future__ import annotations

import numpy as np

from tcp.fitness import unweighted_fitness
from tcp.ga import GAConfig, run_ga
from tcp.hybrid import HybridConfig, run_single_sprint
from tcp.suite import TestSuite


def random_order(suite: TestSuite, rng: np.random.Generator | None = None) -> np.ndarray:
    """Lower bound: a uniformly random execution order (Rothermel et al., 2002)."""
    rng = rng if rng is not None else np.random.default_rng()
    return rng.permutation(suite.n_tests)


def original_order(suite: TestSuite) -> np.ndarray:
    """Lower bound: unmodified development order, i.e. the identity
    permutation (Rothermel et al., 2002)."""
    return np.arange(suite.n_tests)


def greedy_coverage(suite: TestSuite) -> np.ndarray:
    """Coverage-based greedy: at each step, select the remaining test that
    covers the most still-uncovered statements/branches, ties broken by the
    lowest execution cost then original index (Rothermel et al., 2002)."""
    n = suite.n_tests
    remaining = set(range(n))
    covered = np.zeros(suite.n_elements, dtype=bool)
    ordering = np.empty(n, dtype=int)

    for pos in range(n):
        best_test = None
        best_gain = -1
        best_cost = np.inf
        for t in remaining:
            gain = int(np.count_nonzero(suite.coverage_matrix[t] & ~covered))
            if gain > best_gain or (gain == best_gain and suite.cost[t] < best_cost):
                best_test, best_gain, best_cost = t, gain, suite.cost[t]
        ordering[pos] = best_test
        covered |= suite.coverage_matrix[best_test]
        remaining.remove(best_test)

    return ordering


def standalone_ga(
    suite: TestSuite,
    config: GAConfig = GAConfig(),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Search-based: single-objective GA with APFD + coverage fitness only,
    no agile factors (Onyebuchi et al., 2025)."""
    rng = rng if rng is not None else np.random.default_rng()
    ordering, _, _ = run_ga(suite.n_tests, lambda o: unweighted_fitness(o, suite), config, rng)
    return ordering


def hybrid_ga_cso_no_agile(
    suite: TestSuite,
    config: HybridConfig = HybridConfig(),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Search-based hybrid: predecessor GA-CSO method (Onyebuchi et al.,
    2025), i.e. the same hybrid engine as the proposed method but with
    unweighted fitness and no sprint re-initialisation (single static run)."""
    rng = rng if rng is not None else np.random.default_rng()
    ordering, _, _, _ = run_single_sprint(
        suite.n_tests, lambda o: unweighted_fitness(o, suite), config, rng
    )
    return ordering
