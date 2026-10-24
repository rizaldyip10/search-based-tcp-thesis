"""Synthetic test-suite generation, used to validate the algorithms before
real datasets (Siemens Test Suite / Defects4J) are wired in.
"""

from __future__ import annotations

import numpy as np

from tcp.suite import TestSuite


def generate_synthetic_suite(
    n_tests: int = 40,
    n_faults: int = 8,
    n_elements: int = 100,
    budget_fraction: float = 0.5,
    new_or_modified_fraction: float = 0.2,
    coverage_density: float = 0.08,
    rng: np.random.Generator | None = None,
    name: str = "synthetic",
) -> TestSuite:
    """Generate a random but well-formed TestSuite for testing/validation.

    Guarantees every fault is detected by at least one test and every
    statement is covered by at least one test, so metrics never degenerate.
    """
    rng = rng if rng is not None else np.random.default_rng()

    coverage_matrix = rng.random((n_tests, n_elements)) < coverage_density
    # ensure every element is covered by at least one test
    for e in range(n_elements):
        if not coverage_matrix[:, e].any():
            coverage_matrix[rng.integers(0, n_tests), e] = True

    fault_matrix = np.zeros((n_faults, n_tests), dtype=bool)
    for f in range(n_faults):
        n_detectors = rng.integers(1, max(2, n_tests // 10))
        detectors = rng.choice(n_tests, size=n_detectors, replace=False)
        fault_matrix[f, detectors] = True

    cost = rng.uniform(0.5, 5.0, size=n_tests)
    severity = rng.uniform(0.0, 1.0, size=n_faults)
    change_freq = rng.uniform(0.0, 1.0, size=n_faults)

    n_new = max(1, int(round(new_or_modified_fraction * n_tests)))
    is_new_or_modified = np.zeros(n_tests, dtype=bool)
    is_new_or_modified[rng.choice(n_tests, size=n_new, replace=False)] = True

    budget = float(cost.sum() * budget_fraction)

    return TestSuite(
        fault_matrix=fault_matrix,
        coverage_matrix=coverage_matrix,
        cost=cost,
        severity=severity,
        change_freq=change_freq,
        budget=budget,
        is_new_or_modified=is_new_or_modified,
        name=name,
    )


def generate_sprint_sequence(
    n_sprints: int = 4,
    n_tests_per_sprint: int = 40,
    growth_per_sprint: int = 5,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> list[TestSuite]:
    """Generate a sequence of synthetic suites simulating test-suite growth
    and volatility across sprints (mirrors the Defects4J sprint-mapping idea
    in Section 3.4.1.2, but with synthetic data).
    """
    rng = rng if rng is not None else np.random.default_rng()
    suites = []
    for i in range(n_sprints):
        n_tests = n_tests_per_sprint + i * growth_per_sprint
        suites.append(
            generate_synthetic_suite(n_tests=n_tests, rng=rng, name=f"sprint_{i}", **kwargs)
        )
    return suites
