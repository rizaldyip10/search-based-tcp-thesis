"""The proposed method: agile-aware hybrid GA-CSO with the composite
multi-objective fitness function (Section 3.3.2.3). Thin convenience
wrappers around `tcp.hybrid` for the single-sprint and multi-sprint cases.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tcp.fitness import FitnessWeights, composite_fitness
from tcp.hybrid import HybridConfig, run_across_sprints, run_single_sprint
from tcp.suite import TestSuite


def proposed_method(
    suite: TestSuite,
    config: HybridConfig = HybridConfig(),
    weights: FitnessWeights = FitnessWeights(),
    alpha_blend: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run the proposed method on a single static suite (used for the
    Siemens benchmark, Section 3.4.1.1, where there is no sprint sequence)."""
    rng = rng if rng is not None else np.random.default_rng()
    ordering, _, _, _ = run_single_sprint(
        suite.n_tests,
        lambda o: composite_fitness(o, suite, weights, alpha_blend),
        config,
        rng,
    )
    return ordering


def proposed_method_across_sprints(
    suites: Sequence[TestSuite],
    config: HybridConfig = HybridConfig(),
    weights: FitnessWeights = FitnessWeights(),
    alpha_blend: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Run the proposed method across a sprint sequence (used for the
    Defects4J agile-representative dataset, Section 3.4.1.2), exercising the
    sprint re-initialisation mechanism at each sprint boundary."""
    rng = rng if rng is not None else np.random.default_rng()

    def fitness_fn_factory(s: TestSuite):
        return lambda o: composite_fitness(o, s, weights, alpha_blend)

    return run_across_sprints(suites, fitness_fn_factory, config, rng)
