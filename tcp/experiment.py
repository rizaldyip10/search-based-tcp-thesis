"""Phase 3 experimental procedure (Section 3.3.3.3, p.61-62):

    1. Prepare dataset inputs           -- Section 3.4 / tcp.data
    2. Run all six methods 30 times each
    3. Compute APFD, APFDc, CE for each run
    4. Compute the 30-run mean and standard deviation
    5. Apply statistical significance tests  -- tcp.stats

This module implements steps 2-4; `tcp.stats` implements step 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from tcp.baselines import (
    greedy_coverage,
    hybrid_ga_cso_no_agile,
    original_order,
    random_order,
    standalone_ga,
)
from tcp.ga import GAConfig
from tcp.hybrid import HybridConfig
from tcp.method import proposed_method
from tcp.metrics import apfd, apfdc, coverage_effectiveness
from tcp.suite import TestSuite

MethodFn = Callable[[TestSuite, np.random.Generator], np.ndarray]
METRIC_NAMES = ("APFD", "APFDc", "CE")


def default_methods(
    ga_config: GAConfig = GAConfig(),
    hybrid_config: HybridConfig = HybridConfig(),
) -> dict[str, MethodFn]:
    """The six Table 3.6 methods, each exposed as a (suite, rng) -> ordering
    callable so the runner below can treat them uniformly."""
    return {
        "random_order": lambda suite, rng: random_order(suite, rng),
        "original_order": lambda suite, rng: original_order(suite),
        "greedy_coverage": lambda suite, rng: greedy_coverage(suite),
        "standalone_ga": lambda suite, rng: standalone_ga(suite, ga_config, rng),
        "hybrid_ga_cso_no_agile": lambda suite, rng: hybrid_ga_cso_no_agile(
            suite, hybrid_config, rng
        ),
        "proposed_method": lambda suite, rng: proposed_method(suite, hybrid_config, rng=rng),
    }


def _evaluate_ordering(ordering: np.ndarray, suite: TestSuite) -> dict[str, float]:
    return {
        "APFD": apfd(ordering, suite.fault_matrix),
        "APFDc": apfdc(ordering, suite.fault_matrix, suite.cost, suite.severity),
        "CE": coverage_effectiveness(ordering, suite.coverage_matrix, suite.cost, suite.budget),
    }


@dataclass
class TrialResults:
    """Raw per-run metric values: results[method][metric] is an array of
    length n_runs (Section 3.3.3.3, steps 2-3)."""

    suite_name: str
    n_runs: int
    results: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def summary(self) -> dict[str, dict[str, tuple[float, float]]]:
        """Step 4: 30-run mean and standard deviation, per method per metric."""
        return {
            method: {
                metric: (float(np.mean(values)), float(np.std(values, ddof=1)))
                for metric, values in metrics.items()
            }
            for method, metrics in self.results.items()
        }


def run_trials(
    suite: TestSuite,
    methods: dict[str, MethodFn] | None = None,
    n_runs: int = 30,
    seed: int | None = None,
) -> TrialResults:
    """Run every method `n_runs` independent times on `suite` (Section
    3.3.3.3, steps 1-3). Each run gets its own independently-seeded RNG
    stream (via SeedSequence.spawn) so runs are reproducible yet not
    accidentally correlated across methods or runs.
    """
    methods = methods if methods is not None else default_methods()
    seed_sequence = np.random.SeedSequence(seed)
    results: dict[str, dict[str, np.ndarray]] = {}

    for method_name, method_fn in methods.items():
        per_metric: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
        run_seeds = seed_sequence.spawn(n_runs)
        for run_seed in run_seeds:
            rng = np.random.default_rng(run_seed)
            ordering = method_fn(suite, rng)
            scores = _evaluate_ordering(ordering, suite)
            for metric_name, value in scores.items():
                per_metric[metric_name].append(value)
        results[method_name] = {m: np.array(v) for m, v in per_metric.items()}

    return TrialResults(suite_name=suite.name, n_runs=n_runs, results=results)


def run_trials_multi_suite(
    suites: list[TestSuite],
    methods: dict[str, MethodFn] | None = None,
    n_runs: int = 30,
    seed: int | None = None,
) -> dict[str, TrialResults]:
    """Run `run_trials` independently over several suites (e.g. the five
    Siemens programs), keyed by suite name. Each suite gets its own
    seed-sequence child so results stay reproducible but independent."""
    methods = methods if methods is not None else default_methods()
    seed_sequence = np.random.SeedSequence(seed)
    suite_seeds = seed_sequence.spawn(len(suites))
    return {
        suite.name: run_trials(suite, methods, n_runs, seed=int(s.generate_state(1)[0]))
        for suite, s in zip(suites, suite_seeds)
    }
