"""End-to-end smoke test: run all six Table 3.6 methods on one synthetic
suite and report APFD / APFDc / CE, to sanity-check the pipeline before
wiring in real datasets.
"""

import numpy as np

from tcp.baselines import (
    greedy_coverage,
    hybrid_ga_cso_no_agile,
    original_order,
    random_order,
    standalone_ga,
)
from tcp.hybrid import HybridConfig
from tcp.metrics import apfd, apfdc, coverage_effectiveness
from tcp.method import proposed_method
from tcp.synthetic import generate_synthetic_suite

rng = np.random.default_rng(2024)
suite = generate_synthetic_suite(n_tests=40, n_faults=10, n_elements=120, rng=rng)
config = HybridConfig(population_size=40, max_iterations=60)

methods = {
    "random_order": lambda: random_order(suite, rng),
    "original_order": lambda: original_order(suite),
    "greedy_coverage": lambda: greedy_coverage(suite),
    "standalone_ga": lambda: standalone_ga(suite, rng=rng),
    "hybrid_ga_cso_no_agile": lambda: hybrid_ga_cso_no_agile(suite, config, rng),
    "proposed_method": lambda: proposed_method(suite, config, rng=rng),
}

print(f"suite: n_tests={suite.n_tests} n_faults={suite.n_faults} budget={suite.budget:.2f}\n")
print(f"{'method':<24}{'APFD':>8}{'APFDc':>8}{'CE':>8}")
for name, fn in methods.items():
    order = fn()
    a = apfd(order, suite.fault_matrix)
    ac = apfdc(order, suite.fault_matrix, suite.cost, suite.severity)
    ce = coverage_effectiveness(order, suite.coverage_matrix, suite.cost, suite.budget)
    print(f"{name:<24}{a:>8.4f}{ac:>8.4f}{ce:>8.4f}")
