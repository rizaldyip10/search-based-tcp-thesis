"""End-to-end Phase 3 smoke test (Section 3.3.3.3): run all 6 methods 30
times on one synthetic suite, compute APFD/APFDc/CE, and run the full
Shapiro-Wilk -> ANOVA/Kruskal-Wallis -> Tukey HSD pipeline -- all before
real datasets are wired in, to confirm the statistics plumbing is correct.
"""

import numpy as np

from tcp.experiment import default_methods, run_trials
from tcp.ga import GAConfig
from tcp.hybrid import HybridConfig
from tcp.report import format_full_report
from tcp.stats import analyze_trial_results
from tcp.synthetic import generate_synthetic_suite

suite = generate_synthetic_suite(
    n_tests=40, n_faults=10, n_elements=120, rng=np.random.default_rng(2024), name="synthetic_demo"
)

methods = default_methods(
    ga_config=GAConfig(population_size=30, max_iterations=40),
    hybrid_config=HybridConfig(population_size=30, max_iterations=40),
)

trial = run_trials(suite, methods, n_runs=30, seed=2024)
comparisons = analyze_trial_results(trial, alpha=0.05)

print(format_full_report(trial, comparisons))
