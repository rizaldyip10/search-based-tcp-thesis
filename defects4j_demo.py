"""End-to-end demo with REAL Defects4J data (Section 3.4.1.2): builds a
small sprint sequence from actual Commons Lang bugs, caches it to disk, and
runs both the proposed method across sprints (exercising sprint
re-initialisation on genuinely volatile test suites) and a same-suite
Phase 3 comparison against the baselines.

Requires `defects4j` on PATH (see tcp/data/defects4j.py's module docstring
for setup).
"""

import numpy as np

from tcp.data.defects4j import build_sprint_sequence
from tcp.data.files import save_test_suite
from tcp.experiment import default_methods, run_trials
from tcp.ga import GAConfig
from tcp.hybrid import HybridConfig
from tcp.method import proposed_method_across_sprints
from tcp.report import format_full_report
from tcp.stats import analyze_trial_results

BUG_IDS = [1, 3, 4, 5, 6]

print(f"Checking out Commons Lang bugs {BUG_IDS} (one simulated sprint each)...")
suites = build_sprint_sequence(
    "Lang", BUG_IDS, base_workdir="/tmp/d4j_sprint_demo", rng=np.random.default_rng(0)
)

for i, suite in enumerate(suites):
    print(
        f"  sprint {i}: {suite.name:<8} n_tests={suite.n_tests:<3} "
        f"severity={suite.severity[0]:.3f} change_freq={suite.change_freq[0]:.3f}"
    )
    save_test_suite(suite, f"data_cache/defects4j_lang/sprint_{i:02d}")

print("\nRunning the proposed method across the sprint sequence "
      "(sprint re-initialisation active)...")
hybrid_config = HybridConfig(population_size=20, max_iterations=20)
results = proposed_method_across_sprints(suites, hybrid_config, rng=np.random.default_rng(0))
for r in results:
    print(f"  sprint {r['sprint']}: best fitness = {r['fitness']:.4f}")

print("\nPhase 3 comparison (30 runs x 6 methods) on sprint 0's suite "
      f"({suites[0].name})...")
methods = default_methods(
    ga_config=GAConfig(population_size=20, max_iterations=20),
    hybrid_config=hybrid_config,
)
trial = run_trials(suites[0], methods, n_runs=30, seed=0)
comparisons = analyze_trial_results(trial)
print(format_full_report(trial, comparisons))
