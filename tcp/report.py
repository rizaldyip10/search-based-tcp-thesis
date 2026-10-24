"""Plain-text report formatting for TrialResults + ComparisonResult, so
Phase 3 output can be read directly in a terminal or captured into the
thesis's results chapter without pulling in a plotting/table dependency.
"""

from __future__ import annotations

from tcp.experiment import TrialResults
from tcp.stats import ComparisonResult


def format_summary_table(trial: TrialResults) -> str:
    """One row per method, mean +/- std for each metric (Section 3.3.3.3
    step 4)."""
    summary = trial.summary()
    methods = list(summary.keys())
    metrics = list(next(iter(summary.values())).keys())

    header = f"{'method':<24}" + "".join(f"{m:>16}" for m in metrics)
    lines = [f"suite: {trial.suite_name}  (n_runs={trial.n_runs})", header, "-" * len(header)]
    for method in methods:
        row = f"{method:<24}"
        for metric in metrics:
            mean, std = summary[method][metric]
            row += f"{mean:>9.4f}+-{std:<5.4f}"
        lines.append(row)
    return "\n".join(lines)


def format_comparison(result: ComparisonResult) -> str:
    lines = [
        f"metric: {result.metric}  alpha={result.alpha}",
        f"  Shapiro-Wilk p-values: "
        + ", ".join(f"{k}={v:.4f}" for k, v in result.shapiro_p.items()),
        f"  all groups normal: {result.all_groups_normal}",
        f"  omnibus test: {result.omnibus_test}  "
        f"statistic={result.omnibus_statistic:.4f}  p={result.omnibus_p:.4g}  "
        f"significant={result.significant}",
    ]
    if result.posthoc_caveat:
        lines.append(f"  CAVEAT: {result.posthoc_caveat}")
    if result.posthoc:
        lines.append("  Tukey HSD pairwise comparisons:")
        for pair in result.posthoc:
            lines.append(
                f"    {pair.group1:<22} vs {pair.group2:<22} "
                f"meandiff={pair.mean_diff:+.4f} p_adj={pair.p_adj:.4g} "
                f"reject_null={pair.reject_null}"
            )
    return "\n".join(lines)


def format_full_report(trial: TrialResults, comparisons: dict[str, ComparisonResult]) -> str:
    sections = [format_summary_table(trial), ""]
    for metric, comparison in comparisons.items():
        sections.append(format_comparison(comparison))
        sections.append("")
    return "\n".join(sections)
