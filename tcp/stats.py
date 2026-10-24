"""Statistical analysis plan (Section 3.3.3.4, p.62):

    Shapiro-Wilk (normality, per group) -> one-way ANOVA (if all groups
    normal) or Kruskal-Wallis (otherwise) -> Tukey HSD post-hoc, all at
    alpha = 0.05.

NOTE ON THE Kruskal-Wallis -> Tukey HSD PATH: Tukey HSD's own assumptions
(normality, homogeneity of variance) are exactly what triggered falling
back to Kruskal-Wallis in the first place, so running it after a
*significant* Kruskal-Wallis result is statistically inconsistent -- the
standard non-parametric follow-up to Kruskal-Wallis is a rank-based
procedure such as Dunn's test (not available in SciPy/statsmodels without
an extra dependency like `scikit-posthocs`, which isn't in the proposal's
tool table, Table 3.4). This module follows the proposal's pipeline
literally (Table 3.1: "Shapiro-Wilk -> ANOVA / Kruskal-Wallis -> Tukey
HSD") and still runs Tukey HSD after a significant Kruskal-Wallis result,
but flags this explicitly in `ComparisonResult.posthoc_caveat` -- worth
confirming with your supervisor whether Dunn's test should be substituted
for that branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ALPHA_DEFAULT = 0.05


@dataclass
class PosthocPair:
    group1: str
    group2: str
    mean_diff: float
    p_adj: float
    ci_lower: float
    ci_upper: float
    reject_null: bool  # True => groups significantly differ at alpha


@dataclass
class ComparisonResult:
    metric: str
    alpha: float
    group_means: dict[str, float]
    group_stds: dict[str, float]
    shapiro_p: dict[str, float]
    all_groups_normal: bool
    omnibus_test: str  # "ANOVA" or "Kruskal-Wallis"
    omnibus_statistic: float
    omnibus_p: float
    significant: bool
    posthoc: list[PosthocPair] = field(default_factory=list)
    posthoc_caveat: str | None = None


def _shapiro_per_group(groups: dict[str, np.ndarray], alpha: float) -> tuple[dict[str, float], bool]:
    p_values: dict[str, float] = {}
    for name, values in groups.items():
        # Shapiro-Wilk needs n>=3 and non-zero variance. Deterministic
        # methods (original_order/greedy_coverage) produce bit-identical
        # values on every run, but np.std() of identical floats can still
        # be a tiny nonzero residue (~1e-16) from floating-point rounding
        # in the variance computation, which is enough to make scipy divide
        # by (near-)zero internally -- so check exact/near equality directly
        # rather than trusting std().
        if len(values) < 3 or np.all(values == values[0]) or np.std(values) < 1e-12:
            p_values[name] = 0.0
            continue
        _, p = scipy_stats.shapiro(values)
        p_values[name] = float(p)
    all_normal = all(p > alpha for p in p_values.values())
    return p_values, all_normal


def _run_tukey_hsd(groups: dict[str, np.ndarray], alpha: float) -> list[PosthocPair]:
    all_values = np.concatenate(list(groups.values()))
    all_labels = np.concatenate(
        [np.full(len(v), name, dtype=object) for name, v in groups.items()]
    )
    result = pairwise_tukeyhsd(endog=all_values, groups=all_labels, alpha=alpha)
    pairs = []
    for row in result.summary().data[1:]:  # skip header row
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        pairs.append(
            PosthocPair(
                group1=str(group1),
                group2=str(group2),
                mean_diff=float(meandiff),
                p_adj=float(p_adj),
                ci_lower=float(lower),
                ci_upper=float(upper),
                reject_null=bool(reject),
            )
        )
    return pairs


def compare_methods(
    groups: dict[str, np.ndarray], metric: str = "metric", alpha: float = ALPHA_DEFAULT
) -> ComparisonResult:
    """Run the full Shapiro-Wilk -> ANOVA/Kruskal-Wallis -> Tukey HSD
    pipeline on a {method_name: array_of_run_values} dict for one metric.
    """
    if len(groups) < 2:
        raise ValueError("need at least two groups (methods) to compare")

    shapiro_p, all_normal = _shapiro_per_group(groups, alpha)
    group_values = list(groups.values())

    if all_normal:
        statistic, p_value = scipy_stats.f_oneway(*group_values)
        omnibus_test = "ANOVA"
    else:
        statistic, p_value = scipy_stats.kruskal(*group_values)
        omnibus_test = "Kruskal-Wallis"

    significant = bool(p_value < alpha)
    posthoc: list[PosthocPair] = []
    posthoc_caveat = None
    if significant:
        posthoc = _run_tukey_hsd(groups, alpha)
        if omnibus_test == "Kruskal-Wallis":
            posthoc_caveat = (
                "Tukey HSD assumes normality/homogeneity of variance, which this "
                "branch's Shapiro-Wilk result rejected; treat these pairwise "
                "p-values as approximate and consider a rank-based post-hoc "
                "(e.g. Dunn's test) instead -- see module docstring."
            )

    return ComparisonResult(
        metric=metric,
        alpha=alpha,
        group_means={name: float(np.mean(v)) for name, v in groups.items()},
        group_stds={name: float(np.std(v, ddof=1)) for name, v in groups.items()},
        shapiro_p=shapiro_p,
        all_groups_normal=all_normal,
        omnibus_test=omnibus_test,
        omnibus_statistic=float(statistic),
        omnibus_p=float(p_value),
        significant=significant,
        posthoc=posthoc,
        posthoc_caveat=posthoc_caveat,
    )


def analyze_trial_results(trial_results, alpha: float = ALPHA_DEFAULT) -> dict[str, ComparisonResult]:
    """Run `compare_methods` for every metric in a `tcp.experiment.TrialResults`."""
    metrics = next(iter(trial_results.results.values())).keys()
    out: dict[str, ComparisonResult] = {}
    for metric in metrics:
        groups = {method: data[metric] for method, data in trial_results.results.items()}
        out[metric] = compare_methods(groups, metric=metric, alpha=alpha)
    return out
