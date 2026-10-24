import numpy as np
import pytest

from tcp.stats import ALPHA_DEFAULT, analyze_trial_results, compare_methods


def test_compare_methods_detects_significant_difference():
    rng = np.random.default_rng(0)
    groups = {
        "low": rng.normal(loc=0.3, scale=0.02, size=30),
        "high": rng.normal(loc=0.9, scale=0.02, size=30),
    }
    result = compare_methods(groups, metric="APFD")
    assert result.omnibus_test in ("ANOVA", "Kruskal-Wallis")
    assert result.significant is True
    assert result.omnibus_p < ALPHA_DEFAULT
    assert len(result.posthoc) == 1  # one pair for two groups
    assert result.posthoc[0].reject_null is True


def test_compare_methods_detects_no_difference_for_identical_distributions():
    rng = np.random.default_rng(1)
    groups = {
        "a": rng.normal(loc=0.5, scale=0.05, size=30),
        "b": rng.normal(loc=0.5, scale=0.05, size=30),
    }
    result = compare_methods(groups, metric="APFD")
    assert result.significant is False
    assert result.posthoc == []


def test_compare_methods_uses_anova_for_normal_data():
    rng = np.random.default_rng(2)
    groups = {
        "a": rng.normal(loc=0.5, scale=0.05, size=40),
        "b": rng.normal(loc=0.5, scale=0.05, size=40),
        "c": rng.normal(loc=0.5, scale=0.05, size=40),
    }
    result = compare_methods(groups, metric="APFD")
    assert result.all_groups_normal is True
    assert result.omnibus_test == "ANOVA"


def test_compare_methods_uses_kruskal_for_non_normal_data():
    rng = np.random.default_rng(3)
    # uniform distributions reliably fail Shapiro-Wilk's normality test
    groups = {
        "a": rng.uniform(0, 1, size=40),
        "b": rng.uniform(0, 1, size=40),
    }
    result = compare_methods(groups, metric="APFD")
    assert result.all_groups_normal is False
    assert result.omnibus_test == "Kruskal-Wallis"
    assert result.posthoc_caveat is None or result.significant  # caveat only set if significant


def test_compare_methods_requires_at_least_two_groups():
    with pytest.raises(ValueError):
        compare_methods({"only_one": np.array([1.0, 2.0, 3.0])})


def test_analyze_trial_results_covers_all_metrics():
    from tcp.experiment import TrialResults

    rng = np.random.default_rng(4)
    results = {
        "method_a": {
            "APFD": rng.normal(0.5, 0.05, 30),
            "APFDc": rng.normal(0.5, 0.05, 30),
            "CE": rng.normal(0.5, 0.05, 30),
        },
        "method_b": {
            "APFD": rng.normal(0.8, 0.05, 30),
            "APFDc": rng.normal(0.8, 0.05, 30),
            "CE": rng.normal(0.8, 0.05, 30),
        },
    }
    trial = TrialResults(suite_name="demo", n_runs=30, results=results)
    comparisons = analyze_trial_results(trial)
    assert set(comparisons.keys()) == {"APFD", "APFDc", "CE"}
    for comparison in comparisons.values():
        assert comparison.significant is True
