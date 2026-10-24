import numpy as np

from tcp.experiment import TrialResults
from tcp.report import format_comparison, format_full_report, format_summary_table
from tcp.stats import analyze_trial_results


def _demo_trial():
    rng = np.random.default_rng(0)
    results = {
        "random_order": {
            "APFD": rng.normal(0.4, 0.05, 10),
            "APFDc": rng.normal(0.4, 0.05, 10),
            "CE": rng.normal(0.6, 0.05, 10),
        },
        "proposed_method": {
            "APFD": rng.normal(0.9, 0.02, 10),
            "APFDc": rng.normal(0.9, 0.02, 10),
            "CE": rng.normal(0.95, 0.02, 10),
        },
    }
    return TrialResults(suite_name="demo_suite", n_runs=10, results=results)


def test_format_summary_table_contains_method_names_and_suite_name():
    trial = _demo_trial()
    text = format_summary_table(trial)
    assert "demo_suite" in text
    assert "random_order" in text
    assert "proposed_method" in text
    assert "APFD" in text


def test_format_comparison_contains_test_name_and_pvalue():
    trial = _demo_trial()
    comparisons = analyze_trial_results(trial)
    text = format_comparison(comparisons["APFD"])
    assert "omnibus test" in text
    assert "p=" in text


def test_format_full_report_combines_summary_and_comparisons():
    trial = _demo_trial()
    comparisons = analyze_trial_results(trial)
    text = format_full_report(trial, comparisons)
    assert "demo_suite" in text
    assert "metric: APFD" in text
    assert "metric: APFDc" in text
    assert "metric: CE" in text
