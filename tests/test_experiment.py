import numpy as np
import pytest

from tcp.experiment import default_methods, run_trials, run_trials_multi_suite
from tcp.ga import GAConfig
from tcp.hybrid import HybridConfig
from tcp.synthetic import generate_synthetic_suite


@pytest.fixture
def suite():
    rng = np.random.default_rng(42)
    return generate_synthetic_suite(n_tests=20, n_faults=5, n_elements=50, rng=rng)


@pytest.fixture
def fast_methods():
    # small population/iterations so the test suite runs quickly
    return default_methods(
        ga_config=GAConfig(population_size=10, max_iterations=5),
        hybrid_config=HybridConfig(population_size=10, max_iterations=5),
    )


def test_run_trials_shapes_and_metric_ranges(suite, fast_methods):
    trial = run_trials(suite, fast_methods, n_runs=5, seed=0)

    assert trial.suite_name == suite.name
    assert set(trial.results.keys()) == set(fast_methods.keys())
    for method, metrics in trial.results.items():
        assert set(metrics.keys()) == {"APFD", "APFDc", "CE"}
        for metric_name, values in metrics.items():
            assert values.shape == (5,)
            assert np.all(values >= -0.01)  # APFD/APFDc/CE are essentially in [0,1]
            assert np.all(values <= 1.01)


def test_run_trials_is_reproducible_with_same_seed(suite, fast_methods):
    trial1 = run_trials(suite, fast_methods, n_runs=5, seed=123)
    trial2 = run_trials(suite, fast_methods, n_runs=5, seed=123)
    for method in fast_methods:
        assert np.allclose(trial1.results[method]["APFD"], trial2.results[method]["APFD"])


def test_run_trials_differs_with_different_seed(suite, fast_methods):
    trial1 = run_trials(suite, fast_methods, n_runs=5, seed=1)
    trial2 = run_trials(suite, fast_methods, n_runs=5, seed=2)
    # random_order should differ between seeds (it's pure randomness, no search)
    assert not np.allclose(
        trial1.results["random_order"]["APFD"], trial2.results["random_order"]["APFD"]
    )


def test_summary_returns_mean_and_std(suite, fast_methods):
    trial = run_trials(suite, fast_methods, n_runs=5, seed=0)
    summary = trial.summary()
    for method, metrics in summary.items():
        for metric_name, (mean, std) in metrics.items():
            assert isinstance(mean, float)
            assert isinstance(std, float)
            assert std >= 0.0


def test_run_trials_multi_suite(fast_methods):
    rng = np.random.default_rng(7)
    suites = [
        generate_synthetic_suite(n_tests=15, n_faults=3, rng=rng, name="s1"),
        generate_synthetic_suite(n_tests=15, n_faults=3, rng=rng, name="s2"),
    ]
    all_results = run_trials_multi_suite(suites, fast_methods, n_runs=4, seed=0)
    assert set(all_results.keys()) == {"s1", "s2"}
    for trial in all_results.values():
        assert trial.n_runs == 4
