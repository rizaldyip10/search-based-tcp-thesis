import numpy as np
import pytest

from tcp.data.files import load_sprint_sequence, load_test_suite, save_test_suite
from tcp.synthetic import generate_sprint_sequence, generate_synthetic_suite


def test_round_trip_single_suite(tmp_path):
    rng = np.random.default_rng(0)
    suite = generate_synthetic_suite(n_tests=12, n_faults=4, n_elements=30, rng=rng, name="demo")

    out_dir = tmp_path / "demo"
    save_test_suite(suite, out_dir)
    loaded = load_test_suite(out_dir)

    assert loaded.name == "demo"
    assert loaded.n_tests == suite.n_tests
    assert loaded.n_faults == suite.n_faults
    assert np.array_equal(loaded.fault_matrix, suite.fault_matrix)
    assert np.array_equal(loaded.coverage_matrix, suite.coverage_matrix)
    assert np.allclose(loaded.cost, suite.cost)
    assert np.allclose(loaded.severity, suite.severity)
    assert np.allclose(loaded.change_freq, suite.change_freq)
    assert np.array_equal(loaded.is_new_or_modified, suite.is_new_or_modified)
    assert loaded.budget == pytest.approx(suite.budget)


def test_round_trip_sprint_sequence(tmp_path):
    rng = np.random.default_rng(1)
    suites = generate_sprint_sequence(n_sprints=3, n_tests_per_sprint=10, rng=rng)

    root = tmp_path / "sprints"
    for i, s in enumerate(suites):
        save_test_suite(s, root / f"sprint_{i:02d}")

    loaded = load_sprint_sequence(root)
    assert len(loaded) == 3
    for original, reloaded in zip(suites, loaded):
        assert original.n_tests == reloaded.n_tests
        assert np.array_equal(original.fault_matrix, reloaded.fault_matrix)


def test_load_test_suite_without_optional_new_or_modified_file(tmp_path):
    rng = np.random.default_rng(2)
    suite = generate_synthetic_suite(n_tests=8, n_faults=2, rng=rng)
    out_dir = tmp_path / "no_optional"
    save_test_suite(suite, out_dir)
    (out_dir / "new_or_modified.csv").unlink()

    loaded = load_test_suite(out_dir)
    assert not loaded.is_new_or_modified.any()  # defaults to all-False
