"""Defects4J requires the external `defects4j` CLI (a multi-GB, non-trivial
setup -- see tcp/data/defects4j.py's module docstring), so the integration
tests here are skipped unless it's actually installed. The availability
check itself and the pure-Python heuristics run unconditionally.
"""

import numpy as np
import pytest

from tcp.data.defects4j import (
    change_frequency_from_commit_count,
    coverage_matrix_heuristic,
    defects4j_available,
    fault_severity_from_diff_size,
)


def test_defects4j_available_returns_bool():
    assert isinstance(defects4j_available(), bool)


def test_fault_severity_from_diff_size_is_monotonic_and_bounded():
    small = fault_severity_from_diff_size(2)
    medium = fault_severity_from_diff_size(50)
    large = fault_severity_from_diff_size(1000)
    assert 0.0 <= small < medium < large < 1.0


def test_fault_severity_zero_lines_gives_zero_severity():
    assert fault_severity_from_diff_size(0) == pytest.approx(0.0)


def test_change_frequency_from_commit_count_is_monotonic_and_bounded():
    rare = change_frequency_from_commit_count(1)
    common = change_frequency_from_commit_count(20)
    very_common = change_frequency_from_commit_count(200)
    assert 0.0 <= rare < common < very_common < 1.0


def test_change_frequency_zero_commits_gives_zero():
    assert change_frequency_from_commit_count(0) == pytest.approx(0.0)


def test_change_frequency_does_not_saturate_to_one_for_single_file_bugs():
    # Regression test: the first implementation normalised by the max commit
    # count among a bug's OWN touched files, which always degenerates to
    # exactly 1.0 for the common single-file-bug case -- losing all signal.
    single_file_commit_count = 3  # a file touched only 3 times ever
    assert change_frequency_from_commit_count(single_file_commit_count) < 0.3


def test_coverage_matrix_heuristic_matches_by_class_name():
    tests = ["org.example.FooTest::testBar", "org.example.BazTest::testQux"]
    modified_classes = ["org.example.Foo"]
    matrix = coverage_matrix_heuristic(tests, modified_classes)
    assert matrix.shape == (2, 1)
    assert matrix[0, 0] == True  # noqa: E712  # FooTest matches Foo
    assert matrix[1, 0] == False  # noqa: E712  # BazTest does not


@pytest.mark.skipif(not defects4j_available(), reason="defects4j CLI not installed")
def test_build_suite_for_bug_smoke():
    from tcp.data.defects4j import build_suite_for_bug

    suite = build_suite_for_bug("Lang", 1, base_workdir="/tmp/d4j_smoke", rng=np.random.default_rng(0))
    assert suite.n_faults == 1
    assert suite.n_tests > 0


@pytest.mark.skipif(not defects4j_available(), reason="defects4j CLI not installed")
def test_project_mirror_dir_resolves_for_known_projects():
    from tcp.data.defects4j import project_mirror_dir

    for project in ("Lang", "Math", "Chart"):
        mirror = project_mirror_dir(project)
        assert mirror is not None, f"{project} mirror should resolve when defects4j is installed"
        assert mirror.exists()


@pytest.mark.skipif(not defects4j_available(), reason="defects4j CLI not installed")
def test_change_frequency_uses_full_mirror_history_not_shallow_checkout():
    # Regression test: JFreeChart's per-bug checkout only has the handful of
    # synthetic D4J tag commits (no real history, since it's SVN-sourced),
    # which made every Chart bug's change_freq come out as the same
    # near-constant value. Across several real bugs, change_freq must vary.
    from tcp.data.defects4j import build_suite_for_bug

    chart_bugs = [1, 2, 3]
    values = [
        build_suite_for_bug("Chart", bid, "/tmp/d4j_mirror_regression", rng=np.random.default_rng(0)).change_freq[0]
        for bid in chart_bugs
    ]
    assert len(set(round(v, 3) for v in values)) > 1, (
        f"change_freq should differ across bugs touching different files, got {values}"
    )
