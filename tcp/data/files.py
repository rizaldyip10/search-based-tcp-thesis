"""Generic, dataset-agnostic file format for a TestSuite. Both the Siemens
and Defects4J extraction pipelines write into this format, so the rest of
the codebase (experiment runner, Phase 3 statistics) never needs to know
which dataset a suite came from.

Directory layout for one suite, e.g. `data/tcas_v1/`:

    meta.json           {"name": "...", "budget": 12.5 or null}
    coverage.csv         (n_tests x n_elements) 0/1 matrix, no header
    faults.csv           (n_faults x n_tests) 0/1 matrix, no header
    cost.csv              n_tests values, one per line
    severity.csv          n_faults values, one per line
    change_freq.csv       n_faults values, one per line
    new_or_modified.csv   n_tests values of 0/1, one per line (optional)
    test_names.txt         n_tests labels, one per line (optional, for reporting)
    fault_names.txt        n_faults labels, one per line (optional, for reporting)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tcp.suite import TestSuite


def _load_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    return data


def _load_vector(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=1)


def _save_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%g")


def _save_vector(path: Path, vector: np.ndarray) -> None:
    np.savetxt(path, vector, delimiter=",", fmt="%g")


def load_test_suite(directory: str | Path) -> TestSuite:
    """Load a TestSuite from the directory layout documented above."""
    directory = Path(directory)
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    fault_matrix = _load_matrix(directory / "faults.csv").astype(bool)
    coverage_matrix = _load_matrix(directory / "coverage.csv").astype(bool)
    cost = _load_vector(directory / "cost.csv")
    severity = _load_vector(directory / "severity.csv")
    change_freq = _load_vector(directory / "change_freq.csv")

    new_or_modified_path = directory / "new_or_modified.csv"
    is_new_or_modified = (
        _load_vector(new_or_modified_path).astype(bool)
        if new_or_modified_path.exists()
        else None
    )

    return TestSuite(
        fault_matrix=fault_matrix,
        coverage_matrix=coverage_matrix,
        cost=cost,
        severity=severity,
        change_freq=change_freq,
        budget=meta.get("budget"),
        is_new_or_modified=is_new_or_modified,
        name=meta.get("name", directory.name),
    )


def save_test_suite(suite: TestSuite, directory: str | Path) -> None:
    """Write a TestSuite to the directory layout documented above, creating
    the directory if needed. Useful for caching the (expensive) output of
    the Siemens/Defects4J extraction pipelines so experiments don't need to
    recompile/recheckout every run.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "meta.json").write_text(
        json.dumps({"name": suite.name, "budget": suite.budget}, indent=2)
    )
    _save_matrix(directory / "faults.csv", suite.fault_matrix.astype(int))
    _save_matrix(directory / "coverage.csv", suite.coverage_matrix.astype(int))
    _save_vector(directory / "cost.csv", suite.cost)
    _save_vector(directory / "severity.csv", suite.severity)
    _save_vector(directory / "change_freq.csv", suite.change_freq)
    _save_vector(directory / "new_or_modified.csv", suite.is_new_or_modified.astype(int))


def load_sprint_sequence(root_directory: str | Path, sprint_dirs: list[str] | None = None) -> list[TestSuite]:
    """Load a sequence of per-sprint suites from subdirectories of
    `root_directory`. If `sprint_dirs` is omitted, every subdirectory
    containing a meta.json is loaded, sorted by name (so sprint folders
    should be named e.g. sprint_00, sprint_01, ... to sort correctly).
    """
    root_directory = Path(root_directory)
    if sprint_dirs is None:
        sprint_dirs = sorted(
            d.name for d in root_directory.iterdir() if (d / "meta.json").exists()
        )
    return [load_test_suite(root_directory / d) for d in sprint_dirs]
