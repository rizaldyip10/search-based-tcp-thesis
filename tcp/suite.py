"""Data model for a regression test suite used in TCP experiments.

Mirrors the symbols defined in the thesis proposal (List of Symbols, p.15):
T        - test suite
TCj      - the j-th test case, cost(TCj) its execution cost
Fi       - the i-th fault, severity(Fi) and changeFreq(Fi) its attributes
B        - sprint time budget
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TestSuite:
    """A single-sprint snapshot of a regression test suite.

    Attributes:
        fault_matrix: bool array of shape (m_faults, n_tests). fault_matrix[i, j]
            is True iff test j detects fault i. Every fault must be detected by
            at least one test (standard TCP assumption).
        coverage_matrix: bool array of shape (n_tests, k_elements) recording
            which statements/branches each test covers. Used for Coverage
            Effectiveness (CE).
        cost: float array of shape (n_tests,), execution cost (e.g. seconds)
            of each test case, cost(TCj) in the proposal's notation.
        severity: float array of shape (m_faults,) in [0, 1], severity(Fi).
        change_freq: float array of shape (m_faults,) in [0, 1], changeFreq(Fi)
            -- the code-change frequency associated with the code area that
            fault i resides in.
        budget: B, the sprint time budget (max cumulative test execution time
            per CI build). If None, no budget constraint is enforced.
        is_new_or_modified: optional bool array of shape (n_tests,) marking
            tests that are new or modified in the current sprint. Used by the
            sprint re-initialisation mechanism (Section 3.3.2.3).
        name: human-readable identifier for logging/reporting.
    """

    fault_matrix: np.ndarray
    coverage_matrix: np.ndarray
    cost: np.ndarray
    severity: np.ndarray
    change_freq: np.ndarray
    budget: float | None = None
    is_new_or_modified: np.ndarray | None = None
    name: str = "suite"

    def __post_init__(self) -> None:
        self.fault_matrix = np.asarray(self.fault_matrix, dtype=bool)
        self.coverage_matrix = np.asarray(self.coverage_matrix, dtype=bool)
        self.cost = np.asarray(self.cost, dtype=float)
        self.severity = np.asarray(self.severity, dtype=float)
        self.change_freq = np.asarray(self.change_freq, dtype=float)

        m, n = self.fault_matrix.shape
        if self.coverage_matrix.shape[0] != n:
            raise ValueError(
                f"coverage_matrix has {self.coverage_matrix.shape[0]} rows, "
                f"expected {n} (n_tests) to match fault_matrix columns"
            )
        if self.cost.shape != (n,):
            raise ValueError(f"cost must have shape ({n},), got {self.cost.shape}")
        if self.severity.shape != (m,):
            raise ValueError(
                f"severity must have shape ({m},), got {self.severity.shape}"
            )
        if self.change_freq.shape != (m,):
            raise ValueError(
                f"change_freq must have shape ({m},), got {self.change_freq.shape}"
            )
        if not self.fault_matrix.any(axis=1).all():
            undetected = np.where(~self.fault_matrix.any(axis=1))[0]
            raise ValueError(
                f"fault(s) {undetected.tolist()} are not detected by any test case"
            )
        if self.is_new_or_modified is None:
            self.is_new_or_modified = np.zeros(n, dtype=bool)
        else:
            self.is_new_or_modified = np.asarray(self.is_new_or_modified, dtype=bool)
            if self.is_new_or_modified.shape != (n,):
                raise ValueError("is_new_or_modified must have shape (n_tests,)")

    @property
    def n_tests(self) -> int:
        return self.fault_matrix.shape[1]

    @property
    def n_faults(self) -> int:
        return self.fault_matrix.shape[0]

    @property
    def n_elements(self) -> int:
        """Number of coverage elements (statements/branches)."""
        return self.coverage_matrix.shape[1]

    def fault_weights(self, alpha_blend: float = 0.5) -> np.ndarray:
        """wi = alpha * severity(Fi) + (1 - alpha) * changeFreq(Fi). Eq. for wi, p.56."""
        if not 0.0 <= alpha_blend <= 1.0:
            raise ValueError("alpha_blend must be in [0, 1]")
        return alpha_blend * self.severity + (1.0 - alpha_blend) * self.change_freq
