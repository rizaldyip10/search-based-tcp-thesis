"""Builds and caches sprint sequences from REAL Defects4J bugs for all
three agile-representative projects named in the proposal (Table 3.7,
Section 3.4.1.2): Commons Lang, Commons Math, and JFreeChart.

Requires `defects4j` on PATH (see tcp/data/defects4j.py's module docstring
and README.md's toolchain section for setup). Caches each sprint's
TestSuite to data_cache/defects4j_<project>/sprint_NN/ via tcp.data.files,
so later experiment runs can `load_sprint_sequence(...)` instantly instead
of re-checking-out bugs every time.
"""

import sys

import numpy as np

from tcp.data.defects4j import build_sprint_sequence, list_bugs
from tcp.data.files import save_test_suite

# Keep this modest -- each bug is a full project checkout + compile, and
# JFreeChart/Commons Math checkouts are noticeably heavier than Lang's.
N_BUGS_PER_PROJECT = 5

PROJECTS = {
    "Lang": "defects4j_lang",
    "Math": "defects4j_math",
    "Chart": "defects4j_chart",
}


def build_project(project: str, cache_name: str, n_bugs: int) -> None:
    valid_ids = list_bugs(project)
    bug_ids = valid_ids[:n_bugs]
    print(f"\n{project}: {len(valid_ids)} active bugs total, using {bug_ids}")

    suites = build_sprint_sequence(
        project, bug_ids, base_workdir=f"/tmp/d4j_{project.lower()}", rng=np.random.default_rng(0)
    )
    for i, suite in enumerate(suites):
        print(
            f"  sprint {i}: {suite.name:<10} n_tests={suite.n_tests:<3} "
            f"severity={suite.severity[0]:.3f} change_freq={suite.change_freq[0]:.3f}"
        )
        save_test_suite(suite, f"data_cache/{cache_name}/sprint_{i:02d}")


if __name__ == "__main__":
    only = sys.argv[1:] or list(PROJECTS.keys())
    for project in only:
        if project not in PROJECTS:
            print(f"unknown project {project!r}, choose from {list(PROJECTS)}")
            sys.exit(1)
        build_project(project, PROJECTS[project], N_BUGS_PER_PROJECT)
    print("\nDone. Reload any of these instantly with tcp.data.files.load_sprint_sequence(...).")
