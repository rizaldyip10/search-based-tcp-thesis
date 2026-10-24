"""Extraction pipeline for the Defects4J agile-representative dataset
(Section 3.4.1.2): bug-fix commits from Commons Lang, Commons Math, and
JFreeChart, mapped to simulated Scrum sprints.

HOW TO GET THE TOOLCHAIN: Defects4J is a public, no-registration-required
GitHub project (https://github.com/rjust/defects4j), but setting it up is
heavy: clone the repo, run `cpanm --installdeps .` (Perl dependencies),
then `./init.sh` (which itself clones every subject project's full git
history -- several GB, can take a long time). None of that is done by this
module automatically; call `defects4j_available()` to check whether the CLI
is already on PATH before using anything else here.

WHAT THIS MODULE DOES NOT DO (IMPORTANT CAVEATS):

1. Per-test coverage: Defects4J's built-in `defects4j coverage` command
   reports coverage for a *whole* test run, not per individual test case,
   and the proposal's own tool table (Table 3.4) calls for JaCoCo, which
   isn't what D4J's coverage command uses internally. Getting true
   per-test line/branch coverage means instrumenting the build with a
   JaCoCo javaagent and running each relevant test in isolation -- real,
   but substantial extra engineering specific to each project's build
   (Ant vs Maven, classpath layout, etc.). `coverage_matrix_heuristic`
   below is a clearly-labelled STAND-IN: it approximates per-test coverage
   by string-matching each test's fully-qualified class name against the
   bug's modified classes. Replace it with real JaCoCo instrumentation
   before running the actual thesis experiments.
2. Exact `defects4j` CLI flags can differ between D4J 1.x and 2.x. The
   wrapper functions below document the command they issue; re-check
   `defects4j <subcommand> --help` against your installed version if a
   call fails.
3. Test-case granularity: Defects4J's `tests.relevant` export is
   class-granularity (e.g. "org.foo.BarTest" covers every @Test method in
   that class), while `tests.trigger` is method-granularity
   ("org.foo.BarTest::testBaz"). There is no class-only trigger export, so
   this module treats each relevant *class* as one "test case" in the TCP
   sense (matching how these suites are actually executed -- JUnit runs a
   whole class per invocation) and a class "detects" the bug if any of its
   methods is a trigger test. If your thesis wants true method-level TCP,
   you'd need to enumerate every method in each relevant class (via
   `tests.all` cross-referenced with `classes.relevant`) instead.
4. Change-frequency source: computed from `project_repos/<name>.git`
   mirrors (see `_PROJECT_MIRROR_DIR_NAMES`), NOT the per-bug checkout --
   the checkout's own git history is a few synthetic tag commits only for
   SVN-sourced projects like Chart/JFreeChart. This means change-frequency
   counts ALL commits to a file ever, not just those before the bug being
   studied -- a "how churn-prone is this code in general" signal, not a
   strict point-in-time one.

`build_suite_for_bug` assembles one TestSuite per bug (= one simulated
sprint, per Section 3.4.1.2's "bug-fix commits ... mapped to sprint
cycles"): the active test suite is the bug's *relevant* tests, the single
fault is "this bug", detected by its *trigger* tests, severity is derived
from the size of the fix diff, and change-frequency is derived from git
commit history of the files the fix touches.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from tcp.suite import TestSuite


class ToolchainUnavailableError(RuntimeError):
    pass


# Defects4J's own framework/core/Constants.pm hard-codes a check that
# `java -version`'s major version is EXACTLY 11 (not "11 or newer") and
# dies with a one-line, easy-to-miss message otherwise -- if the ambient
# PATH's `java` is some other version (common on modern machines, which
# default to much newer JDKs), every `defects4j` subprocess call below
# would fail. Rather than require the caller's shell to have this set up
# correctly, we detect it and inject a corrected PATH for just these
# subprocess calls, probing common install locations for a real Java 11.
_JAVA11_BIN_CANDIDATES = [
    Path("/opt/homebrew/opt/openjdk@11/bin"),
    Path("/usr/local/opt/openjdk@11/bin"),
]

# `cpanm --installdeps .` falls back to a user-local lib (e.g. ~/perl5) with
# no sudo/write access to the system Perl's site directory; that local lib
# only gets added to @INC with local::lib active or PERL5LIB set, neither of
# which `defects4j`'s own subprocess invocations do for you automatically.
_PERL5LIB_CANDIDATES = [Path.home() / "perl5" / "lib" / "perl5"]

# Per-bug `defects4j checkout` working directories do NOT all carry real
# commit history: for git-native projects (Lang, Math, ...) the checkout's
# git log is a real (if time-truncated, up to that bug's point) slice of
# history, but for SVN-sourced projects (Chart/JFreeChart) it's just the
# handful of synthetic "D4J_..._VERSION" tag commits D4J adds at checkout
# time -- confirmed empirically: Chart_1b had 4 total commits vs. Lang_1b's
# 3574, and every Chart bug's change-frequency came out as the same
# near-constant value as a result. Defects4J's own project_repos/<name>.git
# mirrors have the REAL, full commit history for every supported project
# (verified: commons-lang.git has 7061 commits, jfreechart.git has 4218),
# so git_file_change_frequency below queries these mirrors instead of the
# ephemeral per-bug checkout. Trade-off: this counts ALL commits to a file
# (including ones after the bug being studied), i.e. "how churn-prone is
# this code in general" rather than "how much had it already churned by
# this sprint" -- a deliberate, documented choice; a strict point-in-time
# definition would need per-bug commit-date cutoffs, which this module
# does not implement.
_PROJECT_MIRROR_DIR_NAMES = {
    "Lang": "commons-lang.git",
    "Math": "commons-math.git",
    "Chart": "jfreechart.git",
    "Cli": "commons-cli.git",
    "Codec": "commons-codec.git",
    "Collections": "commons-collections.git",
    "Compress": "commons-compress.git",
    "Csv": "commons-csv.git",
    "Gson": "gson.git",
    "JacksonCore": "jackson-core.git",
    "JacksonDatabind": "jackson-databind.git",
    "JacksonXml": "jackson-dataformat-xml.git",
    "Jsoup": "jsoup.git",
    "JxPath": "commons-jxpath.git",
    "Time": "joda-time.git",
    "Mockito": "mockito.git",
    "Closure": "closure-compiler.git",
}


def _defects4j_root() -> Path | None:
    """Resolve the Defects4J installation root from wherever `defects4j`
    is found on PATH (its real path is <root>/framework/bin/defects4j)."""
    binary = shutil.which("defects4j")
    if binary is None:
        return None
    return Path(binary).resolve().parent.parent.parent


def project_mirror_dir(project: str) -> Path | None:
    """Path to this project's full-history repo mirror
    (project_repos/<name>.git), or None if it can't be resolved (unknown
    Defects4J install location, or `project` not in the known name map --
    add an entry to `_PROJECT_MIRROR_DIR_NAMES` for a new project)."""
    root = _defects4j_root()
    if root is None:
        return None
    mirror_name = _PROJECT_MIRROR_DIR_NAMES.get(project)
    if mirror_name is None:
        return None
    candidate = root / "project_repos" / mirror_name
    return candidate if candidate.exists() else None


def _java_major_version(java_bin: str = "java") -> int | None:
    try:
        result = subprocess.run([java_bin, "-version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r'version "?(?:1\.)?(\d+)', result.stdout + result.stderr)
    return int(match.group(1)) if match else None


def _resolve_java11_bin_dir() -> Path | None:
    """Return a bin directory to prepend to PATH so `java` resolves to
    version 11, or None if the ambient PATH's `java` is already version 11
    (no override needed)."""
    if _java_major_version() == 11:
        return None
    for candidate in _JAVA11_BIN_CANDIDATES:
        if (candidate / "java").exists() and _java_major_version(str(candidate / "java")) == 11:
            return candidate
    return None


def _subprocess_env() -> dict[str, str]:
    """Environment for `defects4j`/git subprocess calls, with a Java 11
    bin directory prepended to PATH if the ambient `java` isn't already 11,
    and a local-lib Perl directory added to PERL5LIB if it exists and isn't
    already on the ambient @INC-affecting path."""
    env = os.environ.copy()
    java11_dir = _resolve_java11_bin_dir()
    if java11_dir is not None:
        env["PATH"] = f"{java11_dir}{os.pathsep}{env.get('PATH', '')}"
    for candidate in _PERL5LIB_CANDIDATES:
        if candidate.exists():
            existing = env.get("PERL5LIB", "")
            if str(candidate) not in existing.split(os.pathsep):
                env["PERL5LIB"] = (
                    f"{candidate}{os.pathsep}{existing}" if existing else str(candidate)
                )
    return env


def defects4j_available() -> bool:
    if shutil.which("defects4j") is None:
        return False
    try:
        result = subprocess.run(
            ["defects4j", "pids"], capture_output=True, text=True, timeout=30, env=_subprocess_env()
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _require_toolchain() -> None:
    if shutil.which("defects4j") is None:
        raise ToolchainUnavailableError(
            "`defects4j` is not on PATH. Clone "
            "https://github.com/rjust/defects4j, install its Perl "
            "dependencies, run ./init.sh, and add framework/bin to PATH."
        )
    java_version = _java_major_version()
    if java_version != 11 and _resolve_java11_bin_dir() is None:
        raise ToolchainUnavailableError(
            f"defects4j requires exactly Java 11 (found major version "
            f"{java_version}), and no Java 11 install was found in the "
            f"usual Homebrew locations ({_JAVA11_BIN_CANDIDATES}). Install "
            f"one, e.g. `brew install openjdk@11`."
        )
    if not defects4j_available():
        raise ToolchainUnavailableError(
            "`defects4j` is on PATH but did not run successfully (e.g. "
            "project repos not initialised -- run ./init.sh)."
        )


def list_bugs(project: str) -> list[int]:
    """`defects4j bids -p <PID>` -- active bug ids for a project (e.g. 'Lang')."""
    _require_toolchain()
    result = subprocess.run(
        ["defects4j", "bids", "-p", project],
        capture_output=True,
        text=True,
        timeout=60,
        env=_subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"defects4j bids failed:\n{result.stderr}")
    return [int(line) for line in result.stdout.split() if line.strip().lstrip("-").isdigit()]


def checkout_bug(project: str, bug_id: int, workdir: str | Path, buggy: bool = True) -> Path:
    """`defects4j checkout -p <PID> -v <id><b|f> -w <workdir>`."""
    _require_toolchain()
    workdir = Path(workdir)
    version = f"{bug_id}{'b' if buggy else 'f'}"
    result = subprocess.run(
        ["defects4j", "checkout", "-p", project, "-v", version, "-w", str(workdir)],
        capture_output=True,
        text=True,
        timeout=300,
        env=_subprocess_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"defects4j checkout failed:\n{result.stderr}")
    return workdir


def _export_property(workdir: Path, prop: str) -> list[str]:
    """`defects4j export -p <prop> -o <tmpfile>`, returning the file's
    non-empty lines. Used for tests.relevant, tests.trigger,
    classes.modified, dir.src.classes, etc.
    """
    with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            ["defects4j", "export", "-p", prop, "-o", str(tmp_path)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=120,
            env=_subprocess_env(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"defects4j export -p {prop} failed:\n{result.stderr}")
        return [line.strip() for line in tmp_path.read_text().splitlines() if line.strip()]
    finally:
        tmp_path.unlink(missing_ok=True)


def relevant_tests(workdir: Path) -> list[str]:
    return _export_property(workdir, "tests.relevant")


def trigger_tests(workdir: Path) -> list[str]:
    return _export_property(workdir, "tests.trigger")


def modified_classes(workdir: Path) -> list[str]:
    return _export_property(workdir, "classes.modified")


def src_classes_dir(workdir: Path) -> str:
    """`defects4j export -p dir.src.classes` -- the project's source root
    (e.g. "src/main/java"), needed because a modified class's fully
    qualified name (e.g. "org.foo.Bar") does NOT map to "org/foo/Bar.java"
    at the repo root; it's rooted under this directory instead."""
    values = _export_property(workdir, "dir.src.classes")
    return values[0] if values else ""


def run_single_test(workdir: Path, test_id: str, timeout: float = 120.0) -> tuple[bool, float]:
    """Run exactly one test via `defects4j test -t <test_id> -w <workdir>`
    and use wall-clock time as a cost proxy. Returns (passed, elapsed_seconds).
    A non-empty `failing_tests` file after the run means the test failed.
    """
    failing_tests_path = Path(workdir) / "failing_tests"
    failing_tests_path.unlink(missing_ok=True)
    start = time.perf_counter()
    result = subprocess.run(
        ["defects4j", "test", "-t", test_id, "-w", str(workdir)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_subprocess_env(),
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"defects4j test -t {test_id} failed to run:\n{result.stderr}")
    passed = not (failing_tests_path.exists() and failing_tests_path.stat().st_size > 0)
    return passed, elapsed


def git_file_change_frequency(
    repo_dir: str | Path, file_paths: list[str], max_commits: int | None = 500
) -> dict[str, int]:
    """Number of commits touching each path (Section 3.3.1.2's "code change
    frequency" operationalisation), via `git log --follow --oneline -- <path>`.
    """
    repo_dir = Path(repo_dir)
    counts: dict[str, int] = {}
    for path in file_paths:
        cmd = ["git", "log", "--follow", "--oneline", "--", path]
        if max_commits is not None:
            cmd[2:2] = [f"-{max_commits}"]
        result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=60)
        counts[path] = len(result.stdout.splitlines()) if result.returncode == 0 else 0
    return counts


def fault_severity_from_diff_size(n_modified_lines: int, scale: float = 50.0) -> float:
    """Derive a severity proxy in (0, 1) from fix-diff size: bigger fixes
    are assumed to indicate more impactful bugs. Saturating so very large
    diffs don't dominate (1 - exp(-lines/scale))."""
    return float(1.0 - np.exp(-max(0, n_modified_lines) / scale))


def change_frequency_from_commit_count(n_commits: float, scale: float = 20.0) -> float:
    """Derive a change-frequency proxy in (0, 1) from a raw commit count on
    an absolute, saturating scale (1 - exp(-commits/scale)) -- e.g. a file
    touched once ever is ~0.05, one touched 20+ times approaches 1.0.

    IMPORTANT: this is NOT the same as normalising by the maximum commit
    count *within this bug's own touched files*, which degenerates to
    exactly 1.0 whenever a bug touches only one file (the common case for
    these projects) and so loses almost all discriminative power across
    bugs -- that was this function's first (wrong) implementation."""
    return float(1.0 - np.exp(-max(0.0, n_commits) / scale))


def _count_diff_lines(workdir: Path) -> int:
    """Approximate fix size via `git diff --stat` against the fixed
    revision, if the checkout is a git repo with both revisions reachable;
    falls back to 0 (-> minimum severity) if unavailable.
    """
    result = subprocess.run(
        ["git", "diff", "--shortstat", "HEAD~1", "HEAD"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return 0
    match = re.search(r"(\d+) insertion", result.stdout)
    insertions = int(match.group(1)) if match else 0
    match = re.search(r"(\d+) deletion", result.stdout)
    deletions = int(match.group(1)) if match else 0
    return insertions + deletions


def coverage_matrix_heuristic(
    test_names: list[str], modified_class_names: list[str]
) -> np.ndarray:
    """COARSE STAND-IN for real per-test coverage (see module docstring,
    caveat 1): a test "covers" a modified class if the class's simple name
    appears in the test's fully-qualified name. One coverage "element" per
    modified class.
    """
    n_tests = len(test_names)
    n_classes = len(modified_class_names)
    matrix = np.zeros((n_tests, max(n_classes, 1)), dtype=bool)
    simple_names = [c.rsplit(".", 1)[-1] for c in modified_class_names]
    for i, test in enumerate(test_names):
        test_class = test.split("::")[0] if "::" in test else test
        for j, simple in enumerate(simple_names):
            if simple and simple in test_class:
                matrix[i, j] = True
    # tests that match nothing still exist in the suite; leave their row all-False
    return matrix


def build_suite_for_bug(
    project: str,
    bug_id: int,
    base_workdir: str | Path,
    run_tests_for_cost: bool = False,
    rng: np.random.Generator | None = None,
    name: str | None = None,
) -> TestSuite:
    """Assemble one simulated-sprint TestSuite for a single Defects4J bug.

    If `run_tests_for_cost` is False (default), a synthetic but labelled
    cost is used (uniform(0.1, 2.0) seconds per test) instead of actually
    executing every relevant test, which can be slow for large suites --
    set it to True once you're ready to pay that cost for real timing data.
    """
    _require_toolchain()
    rng = rng if rng is not None else np.random.default_rng()
    workdir = Path(base_workdir) / f"{project}_{bug_id}b"
    checkout_bug(project, bug_id, workdir, buggy=True)

    tests = relevant_tests(workdir)
    triggers = set(trigger_tests(workdir))
    classes = modified_classes(workdir)
    if not tests:
        raise ValueError(f"{project}-{bug_id}: no relevant tests reported")
    if not triggers:
        raise ValueError(f"{project}-{bug_id}: no trigger tests reported (fault undetectable)")

    # `tests.relevant` is class-granularity (e.g. "org.foo.BarTest") but
    # `tests.trigger` is method-granularity ("org.foo.BarTest::testBaz") --
    # confirmed empirically (Defects4J has no class-only trigger export), so
    # a relevant class "detects" the bug if ANY of its methods is a trigger.
    trigger_classes = {t.split("::")[0] for t in triggers}
    n_tests = len(tests)
    fault_matrix = np.zeros((1, n_tests), dtype=bool)
    for i, t in enumerate(tests):
        if t in trigger_classes:
            fault_matrix[0, i] = True

    if run_tests_for_cost:
        cost = np.zeros(n_tests)
        for i, t in enumerate(tests):
            try:
                _, elapsed = run_single_test(workdir, t)
            except RuntimeError:
                elapsed = 1.0  # fallback if a single test invocation errors
            cost[i] = elapsed
    else:
        cost = rng.uniform(0.1, 2.0, size=n_tests)

    n_diff_lines = _count_diff_lines(workdir)
    severity = np.array([fault_severity_from_diff_size(n_diff_lines)])

    src_dir = src_classes_dir(workdir)
    touched_files = [
        f"{src_dir}/{c.replace('.', '/')}.java" if src_dir else f"{c.replace('.', '/')}.java"
        for c in classes
    ]
    # Query the project's full-history mirror rather than the per-bug
    # checkout -- see _PROJECT_MIRROR_DIR_NAMES's comment for why (the
    # checkout's own history is unusable for SVN-sourced projects).
    history_source = project_mirror_dir(project) or workdir
    change_counts = git_file_change_frequency(history_source, touched_files)
    mean_count = float(np.mean(list(change_counts.values()))) if change_counts else 0.0
    change_freq = np.array([change_frequency_from_commit_count(mean_count)])

    coverage_matrix = coverage_matrix_heuristic(tests, classes)

    total_cost = float(cost.sum())
    budget = total_cost * 0.5 if total_cost > 0 else None

    return TestSuite(
        fault_matrix=fault_matrix,
        coverage_matrix=coverage_matrix,
        cost=cost,
        severity=severity,
        change_freq=change_freq,
        budget=budget,
        name=name or f"{project}-{bug_id}",
    )


def build_sprint_sequence(
    project: str,
    bug_ids: list[int],
    base_workdir: str | Path,
    run_tests_for_cost: bool = False,
    rng: np.random.Generator | None = None,
) -> list[TestSuite]:
    """Build a sequence of per-bug TestSuites, one per simulated sprint, in
    the given bug_id order (sort by bug_id ascending for a chronological
    approximation of sprint order).

    Validates `bug_ids` against `list_bugs(project)` upfront (some bug ids
    in a project's historical range are deprecated/withdrawn and would
    otherwise fail checkout with a buried Perl stack trace partway through
    the sequence).
    """
    rng = rng if rng is not None else np.random.default_rng()
    valid_ids = set(list_bugs(project))
    invalid = [bid for bid in bug_ids if bid not in valid_ids]
    if invalid:
        raise ValueError(
            f"{project} bug id(s) {invalid} are not active (deprecated, "
            f"withdrawn, or out of range) -- call list_bugs({project!r}) "
            f"for the current valid set."
        )
    return [
        build_suite_for_bug(project, bid, base_workdir, run_tests_for_cost, rng)
        for bid in bug_ids
    ]
