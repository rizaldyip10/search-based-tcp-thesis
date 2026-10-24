"""Extraction pipeline for the Siemens Test Suite (Software-artifact
Infrastructure Repository, SIR; Section 3.4.1.1).

HOW TO GET THE DATA: the Siemens programs are distributed through the SIR
portal (https://sir.csc.ncsu.edu), which requires creating a free account
and agreeing to the repository's usage terms -- that step has to be done by
you (a human), not this tool, since it involves account creation. Once you
have an account, download the five subjects used in the proposal (TCAS,
SCHEDULE, SCHEDULE2, PRINTTOKENS, TOTINFO) and note where they land on
disk.

WHY THIS MODULE DOESN'T HARD-CODE SIR'S FOLDER LAYOUT: SIR has changed its
packaging over the years and mirrors vary (e.g. `versions.alt/v1/...` vs
`source.alt/v1/...`, a `testplans.alt/` directory of inputs vs individual
`t*.in` files). Rather than guess a layout I can't verify without the actual
archive in hand, this module takes an explicit, already-resolved mapping:

    golden_source: path to the original (unseeded) program source
    faulty_sources: {fault_name: path to a seeded-fault version's source}
    test_inputs: {test_name: path to a file piped to the program's stdin}
    compile_cmd: how to compile one source file into a binary (defaults to
        a single `.c` file via gcc with gcov instrumentation)

`build_suite_from_c_subject` runs the actual experiment described in
Section 3.3.3.3 step 1 ("prepare dataset inputs"):
    1. Compile the golden version with coverage instrumentation.
    2. Run every test input against the golden binary, recording its
       output (the oracle) and per-test gcov line coverage.
    3. For each faulty version, compile it and run every test input;
       a mismatch against the golden output means that test *detects*
       that fault.
    4. Assemble the TestSuite (fault_matrix, coverage_matrix, cost).

Once you know your downloaded copy's actual layout, write a small adapter
that resolves it into the three dicts above (see
`discover_siemens_subject` for a best-effort example you will likely need
to adjust) and call `build_suite_from_c_subject`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from tcp.suite import TestSuite


class ToolchainUnavailableError(RuntimeError):
    """Raised when a required build tool (gcc/gcov) isn't usable."""


# On some macOS setups, `/usr/bin/gcc` and `/usr/bin/gcov` are shims that
# resolve the real tool via `xcodebuild -find`, which breaks if Xcode.app's
# own frameworks are in a bad state (a dylib-loading error, independent of
# whether the Command Line Tools package itself is installed). The
# standalone Command Line Tools package ships its own clang/gcov directly
# and works even when that resolution path is broken, as long as we supply
# an explicit sysroot (its own bundled SDK) since it isn't the active
# `xcode-select` toolchain. We try the plain `gcc`/`gcov` commands first and
# only fall back to this if those don't actually work.
_CLT_CLANG = Path("/Library/Developer/CommandLineTools/usr/bin/clang")
_CLT_GCOV = Path("/Library/Developer/CommandLineTools/usr/bin/gcov")
_CLT_SDK_CANDIDATES = [
    Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"),
]


def _clt_sdk() -> Path | None:
    for candidate in _CLT_SDK_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _probe_compiler(cc_cmd: list[str]) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.c"
        src.write_text("int main(void) { return 0; }\n")
        binary = Path(tmp) / "probe"
        try:
            compiled = subprocess.run(
                [*cc_cmd, str(src), "-o", str(binary)], capture_output=True, timeout=30
            )
            if compiled.returncode != 0:
                return False
            ran = subprocess.run([str(binary)], capture_output=True, timeout=10)
            return ran.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False


def _resolve_compiler() -> list[str]:
    """Return the argv prefix for a C compiler invocation that actually
    works on this machine, e.g. ["gcc"] or
    ["/Library/.../clang", "-isysroot", "/Library/.../MacOSX.sdk"].
    Raises ToolchainUnavailableError if neither works.
    """
    if shutil.which("gcc") and _probe_compiler(["gcc"]):
        return ["gcc"]
    sdk = _clt_sdk()
    if sdk and _CLT_CLANG.exists():
        fallback = [str(_CLT_CLANG), "-isysroot", str(sdk)]
        if _probe_compiler(fallback):
            return fallback
    raise ToolchainUnavailableError(
        "No working C compiler found. `gcc` failed, and the standalone "
        "Command Line Tools clang fallback also failed. On macOS, run "
        "`xcode-select --install` (needs your password -- this tool "
        "cannot do that for you) and retry."
    )


def _resolve_gcov() -> list[str]:
    if shutil.which("gcov"):
        try:
            if subprocess.run(
                ["gcov", "--version"], capture_output=True, timeout=10
            ).returncode == 0:
                return ["gcov"]
        except (OSError, subprocess.SubprocessError):
            pass
    if _CLT_GCOV.exists():
        try:
            if subprocess.run(
                [str(_CLT_GCOV), "--version"], capture_output=True, timeout=10
            ).returncode == 0:
                return [str(_CLT_GCOV)]
        except (OSError, subprocess.SubprocessError):
            pass
    raise ToolchainUnavailableError("No working `gcov` found (tried PATH and Command Line Tools).")


def gcc_available() -> bool:
    """Best-effort check that a working C compiler is available (either
    plain `gcc` or the Command Line Tools clang fallback -- see above)."""
    try:
        _resolve_compiler()
        return True
    except ToolchainUnavailableError:
        return False


def _require_toolchain() -> None:
    if not gcc_available():
        raise ToolchainUnavailableError(
            "No working C compiler could be found on this machine. On "
            "macOS this usually means the Xcode Command Line Tools are "
            "missing or broken (run `xcode-select --install`, which needs "
            "your password -- this tool cannot do that for you)."
        )


def _compile_with_coverage(source: Path, out_dir: Path, binary_name: str) -> tuple[Path, Path]:
    """Compile with coverage instrumentation. Returns (binary, gcno_path).
    `gcno_path` is discovered rather than assumed, because Apple's clang
    names it `<binary>-<source_stem>.gcno` while GNU gcc names it
    `<source_stem>.gcno`.
    """
    compiler = _resolve_compiler()
    binary = out_dir / binary_name
    before = {p.name for p in out_dir.glob("*.gcno")}
    result = subprocess.run(
        [*compiler, "-fprofile-arcs", "-ftest-coverage", "-O0", str(source), "-o", str(binary)],
        cwd=out_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"compilation of {source} failed:\n{result.stderr}")
    after = {p.name for p in out_dir.glob("*.gcno")}
    new_files = after - before
    stem = source.stem
    candidates = [n for n in new_files if stem in n] or list(new_files)
    if not candidates:
        raise RuntimeError(
            f"compilation of {source} produced no .gcno coverage-notes file "
            f"(looked in {out_dir})"
        )
    return binary, out_dir / sorted(candidates)[0]


def _compile_plain(source: Path, out_dir: Path, binary_name: str) -> Path:
    compiler = _resolve_compiler()
    binary = out_dir / binary_name
    result = subprocess.run(
        [*compiler, "-O0", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"compilation of {source} failed:\n{result.stderr}")
    return binary


def _run_with_input(binary: Path, stdin_path: Path, timeout: float = 10.0) -> tuple[str, float]:
    start = time.perf_counter()
    result = subprocess.run(
        [str(binary)],
        stdin=stdin_path.open("rb"),
        capture_output=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    return result.stdout.decode(errors="replace") + "\n--exit:" + str(result.returncode), elapsed


def _extract_line_coverage(
    gcda_dir: Path, gcno_path: Path, source_name: str, n_lines_hint: int | None = None
) -> np.ndarray:
    """Run gcov on the accumulated .gcda counters (via `gcno_path`, whose
    stem may not match `source_name` -- see `_compile_with_coverage`) and
    return a boolean vector over source lines (True = executed at least once).
    """
    gcov_cmd = _resolve_gcov()
    result = subprocess.run(
        [*gcov_cmd, "-b", str(gcno_path)],
        cwd=gcda_dir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    gcov_file = gcda_dir / f"{source_name}.gcov"
    if result.returncode != 0 or not gcov_file.exists():
        raise RuntimeError(f"gcov failed for {source_name}:\n{result.stderr}")

    executed = []
    for line in gcov_file.read_text(errors="replace").splitlines():
        # gcov line format: "    <count-or-'-'-or-'#####'>:  <lineno>: <src>"
        prefix = line.split(":", 1)[0].strip()
        if prefix in ("-", ""):
            continue  # non-executable line
        if prefix == "#####":
            executed.append(False)
        else:
            try:
                executed.append(int(prefix) > 0)
            except ValueError:
                continue
    if n_lines_hint is not None and len(executed) != n_lines_hint:
        # Padding/truncating defensively; callers should treat n_lines_hint
        # as authoritative since it comes from the golden run.
        if len(executed) < n_lines_hint:
            executed.extend([False] * (n_lines_hint - len(executed)))
        else:
            executed = executed[:n_lines_hint]
    return np.array(executed, dtype=bool)


def build_suite_from_c_subject(
    golden_source: str | Path,
    faulty_sources: dict[str, str | Path],
    test_inputs: dict[str, str | Path],
    budget_fraction: float = 0.5,
    name: str = "siemens_subject",
) -> TestSuite:
    """Build a TestSuite for one Siemens-style C subject by actually
    compiling, running, and diffing -- see module docstring.

    `faulty_sources` keys become fault names (one fault per seeded-fault
    version, matching how SIR structures each subject); `test_inputs` keys
    become test names. Every value is a filesystem path.
    """
    _require_toolchain()

    test_names = sorted(test_inputs)
    fault_names = sorted(faulty_sources)
    n_tests = len(test_names)
    n_faults = len(fault_names)
    if n_tests == 0 or n_faults == 0:
        raise ValueError("need at least one test input and one faulty version")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        golden_dir = tmp_path / "golden"
        golden_dir.mkdir()
        golden_src_copy = golden_dir / Path(golden_source).name
        shutil.copy(golden_source, golden_src_copy)
        golden_binary, golden_gcno = _compile_with_coverage(golden_src_copy, golden_dir, "golden_bin")

        golden_outputs: dict[str, str] = {}
        cost = np.zeros(n_tests)
        gcda_file = golden_gcno.with_suffix(".gcda")
        # gcov names its report "<source-filename-with-extension>.gcov",
        # e.g. "golden.c.gcov" -- NOT "<stem>.gcov".
        source_filename = golden_src_copy.name
        n_source_lines = len(golden_src_copy.read_text(errors="replace").splitlines())
        coverage_matrix = np.zeros((n_tests, n_source_lines), dtype=bool)

        for i, tname in enumerate(test_names):
            # reset per-test counters by removing any stale .gcda before each run
            if gcda_file.exists():
                gcda_file.unlink()
            output, elapsed = _run_with_input(golden_binary, Path(test_inputs[tname]))
            golden_outputs[tname] = output
            cost[i] = elapsed
            coverage_matrix[i] = _extract_line_coverage(
                golden_dir, golden_gcno, source_filename, n_source_lines
            )

        fault_matrix = np.zeros((n_faults, n_tests), dtype=bool)
        for f_idx, fname in enumerate(fault_names):
            faulty_dir = tmp_path / f"fault_{f_idx}"
            faulty_dir.mkdir()
            faulty_src_copy = faulty_dir / Path(faulty_sources[fname]).name
            shutil.copy(faulty_sources[fname], faulty_src_copy)
            faulty_binary = _compile_plain(faulty_src_copy, faulty_dir, "faulty_bin")

            for t_idx, tname in enumerate(test_names):
                output, _ = _run_with_input(faulty_binary, Path(test_inputs[tname]))
                if output != golden_outputs[tname]:
                    fault_matrix[f_idx, t_idx] = True

        undetected = [fault_names[i] for i in range(n_faults) if not fault_matrix[i].any()]
        if undetected:
            raise ValueError(
                f"fault(s) {undetected} are not detected by ANY supplied test input -- "
                "the test pool is inadequate for this seeded fault, or the fault "
                "is actually equivalent to the golden version"
            )

        severity = np.full(n_faults, 0.5)  # simulated; see module note below
        change_freq = np.full(n_faults, 0.5)  # simulated; see module note below
        total_cost = float(cost.sum())
        budget = total_cost * budget_fraction if total_cost > 0 else None

        return TestSuite(
            fault_matrix=fault_matrix,
            coverage_matrix=coverage_matrix,
            cost=cost,
            severity=severity,
            change_freq=change_freq,
            budget=budget,
            name=name,
        )


def simulate_fault_severity(
    fault_names: list[str],
    category_map: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate per-fault severity from fault-type categories, as described
    on p.62 ("Siemens fault severity simulation based on fault-type
    categories rather than real stakeholder assessment").

    `category_map` maps a substring that may appear in a fault's name (e.g.
    the seeded-fault version's commit message or diff type) to a severity
    value in [0, 1]; faults whose name matches no key fall back to a random
    draw. Customise `category_map` to reflect categories you define (e.g.
    off-by-one vs. logic-inversion vs. missing-case) once you've inspected
    the actual seeded diffs.
    """
    rng = rng if rng is not None else np.random.default_rng()
    category_map = category_map or {}
    severities = np.empty(len(fault_names))
    for i, fname in enumerate(fault_names):
        matched = next((v for k, v in category_map.items() if k in fname), None)
        severities[i] = matched if matched is not None else rng.uniform(0.3, 0.9)
    return severities
