# Agile-aware hybrid GA-CSO for Test Case Prioritization

Code artifact for the thesis *"A Multi-Objective Search-Based Method for
Test Case Prioritization to Improve Fault Detection Rate in Agile
Environments"*. Implements RO2 (the method), the Section 3.4 dataset
loaders, and RO3's experiment runner + statistics pipeline (Section 3.3.3).

## Setup

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

59 tests pass with both external toolchains (Siemens's gcc/gcov, and
Defects4J) set up as described below; fewer will pass and some will skip
if your machine doesn't have them yet.

## Layout

```
tcp/
  suite.py       TestSuite data model
  metrics.py     APFD, APFDc, weighted APFD, Coverage Effectiveness
  fitness.py     composite multi-objective fitness F (Eq. 3.1)
  ga.py          Genetic Algorithm (permutation encoding)
  cso.py         Cuckoo Search Optimization (Lévy flight)
  hybrid.py      agile-aware hybrid GA-CSO + sprint re-initialisation
  baselines.py   the 6 methods in Table 3.6
  method.py      convenience wrappers for the proposed method
  synthetic.py   synthetic data generator, used for all current tests
  experiment.py  Phase 3 trial runner: 30 runs x 6 methods, APFD/APFDc/CE
  stats.py       Shapiro-Wilk -> ANOVA/Kruskal-Wallis -> Tukey HSD (Sec 3.3.3.4)
  report.py      plain-text summary/comparison tables
  data/
    files.py       generic CSV-based TestSuite load/save (dataset-agnostic)
    siemens.py     SIR Siemens Test Suite extraction pipeline
    defects4j.py   Defects4J extraction pipeline
tests/            pytest suite (59 passing with both toolchains set up)
demo.py           runs all 6 methods once on one synthetic suite
experiment_demo.py runs the full Phase 3 pipeline (30 runs + stats) on a synthetic suite
defects4j_demo.py runs the full pipeline on REAL Commons Lang bugs (needs defects4j on PATH)
build_real_datasets.py builds + caches sprint sequences for Lang, Math, AND Chart (real bugs)
data_cache/       cached real sprint sequences for all 3 projects (5 bugs each, see below)
```

## This machine's toolchain, as actually set up

Both external toolchains are installed and working on this machine as of
2026-10-19. If you're on a fresh machine, or this breaks after a macOS/Xcode
update, here's exactly what was needed and why (all of this was discovered
by hitting real failures, not assumed upfront):

1. **Xcode Command Line Tools were pointed at a broken Xcode.app**
   (`xcodebuild`'s own frameworks were corrupted, breaking every tool that
   shells out to it for path resolution: `gcc`, `gcov`, `make`, ...). Fixed
   with `sudo xcode-select --switch /Library/Developer/CommandLineTools`
   (needs your password; `tcp/data/siemens.py` also has a PATH-based
   fallback for gcc/gcov specifically, in case this regresses).
2. **The linker's default SDK was a bleeding-edge beta** (`MacOSX27.0.sdk`)
   whose `.tbd` library-stub format this machine's `ld`/tapi parser can't
   read ("unknown architecture"). Native compiles (Perl XS modules, the
   Siemens C pipeline) need `SDKROOT` pointed at a stable SDK instead, e.g.:
   ```bash
   export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk
   ```
3. **Defects4J requires *exactly* Java 11** (a hard check in its own
   `Constants.pm`, not "11 or newer"). Installed via
   `brew install openjdk@11`; `tcp/data/defects4j.py` auto-detects this and
   injects the right `PATH` into its own subprocess calls, so you don't
   need Java 11 to be your default `java`.
4. **`cpanm --installdeps .` (inside the cloned defects4j repo) falls back
   to a user-local Perl lib** (`~/perl5/lib/perl5`) without sudo access to
   the system Perl's site directory, and nothing adds that to `@INC`
   automatically. `tcp/data/defects4j.py` sets `PERL5LIB` for its own
   subprocess calls to work around this; if you run `defects4j` directly
   from a shell yourself, `export PERL5LIB=~/perl5/lib/perl5` first.
5. **`wget`, `ant`, and `svn` aren't preinstalled on macOS.** `defects4j/init.sh`
   needs wget+ant; `svn` is needed separately for checking out JFreeChart
   bugs specifically (its Defects4J repo is SVN-sourced, unlike the
   git-native Lang/Math). `brew install wget ant subversion` (plus
   `brew install cpanminus` for step 4).

With all of that done, Defects4J itself was cloned and initialised with:
```bash
git clone https://github.com/rjust/defects4j.git
cd defects4j
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk  # see point 2
cpanm --installdeps .
./init.sh   # downloads ~730MB total (project repos + Major + test-gen tools); took ~2.5 hours on this connection
export PATH="$PWD/framework/bin:$PATH"   # needed by Python callers too
```

## Getting the real datasets

### Siemens Test Suite (Section 3.4.1.1)

Distributed via the SIR portal (<https://sir.csc.ncsu.edu>), which requires
you to create a free account and accept its terms -- that step has to be
done by you. Once you have the five subjects (TCAS, SCHEDULE, SCHEDULE2,
PRINTTOKENS, TOTINFO) on disk, `tcp/data/siemens.py::build_suite_from_c_subject`
compiles/runs/diffs them for you, **but it does not assume a specific SIR
folder layout** (SIR's packaging has changed over the years and mirrors
vary) -- you resolve your actual download into three dicts
(`golden_source`, `faulty_sources`, `test_inputs`) and pass them in. See
`tests/test_data_siemens.py` for a complete worked example using two tiny
hand-written C programs in place of the real subjects.

Fault severity is simulated per the proposal (p.62); customise
`simulate_fault_severity`'s `category_map` once you've categorised the
seeded faults.

### Defects4J (Section 3.4.1.2) -- working, verified against real bugs from all 3 projects

Set up and verified against real bugs from all three agile-representative
projects named in Table 3.7: Commons Lang, Commons Math, and JFreeChart
(see the toolchain section above for how, including the `svn` requirement
specific to JFreeChart). `tcp/data/defects4j.py::build_suite_for_bug` checks
out one bug, reads its relevant/trigger tests, derives severity from
fix-diff size and change-frequency from git commit history of the touched
files. `build_sprint_sequence` builds a whole sprint sequence from a list
of bug ids, validating them against `list_bugs(project)` first (some bug
ids in a project's range are deprecated). Run `python build_real_datasets.py`
to build + cache 5-bug sprint sequences for all three projects (or
`python build_real_datasets.py Math` for just one), or
`python defects4j_demo.py` for a complete worked example against Commons
Lang specifically, including the Phase 3 statistics pipeline.

Two real bugs were caught and fixed by testing against actual data instead
of assuming the design was right (both have regression tests now):
- Change-frequency was first normalised by the max commit count among a
  bug's *own* touched files, which degenerates to exactly 1.0 for the
  common single-file-bug case (`change_frequency_from_commit_count` now
  uses an absolute saturating scale instead).
- Change-frequency was computed from the per-bug checkout's own git log,
  which is a real (if time-truncated) history slice for git-native
  projects (Lang, Math) but only ~4 synthetic tag commits for SVN-sourced
  JFreeChart -- every Chart bug came out with the same near-constant
  value. Fixed by querying each project's full-history mirror
  (`project_repos/<name>.git`) instead (see `project_mirror_dir` and its
  docstring for the full explanation and the point-in-time trade-off this
  implies).

**Read the module docstring's caveats before trusting results for the
actual thesis experiments**:
- Per-test coverage is currently a coarse class-name-matching heuristic,
  not real JaCoCo instrumentation; replace `coverage_matrix_heuristic`.
- `tests.relevant` is class-granularity, not method-granularity -- each
  "test case" in the resulting TestSuite is really a whole JUnit test
  class, which also means suites can be small for some bugs (we saw
  anywhere from 1 to 152 tests across the 15 real bugs built so far) and
  the per-method `tests.trigger` info is collapsed to "does this class
  contain a triggering method".
- Change-frequency counts ALL commits to a file ever (via the project
  mirror), not just those before the bug being studied -- a general
  churn/volatility signal, not a strict point-in-time one.

### Caching extracted suites

Both pipelines are expensive (compiling per fault version; checking out a
full Defects4J project per bug). Use `tcp/data/files.py::save_test_suite`
to cache the result to disk once, and `load_test_suite` /
`load_sprint_sequence` to reload instantly on subsequent runs.

## Try it

```bash
python demo.py
```
