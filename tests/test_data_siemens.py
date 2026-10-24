"""Validates the Siemens-style compile/run/diff/gcov pipeline end-to-end
using two tiny hand-written C programs (NOT the real Siemens Test Suite,
which requires a gated SIR download -- see tcp/data/siemens.py). Skipped
automatically if a working gcc/gcov toolchain isn't available.
"""

import pytest

from tcp.data.siemens import build_suite_from_c_subject, gcc_available

pytestmark = pytest.mark.skipif(
    not gcc_available(), reason="gcc could not compile/run a trivial program on this machine"
)

GOLDEN_SRC = """
#include <stdio.h>
int classify(int x) {
    if (x < 0) {
        return -1;
    } else if (x == 0) {
        return 0;
    } else {
        return 1;
    }
}
int main(void) {
    int x;
    scanf("%d", &x);
    printf("%d\\n", classify(x));
    return 0;
}
"""

# Seeded fault: the x == 0 branch is wrong (off-by-one), so any test with x == 0
# will produce a different output than the golden version.
FAULTY_SRC = """
#include <stdio.h>
int classify(int x) {
    if (x < 0) {
        return -1;
    } else if (x == 1) {
        return 0;
    } else {
        return 1;
    }
}
int main(void) {
    int x;
    scanf("%d", &x);
    printf("%d\\n", classify(x));
    return 0;
}
"""


@pytest.fixture
def subject_paths(tmp_path):
    golden = tmp_path / "golden.c"
    golden.write_text(GOLDEN_SRC)
    faulty = tmp_path / "faulty.c"
    faulty.write_text(FAULTY_SRC)

    inputs = {}
    for name, value in [("neg", "-5"), ("zero", "0"), ("pos", "7")]:
        p = tmp_path / f"{name}.txt"
        p.write_text(value)
        inputs[name] = p

    return golden, faulty, inputs


def test_build_suite_detects_seeded_fault_with_expected_test(subject_paths):
    golden, faulty, inputs = subject_paths
    suite = build_suite_from_c_subject(
        golden_source=golden,
        faulty_sources={"off_by_one": faulty},
        test_inputs=inputs,
        name="classify_demo",
    )

    assert suite.n_tests == 3
    assert suite.n_faults == 1

    test_names = sorted(inputs)  # loader sorts test names alphabetically
    zero_idx = test_names.index("zero")
    neg_idx = test_names.index("neg")
    pos_idx = test_names.index("pos")

    # only the "zero" input triggers the seeded fault (x == 0 branch changed)
    assert suite.fault_matrix[0, zero_idx] == True  # noqa: E712
    assert suite.fault_matrix[0, neg_idx] == False  # noqa: E712
    assert suite.fault_matrix[0, pos_idx] == False  # noqa: E712

    assert suite.coverage_matrix.shape[0] == 3
    assert suite.cost.shape == (3,)
    assert (suite.cost >= 0).all()


def test_build_suite_raises_when_fault_is_undetectable(subject_paths):
    golden, faulty, inputs = subject_paths
    # only give it inputs that never hit the x == 0 branch difference
    narrow_inputs = {k: v for k, v in inputs.items() if k != "zero"}
    with pytest.raises(ValueError, match="not detected"):
        build_suite_from_c_subject(
            golden_source=golden,
            faulty_sources={"off_by_one": faulty},
            test_inputs=narrow_inputs,
        )
