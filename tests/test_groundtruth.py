"""Verify the ground-truth labelling rules fire where intended.

Every number the benchmark reports is downstream of these labels, so a bug
here does not look like a bug -- it looks like a finding. The fixture below
is one delegation constructed so that each rule has exactly one access that
should trigger it, plus two that should trigger nothing.
"""

from __future__ import annotations

import pytest

from antbench.groundtruth import label_delegation, need_summary
from antbench.schema import (
    ContextAccess,
    ContextKind,
    Delegation,
    NeedLabel,
    ToolCall,
)


def make_delegation() -> Delegation:
    """One delegation exercising all five labels.

    Built fresh per test: label_delegation mutates in place, so sharing an
    instance across tests would leak labels between them.
    """
    return Delegation(
        delegation_id="d1",
        parent_task_id="t1",
        worker_id="w1",
        message="Fix token refresh so expired sessions renew instead of 401ing.",
        issued_at_turn=2,
        repo_head="abc123",
        files_in_diff=["src/auth/session.py"],
        diff=(
            "--- a/src/auth/session.py\n"
            "+++ b/src/auth/session.py\n"
            "@@\n"
            "-    return None\n"
            "+    return refresh_token(self.token)\n"
        ),
        accesses=[
            ContextAccess(
                access_index=0, turn=1, kind=ContextKind.FILE_READ,
                query="src/auth/session.py",
                resolved_paths=["src/auth/session.py"], result_tokens=800,
            ),
            ContextAccess(
                access_index=1, turn=2, kind=ContextKind.SYMBOL_LOOKUP,
                query="refresh_token",
                resolved_paths=["src/auth/tokens.py"], result_tokens=200,
            ),
            # read at turn 3, path edited at turn 7 -> gap of 4
            ContextAccess(
                access_index=2, turn=3, kind=ContextKind.FILE_READ,
                query="src/auth/middleware.py",
                resolved_paths=["src/auth/middleware.py"], result_tokens=400,
            ),
            ContextAccess(
                access_index=3, turn=4, kind=ContextKind.TEST_READ,
                query="tests/test_session.py",
                resolved_paths=["tests/test_session.py"], result_tokens=300,
            ),
            ContextAccess(
                access_index=4, turn=5, kind=ContextKind.GREP,
                query="deprecated_login",
                resolved_paths=["src/legacy/login.py"], result_tokens=1500,
            ),
            ContextAccess(
                access_index=5, turn=6, kind=ContextKind.SYMBOL_LOOKUP,
                query="LegacyAuthAdapter",
                resolved_paths=["src/legacy/adapter.py"], result_tokens=250,
            ),
        ],
        tool_calls=[
            ToolCall(turn=7, name="apply_edit",
                     arguments={"path": "src/auth/middleware.py"}),
            ToolCall(turn=8, name="apply_edit",
                     arguments={"path": "src/auth/session.py"}),
            ToolCall(turn=9, name="run_tests",
                     arguments={"path": "tests/test_session.py"}),
        ],
    )


EXPECTED = [
    NeedLabel.USED_IN_DIFF,
    NeedLabel.RESOLVED_SYMBOL,
    NeedLabel.READ_BEFORE_EDIT,
    NeedLabel.TEST_CONTEXT,
    NeedLabel.UNUSED,
    NeedLabel.UNUSED,
]


@pytest.mark.parametrize("index,expected", list(enumerate(EXPECTED)))
def test_each_rule_fires(index: int, expected: NeedLabel) -> None:
    labelled = label_delegation(make_delegation())
    assert labelled.accesses[index].label is expected


def test_needed_paths_excludes_unused() -> None:
    labelled = label_delegation(make_delegation())
    assert labelled.needed_paths == {
        "src/auth/session.py",
        "src/auth/tokens.py",
        "src/auth/middleware.py",
        "tests/test_session.py",
    }


def test_summary_counts_waste() -> None:
    summary = need_summary(label_delegation(make_delegation()))
    assert summary["accesses"] == 6
    assert summary["needed"] == 4
    assert summary["unused"] == 2
    assert summary["waste_ratio"] == pytest.approx(1 / 3, abs=0.001)
    # only the two UNUSED accesses contribute wasted tokens
    assert summary["wasted_result_tokens"] == 1750


def test_tight_window_demotes_only_read_before_edit() -> None:
    """The window must govern READ_BEFORE_EDIT and nothing else.

    This is what makes the sensitivity sweep interpretable: if narrowing the
    window moved other labels too, a change in totals could not be attributed
    to the window.
    """
    baseline = [a.label for a in label_delegation(make_delegation()).accesses]
    tight = [
        a.label
        for a in label_delegation(
            make_delegation(), read_before_edit_window=2
        ).accesses
    ]

    assert tight[2] is NeedLabel.UNUSED, "gap of 4 should exceed a window of 2"
    assert tight[:2] == baseline[:2]
    assert tight[3:] == baseline[3:]


def test_unbounded_window_keeps_distant_read() -> None:
    distant = make_delegation()
    distant.accesses[2].turn = 0  # gap of 7, beyond the default window of 6
    assert label_delegation(distant).accesses[2].label is NeedLabel.UNUSED

    distant = make_delegation()
    distant.accesses[2].turn = 0
    unbounded = label_delegation(distant, read_before_edit_window=None)
    assert unbounded.accesses[2].label is NeedLabel.READ_BEFORE_EDIT


def test_diff_prefixed_paths_reconcile() -> None:
    """Paths arriving as './src/x.py' must match diff paths as 'src/x.py'.

    Normalisation failures are the highest-risk bug in this module: they make
    everything score UNUSED and inflate the apparent waste ratio.
    """
    d = make_delegation()
    d.files_in_diff = ["./src/auth/session.py"]
    d.accesses[0].resolved_paths = ["src/auth/session.py"]
    assert label_delegation(d).accesses[0].label is NeedLabel.USED_IN_DIFF


def test_broad_access_contributes_only_justifying_paths() -> None:
    """A grep matching many files is evidence about the one that earned its
    label, not about all fifteen it touched.

    Regression from an xarray delegation: a two-file fix reported 28 needed
    paths, including versioneer.py, because a grep for "unicode" matched 15
    files and one of them was in the diff. Inflating need is the dangerous
    direction -- it overstates how much a predictor could have prepared.
    """
    d = Delegation(
        delegation_id="d", parent_task_id="t", worker_id="w", message="m",
        issued_at_turn=0, repo_head="HEAD",
        files_in_diff=["src/core/variable.py"],
        diff="+++ b/src/core/variable.py\n+    return copy(self)\n",
        accesses=[
            ContextAccess(
                access_index=0, turn=1, kind=ContextKind.GREP, query="unicode",
                resolved_paths=[
                    "src/core/variable.py",   # in the diff -- earns the label
                    "versioneer.py",          # collateral match
                    "src/backends/zarr.py",   # collateral match
                ],
                result_tokens=900,
            ),
            ContextAccess(
                access_index=1, turn=2, kind=ContextKind.TEST_READ,
                query="tests", resolved_paths=["tests/test_variable.py"],
                result_tokens=200,
            ),
        ],
        tool_calls=[
            ToolCall(turn=3, name="apply_edit",
                     arguments={"path": "src/core/variable.py"}),
            ToolCall(turn=4, name="run_tests",
                     arguments={"path": "tests/test_variable.py"}),
        ],
    )
    labelled = label_delegation(d)
    assert labelled.accesses[0].label is NeedLabel.USED_IN_DIFF
    assert labelled.needed_paths == {
        "src/core/variable.py",
        "tests/test_variable.py",
    }


def test_test_context_contributes_only_test_paths() -> None:
    """TEST_CONTEXT must not drag non-test collateral into the ground truth."""
    d = Delegation(
        delegation_id="d", parent_task_id="t", worker_id="w", message="m",
        issued_at_turn=0, repo_head="HEAD",
        files_in_diff=[],
        accesses=[
            ContextAccess(
                access_index=0, turn=1, kind=ContextKind.TEST_READ,
                query="copy",
                resolved_paths=["tests/test_copy.py", "src/core/helpers.py"],
                result_tokens=300,
            ),
        ],
        tool_calls=[
            ToolCall(turn=2, name="run_tests",
                     arguments={"path": "tests/test_copy.py"}),
        ],
    )
    labelled = label_delegation(d)
    assert labelled.accesses[0].label is NeedLabel.TEST_CONTEXT
    assert labelled.needed_paths == {"tests/test_copy.py"}
