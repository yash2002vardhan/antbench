"""Workspace behaviour, and its handoff to the ground-truth labeller.

The integration test at the bottom is the important one. `workspace.py`
produces paths and `groundtruth.py` consumes them; if their conventions drift
apart, every access scores UNUSED and the result looks like a dramatic finding
about agent waste rather than a bug. Fixtures cannot catch that -- it needs a
real git repo and a real diff.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from antbench.groundtruth import label_delegation, need_summary
from antbench.schema import ContextKind, Delegation, NeedLabel
from antbench.workspace import Workspace


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal git repo with a source file, a helper, and a test."""
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "auth" / "session.py").write_text(
        "class Session:\n    def refresh(self):\n        return None\n"
    )
    (tmp_path / "src" / "auth" / "tokens.py").write_text(
        'def refresh_token(tok):\n    return tok + "-new"\n'
    )
    (tmp_path / "tests" / "test_session.py").write_text(
        "def test_refresh():\n    assert True\n"
    )

    run = lambda *a: subprocess.run(  # noqa: E731
        a, cwd=tmp_path, capture_output=True, check=True
    )
    run("git", "init", "-q", ".")
    run("git", "add", "-A")
    run("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return tmp_path


@pytest.fixture
def ws(repo: Path) -> Workspace:
    return Workspace(root=repo)


def test_read_logs_path_and_tokens(ws: Workspace) -> None:
    ws.turn = 1
    ws.read_file("src/auth/session.py")
    access = ws.accesses[0]
    assert access.kind is ContextKind.FILE_READ
    assert access.resolved_paths == ["src/auth/session.py"]
    assert access.result_tokens > 0
    assert access.hit


def test_large_file_pages_and_advertises_continuation(ws: Workspace) -> None:
    """A truncated read must tell the worker how to see the rest.

    Regression from the first pilot task: without paging, a worker re-read one
    large module eleven times and never edited anything. The harness, not the
    agent, was the failure.
    """
    big = ws.root / "src" / "big.py"
    big.write_text("\n".join(f"line_{i} = {i}" for i in range(1, 1001)))

    first = ws.read_file("src/big.py")
    assert "lines 1-400 of 1000" in first
    assert "start_line=401" in first
    assert "line_400 = 400" in first
    assert "line_401" not in first

    second = ws.read_file("src/big.py", start_line=401)
    assert "lines 401-800 of 1000" in second
    assert "line_401 = 401" in second

    last = ws.read_file("src/big.py", start_line=801)
    assert "lines 801-1000 of 1000" in last
    assert "start_line=" not in last  # nothing further to advertise

    beyond = ws.read_file("src/big.py", start_line=5000)
    assert ws.accesses[-1].hit is False
    assert "nothing at line" in beyond

    big.unlink()


def test_test_files_classify_as_test_read(ws: Workspace) -> None:
    ws.read_file("tests/test_session.py")
    assert ws.accesses[0].kind is ContextKind.TEST_READ


def test_misses_are_logged_not_swallowed(ws: Workspace) -> None:
    """A worker groping for something absent is evidence about the
    delegation, so misses must survive into the trace."""
    ws.read_file("does/not/exist.py")
    ws.grep("no_such_symbol_anywhere")
    assert [a.hit for a in ws.accesses] == [False, False]
    assert len(ws.accesses) == 2


def test_path_escape_is_refused(ws: Workspace) -> None:
    result = ws.read_file("../../../etc/passwd")
    assert result.startswith("ERROR")
    assert ws.accesses[0].hit is False


def test_symbol_lookup_finds_definition(ws: Workspace) -> None:
    ws.find_symbol("refresh_token")
    assert ws.accesses[0].resolved_paths == ["src/auth/tokens.py"]


def test_edit_requires_unambiguous_match(ws: Workspace) -> None:
    assert ws.apply_edit("src/auth/session.py", "nonexistent", "x").startswith("ERROR")
    assert ws.tool_calls[0].ok is False


def test_diff_and_reset(ws: Workspace) -> None:
    ws.apply_edit("src/auth/session.py", "return None", "return 1")
    diff, paths = ws.diff()
    assert paths == ["src/auth/session.py"]
    assert "return 1" in diff

    ws.reset_repo()
    assert ws.diff()[0] == ""


def test_reset_log_clears_attribution(ws: Workspace) -> None:
    """Records must not leak across delegations."""
    ws.read_file("src/auth/session.py")
    ws.apply_edit("src/auth/session.py", "return None", "return 1")
    ws.reset_log()
    assert ws.accesses == [] and ws.tool_calls == [] and ws.edited_paths == set()
    ws.reset_repo()


def test_workspace_output_labels_correctly(ws: Workspace) -> None:
    """End-to-end: a real trace on a real repo, labelled by groundtruth.

    Guards the path-format contract between the two modules.
    """
    ws.turn = 1
    ws.read_file("src/auth/session.py")       # lands in the diff
    ws.turn = 2
    ws.find_symbol("refresh_token")           # symbol used in an added line
    ws.turn = 3
    ws.read_file("tests/test_session.py")     # test later run
    ws.turn = 4
    ws.grep("legacy_login")                   # pays off in nothing
    ws.turn = 5
    ws.apply_edit(
        "src/auth/session.py", "return None", "return refresh_token(self.tok)"
    )
    ws.turn = 6
    ws.run_tests("tests/test_session.py")

    diff, paths = ws.diff()
    delegation = label_delegation(
        Delegation(
            delegation_id="d1", parent_task_id="t1", worker_id="w1",
            message="fix refresh", issued_at_turn=0, repo_head="HEAD",
            accesses=list(ws.accesses), tool_calls=list(ws.tool_calls),
            files_in_diff=paths, diff=diff,
        )
    )

    assert [a.label for a in delegation.accesses] == [
        NeedLabel.USED_IN_DIFF,
        NeedLabel.RESOLVED_SYMBOL,
        NeedLabel.TEST_CONTEXT,
        NeedLabel.UNUSED,
    ]
    assert delegation.needed_paths == {
        "src/auth/session.py",
        "src/auth/tokens.py",
        "tests/test_session.py",
    }
    assert need_summary(delegation)["waste_ratio"] == pytest.approx(0.25)
    ws.reset_repo()


def test_plain_read_that_surfaces_a_used_symbol_counts(ws: Workspace) -> None:
    """A worker may learn a signature by reading a file, not just by calling
    find_symbol. Both are genuine need.

    Regression from the first live trace: a worker read tokens.py, found
    refresh_token, called it in the fix, and the access scored UNUSED. That
    undercounts need and makes prediction look harder than it is.
    """
    ws.turn = 1
    ws.read_file("src/auth/tokens.py")  # defines refresh_token, never edited
    ws.turn = 2
    ws.apply_edit(
        "src/auth/session.py", "return None", "return refresh_token(self.tok)"
    )

    diff, paths = ws.diff()
    kwargs = dict(
        delegation_id="d", parent_task_id="t", worker_id="w", message="m",
        issued_at_turn=0, repo_head="HEAD", accesses=list(ws.accesses),
        tool_calls=list(ws.tool_calls), files_in_diff=paths, diff=diff,
    )

    with_repo = label_delegation(Delegation(**kwargs), repo_root=ws.root)
    assert with_repo.accesses[0].label is NeedLabel.RESOLVED_SYMBOL

    # Archived traces have no checkout; the rule must degrade, not crash.
    without_repo = label_delegation(Delegation(**kwargs))
    assert without_repo.accesses[0].label is NeedLabel.UNUSED

    ws.reset_repo()


def test_broad_grep_does_not_inherit_symbol_credit(ws: Workspace) -> None:
    """A grep spanning many files must not be credited for symbol payoff.

    Observed in the second pilot run: a grep matching 20 files scored
    RESOLVED_SYMBOL because one of them defined a name in the diff, pushing
    mean needed_paths to 46 against a two-file fix. This direction of error is
    the dangerous one -- it overstates how much a predictor could prepare.
    """
    ws.turn = 1
    ws.grep("refresh")  # matches session.py and tokens.py
    ws.turn = 2
    ws.apply_edit(
        "src/auth/session.py", "return None", "return refresh_token(self.tok)"
    )

    diff, paths = ws.diff()
    labelled = label_delegation(
        Delegation(
            delegation_id="d", parent_task_id="t", worker_id="w", message="m",
            issued_at_turn=0, repo_head="HEAD", accesses=list(ws.accesses),
            tool_calls=list(ws.tool_calls), files_in_diff=paths, diff=diff,
        ),
        repo_root=ws.root,
    )

    grep_access = labelled.accesses[0]
    assert len(grep_access.resolved_paths) > 1
    # session.py is in the diff, so USED_IN_DIFF is correct here -- what must
    # not happen is tokens.py riding along via symbol credit.
    assert grep_access.label is NeedLabel.USED_IN_DIFF

    ws.reset_repo()
