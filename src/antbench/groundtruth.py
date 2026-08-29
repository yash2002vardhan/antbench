"""Derive, after the fact, which context a worker genuinely needed.

This is the part that makes software engineering the right domain for the
benchmark. "What did the agent need?" is a judgement call in most settings; in
a repo it is largely recoverable from evidence the work leaves behind:

  - a file in the final diff was needed, full stop;
  - a file read shortly before editing that same file was needed;
  - a definition the worker looked up and then called was needed;
  - a test the worker read and then ran was needed.

Everything else is provisionally UNUSED. That label is deliberately harsh: it
counts exploration the worker did but did not profit from, which is exactly
the waste a predictive layer should remove. Calling that waste "need" would
let a predictor take credit for prefetching noise.

Known limitation, stated plainly because it bounds the benchmark's claims:
this measures what the worker *used*, not what it *would have used* had
retrieval been free. A worker that never looked something up because looking
it up was expensive leaves no trace here. That biases measured need downward,
so reported prediction ceilings are conservative.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from .schema import ContextAccess, ContextKind, Delegation, NeedLabel

# How many turns may separate a read from the edit it supposedly enabled.
#
# MEASURED RESULT: this parameter turns out not to matter. Sweeping it from 0
# to unbounded over the pilot corpus moved mean need and mean waste by exactly
# zero, because READ_BEFORE_EDIT never fires: it can only apply to a file that
# was edited yet is absent from the final diff, and every successfully edited
# file appears in the diff, where the stronger USED_IN_DIFF claims it first.
# Across 44 successful edits in the corpus, the label was assigned 0 times.
#
# The rule is kept because it costs nothing and would fire on a trace where an
# edit is later reverted or overwritten -- but the ground truth does not rest
# on it, and no reported number depends on this constant. That is a better
# outcome than a defensible-but-arbitrary choice: rerun
# scripts/sweep_window.py after any change to the labelling rules to confirm
# the property still holds.
DEFAULT_READ_BEFORE_EDIT_WINDOW = 6

_TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]+\.py$|_test\.py$")


def _is_test_path(path: str) -> bool:
    return bool(_TEST_PATH.search(path))


def _normalize(path: str) -> str:
    """Repo-relative POSIX form, so diff paths and tool paths compare equal."""
    cleaned = path.strip().lstrip("./")
    return str(PurePosixPath(cleaned)) if cleaned else ""


def _edit_turns_by_path(delegation: Delegation) -> dict[str, list[int]]:
    """Turns at which each path was edited, from the worker's edit calls."""
    edits: dict[str, list[int]] = {}
    for call in delegation.tool_calls:
        if call.name not in {"apply_edit", "write_file"} or not call.ok:
            continue
        path = _normalize(str(call.arguments.get("path", "")))
        if path:
            edits.setdefault(path, []).append(call.turn)
    return edits


def _ran_test_paths(delegation: Delegation) -> set[str]:
    """Test files the worker actually executed, not merely read."""
    ran: set[str] = set()
    for call in delegation.tool_calls:
        if call.name != "run_tests":
            continue
        target = _normalize(str(call.arguments.get("path", "")))
        if target:
            ran.add(target)
    return ran


def _called_symbols(delegation: Delegation) -> set[str]:
    """Symbols appearing in added diff lines.

    Used to decide whether a symbol lookup paid off. Matching on added lines
    only avoids crediting a lookup for a name that was already in the file.
    """
    symbols: set[str] = set()
    for line in delegation.diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        symbols.update(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", line[1:]))
    return symbols


def _label_access(
    access: ContextAccess,
    diff_paths: set[str],
    edit_turns: dict[str, list[int]],
    ran_tests: set[str],
    called_symbols: set[str],
    provided_symbols: set[str],
    read_before_edit_window: int | None,
) -> NeedLabel:
    """Assign the strongest label an access qualifies for.

    Order matters: USED_IN_DIFF is checked first because it is the only label
    grounded in the finished artifact rather than in worker behaviour, and a
    file can legitimately satisfy several conditions at once.

    `read_before_edit_window` of None means unbounded -- any read preceding an
    edit of that path counts. Used as the permissive end of the sweep.
    """
    paths = {_normalize(p) for p in access.resolved_paths if p.strip()}

    if paths & diff_paths:
        return NeedLabel.USED_IN_DIFF

    # Credit an access that surfaced a definition the fix went on to use.
    #
    # Two ways a worker learns a signature: calling find_symbol, or simply
    # reading the file that defines it. Crediting only the former undercounts
    # need -- observed in the first live trace, where a worker read tokens.py,
    # found refresh_token, and called it, yet scored UNUSED. Symbol payoff is
    # about what the access revealed, not which verb revealed it.
    if access.kind is ContextKind.SYMBOL_LOOKUP:
        if access.query.strip() in called_symbols:
            return NeedLabel.RESOLVED_SYMBOL
    elif provided_symbols & called_symbols:
        return NeedLabel.RESOLVED_SYMBOL

    for path in paths:
        for edit_turn in edit_turns.get(path, ()):
            gap = edit_turn - access.turn
            if gap < 0:
                continue
            if read_before_edit_window is None or gap <= read_before_edit_window:
                return NeedLabel.READ_BEFORE_EDIT

    if paths & ran_tests or (
        access.kind is ContextKind.TEST_READ and any(_is_test_path(p) for p in paths)
    ):
        return NeedLabel.TEST_CONTEXT

    return NeedLabel.UNUSED


_DEFINITION = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)


def _definitions_in(path: str, repo_root: Path | None) -> set[str]:
    """Top-level names a file defines, read from the repo at label time.

    Returns empty when the repo is unavailable, which degrades the symbol rule
    to lookup-only rather than failing -- labelling must work on archived
    traces whose checkout is long gone.
    """
    if repo_root is None:
        return set()
    target = repo_root / path
    try:
        if not target.is_file():
            return set()
        return set(_DEFINITION.findall(target.read_text(encoding="utf-8",
                                                        errors="replace")))
    except OSError:
        return set()


def label_delegation(
    delegation: Delegation,
    read_before_edit_window: int | None = DEFAULT_READ_BEFORE_EDIT_WINDOW,
    repo_root: Path | None = None,
) -> Delegation:
    """Label every access in place and return the delegation.

    Mutates rather than copies: traces are large and this runs over all of
    them, and the caller always wants the labelled version. Re-running with a
    different window overwrites prior labels, which is what the sweep relies on.

    `repo_root` enables the symbol-payoff rule for plain file reads by letting
    us see which definitions a file provides. Without it that rule falls back
    to explicit find_symbol calls only, which undercounts need.
    """
    diff_paths = {_normalize(p) for p in delegation.files_in_diff if p.strip()}
    edit_turns = _edit_turns_by_path(delegation)
    ran_tests = _ran_test_paths(delegation)
    symbols = _called_symbols(delegation)

    for access in delegation.accesses:
        provided: set[str] = set()
        # Symbol payoff requires an access that actually *shows* a definition:
        # a whole-file read, not a search hit.
        #
        # Two conditions, both learned from real traces. The access must name
        # exactly one file -- a grep matching twenty files taught the worker no
        # signature, and crediting all twenty because one defines a name in the
        # diff inflates need. And it must be a read: a one-hit grep for
        # `spec_from_loader` landed on a documentation example and was credited
        # as though the worker had learned the symbol there, when a grep returns
        # matching lines rather than a definition.
        if (
            access.kind in (ContextKind.FILE_READ, ContextKind.TEST_READ)
            and len(access.resolved_paths) == 1
        ):
            provided = _definitions_in(_normalize(access.resolved_paths[0]), repo_root)
        access.label = _label_access(
            access, diff_paths, edit_turns, ran_tests, symbols, provided,
            read_before_edit_window,
        )
    return delegation


def need_summary(delegation: Delegation) -> dict[str, int | float]:
    """Per-delegation stats. `waste_ratio` is the headline diagnostic.

    A high waste ratio means the worker spent retrievals on context that did
    not pay off -- the exact cost a predictive layer claims to remove, and so
    an upper bound on how much this delegation has to offer.
    """
    total = len(delegation.accesses)
    unused = sum(1 for a in delegation.accesses if a.label is NeedLabel.UNUSED)
    needed = total - unused
    wasted_tokens = sum(
        a.result_tokens for a in delegation.accesses if a.label is NeedLabel.UNUSED
    )
    return {
        "accesses": total,
        "needed": needed,
        "unused": unused,
        "needed_paths": len(delegation.needed_paths),
        "waste_ratio": round(unused / total, 3) if total else 0.0,
        "wasted_result_tokens": wasted_tokens,
    }
