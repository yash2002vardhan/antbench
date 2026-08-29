"""Predictor contract and scorer arithmetic.

The contract test matters most: a predictor that could see the trace would
score arbitrarily well and the whole benchmark would be meaningless. That
isolation is structural -- predictors take a message and an index, never a
Delegation -- and this pins it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from antbench.predictor import (
    LexicalPredictor,
    Predictor,
    RandomPredictor,
    RepoIndex,
)
from antbench.schema import (
    ContextAccess,
    ContextKind,
    Delegation,
    NeedLabel,
    ToolCall,
)
from antbench.scorer import (
    aggregate,
    is_scorable,
    needed_token_cost,
    score_delegation,
)


@pytest.fixture
def index(tmp_path: Path) -> RepoIndex:
    (tmp_path / "src" / "auth").mkdir(parents=True)
    (tmp_path / "src" / "legacy").mkdir(parents=True)
    (tmp_path / "src" / "auth" / "session.py").write_text(
        "class Session:\n    def refresh(self):\n        return None\n"
    )
    (tmp_path / "src" / "auth" / "tokens.py").write_text(
        "def refresh_token(tok):\n    return tok\n"
    )
    for i in range(8):
        (tmp_path / "src" / "legacy" / f"mod{i}.py").write_text(
            "def helper():\n    return 1\n"
        )
    return RepoIndex.build(tmp_path)


def test_predictors_cannot_see_the_trace() -> None:
    """Structural isolation: predict() takes a message, not a Delegation."""
    params = inspect.signature(Predictor.predict).parameters
    assert set(params) == {"self", "message", "index", "budget"}

    source = inspect.getsource(LexicalPredictor)
    for leak in ("accesses", "tool_calls", "files_in_diff", ".diff"):
        assert leak not in source, f"predictor references trace field {leak}"


def test_index_records_paths_and_definitions(index: RepoIndex) -> None:
    assert "src/auth/session.py" in index.paths
    assert "refresh_token" in index.definitions["src/auth/tokens.py"]


def test_lexical_ranks_the_named_file_first(index: RepoIndex) -> None:
    prediction = LexicalPredictor().predict(
        "The session refresh path returns None; fix it.", index, budget=3
    )
    assert prediction.paths[0] == "src/auth/session.py"


def test_lexical_downweights_ubiquitous_terms(index: RepoIndex) -> None:
    """A term in most files ranks below a term in one.

    There is deliberately no hard cutoff: an earlier version zeroed any term
    appearing in more than a third of files, which discarded `config` on a
    pylint delegation about config-file discovery -- the one word that located
    the answer -- because the repo names many files after it. Log-IDF
    down-weights smoothly instead.
    """
    common = LexicalPredictor().predict("fix the helper", index, budget=5)
    specific = LexicalPredictor().predict("fix refresh_token", index, budget=5)

    # `helper` is defined in 8 of 10 files, so it cannot single anything out.
    assert len(common.paths) > 1
    # `refresh_token` is defined in exactly one, which should win outright.
    assert specific.paths[0] == "src/auth/tokens.py"


def test_lexical_ignores_prose_stopwords(index: RepoIndex) -> None:
    """Common English words must not survive as if they were identifiers.

    They are rare as definition names, so IDF alone rates them informative --
    a pylint delegation once scored on "add", "data", "file", "get", "run".
    """
    prediction = LexicalPredictor().predict(
        "please update the file so that we can get the data", index, budget=5
    )
    assert prediction.paths == []


def test_lexical_handles_message_with_no_identifiers(index: RepoIndex) -> None:
    assert LexicalPredictor().predict("fix it", index, budget=5).paths == []


def test_random_respects_budget_and_is_seeded(index: RepoIndex) -> None:
    first = RandomPredictor(seed=1).predict("x", index, budget=4)
    second = RandomPredictor(seed=1).predict("x", index, budget=4)
    assert len(first.paths) == 4
    assert first.paths == second.paths


def _delegation_needing(paths: list[str], tokens: int = 100) -> Delegation:
    """A completed delegation whose needed_paths are exactly `paths`."""
    return Delegation(
        delegation_id="d1", parent_task_id="t1", worker_id="w1",
        message="m", issued_at_turn=0, repo_head="HEAD",
        completed=True,
        files_in_diff=paths,
        diff="".join(f"+++ b/{p}\n" for p in paths),
        accesses=[
            ContextAccess(
                access_index=i, turn=i + 1, kind=ContextKind.FILE_READ,
                query=p, resolved_paths=[p], result_tokens=tokens,
                label=NeedLabel.USED_IN_DIFF,
            )
            for i, p in enumerate(paths)
        ],
        tool_calls=[
            ToolCall(turn=9, name="apply_edit", arguments={"path": p})
            for p in paths
        ],
    )


class _FixedPredictor(Predictor):
    name = "fixed"

    def __init__(self, paths: list[str]) -> None:
        self._paths = paths

    def predict(self, message, index, budget):  # noqa: ANN001
        from antbench.predictor import Prediction

        return Prediction(paths=self._paths[:budget])


def test_score_counts_hits_at_each_budget(index: RepoIndex) -> None:
    delegation = _delegation_needing(
        ["src/auth/session.py", "src/auth/tokens.py"]
    )
    # Correct answer ranked second: nothing at k=1, one hit from k=3 on.
    predictor = _FixedPredictor(
        ["src/legacy/mod0.py", "src/auth/session.py", "src/legacy/mod1.py"]
    )

    score = score_delegation(delegation, predictor, index, repo="r")
    assert score.needed == 2
    assert score.hits_at[1] == 0
    assert score.hits_at[3] == 1
    assert score.recall_at(3) == pytest.approx(0.5)
    assert score.precision_at(3) == pytest.approx(1 / 3)


def test_repeated_access_to_a_path_counts_once(index: RepoIndex) -> None:
    """A prefetch layer serves a file once, so repeat reads must not
    multiply the token saving attributed to predicting it."""
    delegation = _delegation_needing(["src/auth/session.py"], tokens=100)
    delegation.accesses.append(
        ContextAccess(
            access_index=9, turn=9, kind=ContextKind.FILE_READ,
            query="src/auth/session.py",
            resolved_paths=["src/auth/session.py"], result_tokens=250,
            label=NeedLabel.USED_IN_DIFF,
        )
    )
    # Largest single read, not the sum of both.
    assert needed_token_cost(delegation) == {"src/auth/session.py": 250}


def test_aggregate_macro_averages_recall(index: RepoIndex) -> None:
    """Each delegation counts once regardless of how many files it needed,
    so a sprawling delegation cannot dominate the headline."""
    small = _delegation_needing(["src/auth/session.py"])
    large = _delegation_needing(
        [f"src/legacy/mod{i}.py" for i in range(6)]
    )
    predictor = _FixedPredictor(["src/auth/session.py"])

    scores = [
        score_delegation(small, predictor, index, repo="r"),
        score_delegation(large, predictor, index, repo="r"),
    ]
    summary = aggregate(scores, "fixed")
    # 1.0 on the small one, 0.0 on the large one -> 0.5, not 1/7.
    assert summary["recall@8"] == pytest.approx(0.5)
    assert summary["any_hit@8"] == pytest.approx(0.5)


def test_aggregate_handles_empty_input() -> None:
    assert aggregate([], "none")["delegations"] == 0


def test_blocked_delegation_with_real_need_is_scorable(index: RepoIndex) -> None:
    """A worker that explored well and then judged the task BLOCKED produced
    real evidence about what the delegation required.

    Excluding it would discard the cases where a worker knew what it was
    doing. What disqualifies a delegation is having nothing to predict, not
    the conclusion the worker reached.
    """
    blocked = _delegation_needing(["src/auth/session.py"])
    blocked.completed = False
    blocked.failure_reason = "blocked"
    assert is_scorable(blocked)

    vacuous = _delegation_needing([])
    vacuous.completed = False
    assert not is_scorable(vacuous)


def test_aggregate_reports_completion_mix(index: RepoIndex) -> None:
    done = _delegation_needing(["src/auth/session.py"])
    blocked = _delegation_needing(["src/auth/tokens.py"])
    blocked.completed = False

    predictor = _FixedPredictor(["src/auth/session.py"])
    scores = [
        score_delegation(d, predictor, index, repo="r") for d in (done, blocked)
    ]
    summary = aggregate(scores, "fixed")
    assert summary["delegations"] == 2
    assert summary["completed_in_scored"] == 1
