"""Score predictions against what workers actually needed.

Metrics, and why these:

  recall@k    -- of the files the worker needed, how many were offered. This
                 is the headline: it answers "how much of the need was
                 knowable at delegation time".
  precision@k -- of the files offered, how many were needed. Bounds the waste
                 a prefetch layer would add.
  tokens_saved -- result tokens of correctly predicted accesses. The cost
                 metric, and the one a buyer cares about; recall counts files,
                 but files are not equally expensive.

Reported at several budgets because the operating point is a product decision.
A layer that injects context needs high precision; one that warms a cache
behind the normal retrieval interface can tolerate low precision, since a
wrong guess there costs compute and never pollutes the context window.

Delegations that did not complete are excluded. Their diffs are empty or
near-empty, so the ground truth has almost nothing to score against and every
predictor would look bad for reasons that have nothing to do with prediction.
"""

from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .predictor import Predictor, RepoIndex
from .schema import Delegation, NeedLabel, Trace

BUDGETS = (1, 3, 5, 8)


@dataclass
class DelegationScore:
    """One prediction judged against one delegation's ground truth."""

    delegation_id: str
    repo: str
    needed: int
    predicted: int
    hits_at: dict[int, int] = field(default_factory=dict)
    tokens_saved_at: dict[int, int] = field(default_factory=dict)
    needed_tokens: int = 0
    # Carried so the aggregate can report what the scored population is made
    # of; blocked delegations are included when they produced real need.
    completed: bool = True

    def recall_at(self, k: int) -> float:
        return self.hits_at.get(k, 0) / self.needed if self.needed else 0.0

    def precision_at(self, k: int) -> float:
        offered = min(k, self.predicted)
        return self.hits_at.get(k, 0) / offered if offered else 0.0


def is_scorable(delegation: Delegation) -> bool:
    """Whether a delegation carries enough ground truth to score against.

    The criterion is evidence, not status. `completed` conflates two very
    different outcomes: a worker that ran out of turns mid-orientation (no
    diff, so every access scores UNUSED and the ground truth is vacuous) and
    a worker that explored efficiently, understood the task, and then judged
    it BLOCKED. The second produced real evidence about what the delegation
    required -- one such delegation reached 11 needed paths at 0.15 waste,
    the cleanest access pattern in the pilot -- and excluding it would discard
    exactly the cases where a worker knew what it was doing.

    So: score any delegation whose accesses produced labelled need, whatever
    the worker concluded. Delegations with no needed paths are excluded
    because there is nothing to predict, not because they failed.
    """
    return bool(delegation.needed_paths)


def needed_token_cost(delegation: Delegation) -> dict[str, int]:
    """Result tokens attributable to each needed path.

    A path fetched several times counts once, at its largest result -- a
    prefetch layer would serve it once, so crediting every repeat read would
    inflate the saving.
    """
    cost: dict[str, int] = {}
    for access in delegation.accesses:
        if access.label is None or access.label is NeedLabel.UNUSED:
            continue
        for path in access.resolved_paths:
            cost[path] = max(cost.get(path, 0), access.result_tokens)
    return cost


def score_delegation(
    delegation: Delegation,
    predictor: Predictor,
    index: RepoIndex,
    repo: str,
    budget: int = max(BUDGETS),
) -> DelegationScore:
    """Predict for one delegation and score against its needed paths."""
    needed = delegation.needed_paths
    token_cost = needed_token_cost(delegation)

    prediction = predictor.predict(delegation.message, index, budget)

    score = DelegationScore(
        delegation_id=delegation.delegation_id,
        repo=repo,
        needed=len(needed),
        predicted=len(prediction.paths),
        needed_tokens=sum(token_cost.values()),
        completed=delegation.completed,
    )
    for k in BUDGETS:
        offered = set(prediction.top(k))
        correct = offered & needed
        score.hits_at[k] = len(correct)
        score.tokens_saved_at[k] = sum(token_cost.get(p, 0) for p in correct)
    return score


def score_traces(
    traces: list[Trace],
    predictor: Predictor,
    repos_dir: Path,
    budget: int = max(BUDGETS),
) -> list[DelegationScore]:
    """Score every completed delegation across a corpus.

    Repo indexes are built once per repo, not per delegation -- a real
    predictive layer maintains its index continuously, so rebuilding it each
    time would misrepresent the cost of prediction.
    """
    indexes: dict[tuple[str, str], RepoIndex] = {}
    scores: list[DelegationScore] = []

    for trace in traces:
        slug = trace.repo.replace("/", "__")
        root = repos_dir / slug
        if not root.is_dir():
            continue

        # Index at the trace's own base commit, not at whatever the last task
        # left checked out. A predictor scored against a different commit is
        # being asked about files the worker never saw -- 22 of 30 traces were
        # affected before this was caught, because one checkout was reused
        # across every task in a repository.
        key = (slug, trace.base_commit)
        if key not in indexes:
            subprocess.run(
                ["git", "checkout", "--quiet", "--force", trace.base_commit],
                cwd=root, capture_output=True, timeout=300,
            )
            indexes[key] = RepoIndex.build(root)

        for delegation in trace.delegations:
            if not is_scorable(delegation):
                continue
            scores.append(
                score_delegation(
                    delegation, predictor, indexes[key], trace.repo, budget
                )
            )
    return scores


def aggregate(scores: list[DelegationScore], predictor_name: str) -> dict:
    """Corpus-level metrics.

    Recall is macro-averaged -- each delegation counts once regardless of how
    many files it needed, so a handful of sprawling delegations cannot
    dominate the headline.
    """
    if not scores:
        return {"predictor": predictor_name, "delegations": 0}

    result: dict = {
        "predictor": predictor_name,
        "delegations": len(scores),
        # Blocked-but-informative delegations are scored too, so state how
        # many of the scored population finished.
        "completed_in_scored": sum(1 for s in scores if s.completed),
        "mean_needed_paths": round(
            statistics.mean(s.needed for s in scores), 2
        ),
    }

    for k in BUDGETS:
        recalls = [s.recall_at(k) for s in scores]
        precisions = [s.precision_at(k) for s in scores]
        saved = sum(s.tokens_saved_at.get(k, 0) for s in scores)
        total = sum(s.needed_tokens for s in scores)
        result[f"recall@{k}"] = round(statistics.mean(recalls), 3)
        result[f"precision@{k}"] = round(statistics.mean(precisions), 3)
        result[f"token_saving@{k}"] = round(saved / total, 3) if total else 0.0
        # How often a predictor is useful at all, as distinct from how much it
        # gets right -- a layer that helps rarely but strongly is a different
        # product from one that helps always but weakly.
        result[f"any_hit@{k}"] = round(
            sum(1 for s in scores if s.hits_at.get(k, 0) > 0) / len(scores), 3
        )

    return result
