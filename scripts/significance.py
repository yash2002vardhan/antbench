"""Paired significance test between two predictors.

The headline comparison -- free string matching against a frontier model --
is a claim about a difference, and a difference over 26 delegations needs an
error bar rather than a point estimate. An earlier draft of the pitch asserted
lexical "beats" the LLM; the test below put that at p = 0.071, which supports
"at least as good, for free" and not the stronger sentence.

A paired permutation test is the right instrument here: the two predictors see
identical delegations, the per-delegation differences are not normally
distributed (many exact zeros, a few +/-1.0), and no distributional assumption
is needed. Under the null that the predictors are interchangeable, the sign of
each paired difference is arbitrary, so resampling sign flips gives the exact
reference distribution.

    uv run python scripts/significance.py            # lexical vs llm
    uv run python scripts/significance.py --a random --b lexical
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from antbench.predictor import (  # noqa: E402
    LexicalPredictor,
    LLMPredictor,
    RandomPredictor,
)
from antbench.runner import load_traces  # noqa: E402
from antbench.scorer import score_traces  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
RESAMPLES = 20_000

PREDICTORS = {
    "random": RandomPredictor,
    "lexical": LexicalPredictor,
    "llm": LLMPredictor,
}


def permutation_p(diffs: list[float], resamples: int, seed: int = 0) -> float:
    """Two-sided p under the null that each difference's sign is arbitrary."""
    observed = abs(statistics.mean(diffs))
    rng = random.Random(seed)
    extreme = sum(
        1
        for _ in range(resamples)
        if abs(statistics.mean([d if rng.random() < 0.5 else -d for d in diffs]))
        >= observed
    )
    return extreme / resamples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="lexical", choices=sorted(PREDICTORS))
    parser.add_argument("--b", default="llm", choices=sorted(PREDICTORS))
    parser.add_argument("--budget", type=int, default=8)
    args = parser.parse_args()

    traces = load_traces(DATA / "traces")
    if not traces:
        print("no traces found; run scripts/collect.py first")
        return 1

    scores_a = score_traces(traces, PREDICTORS[args.a](), DATA / "repos")
    scores_b = score_traces(traces, PREDICTORS[args.b](), DATA / "repos")
    if len(scores_a) != len(scores_b):
        print("predictors scored different delegation counts; cannot pair")
        return 1

    k = args.budget
    diffs = [x.recall_at(k) - y.recall_at(k) for x, y in zip(scores_a, scores_b)]
    n = len(diffs)
    mean = statistics.mean(diffs)
    wins = sum(1 for d in diffs if d > 0.01)
    losses = sum(1 for d in diffs if d < -0.01)
    p = permutation_p(diffs, RESAMPLES)

    print(f"{args.a} vs {args.b}, recall@{k}, {n} paired delegations\n")
    print(f"  mean paired difference : {mean:+.3f}")
    print(f"  sd                     : {statistics.stdev(diffs):.3f}" if n > 1 else "")
    print(f"  {args.a} better        : {wins}")
    print(f"  {args.b} better        : {losses}")
    print(f"  tied                   : {n - wins - losses}")
    print(f"  two-sided permutation p: {p:.4f}  ({RESAMPLES:,} resamples)")

    print()
    if p < 0.05:
        print(f"  Separated at p < 0.05: {args.a} differs from {args.b}.")
    else:
        print(
            f"  Not separated at p < 0.05. The supportable claim is that\n"
            f"  {args.a} is at least as good as {args.b}, not that it beats it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
