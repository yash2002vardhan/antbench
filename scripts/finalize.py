"""Score the pre-registered prediction and regenerate every published figure.

Run once, after collection completes. It does three things in order:

  1. Scores the sphinx prediction recorded in pitch/PREDICTION.md, stating
     plainly whether it held, before any other analysis can colour the reading.
  2. Re-scores every predictor on identical data, so the lexical/LLM comparison
     is no longer split across corpus sizes.
  3. Emits the pitch table and the significance tests.

    uv run python scripts/finalize.py            # free predictors
    uv run python scripts/finalize.py --with-llm # includes the paid comparison
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from antbench.predictor import (  # noqa: E402
    LexicalPredictor,
    LLMPredictor,
    RandomPredictor,
    RepoIndex,
)
from antbench.runner import corpus_stats, load_traces  # noqa: E402
from antbench.scorer import aggregate, score_traces  # noqa: E402
from significance import permutation_p  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"

# From pitch/PREDICTION.md, recorded before any sphinx trace was scored.
PREDICTED_LOW, PREDICTED_HIGH = 0.40, 0.60
FALSIFY_BELOW = 0.15


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()

    traces = load_traces(DATA / "traces")
    if not traces:
        print("no traces found")
        return 1

    lexical_scores = score_traces(traces, LexicalPredictor(), DATA / "repos")
    by_repo: dict[str, list[float]] = collections.defaultdict(list)
    for row in lexical_scores:
        by_repo[row.repo].append(row.recall_at(8))

    print("=" * 66)
    print("PRE-REGISTERED PREDICTION: sphinx")
    print("=" * 66)
    sphinx = by_repo.get("sphinx-doc/sphinx", [])
    if not sphinx:
        print("  no sphinx delegations scored yet")
    else:
        observed = statistics.mean(sphinx)
        print(f"  predicted : {PREDICTED_LOW:.2f}-{PREDICTED_HIGH:.2f} recall@8")
        print(f"  falsify   : below {FALSIFY_BELOW:.2f}")
        print(f"  observed  : {observed:.3f}  (n={len(sphinx)} delegations)")
        if observed < FALSIFY_BELOW:
            verdict = "FALSIFIED — the structural proxy does not explain low predictability"
        elif PREDICTED_LOW <= observed <= PREDICTED_HIGH:
            verdict = "HELD — observed inside the predicted interval"
        else:
            verdict = "PARTIAL — outside the interval but above falsification"
        print(f"  verdict   : {verdict}")

    print()
    print("=" * 66)
    print("REPOSITORY STRUCTURE vs PREDICTABILITY")
    print("=" * 66)
    points: list[tuple[float, float]] = []
    print(f"  {'repo':28} {'files':>6} {'defs/file':>9} {'recall@8':>9} {'n':>3}")
    for repo, recalls in sorted(by_repo.items()):
        root = DATA / "repos" / repo.replace("/", "__")
        if not root.is_dir():
            continue
        index = RepoIndex.build(root)
        if not index.definitions:
            continue
        density = statistics.mean(len(v) for v in index.definitions.values())
        recall = statistics.mean(recalls)
        points.append((density, recall))
        print(f"  {repo:28} {len(index.paths):6d} {density:9.1f} "
              f"{recall:9.2f} {len(recalls):3d}")
    if len(points) > 2:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
        if sx and sy:
            r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (len(xs) * sx * sy)
            print(f"\n  corr(defs/file, recall@8) = {r:+.2f} over {len(points)} repos")

    print()
    print("=" * 66)
    print("CORPUS")
    print("=" * 66)
    stats = corpus_stats(traces)
    for key in ("traces", "delegations", "scorable", "completed",
                "mean_accesses_per_delegation", "mean_needed_paths",
                "mean_waste_ratio", "wasted_tokens"):
        print(f"  {key:32} {stats.get(key)}")

    print()
    print("=" * 66)
    print("PITCH TABLE")
    print("=" * 66)
    predictors = [RandomPredictor(), LexicalPredictor()]
    if args.with_llm:
        predictors.append(LLMPredictor())
    scored = {"lexical": lexical_scores}
    costs = {"random": "free", "lexical": "free",
             "llm": "one call per delegation"}
    labels = {"random": "random", "lexical": "lexical (IDF)",
              "llm": "LLM (gpt-5-mini)"}

    print("| predictor | recall@8 | precision@1 | any_hit@8 | cost | n |")
    print("|---|---|---|---|---|---|")
    for predictor in predictors:
        rows = (
            lexical_scores
            if predictor.name == "lexical"
            else score_traces(traces, predictor, DATA / "repos")
        )
        scored[predictor.name] = rows
        a = aggregate(rows, predictor.name)
        bold = "**" if predictor.name == "lexical" else ""
        print(
            f"| {labels[predictor.name]} | {bold}{a['recall@8']:.3f}{bold} | "
            f"{a['precision@1']:.3f} | {a['any_hit@8']:.3f} | "
            f"{costs[predictor.name]} | {a['delegations']} |"
        )

    print()
    print("=" * 66)
    print("SIGNIFICANCE (recall@8, paired permutation)")
    print("=" * 66)
    pairs = [("lexical", "random")]
    if args.with_llm:
        pairs.append(("lexical", "llm"))
    for a_name, b_name in pairs:
        diffs = [
            x.recall_at(8) - y.recall_at(8)
            for x, y in zip(scored[a_name], scored[b_name])
        ]
        p = permutation_p(diffs, 20_000)
        wins = sum(1 for d in diffs if d > 0.01)
        losses = sum(1 for d in diffs if d < -0.01)
        print(
            f"  {a_name} vs {b_name}: mean {statistics.mean(diffs):+.3f}  "
            f"{wins}W/{losses}L/{len(diffs) - wins - losses}T  p={p:.4f}  "
            f"({'separated' if p < 0.05 else 'NOT separated'} at 0.05)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
