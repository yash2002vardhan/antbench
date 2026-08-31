"""Test whether import-graph expansion closes the identified failure class.

The corpus has one clean failure mode: a delegation describes *behaviour*
while the file implementing it is named for its *architectural location*.
Pylint's XDG storage change lives in `config/option_manager_mixin.py` and the
message never says "config"; pytest's import fix lives in `pathlib.py` and the
message says "the import path logic". Lexical matching cannot recover a word
the message does not contain.

The obvious fix: the file that *is* named usually imports the one that is not,
so expand lexical seeds one hop along import edges. This script tests that.

    uv run python scripts/experiment_graph.py

RESULT: it does not work. Expansion costs -0.049 recall@8 over the 57-delegation
corpus, winning on 4 delegations and losing on 15 (p=0.0185). The short version
is that import graphs are too dense (scikit-learn averages 7.0 neighbours per
file, max 203), so expansion trades a known-decent seed for a near-random pick
from dozens of candidates. It helps only where seeds are already worthless.
"""

from __future__ import annotations

import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from antbench.predictor import (  # noqa: E402
    GraphExpandedPredictor,
    LexicalPredictor,
    RepoIndex,
    build_import_graph,
)
from antbench.runner import load_traces  # noqa: E402
from antbench.scorer import aggregate, score_traces  # noqa: E402
from significance import permutation_p  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> int:
    traces = load_traces(DATA / "traces")
    if not traces:
        print("no traces found; run scripts/collect.py first")
        return 1

    scored = {}
    for predictor in (LexicalPredictor(), GraphExpandedPredictor()):
        rows = score_traces(traces, predictor, DATA / "repos")
        scored[predictor.name] = rows
        summary = aggregate(rows, predictor.name)
        print(
            f"  {predictor.name:16} "
            + "  ".join(f"r@{k}={summary[f'recall@{k}']:.3f}" for k in (1, 3, 5, 8))
            + f"  any@8={summary['any_hit@8']:.3f}"
        )

    base, expanded = scored["lexical"], scored["lexical+graph"]
    diffs = [b.recall_at(8) - a.recall_at(8) for a, b in zip(base, expanded)]
    wins = sum(1 for d in diffs if d > 0.01)
    losses = sum(1 for d in diffs if d < -0.01)
    p = permutation_p(diffs, 20_000)
    print(
        f"\n  graph vs lexical: mean {statistics.mean(diffs):+.3f}  "
        f"{wins}W/{losses}L/{len(diffs) - wins - losses}T  p={p:.4f}"
    )

    per_repo: dict[str, list[list[float]]] = collections.defaultdict(
        lambda: [[], []]
    )
    for a, b in zip(base, expanded):
        repo = a.repo.split("/")[-1]
        per_repo[repo][0].append(a.recall_at(8))
        per_repo[repo][1].append(b.recall_at(8))

    print(f"\n  {'repo':12} {'lexical':>8} {'graph':>8} {'delta':>7}")
    for repo, (a_vals, b_vals) in sorted(per_repo.items()):
        ma, mb = statistics.mean(a_vals), statistics.mean(b_vals)
        print(f"  {repo:12} {ma:8.2f} {mb:8.2f} {mb - ma:+7.2f}")

    print("\n  graph density (why expansion is imprecise):")
    for repo in sorted({t.repo for t in traces}):
        root = DATA / "repos" / repo.replace("/", "__")
        if not root.is_dir():
            continue
        index = RepoIndex.build(root)
        graph = build_import_graph(index)
        degrees = [len(v) for v in graph.values()] or [0]
        print(
            f"    {repo:26} nodes={len(graph):5d} "
            f"mean_degree={statistics.mean(degrees):5.1f} max={max(degrees):4d}"
        )

    print(
        "\n  Expansion trades a known-decent seed for a pick from dozens of\n"
        "  neighbours. It helps only where the seeds are already worthless."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
