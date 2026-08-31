"""Regenerate every number cited in the pitch, from the traces on disk.

A benchmark whose headline figures cannot be reproduced from its own artefacts
is an assertion. This prints, in one pass, the corpus statistics, the
predictor table, the significance tests, and the two predictor-independent
corpus properties -- so any figure in PITCH.md can be checked against a single
command.

    uv run python scripts/report.py            # free predictors only
    uv run python scripts/report.py --with-llm # includes the paid comparison
"""

from __future__ import annotations

import argparse
import collections
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
from antbench.runner import corpus_stats, load_traces  # noqa: E402
from antbench.schema import NeedLabel  # noqa: E402
from antbench.scorer import BUDGETS, aggregate, score_traces  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from significance import permutation_p  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()

    traces = load_traces(DATA / "traces")
    if not traces:
        print("no traces found; run scripts/collect.py first")
        return 1

    rule("CORPUS")
    stats = corpus_stats(traces)
    for key in ("traces", "delegations", "scorable", "completed",
                "total_accesses", "mean_accesses_per_delegation",
                "mean_needed_paths", "mean_waste_ratio", "wasted_tokens"):
        print(f"  {key:32} {stats.get(key)}")
    repos = collections.Counter(t.repo for t in traces)
    print(f"  {'repositories':32} {len(repos)}")
    for repo, count in repos.most_common():
        print(f"      {repo:28} {count}")

    rule("PREDICTORS")
    predictors = [RandomPredictor(), LexicalPredictor()]
    if args.with_llm:
        predictors.append(LLMPredictor())

    scored = {}
    header = f"  {'predictor':10}" + "".join(f"  rec@{k}  prec@{k}" for k in BUDGETS)
    print(header)
    for predictor in predictors:
        rows = score_traces(traces, predictor, DATA / "repos")
        scored[predictor.name] = rows
        summary = aggregate(rows, predictor.name)
        line = f"  {predictor.name:10}"
        for k in BUDGETS:
            line += f"  {summary[f'recall@{k}']:5.3f}  {summary[f'precision@{k}']:6.3f}"
        print(line)

    rule("SIGNIFICANCE (recall@8, paired permutation)")
    pairs = [("lexical", "random")]
    if args.with_llm:
        pairs.append(("lexical", "llm"))
    for a, b in pairs:
        if a not in scored or b not in scored:
            continue
        diffs = [
            x.recall_at(8) - y.recall_at(8)
            for x, y in zip(scored[a], scored[b])
        ]
        p = permutation_p(diffs, 20_000)
        wins = sum(1 for d in diffs if d > 0.01)
        losses = sum(1 for d in diffs if d < -0.01)
        verdict = "separated" if p < 0.05 else "NOT separated"
        print(
            f"  {a} vs {b}: mean {statistics.mean(diffs):+.3f}  "
            f"{wins}W/{losses}L/{len(diffs) - wins - losses}T  "
            f"p={p:.4f}  ({verdict} at 0.05)"
        )

    rule("REPOSITORY STRUCTURE vs PREDICTABILITY")
    # Definitions per indexed file, computed from the repo alone with no trace
    # data, as a candidate advance indicator of whether lexical prediction will
    # work on a codebase. See pitch/PREDICTION.md for the pre-registered test.
    from antbench.predictor import RepoIndex  # noqa: PLC0415

    per_repo: dict[str, list[float]] = collections.defaultdict(list)
    for row in scored.get("lexical", []):
        per_repo[row.repo].append(row.recall_at(8))

    print(f"  {'repo':28} {'files':>6} {'defs/file':>9} {'recall@8':>9} {'n':>3}")
    points: list[tuple[float, float]] = []
    for repo, recalls in sorted(per_repo.items()):
        root = DATA / "repos" / repo.replace("/", "__")
        if not root.is_dir():
            continue
        index = RepoIndex.build(root)
        if not index.definitions:
            continue
        density = statistics.mean(len(v) for v in index.definitions.values())
        recall = statistics.mean(recalls)
        points.append((density, recall))
        print(
            f"  {repo:28} {len(index.paths):6d} {density:9.1f} "
            f"{recall:9.2f} {len(recalls):3d}"
        )

    # Is the spread across repositories larger than chance grouping produces?
    # Shuffle the repository labels across delegations and rebuild the same
    # spread statistic; a real repository effect should exceed nearly every
    # relabelling. This is the test behind the per-repo claim in the pitch.
    labelled = [(row.repo, row.recall_at(8)) for row in scored.get("lexical", [])]
    if len({r for r, _ in labelled}) > 1:
        def spread(pairs: list[tuple[str, float]]) -> float:
            groups: dict[str, list[float]] = collections.defaultdict(list)
            for repo, value in pairs:
                groups[repo].append(value)
            return statistics.pstdev([statistics.mean(v) for v in groups.values()])

        observed = spread(labelled)
        labels = [r for r, _ in labelled]
        values = [v for _, v in labelled]
        rng = random.Random(20260829)
        resamples = 20_000
        at_least = sum(
            1
            for _ in range(resamples)
            if spread(list(zip(rng.sample(labels, len(labels)), values))) >= observed
        )
        p_repo = (at_least + 1) / (resamples + 1)
        print(
            f"\n  between-repository spread: sd of per-repo means = {observed:.3f}, "
            f"p={p_repo:.4f}\n  ({resamples:,} label permutations; "
            f"{'separated' if p_repo < 0.05 else 'NOT separated'} at 0.05)"
        )

    if len(points) > 2:
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
        if sx and sy:
            r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (len(xs) * sx * sy)
            print(f"\n  corr(defs per file, recall@8) = {r:+.2f}  "
                  f"over {len(points)} repositories")

    rule("NEED TIMING (share of accesses genuinely needed, by decile)")
    delegations = [d for t in traces for d in t.delegations if d.needed_paths]
    needed: collections.Counter[int] = collections.Counter()
    total: collections.Counter[int] = collections.Counter()
    for d in delegations:
        n = len(d.accesses)
        if n < 5:
            continue
        for i, a in enumerate(d.accesses):
            dec = min(9, 10 * i // n)
            total[dec] += 1
            if a.label is not None and a.label is not NeedLabel.UNUSED:
                needed[dec] += 1
    early = sum(needed[d] for d in range(3)) / max(sum(total[d] for d in range(3)), 1)
    late = sum(needed[d] for d in range(7, 10)) / max(
        sum(total[d] for d in range(7, 10)), 1
    )
    print(f"  first 30%: {early:.2f}    last 30%: {late:.2f}")
    print("  Comparable rates mean spawn-time injection is structurally capped.")

    rule("PITCH TABLE (paste into PITCH.md)")
    print("| predictor | recall@8 | precision@1 | any_hit@8 | cost | n |")
    print("|---|---|---|---|---|---|")
    costs = {"random": "free", "lexical": "free",
             "llm": "one call per delegation"}
    for predictor in predictors:
        a = aggregate(scored[predictor.name], predictor.name)
        label = "lexical (IDF)" if predictor.name == "lexical" else predictor.name
        if predictor.name == "llm":
            label = "LLM (gpt-5-mini)"
        bold = "**" if predictor.name == "lexical" else ""
        print(
            f"| {label} | {bold}{a['recall@8']:.3f}{bold} | "
            f"{a['precision@1']:.3f} | {a['any_hit@8']:.3f} | "
            f"{costs[predictor.name]} | {a['delegations']} |"
        )

    rule("WASTE BY RETRIEVAL KIND")
    kinds: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for d in delegations:
        for a in d.accesses:
            row = kinds[a.kind.value]
            row[0] += 1
            if a.label is NeedLabel.UNUSED:
                row[1] += 1
                row[2] += a.result_tokens
    for kind, (calls, wasted, tokens) in sorted(
        kinds.items(), key=lambda kv: -kv[1][2]
    ):
        print(
            f"  {kind:14} calls={calls:4d}  wasted={wasted:4d} "
            f"({wasted / calls:.0%})  tokens={tokens:8,d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
