"""Corpus properties that hold regardless of any predictor.

These bound what a predictive layer can achieve before a single prediction is
made, so they belong in the methodology rather than the results:

  timing  -- when in a delegation need appears. If need were front-loaded, a
             delegation-time predictor could capture most of it. If it is
             uniform, spawn-time injection is structurally capped and a warm
             cache behind the retrieval interface is required to reach the
             rest.
  kinds   -- which retrieval verbs waste, by call count and by tokens. These
             disagree, and the token column is the one that matters: a verb
             that misses often but cheaply costs less than one that misses
             rarely and expensively.

    uv run python scripts/analyze.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from antbench.runner import load_traces  # noqa: E402
from antbench.schema import Delegation, NeedLabel  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
DECILES = 10
MIN_ACCESSES = 5


def timing(delegations: list[Delegation]) -> None:
    """Fraction of accesses that were genuine need, by decile of delegation."""
    needed: collections.Counter[int] = collections.Counter()
    total: collections.Counter[int] = collections.Counter()

    for delegation in delegations:
        count = len(delegation.accesses)
        if count < MIN_ACCESSES:
            continue
        for position, access in enumerate(delegation.accesses):
            decile = min(DECILES - 1, DECILES * position // count)
            total[decile] += 1
            if access.label is not None and access.label is not NeedLabel.UNUSED:
                needed[decile] += 1

    print("WHEN NEED APPEARS (share of accesses that were genuinely needed)")
    for decile in range(DECILES):
        if not total[decile]:
            continue
        share = needed[decile] / total[decile]
        print(
            f"  {decile * 10:3d}-{decile * 10 + 10:3d}%  "
            f"{needed[decile]:3d}/{total[decile]:3d}  {share:.2f}  "
            f"{'#' * int(share * 40)}"
        )

    early = sum(needed[d] for d in range(3))
    early_total = sum(total[d] for d in range(3))
    late = sum(needed[d] for d in range(7, DECILES))
    late_total = sum(total[d] for d in range(7, DECILES))
    if early_total and late_total:
        print(
            f"\n  first 30%: {early / early_total:.2f}   "
            f"last 30%: {late / late_total:.2f}"
        )
        print(
            "  Comparable rates mean need keeps arriving late, so spawn-time\n"
            "  injection cannot reach all of it -- a warm cache behind the\n"
            "  normal retrieval interface is required for the remainder."
        )


def by_kind(delegations: list[Delegation]) -> None:
    """Waste per retrieval verb, by call count and by tokens."""
    stats: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for delegation in delegations:
        for access in delegation.accesses:
            row = stats[access.kind.value]
            row[0] += 1
            if access.label is NeedLabel.UNUSED:
                row[1] += 1
                row[2] += access.result_tokens

    print("\nWASTE BY RETRIEVAL KIND (ordered by wasted tokens)")
    for kind, (calls, wasted, tokens) in sorted(
        stats.items(), key=lambda kv: -kv[1][2]
    ):
        print(
            f"  {kind:14} calls={calls:4d}  wasted={wasted:4d} "
            f"({wasted / calls:.0%})  wasted_tokens={tokens:8,d}"
        )
    print(
        "\n  Call-count waste and token waste rank differently. Token waste is\n"
        "  what a prefetch layer actually removes."
    )


def main() -> int:
    traces = load_traces(DATA / "traces")
    if not traces:
        print("no traces found; run scripts/collect.py first")
        return 1

    delegations = [d for t in traces for d in t.delegations if d.needed_paths]
    if not delegations:
        print("no delegations with labelled need")
        return 1

    print(f"{len(traces)} traces, {len(delegations)} scorable delegations\n")
    timing(delegations)
    by_kind(delegations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
