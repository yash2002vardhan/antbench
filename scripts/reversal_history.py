"""Re-score the lexical-vs-LLM comparison at several corpus sizes.

The pitch's sharpest argument is that this comparison crossed zero twice as the
corpus grew, so no single corpus size supports a confident claim. That history
was recorded during collection, but `finalize.py` deliberately re-scores only
the final corpus -- which left the progression asserted in prose and absent
from any artefact a reviewer could run.

This closes that gap. Delegations are taken in a fixed order (repo, then
delegation id, matching `score_traces`) and truncated to each prefix size, so
the n=25 row is the first 25 delegations of the same ordering that produces the
n=57 row. That is a reconstruction, not a recording: the original n=25 run
scored whichever delegations happened to exist at the time. It answers the
weaker but checkable question -- does the sign of this difference depend on how
much data you have? -- which is the claim the pitch actually makes.

    uv run python scripts/reversal_history.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from antbench.predictor import LexicalPredictor, LLMPredictor  # noqa: E402
from antbench.runner import load_traces  # noqa: E402
from antbench.scorer import score_traces  # noqa: E402
from significance import permutation_p  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
SIZES = (25, 41, 57)


def main() -> int:
    traces = load_traces(DATA / "traces")
    if not traces:
        print("no traces found; run scripts/collect.py first")
        return 1

    lexical = score_traces(traces, LexicalPredictor(), DATA / "repos")
    llm = score_traces(traces, LLMPredictor(), DATA / "repos")
    total = len(lexical)

    print(f"lexical vs LLM, recall@8, by corpus prefix ({total} available)\n")
    print(f"  {'n':>4}  {'lexical':>8} {'llm':>8} {'diff':>8} {'W/L/T':>10} {'p':>7}")
    for n in [s for s in SIZES if s <= total] + ([total] if total not in SIZES else []):
        a, b = lexical[:n], llm[:n]
        diffs = [x.recall_at(8) - y.recall_at(8) for x, y in zip(a, b)]
        wins = sum(1 for d in diffs if d > 0.01)
        losses = sum(1 for d in diffs if d < -0.01)
        print(
            f"  {n:4d}  "
            f"{statistics.mean(x.recall_at(8) for x in a):8.3f} "
            f"{statistics.mean(y.recall_at(8) for y in b):8.3f} "
            f"{statistics.mean(diffs):+8.3f} "
            f"{f'{wins}/{losses}/{len(diffs)-wins-losses}':>10} "
            f"{permutation_p(diffs, 20_000):7.4f}"
        )

    print(
        "\n  If the sign of `diff` changes across rows, no single corpus size\n"
        "  supports a confident claim about which predictor is better."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
