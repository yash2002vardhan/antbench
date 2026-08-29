"""Score predictors against collected traces.

    uv run python scripts/evaluate.py              # all predictors
    uv run python scripts/evaluate.py --no-llm     # free predictors only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from antbench.predictor import build_predictors  # noqa: E402
from antbench.runner import load_traces  # noqa: E402
from antbench.scorer import BUDGETS, aggregate, score_traces  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the LLM predictor (no API spend)")
    parser.add_argument("--out", default=str(DATA / "results.json"))
    args = parser.parse_args()

    traces = load_traces(DATA / "traces")
    if not traces:
        print("no traces found; run scripts/collect.py first")
        return 1

    completed = sum(
        1 for t in traces for d in t.delegations if d.completed and d.needed_paths
    )
    print(f"{len(traces)} traces, {completed} scorable delegations\n")
    if not completed:
        print("no completed delegations with labelled needs; nothing to score")
        return 1

    results = []
    for predictor in build_predictors(include_llm=not args.no_llm):
        scores = score_traces(traces, predictor, DATA / "repos")
        summary = aggregate(scores, predictor.name)
        results.append(summary)

        print(f"--- {predictor.name} ---")
        print(f"  delegations scored : {summary.get('delegations', 0)}")
        print(f"  mean needed paths  : {summary.get('mean_needed_paths', 0)}")
        for k in BUDGETS:
            print(
                f"  @{k}: recall={summary.get(f'recall@{k}', 0):.3f}  "
                f"precision={summary.get(f'precision@{k}', 0):.3f}  "
                f"tokens={summary.get(f'token_saving@{k}', 0):.3f}  "
                f"any_hit={summary.get(f'any_hit@{k}', 0):.3f}"
            )
        print()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"written to {args.out}")

    lexical = next((r for r in results if r["predictor"] == "lexical"), None)
    llm = next((r for r in results if r["predictor"] == "llm"), None)
    if lexical and llm:
        top = max(BUDGETS)
        gap = llm[f"recall@{top}"] - lexical[f"recall@{top}"]
        print(
            f"\nlexical->llm recall@{top} gap: {gap:+.3f}  "
            f"({lexical[f'recall@{top}']:.3f} -> {llm[f'recall@{top}']:.3f})"
        )
        print(
            "A small gap means prediction is a string-matching problem and a "
            "serving layer needs no model call."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
