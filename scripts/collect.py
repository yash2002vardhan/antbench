"""Collect traces over selected SWE-bench tasks.

    uv run python scripts/collect.py --limit 2      # smoke test
    uv run python scripts/collect.py                # full pilot
    uv run python scripts/collect.py --stats-only   # report on what exists
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from antbench.agents import SUPERVISOR_MODEL, WORKER_MODEL  # noqa: E402
from antbench.runner import (  # noqa: E402
    RunConfig,
    corpus_stats,
    load_traces,
    run_pilot,
)
from antbench.schema import Trace  # noqa: E402
from antbench.tasks import fetch_instances, select_tasks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="trace only the first N tasks")
    parser.add_argument("--repo", action="append", default=None,
                        help="restrict to a repo (repeatable), e.g. pytest-dev/pytest")
    parser.add_argument("--supervisor-model", default=SUPERVISOR_MODEL)
    parser.add_argument("--worker-model", default=WORKER_MODEL)
    parser.add_argument("--no-resume", action="store_true",
                        help="re-trace tasks that already have a trace file")
    parser.add_argument("--stats-only", action="store_true",
                        help="summarise existing traces without running anything")
    parser.add_argument("--per-repo", type=int, default=None,
                        help="cap tasks per repository, filling thinnest first")
    args = parser.parse_args()

    traces_dir = DATA / "traces"

    if args.stats_only:
        stats = corpus_stats(load_traces(traces_dir))
        print(json.dumps(stats, indent=2))
        return 0

    rows = fetch_instances(DATA / "swebench_verified.json")
    tasks = select_tasks(rows, repos=set(args.repo) if args.repo else None)

    if args.per_repo:
        # Balance the corpus rather than deepening whichever repository happens
        # to have the most eligible instances. Per-repo means are the weakest
        # numbers in the pilot -- seaborn and scikit-learn had two delegations
        # each -- so added budget should go where n is smallest.
        already = collections.Counter()
        for path in traces_dir.glob("*.json"):
            try:
                already[Trace.model_validate_json(path.read_text()).repo] += 1
            except Exception:
                continue
        by_repo: dict[str, list] = collections.defaultdict(list)
        for task in tasks:
            by_repo[task.repo].append(task)

        balanced: list = []
        for repo, repo_tasks in by_repo.items():
            room = max(0, args.per_repo - already[repo])
            balanced.extend(repo_tasks[:room + already[repo]][already[repo]:][:room])
        # Thinnest repositories first, so an interrupted run still balances.
        balanced.sort(key=lambda t: (already[t.repo], t.repo, t.instance_id))
        tasks = balanced

    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        print("no tasks matched the filters")
        return 1

    print(f"{len(tasks)} task(s) selected")
    print(f"supervisor={args.supervisor_model}  worker={args.worker_model}")

    config = RunConfig(
        traces_dir=traces_dir,
        repos_dir=DATA / "repos",
        supervisor_model=args.supervisor_model,
        worker_model=args.worker_model,
        resume=not args.no_resume,
    )

    run_pilot(tasks, config)

    print("\n--- corpus ---")
    print(json.dumps(corpus_stats(load_traces(traces_dir)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
