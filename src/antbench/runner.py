"""Drive supervisor and workers over selected tasks, persisting each trace.

Traces are written one JSON file per task, immediately on completion. A pilot
run costs real money and real minutes; losing thirty of them to a crash in the
thirty-first would be the expensive kind of mistake, so nothing is buffered
until the end and an already-traced task is skipped on rerun.
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

from .agents import SUPERVISOR_MODEL, WORKER_MODEL, plan_task, run_delegation
from .groundtruth import label_delegation, need_summary
from .schema import Trace
from .tasks import Task, ensure_checkout
from .workspace import Workspace


@dataclass
class RunConfig:
    traces_dir: Path
    repos_dir: Path
    supervisor_model: str = SUPERVISOR_MODEL
    worker_model: str = WORKER_MODEL
    resume: bool = True


def trace_path(config: RunConfig, task: Task) -> Path:
    return config.traces_dir / f"{task.instance_id}.json"


def run_task(task: Task, config: RunConfig) -> Trace | None:
    """Trace one task: plan, then run each delegation on a clean checkout.

    Returns None if the task was skipped or failed outright. A failure in one
    task never aborts the run -- partial data across many tasks is worth more
    than a clean stop on the first bad repo.
    """
    destination = trace_path(config, task)
    if config.resume and destination.is_file():
        print(f"  skip (already traced)")
        return None

    try:
        checkout = ensure_checkout(task, config.repos_dir)
    except Exception as exc:
        print(f"  checkout failed: {type(exc).__name__}: {exc}")
        return None

    trace = Trace(
        trace_id=str(uuid.uuid4())[:8],
        repo=task.repo,
        base_commit=task.base_commit,
        task_prompt=task.problem_statement,
        source="swebench",
    )

    try:
        plan = plan_task(
            task.problem_statement, task.repo, model=config.supervisor_model
        )
    except Exception as exc:
        print(f"  planning failed: {type(exc).__name__}: {exc}")
        return None

    trace.supervisor_plan = plan.approach
    print(f"  plan: {len(plan.subtasks)} subtasks")

    workspace = Workspace(root=checkout)
    for index, subtask in enumerate(plan.subtasks, 1):
        started = time.time()
        try:
            delegation = run_delegation(
                workspace, subtask,
                parent_task_id=trace.trace_id,
                repo_head=task.base_commit,
                model=config.worker_model,
            )
        except Exception as exc:
            print(f"  [{index}] delegation crashed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            workspace.reset_repo()
            continue

        trace.delegations.append(delegation)
        summary = need_summary(delegation)
        print(
            f"  [{index}] {subtask.title[:44]:46} "
            f"{len(delegation.accesses):2d} acc  "
            # accesses that paid off, not distinct paths -- the two differ by
            # roughly an order of magnitude and reading one as the other makes
            # a healthy delegation look like a labelling bug
            f"{summary['needed']:2d} used  "
            f"{summary['needed_paths']:2d} paths  "
            f"waste={summary['waste_ratio']:.2f}  "
            f"{'done' if delegation.completed else delegation.failure_reason or 'incomplete':10} "
            f"{time.time() - started:5.0f}s"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(trace.model_dump_json(indent=2))
    return trace


def run_pilot(tasks: list[Task], config: RunConfig) -> list[Trace]:
    """Trace every task, reporting progress and surviving individual failures."""
    config.traces_dir.mkdir(parents=True, exist_ok=True)
    traces: list[Trace] = []

    for index, task in enumerate(tasks, 1):
        print(f"\n[{index}/{len(tasks)}] {task.instance_id}  ({task.repo})")
        trace = run_task(task, config)
        if trace is not None:
            traces.append(trace)

    return traces


def load_traces(traces_dir: Path, repos_dir: Path | None = None) -> list[Trace]:
    """Read every persisted trace, re-labelling against the current rules.

    Labels are written into the trace at collection time, but the labelling
    rules keep changing as real traces expose their flaws. Trusting the stored
    labels means every analysis silently reads whatever ground truth was
    current when the trace happened to be collected -- an audit once kept
    flagging a delegation whose label had already been fixed, because it was
    reading a label written hours earlier.

    So the raw accesses are the durable record and labels are derived on read.
    `repos_dir` lets the symbol-payoff rule see file contents; without it that
    rule degrades to explicit lookups, which is why it is passed by default.
    """
    if repos_dir is None:
        repos_dir = traces_dir.parent / "repos"

    traces: list[Trace] = []
    for path in sorted(traces_dir.glob("*.json")):
        try:
            trace = Trace.model_validate_json(path.read_text())
        except Exception as exc:
            print(f"skipping unreadable trace {path.name}: {exc}")
            continue

        root = repos_dir / trace.repo.replace("/", "__")
        for delegation in trace.delegations:
            label_delegation(
                delegation, repo_root=root if root.is_dir() else None
            )
        traces.append(trace)
    return traces


def corpus_stats(traces: list[Trace]) -> dict:
    """Aggregate view of a traced corpus.

    `mean_waste_ratio` is the headline diagnostic: it bounds how much a
    predictive layer could possibly recover on this corpus, since context the
    worker never wasted cannot be saved by prefetching it earlier.
    """
    all_delegations = [d for t in traces for d in t.delegations]
    if not all_delegations:
        return {"traces": len(traces), "delegations": 0}

    # Waste and need are only meaningful where the trace produced labelled
    # need. A worker that ran out of turns mid-orientation leaves no diff, so
    # every access scores UNUSED and its waste ratio approaches 1.0 by
    # construction -- averaging those in would manufacture a headline number
    # out of failures. Judged on evidence rather than on the `completed` flag,
    # so a worker that explored well and then declared BLOCKED still counts;
    # see scorer.is_scorable for the same criterion.
    delegations = [d for d in all_delegations if d.needed_paths]
    if not delegations:
        return {
            "traces": len(traces),
            "delegations": len(all_delegations),
            "scorable": 0,
            "note": "no delegations produced labelled need; waste undefined",
            "total_tokens": sum(t.total_tokens for t in traces),
        }

    summaries = [need_summary(d) for d in delegations]
    accesses = sum(s["accesses"] for s in summaries)
    needed = sum(s["needed"] for s in summaries)

    return {
        "traces": len(traces),
        "delegations": len(all_delegations),
        "scorable": len(delegations),
        "completed": sum(1 for d in all_delegations if d.completed),
        "note": "need/waste computed over delegations with labelled need",
        "total_accesses": accesses,
        "total_needed": needed,
        "mean_accesses_per_delegation": round(accesses / len(delegations), 2),
        "mean_needed_paths": round(
            sum(s["needed_paths"] for s in summaries) / len(delegations), 2
        ),
        "mean_waste_ratio": round(
            sum(s["waste_ratio"] for s in summaries) / len(summaries), 3
        ),
        "wasted_tokens": sum(s["wasted_result_tokens"] for s in summaries),
        "total_tokens": sum(t.total_tokens for t in traces),
    }
