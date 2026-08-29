"""Measure how sensitive the ground truth is to READ_BEFORE_EDIT_WINDOW.

The window is the one arbitrary parameter in the ground truth: how many turns
may separate a read from the edit it supposedly enabled. It governs only the
weakest label (READ_BEFORE_EDIT) -- USED_IN_DIFF needs no window at all.

    uv run python scripts/sweep_window.py

RESULT on the pilot corpus: the table is exactly flat, because the
`rbe labels` column is zero at every window. READ_BEFORE_EDIT can only apply
to a file that was edited yet is absent from the final diff, and every
successfully edited file lands in the diff, where the stronger USED_IN_DIFF
claims it first. So no reported number depends on this constant.

Keep running this after any change to the labelling rules. A non-zero
`rbe labels` column means the property no longer holds and the window has
become load-bearing again -- at which point the sensitivity, not a single
number, is what should be reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from antbench.groundtruth import (  # noqa: E402
    DEFAULT_READ_BEFORE_EDIT_WINDOW,
    label_delegation,
    need_summary,
)
from antbench.runner import load_traces  # noqa: E402
from antbench.schema import NeedLabel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

WINDOWS: tuple[int | None, ...] = (0, 2, 4, 6, 10, 20, None)


def main() -> int:
    traces = load_traces(DATA / "traces")
    if not traces:
        print("no traces found; run scripts/collect.py first")
        return 1

    delegations = [d for t in traces for d in t.delegations]
    repo_root = {
        t.repo.replace("/", "__"): DATA / "repos" / t.repo.replace("/", "__")
        for t in traces
    }
    root_for = {
        d.delegation_id: repo_root[t.repo.replace("/", "__")]
        for t in traces
        for d in t.delegations
    }

    print(f"{len(traces)} traces, {len(delegations)} delegations\n")
    header = (
        f"{'window':>8}  {'scorable':>8}  {'mean need':>9}  "
        f"{'mean waste':>10}  {'rbe labels':>10}"
    )
    print(header)
    print("-" * len(header))

    baseline: dict[str, float] = {}
    total_rbe = 0
    for window in WINDOWS:
        for delegation in delegations:
            label_delegation(
                delegation,
                read_before_edit_window=window,
                repo_root=root_for[delegation.delegation_id],
            )

        scorable = [d for d in delegations if d.needed_paths]
        if not scorable:
            continue
        summaries = [need_summary(d) for d in scorable]
        mean_need = sum(s["needed_paths"] for s in summaries) / len(summaries)
        mean_waste = sum(s["waste_ratio"] for s in summaries) / len(summaries)
        rbe = sum(
            1
            for d in scorable
            for a in d.accesses
            if a.label is NeedLabel.READ_BEFORE_EDIT
        )

        label = "unbounded" if window is None else str(window)
        marker = " *" if window == DEFAULT_READ_BEFORE_EDIT_WINDOW else ""
        print(
            f"{label:>8}  {len(scorable):8d}  {mean_need:9.2f}  "
            f"{mean_waste:10.3f}  {rbe:10d}{marker}"
        )
        total_rbe += rbe
        if window == DEFAULT_READ_BEFORE_EDIT_WINDOW:
            baseline = {"need": mean_need, "waste": mean_waste}

    # Restore the default so the traces on disk stay labelled as collected.
    for delegation in delegations:
        label_delegation(
            delegation,
            read_before_edit_window=DEFAULT_READ_BEFORE_EDIT_WINDOW,
            repo_root=root_for[delegation.delegation_id],
        )

    print("\n* = default")
    if total_rbe == 0:
        print(
            "\nREAD_BEFORE_EDIT never fired: every edited file reached the "
            "final\ndiff, where USED_IN_DIFF claims it first. The window is "
            "not\nload-bearing and no reported number depends on it."
        )
    else:
        print(
            f"\nREAD_BEFORE_EDIT fired {total_rbe} times, so the window IS "
            "load-bearing.\nReport the sensitivity above rather than a single "
            "number."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
