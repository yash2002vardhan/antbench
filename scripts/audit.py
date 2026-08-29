"""Flag delegations whose ground truth looks implausible.

Mechanical labelling can be confidently wrong, and a wrong label does not
look like a bug -- it looks like a finding. Four labelling bugs in this
project's history were caught by reading traces, not by tests: a grep matching
fifteen files once produced 28 "needed" paths for a two-file fix, among them
versioneer.py.

This is the cheap standing version of that inspection. It does not decide
correctness; it surfaces the delegations a human should look at.

    uv run python scripts/audit.py

A flag is a prompt to look, not a verdict. Build helpers and package
__init__.py files are legitimately needed when a fix uses their symbols.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from antbench.runner import load_traces  # noqa: E402
from antbench.schema import Delegation  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"

# Need should exceed diff -- a fix requires reading more than it edits -- but
# not without limit. Calibrated against a corpus whose sound delegations run
# between 1:1 and 3.5:1.
RATIO_SLACK = 4
RATIO_MULTIPLIER = 3

# Paths that are rarely load-bearing for a bug fix. Presence is a prompt to
# look, not evidence of error.
# Deliberately narrow: only paths that cannot plausibly be load-bearing for a
# source fix. Earlier drafts flagged rcsetup.py and package __init__.py files,
# which are legitimately needed when a fix touches config validation or
# re-exports -- a check that cries wolf stops being read.
SUSPECT_MARKERS = (
    "versioneer", "/doc/", "/docs/", "asv_bench", "/benchmarks/",
)


def flags_for(delegation: Delegation) -> list[str]:
    need = sorted(delegation.needed_paths)
    diff = sorted(delegation.files_in_diff)
    found: list[str] = []

    if len(need) > RATIO_MULTIPLIER * max(len(diff), 1) + RATIO_SLACK:
        found.append(f"need({len(need)}) >> diff({len(diff)})")

    suspects = [
        p for p in need if any(marker in f"/{p}" for marker in SUSPECT_MARKERS)
    ]
    if suspects:
        found.append(
            "suspect: " + ", ".join(PurePosixPath(p).name for p in suspects[:3])
        )

    if diff and not set(diff) & set(need):
        # The edited file should almost always appear in need; if it does not,
        # path normalisation between workspace and groundtruth has drifted.
        found.append("no diff file in need (normalisation?)")

    return found


def main() -> int:
    traces = load_traces(DATA / "traces")
    if not traces:
        print("no traces found; run scripts/collect.py first")
        return 1

    delegations = [(t, d) for t in traces for d in t.delegations if d.needed_paths]
    print(f"{len(delegations)} scorable delegations\n")

    flagged = 0
    for trace, delegation in delegations:
        found = flags_for(delegation)
        repo = trace.repo.split("/")[-1]
        status = "; ".join(found) if found else "ok"
        print(
            f"  {repo:12} diff={len(delegation.files_in_diff):2d} "
            f"need={len(delegation.needed_paths):2d}  {status}"
        )
        if found:
            flagged += 1
            print(f"       diff: {sorted(delegation.files_in_diff)}")
            print(f"       need: {sorted(delegation.needed_paths)}")

    print(f"\n{flagged}/{len(delegations)} flagged for inspection")
    if flagged:
        print("Review each before trusting aggregate numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
