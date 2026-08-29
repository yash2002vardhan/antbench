"""Trace schema for anticipatory-context measurement.

The unit of measurement is a *delegation*: the moment a supervisor hands a
subtask to a worker that starts with no context. For each delegation we record

  - what the supervisor said (the only signal a predictor gets to use),
  - every context request the worker subsequently made,
  - which of those requests actually mattered to the finished work.

The last one is the ground truth. It is derived after the fact from the final
diff and the worker's own trajectory, never from a judgement call at logging
time -- see groundtruth.py.

Field ordering is deliberate: `access_index` and `turn` let us reconstruct
"what was known at time T" without replaying the trace.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContextKind(StrEnum):
    """How a worker asked for context.

    These are the retrieval verbs a SWE agent actually uses. Keeping them
    distinct matters because they have very different costs: a symbol lookup is
    cheap and precise, a repo-wide grep is expensive and noisy, and a predictor
    that anticipates only the cheap ones has not earned anything.
    """

    FILE_READ = "file_read"
    GREP = "grep"
    SYMBOL_LOOKUP = "symbol_lookup"
    DIR_LIST = "dir_list"
    TEST_READ = "test_read"


class NeedLabel(StrEnum):
    """Why a context access counts as a genuine need.

    USED_IN_DIFF is the strongest signal and the one derivable mechanically.
    The others cover context that shaped the work without appearing in it --
    a worker that reads an interface to learn a call signature needed that
    file even though it never edited it.
    """

    USED_IN_DIFF = "used_in_diff"
    RESOLVED_SYMBOL = "resolved_symbol"
    READ_BEFORE_EDIT = "read_before_edit"
    TEST_CONTEXT = "test_context"
    UNUSED = "unused"


class ContextAccess(BaseModel):
    """One retrieval a worker performed, with when and what came back.

    `access_index` is per-delegation and monotonic. It is what makes the
    prediction task well-posed: a predictor standing at the delegation sees
    nothing, and is scored against the set of accesses with any label other
    than UNUSED.
    """

    access_index: int
    turn: int
    kind: ContextKind
    query: str
    resolved_paths: list[str] = Field(default_factory=list)
    result_tokens: int = 0
    hit: bool = True
    label: NeedLabel | None = None
    timestamp: str = Field(default_factory=_utcnow)


class ToolCall(BaseModel):
    """A non-retrieval action: running tests, applying an edit.

    Kept separate from ContextAccess because these are what the worker *does*,
    not what it needs to know. Conflating them inflates apparent need.
    """

    turn: int
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    output_excerpt: str = ""
    timestamp: str = Field(default_factory=_utcnow)


class WorkerTurn(BaseModel):
    """One reasoning step. Retained because abandoned attempts are signal.

    Public trace datasets typically keep only the successful path. A worker
    that explored three wrong files before the right one tells us something a
    cleaned trace cannot: that the delegation underdetermined the need.
    """

    turn: int
    thought: str = ""
    abandoned: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Delegation(BaseModel):
    """A supervisor->worker handoff. The prediction unit.

    `message` is the entire input a delegation-time predictor is allowed to
    see, alongside static repo state. Everything else in this model is outcome
    data used for scoring, and must never be fed to the predictor.
    """

    delegation_id: str
    parent_task_id: str
    worker_id: str
    message: str
    issued_at_turn: int
    repo_head: str

    accesses: list[ContextAccess] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    turns: list[WorkerTurn] = Field(default_factory=list)

    files_in_diff: list[str] = Field(default_factory=list)
    diff: str = ""
    completed: bool = False
    failure_reason: str | None = None

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def needed_paths(self) -> set[str]:
        """Ground-truth answer: paths the worker genuinely needed.

        A labelled access contributes only the paths that justify its label,
        not every path it happened to touch. A grep matching fifteen files
        earns USED_IN_DIFF because one of them is in the diff; harvesting all
        fifteen would credit a predictor for naming the fourteen irrelevant
        ones. Observed on an xarray delegation whose two-file fix produced 28
        "needed" paths, among them versioneer.py.

        Narrowing rule by label:
          USED_IN_DIFF     -> only the resolved paths that are in the diff
          RESOLVED_SYMBOL  -> only single-path accesses (a broad match teaches
                              no signature); enforced in groundtruth, so the
                              access already carries exactly one path
          TEST_CONTEXT     -> only the test paths involved
          everything else  -> its resolved paths as-is

        Empty until label_delegation() has run. An unlabelled delegation
        scoring 0.0 recall is a labelling bug, not a prediction failure.
        """
        diff_paths = {p.strip().lstrip("./") for p in self.files_in_diff}
        needed: set[str] = set()

        for access in self.accesses:
            if access.label is None or access.label is NeedLabel.UNUSED:
                continue
            paths = [p.strip().lstrip("./") for p in access.resolved_paths]

            if access.label is NeedLabel.USED_IN_DIFF:
                needed.update(p for p in paths if p in diff_paths)
            elif access.label is NeedLabel.TEST_CONTEXT:
                needed.update(
                    p for p in paths
                    if "test" in PurePosixPath(p).name or "/tests/" in f"/{p}"
                )
            else:
                needed.update(paths)

        return needed


class Trace(BaseModel):
    """One full task: a supervisor decomposing an issue into delegations."""

    trace_id: str
    repo: str
    base_commit: str
    task_prompt: str
    source: Literal["swebench", "synthetic", "live"] = "swebench"

    delegations: list[Delegation] = Field(default_factory=list)
    supervisor_plan: str = ""
    resolved: bool = False

    schema_version: str = "antbench-v1"
    created_at: str = Field(default_factory=_utcnow)

    @property
    def total_tokens(self) -> int:
        """Headline cost metric: what the whole task consumed.

        This is the number the benchmark reports against a prefetch-enabled
        run, so it deliberately counts supervisor and worker tokens alike.
        """
        return sum(
            d.total_prompt_tokens + d.total_completion_tokens for d in self.delegations
        )
