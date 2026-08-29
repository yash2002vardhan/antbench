"""Supervisor/worker agents that produce measurable traces.

The supervisor reads an issue and emits delegations; each worker executes one
delegation against a fresh view of the repo. Workers are deliberately started
cold -- no inherited context -- because that cold start is the thing the
benchmark measures. Warming it up is what a predictive layer would do, and we
cannot measure the benefit of doing so if the baseline already does it.

TURN DEFINITION (load-bearing, referenced by groundtruth.READ_BEFORE_EDIT):
one turn is one model reply. A reply requesting five tools is one turn, and
all five accesses share it. This matters because the ground-truth window
measures distance in turns; counting per-tool instead would silently widen
every window. Defined here, in one place, so the convention is auditable.
"""

from __future__ import annotations

import json
import os
import pathlib
import uuid
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from .groundtruth import label_delegation
from .schema import Delegation, WorkerTurn
from .workspace import Workspace

SUPERVISOR_MODEL = "gpt-5"
WORKER_MODEL = "gpt-5-mini"

# A worker that has not finished by here is recorded as incomplete rather than
# left to spin. Incomplete delegations still carry usable access traces, but
# their waste ratios do not mean anything: waste is measured against what the
# fix used, so a worker with no fix scores near-total waste by construction.
#
# Raised from 25 after the first pilot task, where all four delegations
# terminated exactly at the ceiling -- a limit that binds every time is
# measuring the harness, not the agent.
MAX_WORKER_TURNS = 60


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    path = pathlib.Path.home() / ".openai_key"
    if path.is_file():
        key = path.read_text().strip()
        os.environ["OPENAI_API_KEY"] = key
        return key
    raise RuntimeError("no OPENAI_API_KEY and no ~/.openai_key")


# -- supervisor ------------------------------------------------------------


class Subtask(BaseModel):
    """One unit of delegated work.

    `message` is the entire signal a delegation-time predictor gets. It is
    therefore the supervisor's job to write it as it naturally would -- not to
    over-specify file paths to make prediction easy, which would inflate the
    headline number into meaninglessness.
    """

    title: str = Field(description="Short imperative summary of the subtask.")
    message: str = Field(
        description=(
            "The full instruction handed to a worker who has never seen this "
            "repository. Describe the goal and the observable outcome. Do not "
            "enumerate file paths unless the issue text itself names them."
        )
    )


class Plan(BaseModel):
    approach: str = Field(description="One paragraph on how the issue is split.")
    subtasks: list[Subtask] = Field(description="Two to four independent subtasks.")


SUPERVISOR_PROMPT = """You are a senior engineer triaging an issue in a \
repository you know well.

Break the issue into 2-4 subtasks and delegate each to a separate engineer. \
Each engineer works independently and has NEVER seen this repository before; \
they receive only your message and must discover everything else themselves.

Every subtask MUST be:

  - a concrete change to source code, verifiable by reading or running the \
repository. Never assign investigation, reproduction, documentation, \
changelogs, release coordination, or review.
  - independently completable from a clean checkout. Workers run in parallel \
and cannot see each other's edits, so no subtask may depend on another \
having finished first. Split by area of the codebase, not by stage of work.
  - scoped so a competent engineer could finish it in under twenty tool calls.

If the issue is genuinely too small to split this way, emit a single subtask \
covering the whole fix rather than inventing artificial divisions.

Write each delegation the way you naturally would to a competent colleague: \
state the goal and how to tell it is done. Do not pad the message with file \
paths or code tours to make their job easier -- their discovery process is \
what we are studying."""


def build_supervisor(model: str = SUPERVISOR_MODEL) -> Any:
    return ChatOpenAI(model=model, api_key=load_api_key()).with_structured_output(Plan)


def plan_task(issue: str, repo_name: str, model: str = SUPERVISOR_MODEL) -> Plan:
    supervisor = build_supervisor(model)
    return supervisor.invoke(
        [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"Repository: {repo_name}\n\nIssue:\n{issue}"),
        ]
    )


# -- worker ----------------------------------------------------------------


WORKER_PROMPT = """You are an engineer fixing one subtask in an unfamiliar \
Python repository.

You have no prior knowledge of this codebase. Use the tools to explore it, \
then make the smallest change that accomplishes the subtask.

Work efficiently:

  - Prefer targeted lookups over broad searches.
  - Never re-read a span you have already seen. To see more of a file, call \
read_file again with start_line set past what you read.
  - You have a limited number of turns. Once you can identify the lines that \
need changing, make the edit -- do not keep exploring for certainty.
  - apply_edit needs an exact, unique snippet. If it reports no match, re-read \
that region to copy the text precisely rather than guessing again.

When the change is complete and you have run any relevant tests, reply with \
DONE and a one-line summary. If you become certain the subtask cannot be \
completed, reply with BLOCKED and the reason."""


class WorkerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    turn: int


def make_tools(ws: Workspace) -> list:
    """Bind workspace methods as LangChain tools.

    Every retrieval tool routes through the Workspace so it lands in the
    trace; nothing here touches the filesystem directly.
    """

    @tool
    def read_file(path: str, start_line: int = 1) -> str:
        """Read up to 400 numbered lines of a file, starting at start_line.

        If the result says more lines remain, call again with start_line set
        to the next line number rather than re-reading from the top.
        """
        return ws.read_file(path, start_line)

    @tool
    def grep(pattern: str) -> str:
        """Regex-search all Python files. Returns matching path:line entries."""
        return ws.grep(pattern)

    @tool
    def find_symbol(name: str) -> str:
        """Find where a function or class is defined."""
        return ws.find_symbol(name)

    @tool
    def list_dir(path: str = ".") -> str:
        """List the contents of a directory."""
        return ws.list_dir(path)

    @tool
    def apply_edit(path: str, old: str, new: str) -> str:
        """Replace an exact, unique snippet in a file with new text."""
        return ws.apply_edit(path, old, new)

    @tool
    def run_tests(path: str = "") -> str:
        """Run pytest, optionally on one path. Returns the output."""
        return ws.run_tests(path)

    return [read_file, grep, find_symbol, list_dir, apply_edit, run_tests]


def build_worker_graph(ws: Workspace, model: str = WORKER_MODEL) -> Any:
    """A react loop whose tool node is our instrumented workspace.

    Hand-rolled rather than prebuilt so the turn counter advances exactly once
    per model reply, matching the convention documented at module top.
    """
    tools = make_tools(ws)
    by_name = {t.name: t for t in tools}
    # Bounded per-request: one sphinx delegation hung for 13,801s before the
    # SDK's default timeout fired, against a corpus norm of 45-750s. A stuck
    # request is worth abandoning long before that.
    llm = ChatOpenAI(
        model=model, api_key=load_api_key(), timeout=180, max_retries=2
    ).bind_tools(tools)

    def call_model(state: WorkerState) -> dict:
        turn = state["turn"] + 1
        ws.turn = turn  # every access this reply triggers is stamped with it
        reply = llm.invoke(
            [SystemMessage(content=WORKER_PROMPT), *state["messages"]]
        )
        return {"messages": [reply], "turn": turn}

    def call_tools(state: WorkerState) -> dict:
        last = state["messages"][-1]
        out: list[ToolMessage] = []
        for call in getattr(last, "tool_calls", []) or []:
            impl = by_name.get(call["name"])
            if impl is None:
                content = f"ERROR: unknown tool {call['name']}"
            else:
                try:
                    content = impl.invoke(call["args"])
                except Exception as exc:  # a bad tool call must not kill a trace
                    content = f"ERROR: {type(exc).__name__}: {exc}"
            out.append(ToolMessage(content=str(content), tool_call_id=call["id"]))
        return {"messages": out}

    def route(state: WorkerState) -> str:
        last = state["messages"][-1]
        if state["turn"] >= MAX_WORKER_TURNS:
            return END
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(WorkerState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


def _usage(message: BaseMessage) -> tuple[int, int]:
    meta = getattr(message, "usage_metadata", None) or {}
    return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))


def run_delegation(
    ws: Workspace,
    subtask: Subtask,
    parent_task_id: str,
    repo_head: str,
    model: str = WORKER_MODEL,
) -> Delegation:
    """Execute one subtask on a clean repo and return its labelled-ready trace.

    The repo is reset first so each delegation's diff reflects only its own
    work -- otherwise later delegations inherit earlier edits and the
    diff-derived ground truth becomes meaningless.
    """
    ws.reset_repo()
    ws.reset_log()

    delegation = Delegation(
        delegation_id=str(uuid.uuid4())[:8],
        parent_task_id=parent_task_id,
        worker_id=f"w-{uuid.uuid4().hex[:6]}",
        message=subtask.message,
        issued_at_turn=0,
        repo_head=repo_head,
    )

    graph = build_worker_graph(ws, model)
    final_text = ""
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=subtask.message)], "turn": 0},
            {"recursion_limit": MAX_WORKER_TURNS * 2 + 10},
        )
        messages = result["messages"]
    except Exception as exc:
        # A mid-run failure loses the graph's message list, so turn and token
        # accounting would silently read zero while the workspace still holds
        # real accesses -- a delegation that looks scorable but contributes
        # nothing to cost totals. Recover what the workspace observed.
        delegation.failure_reason = f"{type(exc).__name__}: {exc}"
        messages = []
        if ws.accesses:
            delegation.turns = [
                WorkerTurn(turn=t, thought="[lost to mid-run failure]")
                for t in sorted({a.turn for a in ws.accesses})
            ]

    prompt_total = completion_total = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        p, c = _usage(message)
        prompt_total += p
        completion_total += c
        text = message.content if isinstance(message.content, str) else ""
        delegation.turns.append(
            WorkerTurn(
                turn=len(delegation.turns) + 1,
                thought=text[:2000],
                prompt_tokens=p,
                completion_tokens=c,
            )
        )
        if text:
            final_text = text

    delegation.accesses = list(ws.accesses)
    delegation.tool_calls = list(ws.tool_calls)
    delegation.total_prompt_tokens = prompt_total
    delegation.total_completion_tokens = completion_total

    diff, paths = ws.diff()
    delegation.diff = diff
    delegation.files_in_diff = paths
    delegation.completed = "DONE" in final_text.upper() and bool(paths)
    if not delegation.completed and delegation.failure_reason is None:
        delegation.failure_reason = (
            "blocked" if "BLOCKED" in final_text.upper() else "incomplete"
        )

    # Label before resetting: the symbol-payoff rule needs the checkout to see
    # which definitions each accessed file provides.
    label_delegation(delegation, repo_root=ws.root)

    ws.reset_repo()
    return delegation
