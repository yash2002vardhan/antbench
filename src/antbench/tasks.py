"""Select and materialise SWE-bench Verified instances to trace against.

Task selection is a methodological choice, not plumbing, so the filters are
explicit and defensible:

  - **Multi-file patches only.** A single-file fix cannot be split into
    independent subtasks without inventing the split, and an invented split
    would make delegation-time prediction artificially easy. Only 71 of the
    500 Verified instances touch two or more files, which is itself worth
    reporting: most SWE-bench work is not naturally multi-agent.

  - **django and sympy excluded.** They dominate the set (231 and 75
    instances) and are heavy to clone and install. Letting them dominate the
    pilot would measure those two codebases rather than the phenomenon.

  - **Substantive problem statements.** A two-line issue gives a supervisor
    nothing to decompose, so the delegation would carry no signal either way.

Contamination is a real limitation here: these instances predate the models
and may be memorised. It biases toward *better* prediction, so any headline
number is an optimistic bound -- the mirror of the groundtruth undercount,
and stated for the same reason. Fresh repositories are the fix, and belong in
the full benchmark rather than the pilot.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi

HF_ROWS = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=princeton-nlp%2FSWE-bench_Verified"
    "&config=default&split=test&offset={offset}&length={length}"
)

EXCLUDED_REPOS = {"django/django", "sympy/sympy"}
MIN_PROBLEM_CHARS = 500
# Multi-file gold patches were the original filter: a single-file fix seemed
# unlikely to decompose into independent subtasks. The pilot showed the
# supervisor emits one subtask for most issues anyway (21 of 30 traces), so
# patch breadth turned out not to predict delegation structure. Relaxing this
# is how per-repo samples grow once the multi-file pool is exhausted.
MIN_PATCH_FILES = int(os.environ.get("ANTBENCH_MIN_PATCH_FILES", "2"))

_SSL = ssl.create_default_context(cafile=certifi.where())


@dataclass(frozen=True)
class Task:
    """One SWE-bench instance, reduced to what the harness needs.

    `gold_files` is held for analysis only -- comparing predicted context
    against the files the human fix actually touched. It must never reach the
    supervisor or a worker.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold_files: tuple[str, ...]
    difficulty: str

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.repo}.git"

    @property
    def slug(self) -> str:
        return self.repo.replace("/", "__")


def _patch_files(patch: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE))))


def fetch_instances(cache: Path | None = None) -> list[dict]:
    """All 500 Verified rows, cached to disk after the first fetch."""
    if cache and cache.is_file():
        return json.loads(cache.read_text())

    rows: list[dict] = []
    for offset in range(0, 500, 100):
        url = HF_ROWS.format(offset=offset, length=100)
        with urllib.request.urlopen(url, timeout=90, context=_SSL) as response:
            rows.extend(item["row"] for item in json.load(response)["rows"])

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
    return rows


def select_tasks(
    rows: list[dict],
    limit: int | None = None,
    repos: set[str] | None = None,
) -> list[Task]:
    """Apply the documented filters and return tasks in a stable order.

    Ordering is by repo then instance id -- deterministic, so a rerun with the
    same limit traces the same tasks and results stay comparable.
    """
    tasks: list[Task] = []
    for row in rows:
        if row["repo"] in EXCLUDED_REPOS:
            continue
        if repos and row["repo"] not in repos:
            continue
        if len(row["problem_statement"]) < MIN_PROBLEM_CHARS:
            continue
        files = _patch_files(row["patch"])
        if len(files) < MIN_PATCH_FILES:
            continue
        tasks.append(
            Task(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                problem_statement=row["problem_statement"],
                gold_files=files,
                difficulty=str(row.get("difficulty", "")),
            )
        )

    tasks.sort(key=lambda t: (t.repo, t.instance_id))
    return tasks[:limit] if limit else tasks


def ensure_checkout(task: Task, repos_dir: Path) -> Path:
    """Clone (once per repo) and hard-reset to the task's base commit.

    One checkout is reused across tasks in the same repo; the reset is what
    isolates them. Cloning per task would multiply 40-100 MB by every
    instance for no benefit.
    """
    repos_dir.mkdir(parents=True, exist_ok=True)
    checkout = repos_dir / task.slug

    if not (checkout / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--quiet", task.clone_url, str(checkout)],
            check=True, capture_output=True, timeout=1800,
        )

    subprocess.run(["git", "checkout", "--quiet", "--force", task.base_commit],
                   cwd=checkout, check=True, capture_output=True, timeout=300)
    subprocess.run(["git", "clean", "-qfd"], cwd=checkout,
                   capture_output=True, timeout=120)
    return checkout
