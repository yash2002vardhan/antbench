"""A repo checkout a worker can read and edit, with every access recorded.

This is the instrumentation layer. Workers never touch the filesystem
directly -- every retrieval goes through a Workspace method, which is what
makes the trace complete. A tool that reads a file without logging here is a
silently missing data point, so the rule is: if it returns repo content, it
lives in this class.

Two things are deliberately recorded that a normal agent harness would drop:

  - misses (`hit=False`), because a worker groping for a file that does not
    exist is evidence the delegation underdetermined the need;
  - result sizes in tokens, because the cost a predictive layer removes is
    measured in tokens, not in call counts.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from .schema import ContextAccess, ContextKind, ToolCall

# Cap on returned content. Real agent harnesses truncate; not truncating would
# make wasted-token figures reflect our harness rather than agent behaviour.
MAX_RESULT_CHARS = 12_000
MAX_GREP_HITS = 40
# Lines per read_file call. Roughly 4-8k tokens of source, so a worker can see
# a meaningful span without one read consuming its whole context.
MAX_READ_LINES = 400

_ENCODER = tiktoken.get_encoding("o200k_base")

_SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".tox",
    ".mypy_cache", ".pytest_cache", "build", "dist", ".eggs",
}

_TEST_PATH = re.compile(r"(^|/)tests?/|(^|/)test_[^/]+\.py$|_test\.py$")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text, disallowed_special=()))


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} chars omitted]"


@dataclass
class Workspace:
    """A repo at a known commit, plus the access log for the current worker.

    `accesses` and `tool_calls` accumulate for one delegation at a time;
    `reset_log()` is called between delegations so records attribute correctly.
    """

    root: Path
    turn: int = 0
    accesses: list[ContextAccess] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    edited_paths: set[str] = field(default_factory=set)

    def reset_log(self) -> None:
        self.turn = 0
        self.accesses.clear()
        self.tool_calls.clear()
        self.edited_paths.clear()

    # -- internals ---------------------------------------------------------

    def _record(
        self,
        kind: ContextKind,
        query: str,
        paths: list[str],
        text: str,
        hit: bool,
    ) -> None:
        self.accesses.append(
            ContextAccess(
                access_index=len(self.accesses),
                turn=self.turn,
                kind=kind,
                query=query,
                resolved_paths=paths,
                result_tokens=count_tokens(text),
                hit=hit,
            )
        )

    def _resolve(self, path: str) -> Path | None:
        """Resolve a repo-relative path, refusing anything outside the repo."""
        candidate = (self.root / path.strip().lstrip("/")).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return None
        return candidate

    def _rel(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root.resolve()))

    def _walk(self):
        for path in self.root.rglob("*.py"):
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            yield path

    # -- retrieval (all logged) -------------------------------------------

    def read_file(self, path: str, start_line: int = 1) -> str:
        """Read a file from `start_line`, numbered, capped at MAX_READ_LINES.

        Line-ranged rather than whole-file because whole-file reads on a large
        module truncate, and a truncated read teaches the worker nothing about
        how to see the rest. Observed in the first pilot task: a worker read
        one 3000-line module eleven times, never edited anything, and burned
        its entire turn budget. Real harnesses page; not paging made the
        harness itself the failure.
        """
        target = self._resolve(path)
        if target is None or not target.is_file():
            self._record(ContextKind.FILE_READ, path, [], "", hit=False)
            return f"ERROR: no such file: {path}"

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        rel = self._rel(target)
        kind = (
            ContextKind.TEST_READ if _TEST_PATH.search(rel) else ContextKind.FILE_READ
        )

        begin = max(1, start_line)
        window = lines[begin - 1 : begin - 1 + MAX_READ_LINES]
        if not window:
            body = f"(file has {len(lines)} lines; nothing at line {begin})"
            self._record(kind, path, [rel], body, hit=False)
            return body

        end = begin + len(window) - 1
        numbered = "\n".join(
            f"{n:6d}\t{line}" for n, line in enumerate(window, begin)
        )
        header = f"# {rel} lines {begin}-{end} of {len(lines)}"
        footer = (
            f"\n# ... {len(lines) - end} more lines; "
            f"call read_file(path, start_line={end + 1}) to continue"
            if end < len(lines)
            else ""
        )
        body = f"{header}\n{numbered}{footer}"
        self._record(kind, path, [rel], body, hit=True)
        return body

    def grep(self, pattern: str) -> str:
        """Search .py files for a regex. Misses are logged, not swallowed."""
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            self._record(ContextKind.GREP, pattern, [], "", hit=False)
            return f"ERROR: bad regex: {exc}"

        hits: list[str] = []
        paths: list[str] = []
        for file in self._walk():
            try:
                lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                if regex.search(line):
                    rel = self._rel(file)
                    if rel not in paths:
                        paths.append(rel)
                    hits.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= MAX_GREP_HITS:
                        break
            if len(hits) >= MAX_GREP_HITS:
                break

        body = "\n".join(hits) if hits else "(no matches)"
        self._record(ContextKind.GREP, pattern, paths, body, hit=bool(hits))
        return _truncate(body)

    def find_symbol(self, name: str) -> str:
        """Locate a def/class by name and return its signature line."""
        pattern = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b")
        found: list[str] = []
        paths: list[str] = []
        for file in self._walk():
            try:
                lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                if pattern.search(line):
                    rel = self._rel(file)
                    if rel not in paths:
                        paths.append(rel)
                    found.append(f"{rel}:{lineno}: {line.strip()}")

        body = "\n".join(found) if found else f"(symbol not found: {name})"
        self._record(ContextKind.SYMBOL_LOOKUP, name, paths, body, hit=bool(found))
        return _truncate(body)

    def list_dir(self, path: str = ".") -> str:
        target = self._resolve(path)
        if target is None or not target.is_dir():
            self._record(ContextKind.DIR_LIST, path, [], "", hit=False)
            return f"ERROR: no such directory: {path}"

        entries = sorted(
            f"{child.name}/" if child.is_dir() else child.name
            for child in target.iterdir()
            if child.name not in _SKIP_DIRS
        )
        body = "\n".join(entries) if entries else "(empty)"
        self._record(ContextKind.DIR_LIST, path, [self._rel(target)], body, hit=True)
        return _truncate(body)

    # -- actions (logged as tool calls, not retrievals) --------------------

    def apply_edit(self, path: str, old: str, new: str) -> str:
        """Exact-match replacement.

        Distinct from retrieval: this is doing, not knowing, and conflating
        the two would let the ground truth grade reads against themselves.
        """
        target = self._resolve(path)
        if target is None or not target.is_file():
            self._log_call("apply_edit", {"path": path}, False, "no such file")
            return f"ERROR: no such file: {path}"

        text = target.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            self._log_call("apply_edit", {"path": path}, False, "old text not found")
            return "ERROR: `old` text not found exactly; re-read the file."
        if text.count(old) > 1:
            self._log_call("apply_edit", {"path": path}, False, "ambiguous match")
            return "ERROR: `old` text is ambiguous; include more surrounding context."

        target.write_text(text.replace(old, new, 1), encoding="utf-8")
        rel = self._rel(target)
        self.edited_paths.add(rel)
        self._log_call("apply_edit", {"path": rel}, True, "ok")
        return f"Edited {rel}."

    def run_tests(self, path: str = "") -> str:
        args = ["python", "-m", "pytest", "-x", "-q", "--no-header"]
        if path:
            args.append(path)
        try:
            proc = subprocess.run(
                args, cwd=self.root, capture_output=True, text=True, timeout=300
            )
            output = _truncate((proc.stdout + proc.stderr)[-6000:], 6000)
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            output, ok = "TIMEOUT after 300s", False
        except OSError as exc:
            output, ok = f"ERROR: {exc}", False

        self._log_call("run_tests", {"path": path}, ok, output[:400])
        return output

    def _log_call(self, name: str, args: dict, ok: bool, excerpt: str) -> None:
        self.tool_calls.append(
            ToolCall(turn=self.turn, name=name, arguments=args, ok=ok,
                     output_excerpt=excerpt[:400])
        )

    # -- outcome -----------------------------------------------------------

    def diff(self) -> tuple[str, list[str]]:
        """Working-tree diff and the paths it touches, via git."""
        try:
            proc = subprocess.run(
                ["git", "diff", "--unified=3"],
                cwd=self.root, capture_output=True, text=True, timeout=60,
            )
            text = proc.stdout
        except (subprocess.SubprocessError, OSError):
            return "", sorted(self.edited_paths)

        paths = sorted(set(re.findall(r"^\+\+\+ b/(.+)$", text, re.MULTILINE)))
        return text, paths or sorted(self.edited_paths)

    def reset_repo(self) -> None:
        """Discard edits so the next delegation starts from the base commit."""
        subprocess.run(["git", "checkout", "--", "."], cwd=self.root,
                       capture_output=True, timeout=60)
        subprocess.run(["git", "clean", "-fd"], cwd=self.root,
                       capture_output=True, timeout=60)
