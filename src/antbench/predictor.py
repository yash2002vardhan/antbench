"""Predict, at delegation time, which files a worker will need.

The benchmark's central question: standing at the moment of delegation, with
only the supervisor's message and the repository as it exists, how much of
what the worker eventually needs is already determinable?

Every predictor here obeys one rule -- it sees the delegation message and the
repo, and never `accesses`, `tool_calls`, `diff`, or `files_in_diff`. That
separation is enforced by the call signature rather than by discipline: a
predictor is handed a message and an index, so there is no trace to leak from.

Three predictors, in increasing cost:

  Random     -- a floor. If a real predictor cannot beat sampling files from
                the repo, there is no signal in the delegation message.
  Lexical    -- identifiers mentioned in the message, matched against paths
                and definitions. No model call, microseconds, free.
  LLM        -- one model call reading the message and a repo skeleton.

The gap between Lexical and LLM is the result that matters, not the LLM's
absolute score, and the LLM is here as a measuring instrument rather than as a
proposed serving component. If the two score close, prediction is a
string-matching problem: a production layer needs no model in the serving
path, and the central cost objection to predictive prefetching -- that
predicting could cost more than retrieving -- is answered with evidence rather
than assertion. If the gap is wide, prediction genuinely requires reasoning,
and a serving layer has to budget for it or predict selectively.

Either way the experiment costs well under a dollar, run once, offline.
"""

from __future__ import annotations

import ast
import math
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .agents import load_api_key

PREDICTOR_MODEL = "gpt-5-mini"

# Files offered per prediction. Prefetching is not free -- a predictor allowed
# to name a hundred files would trivially hit high recall while preparing
# nothing useful, so the budget is what makes precision meaningful.
DEFAULT_BUDGET = 8

_SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".tox",
    ".mypy_cache", ".pytest_cache", "build", "dist", ".eggs", "doc", "docs",
}
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

# Prose words that survive IDF because they are rare as *definition names* yet
# common in English. Without this list a pylint delegation about config-file
# discovery scored on "add", "data", "file", "get", "run", "use" and matched
# 300+ files, while `config` -- the one discriminating word -- was discarded
# for appearing in over a third of the repository. IDF alone cannot separate
# these: it measures rarity in code, and the noise here is rarity in prose.
_STOPWORDS = frozenset("""
add all also and any are but can change check code current data default
does each emit ensure file files first for from get has have how instead
into its use used using make makes may more must new not now one only
other others out over pass path paths run runs same second set sets should
some state store temp test tests than that the their them then there these
they this those two update use user value values want way when where which
while will with within without work
""".split())
_DEFINITION = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)


@dataclass
class Prediction:
    """What a predictor offers to prefetch, best first.

    Order matters: the scorer reports precision at several budgets, so a
    predictor that ranks its confident guesses first scores better than one
    returning the same set shuffled.
    """

    paths: list[str]
    rationale: str = ""
    model_calls: int = 0

    def top(self, k: int) -> list[str]:
        return self.paths[:k]


@dataclass
class RepoIndex:
    """Static view of a repository: paths and the names each file defines.

    Built once per repo and reused across delegations. This is knowledge a
    real predictive layer would maintain continuously, so rebuilding it per
    prediction would overstate what prediction costs.
    """

    root: Path
    paths: list[str] = field(default_factory=list)
    definitions: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, root: Path, max_files: int = 4000) -> RepoIndex:
        index = cls(root=root)
        for file in sorted(root.rglob("*.py")):
            if any(part in _SKIP_DIRS for part in file.parts):
                continue
            rel = str(file.relative_to(root))
            index.paths.append(rel)
            try:
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            index.definitions[rel] = set(_DEFINITION.findall(text))
            if len(index.paths) >= max_files:
                break
        return index

    def skeleton(self, max_entries: int = 400) -> str:
        """A compact repo map for the LLM predictor.

        Directories with file counts and representative names. Sending every
        path would blow the context on a large repo and bury the structure the
        model actually reasons over.
        """
        by_dir: dict[str, list[str]] = {}
        for path in self.paths:
            by_dir.setdefault(str(Path(path).parent), []).append(Path(path).name)

        lines: list[str] = []
        for directory in sorted(by_dir):
            names = sorted(by_dir[directory])
            shown = ", ".join(names[:12])
            more = f", (+{len(names) - 12} more)" if len(names) > 12 else ""
            lines.append(f"{directory}/  [{len(names)}]  {shown}{more}")
            if len(lines) >= max_entries:
                lines.append("...")
                break
        return "\n".join(lines)


class Predictor(ABC):
    """Sees a delegation message and a repo index. Never a trace."""

    name: str = "base"

    @abstractmethod
    def predict(self, message: str, index: RepoIndex, budget: int) -> Prediction:
        ...


class RandomPredictor(Predictor):
    """Uniform sample of repo files -- the floor any real signal must clear."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def predict(self, message: str, index: RepoIndex, budget: int) -> Prediction:
        if not index.paths:
            return Prediction(paths=[], rationale="empty repo")
        picks = self._rng.sample(index.paths, min(budget, len(index.paths)))
        return Prediction(paths=picks, rationale="uniform random")


class LexicalPredictor(Predictor):
    """Score files by rare identifiers the delegation message mentions.

    Deliberately simple. It exists to answer whether the delegation message
    carries signal recoverable by string matching alone -- the baseline the
    expensive path has to beat to justify itself.

    Terms are weighted by inverse document frequency. Without it, a message
    carrying a hundred identifiers matches most of the repository and ranks
    whichever file defines the most common names, which in the first pilot
    delegation put generic modules above the file the fix actually touched.
    A term appearing in a third of the repo says nothing about where to look;
    one appearing in two files says a great deal.

    KNOWN FAILURE MODE, not a bug to tune away: this predictor cannot reach a
    file whose name reflects its architectural location rather than the
    behaviour under discussion. Two corpus examples, both scoring 0.00 --
    pytest's import-path fix lives in `src/_pytest/pathlib.py` while the
    message says "the import path logic"; pylint's XDG storage change lives in
    `pylint/config/option_manager_mixin.py` while the message talks about
    `XDG_DATA_HOME` and `~/.pylint.d` and never uses the word "config". No
    weighting scheme recovers a word the message does not contain. Closing
    this needs a different signal -- an import graph, or a symbol-to-module
    map -- not better term scoring.
    """

    name = "lexical"

    def predict(self, message: str, index: RepoIndex, budget: int) -> Prediction:
        terms = {
            t.lower()
            for t in _IDENTIFIER.findall(message)
            if t.lower() not in _STOPWORDS
        }
        if not terms:
            return Prediction(paths=[], rationale="no identifiers in message")

        weight = self._idf(terms, index)
        informative = {t for t in terms if weight[t] > 0}
        if not informative:
            return Prediction(paths=[], rationale="no informative terms")

        scores: dict[str, float] = {}
        for path in index.paths:
            parts = Path(path).parts
            stem = Path(path).stem.lower()
            score = 0.0

            # A term naming the file itself is the strongest lexical signal.
            if stem in informative:
                score += 4.0 * weight[stem]
            for part in {p.lower() for p in parts} & informative:
                score += 1.5 * weight[part]
            # Definitions weigh less than paths: many files define a common
            # name, few are named for one.
            for name in {d.lower() for d in index.definitions.get(path, ())} & informative:
                score += 2.0 * weight[name]
            # Body-text mentions are deliberately not indexed. Scoring every
            # identifier a file contains rewards large files for being large:
            # unnormalised it took one delegation's matches from 56 files to
            # 136 without surfacing a needed one, and normalising by sqrt(file
            # size) left recall@8 unchanged while dropping recall@1 from 0.172
            # to 0.129. Definitions and path parts carry the signal; body text
            # is noise, and indexing it cost a full read of every file.

            if score:
                # Mild preference for shallower files; deep helpers are rarely
                # what a delegation is centrally about.
                score -= 0.05 * len(parts)
                # Discount files that define almost nothing. A repository can
                # carry hundreds of tiny fixtures that exist to be processed
                # rather than read -- pylint's tests/regrtest_data and
                # tests/functional are 75% of its 1194 indexed files -- and a
                # two-function fixture otherwise matches a term exactly as
                # strongly as the 2000-line module that implements it.
                if len(index.definitions.get(path, ())) <= 2:
                    score *= 0.35
                scores[path] = score

        ranked = sorted(scores, key=lambda p: (-scores[p], p))
        return Prediction(
            paths=ranked[:budget],
            rationale=(
                f"{len(informative)}/{len(terms)} informative terms, "
                f"{len(scores)} files matched"
            ),
        )

    @staticmethod
    def _idf(terms: set[str], index: RepoIndex) -> dict[str, float]:
        """Inverse document frequency per term, zero for ubiquitous ones.

        A term in more than a third of files carries no locating information,
        so it is dropped outright rather than merely down-weighted.
        """
        total = max(len(index.paths), 1)
        document_frequency: dict[str, int] = dict.fromkeys(terms, 0)
        for path in index.paths:
            present = {p.lower() for p in Path(path).parts}
            present |= {d.lower() for d in index.definitions.get(path, ())}
            for term in terms & present:
                document_frequency[term] += 1

        weights: dict[str, float] = {}
        for term, freq in document_frequency.items():
            if freq == 0:
                weights[term] = 0.0
            else:
                # No hard ceiling. An earlier version zeroed any term in more
                # than a third of files, which discarded `config` on a pylint
                # delegation about config-file discovery -- the one word that
                # located the answer -- because the repo names many files after
                # it. Log-IDF already down-weights common terms smoothly;
                # cutting them off entirely throws away the signal that a
                # well-organised repository puts in its own filenames.
                weights[term] = math.log(total / freq)
        return weights


class _FilePrediction(BaseModel):
    paths: list[str] = Field(
        description=(
            "Repo-relative paths the engineer will need, most likely first. "
            "Only paths present in the structure shown."
        )
    )
    reasoning: str = Field(description="One or two sentences on the choice.")


LLM_PROMPT = """You predict which files an engineer will need to read.

An engineer who has never seen this repository is about to be given the task \
below. Before they start, you must prepare the files they will need -- they \
have not asked for anything yet, and you cannot observe what they do.

Name at most {budget} repo-relative paths, most likely first. Prefer files \
that must be edited or read to make the change over files that merely mention \
the topic. Only name paths present in the structure shown."""


class LLMPredictor(Predictor):
    """One model call over the delegation message and a repo skeleton.

    Run once per delegation, offline, to establish the ceiling. Not a proposal
    for what a serving layer should do -- see the module docstring.
    """

    name = "llm"

    def __init__(self, model: str = PREDICTOR_MODEL) -> None:
        self._model = model
        self._llm = ChatOpenAI(
            model=model, api_key=load_api_key()
        ).with_structured_output(_FilePrediction)

    def predict(self, message: str, index: RepoIndex, budget: int) -> Prediction:
        try:
            result = self._llm.invoke(
                [
                    SystemMessage(content=LLM_PROMPT.format(budget=budget)),
                    HumanMessage(
                        content=(
                            f"Repository structure:\n{index.skeleton()}\n\n"
                            f"Task given to the engineer:\n{message}"
                        )
                    ),
                ]
            )
        except Exception as exc:
            return Prediction(
                paths=[], rationale=f"failed: {type(exc).__name__}", model_calls=1
            )

        known = set(index.paths)
        # Drop paths that do not exist. A hallucinated path cannot be
        # prefetched, so counting it would credit the predictor for nothing.
        valid = [p for p in dict.fromkeys(result.paths) if p in known]
        return Prediction(
            paths=valid[:budget],
            rationale=result.reasoning[:300],
            model_calls=1,
        )


def build_predictors(include_llm: bool = True) -> list[Predictor]:
    predictors: list[Predictor] = [RandomPredictor(), LexicalPredictor()]
    if include_llm:
        predictors.append(LLMPredictor())
    return predictors


_IMPORT_CACHE: dict[Path, dict[str, set[str]]] = {}


def _module_name(path: str) -> str:
    """Dotted module name for a repo-relative path."""
    stem = path[:-3] if path.endswith(".py") else path
    parts = [p for p in stem.split("/") if p]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def build_import_graph(index: RepoIndex) -> dict[str, set[str]]:
    """Undirected neighbour map over repository import edges.

    Undirected on purpose. A delegation may name the caller and need the
    callee, or name a concept the callee implements and need the caller that
    wires it up; direction is not knowable from the message. Neighbours in
    either direction are candidates.

    Parsed with `ast` rather than regex so `from x import y` and relative
    imports resolve to the same module space as the file paths themselves.
    """
    cached = _IMPORT_CACHE.get(index.root)
    if cached is not None:
        return cached

    module_to_path = {_module_name(p): p for p in index.paths}
    edges: dict[str, set[str]] = {p: set() for p in index.paths}

    for path in index.paths:
        file = index.root / path
        try:
            tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            continue

        own_parts = _module_name(path).split(".")
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # Relative import: resolve against this file's package.
                    base = own_parts[: max(0, len(own_parts) - node.level)]
                    targets = [".".join(base + ([node.module] if node.module else []))]
                elif node.module:
                    targets = [node.module]

            for target in targets:
                # Walk up the dotted name so `a.b.c` also matches package `a.b`.
                parts = target.split(".")
                for cut in range(len(parts), 0, -1):
                    candidate = module_to_path.get(".".join(parts[:cut]))
                    if candidate and candidate != path:
                        edges[path].add(candidate)
                        edges[candidate].add(path)
                        break

    _IMPORT_CACHE[index.root] = edges
    return edges


class GraphExpandedPredictor(LexicalPredictor):
    """Lexical seeds, expanded one hop along import edges.

    Targets the corpus's single identified failure class: a delegation that
    describes behaviour while the file implementing it is named for its
    architectural location. Lexical matching cannot recover a word the message
    never contains -- but the file that *is* named usually imports the one that
    is not.

    Neighbours must *compete* with weak seeds rather than queue behind them.
    An earlier version appended expansion below the seed list and truncated to
    budget; since lexical already fills the budget on most delegations, every
    neighbour was discarded and the predictor was byte-identical to its parent
    on all 41 delegations. The seeds are therefore capped, leaving room the
    expansion can actually occupy.

    A neighbour is ranked by how many distinct seeds reach it and how strong
    those seeds were, on the reasoning that a file several named files all
    import is more likely to be the shared implementation than any single
    seed's private dependency.
    """

    name = "lexical+graph"

    # Fraction of the budget reserved for lexical seeds. The rest is contested
    # by import neighbours. At 1.0 this predictor degenerates to its parent.
    SEED_SHARE = 0.5

    def predict(self, message: str, index: RepoIndex, budget: int) -> Prediction:
        seeded = super().predict(message, index, budget)
        if not seeded.paths or budget <= 2:
            # Nothing to expand from, or too tight a budget for expansion to
            # be anything but dilution.
            return seeded

        keep = max(1, int(budget * self.SEED_SHARE))
        seeds = seeded.paths[:keep]
        graph = build_import_graph(index)
        seen = set(seeds)

        neighbour_score: dict[str, float] = {}
        for rank, seed in enumerate(seeds):
            weight = 1.0 / (1 + rank)  # earlier seeds vouch more strongly
            for neighbour in graph.get(seed, ()):
                if neighbour not in seen:
                    neighbour_score[neighbour] = (
                        neighbour_score.get(neighbour, 0.0) + weight
                    )

        expanded = sorted(neighbour_score, key=lambda p: (-neighbour_score[p], p))
        # Fill from expansion first, then fall back to the demoted seeds so no
        # budget is wasted when the graph is sparse.
        tail = expanded + [p for p in seeded.paths[keep:] if p not in seen]
        paths = (seeds + tail)[:budget]
        return Prediction(
            paths=paths,
            rationale=(
                f"{len(seeds)} seeds + {len(expanded)} import neighbours"
            ),
        )
