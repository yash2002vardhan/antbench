# antbench — measuring anticipatory context in multi-agent SWE

A benchmark for a question no existing memory benchmark asks: **standing at
the moment one agent delegates to another, how much of what the worker will
need is already knowable?**

Every agent memory benchmark — LongMemEval, LoCoMo, MemBench — scores a system
on *given this query, retrieve the right thing*. That framing is reactive by
construction. This one scores *given this trajectory state, prepare what is
needed before the query arrives*.

## Why software engineering

"What did the agent need?" is a judgement call in most domains. In a
repository it is largely recoverable from evidence the work leaves behind: a
file in the final diff was needed, a symbol looked up and then called was
needed, a test read and then run was needed. That makes the ground truth
mechanical rather than annotated, which is what makes a small pilot credible.

## Pipeline

    scripts/collect.py     supervisor decomposes a SWE-bench issue into
                           subtasks; each worker executes one on a clean
                           checkout while every retrieval is logged
    scripts/evaluate.py    three predictors see only the delegation message
                           and the repo, and are scored against what workers
                           actually needed
    scripts/sweep_window.py  sensitivity of the ground truth to its one
                             tunable parameter

## Modules

| file | role |
|---|---|
| `schema.py` | trace format; separates what a predictor may see from what only scoring may see |
| `workspace.py` | the wiretap — every retrieval logged with turn, tokens, hit/miss |
| `groundtruth.py` | mechanical need-derivation, five labels, strongest first |
| `agents.py` | supervisor/worker on LangGraph; one turn = one model reply |
| `tasks.py` | SWE-bench Verified selection and checkout |
| `predictor.py` | random / lexical / LLM, in increasing cost |
| `scorer.py` | recall, precision, token saving at budgets 1/3/5/8 |

## The predictors, and why three

`random` is a floor: if nothing beats sampling files from the repo, the
delegation message carries no signal. `lexical` is IDF-weighted string
matching — free, microseconds, no model call. `llm` is one `gpt-5-mini` call
per delegation, run **offline as a measuring instrument, not as a proposed
serving component**.

The gap between `lexical` and `llm` is the finding. If it is small, prediction
is a string-matching problem and a production layer needs no model in the
serving path — which answers, with evidence, the central objection that
predicting could cost more than retrieving. If it is wide, prediction needs
reasoning and a serving layer must budget for it or predict selectively.

## Result (11 traces, 19 delegations)

| predictor | recall@1 | recall@3 | recall@8 | precision@1 | any_hit@8 | cost |
|---|---|---|---|---|---|---|
| random  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | free |
| lexical | 0.185 | 0.363 | **0.463** | 0.538 | 0.692 | free |
| llm     | 0.221 | 0.396 | 0.441 | 0.684 | 0.632 | one call/delegation |

Lexical is over 26 delegations; the llm row is over the 25 available when that
comparison last ran, before the corpus grew. Re-score both together before
citing the gap.

**Random recovers essentially nothing** — 0.012 recall@8, one lucky hit across
28 delegations, with roughly four needed files among 150-1200 candidates. The
gap to lexical is unambiguous: 20 wins, 0 losses, paired permutation
p < 0.0001. Reproduce with `scripts/report.py`.

**IDF-weighted string matching is not beaten by the LLM predictor**, despite
costing nothing. Per delegation lexical is better on 12, worse on 4, tied on 9;
mean paired difference +0.133 recall@8, paired permutation p = 0.071 over 25
delegations. The sample does not separate them at conventional significance, so
the claim is *at least as good, for free* -- which is what the design decision
turns on. It was not the expected direction.

The mechanism is visible in individual cases. On a pytest delegation the LLM
proposed `python.py`, `config/findpaths.py`, and `compat.py` — reasonable
homes for import logic — while the fix belonged in `pathlib.py`. The model
reasons about where behaviour *ought* to live; lexical matching reads what the
delegation actually *says*. Supervisors name the concepts they are delegating,
those names appear in the repository, and a model's prior about code
organisation competes with that signal rather than adding to it.

### What follows for a serving layer

Prediction here is a string-matching problem. A production prefetch layer
needs no model in the serving path: it needs an IDF index over paths and
definitions, which is a dictionary lookup. That answers, with measurement
rather than assertion, the standing objection that predicting could cost more
than retrieving — predicting is roughly three orders of magnitude cheaper.

The residual gap is specific rather than diffuse. Both pytest delegations fail
for one reason: the message says "the import path logic" and the file is
`src/_pytest/pathlib.py`, a name the message never contains. Neither predictor
bridges that. An import-graph expansion — seed with lexical hits, follow one
hop — would, and remains deterministic and free. That is the next experiment,
not a learned model: with 19 delegations there is no data to train one, and
the measurement says added reasoning subtracts.

## Corpus properties (predictor-independent)

Two facts about the traces bound what any predictive layer can achieve, before
a single prediction is made. `scripts/analyze.py` reports both.

**Need arrives throughout, not up front.** The share of accesses that turn out
to be genuine need is roughly constant across every decile of a delegation —
0.59 in the first 30%, 0.65 in the last 30%. A worker is still discovering
genuinely-needed context in its closing turns. Spawn-time injection is
therefore structurally capped: it cannot reach need that only becomes apparent
after twenty turns of exploration. This is the empirical case for the hybrid
design — inject where confidence is high, and serve the remainder from a warm
cache behind the normal retrieval interface, where a wrong guess costs compute
and never pollutes the context window.

**Waste concentrates by token, not by call.** `grep` misses most often (59% of
calls wasted) but cheaply; `file_read` misses rarely (15%) but expensively,
accounting for 96k of the corpus's wasted tokens against grep's 23k.
`dir_list` is wasted every time. `test_read` is never wasted — when a worker
reads a test file, it needed it. A layer optimising call-count waste would
target the wrong verb.

## Stated limitations

**Undercount.** The ground truth measures what a worker *used*, not what it
*would have used* had retrieval been free. A need the worker skipped because
searching was expensive leaves no trace. This biases measured need downward,
so any prediction ceiling reported here is conservative.

**Contamination.** SWE-bench Verified instances predate the models and may be
memorised. This biases the other way — toward better prediction — so the
headline is bracketed rather than point-estimated. Fresh repositories are the
fix and belong in the full benchmark.

**Scale.** A pilot. 30 tasks across 8 repositories, chosen for multi-file
patches and substantive problem statements, excluding django and sympy so two
codebases cannot dominate.

## Ground-truth parameter

`READ_BEFORE_EDIT_WINDOW` is the only tunable in the labelling rules. Measured
result: it does not matter. Sweeping 0 → unbounded moves need and waste by
exactly zero, because the label never fires — every successfully edited file
reaches the diff, where the stronger `USED_IN_DIFF` claims it first. Rerun
`sweep_window.py` after any labelling change to confirm the property holds.

## Running

    uv sync
    uv run pytest -q                                   # 35 tests
    uv run python scripts/collect.py --limit 1         # one task
    uv run python scripts/collect.py                   # full pilot
    uv run python scripts/evaluate.py --no-llm         # free predictors only
    uv run python scripts/evaluate.py                  # all three
    uv run python scripts/analyze.py                   # corpus properties

Requires an OpenAI key in `OPENAI_API_KEY` or `~/.openai_key`.
