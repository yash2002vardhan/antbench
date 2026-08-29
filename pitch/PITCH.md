# Anticipatory Context

**A benchmark for predictive agent memory.**
Mercor Research Fellowship — APEX. Novel evaluation methodology, extending
APEX-SWE. Requesting the 6-month fellowship.

## The gap

Every agent memory benchmark I can find scores one shape: *given this query,
retrieve the right thing.* LongMemEval-V2 says so outright; LoCoMo, MemBench, PersonaMem and
Vectorize's AMB v1 inherit it. The framing is reactive by construction, and it
is not treated as a gap — Vectorize's March 2026 manifesto enumerating what
memory benchmarks miss does not mention anticipation.

Meanwhile three groups published positive anticipatory results this year:
*Predictive Prefetching for RAG*, *Remember When It Matters*, and *Anticipate
and Learn*. The capability is demonstrated. The measurement is thin.

## What it measures

At the moment a supervisor delegates to a worker that starts cold: **how much
of what that worker will need is already determinable from the delegation
message and the repository?**

The unit is a delegation, not a query. A predictor sees the message and static
repo state, never the trace — isolation enforced by the call signature.

Software engineering makes this tractable. "What did the agent need?" is a
judgement call in most domains; in a repository it is recoverable from evidence
the work leaves behind — a file in the final diff was needed, a symbol looked up
then called was needed, a test read then run was needed. Ground truth is
mechanical, not annotated, which is what lets a pilot this size mean anything.

## Prior work

I built the synthetic arm first. **Janus** (public since June 2026) grades
anticipation against an **exactly computed information-theoretic maximum** —
the provable best score any system could achieve, derived from the true
generating rules and enforced in CI by property tests: no baseline may beat
the oracle, the fast one-pass scorer must agree with the slow reference, seeds
must reproduce, controls must floor. No LLM judge anywhere. That guarantee is
the thing a naturalistic benchmark cannot have, and it is why both arms exist.

It now has five arms, three of them cross-boundary: **prefetch** (which state
comes next), **recall** (reactive retrieval, auto-graded), **serving** (stage
real text into a real LRU cache; graded by exact row-ID match against a fixed
retriever's top-k, so staging "close" content earns nothing), **opener**
(warm-up at session start, learned from that stream's own history), and
**handoff** (after an upstream agent finishes a ticket, is the downstream agent
primed before it asks — learned from behaviour, never declared).

**Handoff is the synthetic counterpart of the benchmark proposed here.** Both
measure whether context can be prepared across an agent boundary before the
receiving agent asks. Janus measures it on generated tickets with a ceiling;
this proposal measures it on real repositories without one.

What Janus also taught me is what the ceiling costs. Its §7.1 documents the
problem: in five of six prefetch workloads an order-2 count model reaches
89-92% of ceiling, because a generated token announces its own latent state.
Trivial methods substitute for memory and good systems tie with bad ones.
Fixing it required *engineering* a regime where counting provably fails — a
corridor of uninformative chores wide enough to defeat any fixed-order n-gram.

The headroom had to be built. On real delegations it is simply there — random
recovers 0.056, string matching 0.501, a nine-fold gap with nothing planted.
That is the argument for a naturalistic arm, and it comes from measurement
rather than assertion.

## Pilot results

41 tasks, 57 delegations, 9 repositories from SWE-bench Verified. Regenerated
in one pass by `scripts/finalize.py --with-llm`.

**Workers waste 34.6% of retrievals** — 350k tokens contributing nothing. Held
between 0.346 and 0.394 at every corpus size from 13 delegations to 57, across
nine repositories added one at a time. It is a property of how agents explore,
not of which repositories were sampled.

| predictor | recall@8 | precision@1 | any_hit@8 | cost |
|---|---|---|---|---|
| random | 0.056 | 0.018 | 0.193 | free |
| lexical (IDF) | 0.501 | 0.509 | 0.807 | free |
| LLM (gpt-5-mini) | 0.558 | **0.684** | **0.947** | one call each |

**Delegation messages carry real signal.** Lexical beats random on 45
delegations and loses on 3, p < 0.0001. Free string matching recovers half of
what a worker will need. This is the load-bearing result and it is unambiguous.

**Whether a frontier model earns its cost on recall is unresolved — and the
instability is itself a finding.** The paired difference has crossed zero twice
as the corpus grew:

| corpus | lexical − LLM | p |
|---|---|---|
| n=25 | +0.133 | 0.071 |
| n=41 | −0.004 | 0.945 |
| n=57 | −0.057 | 0.251 |

No corpus size separates them at 0.05, and the point estimate wanders. Anyone
reporting a single one of those rows would have published a confident claim
that the next fifteen delegations reversed. That is the sharpest argument in
this proposal for why this measurement needs the scale a fellowship buys.

**What has been stable is precision.** The LLM wins precision@1 at every corpus
size — 0.684 against ~0.51 — and surfaces something useful in 95% of
delegations against 81%. That maps onto a design split a serving layer can act
on: lexical for a warm cache, where a wrong guess costs only compute and
prediction must be free; the model for spawn-time injection, where a wrong
guess pollutes the context window and precision is what matters.

**Predictability is a property of the repository.** Per-repo recall@8 spans
50x — requests and xarray at 0.67, pylint at 0.01, separable at p < 0.0001.
Every total failure shares one shape: the message describes *behaviour* while
the file is named for its *architectural location* (`pathlib.py` for import
logic; `config/option_manager_mixin.py` for XDG storage). No term weighting
recovers a word the message never contains, and the LLM does not close this
either.

**Need arrives throughout the trajectory** — 0.62 in the first 30% of a
delegation, 0.69 in the last 30%. Spawn-time injection is structurally capped.

## Three results that went against me

**A pre-registered prediction, wrong.** Definitions per file correlated +0.71
with predictability across eight repositories, so I registered sphinx at
0.40-0.60 before scoring it. It landed at 0.246. On the final nine-repository
corpus the correlation itself fell to +0.65 — the proxy moved along with
everything else. It separates extremes and does not resolve the middle: three
repositories within one definition per file of each other span 0.25 to 0.64.
Recorded in `pitch/PREDICTION.md` before the result.

**A finding that reversed, twice.** At n=25 lexical led the LLM by +0.133 and I
nearly reported reasoning as actively harmful. By n=41 the gap was gone
(−0.004). By n=57 it had crossed to the LLM (−0.057). Three corpus sizes, three
different stories, none separated at 0.05. I have written each of them down and
been wrong about the first two.

**A proposed fix, disproved.** If a delegation names one file and needs
another, the named file usually imports it — so expand seeds one hop along
import edges. Built and measured: **−0.031 recall@8**, 3 wins, 10 losses. Dense
graphs make expansion imprecise (scikit-learn peaks at 203 neighbours); sparse
ones offer nothing but still cost a slot. One-hop adjacency is the wrong
instrument. `scripts/experiment_graph.py`.

## What the fellowship builds

1. **Explain the 50x spread.** A practitioner cannot yet tell in advance
   whether prefetching will pay off on their codebase. The most useful open
   question this pilot produced.
2. **Close the failure class.** Import expansion is ruled out by measurement,
   which narrows the search: next candidate is a symbol-to-module map,
   expanding only to files defining names the message uses.
3. **Measure the counterfactual.** Ground truth records what workers *used*,
   not what they would have used with free retrieval. Running workers with
   prepared context sizes that gap.
4. **Scale and de-contaminate.** Fresh repositories with post-cutoff commits.
   The n=25 to n=41 reversal is the argument for the corpus size this needs.
5. **Cross-validate handoff against delegation.** Janus's handoff arm and this
   benchmark measure the same phenomenon — context prepared across an agent
   boundary before the receiver asks — one with a ceiling on generated tickets,
   one on real repositories without one. Run the same predictor family on both.
   If % of ceiling predicts recall on real delegations, synthetic becomes a
   cheap proxy for expensive naturalistic evaluation. If it does not, that is a
   sharper warning about synthetic agent benchmarks than either arm could issue
   alone. It also closes gaps Janus states about itself: its handoff ceiling is Monte
   Carlo estimated rather than formally exact, and on `flow:deferred` its
   serving ceiling is beaten by a persistence null on chore-to-chore positions
   — root-caused to a small corpus giving the fixed retriever nowhere close to
   land, disclosed in `janus/serving.py`. Only the prefetch arm's ceiling is
   exact. Neither benchmark can run this experiment without the other.
6. **Release.** Dataset, harness, and an APEX leaderboard. A submission is a
   `Predictor`: it receives a delegation message and a repository index, returns
   ranked paths, and never sees the trace — the isolation is enforced by the
   call signature, so the obvious way to cheat is structurally unavailable.
   Scored on recall, precision, and token saving at four budgets.

**Why six months and not three.** Items 1-3 fit in three months: they are
analyses of a corpus that already exists. Item 5 does not — cross-validating
handoff against delegation needs both benchmarks instrumented to a common
predictor interface and a corpus large enough that the comparison does not
reverse again, and it is the one experiment neither project can run alone. The
longer fellowship is what buys that experiment.

## Limitations

**Undercount.** Measures what workers used, not what they would have used with
free retrieval. Biases measured need downward.

**Contamination.** SWE-bench instances may be memorised. Biases the other way.
The two bracket the headline.

**Scale.** 57 delegations across 9 repositories. Enough to establish the floor
result (p < 0.0001) and per-repo variation (p < 0.0001); not enough to resolve
close comparisons, as the lexical-vs-LLM difference crossing zero twice
demonstrates.

## On methodology

Six ground-truth bugs surfaced during this pilot. **Every one was caught by
reading traces, not by tests** — the tests passed throughout. A grep matching
fifteen files once produced 28 "needed" paths for a two-file fix, among them
`versioneer.py`; mean need was inflated 4x until an implausible number was
noticed by eye.

Three artefacts came out of that, and all generalise past this benchmark: an
automated implausibility audit over every delegation; a rule that labels derive
on read rather than persist, so ground truth cannot go stale when the rules
improve; and — after one delegation hung for 13,801 seconds and its exception
destroyed the framework's message list, leaving 43 real accesses with zero
token accounting — **instrumentation must survive the failure of the thing it
instruments**, and every artefact a predictor sees must be pinned to the state
the agent actually saw. That last one was silently wrong for 22 of 30 traces
before it was caught.

## Why me

I build production agent systems. This question came from watching subagents
spawn cold and re-derive context the orchestrator already had, with no way to
tell whether the waste was 5% or 50%. It is 34.6%, and I know that because I
built the instrument.

Janus is public and green in CI; this benchmark's harness, traces, and
analysis scripts are release-ready and ship with the fellowship. Both
reproduce their own numbers from their own artefacts. What I want from the
fellowship is what I cannot get alone: scale, expert annotation, and people who
will tell me where the measurement is wrong. Six ground-truth
bugs were caught by reading traces one at a time — a method that does not
survive a corpus ten times larger.

Not a stepping stone to model building. The part I find absorbing is where a
pre-registered prediction comes back wrong and the question becomes *why the
proxy failed*, not how to make the number larger.
