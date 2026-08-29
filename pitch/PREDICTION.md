# Pre-registered prediction: sphinx

Recorded **before** any sphinx trace was scored, while collection was running.
The point is to test the repository-structure hypothesis as a prediction
rather than to construct an explanation after seeing the answer.

## Hypothesis

Lexical prediction works when a repository's module names share vocabulary
with the way its work is described. A proxy for that, computable from the repo
alone with no trace data: **definitions per indexed file**. A repository whose
files each define many named things offers many hooks for a delegation message
to match; one dominated by small fixture files offers almost none.

## Evidence at the time of writing

| repo | files | defs/file | uniq stems | observed recall@8 |
|---|---|---|---|---|
| xarray | 161 | 32.8 | 0.89 | 0.67 |
| pytest | 216 | 27.3 | 0.86 | 0.67 |
| seaborn | 145 | 16.3 | 0.92 | 0.50 |
| astropy | 776 | 14.4 | 0.71 | 0.56 |
| matplotlib | 883 | 10.5 | 0.95 | 0.64 |
| pylint | 1358 | 5.5 | 0.86 | 0.03 |
| **sphinx** | **547** | **11.1** | **0.61** | *unknown* |

Correlation between definitions per file and recall@8 over the six scored
repositories: **+0.69**.

Pylint is the outlier on both axes — 1358 files at 5.5 definitions each,
because 75% of its index is `tests/regrtest_data` and `tests/functional`
fixture files that exist to be linted rather than read.

## Prediction

Sphinx at 11.1 definitions per file sits with matplotlib (10.5), not with
pylint (5.5). **Predicted recall@8: 0.40–0.60.**

Falsification: sphinx scoring below 0.15 would break the hypothesis and mean
low predictability has a cause this proxy does not capture.

## Why this is worth recording

With n=6 repositories a +0.69 correlation is suggestive, not established, and
it would be easy to narrate whatever sphinx does as confirmation. Committing
to the interval first makes the seventh repository an actual test. If it
holds, the benchmark gains something more useful than another data point: a
cheap, trace-free way to tell in advance whether predictive prefetching will
pay off on a given codebase.


## Interim: scikit-learn (recorded before sphinx scored)

Scikit-learn's two delegations landed after the prediction above was written
and before any sphinx trace was scored, giving an unplanned first test.

At 10.1 definitions per file it sits just below matplotlib (10.5), so the
hypothesis put it in the healthy group. **Observed recall@8: 0.38** — the
lowest of the healthy repositories, but an order of magnitude above pylint's
0.03 and comfortably outside the falsification band.

With seven repositories the correlation is **+0.70**, essentially unchanged
from +0.69 at six. That is one confirmation on n=2 delegations, which is weak
evidence on its own; sphinx contributes seven and remains the real test.


## Outcome: PARTIAL — the interval was too narrow

Observed sphinx recall@8: **0.254**. Above the 0.15 falsification line, below
the predicted 0.40-0.60 band. The direction was right and the magnitude was
wrong.

The correlation itself strengthened, to **+0.71 across eight repositories** —
sphinx sitting mid-table on both axes is consistent with a linear relationship.
But that number now looks like the wrong thing to have emphasised. The three
repositories closest in density span most of the observed range:

| repo | defs/file | recall@8 |
|---|---|---|
| matplotlib | 10.5 | 0.64 |
| scikit-learn | 10.1 | 0.38 |
| sphinx | 11.1 | 0.25 |

Within one definition per file of each other, and recall varies by a factor of
2.5. A correlation of +0.71 over eight points is compatible with that much
scatter; what it does not support is the tight interval I committed to.

**Revised claim.** Definitions per file separates the extremes — pylint at 5.5
is genuinely different from xarray at 32.8 — and does not resolve the middle.
It is a screening signal for "is this repository pathological", not a
predictor of where in the 0.25-0.65 band a normal repository will land. Any
advance indicator worth shipping needs to explain why matplotlib and sphinx
diverge at equal density, and this proxy cannot.

**Why this is the more useful outcome.** A clean confirmation would have
licensed a claim the data does not support: that a single cheap repo statistic
tells you whether predictive prefetching will pay off. Being wrong in a
recorded, bounded way says something sharper — that the extremes are
predictable and the middle is not yet — and it names the open question a
larger corpus would answer. That question is now a fellowship work item rather
than an assumption buried in a pitch.


## Final corpus note (appended after collection completed)

The outcome above was scored at 41 delegations across 8 repositories, where
sphinx read **0.254**. The final corpus — 57 delegations, 9 repositories —
gives **0.246**. `PITCH.md` cites the final figure.

The verdict is unchanged: outside the predicted 0.40-0.60 band, above the 0.15
falsification line, PARTIAL. The structural correlation that motivated the
prediction also moved, from +0.71 at eight repositories to +0.65 at nine —
which is consistent with what the failed prediction already showed, that the
proxy separates extremes and does not resolve the middle.
