# The Proof of Concept: "The Exam"

Fourth document in the set. `WINNING_BACKLOG.md` lists 30 items. This one
answers a narrower question: **if you did that work, what single thing would
you build to prove it?**

The answer is not a demo of the pipeline. It is an *exam* — a fixed,
pre-registered, public set of questions where **some of the questions have no
honest answer**, run against your tool and the alternatives, published with
the raw transcripts and your own failures included.

---

## 1. Why an exam and not a demo

A demo shows the tool working on a case you chose. Every competitor has one,
and no reviewer believes any of them.

The positioning is *"not the tool that answers fastest, the tool whose answer
survives review."* You cannot demo that. You can only demonstrate it
adversarially — by being scored on cases designed to catch tools out, including
cases where the correct behaviour is to **decline**.

That reframing is the whole advantage. Every automated causal tool is
optimised to produce an answer. An exam that scores *"did you correctly refuse
to answer?"* is one nobody else can pass, and it is not a trick: it is the
single most important property of an inference tool used for decisions.

---

## 2. The PoC is a vertical slice, not new work

Everything below maps to existing backlog items. Nothing here is throwaway.

| Exam section | Backlog item | Status in repo |
|---|---|---|
| A — Traps | 11 (negative controls) | `m_bias_collider` fixture exists with exact truth |
| B — Unanswerable | **13 (abstention)** | autobounds partial-ID, unidentifiable capture exist |
| C — Contamination | 17 (Parrot Benchmark) | sign-flip script + analyzer + tests exist |
| D — Degradation | **2 (fail closed)** | the geolift bug is the case; fix is the demo |
| E — Calibration | 6 + 7 (coverage harness) | needs building; scope to 6 estimators |

Confirmed already present and directly reusable:

```
tests/synthetic_datasets/conftest.py   m_bias_collider, ihdp_synthetic,
                                        acic_synthetic, high_dim_sparse,
                                        survival_synthetic  (exact truth)
data/checks.py                          propensity_overlap, overlap_summary,
                                        continuous_positivity_check
data/flags.py                           positivity_violation
sensitivity/dashboard.py                negative_control panel, tipping point
scripts/parrot_signflip_lalonde.py      the contamination case
reporting/preregister.py                OSF + AsPredicted + TTE export
```

---

## 3. The exam

Sixteen questions across five sections. Deliberately small — a benchmark you
can defend beats a benchmark you can brag about.

### Section A — Traps (4 questions). *Is the identification honest?*

Datasets where the obvious analysis is confidently, catastrophically wrong.
All four are canonical from the literature, cited, **not invented for this
exam** — that matters, see §5.

1. **M-bias collider.** `U1 → X ← U2`, `T → X`, `X → Y`. Adjusting for `X`
   induces bias. Fixture already exists with exact truth.
2. **Descendant-of-treatment adjustment.** Conditioning on a post-treatment
   mediator, which attenuates or reverses the total effect.
3. **Positivity cliff.** A covariate region with near-zero treated density.
   Every ML estimator silently extrapolates into it. `overlap_summary` and
   `continuous_positivity_check` already detect this.
4. **Simpson's reversal.** Aggregate and stratified effects have opposite
   signs.

**Correct behaviour:** name the trap, exclude the offending adjustment set or
restrict to the overlap region, and report the right sign with a calibrated
interval.

### Section B — Unanswerable (6 questions). *The killer section.*

Questions **genuinely not identifiable** from the data supplied. There is no
correct point estimate. The only correct answers are a refusal or a bound.

5. Unmeasured confounding, no valid instrument available.
6. Unmeasured confounding, no front-door path available.
7. Target population not transportable from the sample.
8. Positivity violated for the requested contrast, not merely weak.
9. Treatment with no variation in the observed window.
10. A question that is causally malformed — the "treatment" is a
    post-outcome variable.

**Correct behaviour, scored on three axes:**

```
abstention rate        unanswerable questions correctly declined
diagnostic quality     did it name the missing piece? (instrument /
                       mediator / overlap / temporal order)
bound quality          when partial ID applies, are the Manski /
                       autobounds bounds correct and non-trivial?
```

### Section C — Contamination (2 questions). *Is it reasoning or reciting?*

11. **Sign-flipped Lalonde.** Looks like the memorised benchmark; treated
    outcomes negated. Script already exists.
12. **Non-canonical twin.** Same structure, unrecognisable column names.
    Confirms no degradation on unmemorised data.

**Correct behaviour:** neutral rationales before estimation; correct negative
sign after; no confident prior assertion of direction.

### Section D — Degradation (2 questions). *Does silence get reported?*

13. **The geolift case.** `pysyncon` deliberately absent. Today this returns
    a 95% CI of [−1.30, 1.99] against a true effect of 12.0, disclosed as a
    string in a `notes` list.
14. **R bridge absent** for a question that routes to an R-only estimator.

**Correct behaviour:** refuse, name the missing dependency, and do not emit an
interval. Section D is the geolift bug converted from an embarrassment into
the clearest single expression of the product thesis.

### Section E — Calibration (continuous). *Are the intervals real?*

Not questions but a sweep: **6 estimators only** — OLS, DML linear, DML
forest, entropy balancing, a DR meta-learner, synthetic control — across 8
regimes, 500 replications each. Report nominal versus empirical 95% coverage,
bias, RMSE, and mean CI width.

**Correct behaviour:** empirical coverage within Monte Carlo error of 95%. Any
estimator that misses is labelled experimental in the catalog and in the
report.

---

## 4. What you actually ship

Three artifacts. One command each.

**(a) The scorecard.** A single table. This is the thing people screenshot.

```
                          Felisha   Copilot   LLM+code   DoWhy-default
A  Traps  (4)                 4/4       ?/4        ?/4            ?/4
B  Unanswerable (6)
     correctly declined       ?/6       ?/6        ?/6            ?/6
     false answers            ?/6       ?/6        ?/6            ?/6
     named missing piece      ?/6         —          —              —
C  Contamination (2)          ?/2       ?/2        ?/2            ?/2
D  Degradation (2)            ?/2         —          —              —
E  Calibration          6 certified       —          —              —
   false-refusal rate         ?/8       ?/8        ?/8            ?/8
```

The `—` entries are not omissions. They are the point: several of these axes
**cannot be scored** for tools that have no abstention path, no degradation
ledger and no calibration data. That asymmetry is the finding.

**(b) The reproducible report.** One HTML document per question: the DAG, the
identification argument, the estimate or the refusal, the sensitivity panel,
the ground truth revealed at the bottom, and `run.lock.json` attached so any
reader can re-run it. You already generate all of this — the exam just points
it at cases with known answers.

**(c) The coverage chart.** One page: nominal versus empirical coverage for 6
estimators across 8 regimes. Competitors have no equivalent page, because
nobody has produced the numbers.

---

## 5. The five rules that make this a proof and not marketing

This section is the difference between a PoC that convinces methodologists
and one that gets dismissed in a paragraph.

**1. Pre-register the exam using your own exporter.** `reporting/preregister.py`
already emits OSF and AsPredicted formats. Publish the exam design, scoring
rubric and hypotheses *before* running anything. You cannot then move the
goalposts, and the tool pre-registering its own evaluation is a genuinely
elegant self-demonstration of the feature.

**2. Use only canonical traps, cited.** M-bias, collider stratification,
positivity, Simpson's — all textbook. If you invent the traps, the exam is
rigged and a reviewer will say so in one sentence. Standard exam, standard
literature, standard citations.

**3. Score false refusal, not just refusal.** A tool that declines everything
would win a naive abstention benchmark and be worthless. Sections A, C and E
contain **answerable** questions where refusing is *wrong*, and the scorecard
reports a false-refusal rate. Without this the benchmark is not credible and
you should expect to be told so.

**4. Publish your own failures.** A 100% scorecard is not believable. Target
roughly 85–90%, publish every miss with a linked ticket, and say what you
learned. **Publishing your own failures is the single most
credibility-generating move available to you**, and it is exactly what a tool
selling honesty should do.

**5. Run the baselines fairly, and publish the transcripts.** Same data, same
prompt, documented versions, no cherry-picked runs, raw logs for every tool
including the ones that did well. Hold half the exam private and versioned so
it cannot be gamed later — including by you.

---

## 6. The ten-minute live demo

The exam is the artifact. For a room, you need one case.

Use **the collider trap** — it is the most visceral, because the failure mode
is a *sign flip*, not a magnitude error.

```
Here is a dataset. Obvious question, obvious covariates.

  DoWhy default          ATE = +2.1  [1.4, 2.8]   "treatment helps"
  LLM + code interpreter ATE = +2.0  [1.3, 2.7]   "treatment helps"
  Causal-Copilot         ATE = +2.2  [1.5, 2.9]   "treatment helps"

  Felisha                REFUSED that adjustment set.
                         X is a collider on U1 → X ← U2 with T → X.
                         Re-identified without X:
                         ATE = -1.6  [-2.3, -0.9]  "treatment harms"

Ground truth: -1.5
```

Three tools confidently recommended the intervention. The truth is that it
harms. That slide is the entire pitch, and it is not a rhetorical trick — it
is a canonical textbook trap that automated tools genuinely walk into.

---

## 7. Scope discipline

The most likely failure mode is building too much. Explicitly **do not**, for
the PoC:

- Certify all 40+ estimators. **Six.**
- Build a 40-question benchmark. **Sixteen.**
- Run five baselines. **Three**, plus one naive control.
- Write the paper. That is backlog item 18, after the exam exists.
- Add a single estimator.

**Effort: 5–6 weeks.** Roughly: 1 week fail-closed (item 2, needed anyway),
1.5 weeks exam construction (most fixtures exist), 2 weeks the calibration
sweep for 6 estimators, 1 week baselines and transcripts, 0.5 weeks the report
and scorecard.

---

## 8. Honest risks

**The abstention rate might be bad.** You have the machinery, but it has never
been measured — the whole premise of this audit is that unmeasured claims tend
not to hold. If Section B comes back at 2/6, that is the most valuable thing
you could possibly learn, and far better learned privately before publication
than publicly after.

**Over-abstention is a real failure mode.** Rule 3 exists because "refuse
everything" is a degenerate strategy. If the false-refusal rate is high the
tool is safe and useless, which is its own problem.

**Baselines are moving targets.** Frontier models improve monthly. Version
everything, date the scorecard, and plan to re-run it — a benchmark that is
re-run is an asset, one that is published once is a snapshot.

**The traps may be too easy for frontier models.** Increasingly plausible, and
it is fine: Sections B and D are the differentiators, and those are
*architectural*. A model with a code interpreter cannot refuse on grounds of
non-identifiability if it has no identification engine to consult.

---

## 9. Why this is the right PoC

It proves the positioning rather than asserting it. It is adversarial, so
passing it means something. It is scored on axes competitors structurally
cannot be scored on, and it says so honestly rather than hiding the
asymmetry. It reuses machinery that already exists. It is a vertical slice of
work you need regardless. It produces the paper's results as a by-product. And
it publishes its own failures, which is the only move that makes a tool
selling honesty credible.

**One sentence, if it works:**

> On six questions where the honest answer was "not identifiable from this
> data," Felisha declined five and named the missing piece in four. The other
> three tools answered all six.
