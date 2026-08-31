# Cutting Edge & Competitive Position

Companion to `PRODUCTION_ROADMAP.md`. That document asks *"are the existing
claims true?"* This one asks two different questions:

1. **Is the engineering stack current as of now (August 2026)?**
2. **What would make this competitive in the landscape it actually ships into?**

---

## Part 0 — The finding that reframes both questions

From the unit run: `tests/unit` completes in 9 minutes with **1161 passed,
7 failed, 35 skipped**. One of those failures is the most important line in
this whole audit.

```
tests/unit/tasks/test_geolift.py::test_run_geolift_2_se_recovery
AssertionError: true_abs=12.001 not in [-1.299, 1.991]
  notes=['pysyncon unavailable; using OLS-on-donors fallback']
  rmspe_ratio=21.15
```

An optional dependency was missing. The pipeline degraded gracefully, as
designed. The fallback produced a 95% CI of **[-1.30, 1.99] against a true
effect of 12.0** — off by 6×, wrong sign on the lower bound, with an RMSPE
ratio of 21 screaming that the fit was garbage. The only trace in the output
was a string in a `notes` list.

**This is not a bug to fix and move past. It is the product thesis, inverted.**

A tool whose entire pitch is "every number defensible" cannot silently
substitute a method that returns a confidently wrong interval. Graceful
degradation is right for a plot backend, a cache, a report renderer. It is
wrong for an estimator, because a confidently wrong number is the one output
a user cannot detect as broken.

Everything below is downstream of that. The competitive answer and the
engineering answer turn out to be the same answer.

---

## Part 1 — Is the stack current?

Measured against what a well-run Python analytics project looks like in
August 2026.

| Dimension | Felisha today | Current baseline | Gap |
|---|---|---|---|
| Resolver / lockfile | `pip`, `>=` ranges, no lock | `uv` + committed `uv.lock` | **Real** |
| Type checker | `mypy` strict, 426 errors | `ty` (Astral) for the dev loop; mypy as gate | Moderate |
| Python | `requires-python >=3.11` | 3.13/3.14; free-threading supported per PEP 779 | Watch |
| Dataframes | `pandas` in 65 modules, 67 import sites | Narwhals at boundaries; Arrow-native interiors | **Real** |
| Parallelism | `joblib`, `n_jobs=-1` in 7 sites | Explicit thread budgets; no nested oversubscription | **Real** |
| Property-based tests | `hypothesis` declared, used in **1** place | PBT for invariants; SBC for estimators | **Real** |
| Mutation testing | none | `mutmut` on the statistical core | **Real** |
| Observability | 105 `print()`, 9 modules logging, 0 tracing | OTel + GenAI semantic conventions, structlog | **Real** |
| Agent orchestration | `master_loop.py`, 2291 hand-rolled LOC | Durable execution (DBOS / Temporal) | **Real** |
| LLM providers | Ollama / llama.cpp / vLLM / MLX — all local | Local *and* frontier, with routing | **Real** |
| LLM evaluation | Parrot test as a manual document | Evals in CI, scored, versioned | **Real** |
| Supply chain | nothing | SBOM, `osv-scanner`, sigstore, Trusted Publishing | **Real** |
| Agent-callable API | none (CLI/TUI only) | **MCP server** | **Real** |

### 1.1 The four that actually matter

Most of that table is hygiene. Four items change what the tool *is*.

**(a) An MCP server. The single highest-leverage change available.**

MCP went from ~2M monthly SDK downloads at launch (Nov 2024) to ~97M by
March 2026, with native support across Anthropic, OpenAI, Google and
Microsoft flagship models. In 2026 the distribution channel for an
analytical tool is *other agents*.

Right now, an analyst's agent that needs a defensible causal estimate cannot
call Felisha. It will instead write its own `statsmodels` regression and
report a naive coefficient with no identification argument, no sensitivity
analysis, and no positivity check. That is the competitive threat, and an MCP
server is the direct answer to it.

The tool surface writes itself from the existing Roadmap steps:

```
causal.profile(data)            → DataFlags, roles, missingness, positivity
causal.propose_dag(data, ?q)    → candidate DAGs + CI audit + conflicts
causal.identify(dag, estimand)  → strategy, adjustment set, or a reason it fails
causal.estimate(...)            → point, SE, CI, method card, degradation ledger
causal.sensitivity(estimate)    → E-value, OVB, Γ, tipping point, verdict
causal.report(study)            → the full defensible document
```

That is a small wrapper over code that already exists, and it turns Felisha
from an application into infrastructure. `identify()` returning *"not
identifiable, and here is the missing piece"* is a genuinely valuable tool
call that nothing else in the ecosystem offers.

**(b) Durable execution instead of a hand-rolled loop.**

`master_loop.py` is 2291 lines, supported by ~820 more in
`loop_observability/` (`circuit_breaker.py`, `budget.py`, `postmortem.py`).
That is a bespoke reimplementation of retries, checkpointing, budgets and
crash forensics — the exact feature set of a durable execution engine.

Runs are multi-hour, LLM-driven, and currently die with the process. The 2026
pattern is to journal each step so the workflow resumes exactly where it
stopped, with LLM calls wrapped as journaled activities that are never
re-executed on replay (they are non-deterministic, so replay must return the
cached result).

Recommendation: **DBOS first.** It runs in-process as a library, persisting
execution state to Postgres or SQLite, with no new infrastructure — which
matters enormously for a tool that must stay `pip install`-able for a solo
researcher. Reach for Temporal only if you outgrow it. The payoff is
concrete: a 6-hour `--experiments 20` run survives a laptop sleep, and the
postmortem record becomes a query instead of a bespoke file format.

**(c) A frontier-model path, and evals that prove which model to use.**

The local-only stance was a defensible philosophical choice in 2024. In 2026
it is a quality ceiling on precisely the steps where reasoning quality
dominates: DAG proposal, identification strategy, domain inference,
executive synthesis. Competitors use frontier models.

The fix is not "switch to the cloud" — local inference is a real feature for
sensitive data, and should stay the default. The fix is to make the engine
layer genuinely model-agnostic (it is already 80% there: `engines/base.py`
defines a clean `InferenceEngine` Protocol with an eager-failure contract),
add hosted adapters behind it, and then **let evidence choose per step**:

- Route by step difficulty. Synthesis and DAG proposal get the strongest
  model available; column-level utility calls stay local and cheap.
- Use prompt caching on the large stable prefixes — the dataset context
  block and catalog markdown are re-sent on every planner call and are
  ideal cache targets. The existing `cache_prefix_key` already computes the
  digest needed to verify prefix reuse.
- Batch the embarrassingly parallel calls (per-column investigator passes)
  rather than looping serially.

**(d) Property-based testing and simulation-based calibration.**

`hypothesis` is a declared dev dependency used in exactly one place. For a
statistics library this is the largest available testing win, because
estimators have real invariants that are trivial to state and brutal to
violate:

- Scale equivariance: multiply `Y` by `c` → the ATE scales by `c`.
- Location invariance: add a constant to `Y` → the ATE is unchanged.
- Null recovery: a DGP with zero true effect → CI contains 0 at the nominal
  rate.
- Permutation: shuffling row order must not change the estimate.
- Column renaming must not change the estimate.
- Monotonicity: more confounding, held otherwise fixed → larger bias in the
  unadjusted estimate.

Add **mutation testing** (`mutmut`) over the statistical core. Given that
the current assertions use 20% tolerance bands, there is a real chance some
tests do not constrain their code at all — mutation testing is the only
thing that detects an assertion which cannot fail.

### 1.2 The ones to be deliberate about

**Free-threaded Python: track, do not adopt yet.** Free-threading is
officially supported as of 3.14 (PEP 779, Phase II) and is no longer
experimental, but the scientific ecosystem is still catching up and the
free-threaded build carries roughly 6–9% single-thread overhead. The real
win here is not the GIL — it is that `n_jobs=-1` appears in 7 sites with no
global thread budget, so nested parallelism oversubscribes CPU (visible in
the 40-minute suite, where four `loky` workers each ran at ~40% CPU). Fix
the oversubscription now with an explicit budget; revisit 3.14t when
NumPy/scikit-learn/pandas free-threaded wheels are boring.

**Pandas → Narwhals at the boundary, not a rewrite.** 65 modules import
pandas. A migration would be enormous and would risk the statistical core
for little gain. But the *connector and profiler* layer is where users want
to hand you a Polars frame, a DuckDB relation, or an Arrow table without a
`.to_pandas()` round-trip on a 40 GB dataset. Narwhals there is a contained,
high-value change.

**`ty` for the inner loop, `mypy` for the gate.** ty is 10–60× faster and
Astral use it in production, but it is still beta, its typing-spec
conformance trails mypy, and it checks unannotated function bodies that mypy
skips — so a mypy-clean codebase surfaces new errors on its first ty run.
With 426 mypy errors outstanding, adding a second checker now is not the
move. Get mypy green via the ratchet in Phase 0, then add ty as the fast
pre-commit check.

---

## Part 2 — Competitive position

### 2.1 The landscape

**Substrate, not competition.** PyWhy (DoWhy, EconML, causal-learn, led by
Microsoft Research) is what Felisha is *built on* — roughly 24K GitHub stars
across the family. Also in this tier: Salesforce CausalAI, AWS's use of
DoWhy for root-cause analysis, the Databricks causal accelerator, CausalNex,
py-tetrad from CMU. These are libraries. Felisha is an orchestration layer
above them, and that is the correct position.

**The direct competitor.** *Causal-Copilot* (arXiv 2504.13263) is an
autonomous LLM agent that automates the full causal pipeline — discovery,
inference, algorithm selection, hyperparameter optimisation, interpretation
— across 20+ methods for tabular and time-series data, with published
benchmark results against baselines. This is the same shape as `causalrag
auto`. Felisha has more methods (40+ vs 20+) and considerably more
methodological rigor. Causal-Copilot has an arXiv paper with numbers.

**The fast-moving research frontier.** *Causal Ensemble Agent* (arXiv
2606.10607, 2026) pools statistical discovery experts and uses an LLM as a
meta-referee to reweight them near the decision boundary — which overlaps
directly with Felisha's Markov-boundary triangulation and DAG-conflict
reporting. There is also a growing body of work on LLM agents for confounder
discovery and subgroup analysis. This area is being actively commoditised.

**Commercial.** causaLens (DecisionOS) sells causal AI into retail, CPG,
pricing and supply chain. Enterprise buyers in this space want governance and
auditability, not novel estimators.

**Benchmarks matter here.** Cladder and similar suites now exist to test
whether LLMs can reason causally across Pearl's ladder. In this field,
credibility is established by appearing in a benchmark table.

### 2.2 Where Felisha genuinely wins

These are real, and none of them is "we have more estimators."

1. **The Roadmap discipline is the moat.** Nobody else structures output as
   a typed Q1–Q8 walk where every estimate carries its identifiability
   proof, sensitivity verdict, anomaly audit and refutation results in one
   serialisable record. Causal-Copilot *selects an algorithm*. Felisha
   produces *an argument*. Those are different products, and only one of
   them survives a journal reviewer or a regulator.

2. **Sensitivity breadth is unmatched.** E-value with scale routing,
   sensemakr, Chernozhukov–Cinelli–Newey OVB, Zhao Γ for matched designs,
   Rosenbaum bounds, Manski partial-ID, tipping-point auto-fire,
   negative-control falsification, always-valid CIs, multiple-testing
   adjustment. Competitors report point estimates. Felisha reports *how
   wrong the assumptions would have to be to overturn the finding.* That is
   the question decision-makers actually ask.

3. **The Parrot Diagnostic has no equivalent anywhere.** A sign-flipped
   Lalonde harness that detects whether a model is reasoning over supplied
   data or reciting a memorised benchmark. Everyone in this field is quietly
   worried about exactly this contamination problem. Nobody has shipped a
   test for it.

4. **Preregistration export** (OSF / AsPredicted / Hubbard NEJM TTE) is
   unusual to the point of being unique, and it is the wedge into clinical
   and policy users — the buyers with the most acute need and the least
   tolerance for a black box.

5. **Local-first inference** is a genuine differentiator for regulated data
   that cannot leave the building. Keep it as the default, not the ceiling.

### 2.3 Where Felisha loses today

1. **Not agent-callable.** No MCP server, no Python API. In 2026 this is the
   distribution problem, and it is the one that compounds.
2. **No published numbers.** Competitors have benchmark tables; Felisha has
   README assertions — several of which, per the companion document, are not
   currently true. In this field that asymmetry is fatal to credibility.
3. **Reasoning ceiling** from local-only models on the steps where reasoning
   quality dominates.
4. **Installation friction.** R 4.4 plus ~30 CRAN packages installed by hand
   gates ~20 of the 40+ advertised estimators. Competitors are one
   `pip install`.
5. **Single-user.** CLI and TUI only; no hosted or collaborative surface, no
   way for a team to review a run.
6. **The trust bug is live.** The geolift failure above is exactly the defect
   a skeptical evaluator would find first, and it undermines the core claim
   directly.

### 2.4 The strategic call

**Automation is being commoditised. Defensibility is not.**

Causal-Copilot already automates method selection. Frontier models with code
interpreters get better at ad-hoc causal analysis every month. Competing on
*"we automate causal analysis"* means competing against that curve with a
solo-maintained codebase. That race is not winnable and it is not the
interesting one.

What is *not* commoditised — and is getting scarcer as automated analysis
gets more abundant — is the ability to hand a regulator, a journal reviewer,
or a CFO a document that survives scrutiny. Abundant cheap analysis makes
*verified* analysis more valuable, not less.

So the position is:

> **Not the tool that answers fastest. The tool whose answer survives review.**

Four moves follow directly, in order:

1. **Fix the trust bug properly, and make the fix the feature.** Estimators
   fail closed by default; every degradation becomes a typed record that
   surfaces in the report; `--strict` for publication-bound work. A `notes`
   string is not a disclosure. This is the single change most aligned with
   the positioning, and it is a live defect regardless.

2. **Publish the certification matrix** (Phase 1 of the companion roadmap).
   Measured coverage, bias and RMSE for every estimator across regimes,
   regenerated nightly. This converts "40+ estimators" from a marketing
   number into an evidence table, and it is the artifact that makes the
   defensibility claim checkable rather than asserted. Nobody in this space
   ships this.

3. **Ship the MCP server.** Become the thing other agents call when they
   need an estimate that will hold up. Small wrapper, large strategic
   effect.

4. **Turn the Parrot Diagnostic into a public benchmark.** Version it, score
   every candidate model on parroting rate, DAG quality against
   expert-labelled graphs, identification correctness, and synthesis
   faithfulness. Publish the leaderboard.

   A tool that can state *"this model parrots memorised benchmarks 34% of
   the time, this one 4%"* owns a question the entire field is asking and
   nobody has answered. That is the paper that gets cited, and citation is
   how tools get adopted in this domain.

### 2.5 What not to do

- **Do not add estimators.** The 1610-item backlog is real and it is a trap.
  An uncertified 60th estimator is worth less than a certified 40th, and
  breadth is already not the differentiator.
- **Do not rewrite the dataframe layer.** Narwhals at the boundary; leave
  the statistical core alone.
- **Do not chase a UI.** The competitive surface is the API and the
  artifact, not the terminal. The TUI is already good enough.
- **Do not drop local inference.** It is a differentiator for the buyers who
  care most. Add frontier models beside it, not instead of it.

---

## Sequencing against the companion roadmap

| When | From `PRODUCTION_ROADMAP.md` | Added here |
|---|---|---|
| Weeks 1–2 | Phase 0 — make CI real | Fix the geolift trust bug; guard the 4 dep-gated tests; `uv.lock` |
| Weeks 3–7 | Phase 1 — coverage harness | Hypothesis invariants + mutation testing; publish the certification matrix |
| Weeks 5–8 | Phase 2 — reproducibility | Containers incl. R; SBOM + `osv-scanner` + Trusted Publishing |
| Weeks 7–10 | Phase 3 — loud failure | Typed degradation ledger + `--strict`; OTel tracing |
| Weeks 9–14 | Phase 4 — product surface | **MCP server** and the stable Python API, together |
| Ongoing | Phase 5 — differentiators | Durable execution; frontier adapters + routing; the Parrot benchmark |

The two documents converge on the same first move. Phase 0 makes the
existing claims true; fixing the geolift fallback makes the central claim
*defensible*. Neither is optional, and both fit in a fortnight.
