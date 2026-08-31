# The Backlog: Best-in-Industry, and Ahead of Causal-Copilot

Third document in the set. `PRODUCTION_ROADMAP.md` asks *"are the claims
true?"*; `CUTTING_EDGE_AND_COMPETITIVE.md` asks *"is the stack current and
where do we win?"*. This one is just the list, ordered, with effort.

**The key discovery that shapes it:** most of the machinery needed to beat the
top competitor is already written and simply unpublished and uncertified.

| Already built | Lines | What it is |
|---|---:|---|
| `estimators/causaltune_select.py` | 380 | Energy score + ERUPT Pareto selection, **no ground truth required** |
| `estimators/learned_router.py` | 392 | GBM trained on dispatch telemetry, augments the rule cascade |
| `audits/method_coverage.py` | 431 | (estimand × flag combo) → reachable estimators matrix |
| `scripts/run_parrot_test.py` | 394 | Parrot driver + sign-anticipation ratio |
| `tests/integration/test_parrot_analyzer.py` | — | Analyzer tests that run **without** Ollama |

Causal-Copilot's paper states its own principal limitation: *"the algorithm
selection mechanism, while sophisticated, still relies on heuristic rules
derived from theoretical properties and empirical data."*

You have two **empirical** selectors already implemented against that exact
weakness, and neither has ever been benchmarked or published. That is the
opening.

---

## Tier 0 — Credibility floor

Nothing below this line counts until these are true. Three weeks.

**1. Connect CI and pin the toolchain.** `git mv causalrag/.github .github`;
exact pins for `ruff`/`mypy`; commit `uv.lock`. Autofix the ~394 mechanical
lint findings, waive `N803`/`N806` (statistical notation) and `RUF001–003`
(citation dashes) with recorded reasons. Add the 18-library mypy override
block plus `pandas-stubs` (−156 errors), delete stale ignores (−32), ratchet
the rest. *Effort: 1 week. Unblocks: everything.*

**2. Fail closed on estimator degradation — the geolift bug.** An absent
`pysyncon` produced a 95% CI of [−1.30, 1.99] against a true effect of 12.0,
disclosed as a string in `notes`. Estimators must refuse rather than
substitute. Add a typed `DegradationRecord` that surfaces in the report, and
a `--strict` mode where any degradation is an error. Gate on internal
diagnostics too: an `rmspe_ratio` of 21 should refuse to emit a CI at all.
*Effort: 1 week. This is the single highest-alignment change in the list.*

**3. Fix the other 6 failures.** Four are missing-dep guards that should
`skipif` (3× `rpy2` in `test_q7_pin_adjustment_set.py`, 1× `duckdb`). One is
a stale model assertion — rewrite `test_hardware.py` against *tier
properties* (parameter band, quantization, VRAM fit), never model names, so
the map can be refreshed without breaking tests. One is the
`cluster_robust_se > naive_se` assertion, which is a common case rather than
a theorem — root-cause it. *Effort: 2 days.*

**4. Make it installable.** `LICENSE` (MIT, matching the declaration), fix
the `pyproject` URLs to `brennansk1/felisha`, reconcile `version` and
`Development Status` with reality, move the 1.8 MB of design PDFs out of the
root. Container images: Python-only, plus a full image with R 4.4 and the ~30
CRAN packages pinned via `renv.lock`. *Effort: 3 days. This is what unblocks
every other human.*

**5. Split the suite.** `pytest-xdist` + markers for a sub-5-minute PR gate;
heavy suites nightly; `pytest-timeout`; a global thread budget so
`n_jobs=-1` in 7 sites stops oversubscribing. Wire
`audits/end_to_end_flow.py` in as a required check — it was designed to be
the ship gate. *Effort: 3 days.*

---

## Tier 1 — What makes it best in industry

The certification matrix. This is the flagship, and nobody in this space
ships it. Six weeks.

**6. The coverage harness.** For each (estimator × DGP): 500+ replications
from known truth, asserting empirical 95%-CI coverage within Monte Carlo
error of nominal. Track bias, RMSE and CI width alongside. This replaces
`abs(point - true) < 0.20 * abs(true_ate)` with the claim the tool actually
makes. *Effort: 2 weeks. Highest-value single item in the document.*

**7. The DGP library.** Span both what each method assumes and where it
should break: confounding strength, positivity violations, non-linearity,
effect heterogeneity, treatment/outcome dtype crosses, `n` from 100 to 100k,
`p` from 5 to 5000, clustered and panel structures. *Effort: 1.5 weeks.*

**8. Upgrade `method_coverage.py` from routing to calibration.** It already
answers *"is there a route for this cell?"*. Extend each cell to carry
measured coverage, bias and RMSE from item 6. The audit becomes the
certification matrix rather than a separate build. *Effort: 1 week — and it
is why item 6 is cheaper than it looks.*

**9. Property-based invariants.** `hypothesis` is a declared dev dependency
used in exactly one place. Estimators have invariants that are trivial to
state and brutal to violate: scale equivariance (`Y × c` → `ATE × c`),
location invariance, null recovery at the nominal rate, permutation and
column-rename invariance, monotonicity of bias in confounding strength.
*Effort: 1 week.*

**10. Mutation testing on the statistical core.** With 20%-tolerance
assertions there is a real chance some tests cannot fail. `mutmut` over
`estimators/`, `sensitivity/` and `roadmap/` is the only thing that detects
that. *Effort: 4 days.*

**11. Negative controls as first-class tests.** Assert methods fail *loudly*
under their own violated assumptions. The M-bias test is the right shape —
generalise it. A tight CI under a positivity violation is a bug that no
current test can see. *Effort: 1 week.*

**12. Publish the matrix.** Versioned in `docs/`, regenerated nightly, with
coverage regression failing the build. Any estimator that cannot be certified
is labelled experimental **in the catalog and in the report**. *Effort: 3
days. Converts "40+ estimators" from a marketing number into an evidence
table.*

---

## Tier 2 — Beat Causal-Copilot specifically

Attack the axes where you are structurally stronger, not where you overlap.
Eight weeks, overlapping Tier 1.

**13. The abstention benchmark — the killer metric.**

Every automated causal tool always returns an answer. That is their shared
weakness, and it is unmeasured because nobody wants to be scored on it.

Felisha is architecturally able to say *"not identified, and here is the
missing piece"*: it has autobounds partial-ID fallback, unidentifiable-
proposal capture that feeds the next critic call, collider/descendant
filtering, and positivity checks. Build a benchmark of questions that are
**genuinely unidentifiable from the supplied data** — unmeasured confounding
with no instrument, no front-door path, positivity violated, target
population not transportable — and score:

```
abstention rate        fraction of unidentifiable questions correctly refused
false-answer rate      fraction answered with a point estimate anyway
diagnostic quality     did it name the missing piece correctly?
```

A tool that refuses 90% of the unanswerable questions while a competitor
answers 100% of them wins the only comparison that matters to a regulated
buyer. **Nobody publishes this number. You should define the metric.**
*Effort: 2 weeks.*

**14. Benchmark empirical selection against heuristic selection.** Their
stated limitation is heuristic rules. You have `causaltune_select.py`
(energy score + ERUPT, ground-truth-free) and `learned_router.py`
(telemetry-trained). Finish both, wire them into the cascade as the primary
signal with the rules as the floor, and publish head-to-head: *regret
against the oracle estimator choice*, measured on the Tier-1 DGP library.
The claim becomes *"we select by measured out-of-sample performance on your
data, not a rule table."* *Effort: 2 weeks. The code is already written.*

**15. Benchmark calibration, not point accuracy.** Causal-Copilot reports
point-estimate accuracy against baselines. Report coverage. It is the harder,
more honest axis, it is where a rigor-first tool wins, and item 6 already
produces the numbers. *Effort: 3 days on top of Tier 1.*

**16. Benchmark unmeasured-confounding robustness.** Their paper concedes
weakness under "complex unmeasured confounding." You have E-values with
scale routing, sensemakr, CCN OVB, Zhao Γ, Rosenbaum bounds, Manski
partial-ID, tipping-point auto-fire and negative-control falsification —
breadth nothing else has. Score: as unmeasured confounding is dialled up in
a known DGP, which tool's reported uncertainty widens honestly and which
stays falsely confident? *Effort: 1.5 weeks.*

**17. The Parrot Benchmark, public and versioned.** You are ~80% there: a
394-line driver, a sign-anticipation ratio, and analyzer tests that run
without Ollama. What is missing is the sweep and the publication. Score every
candidate model on parroting rate, DAG quality against expert-labelled
graphs, identification correctness and synthesis faithfulness. Publish a
leaderboard.

A tool that can state *"this model parrots memorised benchmarks 34% of the
time, this one 4%"* answers a question the whole field is asking and nobody
has. *Effort: 2 weeks. This is the paper that gets cited, and citation is how
tools get adopted in this domain.*

**18. Write the paper.** Items 13–17 are a publication: an automated causal
analysis system evaluated on calibration, abstention, confounding robustness
and model parroting rather than point accuracy. Causal-Copilot has arXiv
numbers and you have README assertions; that asymmetry is currently fatal to
credibility and this closes it. *Effort: 3 weeks.*

---

## Tier 3 — Highly useful

Adoption. Rigor nobody can reach is worth nothing. Nine weeks.

**19. MCP server.** The single highest-leverage usefulness change. MCP went
~2M → ~97M monthly SDK downloads by March 2026 with native support across
the major flagships; in 2026 the distribution channel for an analytical tool
is other agents. Today an analyst's agent that needs a defensible estimate
cannot call you — it writes its own `statsmodels` regression instead. Surface
`profile`, `propose_dag`, `identify`, `estimate`, `sensitivity`, `report`.
`identify()` returning *"not identifiable, here is the missing piece"* is a
tool call nothing else in the ecosystem offers. *Effort: 1.5 weeks — a thin
wrapper over existing code.*

**20. A stable Python API.** `__init__.py` is three lines exporting
`__version__`; every capability is CLI/TUI-only, so notebook users import
internals with no stability guarantee. Curate `Study`, `discover`,
`estimate`, `sensitivity`, `report` with semver. *Effort: 1 week.*

**21. Report faithfulness verification.** Automatically check that every
numeric claim in the executive synthesis traces to a record in the protocol.
This closes the last hallucination gap — the one in the text the
decision-maker actually reads — and it is a natural Layer 5 on the existing
guard stack. *Effort: 1.5 weeks.*

**22. Structured logging and tracing.** Replace 105 `print()` calls with
`structlog`, JSON output, a `run_id` on every record, OpenTelemetry spans
around LLM calls and estimator fits. Multi-hour autonomous runs become
debuggable. *Effort: 1 week.*

**23. Frontier models beside local, with routing.** `engines/base.py` is
already 80% there — a clean Protocol with an eager-failure contract. Add
hosted adapters, route by step difficulty (strongest model for DAG proposal
and synthesis; local and cheap for column-level utility calls), and cache the
large stable prefixes — the dataset context block and catalog markdown are
re-sent on every planner call, and `cache_prefix_key` already computes the
digest needed to verify reuse. Keep local as the default: it is a real
feature for data that cannot leave the building. *Effort: 1.5 weeks.*

**24. Durable execution.** `master_loop.py` is 2,291 lines plus ~820 in
`loop_observability/` — a hand-rolled reimplementation of retries,
checkpointing, budgets and crash forensics. Adopt **DBOS**: in-process
library, state in Postgres or SQLite, no new infrastructure, so the tool
stays `pip install`-able for a solo researcher. Temporal only if you outgrow
it. A six-hour run then survives a laptop sleep. *Effort: 2 weeks.*

**25. `causalrag verify <run.lock.json>`.** Restore the recorded
environment, re-run, diff the estimates. Reproducibility becomes an
executable assertion rather than a JSON file. This is the feature a clinical,
policy or audit user asks for first. *Effort: 1 week.*

**26. Redact credentials.** Inline DB credentials flow from `SQLConnector`
into `DatasetSpec.source` and the provenance manifest, both serialised
verbatim into files users are told to share as reproducibility evidence.
Strip URL userinfo at the boundary; read secrets from the environment; add a
test asserting no artifact can contain a password. *Effort: 2 days.*

**27. Documentation site.** MkDocs or Quarto, with the Tier-1 certification
data inline, worked examples per domain, and an explicit
assumptions-and-limitations page. *Effort: 1.5 weeks.*

**28. Narwhals at the connector boundary.** 65 modules import pandas;
migrating the core would risk the statistical engine for little gain. But the
connector and profiler layer is where users want to hand you a Polars frame
or an Arrow table without a `.to_pandas()` round-trip on 40 GB. *Effort: 1
week. Do not touch the statistical core.*

---

## Tier 4 — Category expansion

Only once Tiers 0–2 hold.

**29. Active experimental design.** The EIG machinery in `loop_scoring/`
already computes information gain over hypotheses. Turn it outward: *"run
this experiment next, at this sample size, and here is the expected
information gain."* `feasibility/power.py` already has the power × MDE grid.
This converts the tool from an analyser of existing data into a planner of
new data collection — a substantially larger market, and nothing in the
competitive set does it. *Effort: 4 weeks.*

**30. Team surface.** Immutable, addressable, diffable runs; "what changed
between run 41 and 42" is the second question every real user asks. Server
mode (FastAPI + job queue + artifact storage) with the TUI and CLI as clients
of the same API. *Effort: 4 weeks.*

---

## What not to do

- **Do not add estimators.** The 1610-item backlog is a trap. An uncertified
  60th is worth less than a certified 40th, and breadth is already not the
  differentiator.
- **Do not rewrite the dataframe layer.** Boundary only.
- **Do not chase a GUI.** The competitive surface is the API and the
  artifact. The TUI is already good enough.
- **Do not drop local inference.** Add frontier models beside it.
- **Do not compete on automation speed.** Causal-Copilot already automates
  selection, and frontier models with code interpreters improve monthly.
  That race is not winnable solo, and it is not the interesting one.

---

## If you only do four things

1. **Item 1** — connect CI. Until automation runs, every claim is an
   assertion.
2. **Item 2** — fail closed on degradation. The geolift CI excluding truth by
   6× is the exact defect a skeptical evaluator finds first, and fixing it
   *is* the positioning.
3. **Item 6 + 12** — the coverage harness and the published certification
   matrix. This is what makes the tool best in industry, and nobody else
   ships it.
4. **Item 13** — the abstention benchmark. Define the metric the whole
   category is avoiding, on the axis where you are architecturally unique.

The through-line: **not the tool that answers fastest, the tool whose answer
survives review.** Every item above either makes that claim true or makes it
checkable.
