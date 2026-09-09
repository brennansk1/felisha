# Upgrade Brief — Phase 1

**Repository:** `brennansk1/felisha` · package root `causalrag/`
**Branch:** `claude/beta-production-roadmap-b1u0gb`
**Audit baseline:** commit `058fac6` (measured 2026-08-31)
**Brief written:** 2026-09-09

This file is self-contained. Someone — or an agent — reading only this file
should be able to execute the work. It has three parts:

1. **Context** — what this project is and what the audit found.
2. **The research protocol** — ten research tasks, to be completed and
   written to disk *before* implementation code is written.
3. **The implementation plan** — what to build, in what order, with gates.

Companion documents already in `causalrag/docs/`, in reading order:
`PRODUCTION_ROADMAP.md` (the audit), `CUTTING_EDGE_AND_COMPETITIVE.md`
(stack currency + competitive position), `WINNING_BACKLOG.md` (30 items,
4 tiers), `PROOF_OF_CONCEPT.md` (the Exam spec). This brief is the
execution plan for the first slice of that backlog.

---

## Part 1 — Context

### 1.1 What the project is

`causalrag` (product name "Felisha") is an LLM-assisted causal-inference
pipeline implementing the Petersen–van der Laan Causal Roadmap end-to-end
(Q1–Q8). ~57,000 LOC of source across 350 Python modules, ~25,000 LOC of
tests. Key surfaces:

```
src/causalrag/
  core/            protocol, graph, estimand, flags, ledger, roles   (2,723)
  discovery/       profiler, investigator, expert, Markov boundary   (2,739)
  identify/        DoWhy + ananke reconcile, transportability        (1,660)
  estimators/      python/ + rbridge/ + catalog + routers            (1,532)
  sensitivity/     E-value, sensemakr, OVB, Rosenbaum, Manski        (3,720)
  roadmap/         q5_identify, q6_estimand, q7_estimate, q8         (1,468)
  tasks/           rca, impact, uplift, mmm, geolift                 (3,428)
  reporting/       synthesis, render_html, quarto, preregister       (3,518)
  llm/             engines/, cache, guards, honesty, dspy            (4,309)
  master_loop.py   autonomous K-experiment loop                      (2,291)
  audits/          end_to_end_flow, method_coverage, island_detector (1,618)
```

### 1.2 What is genuinely strong — do not break these

- **Four-layer hallucination guard** (`llm/guards.py`): prevention → schema
  → semantic → *statistical*. Layer 4 tests the LLM's implied conditional
  independencies against the data and surfaces disagreement without letting
  either side silently win. Most tools stop at Layer 2.
- **The Parrot Diagnostic** — sign-flipped Lalonde falsification harness
  detecting whether the model reasons over supplied data or recites a
  memorised benchmark. No known equivalent anywhere.
- **Audit-trail-first design** — decision ledger, identification narration,
  sensitivity verdicts as first-class serialised records.
- **`extra="forbid"`** on the Pydantic protocol models — corruption fails
  loudly rather than dropping fields.
- **Eager `EngineNotAvailable`** (`llm/engines/base.py`) — the engine layer
  refuses to silently fall back, with an explicit comment that silent
  degradation hides misconfiguration. That instinct, applied consistently,
  is the fix for the central defect below.
- **Correct HTML escaping** — all 83 interpolation sites in
  `reporting/render_html.py` route through `_e()`.
- **Preregistration export** (`reporting/preregister.py`) — OSF,
  AsPredicted, Hubbard NEJM TTE. Unusual to the point of unique.

### 1.3 Verified findings — the basis for this plan

All measured by running the project's own tooling. **Every number below is
dated 2026-08-31 and MUST be re-verified before use — see R1.**

| # | Finding | Evidence |
|---|---|---|
| F1 | **CI has never executed.** Workflow sits at `causalrag/.github/workflows/ci.yml`; GitHub Actions only discovers workflows at repo root `.github/workflows/`. Actions API returned `total_count: 0`. | Every "CI fails on regressions" / "audit GREEN" claim is unenforced |
| F2 | **Both static CI steps would fail today.** `ruff check src tests` → 997 findings. `mypy src/causalrag` → 426 errors in 106 of 194 files. | Gate fails before tests run |
| F3 | 997 ruff findings are mostly **version drift**, not rot. `ruff>=0.4` resolves to 0.15.8; rule *families* are selected so new rules auto-opt-in. `F401`/`F841` return **zero** hits in `src/`. | `N806` 156, `I001` 109, `RUF002` 99, `RUF015` 85, `UP037` 82, `RUF100` 68 |
| F4 | **156 of 426 mypy errors are 18 missing overrides.** pandas 66, sklearn 26, scipy 15, pyarrow 14, tigramite 7, statsmodels 5, rpy2 4, dowhy 3, causallearn 3, pysyncon 2, pymc_bart 2, pymc 2, arviz 2, sqlalchemy, sentence_transformers, sensemakr, ibis, duckdb. A further 32 are stale `unused-ignore`. | Config fix clears 37% |
| F5 | **7 unit tests fail.** `tests/unit`: 1161 passed, 7 failed, 35 skipped, 9m00s. Docs claim "1165+ passing". | See F6–F8 |
| F6 | 4 of those are **missing-dep guards that should skip**: 3× `rpy2` in `tests/unit/roadmap/test_q7_pin_adjustment_set.py`, 1× `duckdb` in `test_connectors_extended.py`. Three test a *core Roadmap step* and can only run with R installed. | `RBridgeError: rpy2 not installed` |
| F7 | 1 is a **stale model assertion**: `test_hardware.py::test_selector_returns_qwen3_14b_at_tier2` expects a 14B model, tier map returns `qwen3.5:9b`. The "May 2026 landscape" map rotted in 3 months. | Test against tier *properties*, never model names |
| F8 | **2 are statistical, and one is the central defect.** See §1.4. | |
| F9 | **`run.lock.json` records an environment nobody can rebuild.** No lockfile, no container. R bridge (~20 of 40+ advertised estimators) has zero CI coverage; `_r.py` documents its singleton session as not thread-safe. | `dowhy>=0.11` resolves to whatever shipped today |
| F10 | **237 `except Exception` handlers; 36 swallow silently.** estimators/ 67, llm/ 33, tasks/ 21, identify/ 15, sensitivity/ 13. | Silent substitution changes scientific interpretation |
| F11 | **105 `print()` calls; 9 modules import `logging`.** No structured logs, no run correlation ID, no tracing. | |
| F12 | **Credentials land in shareable artifacts.** Inline-credential SQLAlchemy URLs flow `SQLConnector` → `DatasetSpec.source` (`core/protocol.py:35`) → `ManifestBuilder.dataset_path` (`provenance/manifest.py:182`), both serialised verbatim into `study.causalrag.yaml` and `run.lock.json`. No redaction anywhere. | |
| F13 | **No `LICENSE` file** despite `license = { text = "MIT" }`. `version = "0.1.0a0"` and `Development Status :: 2 - Pre-Alpha` against a README claiming v1.0. `pyproject` URLs point at `brennanskelley/causalrag`; actual repo is `brennansk1/felisha`. No tags, no releases. 1.8 MB of design PDFs at repo root. | Blocks every legal review |
| F14 | **No coverage testing anywhere.** All statistical assertions are single-draw point-estimate tolerances, e.g. `assert abs(result.point_estimate - true_ate) < 0.20 * abs(true_ate)`. No test asserts CI coverage. | Largest correctness gap |
| F15 | Full suite (adding `tests/integration` + `tests/synthetic_datasets`) **exceeded 40 minutes** without completing — joblib `loky` workers alive 10–18 min each at ~40% CPU. `n_jobs=-1` in 7 sites with no global thread budget. No `pytest-xdist`, no `pytest-timeout`; `pytest-cov` installed with no threshold. | |

### 1.4 The central defect

This single test failure is the reason this plan is ordered the way it is.

```
tests/unit/tasks/test_geolift.py::test_run_geolift_2_se_recovery
AssertionError: true_abs=12.001 not in [-1.299, 1.991]
  notes=['pysyncon unavailable; using OLS-on-donors fallback']
  rmspe_ratio=21.15
```

An optional dependency was absent. The pipeline did what it was designed to
do — degraded gracefully to an OLS-on-donors fallback. That fallback
returned a **95% confidence interval of [−1.30, 1.99] against a true effect
of 12.0** — excluding truth by a factor of six, with the wrong sign on the
lower bound, and an `rmspe_ratio` of 21 indicating the fit was garbage. The
only disclosure was a string in a `notes` list.

**Graceful degradation is correct for a plot backend, a cache, a report
renderer. It is wrong for an estimator, because a confidently wrong number
is the one output a user cannot detect as broken.**

The second statistical failure is subtler and needs root-causing:

```
tests/unit/estimators/test_hierarchical.py
  assert diag["cluster_robust_se"] > diag["naive_se"]
  assert 0.14350162324283533 > 0.14682146547062208
```

The test encodes "clustering inflates the SE." That is the common case, not
a theorem — cluster-robust SEs can fall below naive ones with few clusters
or negative within-cluster correlation. Either the assertion is wrong or the
computation is. Only a coverage study distinguishes them.

### 1.5 Positioning — the constraint every decision answers to

> **Not the tool that answers fastest. The tool whose answer survives review.**

Automation is being commoditised: Causal-Copilot (arXiv 2504.13263) already
automates method selection across 20+ methods with published benchmarks, and
frontier models with code interpreters improve monthly. Competing on
automation speed is not winnable solo. What is not commoditised — and grows
scarcer as automated analysis grows abundant — is an answer that survives
scrutiny.

Every item in this plan either makes that claim true or makes it checkable.

---

## Part 2 — The research protocol

**Rule: no implementation code until the research task covering it is
complete and written to disk.** Each task below names its question, sources,
deliverable path and acceptance criteria. Deliverables go in
`causalrag/docs/research/` and are committed — they are reviewable
artifacts, not scratch notes.

Research is *not* optional preamble. Two of these tasks (R6, R7) determine
whether the flagship benchmark is valid at all; getting them wrong
invalidates the work that follows.

### R1 — Re-establish ground truth

**Question:** Are the F1–F15 numbers still accurate on today's dependency
resolution?

**Method.** Build a clean environment and re-run everything. Note the
container `pip` may be a `uv` shim that installs into isolated tool envs —
build an explicit venv:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e ".[dev,estimators]"
.venv/bin/python -c "import causalrag; print(causalrag.__file__)"

.venv/bin/ruff --version && .venv/bin/ruff check src tests --statistics
.venv/bin/python -m mypy src/causalrag 2>&1 | tail -3
.venv/bin/python -m mypy src/causalrag 2>/dev/null \
  | grep -oE "\[[a-z-]+\]$" | sort | uniq -c | sort -rn
.venv/bin/python -m mypy src/causalrag 2>/dev/null \
  | grep -E "import-untyped|import-not-found" \
  | grep -oE '"[a-zA-Z0-9_.]+"' | tr -d '"' | cut -d. -f1 | sort | uniq -c | sort -rn

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/unit -q
```

Confirm F1 independently via the Actions API (workflow runs for the repo
should still be `total_count: 0`), and confirm no `.github` exists at repo
root.

**Deliverable:** `docs/research/R1-ground-truth.md` — a table of every
F-number with today's measurement, and an explicit list of any that changed.

**Acceptance:** every F-number either confirmed with a fresh command output,
or corrected with the new value and an explanation.

### R2 — Toolchain pinning and rule waivers

**Question:** Which exact tool versions, and which ruff rules are genuinely
wrong for this domain versus genuinely worth fixing?

**Research.** For each of the top rule families in F3, decide fix-vs-waive
with a written reason. The prior from the audit: `N803`/`N806` should be
waived (statistical notation `X`, `Y`, `T`, `ATE`, `SE` is *correct* here and
lowercasing it would make the code worse) and `RUF001`–`RUF003` should be
waived (bibliographic en-dashes in citations like "Petersen–van der Laan").
Verify that prior rather than assuming it. Confirm the ~394 autofixable
count and inspect a sample of each autofix class before applying in bulk.

Separately assess whether to add `ty` (Astral's checker; beta as of Dec
2025, 10–60× faster, Astral use it in production, but typing-spec
conformance trails mypy and it checks unannotated bodies mypy skips, so a
mypy-clean repo surfaces new errors). Recommended prior: **mypy stays the
gate; add `ty` only as a pre-commit fast check, and only after mypy is
green.** With 426 errors outstanding, a second checker now is not the move.

**Deliverable:** `docs/research/R2-toolchain.md` — exact pins, the waiver
table with reasons, the mypy override block ready to paste, and a
recommendation on `ty` with justification.

**Acceptance:** every waiver has a stated reason a reviewer would accept;
the override list is derived from R1's measured library list, not this
brief's.

### R3 — CI architecture

**Question:** What is the correct workflow layout for a repo where the
package lives one directory down, and what should gate a PR?

**Research.** GitHub Actions workflow discovery (root-only `.github/workflows/`),
`paths` filters, `defaults.run.working-directory`, matrix strategy, `uv`
caching, required status checks, and whether a merge queue is warranted for
a single-maintainer repo (probably not — note the decision). Determine how
to split: a sub-5-minute PR gate (`pytest -m "not slow and not integration"`
with `pytest-xdist`) versus a nightly full suite. Research `pytest-timeout`
defaults and how to impose a **global thread budget** so the 7 `n_jobs=-1`
sites stop oversubscribing in a container (F15).

Also research the container story: a Python-only image and a full image with
R 4.4 + ~30 CRAN packages pinned via `renv.lock`. Determine whether the R
job runs on every PR or nightly only.

**Deliverable:** `docs/research/R3-ci.md` plus a draft `.github/workflows/`
tree (not yet moved into place).

**Acceptance:** the draft workflow, run locally with `act` or reasoned
through step by step, would pass on the current tree *after* R2's fixes.

### R4 — Fail-closed design (the central defect)

**Question:** What is the right typed model for "this estimate is degraded
or refused," and where does it need to apply?

**Research, in three parts.**

*(a) Enumerate and classify.* Produce a complete inventory of the 237
`except Exception` sites and the 36 silent swallows, each classified as:
**estimator-critical** (a wrong number can reach the user — must fail
closed), **presentation** (plot backend, cache, formatting — graceful
degradation is correct), or **infrastructure** (retry/connection). Do not
guess the classification; read each site.

*(b) Survey prior art.* How does mature statistical software signal "I could
not do this reliably"? Look at: R's `warning()`/`condition` system and how
`grf`, `MatchIt`, `WeightIt`, `did` use it; `statsmodels` convergence flags
and `mle_retvals`; scikit-learn's `ConvergenceWarning`; DoubleML's and
`econml`'s diagnostics. The question to answer: is the right primitive an
exception, a typed result variant, or a required-to-be-read diagnostic
field? Consider that `EstimationResult` already exists and carries
diagnostics — the design should extend it, not sit beside it.

*(c) Internal-diagnostic gates.* The geolift case had `rmspe_ratio=21`, a
loud internal signal that the fit was worthless. Research, per estimator
family, what diagnostic exists and what threshold means "do not emit an
interval." For synthetic control specifically, research the literature on
pre-treatment fit (RMSPE ratio) as a validity gate and what cut-off is
defensible and citable.

**Deliverable:** `docs/research/R4-fail-closed.md` — the classified
inventory as a table, the prior-art survey, a proposed `DegradationRecord`
schema, the `--strict` semantics, and a per-family diagnostic-gate table
with cited thresholds.

**Acceptance:** the design says exactly what happens to the geolift case,
and a reviewer can see why the threshold chosen is defensible rather than
arbitrary.

### R5 — Coverage harness methodology

**Question:** How do you correctly measure and assert CI coverage across a
matrix of estimators and regimes?

**Research.** Simulation-based calibration and coverage-study methodology:
Talts et al. on SBC; Morris, White & Crowther on designing simulation
studies for statistical methods (this is the canonical reference and should
drive the design); how to size replications for a target Monte Carlo error
on a coverage estimate; how to build a confidence interval *for the coverage
estimate itself* so "coverage is 93.1%" comes with its own uncertainty;
and how to handle multiplicity when testing 6 estimators × 8 regimes = 48
coverage claims simultaneously without either drowning in false alarms or
missing real miscalibration.

Then survey **how peer packages validate themselves**: does `grf` publish
coverage results? `econml`? `DoubleML`? The ACIC data challenges? Determine
whether a published certification matrix genuinely has no precedent (the
audit's claim) or whether prior art exists to conform to.

**Deliverable:** `docs/research/R5-coverage-methodology.md` — the design
with citations, replication count with its MC-error justification, the
coverage-CI construction, the multiplicity approach, and the peer-package
survey.

**Acceptance:** the replication count is derived from a target MC error, not
picked round. The multiplicity decision is explicit.

### R6 — DGP library design ⚠ high-stakes

**Question:** For each of the 6 PoC estimators, what DGPs exercise their
assumptions, and what DGPs break them — with exactly computable ground truth?

The 6: OLS, DML linear, DML forest, entropy balancing, a DR meta-learner,
synthetic control.

**Research.** For each estimator, enumerate its identifying and regularity
assumptions from the primary literature, then design a DGP that violates
each one *individually* while keeping the true effect analytically known.
Dimensions to span: confounding strength, positivity/overlap, nonlinearity,
effect heterogeneity, treatment and outcome dtypes, `n` from 100 to 100k,
`p` from 5 to 5000, clustered and panel structure.

**Reuse what exists.** `tests/synthetic_datasets/conftest.py` already
provides fixtures with exact truth: `m_bias_collider`, `ihdp_synthetic`,
`acic_synthetic`, `high_dim_sparse`, `survival_synthetic`,
`mixed_types_dataset`. Start from these rather than rebuilding.

**Why this is high-stakes:** an error here is silent and invalidates every
number in the certification matrix. This task and R7 are the two where the
cost of being wrong is highest — see §4.3 on delegating them.

**Deliverable:** `docs/research/R6-dgp-library.md` — per estimator, an
assumption table; per DGP, the generative equations, the closed-form true
ATE, the assumption it violates, and the citation. Plus a coverage map
showing every (estimator, assumption) pair is exercised.

**Acceptance:** every true effect is derived in closed form and verified
numerically at large `n` against the analytic value. Every DGP cites the
paper whose assumption it targets.

### R7 — Provable non-identifiability ⚠ high-stakes

**Question:** How do you construct questions that are *provably* not
identifiable, and prove it?

This underwrites the abstention benchmark — the flagship differentiator. If
a question turns out to be identifiable via a path not considered, the
benchmark is wrong and the central claim collapses.

**Research.** The completeness results for identification: Shpitser–Pearl's
ID algorithm and its completeness, the hedge criterion for
non-identifiability, Tian–Pearl, and how `ananke` and `Y0` implement these
(both are already integrated in `identify/`). Establish a *mechanical
procedure* for certifying that a given (graph, estimand) pair is
non-identifiable, rather than arguing it informally.

Then design the 6 questions: unmeasured confounding with no instrument;
unmeasured confounding with no front-door; non-transportable target
population (Bareinboim–Pearl transportability — already in
`identify/transportability.py`); positivity violated for the requested
contrast; no treatment variation in the window; a causally malformed
question where the "treatment" is post-outcome.

Also research the **scoring rubric**: what counts as "correctly named the
missing piece", and how to score partial-identification bounds (are the
Manski/autobounds bounds correct *and* non-trivial?).

**Deliverable:** `docs/research/R7-nonidentifiability.md` — the
certification procedure, the 6 questions with a machine-checkable
non-identifiability proof each, and the scoring rubric.

**Acceptance:** each question's non-identifiability is certified by running
the ID algorithm (via `ananke`/`Y0`), not by argument. Independently
re-derivable by a reviewer.

### R8 — Competitive baselines

**Question:** How do we run Causal-Copilot and the other baselines fairly?

**Research.** Get Causal-Copilot (arXiv 2504.13263) actually installed and
running. Document its version/commit, its input interface, its configuration
surface, and what it needs to be given a fair shot. Do the same for a
DoWhy-default baseline and an LLM-with-code-interpreter baseline. Determine
a prompt-parity protocol: same data, same question, documented versions, no
cherry-picked runs, raw transcripts retained for every tool.

Note honestly which scorecard axes **cannot** be scored for a tool with no
abstention path, no degradation ledger and no calibration data — that
asymmetry is a finding, and it must be presented as such rather than as a
zero.

**Deliverable:** `docs/research/R8-baselines.md` — install and run
instructions per baseline, pinned versions, the parity protocol, and the
scoreable-axis matrix.

**Acceptance:** each baseline demonstrably runs end-to-end on one shared
example, with the transcript saved.

### R9 — Prior art: do not reinvent a benchmark

**Question:** Should the Exam conform to, extend, or replace an existing
benchmark?

**Research.** The ACIC data challenges, IHDP, LaLonde/Dehejia–Wahba,
Cladder (causal reasoning across Pearl's ladder), CausalBench, and any 2026
successors. Determine whether an existing suite already covers Sections A
and E, so effort concentrates on Sections B and D — which are the novel
parts. Being a *recognised extension* of an accepted benchmark is far
stronger than inventing one wholesale.

**Deliverable:** `docs/research/R9-prior-art.md` — the survey, and an
explicit conform/extend/replace decision per Exam section with reasons.

**Acceptance:** a clear statement of what is novel here and what is
borrowed, with citations.

### R10 — Packaging, secrets, and supply chain

**Question:** What is needed to make this installable and safe to share?

**Research.** URL-userinfo redaction at the boundary for F12 (parse and strip
credentials before they reach `DatasetSpec.source` or `ManifestBuilder`);
secrets from environment or keyring; the `LICENSE`/version/URL corrections
in F13; `uv.lock`; SBOM (CycloneDX) and `osv-scanner`; PyPI Trusted
Publishing. Determine where the 1.8 MB of design PDFs should live instead.

**Deliverable:** `docs/research/R10-packaging.md`.

**Acceptance:** includes a concrete test design proving no artifact can
contain a password.

---

## Part 3 — Implementation plan

Execute only after the covering research task is committed. Each stage has a
gate that must pass before the next begins.

### Stage 0 — Credibility floor (~3 weeks) · needs R1–R3, R10

| Step | Work | Gate |
|---|---|---|
| 0.1 | `git mv causalrag/.github .github`; re-scope paths per R3 | A workflow run appears in the Actions API for the first time |
| 0.2 | Pin toolchain to R2's exact versions; commit `uv.lock` | `uv.lock` present; CI installs from it |
| 0.3 | Apply R2's autofixes and waivers | `ruff check src tests` exits 0 |
| 0.4 | Apply R2's mypy override block + `pandas-stubs`; delete stale ignores; add per-module exemptions for the residue with a CI check that the exemption list may only shrink | `mypy src/causalrag` exits 0 |
| 0.5 | **Fail-closed per R4** — `DegradationRecord`, `--strict`, diagnostic gates. Replace the 36 silent swallows first; narrow estimator-critical handlers to specific exception types | The geolift case refuses instead of emitting `[−1.30, 1.99]` |
| 0.6 | Fix F6 (4 `skipif` guards), F7 (rewrite `test_hardware.py` against tier *properties*), F8 (root-cause the cluster-robust SE assertion) | `pytest tests/unit` exits 0 |
| 0.7 | R10: `LICENSE`, `pyproject` URLs, version/status reconciliation, credential redaction, move the PDFs | A redaction test passes; metadata matches reality |
| 0.8 | Suite split per R3: xdist PR gate, `pytest-timeout`, global thread budget, nightly heavy suite; wire `audits/end_to_end_flow.py` in as a required check | PR gate under 5 minutes, green |

**Stage 0 exit:** a green CI badge on `main` that means something, and the
central defect closed.

### Stage 1 — The Exam (~5–6 weeks) · needs R5–R9

Build in this order; each section is independently shippable.

| Step | Work | Gate |
|---|---|---|
| 1.1 | **Section D** (2 questions) — degradation. Falls straight out of 0.5 | Both cases refuse and name the missing dependency |
| 1.2 | **Section E** — coverage harness per R5, DGP library per R6, **6 estimators only** | Empirical coverage within MC error for the estimators that pass; failures labelled experimental in the catalog *and* the report |
| 1.3 | Extend `audits/method_coverage.py` from routing to calibration — each cell carries measured coverage/bias/RMSE | The audit *is* the certification matrix |
| 1.4 | **Section A** (4 questions) — canonical traps per R9, starting from `m_bias_collider` | Correct sign recovered on all four, or an explicit refusal |
| 1.5 | **Section B** (6 questions) — abstention per R7 | Each question's non-identifiability certified by the ID algorithm |
| 1.6 | **Section C** (2 questions) — contamination, from the existing sign-flip script and analyzer | Neutral pre-estimation rationales; correct negative sign after |
| 1.7 | Baselines per R8; scorecard; per-question reproducible report with `run.lock.json` | Every baseline transcript saved |
| 1.8 | Hypothesis invariants (scale equivariance, location invariance, null recovery, permutation and rename invariance, monotonicity of bias in confounding) and `mutmut` over `estimators/`, `sensitivity/`, `roadmap/` | Mutation score recorded; assertions that cannot fail are fixed |

**Stage 1 exit:** the scorecard, the reproducible report, and the coverage
chart — with our own failures published.

### 3.1 The five credibility rules — non-negotiable

From `PROOF_OF_CONCEPT.md`. Two of these constrain us, not the competition,
and that is the point.

1. **Pre-register the Exam using our own exporter.** `reporting/preregister.py`
   already emits OSF and AsPredicted. Publish design, rubric and hypotheses
   *before* running anything.
2. **Only canonical traps, cited.** Invent them and the exam is rigged.
3. **Score false-refusal rate, not just refusal rate.** "Decline everything"
   would otherwise win our own benchmark. Sections A, C and E contain
   *answerable* questions where refusing is wrong.
4. **Publish our own failures.** Target ~85–90%, every miss with a linked
   ticket. A 100% scorecard is not believable.
5. **Run baselines fairly and publish transcripts.** Hold half the Exam
   private and versioned so it cannot be gamed later — including by us.

### 3.2 Do not

- **Do not add estimators.** The 1,610-item method-coverage backlog is a
  trap. An uncertified 60th is worth less than a certified 40th.
- **Do not rewrite the dataframe layer.** 65 modules import pandas. Narwhals
  at the *connector boundary* only, later, and never in the statistical core.
- **Do not chase a GUI.** The competitive surface is the API and the
  artifact. The TUI is already good enough.
- **Do not drop local inference.** It is a differentiator for data that
  cannot leave the building — and see §4.2 for a compliance reason.
- **Do not certify all 40+ estimators in Stage 1.** Six.
- **Do not write the paper yet.** After the Exam exists.

---

## Part 4 — Execution notes

### 4.1 Environment gotchas found the hard way

- `pip` in some containers is a `uv` shim installing into isolated tool
  envs — packages install "successfully" and are not importable. Build an
  explicit venv and invoke it by absolute path.
- Piping `pytest` through `tail` buffers all output until completion; a long
  run looks hung when it is fine. Write to a file and poll the file.
- `tests/unit` takes 9 minutes; the full suite exceeds 40. Set
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` when
  measuring, or the `loky` workers oversubscribe.
- `anthropic.com` may be egress-blocked; `platform.claude.com` docs are
  reachable and are the better source anyway.

### 4.2 Model routing for this work

Anthropic's guidance: start with **Opus 5** (`claude-opus-5`, $5/$25 per
MTok); use **Fable 5.1** (`claude-fable-5-1`, $10/$50, cache reads
$0.25/MTok, 1M context, batch $5/$25) for demanding reasoning and
long-horizon agentic work, or when Opus 5 at higher effort falls short.

**Worth Fable 5.1:** R6 (DGP design) and R7 (non-identifiability) — the two
high-stakes tasks where an error is silent and invalidates everything
downstream; adversarial review of the identification core (`identify/` +
`roadmap/q5_identify.py` + `core/graph.py` + `core/estimand.py` fit in one
1M-context pass); vision QA on rendered reports (dense charts, chip strips,
DAG plots); and orchestrating the multi-day certification sweep, where the
cheap cache reads and batch pricing matter.

**Not worth it:** Stage 0.1–0.4 and 0.6–0.7 are mechanical. Note Fable 5.1
tends to rewrite whole files for small edits — actively wrong for a
minimal-diff refactor. Use Opus 5 or smaller.

**If Fable/Mythos goes into the product** (frontier adapters,
`WINNING_BACKLOG.md` item 23), three hard constraints:

- **30-day data retention; not available under zero data retention unless
  expressly authorized; designated Covered Models.** For clinical/policy
  users — the stated wedge — this may be disqualifying. Route derived
  artifacts (a DAG, an estimand, summary statistics), never raw cohort data.
  This is a compliance reason to keep local-first as the default, not just a
  philosophical one.
- **Forced tool use is unsupported** — `tool_choice` of `any`/`tool` returns
  400. `guards.py` Layers 1–2 and `OllamaClient.parse` depend on schema
  enforcement, so a Fable adapter must use structured outputs or strict tool
  use with `tool_choice: auto`.
- **Refusals are a typed outcome** — HTTP 200 with `stop_reason: "refusal"`
  and a `stop_details` object naming the policy area. This fits the
  architecture well: add an `EngineRefused` sibling to the existing eager
  `EngineNotAvailable` contract, and wire it to the refusal channel already
  sketched in `llm/honesty.py`. Not billed if refused before output;
  permitted fallback targets are Opus 4.8 and Opus 5.

### 4.3 Git conventions for this work

Develop on `claude/beta-production-roadmap-b1u0gb`. Push with
`git push -u origin claude/beta-production-roadmap-b1u0gb`; on network
failure retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s). Do
not open a pull request unless explicitly asked. Commit research
deliverables separately from implementation so each is reviewable on its own.

### 4.4 The four things that matter most

If the plan has to be cut, keep these:

1. **Stage 0.1** — connect CI. Until automation runs, every quality claim in
   the repository is an assertion rather than a fact.
2. **Stage 0.5** — fail closed. The geolift CI excluding truth by 6× is the
   exact defect a skeptical evaluator finds first, and fixing it *is* the
   positioning.
3. **Stage 1.2 + 1.3** — the coverage harness and the published
   certification matrix. This is what makes the tool best in industry, and
   nobody else ships it.
4. **Stage 1.5** — the abstention benchmark. It defines the metric the whole
   category is avoiding, on the axis where this architecture is unique.

---

## Appendix — Reusable machinery already in the tree

Verified present. Start from these rather than building new.

| Path | Lines | What it gives you |
|---|---:|---|
| `estimators/causaltune_select.py` | 380 | Energy score + ERUPT Pareto selection, **no ground truth required** |
| `estimators/learned_router.py` | 392 | GBM trained on dispatch telemetry, augments the rule cascade |
| `audits/method_coverage.py` | 431 | (estimand × flag combo) → reachable estimators matrix — the certification matrix skeleton |
| `audits/end_to_end_flow.py` | 689 | Orphan flags, unreachable estimators, panels missing from the report |
| `scripts/run_parrot_test.py` | 394 | Parrot driver + sign-anticipation ratio |
| `tests/integration/test_parrot_analyzer.py` | — | Analyzer tests that run **without** Ollama |
| `tests/synthetic_datasets/conftest.py` | — | `m_bias_collider`, `ihdp_synthetic`, `acic_synthetic`, `high_dim_sparse`, `survival_synthetic`, `mixed_types_dataset` — all exact truth |
| `data/checks.py` | — | `propensity_overlap`, `overlap_summary`, `continuous_positivity_check` |
| `data/flags.py` | — | `positivity_violation` |
| `sensitivity/dashboard.py` | 676 | Negative-control panel, tipping point, E-value routing |
| `identify/` | 1,660 | DoWhy + ananke/Y0 reconcile, transportability, autobounds partial ID, c-component decomposition |
| `reporting/preregister.py` | — | OSF + AsPredicted + Hubbard NEJM TTE export — use it to pre-register the Exam |
| `llm/engines/base.py` | — | `InferenceEngine` Protocol with the eager-failure contract to extend |
| `loop_scoring/` | 1,079 | EIG, Thompson sampling, MCTS — for later active-design work |
| `feasibility/power.py` | — | Power × MDE grid — for later active-design work |

**Note on Causal-Copilot's stated limitation:** its paper concedes that
*"the algorithm selection mechanism, while sophisticated, still relies on
heuristic rules derived from theoretical properties and empirical data."*
Two **empirical** selectors already exist in this tree against exactly that
weakness, and neither has ever been benchmarked. That is the opening, and it
is finishing work rather than construction.
