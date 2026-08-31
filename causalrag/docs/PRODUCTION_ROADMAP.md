# Beta → Production Roadmap

**Status of this document.** An engineering audit of the repository as it
stands, followed by a phased plan to production. Every claim in the
"Findings" section was verified by running the tooling in this repo, not
inferred from documentation.

---

## 1. The thesis

**The science is roughly two years ahead of the engineering.**

This is not a project that needs more methods. It has 40+ estimators, a
four-layer hallucination guard, a falsification harness (the Parrot
Diagnostic) that most LLM-for-science tools do not bother to build, a
provenance manifest, and a decision ledger. The method-coverage audit has
already surfaced a 1610-item backlog. Adding to it is the lowest-value work
available.

What is missing is the layer that converts "impressive research code" into
"a number someone will stake a decision on":

1. **Nothing enforces the quality gates.** CI has never executed once.
2. **Nothing proves the statistics are calibrated.** Point estimates are
   spot-checked; coverage is never tested.
3. **Nothing makes a run reproducible on another machine.** No lockfile, no
   container — in a tool whose entire pitch is reproducibility.
4. **Failures are quiet.** 237 broad exception handlers, 36 of which swallow
   silently, in a pipeline designed to fall back to the "next-best estimator."

Those four, in that order, are the roadmap. Everything else is downstream.

---

## 2. Findings

### 2.1 Blocking — the ship gate is not connected

**CI has never run. Not once.**

The workflow is committed at `causalrag/.github/workflows/ci.yml`. GitHub
Actions only discovers workflows at `<repo-root>/.github/workflows/`. The
repository root is one level up, and has no `.github` directory at all.

```
$ git ls-files | grep workflow
causalrag/.github/workflows/ci.yml     # ← one level too deep

GitHub Actions API, list workflow runs for brennansk1/felisha:
{"total_count": 0, "workflow_runs": []}
```

Consequences, all of which are currently load-bearing claims in `README.md`
and `CHANGELOG.md`:

- "CI fails on regressions" — it cannot; it has never started.
- "audit GREEN" — the end-to-end flow audit is not a gate, it is a script
  nobody runs automatically.
- "1165+ unit tests passing" — unverified by any automated process.

**And the gate would fail today if it were connected.** Running the two
static checks exactly as `ci.yml` specifies them:

| CI step | Command | Result |
|---|---|---|
| Lint | `ruff check src tests` | **997 errors** |
| Type check | `mypy src/causalrag` | **426 errors in 106 files** |

Both steps would fail before the test step ever ran.

### 2.2 The toolchain is unpinned, so the lint surface moves on its own

`pyproject.toml` declares `ruff>=0.4` and `mypy>=1.10` with no upper bound
and no lockfile. `ruff` 0.4 shipped in April 2024; the current resolution is
0.15.8. The project selects rule *families* (`select = ["E","F","I","B","UP","N","SIM","RUF"]`),
so every new rule Astral adds to `RUF`, `UP`, `SIM`, or `N` is silently
opted into on the next `pip install`.

That is where most of the 997 come from — new rules, not new bugs:

| Rule | Count | Nature |
|---|---:|---|
| `N806` non-lowercase-variable | 156 | Statistical notation (`X`, `Y`, `ATE`, `SE`) — should be waived, not "fixed" |
| `I001` unsorted-imports | 109 | Autofixable |
| `RUF002` ambiguous-unicode-docstring | 99 | Citations with en-dashes (Petersen–van der Laan) — waive |
| `RUF015` unnecessary-iterable-allocation | 85 | Autofixable |
| `UP037` quoted-annotation | 82 | Autofixable |
| `RUF100` unused-noqa | 68 | Autofixable |

**Good news in the detail:** `F401` (unused import) and `F841` (unused
variable) return **zero hits in `src/`**. The real-defect rules are clean;
the noise is stylistic churn from version drift. There is no rot here — only
an unpinned toolchain.

### 2.3 Type coverage is the one genuinely large debt — but a third of it is config

`mypy` runs in `strict = true` mode over 194 files and finds 426 errors. The
distribution matters:

| Error code | Count | What it actually is |
|---|---:|---|
| `import-untyped` | 132 | Missing `ignore_missing_imports` overrides |
| `no-untyped-call` | 88 | Calls into the above |
| `unused-ignore` | 32 | Stale `# type: ignore` from older mypy |
| `type-arg` | 27 | Bare `list`, `dict`, `App` |
| `no-any-return` | 26 | Real, low-severity |
| `import-not-found` | 24 | Optional deps not installed |
| `assignment`, `arg-type`, `operator`, `index` | 62 | **Real type defects worth reading** |

The overrides block covers only four modules (`econml`, `networkx`,
`pynvml`, `psutil`). The libraries actually triggering errors are:

```
pandas 66 · sklearn 26 · scipy 15 · pyarrow 14 · tigramite 7 · statsmodels 5
rpy2 4 · dowhy 3 · causallearn 3 · pysyncon 2 · pymc_bart 2 · pymc 2 · arviz 2
sqlalchemy · sentence_transformers · sensemakr · ibis · duckdb
```

**156 of 426 errors (37%) clear with one 18-line config block plus
`pandas-stubs`.** Another 32 clear by deleting stale ignores. The residue —
roughly 60 real type defects, concentrated in `estimators/` (186 errors
total, the largest single cluster) — is genuine work.

### 2.4 Statistical validation stops one step short of the claim that matters

There is a real validation suite (`tests/synthetic_datasets/`) covering
Lalonde, IHDP, ACIC, M-bias colliders, high-dimensional `n ≪ p`, mixed
dtypes, and survival routing. This is well above average.

But every assertion is a **single-draw point-estimate tolerance**:

```python
# tests/synthetic_datasets/test_acic.py:52
assert abs(result.point_estimate - true_ate) < 0.20 * abs(true_ate)

# tests/synthetic_datasets/test_ihdp_cate.py:85
assert abs(result.point_estimate - true_ate) < 0.2 * abs(true_ate) + 1.0
```

One dataset, one seed, one fit, a 20% band. That tests *"did this run
crash or return something wildly wrong."* It does not test the thing the
tool exists to deliver.

**No test anywhere asserts confidence-interval coverage.** For a causal
inference tool, calibrated uncertainty *is* the product. A DML estimator
whose 95% CI covers truth 71% of the time will pass every test in this repo
and produce confidently wrong decisions forever. That is the single largest
correctness gap in the codebase, and it is invisible to the current suite.

### 2.5 Reproducibility is asserted, not achieved

`run.lock.json` hashes data, DAG, estimand, RNG seed, git SHA, package
versions, model digests, and prompts. The design intent is exactly right.

But nothing can *act* on that manifest:

- No dependency lockfile (`uv.lock` / `requirements.txt` / `poetry.lock`).
  `dowhy>=0.11` resolves to whatever shipped this morning.
- No container image. The R bridge needs R 4.4+ and ~30 CRAN packages
  installed by hand from a README code block.
- No `causalrag verify <run.lock.json>` command to restore an environment
  and reproduce a prior estimate.

A manifest that records an environment nobody can reconstruct documents
irreproducibility precisely. Independently: the R bridge — roughly 20 of the
40+ advertised estimators — is **never exercised in CI**, and `_r.py`
documents its singleton R session as explicitly not thread-safe.

### 2.6 Failures are quiet by construction

237 `except Exception` handlers; 36 immediately `pass` or `continue`.

Individually defensible — the README's "every optional dep degrades
gracefully" is a deliberate, reasonable design choice. In aggregate, in a
pipeline that auto-routes to the "next-best estimator" on failure, it means:

> A user can receive an estimate from a fallback method, and have no way to
> learn that the method they intended silently failed to fit.

Distribution: `estimators/` 67, `llm/` 33, `tasks/` 21, `identify/` 15,
`sensitivity/` 13. These are precisely the paths where a silent substitution
changes the scientific interpretation of the output.

Compounding it: **105 `print()` calls in `src/`, against 9 modules that
import `logging`.** There is no structured logging, no run correlation ID, no
way to reconstruct after the fact why a given run chose the estimator it
chose.

### 2.7 Secrets land in artifacts

`SQLConnector` takes an inline-credential SQLAlchemy URL:

```
postgresql+psycopg://user:pw@host/db
```

That string flows into `DatasetSpec.source` (`core/protocol.py:35`) and
`ManifestBuilder.dataset_path` (`provenance/manifest.py:182`), both of which
are serialised verbatim — to `study.causalrag.yaml` and `run.lock.json`.
Those files are the things users are encouraged to commit and share as
evidence of reproducibility. There is no redaction anywhere in the codebase,
and no environment-variable path for credentials.

### 2.8 The test suite cannot function as a merge gate

Measured in this container: the full `pytest` run exceeded **40 minutes**
without completing, and `pytest tests/unit` alone — pinned to single-threaded
BLAS — exceeded **35 minutes** without completing. Neither produced a pass/fail
count within the time available, so the documented "1165+ unit tests passing"
could not be independently confirmed here.

Process inspection showed the suite was not hung — it was genuinely computing,
with joblib `loky` workers alive for 10–18 minutes each at ~40% CPU, fitting
real causal forests inside the test process.

- No `pytest-xdist`; the suite is serial.
- Estimator tests fit real forests at default `n_jobs`, oversubscribing CPU
  in any containerised runner.
- `pytest-cov` is a declared dev dependency but no coverage threshold is
  configured or enforced.
- No `pytest-timeout`, so one pathological fit hangs the whole run.

A 40-minute serial gate will be bypassed by whoever is shipping under
pressure, which in practice means it is not a gate.

### 2.9 Packaging and distribution

| Claim | Reality |
|---|---|
| README: "Status — v1.0" | `version = "0.1.0a0"` |
| README: "v1.0 sprint plan complete" | `Development Status :: 2 - Pre-Alpha` |
| `pyproject`: `license = { text = "MIT" }` | **No `LICENSE` file exists** |
| `pyproject` URLs → `github.com/brennanskelley/causalrag` | Actual repo is `brennansk1/felisha` |
| — | No git tags, no releases, no published package |
| — | No `CONTRIBUTING.md`, no `SECURITY.md` |
| — | 1.8 MB of PDF/text design docs committed at repo root |

A declared MIT license with no license file is not a license. No
organisation's legal review clears that, which blocks adoption regardless of
technical merit.

### 2.10 No programmatic surface

`src/causalrag/__init__.py` is three lines and exports `__version__`. Every
capability is reachable only through the CLI or the TUI. A user who wants to
call this from a notebook or a pipeline must import internals
(`causalrag.roadmap.q7_estimate.estimate`, `causalrag.estimators.python.select`),
none of which carry any API-stability guarantee.

Also: the LLM layer supports Ollama, llama.cpp, vLLM, and MLX — **all
local**. There is no adapter for a frontier hosted model. For the reasoning-
heaviest steps in the pipeline (DAG proposal, domain inference, executive
synthesis) that is a real ceiling on output quality, and it is the gap most
directly relevant to being "cutting edge."

---

## 3. What is genuinely strong — preserve these

An honest audit records what not to break:

- **Four-layer hallucination guard** (`llm/guards.py`) — prevention →
  schema → semantic → *statistical*. Layer 4 testing the LLM's implied
  conditional independencies against the data, and surfacing disagreement
  without letting either side silently win, is the correct architecture.
  Most tools in this space stop at Layer 2.
- **The Parrot Diagnostic** — a sign-flipped Lalonde falsification harness
  that detects whether the LLM is reasoning over supplied data or reciting a
  memorised benchmark. Genuinely novel. This should become a scored,
  per-model regression gate, not a one-off document.
- **Audit-trail-first design** — decision ledger, identification narration,
  sensitivity verdicts as first-class serialised records.
- **HTML reporting escapes correctly** — all 83 interpolation sites route
  through `_e()`. LLM-authored text into HTML is a classic injection vector
  and it is handled.
- **`extra="forbid"` on the Pydantic models** — protocol corruption fails
  loudly instead of silently dropping fields.
- **Eager `EngineNotAvailable`** — the engine layer refuses to silently fall
  back, with an explicit comment that silent degradation hides
  misconfiguration. That instinct, applied consistently, is the fix for §2.6.
- **The flow audit concept** — orphan flags, unreachable estimators, panels
  absent from the report. The idea is excellent; it just needs to be wired
  to a runner that actually executes.

---

## 4. The plan

Sequenced so that each phase makes the next one verifiable. Effort assumes
one experienced engineer; the phases overlap where noted.

### Phase 0 — Make the gate real (1–2 weeks)

Nothing else can be trusted until the automation runs. Order matters.

1. **`git mv causalrag/.github .github`** and re-scope paths in `ci.yml`.
   One command; converts every documented gate from aspiration to fact.
2. **Pin the toolchain.** `ruff==0.15.8`, `mypy==1.18.*`, exact pins for all
   dev tools. Add `uv.lock`. Version drift stops being a source of red.
3. **Get lint to actually-green, honestly.** Autofix the ~394 mechanical
   findings. Then *waive the rules that are wrong for this domain* rather
   than contorting the code: `N803`/`N806` (statistical notation is correct
   as `X`, `Y`, `ATE`) and `RUF001`–`RUF003` (bibliographic en-dashes).
   Record the waivers with reasons in `pyproject.toml`.
4. **Type-check: ratchet, do not boil the ocean.** Add the 18-library
   override block and `pandas-stubs` (−156). Delete stale ignores (−32).
   Then set `strict = true` globally with explicit
   `disallow_untyped_defs = false` overrides for the ~10 worst modules, and
   add a CI check that the override list may only shrink. Green from day
   one, monotonically improving.
5. **Split the suite.** `pytest -m "not slow and not integration"` as the PR
   gate under 5 minutes with `pytest-xdist`; the full suite nightly. Add
   `pytest-timeout`. Force `n_jobs=1` in estimator fixtures.
6. **Wire `audits/end_to_end_flow.py` into CI as a required check.** It was
   designed to be the ship gate; make it one.
7. **Housekeeping with outsized effect:** add `LICENSE` (MIT, matching the
   declaration), fix the `pyproject` repository URLs, reconcile
   `version`/`Development Status` with reality, move the 1.8 MB of design
   PDFs out of the repo root.

**Exit criterion:** a green CI badge on `main` that means something.

### Phase 1 — Prove the statistics (3–5 weeks; the highest-value phase)

This is what would make the tool genuinely cutting edge, because almost
nobody ships it.

1. **A coverage harness.** For each (estimator × DGP) pair: simulate N=500+
   replications from a known ground truth, fit, and assert the empirical
   95%-CI coverage lands within Monte Carlo error of nominal. Track bias,
   RMSE, and CI width alongside it. This replaces
   `abs(point - true) < 0.2*|true|` with the claim the tool actually makes.
2. **A DGP library** spanning the conditions each method assumes and the
   conditions where it should fail: confounding strength, positivity
   violations, non-linearity, effect heterogeneity, treatment/outcome
   dtype crosses, n from 100 to 100k, p from 5 to 5000.
3. **Negative controls as first-class tests.** Assert methods *fail
   loudly* under their own violated assumptions — the M-bias test is the
   right shape; generalise it. A method that returns a tight CI under a
   positivity violation is a bug, and today no test can see it.
4. **A published certification matrix.** For every one of the 40+
   estimators: measured coverage, bias, and RMSE per regime, versioned in
   `docs/` and regenerated nightly. Turn the marketing claim ("40+
   estimators") into an evidence table. Any method that cannot be
   certified gets labelled experimental in the catalog and in the report.
5. **Calibration gates in CI.** Coverage regression fails the nightly
   build. This is the single most valuable check the project can own.

**Exit criterion:** for any estimate the tool emits, you can cite measured
coverage for that method in that regime.

### Phase 2 — Reproducibility that survives leaving your laptop (2–3 weeks, overlaps Phase 1)

1. **Lock everything.** `uv.lock` committed; `renovate`/`dependabot` for
   controlled bumps.
2. **Container images.** A Python-only image, and a full image with R 4.4 +
   the ~30 CRAN packages pinned via `renv.lock`. This is what makes the R
   half of the catalog real for anyone who is not you.
3. **Test the R bridge in CI.** A container job with R installed. ~20
   advertised estimators currently have zero automated coverage.
4. **`causalrag verify <run.lock.json>`** — restore the recorded
   environment, re-run, and diff the estimates. Reproducibility becomes an
   executable assertion instead of a JSON file. This is the feature a
   regulated user (clinical, policy, audit) will ask for first.
5. **Redact credentials at the boundary.** Parse connection URLs, strip
   userinfo before it reaches `DatasetSpec.source` or the manifest, and read
   secrets from the environment or a keyring. Add a test asserting no
   artifact can contain a password.

### Phase 3 — Make failure loud (2–3 weeks)

1. **Structured logging.** Replace 105 `print()` calls with `structlog`,
   JSON output, a `run_id` on every record, and OpenTelemetry spans around
   LLM calls and estimator fits. Long autonomous runs become debuggable.
2. **A typed degradation ledger.** Every fallback — missing dep, failed fit,
   estimator swap, circuit-breaker trip — becomes a structured record that
   surfaces in the report: *"This estimate came from WeightIt because
   `dml.forest` failed to converge (n=142, 3 covariates constant)."*
   Replace the 36 silent swallows first; narrow the remaining broad
   handlers to specific exception types.
3. **Fail-closed as a mode.** `--strict` makes any degradation an error
   rather than a fallback. Regulated users need to run this way, and it is
   the honest default for a publication-bound analysis.
4. **Budget and cost enforcement.** `loop_observability/budget.py` exists;
   make exceeding a budget a hard stop with a resumable checkpoint.

### Phase 4 — A product surface (4–6 weeks)

1. **A real Python API.** Curate `causalrag/__init__.py` into a stable,
   documented, semver-guaranteed surface: `Study`, `discover`, `estimate`,
   `sensitivity`, `report`. The notebook is where analysts live, and today
   they must import internals.
2. **Server mode.** A multi-hour autonomous LLM run does not belong in a TUI
   session that dies with the SSH connection. FastAPI + a job queue
   (Celery/Arq) + artifact storage (S3-compatible) + resumable checkpoints,
   with the TUI and CLI as clients of the same API.
3. **Versioned artifacts.** Studies and runs become immutable, addressable,
   diffable objects. "What changed between run 41 and run 42" is the
   question every real user asks second.
4. **Documentation site.** MkDocs or Quarto: method reference with the
   Phase-1 certification data inline, worked examples per domain, an
   explicit assumptions-and-limitations page.

### Phase 5 — The cutting-edge differentiators (ongoing)

Only credible once Phases 0–3 hold. Ranked by defensibility:

1. **Model-agnostic LLM layer + a scored eval harness.** Add hosted
   frontier-model adapters alongside the local engines. Then promote the
   Parrot Diagnostic into a versioned benchmark that scores every candidate
   model on parroting, DAG quality against expert-labelled graphs,
   identification-strategy correctness, and synthesis faithfulness. **A tool
   that can tell you "this model parrots memorised benchmarks 34% of the
   time, this one 4%" owns a category no one currently occupies.**
2. **Assumption-aware agentic critique.** The `--paranoid` multi-agent
   debate is the right seed. Point it specifically at assumption
   violations, with the Layer-4 statistical audit as the referee, so the
   debate is grounded in data rather than in rhetoric.
3. **Report faithfulness verification.** An automated check that every
   numeric claim in the executive synthesis traces to a specific record in
   the protocol. This closes the last hallucination gap — the one at the
   very end, in the text the decision-maker actually reads.
4. **Active experimental design.** The EIG machinery already computes
   information gain over hypotheses. Turning that outward — "run *this*
   experiment next, with *this* sample size" — converts the tool from an
   analyser of existing data into a planner of new data collection. That is
   a substantially larger market.
5. **Then, and only then, more methods.** The 1610-item backlog is real, but
   an uncertified 60th estimator is worth less than a certified 40th.

---

## 5. Sequencing summary

| Phase | Focus | Effort | Unblocks |
|---|---|---|---|
| 0 | Make the gate real | 1–2 wk | Everything. Do this first. |
| 1 | Prove the statistics | 3–5 wk | Any claim about correctness |
| 2 | Reproducibility | 2–3 wk | Regulated / audited users |
| 3 | Loud failure | 2–3 wk | Debuggability, trust in fallbacks |
| 4 | Product surface | 4–6 wk | Adoption beyond the author |
| 5 | Differentiators | ongoing | Category leadership |

Phases 1 and 2 overlap. Realistic path to defensible production grade:
**roughly one quarter of focused work**, with Phase 0 delivering
disproportionate value in the first fortnight.

## 6. The single highest-leverage change

If only one thing gets done: **`git mv causalrag/.github .github`, pin the
toolchain, and get CI actually green.**

Not because lint matters more than causal inference, but because until
automation runs, every other quality claim in this repository — 1165 tests,
audit GREEN, CI fails on regressions — is an assertion rather than a fact.
Phase 1's coverage harness is the most *valuable* work, but it is only
trustworthy once something runs it on every commit.
