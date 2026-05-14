# CausalRoadmap

An LLM-assisted causal-inference pipeline that drives a dataset all the way from raw
observations to a domain-aware executive synthesis — implementing the Petersen–van der
Laan Causal Roadmap end-to-end (Q1–Q8) with a deep estimator catalog and an
autonomous "master mode" that runs N experiments back-to-back.

Built around a few load-bearing ideas:

* **Hardware-aware local LLM orchestration** via Ollama (no cloud round-trips).
* **A typed Causal Roadmap walk per hypothesis** — every estimate carries its
  identifiability assumptions, sensitivity verdict, and refutation results.
* **A method catalog of 23+ estimators** spanning Python (EconML DML / forests /
  meta-learners / BART) and R (`grf`, `lmtp`, `MatchIt`, `WeightIt`, `mediation`,
  `survRM2`, `bartCause`, `marginaleffects`, `sensemakr`, `EValue`, `tipr`, `bnlearn`).
* **An autonomous master loop** that plans a prioritized candidate queue, runs
  propose-K / critique / commit each turn, and supports foundation-recursion
  chains with per-chain bookkeeping and red-sensitivity auto-robustness checks.
* **Domain-agnostic executive synthesis** — the final report translates findings
  into the language of whatever field the data is from (clinical, policy,
  business, ecology, engineering, marketing, operations, …).

## Status — May 2026 sprint week 3-4

Beta. Roadmap walks, R bridge, master mode, and domain-aware synthesis are all
live. Recent audit-driven hardening covered: SE-anchored refutation thresholds,
IV partial-F first-stage diagnostic, E-value scale routing per estimator,
collider/descendant/mediator safety guard in identification, bootstrap CIs for
meta-learners, richer SuperLearner fallback for `lmtp`, dtype-aware outcome
classification, fabricated-id validation in synthesis, multi-chain foundation
bookkeeping, and an expanded `DataFlag` system with detection.

226+ unit tests, all passing.

## Install

```bash
pip install -e ".[dev,estimators]"
```

R bridge (optional but recommended — unlocks ~15 estimators):

```bash
# R 4.4+ required
Rscript -e 'install.packages(c("grf","lmtp","MatchIt","WeightIt","marginaleffects", \
  "sensemakr","mediation","survRM2","EValue","tipr","bartCause","bnlearn","cobalt"), \
  repos="https://cloud.r-project.org", type="binary")'
```

## Quick start (CLI)

```bash
causalrag init my_study
cd my_study
causalrag validate
causalrag doctor

# Deterministic single-pass run on a CSV
causalrag run data/cohort.csv --treatment T --outcome Y

# Autonomous master mode (LLM-driven, K experiments + foundation chains)
causalrag auto run data/cohort.csv --experiments 10 --foundation
```

Both `run` and `auto run` produce:

* `study.causalrag.yaml` — every hypothesis, identification result, estimate,
  sensitivity verdict, decision-ledger entry.
* `executive_synthesis.json` (auto only) — domain-aware ranked findings.
* `reports/<project>_<timestamp>.html` — executive summary on top, technical
  Roadmap walks below.

## TUI

```bash
causalrag tui
```

A Textual terminal app shows a phase tracker (Discover → Feasibility →
Hypothesize → Estimate → Sensitivity → Report), a streaming log view, and a
slash-command composer. Every command below maps directly to a CLI verb but
runs inside the persistent TUI session with live progress updates.

### Slash commands

| Command | Phase | What it does |
|---|---|---|
| `/init <name>` | 0 | Scaffold a new study directory with `study.causalrag.yaml`, `reports/`, `.causalrag/`. |
| `/doctor` | 0 | Diagnostic: Python/R env, Ollama reachability, hardware probe (CPU/GPU/RAM), cassette dir, registered estimator count. |
| `/discover <data>` | 1 | Profile dataset → LLM investigator (per-column semantic role) → domain-expert brief (candidate DAGs + identification warnings) → Layer-4 audit (conditional-independence checks). Emits `DataFlag`s, candidate graphs, role assignments. |
| `/feasibility` | 2 | Power × MDE grid across admissible (T, Y) pairs given the data. Surfaces which research questions are statistically reachable at this sample size. |
| `/hypothesize` | 3 | Ranked, scoped hypotheses. `--mode master` lets the LLM enumerate the credible set; otherwise hypotheses are constructed deterministically from feasibility output. |
| `/estimate` | 4 | Walk one hypothesis through Roadmap Q5 (identify) → Q6 (statistical estimand) → Q7 (estimate + refute). Picks the best estimator from the catalog via the rule cascade unless `--prefer <id>` is passed. |
| `/sensitivity` | 5 | E-value (scale-routed per estimator + outcome), sensemakr robustness value, verdict aggregation (green/yellow/red), multiverse triangulation across DAG × estimator × adjustment. |
| `/report` | 6 | Render HTML (default) or Markdown. Executive synthesis is rendered at the top if `executive_synthesis.json` is present. |
| `/run <data> [--treatment T] [--outcome Y]` | 0–6 | Deterministic single-pass: discovery → feasibility → hypothesize → estimate → sensitivity → report. One LLM call per phase. |
| **`/auto run <data> --experiments K [--foundation]`** | 0–6 | **Autonomous master mode.** See below. |
| `/help` | — | List of commands. |
| `/clear` | — | Clear the log view. |
| `/quit` | — | Exit the TUI. |

### `/auto` — the master loop in detail

```bash
/auto run data/cohort.csv \
    --experiments 10 \
    --foundation \
    --max-foundation-iterations 8 \
    --max-foundation-depth 4
```

Flags:

| Flag | Default | Meaning |
|---|---|---|
| `--experiments K` | 5 | Total successful experiments to run. Hard cap. |
| `--foundation` | off | Enable foundation-recursion follow-ups (CATE-on-modifier, mediation decomposition, robustness child on red sensitivity). Without this flag, every experiment is an independent root. |
| `--max-foundation-iterations N` | 8 | Cumulative budget for foundation experiments across all chains. |
| `--max-foundation-depth D` | 4 | Longest chain depth a single foundation thread can reach. |
| `--question "..."` | none | Research question to seed discovery. |

What the loop actually does each turn:

1. **Plan** — one up-front LLM call enumerates 15–30 credible candidate
   experiments covering the dataset. A deterministic scorer ranks them by

   ```
   score = 0.40·impact + 0.25·identifiability + 0.20·power_proxy + 0.15·novelty − cost
   ```

   where `identifiability` is the strength of the available identification
   strategy (backdoor with named confounders vs IV vs front-door vs nothing),
   `power_proxy` blends the LLM's hint with the sample-size band, `novelty`
   penalizes already-tested triples, `cost` mildly downweights expensive
   estimators (forests / BART).

2. **Propose-K → critique → commit** — each turn the top-K candidates go to
   a critic LLM agent that checks: already-tested? identifiability sniff
   (NDE/NIE needs mediator, LATE needs instrument, MTP needs continuous T)?
   recommended method actually in the catalog? min sample size feasible?
   Rejected candidates are vetoed; the highest-scoring survivor runs.

3. **Roadmap walk** — the chosen experiment walks Q5 (identify, with
   collider/descendant/mediator safety guard) → Q6 (statistical estimand)
   → Q7 (estimate with cross-fitting + refutation with SE-anchored
   thresholds) → Q8 (E-value on the correct scale for this estimator
   + sensemakr + verdict aggregation).

4. **Foundation firing** (when `--foundation` is set) — a deterministic
   rule decides whether to auto-fire a follow-up child:

   ```
   fire iff
       parent.sensitivity_verdict ≠ red
     AND |parent.point/SE| > 1.96
     AND chain.depth < max_foundation_depth
     AND chain.null_streak < 2
     AND chain.info_gain_streak_below_eps < 2
     AND foundation_iterations_used < max_foundation_iterations
   ```

   When the rule fires, a focused LLM call specifies *which* follow-up
   (CATE on the strongest modifier, mediator decomposition, etc.). Chains
   are tracked per-`chain_id`; interleaved independent experiments do NOT
   reset another chain's depth. Each chain ends when budgets are met,
   when info-gain plateaus (Δpoint/SE < 0.3 twice in a row), or when the
   null-streak hits the threshold.

5. **Auto-robustness on red** — when sensitivity comes back RED, the loop
   *automatically* schedules a robustness re-run with a different
   identification strategy (e.g., WeightIt if parent used DML, or
   LinearDML if parent used a forest). The user does not have to ask.

6. **Recovery and dead-end handling** — estimator fit failures trigger a
   one-shot retry with the auto-cascade (no LLM hand-holding).
   Unidentifiable proposals capture the missing piece (instrument?
   mediator? descendant in adjustment set?) and surface it back to the
   next critic call. `--max-consecutive-failures` (default 3) trips the
   safety circuit-breaker.

7. **Multiple-testing adjustment** — after the loop ends, BH/BY/Bonferroni
   (per `protocol.multiple_testing`) is applied across all `p_value`s in
   the queue. The adjusted p-values land on each
   `EstimationResult.diagnostics` and are quoted in the synthesis prompt.

8. **Domain-aware synthesis** — the reasoning LLM infers the dataset's
   domain (clinical / business / policy / ecology / engineering / …) and
   writes the executive summary in that field's vocabulary. Findings are
   ranked by impact × confidence. A deterministic override forces
   `confidence="low"` if the CI crosses zero, sensitivity is red, or
   n_used < 100 — the LLM cannot mark a fragile finding as high
   confidence.

Output for an `/auto` run is the same set of files as `/run` plus
`executive_synthesis.json` and a richer decision ledger showing every
chain, every critic rejection, and every foundation firing.

## Pipeline architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1 — discovery                                                 │
│   profile → investigator → expert brief → candidate DAGs → flags    │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 2 — feasibility (power × identifiability × sample-size)       │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 3 — hypothesize  (manual / automated / master-mode queue)     │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 4 — Causal Roadmap walks  (Q5 identify → Q6 statistical       │
│   estimand → Q7 estimate + refute → Q8 interpret + sensitivity)     │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 5 — multiple-testing adjustment + cross-experiment analysis   │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 6 — executive synthesis (domain-aware) + HTML/MD report       │
└─────────────────────────────────────────────────────────────────────┘
```

## Layout

```
src/causalrag/
├── core/           # framework primitives — StudyProtocol, RoadmapWalk,
│                     EstimationResult, Decision ledger, CausalGraph
├── data/           # ingestion (csv/parquet/duckdb/sql/feather/json/excel)
│                     + dataset profiling + DataFlag detectors
├── discovery/      # Stage 1c investigator + Stage 1e domain-expert brief
├── feasibility/    # power × identifiability × sample-size check per (T,Y)
├── hypothesize/    # manual queue + LLM-driven master hypothesize
├── roadmap/        # Q5–Q8 — identify, derive statistical estimand,
│                     estimate, refute, interpret
├── estimators/     # method catalog
│   ├── python/     # OLS, DML (linear/sparse/forest), BART, meta-learners
│   └── rbridge/    # grf, lmtp, MatchIt, WeightIt, mediation, survRM2,
│                     bartCause, marginaleffects, bnlearn, sensemakr
├── sensitivity/    # E-value (scale-routed), sensemakr, verdict aggregator,
│                     multiple-testing adjustment, anomaly audit
├── reporting/      # executive synthesis (domain-aware), HTML / Markdown
├── llm/            # Ollama client + shared honesty preamble
├── tui/            # textual TUI — operator console for /run, /auto, …
├── master_loop.py  # the autonomous propose-K / critique / commit loop
└── cli/            # Typer app
```

## Estimator catalog

`causalrag explain --method <id>` shows the use-case for any registered method.
A non-exhaustive sample:

| Estimator | Use case |
|---|---|
| `python.dml.linear` | DML with linear final stage — defensible default for ATE |
| `python.dml.causal_forest` | EconML CausalForestDML for HTE / CATE |
| `python.dml.sparse_linear` | Lasso final stage for high-dim adjustment |
| `python.bart.dml` | Bayesian causal forest via PyMC-BART |
| `python.dr.dr_learner` | EconML DRLearner with cross-fit (cv=5) |
| `python.meta.{s,t,x}_learner` | Meta-learners w/ bootstrap CIs |
| `python.linear.ols` | HC3 OLS — small-sample honest default |
| `rbridge.grf.causal_forest` | reference grf::causal_forest |
| `rbridge.grf.causal_survival_forest` | survival CATE (Cui-Athey-Tibshirani 2023) |
| `rbridge.grf.instrumental_forest` | IV-CATE with partial-F diagnostic |
| `rbridge.grf.multi_arm_causal_forest` | multi-arm treatment |
| `rbridge.lmtp.{shift,policy,mixture,sdr,contrast}` | stochastic interventions / MTP |
| `rbridge.matchit` | MatchIt + marginaleffects post-match g-computation |
| `rbridge.weightit` | propensity weighting (GLM / GBM / CBPS / EBAL / BART / SuperLearner) |
| `rbridge.bartcause` | Bayesian causal forest with calibrated posterior CIs |
| `rbridge.mediation` | NDE / NIE |
| `rbridge.survrm2` | RMST contrast |
| `rbridge.marginaleffects.slopes` | continuous-T marginal slopes |

## Master mode (autonomous "drop a dataset → K experiments")

```bash
causalrag auto run data/cohort.csv --experiments 10 --foundation
```

What happens:

1. **Discovery** — LLM investigator + domain-expert brief + candidate DAGs.
2. **Candidate queue** — one LLM call enumerates 15–30 credible experiments;
   a deterministic scorer (`impact × identifiability × power × novelty − cost`)
   ranks them.
3. **Iterative loop** — each turn the top-K candidates go through a critic
   agent (catalog-validity, already-tested guard, identifiability sniff-test),
   the highest-scored survivor runs the Roadmap.
4. **Foundation chains** — when a parent comes back significant + non-red,
   a deterministic rule fires a follow-up (CATE on the strongest modifier,
   mediator decomposition, …). Chains are tracked per-id (multi-chain
   bookkeeping); independent experiments do NOT reset another chain's depth.
   When sensitivity comes back RED, the loop auto-schedules a robustness
   re-run with a different identification strategy.
5. **Recovery** — estimator-swap retries on fit failures; unidentifiable
   proposals surface the reason back to the next propose call; failures
   are captured into the history so the LLM can avoid the same dead-end.
6. **Multiple-testing** — BH/BY/Bonferroni adjustment across the K
   experiments before synthesis.
7. **Synthesis** — the LLM infers the domain (clinical / policy / business /
   ecology / engineering / …) and writes findings in that field's
   vocabulary, ranked by impact × confidence. The deterministic confidence
   override forces `low` if CI crosses zero or sensitivity is red.

## Design principles

* **Honest provenance.** Every LLM call, every estimator decision, every
  refutation result lives on the protocol with a Decision-ledger entry —
  reproducibility derives entirely from the YAML.
* **Refuse gracefully.** The pipeline never silently substitutes a worse
  estimator or fabricates an identifiability claim. Failures are captured
  with a reason and surfaced to the LLM so it can adapt.
* **Domain agnosticism.** The synthesis layer infers the audience from the
  data — a clinical dataset produces clinical implications, a sales dataset
  produces operator actions; the math underneath is the same.
* **Senior-statistician standards.** Refutation pass/fail thresholds are
  SE-anchored. IV first-stage uses partial F, not rank correlation. E-value
  is computed on the right scale for the estimator. Colliders, descendants,
  and mediators are programmatically excluded from adjustment sets.

## Tests

```bash
.venv/bin/python -m pytest tests/unit -q
```

226+ unit tests across the estimator catalog, identification gate, sensitivity,
refutation thresholds, flag detection, synthesis robustness, multiple-testing,
and the autonomous loop's prioritization scorer.

## Further reading

* `CausalRoadmap_PDD_v0.3.pdf` — the product design document (full architecture
  §13, four-week sprint plan §33).
* `docs/` — developer notes (when present).

## License

Internal / TBD.
