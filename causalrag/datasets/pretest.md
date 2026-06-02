# Pretest plan — CausalRoadmap v1.0 local integration test

**Purpose.** Validate the full auto-pilot pipeline (`causalrag run`) end-to-end on three
real datasets from three different domains. This file is the *pre-registration*: it records
what each dataset is, the known/expected causal structure, and **what auto mode should catch
on its own**. A companion `posttest.md` will record what auto mode *actually* caught, graded
against the expectations below.

The whole point of this exercise: see how much of the causal setup the pipeline infers
without hand-holding. We deliberately do **not** pre-wire treatment/outcome/graph — auto mode
should discover them. Where we *do* know the answer (the RCT, the IV benchmark), we grade the
recovered effect against it.

---

## Environment (read before running)

- **Interpreter:** always `causalrag/.venv/bin/python` (Python 3.12). Never the system `python3`
  (it is 3.9; `StrEnum`/`datetime.UTC` import errors there are environment artifacts, not bugs).
- **LightGBM/libomp:** on macOS the nuisance `auto` library avoids the LightGBM `stacked-rich`
  path (dual-OpenMP segfault) and uses sklearn-native `stacked-fast`. Already fixed in
  `estimators/python/nuisance.py` — do not revert.
- **Do not run the full pytest suite** to "warm up" — it exercises heavy estimator/integration
  paths. Run the pipeline directly per the commands below.
- **LLM inference** (role inference, DAG proposal, hypothesis generation) needs Ollama running
  at `http://127.0.0.1:11434`. This Mac probes as **tier T1 → `qwen3.5-9b` / `qwen3.5-9b-reasoning`**.
  For a no-LLM heuristic run (faster, deterministic, no Ollama), add `--no-llm`.

### Run commands

```bash
cd causalrag
# Full auto-pilot (LLM-assisted role + DAG inference). Pulls the T1 models via Ollama.
.venv/bin/python -m causalrag run datasets/healthcare_oncology/tcga_brca.parquet \
    -q "Does prior treatment affect survival in breast-cancer patients?"

.venv/bin/python -m causalrag run datasets/business_intelligence/hillstrom_email.csv \
    -q "Does the email campaign drive site visits and spend?"

.venv/bin/python -m causalrag run datasets/economics/pension_401k.csv \
    -q "What is the effect of 401(k) participation on net financial assets?"

# Heuristic / deterministic (no Ollama needed): append --no-llm to any of the above.
```

Each run auto-scaffolds a project, writes a `study.causalrag.yaml` with a `decision_ledger`
(every auto choice tagged `source=auto`), and emits an HTML report. The decision ledger + report
are what `posttest.md` will be scored from.

---

## Dataset 1 — TCGA-BRCA (healthcare / oncology) — *observational, confounded, high-dim*

- **Path:** `datasets/healthcare_oncology/tcga_brca.parquet`
- **Source:** ported from the predecessor CausalRAG TCGA-BRCA project (ACIC 2026 poster).
- **Shape:** 1,098 rows × 292 columns. Loads via `ParquetConnector`.
- **Domain:** precision oncology — breast-cancer patients (TCGA).

| Role | Candidate column(s) | Notes |
|------|--------------------|-------|
| Treatment | `prior_treatment` (Yes/No), or a therapy/drug indicator | binary; observational, not randomized |
| Outcome | `vital_status` (Alive 945 / Dead 152) or `days_to_death` (151 non-null) | survival; **right-censored** |
| Confounders | `age_at_diagnosis`, `ajcc_pathologic_stage`/`_t`/`_n`/`_m`, `tumor_grade`, `race`, `morphology`, `year_of_diagnosis` | classic clinical confounders |
| Nuisance / high-dim | many columns hold gene-expression / mutation **arrays** | array-valued cells; must be encoded or dropped |

- **Ground truth:** none (real-world observational; no known ATE). This dataset tests
  *robustness*, not number-recovery.
- **What auto mode should catch:**
  - infer a survival/treatment framing; pick treatment + outcome sensibly;
  - assemble a confounder set that includes **stage/grade/age** (omitting these = textbook bias);
  - raise the **HIGH-DIM** flag (≫ columns; array-valued features);
  - flag **censoring / survival** handling on `days_to_death`;
  - because it's observational, run **sensitivity to unmeasured confounding** (E-value) and
    report it prominently;
  - possibly warn on **class imbalance** (152/1098 deaths).
- **Known pitfalls to watch:** array/object columns crashing profilers (e.g. `nunique` on an
  ndarray cell); treating `days_to_death` as a plain continuous outcome (ignores censoring);
  several all-null columns (`cause_of_death`, `tumor_stage`, `ajcc_clinical_*`).

---

## Dataset 2 — Hillstrom MineThatData email (business intelligence / marketing) — *RCT, known answer*

- **Path:** `datasets/business_intelligence/hillstrom_email.csv`
- **Source:** Kevin Hillstrom MineThatData E-Mail Analytics challenge (2008); a real randomized experiment.
- **Shape:** 64,000 rows × 12 columns. Loads via `CSVConnector`.
- **Domain:** e-commerce marketing.

| Role | Column(s) | Notes |
|------|-----------|-------|
| Treatment | `segment` — `Mens E-Mail` (21,307) / `Womens E-Mail` (21,387) / `No E-Mail` (21,306) | **randomized**, ~⅓ each; binarize to email-vs-none for ATE |
| Outcomes | `visit` (mean 0.147), `conversion` (mean 0.009), `spend` (mean \$1.05) | site visit, purchase, revenue |
| Pre-treatment covariates | `recency`, `history_segment`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel` | for variance reduction only — not needed for identification |

- **Ground truth (this is the gradable one):** because treatment is randomized, the ATE is the
  raw difference in means — **no confounder adjustment required**. Published/replicated result:
  any-email vs no-email lifts the **visit rate from ≈10.6% to ≈18.3% (≈ +7–8 percentage points)**;
  conversion lift ≈ +0.5–0.7 pp; modest positive spend lift. **The pipeline's recovered ATE on
  `visit` should land near +0.075 with a tight CI.**
- **What auto mode should catch:**
  - detect that treatment is **independent of covariates** (randomized) → identification is
    trivially valid, adjustment optional;
  - pick treatment = email/segment, outcome = `visit` (and/or conversion/spend), estimand = **ATE**;
  - recover the ~+7–8 pp visit effect;
  - surface the **uplift / heterogeneity** angle (mens vs womens creative).
- **Known pitfalls to watch:** 3-arm treatment (auto may binarize — fine, but note which contrast);
  `conversion` is very rare (0.9%) → wide CIs; `spend` is zero-inflated and heavy-tailed.

---

## Dataset 3 — 401(k) pension (economics / public finance) — *observational + instrument, IV benchmark*

- **Path:** `datasets/economics/pension_401k.csv`
- **Source:** SIPP 1991; Chernozhukov et al. DML replication data (`VC2015/DMLonGitHub`).
- **Shape:** 9,915 rows × 14 columns. Loads via `CSVConnector`.
- **Domain:** household finance / retirement-savings policy.

| Role | Column | Notes |
|------|--------|-------|
| Treatment | `p401` — participates in 401(k) (rate 0.262) | endogenous (self-selected) |
| **Instrument** | `e401` — employer *offers* 401(k) (rate 0.371) | plausibly exogenous; instruments `p401` |
| Outcome | `net_tfa` — net total financial assets (mean ≈ \$18,052) | heavy-tailed, can be negative |
| Confounders | `age`, `inc`, `fsize`, `educ`, `db`, `marr`, `twoearn`, `pira`, `hown`, (`nifa`, `tw`) | **income (`inc`) is the key confounder** |

- **Ground truth (literature benchmark):** DML estimates put the effect of **eligibility
  (`e401`) on `net_tfa` at ≈ +\$8,000–\$10,000 (ATE ~\$9k)**, and the **participation (`p401`,
  instrumented by `e401`) LATE at ≈ +\$10,000–\$13,000**. Naïve unadjusted comparisons overstate
  this because high-income people both participate more and save more.
- **What auto mode should catch:**
  - identify `e401` as an **instrument** for `p401` (the IV structure is the crux here);
  - distinguish the clean **eligibility/ITT effect** from the **participation/LATE effect**;
  - include **income** in the adjustment set (omitting it = large upward bias);
  - select a **DML / IV** estimator and land in the ~\$9k–\$13k band;
  - flag **selection-on-income** confounding.
- **Known pitfalls to watch:** confusing the instrument (`e401`) with a confounder; treating
  `p401` as exogenous (no IV) → biased; `net_tfa` outliers/negatives distorting linear nuisance.

---

## What `posttest.md` will record (grading rubric)

For each dataset, capture from the `decision_ledger` + report:

1. **Role inference** — treatment / outcome / instrument auto picked, vs the expected roles above (correct / partial / wrong).
2. **Confounder set** — did it include the must-have adjusters (BRCA: stage/age; 401k: income)?
3. **Graph / identification** — DAG proposed, identification strategy (backdoor vs IV vs RCT),
   and any non-identifiability or assumption warnings.
4. **Estimand & estimator** — estimand class chosen, estimator selected, and why.
5. **Effect estimate vs benchmark** — point estimate + CI against the known answer
   (Hillstrom ≈ +0.075 visit; 401k ≈ +\$9k–\$13k; BRCA: directional/plausibility only).
6. **Flags raised** — HIGH-DIM, censoring, randomization, rare-outcome, heavy-tail, imbalance.
7. **Sensitivity** — E-value / robustness output, esp. for the observational BRCA + 401k cases.
8. **Operational** — runtime, peak memory, any crash / gap / unrefined-code surfaced (these feed
   back into the audit backlog).

**Pass bar:** auto mode recovers the Hillstrom RCT effect within CI, lands the 401k effect in the
literature band when the IV structure is used, and produces a defensible (if not numerically
gradable) BRCA analysis with sensitivity reported — all without manual `--treatment`/`--outcome`
hints. Anything auto mode misses here becomes a prioritized backlog item.

---

# Coverage roadmap — what must pass before the project is "1000% done"

The first three datasets only exercised **backdoor adjustment + DML** (plus one RCT). The project
implements ~30 estimators across many identification strategies; *coverage is about
`(estimand × identification strategy × data shape)`, not domain labels.* Below is the full matrix we
must clear. Most canonical datasets ship inside the installed **`causaldata`** package (tiny, no
large downloads) — exported to CSV under `datasets/<regime>/`. Three regimes **reuse** data we
already have. Each new regime is expected to surface code gaps (as G7–G10 did); that is the point.

Legend: ✅ done · ⏳ in progress · ☐ todo

## MUST (core methods the pipeline advertises)

| # | Regime / estimand | Data (source) | Benchmark / standard | Stresses | Status |
|---|---|---|---|---|---|
| 1 | **IV / LATE** | **reuse** 401k: `p401` treatment, `e401` instrument, `net_tfa` | participation LATE ≈ **+$10k–13k** | IV estimator, LATE estimand, instrument detection | ✅ **PASS via auto-pilot** (+$8,749 [7.4k,10.1k]); G13/G14 fixed |
| 2 | **Difference-in-Differences (2-group)** | `organ_donations` | sign/CI | panel/DiD identification | ⛔ **feature gap (G15)** — non-DAG design not wired |
| 3 | **Staggered-adoption DiD** | `castle` | small + on homicide | `rbridge.did.callaway_santanna` | ⛔ **feature gap (G15)** — estimator reachable, no detection/identification/routing |
| 4 | **Regression Discontinuity** | `gov_transfers` | discontinuity at cutoff | `rbridge.rd.rdrobust` | ⛔ **feature gap (G15)** — non-DAG design not wired |
| 5 | **CATE / heterogeneous / uplift** | **reuse** Hillstrom (mens vs womens) | heterogeneity by segment | meta-learners (S/T/X), causal forest, `uplift` task | ✅ runs + correct ATEs (G6 fixed); explicit segment-CATE breakdown = follow-up |
| 6 | **Survival / censored time-to-event** | **reuse** BRCA `days_to_death`+`event` | directional; censoring-aware | survival/RMST routing | ⚠️ censoring detected but NOT routed (G15-class) + G10 crash |

## SHOULD (rounds out the surface)

| # | Regime | Data | Standard | Status |
|---|---|---|---|---|
| 7 | **Synthetic control** | `texas` | SC gap vs donor pool | ⛔ **feature gap (G15)** — non-DAG design not wired |
| 8 | **Clean confounded observational (known effect)** | `nhefs` (causaldata; smoking-cessation → weight) | **≈ +3.4 kg** (Hernán *What If*) — a numeric backdoor benchmark | ☐ |
| 9 | **Mediation (NDE/NIE, frontdoor)** | `nhefs` mediation framing, or simulated front-door | decomposition runs | ☐ |
| 10 | **Multi-arm / continuous (dose) treatment** | Hillstrom 3-arm `segment`, or a dose dataset | **the deferred G7** contrast policy | ☐ |
| 11 | **Heavy missingness** | inject/῾use a high-missing dataset | `HEAVY_MISSINGNESS` flag → `hist-gbm` path | ☐ |
| 12 | **Non-CSV/Parquet connectors** | load 401k via **SQLite** + **DuckDB** + **Excel** | SQL/DuckDB/Excel connector code paths | ✅ PASS (SQLite + DuckDB + Excel all load 9915×14) |

## LOWER priority
13. Interference / spillovers (network data) · 14. Transportability / external validity (multi-site) ·
15. Proximal causal inference (measurement-error proxies).

## Cross-cutting blockers that gate "done" (from posttest)
- **G9** — wide-dataset discovery (200+ cols) fails on a small model; needs batched column
  classification or large-model routing. **Gates any high-dimensional domain.**
- **G10** — R-bridge estimator result-parsing crash on a positivity edge case.
- Lower: G1 (bnlearn discretization), G6 (LLM proposes off-graph outcomes).

**Definition of done:** regimes 1–6 PASS (numeric where a benchmark exists, defensible otherwise),
7–12 at least run cleanly end-to-end, and G9 resolved. Results recorded in `posttest.md`.
