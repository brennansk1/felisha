# CausalRoadmap Pipeline — Reference for TUI Design

This document is a single-source reference for every stage, input, output,
decision point, and visual surface in the CausalRoadmap pipeline. It exists so
a TUI designer (modeled after the Claude Code TUI) can render the right state
at the right time, surface the right decisions to the analyst, and persist the
right artifacts after each step.

The pipeline follows the **Petersen–van der Laan Causal Roadmap** (8 steps)
embedded in **7 architectural phases** (numbered 0–6). Both numbering systems
are referenced below.

---

## 1. High-level state machine

```
              ┌──────────────────────────────────────────────────────────┐
              │                  StudyProtocol  (study.causalrag.yaml)   │
              │  central, on-disk, every command reads and updates it    │
              └──────────────────────────────────────────────────────────┘
                                         ▲
                                         │  read/write
   ┌────────┐    ┌───────────┐    ┌──────┴───────┐    ┌──────────┐
   │ Phase 0│ →  │  Phase 1  │ →  │   Phase 2    │ →  │ Phase 3  │ →  ...
   │  init  │    │ discover  │    │  feasibility │    │hypothesize│
   └────────┘    └───────────┘    └──────────────┘    └──────────┘
        │              │                 │                  │
        ▼              ▼                 ▼                  ▼
    skeleton    DiscoveryReport   FeasibilityReport   HypothesisQueue
                + flags + DAGs    (admissible pairs)  (ranked, scoped)

   ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌───────────┐
   │ Phase 4  │ →  │ Phase 4  │ →  │   Phase 5    │ →  │  Phase 6  │
   │identify  │    │ estimate │    │ sensitivity  │    │  report   │
   │ (Step 5) │    │(Step 6+7)│    │  (Step 8)    │    │           │
   └──────────┘    └──────────┘    └──────────────┘    └───────────┘
        │              │                 │                  │
        ▼              ▼                 ▼                  ▼
  Identification  EstimationResult   SensitivityVerdict   HTML/PDF/Quarto
    (gates 7)     + refutations      + Layer-4 audit      + provenance log
```

Each phase is invoked by a CLI command; each command:
- Reads the current `study.causalrag.yaml` (or scaffolds a fresh one for `init`).
- Mutates only the slice of state it owns.
- Writes the updated YAML back atomically.
- Optionally emits a sidecar JSON in `.causalrag/` for richer downstream tools.

---

## 2. CLI commands and their visual surfaces

| Command | Phase | Reads | Writes | Streaming output |
|---|---|---|---|---|
| `causalrag init <name>` | 0 | nothing | scaffold + `study.causalrag.yaml` + `.causalrag/` | "Initialized" panel |
| `causalrag doctor` | — | hardware + Ollama | `.causalrag/hardware.json` (with `--save`) | tier table + missing-models alert |
| `causalrag validate` | — | YAML | nothing | round-trip check |
| `causalrag discover <data>` | 1 | YAML + dataset | discovery report + DAGs + flags + variables → YAML + `.causalrag/discovery.json` | column table + flags + expert brief panel + audit warnings |
| `causalrag feasibility` (Week 4) | 2 | YAML | feasibility report → YAML | MDE / power-target table + admissible pairs |
| `causalrag hypothesize` (Week 4) | 3 | YAML | hypothesis queue → YAML | ranked-hypothesis table + counterfactual-share split |
| `causalrag estimate` | 4 | YAML + dataset | RoadmapWalk → YAML | estimator + adjustment + point estimate + CI table |
| `causalrag sensitivity` | 5 | YAML + dataset | q8 interpretation → YAML | E-value + sensemakr + verdict colour table |
| `causalrag report` (Week 4) | 6 | YAML | HTML/PDF in `reports/` | progress bar + path to artifact |
| `causalrag run <data>` (Week 4) | 1→6 | YAML + dataset | everything | composite — runs all phases in sequence |

`run` is the one-shot pipeline; the others are stepwise for analysts who want
to inspect/override between stages. The TUI should support both modes.

---

## 3. Stage-by-stage detail

### Phase 0 — Initialization (`init`)

**What happens.** Creates the project skeleton:

```
<project>/
├── study.causalrag.yaml         # central protocol (Pydantic-validated)
├── data/                         # user-provided datasets land here
├── reports/                      # rendered output (HTML, PDF, Quarto)
└── .causalrag/
    ├── cassettes/                # LLM record/replay (sha256-keyed)
    ├── history.jsonl             # append-only log of every hypothesis
    ├── llm.lock                  # model digest + config (created by discover)
    ├── hardware.json             # doctor's structured snapshot
    └── discovery.json            # discovery sidecar
```

**Decisions.** Tier preference (`data-scientist | academic | domain-expert`).
Tier only affects defaults; every capability is reachable from every tier.

**TUI surface.** A one-shot `Panel.fit` with project name, path, tier.

### Phase 1 — Discovery (`discover`)

The most complex phase. Five sub-stages (1a–1e), all flowing through
`causalrag.discovery.run_discovery`:

#### Stage 1a — Connector ingestion (`data/connectors/`)

Connector protocol: `to_arrow()`, `describe()`, `supports_lazy()`. URIs are
dispatched via `from_uri(source)`:

- `csv://`, bare `*.csv`, `*.tsv` → `CSVConnector`
- `parquet://`, bare `*.parquet` → `ParquetConnector`
- (v0.5+) SQL, Mongo, FHIR, REST, HF Datasets, Stata/SAS/SPSS — protocol
  contract is already in place; readers ship across v0.5–v1.0.

**TUI surface.** "Loaded `<path>`: <n_rows> × <n_cols>".

#### Stage 1b — Deterministic statistical profile (`data/profiler.py`)

Designed to complete in <10 s for 10 M rows. Per-column `ColumnProfile`:

- Inferred logical dtype: `binary | ordinal | categorical | continuous | count
  | time_to_event | date | datetime | identifier | text`.
- Cardinality, missingness rate, top-20 value frequencies.
- Continuous: mean, SD, p5/p25/p50/p75/p95, skew, kurtosis, Tukey outlier count.
- Categorical: mode, entropy.
- Heuristic flags: `suspected_identifier`, `suspected_time_column`,
  `suspected_event_indicator`, `is_binary_01`, `constant`.

Dataset-level signals:

- `column_pairs_high_corr` (|r| ≥ 0.9).
- `censoring_pairs` (time + event column matches).
- `missingness_clusters` (columns that go missing together — MAR/MNAR hint).
- `n_exact_duplicate_rows` (after identifier drop).
- `string_formats` per text column: `identifier_like | email_like | url_like
  | zip_like | date_like | free_text | categorical_like`.

**TUI surface.** A condensed table per column (`name`, `logical_dtype`,
`missing %`, `cardinality`, hint icons). Bottom strip: dataset-level signals
(duplicates, censoring pairs, high-corr clusters).

#### Stage 1c — LLM investigator (`discovery/investigator.py`)

**Routed to the general-purpose / discovery slot** (instruction-tuned model;
qwen3:14b at T2, qwen3:32b at T3, llama3.3:70b at T4+). Receives:

- The compact `ColumnProfile` JSON (~5–20 KB).
- A 10-row sample.
- Optional user research question.

Emits an `InvestigatorReport`:

- `domain_tag` (one of: `clinical | financial | marketing | education |
  manufacturing | social_science | web_analytics | environmental | other`).
- Per column: `domain_meaning`, `domain_tag`, `value_interpretation`,
  `temporal_position` (`baseline | pre_treatment | treatment_era |
  post_treatment | outcome | unknown`), `watch_for` warning tags,
  `proposed_role`.

**Hallucination guards.** Layer 1 (prevention via JSON schema), Layer 2
(Pydantic retry-on-validation-error), Layer 3 (every referenced column must
exist in the profile).

**TUI surface.** A spinner while the LLM is generating. After: a column table
showing role + temporal-position assignments, plus `watch_for` warning chips.

#### Stage 1d — Flag emission (`data/flags.py`)

Deterministic emitter — has **veto** authority over the LLM (PDD §15.1).
Looks at the profile + treatment/outcome hints:

- Treatment-type flag (binary, categorical, continuous, mixture).
- Outcome-type flag (binary, continuous, count, censored).
- Structural: `SMALL_SAMPLE` (n<200), `HIGH_DIMENSIONAL` (p²>n),
  `HEAVY_MISSINGNESS` (>20% on any column), `HEAVY_CENSORING` (>70%),
  `POSITIVITY_VIOLATION` (set later by Step 5/7 overlap diagnostic).

**TUI surface.** A row of flag chips at the top of the discovery panel.

#### Stage 1e — Domain expert (`discovery/expert.py`)

**Routed to the reasoning / hypothesize slot** (deepseek-r1, qwq, qwen3:thinking).
Synthesizes across all columns. Receives:

- Full investigator report + Stage 1b profile + temporal lattice +
  correlation clusters + duplicate count + string formats + per-domain
  pitfall list.

Emits a `DomainExpertBrief`:

- One-paragraph `domain_summary`.
- Ranked `treatments` with suitability scores + typical questions.
- Ranked `outcomes` with measurement/censoring notes.
- `confounders` per (treatment, outcome) pair.
- `mediators` and `effect_modifiers` (distinguished from confounders).
- `unmeasured_confounders` with name, reason, observed proxies.
- K candidate DAGs (default K=3), each as an edge list with rationale.
- `identification_warnings` (domain-specific identification hazards).

**Hallucination guards.**

- Layer 3: every referenced column must appear in the investigator report.
- Layer 4: every claimed confounder gets a marginal-association test on the
  data; verdicts (`supported | contradicted | inconclusive`) surface to the
  analyst without overriding the LLM.
- DAG edge audit (`audit_dag_edges`): partial correlation under the
  remaining-parents conditioning set; supported/contradicted/inconclusive.

**TUI surface.** A multi-panel layout:
- Brief panel (Markdown render of `domain_summary`).
- Treatment/outcome ranked lists.
- Confounder table with audit verdicts beside each row.
- Candidate-DAG carousel with graphviz render.
- Identification-warning list (yellow chips).
- Audit summary: "N edges contradicted by data".

### Phase 2 — Feasibility filter (Week 4, `feasibility/`)

MDE calculator per data-flag combo (binary-ATE, continuous-ATE,
subgroup-CATE). Modes:

- `default` (statistician-set thresholds).
- `manual` (analyst overrides).
- `llm_calibrated` (LLM proposes thresholds based on domain).

Outputs `FeasibilityReport` with `admissible_pairs` (treatment, outcome)
that have power ≥ target.

**TUI surface.** A grid: rows = treatment candidates, columns = outcomes,
each cell colored green/yellow/red by power.

### Phase 3 — Hypothesis generation (Week 4, `hypothesize/`)

Three modes:

- `manual`: analyst writes hypotheses directly.
- `automated`: LLM (reasoning slot) proposes ranked hypotheses with
  counterfactual share.
- `hybrid`: LLM critiques + extends manual hypotheses.

Outputs `HypothesisQueue` — ranked, scoped, with impact scores.

**TUI surface.** Ranked list with `[priority] | hypothesis | rationale |
estimand`. Analyst can pin / unpin / re-rank interactively.

### Phase 4 — Per-hypothesis Causal Roadmap walk

Eight steps per hypothesis. Each step is a distinct submodule in
`roadmap/q1_question.py` … `roadmap/q8_interpret.py`. Steps 1–4, 8 may invoke
the LLM (always with schema guards). Steps 5–7 are purely statistical.

**Step 5 — Identifiability (`q5_identify.py`).** DoWhy's `identify_effect`
on the chosen CausalGraph. Returns:

- `identifiable: bool` — **hard gate**, blocks Step 7 by default.
- `strategy`: `backdoor | frontdoor | iv | mediation | non-identifiable | unsupported`.
- `adjustment_set`, `instrument`, `mediator` (when applicable).
- `estimand_expression` (do-calculus form).

**Step 6 — Statistical estimand (`q6_statistical_estimand.py`, Week 4).**
Translate to canonical functional + adjustment set.

**Step 7 — Estimate (`q7_estimate.py`).** This is where it all converges:

```
                df ── auto_preprocess ──┐
                                        │
   adjustment_set ── select_variables ──┤
   (post-double / lasso-intersection /  │
    correlation-pruning / none / auto)  │
                                        ▼
                        ┌── overlap_summary ──┐
                        │  (positivity check, │
                        │   balance diagnostic)│
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌── select_estimator ──┐
                        │   (auto cascade or   │
                        │    user `prefer=`)   │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌──── estimator.fit() ────┐
                        │  LinearDML / Forest /   │
                        │  Sparse / Meta / BART   │
                        │  with SuperLearner      │
                        │  stacked nuisance       │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌── estimator.estimate() ──┐
                        │   EstimationResult       │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌── refutations ──┐
                        │ placebo / RCC /  │
                        │ subset bootstrap │
                        └──────────────────┘
```

**TUI surface.** A vertical waterfall — each stage emits a small badge
(green/yellow/red), and the final card shows:

- Estimator id, family, backend version.
- Point estimate, 95% CI, p-value.
- Adjustment set used (after selection).
- Identification strategy.
- Overlap verdict (green/yellow/red) + propensity range.
- Refutation results: 3 checks, each pass/fail with deltas.

**Step 8 — Interpretation (`q8_interpret.py`).** Translates the numbers back
into the question's language. Hooks into the sensitivity verdict for the
narrative qualifier.

### Phase 5 — Sensitivity (`sensitivity/`)

Runs in parallel after Step 7. Three methods (more in v0.5):

| Method | Module | When | Scales |
|---|---|---|---|
| E-value | `evalue.py` | always for ATE | `risk_ratio`, `odds_ratio`, `hazard_ratio`, `standardized` (auto-converted to Cohen's d via outcome SD) |
| Sensemakr (partial R²) | `sensemakr_py.py` | linear-form estimates | benchmark covariates auto-pick |
| Multiverse | `multiverse.py` (Week 4) | when ≥2 estimators agree | specification-curve plot |
| Rosenbaum bounds | `rosenbaum.py` (v0.5, R) | matched analyses | Γ search |
| PyMC Bayesian | `bayes.py` (v0.5) | binary outcome | bias-parameter priors |

Verdict aggregator (`verdict.py`):

- `min` rule (default): pick the most pessimistic component.
- `average` rule: ordinal mean.
- `strict` rule: green only if every component is green.

**TUI surface.** Verdict card with color stripe + components panel. Below:
per-method detail card (E-value, robustness value, etc.).

### Phase 6 — Reporting (Week 4, `reporting/`)

Renders to HTML / PDF / Quarto / Markdown via Jinja2 templates. Includes:

- Cover card (project, tier, dataset).
- Discovery summary (column roles, flags, candidate DAGs, audit verdicts).
- Hypothesis queue.
- Per-hypothesis Roadmap walk (collapsible).
- Sensitivity verdict.
- Analyst-decision ledger (every default accepted, every override).
- Provenance section (model digests, seeds, R/Python versions, timestamps).
- Limitations & Failure Modes page (from PDD §31, ships from v0.1).

---

## 4. Persisted state — `study.causalrag.yaml`

The schema (Pydantic-validated) — every field a TUI may want to inspect:

| Field | Type | Phase that writes it |
|---|---|---|
| `name`, `version`, `created`, `updated` | str / datetime | 0 |
| `tier` | `data-scientist | academic | domain-expert` | 0 |
| `research_question` | str / None | 1 (or analyst-typed) |
| `dataset` | `DatasetSpec` (source, sha256, n_rows, n_cols, columns) | 1 |
| `discovery` | `DiscoveryReport` (columns, brief, K DAGs, flags) | 1 |
| `feasibility` | `FeasibilityReport` (admissible pairs, n_floor, alpha) | 2 |
| `hypothesis_queue` | tuple[Hypothesis] (ranked) | 3 |
| `roadmap_walks` | dict[str, RoadmapWalk] (per hypothesis id) | 4 |
| `flags` | set[DataFlag] (union of phase emissions) | 1, 2, 4 |
| `candidate_graphs` | tuple[CausalGraph] | 1, 4 |
| `selected_graph_index` | int | analyst-decided |
| `multiple_testing` | `bh | by | bonferroni | none` | analyst |
| `counterfactual_ratio` | float ∈ [0, 1] | analyst |
| `llm` | `LLMConfig` (backend, models, digests, seed, tier) | 1 |
| `decision_ledger` | tuple[Decision] | every phase |
| `overrides` | tuple[Override] | analyst |

The TUI can render any subset as a sidebar tree (think VS Code's outline
panel). Each leaf clicks through to the underlying detail view.

---

## 5. LLM model-slot routing

Three slots, with hardware-aware defaults at each tier:

| Slot | Used by | Properties |
|---|---|---|
| `discovery` | Stage 1c investigator | general-purpose, instruction-tuned, fast structured JSON. Default: qwen3:14b (T2) → qwen3:32b (T3) → llama3.3:70b (T4+) |
| `hypothesize` | Stage 1e expert, Phase 3 generator | reasoning ("thinking") model, slower but deeper. Default: deepseek-r1:14b (T2) → deepseek-r1:32b-q5 (T3) → r1:70b-distill (T4+) |
| `utility` | JSON repair, schema fix, summaries | smallest available. Default: qwen3:4b → qwen3:8b at higher tiers |

**Tier table** (PDD §16.2 — the 16B class is the floor):

| Tier | Effective VRAM | Discovery default | Hypothesize default | Quant |
|---|---|---|---|---|
| T0 | <12 GB | qwen3:4b | qwen3:8b | Q4_K_M |
| T1 | 16–32 GB RAM, no GPU | qwen3:8b | qwen3:14b | Q4_K_M |
| **T2 floor** | 12–16 GB VRAM or 32 GB unified | qwen3:14b | deepseek-r1:14b | Q4_K_M |
| T3 prosumer | 24 GB | qwen3:32b | r1:32b | Q5_K_M |
| T4 workstation | 48 GB | llama3.3:70b | r1:70b-distill | Q5_K_M |
| T5 server | 80+ GB | llama3.3:70b | r1:70b-distill | Q8_0 |

The selector falls back to an installed model when the tier default is not
pulled (substring-family match).

**TUI surface.** Doctor command should render this table with installed-✓
indicators. Estimate / discover should show which model is actively being
used in the streaming progress.

---

## 6. Four-layer hallucination guard

Every LLM call passes through (`llm/guards.py` + `OllamaClient.parse`):

| Layer | Where | Failure handling |
|---|---|---|
| 1. Prevention | `format=<json_schema>` passed to Ollama 0.4+ | structural — model can only emit JSON matching the schema |
| 2. Schema | Pydantic `model_validate` with retry-on-error | retry once with the validation error in the prompt; on persistent failure raise `SchemaValidationFailed` |
| 3. Semantic | `check_columns_exist`, `check_temporal_consistency`, `check_iv_relevance` | unknown columns / temporal violations are logged + dropped or downgraded |
| 4. Statistical | `audit_dag_edges` partial-correlation tests + confounder marginal-association tests | verdicts surfaced to analyst (`supported / contradicted / inconclusive`); LLM is NOT silently overridden |

**TUI surface.** A thin status bar at the bottom of every LLM-driven screen
showing layer compliance (Layer 1 ✓ / Layer 2 ✓ / Layer 3 ✓ / Layer 4 ⚠ if
contradictions present).

---

## 7. Estimator catalog — what's registered

```
python.dml.linear           ATE, CATE   binary or continuous treatment
python.dml.causal_forest    ATE, CATE   non-linear CATE; ≥3 modifiers, n≥500
python.dml.sparse_linear    ATE, CATE   HIGH_DIMENSIONAL flag
python.meta.t_learner       ATE, CATE   binary treatment
python.meta.s_learner       ATE, CATE   binary treatment, biased toward null
python.meta.x_learner       ATE, CATE   rare treatment (prevalence <15%)
python.dr.dr_learner        ATE, CATE   doubly-robust
python.bart.dml             ATE, CATE   Bayesian credible intervals
                                        (optional `bart` extra)
```

`select_estimator()` auto-cascade (see `estimators/python/select.py`):

1. HIGH_DIMENSIONAL → `sparse_linear`
2. ≥3 modifiers + n≥500 → `causal_forest`
3. SMALL_SAMPLE → `linear` (forests/learners overfit)
4. Bayesian intervals requested → `bart` (or `dr_learner` fallback)
5. Rare treatment → `x_learner`
6. Default → `linear`

User override: `--prefer <id>` (exact) or `--prefer <family>`
(dml/forest/sparse/meta/bart).

---

## 8. Nuisance-estimator (SuperLearner) library

Configurable via `nuisance_library=`:

| Library | Composition | When auto picks it |
|---|---|---|
| `single-gbm` | tuned GradientBoosting | n<500 |
| `hist-gbm` | sklearn HistGradientBoosting | HEAVY_MISSINGNESS + n<500 (native NaN handling) |
| `stacked-default` | GBM + RF + ElasticNet/Logit, NNLS meta | n≥500, no lightgbm |
| `stacked-fast` | HistGBM + RF + ElasticNet/Logit | speed-priority |
| `stacked-rich` | GBM + LightGBM + RF + EN/Logit | n≥500, lightgbm installed |
| `bart` | pymc-bart | calibrated UQ wanted |
| `auto` | resolves to one of the above | default |

**TUI surface.** Settings sub-panel under estimate: dropdown for library,
with "auto: stacked-rich" subtitle showing the resolution.

---

## 9. Variable selection methods

Configurable via `selection=`:

| Method | Mechanism | When auto picks it |
|---|---|---|
| `post_double_selection` | Belloni-Chernozhukov-Hansen 2014: Lasso(Y~W) ∪ Lasso(T~W) | HIGH_DIMENSIONAL or |W|>20 |
| `lasso_intersection` | intersection of the two Lassos (strict) | manual |
| `correlation_pruning` | drop |r|≥0.9 pairs, keep less-missing | 5 ≤ |W| ≤ 20 |
| `none` | pass-through | |W|<5 |
| `auto` | resolves to one of the above | default |

**TUI surface.** Selection-method dropdown + a "dropped variables" table
with reasons.

---

## 10. Diagnostics surfaced on every `EstimationResult`

```python
result.diagnostics = {
    "selected_estimator_id":  ...,    # which one auto-picked
    "prefer_override":        ...,    # user override id (if any)
    "identification_strategy":...,    # backdoor | frontdoor | iv | ...
    "adjustment_set_initial": [...],  # before selection
    "adjustment_set_used":    [...],  # after selection
    "preprocessing": {                # what auto_preprocess did
        "transforms": [...],          # per-column transforms
        "new_columns_from": {...},    # original -> derived (one-hot, dates)
    },
    "variable_selection": {           # selection report
        "method": "post_double_selection",
        "selected": [...],
        "dropped":  [...],
        "reasons":  {col: why_dropped, ...},
        "notes":    [...],
    },
    "overlap": {
        "positivity": {
            "verdict": "green/yellow/red",
            "note":    "...",
            "propensity_min":  0.04,
            "propensity_max":  0.96,
            "pct_extreme":     0.02,
        },
        "balance": [
            {"covariate": "age",
             "std_diff_unweighted": 0.18,
             "std_diff_weighted":   0.04,
             "imbalanced": False},
            ...
        ],
        "worst_imbalance": 0.18,
    },
    ...estimator-specific keys...
}

result.refutations = {
    "placebo_treatment":      {"original": X, "refuted": ~0, "passed": True},
    "random_common_cause":    {"original": X, "refuted": X', "passed": ...},
    "subset_bootstrap":       {"original": X, "refuted": X', "passed": ...},
    "n_passed": 3,
}
```

**TUI surface.** A two-column results layout: headline numbers on the left,
diagnostics tree on the right (collapsible). Each diagnostic key clicks
through to a detail panel.

---

## 11. Provenance — every estimate carries

- `estimator_id` (registry key)
- `estimand_class` (ATE / CATE / RMST_CONTRAST / …)
- `n_used` (after dropna)
- `backend_version` (e.g. "econml 0.16.0")
- `r_session_metadata` (for R-bridged estimators)
- `fit_seconds` (wall-clock)
- `timestamp` (UTC)

Plus, indirectly through the protocol:
- `llm.model_digest` (sha256 of the Ollama model)
- `llm.seed`, `llm.temperature`
- `llm.prompt_pack_version`

**TUI surface.** A "Provenance" footer on every results card — fixed, small,
always visible so the analyst can quote it in publications.

---

## 12. Decision points (where the analyst can intervene)

In stepwise mode, the analyst gets a prompt at each of these natural breaks:

1. **After `init`** — tier choice; not blocking but tier defaults are about
   to drive prompt verbosity.
2. **After `discover`** — confirm/override:
   - `domain_tag` (if mis-tagged).
   - Each column's `proposed_role`.
   - The selected candidate DAG (default rank 1).
   - Contradicted DAG edges (drop or keep).
3. **After `feasibility`** — accept admissible pairs or rerun with different
   thresholds.
4. **After `hypothesize`** — pin / unpin / reorder the queue.
5. **After `identify`** (Step 5) — if non-identifiable, two options:
   - Edit the DAG (returns to Step 2).
   - Override with `--allow-nonidentifiable` (recorded in `overrides`).
6. **After `estimate`** — accept or rerun with a different estimator
   (`--prefer`), selection method (`--selection`), or nuisance library
   (`--nuisance-library`).
7. **After `sensitivity`** — accept the verdict or run additional methods
   (multiverse, Bayesian sensitivity).
8. **Before `report`** — pick output format (HTML / PDF / Quarto / Markdown).

**TUI surface.** Each decision point is a modal/drawer with three buttons:
[Accept default] [Override interactively] [Skip — leave as-is].

---

## 13. Failure modes the pipeline guards against (PDD §31)

For each, the TUI should render a "guard active" badge on the relevant
phase:

| Failure mode | Mitigation surface |
|---|---|
| Causal Parrot (LLM memorizes DAG) | Layer-4 audit verdicts panel after Stage 1e |
| Post-treatment confounder bias | Temporal-order check during Stage 1c + Step 4 |
| Overconfident automation | Analyst-decision ledger rendered prominently |
| Hidden non-identifiability | Step 5 hard gate + red banner |
| LLM version drift | Model digest pinned in `llm.lock`; mismatch warning |
| R-version skew | R session metadata recorded per R-bridged estimate |
| Multiple-testing inflation | BH correction across hypotheses (default `bh`) |
| CausalProbe-style brittleness | Domain-tag-aware prompts + "LLM confidence low" banner |
| Analyst abdication | Decision ledger renders top-of-report in academic tier |

---

## 14. CLI exit codes (for TUI status mapping)

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Validation failure (corrupt YAML, etc.) |
| 2 | Missing required input (no protocol, no dataset) |
| 3 | Cassette miss when replay-only |
| 4 | LLM schema validation failed after retries |
| 5+ | Unexpected exception |

---

## 15. Known v0.1 bugs / limitations (status as of 2026-05)

These are documented because the TUI should grey-out or message-around them
rather than fail mysteriously:

- **Mixed-types preprocessing**: free-text columns containing the literal
  word "date" may be classified as `date_like` and partially decomposed
  instead of dropped. Workaround: pass `drop_free_text=True` (the default)
  and supply non-ambiguous column names.
- **Censoring-pair detection** only matches names of the form
  `<base>_days` + `<base>_event` (or similar). Other survival schemas
  require explicit `--treatment-outcome` flagging.
- **Phase 2/3/6** (`feasibility`, `hypothesize`, `report`) commands land in
  Week 4 of the §33 sprint. They are referenced throughout this doc but
  not yet implemented.
- **R bridge** (`estimators/rbridge/`) is v0.5; survival estimators
  (CSF, lmtp, tmle3, sensemakr-extended) will not appear in the candidate
  list until then.
- **DoWhy's identify_effect** may flag a non-identifiable estimand as
  identifiable through `backdoor` when the chosen DAG contains an
  unmeasured confounder as a node. The Step 5 module passes the situation
  through; the analyst must inspect `adjustment_set` to confirm none of
  the chosen variables are unobserved.

---

## 16. Where to look in the codebase

```
src/causalrag/
├── core/                  # framework primitives — Layer 1
│   ├── protocol.py        # StudyProtocol — central YAML object
│   ├── flags.py           # DataFlag StrEnum
│   ├── roles.py           # VariableRole + VariableSpec
│   ├── estimand.py        # CausalEstimand, StatisticalEstimand
│   ├── graph.py           # CausalGraph (NetworkX wrapper)
│   ├── result.py          # EstimationResult, MultiverseResult
│   └── registry.py        # estimator dispatch
├── data/                  # ingestion + profiling — Layer 2
│   ├── connectors/        # CSV, Parquet (more in v0.5)
│   ├── profiler.py        # Stage 1b
│   ├── flags.py           # Stage 1d
│   ├── features.py        # auto_preprocess (Stage 1f)
│   ├── selection.py       # post-double-selection, etc.
│   └── checks.py          # positivity, balance, overlap
├── discovery/             # Phase 1 LLM agent
│   ├── investigator.py    # Stage 1c
│   └── expert.py          # Stage 1e
├── feasibility/           # Phase 2 (Week 4)
├── hypothesize/           # Phase 3 (Week 4)
├── roadmap/               # Phase 4 — the 8 Roadmap steps
│   ├── q5_identify.py     # DoWhy identify_effect
│   └── q7_estimate.py     # full Step 7 orchestrator
├── estimators/
│   ├── base.py            # CausalEstimator Protocol
│   ├── python/
│   │   ├── dml.py         # LinearDML, CausalForestDML, SparseLinearDML
│   │   ├── meta.py        # T/S/X/DR learners
│   │   ├── bart.py        # Bayesian BART
│   │   ├── nuisance.py    # SuperLearner library
│   │   └── select.py      # auto-cascade selector
│   └── rbridge/           # v0.5 R-bridged adapters
├── sensitivity/
│   ├── evalue.py
│   ├── sensemakr_py.py
│   └── verdict.py
├── llm/
│   ├── hardware.py        # HardwareProfile + tier mapping
│   ├── selector.py        # 3-slot model selector
│   ├── cassette.py        # record/replay
│   ├── ollama_client.py   # sync httpx client + JSON-schema retry
│   └── guards.py          # 4-layer hallucination guard
├── reporting/             # Phase 6 (Week 4)
└── cli/
    ├── main.py            # Typer app
    └── doctor.py          # environment audit shim
```

Phase 6 reporting templates (Week 4) will live under `src/causalrag/reporting/templates/`.
