# Your first causal study in 5 minutes

This walks the full CausalRoadmap pipeline against the Lalonde NSW dataset
(real RCT data; experimental ATE ≈ $1,790). The same flow runs against any
tabular CSV/Parquet you point it at.

## 0. Install

```bash
pip install -e ".[dev,estimators,sensitivity]"
ollama pull qwen3:14b
ollama pull deepseek-r1:14b   # optional but recommended at T2+
```

## 1. Scaffold + sanity check

```bash
causalrag init lalonde-study --tier academic
cd lalonde-study
causalrag doctor
```

You should see a tier (T0–T5), recommended models, and any "missing models"
chips so you can `ollama pull` them.

## 2. Drop your dataset and discover

```bash
# Get the Lalonde NSW (445 rows)
python -c "from causaldata import nsw_mixtape; \
  nsw_mixtape.load_pandas().data.drop(columns=['data_id']).to_csv('data/lalonde.csv', index=False)"

causalrag discover data/lalonde.csv \
    --treatment treat --outcome re78 \
    --question "Does the NSW training program raise 1978 earnings?"
```

What this runs (PDD Phase 1):

- **1a** Connector ingestion.
- **1b** Deterministic profile per column (dtype, missingness, top values,
  Tukey outliers, censoring-pair detection, duplicate-row count).
- **1c** LLM investigator (general-purpose model) — column-by-column
  semantic meanings, temporal positions, role proposals.
- **1d** Flag emission (deterministic; vetoes LLM disagreements).
- **1e** LLM domain expert (reasoning model) — confounder / mediator /
  effect-modifier list, K=3 candidate DAGs, identification warnings.

Output: a column table, flag chips, the domain expert brief, plus a
**Layer-4 audit** that surfaces every LLM-proposed DAG edge that the data
contradicts.

## 3. Feasibility

```bash
causalrag feasibility --alpha 0.05 --power 0.80
```

Runs the power × MDE grid for every (treatment, outcome) candidate. Tags
pairs as `admissible` / `borderline` / `underpowered`.

## 4. Hypothesize

```bash
causalrag hypothesize --counterfactual-ratio 0.30
```

Ranks the admissible pairs into a hypothesis queue with impact scores.
Automated mode reuses the expert brief; pass `--mode manual` to skip the
LLM step.

## 5. Estimate

```bash
causalrag estimate --treatment treat --outcome re78
```

Walks Steps 5-7 of the Roadmap:

- **Step 5** — DoWhy `identify_effect` says backdoor / frontdoor / IV /
  non-identifiable. Non-identifiable is a **hard gate** by default;
  pass `--allow-nonidentifiable` to override (recorded in the ledger).
- **auto_preprocess** — one-hot, standardize, log-transform skewed
  outcomes, decompose dates.
- **select_variables** — post-double-selection (Belloni-Chernozhukov-
  Hansen 2014) when |W| > 20, correlation_pruning for moderate W, none
  for small W.
- **overlap_summary** — propensity overlap + balance, green/yellow/red.
- **select_estimator** — auto-picks LinearDML / CausalForestDML /
  SparseLinearDML / X-learner / DR-learner / BART based on flags + n +
  modifier count. User override: `--prefer <id-or-family>`.
- **estimator.fit** — SuperLearner-stacked nuisance, cross-fitted.
- **refutations** — placebo treatment + random common cause +
  subset bootstrap.

The output card shows estimator, strategy, adjustment set, point
estimate, 95% CI, p-value, and the refutations table.

## 6. Sensitivity

```bash
causalrag sensitivity --treatment treat --outcome re78
```

- **E-value** (VanderWeele-Ding) — how strong an unmeasured confounder on
  both arms would have to be to nullify the effect.
- **Sensemakr** (Cinelli-Hazlett) — partial-R² robustness value benchmarked
  against observed covariates.
- **Verdict** — green / yellow / red under a configurable aggregation rule
  (`min` by default — most-pessimistic wins).
- **Step 8 narrative** — a structured interpretation of the effect with
  named assumptions, magnitude in outcome units, and a robustness
  paragraph that respects the verdict color.

## 7. Report

```bash
causalrag report --format html
open reports/lalonde-study_*.html
```

A self-contained HTML file with:

- Cover card (project, tier, dataset, generated-at).
- Research question + domain brief.
- Variable spec table + flag chips + candidate DAG list.
- Feasibility summary.
- Hypothesis queue (top 10).
- Per-hypothesis Roadmap walks (collapsible) with identification, estimate
  card, refutations, Step 8 narrative.
- **BH-adjusted summary table** when multiple hypotheses were estimated
  (the headline conclusion always quotes the BH-adjusted p-value).
- Analyst-decision ledger + overrides table.
- Provenance (model digests, seed, R/Python versions, timestamps).
- "Limitations & Failure Modes" appendix (ships from v0.1 always).

## 8. TUI alternative

Prefer a terminal UI?

```bash
causalrag tui
```

`/init`, `/doctor`, `/discover`, `/feasibility`, `/hypothesize`,
`/estimate`, `/sensitivity`, `/report` — same commands, with `/` to open
the menu, `Tab` to autocomplete, `↑/↓` for history.

## What you've just done

You ran a complete Petersen-van der Laan Causal Roadmap analysis with:

1. LLM-assisted column understanding and DAG proposal — using the **best**
   reasoning model your hardware can fit.
2. **Four-layer hallucination guard** at every LLM call site (JSON-schema
   enforcement, Pydantic retry, semantic column/temporal checks,
   statistical CI audit of every proposed edge).
3. Auto-preprocessing + SuperLearner-stacked nuisance + post-double
   variable selection — defaults that match Belloni-Chernozhukov-Hansen
   and Targeted-Learning conventions.
4. Estimator auto-selection respecting `BINARY_TREATMENT`,
   `HIGH_DIMENSIONAL`, `SMALL_SAMPLE`, `RIGHT_CENSORED_OUTCOME` flags.
5. Mandatory refutations + dual-method sensitivity verdict.
6. Full provenance: every LLM call carries a model digest + seed; every
   estimate carries a backend version + adjustment set; every analyst
   decision lands in the ledger.

That's the methodological bar of a competent applied causal-inference
paper, executed in under five minutes of interactive work.

## Next reading

- `docs/PIPELINE.md` — TUI/UX design reference covering every phase,
  decision point, and persisted state field.
- The PDD (`CausalRoadmap_PDD_v0.3.pdf`) — methodology depth, especially
  §10 (Roadmap), §13 (architecture), §16 (LLM orchestration), §31
  (failure modes).
