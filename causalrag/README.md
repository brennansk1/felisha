# CausalRoadmap (Felisha)

An LLM-assisted causal-inference pipeline that drives a dataset all the way from
raw observations to a domain-aware executive synthesis — implementing the
Petersen–van der Laan Causal Roadmap end-to-end (Q1–Q8) with a deep estimator
catalog, an autonomous "master mode" that runs K experiments back-to-back, and
an audit-trail-first architecture that makes every number defensible.

Built around a few load-bearing ideas:

* **Hardware-aware local LLM orchestration** via Ollama / llama.cpp server /
  vLLM / MLX-LM (no cloud round-trips required).
* **A typed Causal Roadmap walk per hypothesis** — every estimate carries its
  identifiability proof, sensitivity verdict, anomaly audit, and refutation
  results in a single serialisable record.
* **A method catalog of 40+ estimators** spanning Python (DML linear / sparse
  / forest, BART, S/T/X/DR meta-learners, conformal ITE, distributional
  regression, hierarchical / multilevel DML, network-interference, front-door
  g-formula, proximal CI, synthetic-control SCM / ASCM / SDiD) and R
  (`grf`, `lmtp`, `MatchIt`, `WeightIt` (entropy-balancing default),
  `mediation`, `survRM2`, `bartCause`, `marginaleffects`, `sensemakr`,
  `EValue`, `tipr`, `bnlearn`, the modern DiD stack — Callaway-Sant'Anna /
  Borusyak-Jaravel-Spiess / de Chaisemartin-D'Haultfoeuille / HonestDiD,
  the `rdrobust` family, `tmle3` + `sl3` + `tmle3mediate`, TrialEmulation).
* **An autonomous master loop** that plans a prioritized candidate queue, runs
  propose-K / critique / commit with Self-Refine, supports foundation-recursion
  chains with EIG-based continuation + Thompson sampling, and auto-fires
  robustness re-runs on red sensitivity. Multi-agent debate available behind
  `--paranoid`. MCTS with progressive widening optional.
* **Domain-agnostic executive synthesis** — the final report translates findings
  into the language of whatever field the data is from (clinical / policy /
  business / ecology / engineering / marketing / operations / …).
* **Ship-gate audit** — an end-to-end flow audit verifies every flag is routed,
  every catalog estimator is reachable, every sensitivity panel surfaces in
  the report, and every brief field is consumed downstream. CI fails on regressions.

## Status — v1.0 (May 2026)

**v1.0 sprint plan is complete.** Every P0/P1 ticket from
`docs/SPRINT_PLAN_V1.md` has shipped — Sprints 1 through 9.5. **1165+ unit
tests passing, end-to-end flow audit GREEN.** Two items deliberately deferred
to v1.1 per the SEDR's own pro/con analysis (6.8 MR-RAPS for Mendelian
randomization; full DSPy compile-time optimisation against curated gold sets).

What ships in v1.0:

* **Discovery & profiling** — `causal-learn` CI back-end (RCoT / KCIT / CCIT /
  CMIknnMixed routing); Markov-boundary cross-check (single + KIAMB multi-MB
  + bootstrap stability for high-dim); five domain DAG templates (clinical
  TTE, MMM, attribution, spatiotemporal, engineering trace); Tigramite
  PCMCI+ / LPCMCI / J-PCMCI+ for time-series; missing-data diagnostic; continuous-T
  positivity (kernel density support); DAG-mismatch alerts; ICH-E9(R1)
  estimand schema with target-trial-emulation protocol; ADMG bidirected
  edges (proper latent confounder nodes); multi-mediator chains; time-varying
  / panel layering; effect-modifier topology distinct from confounder.
* **Estimator catalog** — 40+ methods (see [Catalog](#estimator-catalog) below).
* **Identification** — DoWhy primary + `ananke` / `Y0` reconcile cross-check;
  Bareinboim-Pearl transportability for population transfer; Duarte autobounds
  partial-identification fallback when point-ID fails; c-component decomposition
  for large DAGs (>10 nodes); multiverse + multiverse-of-DAGs / Bayesian
  model averaging.
* **Autonomous master loop** — Plan → propose-K → critique → commit → foundation
  recursion → synthesis. EIG-based chain continuation (Lindley 1956 /
  Chaloner-Verdinelli 1995); Thompson sampling across chain roots; Bayesian
  saturation stopping; Self-Refine / Reflexion on the critic agent; MCTS with
  progressive widening (opt-in `--master-mcts`); RAG over prior runs;
  cost-aware budget tracker; postmortem record; per-family circuit-breaker;
  multi-agent debate behind `--paranoid`.
* **Sensitivity** — unified dashboard aggregating E-value (scale-routed per
  estimator + outcome dtype), sensemakr, Chernozhukov-Cinelli-Newey OVB,
  Zhao 2019 Γ for matched designs, Rosenbaum bounds, Manski partial-ID bounds,
  refutations (SE-anchored thresholds), tipping-point auto-fire, negative-
  control falsification, anomaly audit, multiple-testing (BH/BY/Bonferroni),
  e-value closed testing, always-valid CIs (Howard-Ramdas / Waudby-Smith-Ramdas).
* **BI tasks** — root-cause attribution (DoWhy-GCM + multiply-robust), causal
  forecasting / impact (`CausalImpact` + ASCM + matrix completion), uplift /
  policy targeting (Qini / AUUC / `policytree` / `mcf`), marketing-mix
  modelling (Robyn / Meridian / PyMC-Marketing + NNLS fallback), GeoLift
  incrementality, A/B platform ingestion (Eppo / Statsig / Optimizely /
  GrowthBook), warehouse connectors via `ibis-framework` (BigQuery / Snowflake
  / Redshift / Postgres / DuckDB / Databricks).
* **Reporting** — domain-aware executive synthesis, Quarto multi-format output
  (HTML / PDF / Word), OSF + AsPredicted + Hubbard NEJM 2024 TTE
  preregistration export, Jupyter notebook export via `jupytext`,
  reproducibility manifest (`run.lock.json`) hashing data / DAG / estimand /
  RNG / git / packages / models / prompts.
* **TUI** — Textual app with multi-pane layout (chain forest / leaderboard /
  queue panel / flag chips / streaming log), in-terminal plots via
  `textual-plotext`, tutorial mode on packaged Lalonde / IHDP, study save /
  load / branch, hover-help on flag chips with full semantic descriptions,
  live elapsed-timer during long LLM calls, hint-augmented error messages.
* **LLM scaffolding** — engine abstraction over Ollama / llama.cpp server /
  vLLM / MLX-LM with auto-selection chain; EAGLE-2/3 speculative-decoding
  adapter with curated draft-model registry; three-layer cache (engine prefix
  KV / semantic / cassette replay with model-digest); shared honesty preamble
  + refusal channel; DSPy + Outlines opt-in module catalog covering all 8
  prompt sites; YAML-driven `FlagRegistry` with full metadata; hardware-tier
  map T0–T5 for May 2026 model landscape.
* **Audits** — end-to-end flow audit (the v1.0 ship gate — orphan flags,
  unreachable estimators, panels not in report, brief fields not routed);
  method-coverage matrix (1610 v1.1 ticket candidates surfaced); island
  detector for dead code.

Pipeline reaches the senior-data-scientist bar on every Roadmap step —
every number defensible, every assumption explicit, every flag routed.

## Install

```bash
pip install -e ".[dev,estimators]"
```

R bridge (optional but recommended — unlocks ~20 R-backed estimators):

```bash
# R 4.4+ required
Rscript -e 'install.packages(c(
  "grf","lmtp","MatchIt","WeightIt","marginaleffects",
  "sensemakr","mediation","survRM2","EValue","tipr",
  "bartCause","bnlearn","cobalt",
  "did","didimputation","DIDmultiplegt","HonestDiD","bacondecomp",
  "rdrobust","rdmulti","rddensity",
  "tmle3","sl3","tmle3mediate",
  "TrialEmulation","augsynth","synthdid"
), repos="https://cloud.r-project.org", type="binary")'
```

Optional Python deps unlock additional functionality:

| Package | What it enables |
|---|---|
| `ananke-causal` | Second ID engine (Tian-ID, c-component, transportability) |
| `autobounds` | Partial-identification bounds when point ID fails |
| `causaldata` | Lalonde NSW + other canonical datasets for tutorials / tests |
| `causalimpact` / `tfcausalimpact` | Bayesian structural-TS impact analysis |
| `pysyncon` | Synthetic-control / ASCM / SDiD (Python; alternative to R `augsynth`) |
| `tigramite` | Time-series causal discovery (PCMCI+ / LPCMCI / J-PCMCI+) |
| `pymc` + `pymc-bart` | Python BART path for `python.bart.dml` |
| `pymc-marketing` | MMM (Python; alternative to R `Robyn`) |
| `policytree` / `mcf` (R) | Policy-tree targeting |
| `ibis-framework[<backend>]` | Warehouse ingestion (BigQuery / Snowflake / …) |
| `dspy` + `outlines` | Compile-time prompt optimisation + grammar-constrained decoding |
| `sentence-transformers` | Semantic prompt cache fuzzy lookup |
| `plotext` / `textual-plotext` | In-terminal plots in the TUI |
| `mlx-lm` | Apple-Silicon-native LLM engine |
| `jupytext` | Notebook export companion `.py` file |

Every optional dep degrades gracefully — a missing package logs a hint
and the pipeline routes to the next-best estimator / fallback.

## Quick start (CLI)

```bash
causalrag init my_study
cd my_study
causalrag validate
causalrag doctor

# Deterministic single-pass run on a CSV
causalrag run data/cohort.csv --treatment T --outcome Y

# Autonomous master mode — LLM-driven, K experiments + foundation chains
causalrag auto data/cohort.csv \
    --experiments 10 \
    --foundation \
    --multiple-mb 3 \
    --question "What raises retention 30 days after the trial period?"

# High-dim mode for n ≪ p (oncology / brain imaging)
causalrag auto data/tcga_brca.csv --experiments 5 --high-dim-mode

# Regenerate the executive synthesis without re-running estimation
causalrag synthesize --project ./my_study

# Inspect the catalog for a specific estimator
causalrag explain --method rbridge.grf.causal_survival_forest
causalrag explain --all
```

Output:

* `study.causalrag.yaml` — every hypothesis, identification result, estimate,
  sensitivity verdict, decision-ledger entry, anomaly audit, sensitivity
  interpretation, identification narration.
* `executive_synthesis.json` (auto only) — domain-aware ranked findings.
* `run.lock.json` — reproducibility manifest (data / DAG / estimand / RNG /
  git / package / model / prompt hashes).
* `reports/<project>_<timestamp>.html` — executive summary on top, Quarto-style
  technical Roadmap walks below, sensitivity-panel chip strip per walk,
  method cards per estimator, decision ledger, identification narration,
  anomaly audit, honest caveats.
* `studies/<name>/<branch>/...` — when `causalrag study save/branch` was used.

### CLI subcommands

| Command | Phase | What it does |
|---|---|---|
| `causalrag init <name>` | 0 | Scaffold a new study directory. |
| `causalrag doctor` | 0 | Hardware probe + Ollama reachability + R-bridge inventory + registered-estimator count. |
| `causalrag validate` | 0 | Round-trip the `study.causalrag.yaml` through Pydantic to catch corruption. |
| `causalrag discover <data>` | 1 | Run Phase 1 on a CSV / URI; persist DiscoveryReport. |
| `causalrag estimate` | 4 | Walk one hypothesis through Q5-Q7. |
| `causalrag sensitivity` | 5 | Run E-value + sensemakr + multiverse on the latest estimate. |
| `causalrag run <data>` | 0–6 | Deterministic single-pass through every phase. |
| `causalrag auto <data>` | 0–6 | Autonomous master loop — see below. |
| `causalrag synthesize` | 5+ | Regenerate just the executive synthesis on an existing protocol. |
| `causalrag explain --method <id>` | — | Look up a catalog entry — use case, citations, flag triggers. |
| `causalrag explain --all` | — | Dump the full catalog table. |
| `causalrag tui [--auto]` | — | Launch the Textual app. |

### `causalrag auto` — flags

| Flag | Default | Meaning |
|---|---|---|
| `--experiments K` / `-K K` | 5 | Total successful experiments. Hard cap. |
| `--foundation` | off | Enable foundation-recursion follow-ups (CATE-on-modifier, mediator decomp, red-sensitivity robustness child). |
| `--max-foundation-iterations N` | 8 | Cumulative budget across all chains. |
| `--max-foundation-depth D` | 4 | Longest chain depth a single foundation thread can reach. |
| `--queue-size N` | 18 | Candidates the up-front planner enumerates. |
| `--propose-k N` | 3 | Critic reviews the top-K candidates each turn. |
| `--critic/--no-critic` | on | Enable / disable the propose-K critic. |
| `--multiple-mb N` | 1 | Discover up to N distinct Markov boundaries per target (Phase 2 of the MB roadmap). |
| `--high-dim-mode` | off | Stability subsampling + iamb.fdr for the MB layer. For n ≪ p. |
| `--mb-bootstraps N` | 20 | Bootstrap iterations under `--high-dim-mode`. |
| `--mb-stability F` | 0.6 | Selection-frequency threshold in stability subsampling. |
| `--question "..."` | none | Research question to seed discovery. |
| `--base-url URL` | `http://127.0.0.1:11434` | Ollama / engine endpoint. |

## TUI

```bash
causalrag tui            # default layout
causalrag tui --auto     # mounts queue + chain-forest side panels
```

A Textual terminal app with a phase tracker, a streaming log, a slash-command
composer, and optional side panels.

### Slash commands

| Command | Phase | What it does |
|---|---|---|
| `/init <name>` | 0 | Scaffold a new study. |
| `/doctor` | 0 | Diagnostic (env + Ollama + R + hardware + catalog). |
| `/discover <data>` | 1 | Profile → investigator → expert brief → DAG audit. |
| `/feasibility` | 2 | Power × MDE grid over admissible (T, Y) pairs. |
| `/hypothesize` | 3 | Manual or `--mode master` LLM-enumerated hypothesis queue. |
| `/estimate` | 4 | Q5 → Q6 → Q7 walk; estimator picked by rule cascade or `--prefer`. |
| `/sensitivity` | 5 | E-value, sensemakr, multiverse, Rosenbaum, Manski, tipping-point. |
| `/report` | 6 | Render HTML / Markdown / Quarto. |
| `/run <data>` | 0–6 | One-shot deterministic pipeline. |
| **`/auto run <data> --experiments K [--foundation]`** | 0–6 | **Autonomous master loop.** |
| `/synthesize` | 5+ | Regenerate the executive synthesis on the current protocol. |
| `/explain <method>` | — | Show the catalog entry for an estimator id. |
| `/study save <name>` | — | Snapshot the current project as a named study. |
| `/study load <name>` | — | Restore a saved study. |
| `/study branch <new>` | — | Branch the current study for what-if exploration. |
| `/layout [show\|hide\|queue\|chains]` | — | Toggle `--auto` side panels. |
| `/help`, `/?` | — | List commands. |
| `/clear` | — | Clear the log. |
| `/quit` | — | Exit. |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `/` | Open the slash-command menu. |
| `Tab` | Autocomplete slash name **or** file-path argument (for `/init`, `/discover`, `/run`, `/auto`). |
| `↑` / `↓` | History — on an empty arg slot, recalls the last value used for that command. |
| `Ctrl-K` | Focus the input. |
| `Ctrl-L` | Clear the log. |
| `Ctrl-G` | Scroll the log to bottom. |
| `Ctrl-T` | Toggle `--auto` side panels. |
| `Ctrl-C` | Quit. |

While a worker is running the hint strip shows a live `elapsed Ns` counter so
long LLM calls don't look frozen. Errors get a one-line recovery hint when the
failure matches a known pattern (Ollama down → `ollama serve`; dataset missing
→ path check; invalid LLM JSON → `--no-cache`; etc.).

### Side panels (`--auto` mode)

* **Candidate queue panel** — top-5 candidates with deterministic score
  components (impact × identifiability × power × novelty − cost); rows
  strike-through as candidates complete.
* **Chain forest panel** — completed walks grouped by `chain_id`, indented
  by depth, with sensitivity-verdict glyph (●/◐/○).
* **Leaderboard** — when multiple estimators run on the same hypothesis
  (typical after a red-sensitivity robustness swap), they appear side-by-side
  with point / SE / CI / verdict / energy score / ERUPT.
* **Flag chip bar** — top-of-screen chips with hover tooltips showing each
  flag's semantic meaning, implication, and routes.

### Tutorial mode

```bash
causalrag tui --tutorial lalonde   # walks Lalonde NSW step-by-step
causalrag tui --tutorial ihdp      # IHDP semi-synthetic CATE benchmark
```

Each tutorial steps through `init → discover → hypothesize → estimate →
sensitivity → report` with inline hint cards.

## `/auto` — the master loop in detail

```bash
/auto run data/cohort.csv \
    --experiments 10 \
    --foundation \
    --max-foundation-iterations 8 \
    --max-foundation-depth 4 \
    --multiple-mb 3
```

What the loop does each turn:

1. **Plan** — one LLM call enumerates 15–30 credible candidates. A deterministic
   scorer ranks:

   ```
   score = 0.40·impact + 0.25·identifiability + 0.20·power_proxy + 0.15·novelty − cost
   ```

   `identifiability` is the strength of the available identification strategy
   (backdoor with named confounders / IV / front-door / nothing). `power_proxy`
   blends the LLM hint with the sample-size band. `novelty` penalises
   already-tested (T, Y, estimand) triples. `cost` mildly downweights expensive
   estimators (forests / BART).

2. **Dedupe** — exact and near-duplicate candidates merged via a deterministic
   pre-pass + optional LLM refinement.

3. **Propose-K → critique → commit** — top-K candidates go to a critic agent
   (already-tested guard, identifiability sniff-test, catalog-validity check,
   min-n check). Rejected candidates are vetoed; the highest-scoring survivor
   runs. Self-Refine fires on a critic rejection — the planner gets a
   structured reflection and can revise instead of re-proposing.

4. **Roadmap walk** — chosen experiment walks Q5 (identify with collider /
   descendant / mediator filter + ananke cross-check) → Q6 (statistical
   estimand) → Q7 (estimate with cross-fitting + SE-anchored refutations) →
   Q8 (E-value on the correct scale + sensemakr + Zhao Γ when matching +
   anomaly audit + tipping-point on yellow/red + negative-control scan).

5. **Foundation firing** (when `--foundation`) — deterministic rule:

   ```
   fire iff
       parent.sensitivity_verdict not in {errored, unknown}
     AND |parent.point/SE| > 1.96
     AND chain.depth < max_foundation_depth
     AND chain.null_streak < 2
     AND chain.info_gain_streak_below_eps < 2
     AND foundation_iterations_used < max_foundation_iterations
   ```

   EIG (Lindley) + Thompson sampling pick which chain to drill into next when
   multiple are alive. Chains are tracked per-`chain_id`; interleaved
   independent experiments do NOT reset another chain's depth.

6. **Auto-robustness on red** — when sensitivity comes back RED, the loop
   *automatically* schedules a robustness re-run with a different estimator
   family (DML → WeightIt; forest → DML linear; etc.). User does not have
   to ask.

7. **Recovery** — estimator-swap retry on fit failure; unidentifiable
   proposals capture the missing piece (instrument? mediator? descendant in
   adjustment set?) and surface it to the next critic call; circuit-breaker
   blacklists a family after three consecutive failures.

8. **Multiple-testing** — BH / BY / Bonferroni applied across the loop's
   p-values before synthesis. Adjusted p-values quoted in the synthesis prompt.

9. **Cross-experiment analysis** — surface contradictions, reinforcements,
   and chain narratives before the synthesis call.

10. **Domain-aware synthesis** — reasoning LLM infers the domain (clinical /
    business / policy / ecology / engineering / education / marketing /
    operations / social science / physical science / other) and writes findings
    in that field's vocabulary. Deterministic confidence override forces
    `confidence="low"` if CI crosses zero / sensitivity is red / n_used < 100.

## Pipeline architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1 — discovery                                                    │
│   profile → investigator (LLM) → expert brief (LLM) → candidate DAGs   │
│   → Layer-4 CI audit (causal-learn) → DataFlag detection               │
│   → Markov boundary cross-check → DAG-conflict report                  │
│   → missingness diagnostic → continuous-T positivity                   │
├────────────────────────────────────────────────────────────────────────┤
│ Phase 2 — feasibility (power × identifiability × sample-size grid)     │
├────────────────────────────────────────────────────────────────────────┤
│ Phase 3 — hypothesize (manual / automated / master-mode queue)         │
│   → dedupe → propose-K critique commit → Self-Refine on reject        │
├────────────────────────────────────────────────────────────────────────┤
│ Phase 4 — Causal Roadmap walks                                         │
│   Q5 identify (DoWhy + ananke reconcile, collider/descendant guard)    │
│   → autobounds partial-ID fallback if point-ID fails                   │
│   → Q5 narration (LLM)                                                 │
│   Q6 statistical estimand                                              │
│   Q7 estimate (catalog dispatch, cross-fitting) + SE-anchored          │
│       refutations + tipping-point + negative-control scan              │
│   Q8 interpret + sensitivity dashboard + anomaly audit                 │
├────────────────────────────────────────────────────────────────────────┤
│ Phase 5 — multiple-testing adjustment + cross-experiment analysis      │
│   → contradictions / reinforcements / chain narratives                 │
├────────────────────────────────────────────────────────────────────────┤
│ Phase 6 — executive synthesis (domain-aware) + Quarto / HTML / MD      │
│   → method cards / preregistration export / notebook export           │
│   → run.lock.json reproducibility manifest                             │
└────────────────────────────────────────────────────────────────────────┘
```

## Repository layout

```
src/causalrag/
├── core/                 # StudyProtocol, RoadmapWalk, EstimationResult,
│                           Decision ledger, CausalGraph (with bidirected),
│                           CausalEstimand (ICH-E9 fields), DataFlag enum,
│                           flag_descriptions, flag_registry (+ YAML),
│                           dag_constructors, effect_modifier_topology,
│                           temporal_graph, interference, registry
├── data/                 # ingestion + profiling + DataFlag detectors
│   ├── connectors/       # csv, parquet, duckdb, sql, feather, json, excel,
│   │                       ibis_backend (BigQuery/Snowflake/Postgres/…)
│   ├── flags.py          # detectors per flag
│   ├── missingness.py    # MICE / IPCW / refuse recommendation
│   ├── ab_platforms.py   # Eppo / Statsig / Optimizely / GrowthBook
│   └── checks.py         # positivity (binary + continuous-T)
├── discovery/            # Stage-1 LLM agents + DAG templates + MB
│   ├── investigator.py
│   ├── expert.py
│   ├── markov_boundary.py
│   ├── dag_conflicts.py
│   ├── ci_backend.py     # causal-learn router
│   ├── timeseries_cd.py  # Tigramite wrapper
│   └── dag_templates/    # clinical_tte, mmm, attribution,
│                           spatiotemporal, engineering_trace
├── feasibility/          # power × MDE simulation
├── hypothesize/          # manual + master mode + dedupe
├── identify/             # decomposition, autobounds_bridge,
│                           ananke_bridge, transportability
├── roadmap/              # Q5 identify, Q6 statistical estimand,
│                           Q7 estimate + refute, Q8 interpret,
│                           identification_narration
├── estimators/           # method catalog
│   ├── python/           # OLS, DML, BART, meta-learners, conformal ITE,
│   │                       hierarchical, interference, frontdoor, proximal,
│   │                       distributional, synthetic_control
│   ├── rbridge/          # grf, lmtp, MatchIt, WeightIt, mediation, survRM2,
│   │                       bartCause, marginaleffects, bnlearn, did_modern,
│   │                       rd, tmle3, trial_emulation, discovery_r, sensitivity_r
│   ├── catalog.py
│   ├── causaltune_select.py    # energy score + ERUPT
│   ├── learned_router.py       # XGBoost / GBM router
│   └── success_classifier.py   # per-estimator success classifier
├── sensitivity/          # evalue (scale-routed), sensemakr_py, verdict,
│                           multiple_testing, anomaly_audit, interpretation,
│                           zhao_value, sequential (anytime-valid CIs),
│                           evalue_closed, dashboard
├── multiverse/           # specification curve (specr), dag_bma
├── master_loop.py        # autonomous propose-K / critique / commit / synth
├── loop_observability/   # postmortem, circuit_breaker, budget
├── loop_scoring/         # eig (Lindley), bandit (Thompson + UCB1), mcts
├── llm/                  # ollama_client, honesty, self_refine, multi_agent,
│   │                       cache, spec_decoding, dspy_modules, rag_history,
│   │                       hardware_tiers
│   └── engines/          # ollama, llamacpp, vllm, mlx adapters
├── reporting/            # synthesis, render_html, render_md, quarto,
│                           preregister, notebook_export, cross_experiment
├── tasks/                # rca, impact, uplift, mmm, geolift
├── tui/                  # Textual app, widgets/, tutorial.py, errors.py,
│                           completion.py
├── audits/               # end_to_end_flow, method_coverage, island_detector
├── provenance/           # manifest.py (run.lock.json builder)
└── cli/                  # Typer app + studies.py (save/load/branch)
```

## Estimator catalog

`causalrag explain --method <id>` shows the canonical use case + citations
for any registered method. `causalrag explain --all` dumps the full table.

A representative sample:

| Estimator id | Use case |
|---|---|
| `python.linear.ols` | HC3 OLS — small-sample honest default. |
| `python.dml.linear` | DML linear final stage — defensible ATE default. |
| `python.dml.causal_forest` | EconML CausalForestDML for HTE / CATE. |
| `python.dml.sparse_linear` | Lasso final stage for high-dim adjustment. |
| `python.bart.dml` | Bayesian causal forest via PyMC-BART. |
| `python.dr.dr_learner` | EconML DRLearner with cross-fit (cv=5). |
| `python.meta.{s,t,x}_learner` | Meta-learners with bootstrap CIs. |
| `python.conformal.ite` | Lei-Candès weighted-conformal CATE intervals. |
| `python.hierarchical.dml` | Cluster-aware DML, cluster-robust SE. |
| `python.interference.aronow_samii` | Partial-interference direct effect. |
| `python.interference.savje` | Sävje-Aronow-Hudgens general interference. |
| `python.frontdoor` | Pearl front-door g-formula + bootstrap. |
| `python.proximal.regression` | Liu-Tchetgen-Tchetgen 2024 two-stage PCI. |
| `python.synth_control.{scm,ascm,sdid}` | Synthetic control / ASCM / SDiD. |
| `python.firpo.rif_quantile` | Firpo 2007 unconditional-quantile partial effect. |
| `python.cfvm.counterfactual_dist` | Chernozhukov-Fernández-Val-Melly counterfactual distributions. |
| `python.dfl.reweighting` | DiNardo-Fortin-Lemieux distributional reweighting. |
| `rbridge.grf.causal_forest` | Reference `grf::causal_forest`. |
| `rbridge.grf.causal_survival_forest` | Survival CATE (Cui-Athey-Tibshirani 2023). |
| `rbridge.grf.instrumental_forest` | IV-CATE with partial-F first-stage diagnostic. |
| `rbridge.grf.multi_arm_causal_forest` | Multi-arm treatment. |
| `rbridge.lmtp.{shift,policy,mixture,sdr,contrast}` | Stochastic interventions / MTP. |
| `rbridge.matchit` | MatchIt + `marginaleffects` post-match g-comp. |
| `rbridge.weightit` | Propensity weighting (EBAL default — Zhao 2017 doubly-robust). |
| `rbridge.bartcause` | Bayesian causal forest with calibrated posterior CIs. |
| `rbridge.mediation` | NDE / NIE via `mediation`. |
| `rbridge.survrm2` | RMST contrast. |
| `rbridge.marginaleffects.slopes` | Continuous-T marginal slopes. |
| `rbridge.tmle3` | Targeted MLE (van der Laan-Rose) via `tmle3` + `sl3`. |
| `rbridge.tmle3.mediation` | NDE / NIE via `tmle3mediate`. |
| `rbridge.did_modern.callaway_santanna` | Modern staggered DiD. |
| `rbridge.did_modern.bjs_imputation` | Borusyak-Jaravel-Spiess imputation DiD. |
| `rbridge.did_modern.dCDH` | de Chaisemartin-D'Haultfoeuille negative-weight diagnostics. |
| `rbridge.did_modern.honest_did` | HonestDiD parallel-trends robustness. |
| `rbridge.rd.rdrobust` | Calonico-Cattaneo-Titiunik sharp / fuzzy RDD. |
| `rbridge.trial_emulation` | TrialEmulation ITT / PP / as-treated. |

The dispatch rule cascade (`estimators/python/select.py::_rule_cascade`) picks
the right estimator from this catalog given the active flag set + estimand
class. The Sprint 9.5.2 method-coverage audit confirms every (estimand × flag
combination) is reachable from at least one rule path.

## Audits — the v1.0 ship gate

```bash
python -c "from causalrag.audits.end_to_end_flow import audit_pipeline_flow; print(audit_pipeline_flow())"
```

Three audits run statically (no pipeline execution needed):

* **`audits/end_to_end_flow.py`** — walks the directed graph
  `discovery_signal → DataFlag → router → estimator → sensitivity panel →
  synthesis prompt → HTML report`. Flags orphaned producers, orphaned
  consumers, unreachable estimators, sensitivity panels missing from
  synthesis / HTML, brief fields not routed downstream. **Severity GREEN
  on the v1.0 pipeline.**
* **`audits/method_coverage.py`** — sparse matrix of (estimand × flag
  combination) → reachable estimators. Empty cells become v1.1 ticket
  candidates. Current pipeline: 2205 cells, 595 covered (27%), 1610
  candidates surfaced.
* **`audits/island_detector.py`** — AST + grep static analysis for
  modules / functions defined but never referenced (dead code), or only
  referenced from tests (test-only).

Run them in CI to fail the build on regressions.

## Design principles

* **Honest provenance.** Every LLM call, every estimator decision, every
  refutation result lives on the protocol with a Decision-ledger entry
  carrying timestamp + source + rationale. The `run.lock.json` manifest
  hashes data + DAG + estimand + RNG + git SHA + package versions + model
  digests + prompt digests — reproducibility derives entirely from those
  artifacts.
* **Refuse gracefully.** The pipeline never silently substitutes a worse
  estimator or fabricates an identifiability claim. Failures are captured
  with a reason (`failure_reason`) and surfaced to the next critic call so
  the LLM can adapt. The "errored" sensitivity verdict is distinct from
  "red".
* **Domain agnosticism.** The synthesis layer infers the audience from the
  data — a clinical dataset produces care-pathway implications, a sales
  dataset produces operator actions, an ecology dataset produces follow-up
  study designs. The math underneath is the same; only the vocabulary
  differs.
* **Senior-statistician standards.** Refutation pass/fail thresholds are
  SE-anchored. IV first-stage uses partial F, not Kendall's tau. E-value
  is computed on the right scale for the estimator + outcome dtype.
  Colliders, descendants, and mediators are programmatically excluded
  from adjustment sets. DiD picks the right modern variant from the
  Roth-Sant'Anna 2025 decision tree.
* **Audit-first.** The Sprint 9.5 ship-gate audit must be GREEN before any
  release. Adding a new flag without a detector / route / description
  fails CI. Adding a new estimator without a rule path fails CI.

## Tests

```bash
.venv/bin/python -m pytest tests/unit -q
```

**1165+ unit tests passing.** ~30 skipped for optional R packages /
pymc / autobounds / ananke / mlx_lm / plotext / dspy / outlines.

Tests cover the estimator catalog, identification gate (both DoWhy and
ananke paths), sensitivity dashboard (all 9 panels), refutation thresholds,
flag detection, synthesis robustness with fabricated-id validation,
multiple-testing adjustment, master-loop helpers (EIG / Thompson / chain
bookkeeping / dedupe / circuit-breaker), every BI task, every multiverse
module, every audit, every TUI widget.

Integration tests under `tests/integration/` include a fake-LLM end-to-end
master loop (`RUN_FAKE_LOOP_INTEGRATION=1`), parrot diagnostic
harness (`RUN_PARROT_TESTS=1`), and the lalonde / IHDP / ACIC / m-bias /
high-dim / mixed-types synthetic-dataset benchmarks.

## Further reading

* `docs/SPRINT_PLAN_V1.md` — canonical v1.0 roadmap with every shipped
  ticket and the deferred-to-v1.1 candidates.
* `docs/PARROT_TEST.md` — design + how-to-run for the sign-flip / non-
  canonical-dataset parrot diagnostic.
* `docs/PIPELINE.md` — phase-by-phase developer notes.
* `docs/TUTORIAL.md` — walkthrough of running on a packaged dataset.
* `CHANGELOG.md` (repo root) — round-by-round shipping history.
* `CausalRoadmap_PDD_v0.3.pdf` (repo root) — the product design document
  (full architecture §13, original four-week sprint plan §33, methodology
  references).

## Citation

If you use this pipeline in academic work, cite the underlying methods.
The `causalrag explain --method <id>` output lists the canonical references
per estimator (Chernozhukov et al. 2018, van der Laan-Rose, Callaway-Sant'Anna,
Calonico-Cattaneo-Titiunik, Pearl, Bareinboim, Imbens-Rubin, …) and the
method cards in the HTML report carry the same.

The pipeline itself is provisionally cited as: **CausalRoadmap / Felisha v1.0,
2026, https://github.com/brennansk1/felisha**.

## License

Internal / TBD.
