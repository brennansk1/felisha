# CausalRoadmap v1.0 — Sprint Plan

Canonical roadmap from the May 2026 senior-engineering design review (SEDR)
plus the complex-/large-DAG addendum (§ Sprint 6.5, below). Supersedes the
original four-week §33 sprint plan, which delivered the bootstrap.

The guiding philosophy: **CausalRoadmap should behave like a PhD-level
causal-inference data scientist who is also disciplined about
reproducibility and computation.**

* Prefer doubly-robust, cross-fit, target-parameter-explicit estimators
  over single-equation regression.
* Refuse to "answer" without an explicit estimand, identification proof,
  and falsification step.
* Record everything (data hash, RNG, model digest, prompt, prior) so a
  reviewer can reproduce or contest any number.
* Be opinionated where the literature is, and humble where it is not.

Priorities: **P0 = required for credible v1.0**, **P1 = best-in-class
stretch**, **P2 = v1.1 follow-up**.

---

## Status — May 2026

* Bootstrap (original §33 plan) — **complete**. Discovery, R bridge,
  master loop, domain-aware synthesis, MB layer all shipped. 383 unit
  tests passing.
* SEDR — **received May 14**. Captured here. Implementation starts
  Sprint 1.
* Complex-DAG addendum — **new in this plan**. The SEDR enumerates
  estimators and identification engines but does not address how the
  pipeline handles complex topologies (latents as bidirected edges,
  multi-mediator chains, panel layering, network interference,
  large DAGs > 50 nodes, domain-specific DAG templates). Sprint 6.5
  closes this.

---

## Sprint 1 — Foundations (P0)

Goal: every number Felisha produces is reproducible, every flag is
audited, and the LLM scaffolding is compiled rather than templated.

| Ticket | Component | Scope |
|---|---|---|
| 1.1 | `core/flags/registry.py` | YAML-driven `FlagRegistry` with parent/child hierarchy, implication closure, confidence + provenance, deprecation policy. Backward-compatible `StrEnum` generation so existing code still compiles. Third parties register via `causalroadmap.flags` entry points. |
| 1.2 | `provenance/manifest.py` | `run.lock.json` capturing data Merkle hash, DAG hash, estimand hash, RNG seeds, code SHA, lockfiles, model digests, prompt hashes (incl. DSPy compiled-prompt hashes). |
| 1.3 | `discovery/ci_backend.py` | `causal-learn` CI module integration (RCoT default, KCIT for small n, CCIT for mixed-types, CMIknnMixed for TS). Replaces ad-hoc partial-correlation tests in `discovery/markov_boundary.py`. |
| 1.4 | `reporting/quarto.py` | Quarto template (HTML + PDF) that embeds sensemakr contour, love plot, propensity overlap, CATE calibration, method cards, and the executive synthesis. |
| 1.5 | `reporting/preregister.py` | OSF JSON + AsPredicted markdown export of the target-trial protocol. |
| 1.6 | `core/estimand.py` | ICH-E9(R1) fields on `CausalEstimand`: population, endpoint, intercurrent-event strategy (treatment-policy / composite / hypothetical / principal-stratum / while-on-treatment), summary measure, treatment-condition. Target-trial-emulation 7 elements (eligibility / strategies / assignment / follow-up / outcome / contrast / analysis plan) as a Pydantic schema. |
| 1.7 | `llm/dspy_module.py` | DSPy + Outlines integration for planner / critic / foundation-followup. Compile against a 50–100 example gold set. Schema-repair sub-agent for hard parse failures. Semantic prompt cache. |
| 1.8 | `master_loop/postmortem.py` | Structured `PostmortemRecord` written to the manifest on abnormal termination; per-estimator-family circuit breaker. |

**Exit criteria:** every existing `/auto` run produces a `run.lock.json`
that can be replayed bit-identically; flag system is YAML-driven; one
Quarto report renders end-to-end.

---

## Sprint 2 — Estimator catalog v1.0 (P0)

Goal: close the largest gaps against the modern consensus.

| Ticket | Component | Scope |
|---|---|---|
| 2.1 | `estimators/rbridge/tmle3.py` | Full `tlverse` integration: `tmle3` + `sl3` + `tmle3mediate`. Doubly-robust ATE/ATT, mediation, longitudinal. |
| 2.2 | `estimators/rbridge/did.py` | Modern DiD stack: `did::att_gt` (Callaway-Sant'Anna), `didimputation` (Borusyak-Jaravel-Spiess), `DIDmultiplegt` (de Chaisemartin-D'Haultfoeuille), `bacondecomp` (Goodman-Bacon), `HonestDiD` (Rambachan-Roth). Opinionated "which DiD when" dispatch (decision tree below). |
| 2.3 | `estimators/rbridge/synthcontrol.py` | `pysyncon` (SCM, Ridge ASCM, SDiD); R-bridge to `augsynth` for staggered ASCM; `synthdid` for Arkhangelsky et al. |
| 2.4 | `estimators/rbridge/rd.py` | `rdrobust`, `rdbwselect`, `rdmulti`, `rddensity` — Calonico-Cattaneo-Titiunik stack. |
| 2.5 | `estimators/python/conformal_ite.py` | Lei-Candès weighted conformal + Jonkers CCT-learner 2024 + Alaa-Ahmad-van der Laan conformal meta-learners (NeurIPS 2023). |
| 2.6 | `sensitivity/dashboard.py` | sensemakr contour, Chernozhukov-Cinelli-Newey OVB (via DoWhy 0.12), Rosenbaum Γ for matching runs, Manski bounds. |
| 2.7 | `estimators/select.py` | Adopt **CausalTune `energy_score` / ERUPT** as the default cross-validation metric for ATE / policy-value selection. |
| **2.8** | `identify/autobounds_bridge.py` | **NEW — Duarte autobounds partial-identification engine (JASA 2024).** When DoWhy reports non-identifiable and both T and Y are discrete (≤ ~10 categories), invoke `autobounds` to return informative bounds rather than refusing. New flag `POINT_ID_FAILED_DISCRETE` routes here. Cap DAG size at 10 nodes for v1.0. The data-fusion capability (observational + experimental + prior-study summaries) is deferred to v1.1. |
| **2.9** | `estimators/rbridge/weighting.py` | **NEW (default flip).** Set entropy balancing (`method="ebal"`) as the WeightIt default per Zhao 2017 (entropy balancing is doubly robust). Update the method card. Zero new code beyond the one-line default change + a citation paragraph. |

**"Which DiD when" decision tree** (consensus from Roth-Sant'Anna 2025
Practitioner's Guide):

```
if STAGGERED_ADOPTION and NEVER_TREATED_AVAILABLE:
    primary   = Callaway-Sant'Anna (did::att_gt)
    secondary = Borusyak-Jaravel-Spiess (didimputation)
    diagnostic = Goodman-Bacon decomp + Sun-Abraham IW
elif STAGGERED_ADOPTION and not NEVER_TREATED_AVAILABLE:
    primary = de Chaisemartin-D'Haultfoeuille (DIDmultiplegt)
    report negative-weight share
elif CONTINUOUS_TREATMENT and STAGGERED_ADOPTION:
    primary = Callaway-Goodman-Bacon-Sant'Anna 2024 (continuous staggered)
elif TWO_PERIODS:
    primary = doubly-robust DiD (Sant'Anna-Zhao DRDID)
ALL: report Rambachan-Roth 2023 honest parallel-trends CI, NOT a pretest p-value.
```

**Exit criteria:** tmle3 is wired and a binary-T continuous-Y demo
recovers the known ATE within 1 SE. DiD/RD/SCM stacks each have a
smoke test on a packaged dataset (Card-Krueger / Lalonde / Abadie-
Diamond-Hainmueller California Prop-99).

---

## Sprint 3 — Auto-mode upgrades (P0)

Goal: the master loop becomes principled rather than heuristic.

| Ticket | Component | Scope |
|---|---|---|
| 3.1 | `master_loop/eig.py` | Expected-Information-Gain (Lindley 1956 / Chaloner-Verdinelli 1995) replaces `info_gain_streak_below_eps`. Approximate EIG = ½·log(σ² / (σ²·s²/(σ²+s²))) for Gaussian-approximate posteriors. |
| 3.2 | `master_loop/bandit.py` | Thompson sampling across chain roots for budget allocation. Beta/Gaussian posterior on `|point/SE|` per chain. |
| 3.3 | `llm/self_refine.py` | Reflexion / Self-Refine loop on the critic agent: propose → tool-call identify → on failure, structured reflection → planner revises. |
| 3.4 | `master_loop/budget.py` | Cost-aware tracker — per-experiment LLM tokens, wallclock, peak RAM, R-bridge time. CLI flag `--budget tokens=200k,wall=15min`. |
| 3.5 | `llm/rag_history.py` | Sentence-embedding RAG over prior `(flags, hypothesis, estimator, verdict)` tuples; injected as `dspy.Predict` few-shot for planner. |
| 3.6 | `master_loop/saturation.py` | Bayesian saturation stopping: posterior probability that the next chain step would shrink the credible interval by less than ε. Replaces the null-streak heuristic. |

**Exit criteria:** the EIG and Thompson-sampling paths are exercised in
the lalonde + Adult Census smoke tests and produce richer chain
forests than the heuristic baseline.

---

## Sprint 4 — TUI v1.0 (P0/P1)

Goal: the TUI passes the "this looks like a real research workbench" bar.

| Ticket | Component | Scope |
|---|---|---|
| 4.1 | `tui/widgets/forest_view.py` | Multi-pane layout — chain forest left, current walk right, log bottom, flag chips top. Already partially in via `--auto`; expand to all phases. |
| 4.2 | `tui/widgets/plots.py` | In-terminal plots via `textual-plotext` (power curves, love plots, propensity overlap density, sensemakr contour, CATE PDP). |
| 4.3 | `tui/studies.py` | Study save/load/branch (`studies/<name>/{manifest.lock, dag.json, hypotheses.jsonl, runs/<chain_id>/...}`). `/study save`, `/study load`, `/study branch what-if-strict-trimming`. |
| 4.4 | `tui/widgets/leaderboard.py` | Side-by-side estimator comparison (AutoGluon-style leaderboard rendered for any hypothesis that ran multiple estimators). |
| 4.5 | `tui/tutorial.py` | `--tutorial` walks a new user through init → discover → hypothesize → estimate → sensitivity → report on packaged Lalonde / IHDP, with inline hints. |
| 4.6 | `tui/widgets/flag_chips.py` | Hover-help on flag chips — definition + papers + which estimators it routes to. |
| 4.7 | `tui/export.py` | `/export notebook` writes the run as a `.ipynb` via `jupytext`. |

---

## Sprint 5 — BI persona (P0/P1)

| Ticket | Component | Scope |
|---|---|---|
| 5.1 | `data/connectors/ibis.py` | `ibis-framework` adapter (BigQuery, Snowflake, Redshift, Postgres, DuckDB, Databricks SQL). Sampling-aware: profile on trimmed 100k–1M sample; fit on the full set. |
| 5.2 | `data/connectors/ab_platforms.py` | A/B-platform ingest schemas (Eppo, Statsig, Optimizely, GrowthBook). |
| 5.3 | `tasks/rca.py` | "Why did metric X change?" mode — DoWhy-GCM anomaly-attribution + Quintas-Martínez multiply-robust distribution-change attribution. |
| 5.4 | `tasks/impact.py` | Causal forecasting — `CausalImpact` (Bayesian STS) + augmented SCM + matrix completion. CLI: `/impact when=2024-06-01 window=90d`. |
| 5.5 | `tasks/uplift.py` | Policy targeting — `policytree`, `mcf` (Bodory-Mascolo-Lechner 2024), Qini curve, AUUC, expected-policy-value. |

---

## Sprint 6 — Identification, multiverse, advanced sensitivity (P1)

| Ticket | Component | Scope |
|---|---|---|
| 6.1 | `identify/engine.py` | `ananke` / `Y0` as second ID engine. Reconcile vs DoWhy; agreement-check field on `IDResult`. |
| 6.2 | `multiverse/specr.py` | `specr` 1.0 + `multiverse` (Sarma-Kale-Moon). Specification curve + Simonsohn-Simmons-Nelson joint inference WHERE principled-equivalence holds; surface that caveat. |
| 6.3 | `multiverse/dag_bma.py` | Multiverse-of-DAGs — bootstrapped CD posterior + BMA-weighted estimate. |
| 6.4 | `identify/transportability.py` | Bareinboim-Pearl transportability for `TARGET_POPULATION_DIFFERS`. |
| 6.5 | `estimators/python/proximal.py` | Liu-Tchetgen-Tchetgen 2024 proximal CI two-stage regression when `PROXIMAL_PAIR_AVAILABLE`. |
| 6.6 | `estimators/python/frontdoor.py` | Front-door estimation via g-formula + bootstrap CI. |
| **6.7** | `sensitivity/dashboard.py` | **NEW — Zhao 2019 sensitivity value (JASA).** Add a `zhao_sensitivity_value` panel alongside E-value and sensemakr RV. Returns the single Γ threshold at which the matched-pair inference becomes inconclusive. Compute only when the estimator path was matching (`rbridge.matchit`). Asymptotic-normal CI for Γ itself. Skip the design-sensitivity-power piece (deferred to v1.1). |
| **6.8** | `identify/mr_bridge.py` | **NEW (v1.1 candidate, behind `MENDELIAN_RANDOMIZATION` flag).** MR-RAPS / GRAPPLE bridge for Mendelian randomization with horizontal-pleiotropy handling. GWAS-summary-statistics ingestion path is a separate ticket. Trigger: first clinical / epi user request. |

---

## Sprint 6.5 — Complex / large-DAG handling (P0/P1, **new — not in SEDR**)

Goal: address topologies the current pipeline doesn't model. The MB
work just shipped helps with adjustment-set selection; this sprint
expands the **graph language** itself.

| Ticket | Pattern | Why it matters | Scope |
|---|---|---|---|
| 6.5.1 | **ADMGs with latents** as proper graph elements | Today `investigator.unmeasured_confounders` is descriptive prose; identification can't see it. | `core/graph.py` adds bidirected edges (`<->`). Investigator labels latent edges from the brief. `q5_identify` passes ADMGs to `ananke` (Sprint 6.1) and runs Tian-Shpitser ID*. |
| 6.5.2 | **Multi-mediator chains** | `CandidateExperiment.mediator: str \| None` is single-mediator. Real mediation chains (T → M1 → M2 → Y) and sequential mediation can't be expressed. | `CandidateExperiment.mediators: tuple[str, ...]` with order. Estimator dispatch routes multi-mediator to `paths`/`cmaverse` (R) or VanderWeele-Vansteelandt sequential decomposition. |
| 6.5.3 | **Time-varying confounding** with temporal layering | `TIME_VARYING_TREATMENT` flag exists but no estimator declares support and the graph builder has no time axis. | `core/graph.py` adds `time_index: int \| None` per node. `_build_graph_for_proposal` constructs a time-layered graph from the panel structure. Routes to LTMLE / `gfoRmula` / lmtp longitudinal specs. |
| 6.5.4 | **Network / partial-interference graphs** | No interference handling at all. | New `core/graph.py::InterferenceGraph` carrying unit adjacency. Aronow-Samii exposure mapping per unit. Estimator wrappers: Aronow-Samii 2017, Hudgens-Halloran 2008, Sävje-Aronow-Hudgens 2021. New flag: `INTERFERENCE_NETWORK_KNOWN` (adjacency provided) vs `INTERFERENCE_SUSPECTED_UNKNOWN`. |
| 6.5.5 | **Hierarchical / multilevel structure** | `CLUSTERED` flag exists, no routing. | Two-level structure on graph (unit, cluster). Routes to `WeightIt + tmle3` with cluster-robust SEs, multilevel TMLE (van der Laan 2022), and Bayesian hierarchical models (`brms`, `PyMC`). |
| 6.5.6 | **Large DAGs (>50 nodes)** — scope reduction | The pipeline currently passes the full DAG to DoWhy. On big graphs this is slow and can mask identifiability bugs. | (a) **c-component decomposition** via ananke — identification only needs the c-components touching (T, Y). (b) **Modular subgraph extraction** — restrict to the union of the directed paths between T and Y plus ancestors of any node in the adjustment set. (c) **d-separation-driven pruning** — drop nodes that are d-separated from both T and Y given the candidate adjustment set. |
| 6.5.7 | **Domain-specific DAG templates** | LLM-driven DAG construction is unreliable on novel topologies. Templates encode field expertise. | New module `discovery/dag_templates/`. Each template is a Pydantic model with placeholder nodes the LLM fills. Initial templates: clinical TTE (eligibility window → washout → randomization → follow-up windows → outcome), MMM (channels → reach → conversion → revenue), attribution (touchpoints with sequential mediation), ecological spatiotemporal (lat/lon × time-lag), engineering trace (service-A → service-B → SLO). |
| 6.5.8 | **Front-door / instrumental DAG construction** | Already a Sprint-6 estimator target; this ticket constructs the topology rather than just the estimator. | When the brief names an instrument Z, the graph builder constructs Z → T → Y plus latents `U <-> T, U <-> Y`. When a front-door mediator M exists, constructs T → M → Y with `U <-> T, U <-> Y`. Routes match the topology. |
| 6.5.9 | **Effect-modifier topology** vs confounder-misclassification | Today modifiers leak into confounders. | Modifier nodes get role `EFFECT_MODIFIER` and edges `T → Y` are annotated as moderated by the modifier rather than the modifier becoming a confounder. Estimator dispatch routes to CATE-capable estimators. |
| 6.5.10 | **DAG-mismatch alerts** | When the MB pass + the LLM brief + the bnlearn structure learner disagree, surface it. | New `Conflict` record per (T, Y) — three sources weighted by the soft-flag confidence machinery. Reviewer-ready in the report. |

**Domain-specific guidance** (informs Sprint-7 work):

* **Clinical / oncology genomics (n ≪ p)**: MB Phase 3 (just shipped)
  is the first step. Add c-component decomposition (6.5.6a) so a
  20k-gene + clinical-covariates DAG is tractable. Pathway-aware DAG
  templates (KEGG / Reactome) as a domain library.
* **Policy / economics (panel, staggered)**: Time-varying layering
  (6.5.3) + the modern DiD stack (Sprint 2.2). Multiverse-of-DAGs
  (Sprint 6.3) for robustness across plausible identification
  assumptions.
* **Marketing / BI (attribution sequences)**: Multi-mediator chains
  (6.5.2) + sequential mediation. Network interference (6.5.4) for
  social-graph spillovers.
* **Engineering / ops (service traces)**: Trace-graph template
  (6.5.7) — OpenTelemetry traces as DAG input.
* **Ecology / environmental (spatiotemporal)**: Tigramite PCMCI+
  (Sprint 7.1) + spatial neighbour graphs (6.5.4 generalized).

**Exit criteria for Sprint 6.5:**
1. A test DAG with 100 nodes, 4 latents, 3 mediators in a chain
   completes identification + estimation in under 30 s.
2. Lalonde re-run produces an ADMG that explicitly carries
   "U <-> sex, U <-> race" latents from the brief.
3. The clinical-TTE template scaffolds a target-trial DAG from a CSV
   in one command.

---

## Sprint 7 — Domain extensions (P1/P2)

| Ticket | Domain | Scope |
|---|---|---|
| 7.1 | Time-series | Tigramite v5 wrapper (PCMCI+, LPCMCI, J-PCMCI+, RPCMCI). |
| 7.2 | Clinical | TrialEmulation R bridge (Rezvani et al. 2024); CONSORT-AI / STaRT-RWE / TARGET 2024 reporting templates; cloning-censoring-weighting for per-protocol. |
| 7.3 | Marketing | MMM wrappers — Robyn (R / Meta), Meridian (Python / Google 2024), PyMC-Marketing (full Bayesian). User picks based on team. |
| 7.4 | Marketing | GeoLift incrementality + ghost-ads + ITT-vs-PP per-protocol. |
| 7.5 | Network interference | Aronow-Samii / Hudgens-Halloran / Sävje-Aronow-Hudgens Python implementation (no production library covers all of these). |
| 7.6 | Dose-response | CCT-learner + GPS + npcausal wrappers. |
| 7.7 | Distributional | Firpo unconditional-quantile, Chernozhukov-Fernández-Val-Melly counterfactual distributions, `bamlss` distributional regression. |

---

## Sprint 8 — Inference engine + local-model strategy (P1)

| Ticket | Component | Scope |
|---|---|---|
| 8.1 | `llm/engines/` | Engine abstraction with adapters: **llama.cpp server (default)**, Ollama (compat), vLLM (server mode), MLX-LM (Mac), TabbyAPI (single-GPU EXL2), sglang. |
| 8.2 | `llm/spec_decoding.py` | EAGLE-2/3 speculative decoding with draft-model auto-selection. 1.4–2× speedup at batch=1 per published benchmarks. |
| 8.3 | `llm/hardware_tiers.py` | Re-benched T0–T5 tier map for Qwen3-Thinking, DeepSeek-R1-Distill, GLM-4-32B-0414, Phi-4, Mistral-Small-3, Gemma 3, Llama 3.3 70B. CI re-benches on model release. |
| 8.4 | `llm/multi_agent.py` | Multi-agent debate (planner ↔ skeptic ↔ statistician) behind `--paranoid`. 3× tokens; gated. |
| 8.5 | `llm/cache.py` | Three-layer cache — engine prefix-KV-cache, semantic cache on `(prompt_template, hash(inputs))`, existing cassette replay with model-digest checks. |

---

## Sprint 9.5 — End-to-end flow audit (P0 final gate)

| Ticket | Component | Scope |
|---|---|---|
| **9.5.1** | `audits/end_to_end_flow.py` | **End-to-end flag/method routing audit.** Walk every discovery output, every emitted `DataFlag`, every detector, every estimator, every sensitivity panel, every reporting hook. Build a directed graph `discovery_signal → flag → router → estimator/diagnostic → sensitivity → synthesis`. For every node in that graph, confirm at least one inbound producer and one outbound consumer (no orphaned producers, no orphaned consumers). Surface a `FlowAuditReport` per CI run that lists: (a) flags emitted by some detector but consumed by zero rules, (b) flags consumed by some rule but emitted by zero detector, (c) estimators registered in the catalog but not reachable from any rule path, (d) sensitivity panels in the dashboard not surfaced in any report path, (e) discovery brief fields (mediators, instruments, unmeasured confounders, negative controls, target population) that are not routed into the master loop / Q5 / estimator selection. |
| **9.5.2** | `audits/method_coverage.py` | **Method-coverage report.** For each (estimand class × flag combination) the pipeline could plausibly encounter, list the estimator(s) and sensitivity tests routed to it. Render a sparse matrix; cells with zero coverage are filed as v1.1 tickets. |
| **9.5.3** | `audits/island_detector.py` | **Island detector.** Static graph analysis identifies modules / functions that have no inbound caller in the runtime path of `/auto`. Surfaces dead code, partially-wired features, and "implemented but not invoked" gaps. |

**Exit criteria:** every flag the discovery layer can emit has at least one estimator or diagnostic route. Every catalog estimator is reachable from the rule cascade for at least one flag combination. Every brief field (mediator / instrument / negative control / target-population indicator) is consumed somewhere downstream. Every sensitivity panel ends up in the executive synthesis prompt and the HTML report. The audit runs in CI and fails the build on regressions.

This sprint is the **v1.0 ship gate** — no release without a clean flow audit.

---

## Sprint 9 — Stretch (P2)

| Ticket | Component | Scope |
|---|---|---|
| 9.1 | `master_loop/mcts.py` | MCTS with progressive widening + Bayesian-surprise value (AutoDiscovery / ReST-MCTS* pattern). Behind `--master-mcts`. |
| 9.2 | `dispatch/learned_router.py` | ML-learned router on dispatch telemetry. XGBoost on flags vector → estimator probability. Explainability shim (SHAP) required. |
| 9.3 | `master_loop/estimator_classifier.py` | Per-estimator success classifier on `(flags, n, p, missingness)`. |
| 9.4 | `sensitivity/sequential.py` | Always-valid plug-in inference for AIPW/TMLE via Howard-Ramdas confidence sequences. |
| 9.5 | `multiple_testing/e_value.py` | E-value closed testing (FWER under continuous monitoring). |

---

## New flags to add (Sprint 1 + 6.5)

Grouped per the SEDR Part 3 + complex-DAG additions:

**Treatment timing & dynamics:** `ANTICIPATION_EFFECTS`,
`LAGGED_TREATMENT_RESPONSE`, `LEAD_LAG_RESPONSE`,
`TREATMENT_REVERSAL_ALLOWED`, `TIME_VARYING_CONFOUNDING_PRESENT`.

**Measurement error:** `EXPOSURE_MEASUREMENT_ERROR_CLASSICAL`,
`EXPOSURE_MEASUREMENT_ERROR_BERKSON`, `OUTCOME_MEASUREMENT_ERROR`,
`DIFFERENTIAL_MEASUREMENT_ERROR`.

**Selection biases:** `COLLIDER_STRATIFICATION_RISK`,
`ATTRITION_PRESENT`, `HEALTHY_WORKER_BIAS_SUSPECTED`,
`LEFT_TRUNCATION`.

**Outcome scales:** `ORDINAL_OUTCOME`, `MULTIVARIATE_OUTCOME`,
`COMPOSITIONAL_OUTCOME`, `FUNCTIONAL_OUTCOME`, `LONG_TAILED_OUTCOME`.

**Survey / sampling design:** `SURVEY_WEIGHTS_PRESENT`,
`STRATIFIED_SAMPLING`, `CLUSTER_SAMPLING`, `MULTI_STAGE_SAMPLING`.

**Treatment dose & shape:** `DOSE_RESPONSE_EXPECTED_MONOTONE`,
`DOSE_RESPONSE_EXPECTED_NONMONOTONE`, `THRESHOLD_EFFECT_SUSPECTED`.

**Heterogeneity hints:** `PRESPECIFIED_SUBGROUPS`, `EXPLORATORY_HTE`.

**Confounding strength:** `STRONG_KNOWN_CONFOUNDER`,
`WEAK_OVERLAP_SUSPECTED`, `SPARSE_PROPENSITY_FEATURES`.

**Causal-discovery diagnostic:** `TETRAD_VIOLATION`, `MAG_AMBIGUITY`,
`LATENT_CONFOUNDER_SUSPECTED`.

**Identification:** `FRONT_DOOR_AVAILABLE`,
`TRANSPORTABILITY_REQUIRED`, `PROXIMAL_PAIR_AVAILABLE`.

**Complex DAG (new in 6.5):** `INTERFERENCE_NETWORK_KNOWN`,
`INTERFERENCE_SUSPECTED_UNKNOWN`, `MULTI_MEDIATOR_CHAIN`,
`LARGE_DAG` (>50 nodes), `PATHWAY_AWARE_TEMPLATE_AVAILABLE`.

---

## Epistemic hygiene (carried from the SEDR)

1. **LLM-driven CD is unsettled.** Wan et al. (Feb 2025) shows LLM
   contributions are most reliable as (a) orientation hints for
   edges already identified by statistical CD and (b) forbidden-edge
   constraints. Vashishtha et al. ICLR 2025 frames it as causal-order
   priors. Felisha never lets an LLM orient against a passed Layer-4
   CI test.
2. **"Honest" causal trees aren't magic.** Cattaneo-Klusowski-Yu
   (March 2026) prove adaptive CART causal trees can't achieve
   polynomial-in-n uniform convergence and honesty is regularisation,
   not a guarantee. Keep `honesty=true` as default but expose the
   knob and surface the caveat in the method card.
3. **Speculative decoding gains are workload-dependent.** EAGLE-3
   gives 1.4–2× at batch=1 / long generations but degrades on high-
   concurrency or short outputs. Verify on each model release.
4. **Multiverse joint inference requires principled equivalence.**
   Del Giudice-Gangestad 2021. Felisha exposes the multiverse as
   exploratory robustness by default; Simonsohn joint test only when
   the user affirms principled equivalence.
5. **Target-trial emulation is a framework, not a method.** The
   protocol elements (eligibility, treatment strategies, assignment,
   follow-up, outcome, contrast, analysis plan) must be filled before
   estimation; the heavy lifting (immortal-time avoidance,
   per-protocol weighting, cloning-censoring) still happens in the
   estimator layer.

---

## Key references (the modern shortlist)

Discovery & identification: DoWhy 0.12, `ananke`, `Y0`, `causal-learn`,
Tigramite v5, `causica` / DECI, `bnlearn`, `pcalg`.

Estimation: `tlverse` (`tmle3` + `sl3` + `tmle3mediate` + `ltmle`),
`grf`, EconML, `did` + `didimputation` + `DIDmultiplegt` +
`bacondecomp` + `HonestDiD`, `augsynth` + `pysyncon` + `synthdid`,
`rdrobust` family, `bartCause` + `stochtree` (BCF), `lmtp`,
`policytree` + `mcf`, `causalml`, `causaltune`.

Conformal: Lei-Candès 2021, Alaa et al. 2023, Jonkers et al. 2024 v6.

Sensitivity: `sensemakr`, `EValue`, `tipr`, Rosenbaum bounds, Manski
bounds, proximal CI (Tchetgen-Tchetgen 2024).

LLM scaffolding: DSPy, Outlines, Instructor, Pydantic-AI, LangGraph,
llama.cpp server, vLLM, MLX-LM, EAGLE-2/3, sglang.

Reporting: Quarto, OSF preregistration JSON, CONSORT-AI / STaRT-RWE /
TARGET 2024.

BI / domains: `ibis-framework`, Eppo / Statsig / Optimizely /
GrowthBook exports, DoWhy-GCM, `CausalImpact`, Robyn / Meridian /
PyMC-Marketing, GeoLift, TrialEmulation.
