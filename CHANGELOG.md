# CHANGELOG

All notable changes to Felisha / CausalRoadmap. Entries are grouped by
"round" — a coherent batch of work shipped together.

## v1.0 (May 2026) — sprint plan complete

Every P0/P1 ticket from `docs/SPRINT_PLAN_V1.md` is shipped. Headline
features:

- 40+ estimators across Python and R; modern DiD stack; synthetic
  control / ASCM / SDiD; rdrobust family; tlverse `tmle3` + `sl3` +
  `tmle3mediate`; conformal ITE; proximal CI; front-door g-formula;
  distributional regression; hierarchical / multilevel; network
  interference (Aronow-Samii, Sävje-Aronow-Hudgens); Duarte autobounds
  partial-identification fallback.
- Identification with DoWhy + `ananke`/`Y0` reconciliation;
  transportability; multiverse + multiverse-of-DAGs/BMA;
  c-component decomposition for large graphs; ADMGs with proper
  bidirected edges; multi-mediator chains; time-varying / panel
  layering; effect-modifier topology; domain DAG templates.
- Autonomous master loop with EIG-based chain continuation, Thompson
  sampling across roots, Self-Refine critic loop, MCTS with
  progressive widening, RAG over prior runs, postmortem record,
  per-family circuit breaker, multi-agent debate behind `--paranoid`.
- Sensitivity dashboard aggregating E-value (scale-routed),
  sensemakr / Chernozhukov-Cinelli-Newey OVB, Zhao 2019 Γ, Rosenbaum,
  Manski, refutations, anomaly audit, tipping-point, negative-control
  scan, e-value closed testing, always-valid CIs (Howard-Ramdas /
  Waudby-Smith-Ramdas), multiple-testing adjustment.
- BI tasks — root-cause attribution, causal forecasting / impact,
  uplift / policy targeting, MMM wrappers (Robyn / Meridian /
  PyMC-Marketing), GeoLift incrementality.
- Reporting — Quarto multi-format output, OSF / AsPredicted /
  Hubbard NEJM 2024 TTE preregistration export, Jupyter notebook
  export, run.lock.json reproducibility manifest.
- TUI — multi-pane layout, in-terminal plots via `textual-plotext`,
  tutorial mode, study save/load/branch, hover-help on flag chips.
- LLM scaffolding — engine abstraction (Ollama / llama.cpp / vLLM /
  MLX-LM), EAGLE-2/3 speculative-decoding adapter, three-layer
  cache, DSPy + Outlines opt-in module catalog covering all 8 prompt
  sites, YAML FlagRegistry with full metadata.
- Audits — end-to-end flow audit (ship gate, GREEN), method-coverage
  matrix (1610 v1.1 ticket candidates), island detector.

Deferred to v1.1:
- 6.8 MR-RAPS / GRAPPLE for Mendelian randomization (epidemiology-
  specific; waits for a clinical user request per pro/con analysis).
- Full DSPy compile-time optimisation against gold sets (scaffolding
  shipped; gold-set curation is the next gating task).
- TUI plotext as a hard dep (currently optional with ASCII fallback).
- Sub-agent test stubs for 2.1 / 9.2 / 9.3 / 9.5 replaced with full
  behavioural tests (the Wave 7 rate-limit cut those short — code
  is functional + import-smoked, behavioural coverage upgrade for v1.1).

1165+ unit tests passing. 30 skipped (rpy2 / pymc / ananke / autobounds
/ mlx_lm / plotext / dspy / outlines optional). Audit GREEN.

## Round 4 — wire round-2 LLM modules + smoke-test follow-ups (May 2026)

### Wired
- `narrate_identification` runs inside `_run_one_experiment` right
  after `identify_effect`, attaching plain-language identification
  prose to `walk.q5_identification["narration"]`.
- `audit_for_anomalies` runs deterministic pre-screen + optional LLM
  refinement and lands on `result.diagnostics["anomaly_audit"]`.
- `interpret_sensitivity` produces domain-aware sensitivity prose;
  deterministic verdict colour is pinned, LLM enriches the narrative
  only.
- `analyze_cross_experiment` runs before `synthesize_insights` and
  feeds contradictions / reinforcements / chain narratives into the
  synthesis prompt.
- `dedupe_candidates` runs after the candidate-queue planner.
- `diagnose_missingness` runs in `run_discovery`, with
  `HEAVY_MISSINGNESS` promoted to a flag when per-column missing
  rate ≥ 20%.
- New CLI subcommand `causalrag auto run <data>` mirrors the TUI.

### Fixed (from lalonde smoke test #1)
- `_auto_robustness_candidate` now picks a categorically-different
  estimator family from the parent; refuses to schedule if no valid
  swap is registered. Was: same DML fell through on every red parent
  → 5 identical re-runs.
- `_is_duplicate_followup` guard runs on every pending follow-up
  (robustness child AND foundation follow-up).
- Red sensitivity no longer short-circuits substantive follow-ups.
- Auto-infer T/Y from investigator role assignments so flag detectors
  fire automatically in master mode.

### Fixed (from lalonde smoke test #2)
- Refuse to fire foundation children when parent verdict is
  `errored` / `unknown`.
- Refuse foundation candidates when the LLM returned a partial
  proposal (missing T/Y/estimand).

### Added
- Working-memory compressor (`_compress_history`) — bounded prompt
  growth at K ≥ 8 experiments.
- Chain-forest view rendered in critic + foundation prompts.
- Decision-ledger summary surfaced in critic prompt so the LLM
  avoids re-trying dead-end estimator families.
- Tipping-point auto-fire on yellow/red sensitivity via tipr.
- Negative-control outcome scan on yellow/red sensitivity.
- Walk diagnostics surfaced in HTML report: narration, anomaly
  audit, sensitivity interpretation, chain linkage, adjusted
  p-values, sensitivity verdict chip, failure reason.

### Tests
- 17 new unit tests for master_loop helpers.
- 336 unit tests passing.

---

## Round 3 — file-isolated audit fixes (May 2026)

Eight parallel agents shipped audit-driven fixes:

- **E-value scale routing** (`sensitivity/evalue.py`) — remove silent
  ±10 clamp, add `evalue_for_estimator` helper that routes scale by
  estimator id + outcome dtype, add `risk_difference` scale +
  `baseline_risk` kwarg. 18 tests.
- **IV first-stage F-statistic** (`estimators/rbridge/grf.py`) —
  partial F with Stock-Yogo / Olea-Pflueger verdicts. 7 tests.
- **SE-anchored refutation thresholds** (`roadmap/q7_estimate.py`)
  — placebo / random-common-cause / subset-bootstrap all
  calibrated to original SE. Protocol flags preserved through
  refutation refits. 6 tests.
- **LMTP defensible defaults** (`estimators/rbridge/lmtp.py`) —
  refuse to run with the `SL.glm + SL.mean` straw-man unless
  `allow_minimal_learners=True`; `LMTPShift` defaults to contrast
  (real ATE) instead of policy mean. 5 tests.
- **Meta-learner honest CIs** (`estimators/python/meta.py`) —
  bootstrap fallback when EconML's `effect_interval` is
  unavailable, DRLearner cv bumped 3 → 5, W/X confounders/modifiers
  split. 4 tests.
- **DataFlag system expansion** (`core/flags.py`, `data/flags.py`,
  `estimators/python/select.py`, `discovery/expert.py`) — 7 new
  flags (RARE_OUTCOME, IMBALANCED_TREATMENT, BOUNDED_OUTCOME,
  ZERO_INFLATED_OUTCOME, EFFECT_MODIFICATION_OF_INTEREST,
  DIFF_IN_DIFF_CANDIDATE, STAGGERED_ADOPTION) with detectors +
  selector routes. INSTRUMENTAL_CANDIDATE_PRESENT and
  MEDIATOR_PROPOSED actually emitted. 33 tests.
- **Synthesis robustness** (`reporting/synthesis.py`) — dtype-aware
  unit classification, ATT/ATC-aware magnitude scaling, fabricated-id
  validation, deterministic confidence override, graceful failure
  stub. 18 tests.
- **Q5 collider/descendant guard** (`roadmap/q5_identify.py`,
  `core/graph.py`) — programmatic filter excludes descendants of T,
  mediators, and colliders from the adjustment set. Accepts
  candidate_graphs= for multi-DAG identification. 10 tests.

### Plus
- Round 2 LLM modules built (identification narration, sensitivity
  interpretation, anomaly audit, cross-experiment analysis,
  dedupe, multiple-testing, shared honesty preamble). Wired in
  Round 4.
- TUI live candidate-queue panel + chain-forest sidebar. 11 tests.
- Continuous-T positivity diagnostic. 6 tests.
- Pin adjustment set on backdoor identification. 4 tests.
- BART convergence diagnostics surfaced. New skipped-without-pymc
  tests.
- Multiple-testing adjustment wired into master loop + synthesis
  prompt. 9 tests.

---

## Round 2 — domain-aware executive synthesis (May 2026)

- New `reporting/synthesis.py` — domain-agnostic synthesis layer.
  LLM infers domain (clinical / business / policy / ecology /
  engineering / ...) and writes findings in that field's language.
  Deterministic confidence override on CI-crosses-zero / red
  sensitivity / n < 100.
- README rewritten to reflect Beta status, full pipeline architecture,
  master-loop deep dive, slash-command reference.

---

## Round 1 — initial master loop architecture (May 2026)

- Petersen–van der Laan Causal Roadmap walks (Q1–Q8) per hypothesis.
- 23-estimator catalog: EconML DML / forests / meta-learners / BART
  in Python; grf / lmtp / MatchIt / WeightIt / mediation / survRM2 /
  bartCause / marginaleffects / sensemakr / EValue / tipr / bnlearn
  via R bridge.
- Autonomous `/auto` mode: candidate queue, propose-K → critique →
  commit, foundation recursion with multi-chain bookkeeping, auto-
  robustness on red sensitivity, BH multiple-testing adjustment.
- TUI (Textual) with slash-command composer, phase tracker, log view.
