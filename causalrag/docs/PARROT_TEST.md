# Parrot Diagnostic — LLM Causal-Reasoning Falsification Test

## What this test is

The **Parrot Diagnostic** is a falsification harness for the LLM-guided
phases of the CausalRoadmap pipeline. It asks the question:

> When the LLM proposes hypotheses for a dataset, is it *reasoning over
> the data we just handed it*, or is it *retrieving the canonical
> textbook answer for a dataset whose name it recognized*?

Large LLMs have memorized canonical causal-inference benchmarks
(Lalonde NSW, IHDP, ACIC, Card 1995) almost verbatim. If we hand the
pipeline a csv that *looks* like Lalonde NSW but contains a
sign-flipped outcome, a **parroting** model will still happily propose
"training increases earnings" hypotheses with confidently positive
expected effects. A **reasoning** model will frame its rationales
neutrally and let the estimator stage discover the true (negative)
effect.

The diagnostic also includes a non-canonical synthetic-health smoke
test to confirm the pipeline doesn't degrade on a dataset the model
has *not* memorized.

## Why it matters

The §6 "honesty layer" of the CausalRoadmap protocol exists to guard
against exactly this failure mode. Without an independent falsification
test, we have no way to know whether the LLM phases are doing real
work or producing impressive-sounding boilerplate. The Parrot
Diagnostic gives us a binary pass/fail signal that:

- Catches retrieval-style answers on memorized datasets.
- Verifies that the candidate-queue planner frames its rationales
  *before* the estimator runs (i.e. that the LLM isn't asserting a
  direction it cannot yet know).
- Confirms the pipeline still produces a defensible point estimate on
  a non-canonical dataset.

## The two legs

### 1. Sign-flipped Lalonde (the falsification leg)

- **Construction.** Load Dehejia-Wahba NSW via `causaldata`. For every
  treated row, multiply `re78` by `-1`. Persist to
  `lalonde_signflipped.csv`. Covariates and the treatment column are
  untouched, so the dataset *looks* like Lalonde to anything that
  reads the column names.
- **Expected truth.** Naive diff-in-means is now decisively negative
  (control mean ~ +$4,555; treated mean ~ −$6,349; ground-truth ATE on
  the flipped file is ~ −$10,900).
- **Parrot signature.** An LLM whose candidate queue keeps emitting
  rationales like *"training increases earnings"* / *"we expect a
  positive ATE"* is parroting — those statements cannot be true on
  the data we actually persisted.
- **Reasoning signature.** Neutral rationales (*"the data will reveal
  whether the training intervention affected earnings"*, *"effect
  direction depends on the post-period earnings distribution"*),
  followed by an estimator stage that lands on a negative point
  estimate.

### 2. Synthetic health (the non-canonical smoke leg)

- **Construction.** `scripts/generate_synthetic_health.py` builds an
  800-row CSV with columns the model has never seen verbatim
  (`subject_id`, `statin_adherence`, `cardiac_event_5yr`, …). The DGP
  is a logistic-hazard model where adherence has a true logit-scale
  effect of −0.45 and an interaction with BMI. The marginal ATE on
  the probability scale is small (~ −0.03 to −0.10 across seeds) —
  small enough that no retrieval shortcut produces it.
- **Pass criteria.** Pipeline must complete, persist an estimate, and
  land within ±0.15 of the oracle marginal ATE.

## How to run

### Prerequisites

- A live local Ollama server (default at `http://127.0.0.1:11434`).
- `causalrag` installed with the `real-data` extra:
  ```bash
  pip install -e '.[real-data,estimators,sensitivity]'
  ```
- An LLM slot that the discovery + expert selector can resolve (run
  `causalrag doctor` to confirm).

### Enable + run

```bash
RUN_PARROT_TESTS=1 pytest tests/integration/test_parrot_harness.py -v
```

Optional environment knobs:

| Env var               | Effect                                                                 |
|-----------------------|------------------------------------------------------------------------|
| `RUN_PARROT_TESTS=1`  | Required — without this the two slow tests are skipped.                |
| `OLLAMA_BASE_URL=…`   | Override the Ollama endpoint (default `http://127.0.0.1:11434`).        |
| `PARROT_ARTIFACTS=…`  | Persist run artifacts at this path (instead of pytest's tmp dir).      |
| `PARROT_SKIP_HEALTH=1`| Run only the Lalonde leg (saves ~15 min).                              |

To run the same diagnostic outside pytest:

```bash
python scripts/run_parrot_test.py --artifacts artifacts/parrot_runs
```

### Expected runtime

| Leg                       | Runtime (workstation w/ Ollama on a 24 GB GPU) |
|---------------------------|------------------------------------------------|
| Sign-flipped Lalonde      | ~15 min                                        |
| Synthetic health smoke    | ~15 min                                        |
| **Total** (both legs)     | ~30 min                                        |

Plan for ~45 min if you're running on CPU-backed Ollama or a smaller
GPU; the master loop's candidate-queue planning is the heaviest call.

## What to look for in the output

After the run, four artifacts live under
`artifacts/parrot_runs/<dataset>/`:

- **`study.causalrag.yaml`** — the persisted protocol. The
  `candidate_queue` section is the parrot evidence: read each
  candidate's `impact_rationale` and check whether the model declared
  a sign *before* seeing any estimate. Read the
  `roadmap_walks.<id>.q7_estimates[0].point_estimate` to see what the
  estimator returned.
- **`executive_synthesis.json`** — if the master loop produced one,
  this contains the headline numerical result.
- **Console output** — the harness prints a `ParrotResult.summary()`
  block:
  ```
  sign-flipped Lalonde:
    rationales scanned        = 22
    sign-anticipating         = 4
    sign-anticipation ratio   = 18.18%
    point estimate            = -8423.14
    95% CI                    = [-10210.41, -6635.87]
  ```

### Sign-anticipation ratio

The headline metric. For each string in
`protocol.candidate_queue[*].impact_rationale` (and
`identifiability_rationale` / `power_rationale`) plus every
foundation-followup `note` in the decision ledger, we check whether
the text contains a directional word:

```
increase / increases / increasing / decrease / decreases / raises /
lowers / lowered / positive / negative / boost / reduce / improve /
worsen
```

The ratio is `(hits) / (total non-empty rationales)`.

| Sign-anticipation ratio | Interpretation                                  |
|-------------------------|-------------------------------------------------|
| **< 30 %**              | PASSING — model frames hypotheses neutrally.    |
| **30 – 50 %**           | INCONCLUSIVE — investigate prompt design.       |
| **> 50 %**              | FAILING — model is parroting memorized priors.  |

## Pass / fail criteria (test harness)

### `test_signflipped_lalonde_falsifies_parrot`

Both of these must hold:

1. `point_estimate < 0` on the sign-flipped data. Falsification: the
   data demands it.
2. `sign_anticipation_ratio < 0.30`.

If criterion 1 fails: the pipeline produced a positive estimate on
data whose true ATE is sharply negative. This is the parrot smoking
gun.

If only criterion 2 fails: the estimator stage recovered the right
answer but the LLM phases were asserting unjustified priors. This is
also parrot behavior — the model got the right answer *despite* its
rationales, not *because of* them.

### `test_synthetic_health_smoke_runs_clean`

- `|point_estimate - oracle_ATE| < 0.15` (oracle is in the truth json
  written alongside the csv).
- `sign_anticipation_ratio < 0.50` (looser bound — a reasoning model
  *might* legitimately anticipate that statins lower cardiac risk
  from biomedical priors; the parrot signature here is closer to
  100 %).

## What "passing" vs "failing" implies

- **Passing both legs.** The LLM phases of the pipeline are doing real
  causal-reasoning work — they're framing hypotheses neutrally on a
  dataset whose name they recognize but whose data they have not seen,
  and they cope cleanly with a non-canonical schema.

- **Failing sign-flipped Lalonde, passing synthetic health.** The
  model retrieves canonical answers when it recognizes a dataset but
  reasons fine on novel data. Mitigation: improve the discovery
  prompt to discourage prior assertion, or pin a more reasoning-heavy
  model in the `expert` slot.

- **Failing both legs.** Either the estimator stage is broken (likely
  if the synthetic-health point estimate is wildly off) or the model
  is parroting on *anything* the dataset description suggests
  (mitigation: review `master_loop._PLANNER_SYSTEM` and the
  candidate-queue planner prompt for direction-asserting language).

- **Passing Lalonde, failing synthetic health.** Pipeline plumbing
  issue on novel schemas (preprocessing, type inference) — not a
  parrot problem. Read the per-phase events in the console output to
  find the failing phase.

## Files

- `scripts/parrot_signflip_lalonde.py` — generator for the sign-flipped csv.
- `scripts/generate_synthetic_health.py` — generator for the
  non-canonical synthetic health csv (+ ground-truth json).
- `scripts/run_parrot_test.py` — non-pytest driver. Also defines the
  `count_sign_anticipating` analyzer reused by the pytest harness.
- `tests/integration/test_parrot_harness.py` — pytest entry point
  (slow; gated by `RUN_PARROT_TESTS`).
- `tests/integration/test_parrot_analyzer.py` — fast unit tests for
  the sign-anticipation regex / counter. Runs by default in CI.
