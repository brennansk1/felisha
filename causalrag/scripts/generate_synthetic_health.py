"""Generate a non-canonical synthetic health dataset for the parrot diagnostic.

This is the "smoke test" companion to the sign-flipped Lalonde test. The
goal here is NOT biological realism — the goal is a dataset whose
schema and column names do NOT appear verbatim in any public corpus the
LLM was trained on, so retrieval / memorization is impossible and the
model must read the actual data.

Schema (800 rows)
-----------------
- ``subject_id``                 — synthetic primary key (``S-00000`` …)
- ``age``                        — 45–85, gamma-shifted, integer
- ``sex``                        — 0/1 (50/50)
- ``bmi``                        — continuous, mean ~28, sd ~5
- ``blood_pressure_baseline``    — continuous mmHg, mean ~135, sd ~18
- ``insurance_type``             — categorical {private, medicare, medicaid, none}
- ``smoker_yes_no``              — binary 0/1
- ``diabetes_yes_no``            — binary 0/1
- ``exercise_freq_per_week``     — integer 0–7
- ``statin_adherence``           — binary TREATMENT (did the patient take their statin?)
- ``cardiac_event_5yr``          — binary OUTCOME (event in 5 years)

Data-generating process
-----------------------
Adherence propensity is a logistic function of insurance, age, and
exercise frequency (sicker / better-resourced patients are more likely
to adhere). The cardiac-event hazard is driven by:

    logit(P(event)) =
        β0
        + 0.04 * (age - 60)
        + 0.55 * smoker
        + 0.60 * diabetes
        + 0.08 * (bmi - 27)
        + 0.012 * (bp - 130)
        - 0.18 * exercise_freq_per_week
        + τ_true * statin_adherence
        + 0.20 * (statin_adherence) * (bmi - 27) / 5   # heterogeneity

with ``τ_true = -0.45`` (logit-scale, i.e. statins reduce risk). The
*marginal* ATE on the probability scale is around −0.07 to −0.10
depending on draw — *not* a number anyone would retrieve from
training. The interaction with BMI means CATE is more negative for
higher-BMI patients, which the pipeline should be able to surface
after the smoke run if heterogeneity hypotheses are queued.

Usage
-----
    python scripts/generate_synthetic_health.py --out artifacts/synthetic_health.csv --seed 2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_DEFAULT = 800


def generate(n: int = N_DEFAULT, seed: int = 2026) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build the synthetic frame.

    Returns
    -------
    (df, truth)
        ``truth`` carries the logit-scale coefficients + empirical
        marginal ATE so the harness can compare against ground truth.
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.gamma(shape=8.0, scale=4.0, size=n) + 40, 45, 85).round().astype(int)
    sex = rng.binomial(1, 0.50, size=n)
    bmi = np.clip(rng.normal(loc=28.0, scale=5.0, size=n), 16.0, 55.0)
    bp = np.clip(rng.normal(loc=135.0, scale=18.0, size=n), 85.0, 220.0)
    insurance = rng.choice(
        ["private", "medicare", "medicaid", "none"],
        size=n,
        p=[0.45, 0.30, 0.18, 0.07],
    )
    smoker = rng.binomial(1, 0.22, size=n)
    diabetes = rng.binomial(1, 0.18 + 0.004 * (bmi - 25), size=n).clip(0, 1)
    exercise = rng.integers(low=0, high=8, size=n)

    # Adherence propensity — driven by access (insurance), age, exercise habit
    insurance_score = pd.Series(insurance).map(
        {"private": 0.6, "medicare": 0.4, "medicaid": -0.1, "none": -0.8}
    ).to_numpy()
    adherence_logit = (
        -0.2
        + 0.025 * (age - 60)
        + 0.12 * exercise
        + insurance_score
        - 0.15 * smoker
    )
    p_adhere = 1.0 / (1.0 + np.exp(-adherence_logit))
    statin_adherence = (rng.uniform(size=n) < p_adhere).astype(int)

    # Outcome — logit hazard with a true negative ATE on logit scale
    tau_true = -0.45
    event_logit = (
        -2.4
        + 0.04 * (age - 60)
        + 0.55 * smoker
        + 0.60 * diabetes
        + 0.08 * (bmi - 27)
        + 0.012 * (bp - 130)
        - 0.18 * exercise
        + tau_true * statin_adherence
        + 0.20 * statin_adherence * (bmi - 27) / 5.0
    )
    p_event = 1.0 / (1.0 + np.exp(-event_logit))
    cardiac_event_5yr = (rng.uniform(size=n) < p_event).astype(int)

    # Empirical marginal-ATE under both treatment assignments (oracle)
    logit_t1 = event_logit + tau_true * (1 - statin_adherence) + 0.20 * (1 - statin_adherence) * (bmi - 27) / 5.0
    logit_t0 = event_logit - tau_true * statin_adherence - 0.20 * statin_adherence * (bmi - 27) / 5.0
    p_t1 = 1.0 / (1.0 + np.exp(-logit_t1))
    p_t0 = 1.0 / (1.0 + np.exp(-logit_t0))
    marginal_ate = float((p_t1 - p_t0).mean())

    df = pd.DataFrame(
        {
            "subject_id": [f"S-{i:05d}" for i in range(n)],
            "age": age,
            "sex": sex.astype(int),
            "bmi": bmi.round(2),
            "blood_pressure_baseline": bp.round(1),
            "insurance_type": insurance,
            "smoker_yes_no": smoker.astype(int),
            "diabetes_yes_no": diabetes.astype(int),
            "exercise_freq_per_week": exercise.astype(int),
            "statin_adherence": statin_adherence,
            "cardiac_event_5yr": cardiac_event_5yr,
        }
    )

    truth = {
        "tau_true_logit": tau_true,
        "marginal_ate_probability": marginal_ate,
        "n": n,
        "seed": seed,
    }
    return df, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/synthetic_health.csv"),
    )
    parser.add_argument("--n", type=int, default=N_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    df, truth = generate(n=args.n, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Write the ground-truth alongside so the harness can verify.
    import json

    truth_path = args.out.with_suffix(".truth.json")
    truth_path.write_text(json.dumps(truth, indent=2))

    print(
        f"Wrote {len(df)} rows to {args.out}.\n"
        f"  tau_true (logit)               = {truth['tau_true_logit']}\n"
        f"  marginal ATE (probability)     = {truth['marginal_ate_probability']:.4f}\n"
        f"  ground truth written to        = {truth_path}"
    )


if __name__ == "__main__":
    main()
