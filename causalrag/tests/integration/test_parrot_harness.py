"""Test harness for the LLM-causal-parrot diagnostic.

This file is the *pytest entry point* for the parrot test described in
``docs/PARROT_TEST.md``. It is SLOW (15+ minutes per dataset on a
workstation Ollama) and depends on a live local Ollama, so it's gated
behind both a marker and an env-var opt-in:

    RUN_PARROT_TESTS=1 pytest tests/integration/test_parrot_harness.py -v

The harness logic itself lives in ``scripts/run_parrot_test.py`` —
this file is a thin wrapper so CI / pytest collection both see it.

Two test cases:

- ``test_signflipped_lalonde_falsifies_parrot``: the FALSIFICATION leg.
  On the sign-flipped Lalonde csv, a parroting LLM will still propose
  "training raises earnings" hypotheses (memorized from training data).
  A reasoning LLM frames its hypotheses neutrally. The pass criteria
  are tight:
    1. Estimated ATE on the sign-flipped data MUST be negative.
    2. Sign-anticipation ratio in the candidate-queue rationales must
       be < 30%.

- ``test_synthetic_health_smoke_runs_clean``: the NON-CANONICAL leg.
  The model has not seen this dataset (it's generated locally with a
  novel schema). We just check the pipeline completes end-to-end,
  produces a finite estimate, and that the sign-anticipation ratio is
  low enough that the rationales were not pre-baked.

Skipped by default. If you only have one GPU and limited time, skip
the health leg by setting ``PARROT_SKIP_HEALTH=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.live_llm,
    pytest.mark.skipif(
        not os.environ.get("RUN_PARROT_TESTS"),
        reason=(
            "Parrot diagnostic is slow (~15 min per dataset) and needs live "
            "Ollama. Set RUN_PARROT_TESTS=1 to enable."
        ),
    ),
]


def _import_harness():
    """Load the driver module from ``scripts/``.

    The scripts/ directory is sibling to src/ — not a package, so we
    bring it in via importlib rather than a plain ``from scripts import …``.
    """
    import importlib.util
    import sys

    here = Path(__file__).resolve()
    scripts_dir = here.parents[2] / "scripts"
    name = "run_parrot_test_mod"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, scripts_dir / "run_parrot_test.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load scripts/run_parrot_test.py from {scripts_dir}")
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so dataclasses can resolve forward references.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def harness():
    return _import_harness()


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory) -> Path:
    """Persistent-per-session artifacts root. Outputs survive the run so
    you can inspect ``study.causalrag.yaml`` + ``executive_synthesis.json``."""
    base = os.environ.get("PARROT_ARTIFACTS")
    if base:
        out = Path(base)
        out.mkdir(parents=True, exist_ok=True)
        return out
    return tmp_path_factory.mktemp("parrot_runs")


@pytest.fixture(scope="module")
def signflipped_lalonde_project(harness, artifacts_dir: Path) -> Path:
    """Build the sign-flipped csv and run /auto once for the whole module."""
    from scripts.parrot_signflip_lalonde import build_signflipped, load_nsw  # type: ignore

    nsw = load_nsw()
    flipped = build_signflipped(nsw)
    csv_path = artifacts_dir / "lalonde_signflipped.csv"
    flipped.to_csv(csv_path, index=False)

    project = artifacts_dir / "lalonde_signflipped"
    harness.run_pipeline(
        dataset_path=csv_path,
        project_dir=project,
        research_question=(
            "Estimate the average effect of the NSW training program (treat) "
            "on 1978 earnings (re78) in this dataset."
        ),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    return project


@pytest.fixture(scope="module")
def synthetic_health_project(harness, artifacts_dir: Path) -> tuple[Path, dict]:
    """Build the synthetic health csv and run /auto once."""
    if os.environ.get("PARROT_SKIP_HEALTH"):
        pytest.skip("PARROT_SKIP_HEALTH=1 set")
    from scripts.generate_synthetic_health import generate  # type: ignore

    df, truth = generate()
    csv_path = artifacts_dir / "synthetic_health.csv"
    df.to_csv(csv_path, index=False)
    (artifacts_dir / "synthetic_health.truth.json").write_text(json.dumps(truth, indent=2))

    project = artifacts_dir / "synthetic_health"
    harness.run_pipeline(
        dataset_path=csv_path,
        project_dir=project,
        research_question=(
            "Estimate the average effect of statin adherence on 5-year cardiac "
            "event risk in this cohort."
        ),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    return project, truth


# ──────────────────────────────────────────────────────────────────────────


def test_signflipped_lalonde_falsifies_parrot(harness, signflipped_lalonde_project: Path) -> None:
    """The headline parrot test.

    Pass criteria:
      1. point estimate is negative (the data demands it; only retrieval
         from memorized Lalonde priors would produce a positive number)
      2. sign-anticipation ratio across LLM rationales is < 30%
    """
    result = harness.diagnose_run(
        dataset_label="sign-flipped Lalonde",
        protocol_yaml=signflipped_lalonde_project / "study.causalrag.yaml",
        synthesis_path=signflipped_lalonde_project / "executive_synthesis.json",
    )

    # Surface diagnostics on failure
    print("\n" + result.summary())
    if result.matched_phrases:
        print("Sign-anticipating phrases (first 5):")
        for phrase in result.matched_phrases[:5]:
            print(f"  • {phrase}")

    assert result.n_rationales > 0, (
        "No LLM rationales were captured — was the master loop or candidate "
        "queue actually populated? Check the protocol YAML."
    )
    assert result.point_estimate is not None, "No point estimate persisted."
    assert result.point_estimate < 0, (
        f"FAIL: point estimate {result.point_estimate:+.4f} is non-negative on "
        "sign-flipped Lalonde — this is the parrot signature (model used the "
        "memorized canonical answer instead of the data)."
    )
    assert result.sign_anticipation_ratio < 0.30, (
        f"FAIL: sign-anticipation ratio {result.sign_anticipation_ratio:.0%} ≥ "
        "30%. The LLM is asserting direction BEFORE running experiments — "
        "this is parrot behavior. Reasoning models keep rationales neutral."
    )


def test_synthetic_health_smoke_runs_clean(harness, synthetic_health_project) -> None:
    """Non-canonical smoke test.

    No-retrieval dataset → pipeline must:
      * complete and persist an estimate
      * not over-confidently anticipate a sign (ratio < 50%; this leg is
        looser since the LLM might reasonably anticipate a negative
        effect for statins)
      * land within a generous tolerance of the oracle marginal ATE
    """
    project, truth = synthetic_health_project
    result = harness.diagnose_run(
        dataset_label="synthetic health",
        protocol_yaml=project / "study.causalrag.yaml",
        synthesis_path=project / "executive_synthesis.json",
    )
    print("\n" + result.summary())
    print(f"  oracle marginal ATE        = {truth['marginal_ate_probability']:+.4f}")

    assert result.point_estimate is not None, "No estimate from synthetic health run."
    # Generous oracle bound — observational, n=800, model selection variance.
    oracle = truth["marginal_ate_probability"]
    assert abs(result.point_estimate - oracle) < 0.15, (
        f"Estimate {result.point_estimate:+.4f} too far from oracle "
        f"{oracle:+.4f} (>0.15 abs delta) — pipeline likely broken on a "
        "non-canonical dataset."
    )
    # Softer ceiling on this leg — see docstring.
    assert result.sign_anticipation_ratio < 0.50, (
        f"sign-anticipation ratio {result.sign_anticipation_ratio:.0%} ≥ 50% "
        "even on a novel dataset — investigate prompt design."
    )


# ──────────────────────────────────────────────────────────────────────────
# Unit tests for the sign-anticipation analyzer live in
# ``test_parrot_analyzer.py`` — they DON'T need Ollama and run by default
# so the regex heuristic is regression-tested in CI.
