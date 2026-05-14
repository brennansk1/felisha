"""Tutorial mode for the CausalRoadmap TUI.

`causalrag tui --tutorial` walks a new user through the full roadmap on a
packaged dataset (Lalonde NSW job-training, or an IHDP-flavored
semi-synthetic CATE benchmark). Each step pairs a markdown-formatted
prompt with the exact slash command the user should run and a one-line
hint that explains *why*.

The module is intentionally self-contained and has zero hard dependencies
on Textual — the TUI app pulls in `render_tutorial_step` and the
`Tutorial` dataclass and decides how to surface them. This keeps the
business logic unit-testable without spinning up the full app.

Dataset loaders return ``(df, info)``:

- ``df`` is a pandas DataFrame ready to be written to CSV.
- ``info`` is a dict with metadata: ``name``, ``treatment``, ``outcome``,
  ``true_ate`` (when known), ``source`` (``"causaldata"`` or
  ``"synthetic"``), and a short ``description``.

If the ``causaldata`` package is unavailable the Lalonde loader falls
back to a small synthetic Lalonde-shaped frame so the tutorial still
runs end-to-end on a clean install. The IHDP loader is always synthetic
(IHDP is itself a semi-synthetic benchmark by construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


# --- Dataset loaders -------------------------------------------------------


def _synthetic_lalonde(n: int = 445, seed: int = 1975) -> pd.DataFrame:
    """Lalonde-NSW-shaped synthetic frame used when ``causaldata`` is absent.

    Columns mirror the canonical Dehejia-Wahba NSW sample
    (``treat``, ``age``, ``educ``, ``black``, ``hisp``, ``married``,
    ``nodegree``, ``re74``, ``re75``, ``re78``) so the same downstream
    treatment/outcome hints work whether the user has the real data or
    not. The true ATE is positive (~1700) to match the real Dehejia-Wahba
    result.
    """
    rng = np.random.default_rng(seed)
    treat = rng.binomial(1, 0.4, size=n).astype(float)
    age = rng.integers(17, 55, size=n).astype(float)
    educ = rng.integers(3, 16, size=n).astype(float)
    black = rng.binomial(1, 0.84, size=n).astype(float)
    hisp = rng.binomial(1, 0.06, size=n).astype(float)
    married = rng.binomial(1, 0.17, size=n).astype(float)
    nodegree = (educ < 12).astype(float)
    re74 = np.clip(rng.normal(2100, 5000, size=n), 0, None)
    re75 = np.clip(0.6 * re74 + rng.normal(500, 3000, size=n), 0, None)
    true_ate = 1700.0
    re78 = np.clip(
        0.5 * re74
        + 0.4 * re75
        + 120.0 * age
        + 350.0 * educ
        + treat * true_ate
        + rng.normal(0, 4500, size=n),
        0,
        None,
    )
    return pd.DataFrame(
        {
            "treat": treat,
            "age": age,
            "educ": educ,
            "black": black,
            "hisp": hisp,
            "married": married,
            "nodegree": nodegree,
            "re74": re74,
            "re75": re75,
            "re78": re78,
        }
    )


def load_lalonde() -> tuple[pd.DataFrame, dict]:
    """Load Lalonde NSW via ``causaldata`` with a synthetic fallback."""
    info: dict = {
        "name": "lalonde",
        "treatment": "treat",
        "outcome": "re78",
        "description": (
            "NSW job-training program (Dehejia-Wahba sample) — effect of "
            "training on 1978 earnings."
        ),
    }
    try:  # pragma: no cover - exercised only when causaldata is installed
        import causaldata  # type: ignore

        df = causaldata.nsw_mixtape.load_pandas().data
        if "data_id" in df.columns:
            df = df.drop(columns=["data_id"])
        df = df.reset_index(drop=True)
        info["source"] = "causaldata"
        info["true_ate"] = None  # real data — truth unknown
        return df, info
    except Exception:
        df = _synthetic_lalonde()
        info["source"] = "synthetic"
        info["true_ate"] = 1700.0
        return df, info


def load_ihdp() -> tuple[pd.DataFrame, dict]:
    """Load an IHDP-flavored semi-synthetic CATE benchmark.

    Real-shape covariates with a synthetic outcome whose CATE is
    heterogeneous along the first two continuous covariates. Fixed seed
    so the "ground truth" ATE is exact across runs.
    """
    rng = np.random.default_rng(11)
    n = 747  # canonical IHDP sample size
    p_cont = 6
    p_bin = 19
    x_cont = rng.normal(size=(n, p_cont))
    x_bin = rng.binomial(1, 0.5, size=(n, p_bin))
    treat = rng.binomial(
        1, 1 / (1 + np.exp(-0.3 * x_cont.sum(axis=1))), size=n
    ).astype(float)
    cate_per_row = 3.0 + 1.5 * x_cont[:, 0] + 0.5 * x_cont[:, 1]
    base = 2.0 * x_cont.sum(axis=1) + 0.3 * x_bin.sum(axis=1)
    y = base + treat * cate_per_row + rng.normal(scale=1.0, size=n)
    true_ate = float(cate_per_row.mean())
    df = pd.DataFrame(
        {
            **{f"x_cont_{i}": x_cont[:, i] for i in range(p_cont)},
            **{f"x_bin_{i}": x_bin[:, i] for i in range(p_bin)},
            "treat": treat,
            "y": y,
        }
    )
    info = {
        "name": "ihdp",
        "treatment": "treat",
        "outcome": "y",
        "description": (
            "IHDP-flavored semi-synthetic CATE benchmark — heterogeneous "
            "treatment effect along x_cont_0/x_cont_1."
        ),
        "source": "synthetic",
        "true_ate": true_ate,
    }
    return df, info


# --- Dataclasses -----------------------------------------------------------


@dataclass
class TutorialStep:
    """A single step in the guided tour."""

    name: str
    phase: int
    prompt: str  # markdown prompt shown to the user
    expected_command: str  # the slash command they should run
    hint: str
    automated: bool = False  # if True, run for them after a confirmation


@dataclass
class Tutorial:
    """A complete tutorial walk."""

    name: str
    description: str
    dataset_loader: Callable[[], tuple[pd.DataFrame, dict]]
    steps: list[TutorialStep] = field(default_factory=list)
    cleanup: Callable[[], None] | None = None


# --- Lalonde tutorial ------------------------------------------------------


LALONDE_TUTORIAL: Tutorial = Tutorial(
    name="lalonde",
    description="NSW job-training program causal effect on 1978 earnings.",
    dataset_loader=load_lalonde,
    steps=[
        TutorialStep(
            name="init",
            phase=0,
            prompt=(
                "## Step 1 — Initialize a project\n\n"
                "Every CausalRoadmap run lives inside a `StudyProtocol` "
                "(`study.yaml`). We'll scaffold one for the Lalonde "
                "tutorial.\n\n"
                "**Try:** `/init lalonde-tutorial --tier=academic`"
            ),
            expected_command="/init lalonde-tutorial --tier=academic",
            hint=(
                "The protocol pins your research question, dataset, "
                "discovery DAG, and every estimate so the run is reproducible."
            ),
            automated=True,
        ),
        TutorialStep(
            name="discover",
            phase=1,
            prompt=(
                "## Step 2 — Discover the data\n\n"
                "Profile the columns, infer roles, and (with an LLM) sketch "
                "candidate DAGs.\n\n"
                "**Try:** `/discover lalonde.csv --treatment treat "
                "--outcome re78 --no-llm`"
            ),
            expected_command=(
                "/discover lalonde.csv --treatment treat --outcome re78 --no-llm"
            ),
            hint=(
                "`--no-llm` keeps the tutorial offline-deterministic; drop "
                "it once you have Ollama running to see Layer-1 + Layer-3 "
                "annotations."
            ),
        ),
        TutorialStep(
            name="hypothesize",
            phase=3,
            prompt=(
                "## Step 3 — Generate hypotheses\n\n"
                "Rank the treatment/outcome pairs worth estimating "
                "(here, just `treat → re78`).\n\n"
                "**Try:** `/hypothesize --mode automated`"
            ),
            expected_command="/hypothesize --mode automated",
            hint=(
                "Automated mode reads the discovery report and proposes "
                "an ATE on every admissible pair, scored by impact."
            ),
        ),
        TutorialStep(
            name="estimate",
            phase=4,
            prompt=(
                "## Step 4 — Estimate the ATE\n\n"
                "Walk the Roadmap: identify the effect (Q5), pick an "
                "estimator (Q6), fit, refute.\n\n"
                "**Try:** `/estimate --treatment treat --outcome re78`"
            ),
            expected_command="/estimate --treatment treat --outcome re78",
            hint=(
                "Look for the refutation row — placebo / random-common-cause "
                "/ subset-bootstrap should all PASS on a clean run."
            ),
        ),
        TutorialStep(
            name="sensitivity",
            phase=5,
            prompt=(
                "## Step 5 — Stress-test it\n\n"
                "How big would an unmeasured confounder have to be to "
                "explain away the effect?\n\n"
                "**Try:** `/sensitivity --treatment treat --outcome re78`"
            ),
            expected_command="/sensitivity --treatment treat --outcome re78",
            hint=(
                "Green = robust; yellow = fragile; red = explained away by "
                "plausible unmeasured confounding."
            ),
        ),
        TutorialStep(
            name="report",
            phase=6,
            prompt=(
                "## Step 6 — Render the report\n\n"
                "Bundle every card into a shareable HTML report.\n\n"
                "**Try:** `/report --format html`"
            ),
            expected_command="/report --format html",
            hint=(
                "The report lives under `reports/` and embeds the full "
                "Roadmap walk plus refutations and sensitivity."
            ),
            automated=True,
        ),
    ],
)


# --- IHDP tutorial ---------------------------------------------------------


IHDP_TUTORIAL: Tutorial = Tutorial(
    name="ihdp",
    description="IHDP semi-synthetic CATE benchmark.",
    dataset_loader=load_ihdp,
    steps=[
        TutorialStep(
            name="init",
            phase=0,
            prompt=(
                "## Step 1 — Scaffold an IHDP project\n\n"
                "We'll work the Roadmap on a semi-synthetic frame whose "
                "true ATE is known.\n\n"
                "**Try:** `/init ihdp-tutorial --tier=academic`"
            ),
            expected_command="/init ihdp-tutorial --tier=academic",
            hint=(
                "Because the outcome is synthetic, we can compare our "
                "point estimate against ground truth at the end."
            ),
            automated=True,
        ),
        TutorialStep(
            name="discover",
            phase=1,
            prompt=(
                "## Step 2 — Profile 25 covariates\n\n"
                "IHDP has 6 continuous + 19 binary covariates. Discovery "
                "will flag them and propose a DAG skeleton.\n\n"
                "**Try:** `/discover ihdp.csv --treatment treat "
                "--outcome y --no-llm`"
            ),
            expected_command=(
                "/discover ihdp.csv --treatment treat --outcome y --no-llm"
            ),
            hint=(
                "With this many covariates the auto variable-selection "
                "step will prune redundant adjustments before fitting."
            ),
        ),
        TutorialStep(
            name="hypothesize",
            phase=3,
            prompt=(
                "## Step 3 — Pose the CATE hypothesis\n\n"
                "Effect heterogeneity is the point of IHDP — let's queue "
                "a CATE estimand.\n\n"
                "**Try:** `/hypothesize --mode automated`"
            ),
            expected_command="/hypothesize --mode automated",
            hint=(
                "The hypothesis ranker boosts CATE proposals when the "
                "discovery audit detects effect modifiers."
            ),
        ),
        TutorialStep(
            name="estimate",
            phase=4,
            prompt=(
                "## Step 4 — Fit a causal forest\n\n"
                "With ≥3 modifiers and n ≥ 500 the dispatcher should "
                "route to a forest-based estimator.\n\n"
                "**Try:** `/estimate --treatment treat --outcome y "
                "--estimand CATE`"
            ),
            expected_command=(
                "/estimate --treatment treat --outcome y --estimand CATE"
            ),
            hint=(
                "True ATE here is ≈ 3.0 — your point estimate should "
                "land within roughly ± 0.5."
            ),
        ),
        TutorialStep(
            name="sensitivity",
            phase=5,
            prompt=(
                "## Step 5 — Robustness check\n\n"
                "Run E-value + sensemakr against the headline ATE.\n\n"
                "**Try:** `/sensitivity --treatment treat --outcome y`"
            ),
            expected_command="/sensitivity --treatment treat --outcome y",
            hint=(
                "On a well-identified synthetic DGP the E-value should "
                "be comfortably > 2 and the verdict should be green."
            ),
        ),
        TutorialStep(
            name="report",
            phase=6,
            prompt=(
                "## Step 6 — Export the walk\n\n"
                "Render the full Roadmap walk to HTML.\n\n"
                "**Try:** `/report --format html`"
            ),
            expected_command="/report --format html",
            hint=(
                "Open the report in a browser — every Q-step has its "
                "own card with provenance back to discovery."
            ),
            automated=True,
        ),
    ],
)


# --- Registry --------------------------------------------------------------


_TUTORIALS: dict[str, Tutorial] = {
    LALONDE_TUTORIAL.name: LALONDE_TUTORIAL,
    IHDP_TUTORIAL.name: IHDP_TUTORIAL,
}


def list_tutorials() -> list[str]:
    """Return the names of all registered tutorials, in stable order."""
    return sorted(_TUTORIALS.keys())


def get_tutorial(name: str) -> Tutorial:
    """Look up a tutorial by name.

    Raises ``KeyError`` with a helpful message when the name is unknown.
    """
    try:
        return _TUTORIALS[name]
    except KeyError as exc:
        available = ", ".join(list_tutorials())
        raise KeyError(
            f"Unknown tutorial {name!r}. Available: {available}"
        ) from exc


# --- Rendering -------------------------------------------------------------


def render_tutorial_step(step: TutorialStep) -> str:
    """Render a single step as a markdown hint card for the TUI.

    The output is plain markdown (no Rich/Textual types) so callers can
    pipe it into a Markdown widget, a Rich console, or even stdout.
    """
    auto_chip = " *(automated)*" if step.automated else ""
    lines: list[str] = [
        f"### Tutorial · {step.name} · phase {step.phase}{auto_chip}",
        "",
        step.prompt.strip(),
        "",
        f"**Run:** `{step.expected_command}`",
        "",
        f"> Hint — {step.hint}",
    ]
    return "\n".join(lines)


__all__ = [
    "IHDP_TUTORIAL",
    "LALONDE_TUTORIAL",
    "Tutorial",
    "TutorialStep",
    "get_tutorial",
    "list_tutorials",
    "load_ihdp",
    "load_lalonde",
    "render_tutorial_step",
]
