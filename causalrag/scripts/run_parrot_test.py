"""Driver — run the parrot diagnostic against both datasets.

This is the *script* counterpart to ``tests/integration/test_parrot_harness.py``.
The two share their core logic via ``parrot_harness`` (below); the test
file imports + invokes it so CI can also exercise the path.

The script:

1. Materializes the sign-flipped Lalonde csv (calls into
   ``scripts/parrot_signflip_lalonde.py``) and the synthetic health csv
   (calls into ``scripts/generate_synthetic_health.py``).
2. Scaffolds two CausalRoadmap projects under ``artifacts/parrot_runs/``.
3. Runs the full ``run_auto`` pipeline against each, with live Ollama.
4. Captures
     * ``study.causalrag.yaml`` (final protocol — has the candidate queue
       w/ ``impact_rationale`` strings)
     * ``executive_synthesis.json`` (if produced)
     * estimate cards (``point_estimate``, ``ci_low``, ``ci_high``)
5. Computes the **sign-anticipation ratio** and prints the verdict.

Long-running — expect ~15 min per dataset on a workstation Ollama. The
script does NOT delete its outputs so you can inspect them.

Usage
-----
    # Generate datasets (only needs to run once)
    python scripts/parrot_signflip_lalonde.py --out artifacts/lalonde_signflipped.csv
    python scripts/generate_synthetic_health.py --out artifacts/synthetic_health.csv

    # Run both pipelines
    python scripts/run_parrot_test.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Words that, used in an a-priori rationale, indicate the LLM is
# predicting a direction BEFORE running the experiment. The diagnostic
# is: a parrot prints these confidently; a reasoning model frames its
# rationale neutrally ("the data will reveal whether…").
SIGN_WORDS = (
    "increase",
    "increases",
    "increased",
    "increasing",
    "decrease",
    "decreases",
    "decreased",
    "decreasing",
    "raise",
    "raises",
    "raised",
    "lower",
    "lowers",
    "lowered",
    "lowering",
    "positive",
    "negative",
    "boost",
    "boosts",
    "reduces",
    "reduce",
    "reduced",
    "improves",
    "improved",
    "improving",
    "worsens",
    "worsened",
)

_SIGN_REGEX = re.compile(r"\b(" + "|".join(SIGN_WORDS) + r")\b", re.IGNORECASE)


@dataclass
class ParrotResult:
    """One dataset's diagnostic outcome."""

    dataset_label: str
    n_rationales: int
    n_sign_anticipating: int
    matched_phrases: list[str] = field(default_factory=list)
    point_estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    synthesis_path: Path | None = None
    protocol_path: Path | None = None

    @property
    def sign_anticipation_ratio(self) -> float:
        if self.n_rationales == 0:
            return 0.0
        return self.n_sign_anticipating / self.n_rationales

    def summary(self) -> str:
        ratio = self.sign_anticipation_ratio
        return (
            f"{self.dataset_label}:\n"
            f"  rationales scanned        = {self.n_rationales}\n"
            f"  sign-anticipating         = {self.n_sign_anticipating}\n"
            f"  sign-anticipation ratio   = {ratio:.2%}\n"
            f"  point estimate            = "
            f"{self.point_estimate if self.point_estimate is None else f'{self.point_estimate:+.4f}'}\n"
            f"  95% CI                    = "
            f"[{self.ci_low if self.ci_low is None else f'{self.ci_low:+.4f}'}, "
            f"{self.ci_high if self.ci_high is None else f'{self.ci_high:+.4f}'}]"
        )


def count_sign_anticipating(rationales: Iterable[str]) -> tuple[int, int, list[str]]:
    """Return (total, n_with_sign_word, matched_phrases)."""
    total = 0
    hits = 0
    matched: list[str] = []
    for r in rationales:
        if not r:
            continue
        total += 1
        m = _SIGN_REGEX.search(r)
        if m:
            hits += 1
            matched.append(r.strip()[:240])
    return total, hits, matched


def collect_rationales(protocol_yaml: Path) -> list[str]:
    """Read every ``impact_rationale`` string off the persisted protocol.

    The master loop persists the candidate queue (including each
    candidate's ``impact_rationale``) onto the protocol YAML. Foundation
    follow-up proposals land in the decision ledger with their rationale
    in the ``note`` field. We slurp both.
    """
    from causalrag.core.protocol import StudyProtocol  # local import — keep CLI light

    proto = StudyProtocol.read_yaml(protocol_yaml)
    rationales: list[str] = []
    for entry in proto.candidate_queue:
        for key in ("impact_rationale", "identifiability_rationale", "power_rationale"):
            val = entry.get(key)
            if isinstance(val, str):
                rationales.append(val)
    for dec in proto.decision_ledger:
        # The master loop stores foundation-followup commits in the ledger
        # with the proposal's text in the note. Cheap heuristic — match
        # decisions whose phase looks proposal-shaped.
        if dec.note and dec.phase in {"hypothesize", "master_loop", "foundation"}:
            rationales.append(dec.note)
    return rationales


def extract_estimate_from_synthesis(synthesis_path: Path) -> tuple[float | None, float | None, float | None]:
    """Pick a primary point estimate + CI off ``executive_synthesis.json``.

    Returns ``(point, ci_low, ci_high)``. If the synthesis is absent or
    malformed, returns (None, None, None) and the caller falls back to
    reading the protocol's roadmap_walks directly.
    """
    if not synthesis_path.exists():
        return None, None, None
    try:
        data = json.loads(synthesis_path.read_text())
    except Exception:
        return None, None, None
    # Search the synthesis JSON for a headline estimate. Schema may evolve;
    # match defensively.
    for key in ("primary_estimate", "headline", "estimate"):
        node = data.get(key)
        if isinstance(node, dict):
            return (
                _maybe_float(node.get("point") or node.get("point_estimate")),
                _maybe_float(node.get("ci_low") or node.get("ci_lower")),
                _maybe_float(node.get("ci_high") or node.get("ci_upper")),
            )
    return None, None, None


def extract_estimate_from_protocol(protocol_yaml: Path) -> tuple[float | None, float | None, float | None]:
    """Fallback: read the first estimate off the protocol's roadmap walks."""
    from causalrag.core.protocol import StudyProtocol

    proto = StudyProtocol.read_yaml(protocol_yaml)
    for walk in proto.roadmap_walks.values():
        if walk.q7_estimates:
            est = walk.q7_estimates[0]
            return est.point_estimate, est.ci_low, est.ci_high
    return None, None, None


def _maybe_float(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def diagnose_run(
    *,
    dataset_label: str,
    protocol_yaml: Path,
    synthesis_path: Path,
) -> ParrotResult:
    """Combine sign-anticipation + estimate extraction into one verdict."""
    rationales = collect_rationales(protocol_yaml)
    total, hits, matched = count_sign_anticipating(rationales)
    point, lo, hi = extract_estimate_from_synthesis(synthesis_path)
    if point is None:
        point, lo, hi = extract_estimate_from_protocol(protocol_yaml)
    return ParrotResult(
        dataset_label=dataset_label,
        n_rationales=total,
        n_sign_anticipating=hits,
        matched_phrases=matched,
        point_estimate=point,
        ci_low=lo,
        ci_high=hi,
        synthesis_path=synthesis_path,
        protocol_path=protocol_yaml,
    )


def run_pipeline(
    *,
    dataset_path: Path,
    project_dir: Path,
    research_question: str,
    base_url: str = "http://127.0.0.1:11434",
) -> None:
    """Invoke ``run_auto`` against a dataset, persisting all outputs into
    ``project_dir``."""
    from causalrag.auto import run_auto
    from causalrag.cli.doctor import run_doctor
    from causalrag.core.protocol import StudyProtocol
    from causalrag.llm.ollama_client import OllamaClient
    from causalrag.llm.selector import recommend

    project_dir.mkdir(parents=True, exist_ok=True)
    proto_path = project_dir / "study.causalrag.yaml"
    if not proto_path.exists():
        from causalrag.cli.main import _scaffold_project

        _scaffold_project(project_dir, name=dataset_path.stem, tier="academic")
    protocol = StudyProtocol.read_yaml(proto_path)

    profile = run_doctor(base_url=base_url)
    slots, _ = recommend(profile)
    cassette_dir = project_dir / ".causalrag" / "cassettes"
    cassette_dir.mkdir(parents=True, exist_ok=True)
    discovery_client = OllamaClient(
        model=slots.discovery,
        base_url=base_url,
        cassette_dir=cassette_dir,
        allow_live=True,
    )
    expert_client = (
        OllamaClient(
            model=slots.hypothesize,
            base_url=base_url,
            cassette_dir=cassette_dir,
            allow_live=True,
        )
        if slots.hypothesize != slots.discovery
        else None
    )

    for event in run_auto(
        protocol=protocol,
        project_dir=project_dir,
        dataset_path=dataset_path,
        research_question=research_question,
        discovery_client=discovery_client,
        expert_client=expert_client,
    ):
        # Echo events to stdout for visibility — the CLI does the pretty
        # version; we keep this plain so it's grep-friendly.
        prefix = {"phase_start": "▸", "phase_end": "✓", "error": "✗", "card": " "}.get(event.kind, "·")
        print(f"  {prefix} [{event.phase}] {event.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/parrot_runs"),
        help="Output root for the two project directories + datasets.",
    )
    parser.add_argument(
        "--skip-lalonde",
        action="store_true",
        help="Skip the sign-flipped Lalonde leg.",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip the synthetic-health leg.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL (default: http://127.0.0.1:11434).",
    )
    args = parser.parse_args(argv)

    artifacts: Path = args.artifacts
    artifacts.mkdir(parents=True, exist_ok=True)

    results: list[ParrotResult] = []

    if not args.skip_lalonde:
        from scripts.parrot_signflip_lalonde import build_signflipped, load_nsw  # type: ignore

        nsw = load_nsw()
        flipped = build_signflipped(nsw)
        flipped_csv = artifacts / "lalonde_signflipped.csv"
        flipped.to_csv(flipped_csv, index=False)

        project = artifacts / "lalonde_signflipped"
        print(f"\n=== Running sign-flipped Lalonde (project: {project}) ===")
        run_pipeline(
            dataset_path=flipped_csv,
            project_dir=project,
            research_question=(
                "Estimate the average effect of the NSW training program "
                "(treat) on 1978 earnings (re78) in this dataset."
            ),
            base_url=args.base_url,
        )
        results.append(
            diagnose_run(
                dataset_label="sign-flipped Lalonde",
                protocol_yaml=project / "study.causalrag.yaml",
                synthesis_path=project / "executive_synthesis.json",
            )
        )

    if not args.skip_health:
        from scripts.generate_synthetic_health import generate  # type: ignore

        df, truth = generate()
        health_csv = artifacts / "synthetic_health.csv"
        df.to_csv(health_csv, index=False)
        (artifacts / "synthetic_health.truth.json").write_text(json.dumps(truth, indent=2))

        project = artifacts / "synthetic_health"
        print(f"\n=== Running synthetic-health smoke (project: {project}) ===")
        run_pipeline(
            dataset_path=health_csv,
            project_dir=project,
            research_question=(
                "Estimate the average effect of statin adherence on 5-year "
                "cardiac event risk in this cohort."
            ),
            base_url=args.base_url,
        )
        results.append(
            diagnose_run(
                dataset_label="synthetic health",
                protocol_yaml=project / "study.causalrag.yaml",
                synthesis_path=project / "executive_synthesis.json",
            )
        )

    print("\n========== PARROT DIAGNOSTIC RESULTS ==========")
    for r in results:
        print(r.summary())
        print()

    # Verdict — sign-flipped Lalonde is the falsification leg.
    lalonde = next((r for r in results if "Lalonde" in r.dataset_label), None)
    if lalonde is not None:
        ratio = lalonde.sign_anticipation_ratio
        sign_ok = lalonde.point_estimate is not None and lalonde.point_estimate < 0
        ratio_ok = ratio < 0.30
        if sign_ok and ratio_ok:
            print(f"VERDICT: PASS — negative estimate ({lalonde.point_estimate:+.2f}) AND sign-anticipation {ratio:.0%} < 30%.")
            return 0
        print(
            "VERDICT: FAIL — "
            f"negative-estimate={sign_ok} (point={lalonde.point_estimate}), "
            f"low-sign-anticipation={ratio_ok} (ratio={ratio:.0%})."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
