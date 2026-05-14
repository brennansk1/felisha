"""End-to-end master loop test with a fake (cassette-style) LLM client.

Verifies the loop wiring without needing a real Ollama: discovery →
candidate queue → critic → commit → run_one_experiment → sensitivity →
synthesis. All LLM calls go through a deterministic fake that returns
prepared responses for each schema.

Skipped unless ``RUN_FAKE_LOOP_INTEGRATION`` env var is set, because it
still exercises the actual estimator stack (EconML DML), which takes
several seconds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from causalrag.core.protocol import StudyProtocol


SKIP = not bool(os.environ.get("RUN_FAKE_LOOP_INTEGRATION"))


@pytest.mark.skipif(SKIP, reason="set RUN_FAKE_LOOP_INTEGRATION=1 to run")
def test_fake_client_master_loop_end_to_end(tmp_path: Path) -> None:
    """Run a small fake-LLM /auto end-to-end and verify the protocol."""
    from causalrag.master_loop import LoopConfig, run_master_loop

    # 1. Synthetic dataset — n=400, binary T, continuous Y, true ATE = 2.0
    rng = np.random.default_rng(42)
    n = 400
    age = rng.normal(40, 10, size=n)
    educ = rng.normal(12, 3, size=n)
    propensity_logit = 0.05 * age - 0.1 * educ
    treat = (rng.uniform(size=n) < 1 / (1 + np.exp(-propensity_logit))).astype(int)
    y = 2.0 * treat + 0.5 * age + 0.3 * educ + rng.normal(scale=2.0, size=n)
    df = pd.DataFrame({"age": age, "educ": educ, "treat": treat, "y": y})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    # 2. Fake LLM client — returns canned responses per schema
    from causalrag.llm.ollama_client import OllamaClient, LLMResponse

    class FakeClient:
        model = "fake"

        def parse(
            self,
            *,
            prompt: str,
            schema: Any,
            system: str = "",
            json_schema: dict[str, Any] | None = None,
            extra_options: dict[str, Any] | None = None,
        ) -> LLMResponse:
            name = schema.__name__
            payload: dict[str, Any]
            if name == "InvestigatorReport":
                payload = {
                    "domain_tag": "social_science",
                    "columns": [
                        {
                            "column": "age",
                            "domain_meaning": "age in years",
                            "temporal_position": "pre",
                            "proposed_role": "confounder",
                        },
                        {
                            "column": "educ",
                            "domain_meaning": "years of education",
                            "temporal_position": "pre",
                            "proposed_role": "confounder",
                        },
                        {
                            "column": "treat",
                            "domain_meaning": "binary treatment",
                            "temporal_position": "during",
                            "proposed_role": "treatment",
                        },
                        {
                            "column": "y",
                            "domain_meaning": "continuous outcome",
                            "temporal_position": "post",
                            "proposed_role": "outcome",
                        },
                    ],
                }
            elif name == "DomainExpertBrief":
                payload = {
                    "domain_summary": "Synthetic dataset with binary T and continuous Y.",
                    "candidate_dags": [
                        {
                            "rank": 1,
                            "edges": [
                                ["age", "treat"],
                                ["age", "y"],
                                ["educ", "treat"],
                                ["educ", "y"],
                                ["treat", "y"],
                            ],
                            "rationale": "Standard backdoor",
                            "distinguishing_edges": [],
                        }
                    ],
                    "confounders": [{"name": "age", "reason": "drives both"}, {"name": "educ", "reason": "drives both"}],
                    "mediators": [],
                    "unmeasured_confounders": [],
                    "effect_modifiers": [],
                    "identification_warnings": [],
                }
            elif name == "CandidateQueue":
                payload = {
                    "candidates": [
                        {
                            "candidate_id": "c1",
                            "research_question": "Does treat raise y?",
                            "treatment": "treat",
                            "outcome": "y",
                            "estimand_class": "ATE",
                            "modifiers": [],
                            "mediator": None,
                            "instrument": None,
                            "recommended_method": "python.dml.linear",
                            "impact_rationale": "Headline ATE",
                            "identifiability_rationale": "Backdoor via age + educ",
                            "power_rationale": "n=400 is adequate",
                            "impact_hint": 0.9,
                            "identifiability_hint": 0.85,
                            "power_hint": 0.8,
                        }
                    ],
                }
            elif name == "CriticBatch":
                payload = {
                    "verdicts": [
                        {
                            "candidate_id": "c1",
                            "keep": True,
                            "rejection_reason": None,
                            "revised_recommended_method": None,
                            "risks": [],
                        }
                    ]
                }
            elif name == "DedupePlan":
                payload = {"pruned": [], "merged": []}
            elif name == "ExecutiveSynthesis":
                payload = {
                    "inferred_domain": "social_science",
                    "tldr": "Treatment raises y by ~2 units on average.",
                    "findings": [
                        {
                            "rank": 1,
                            "hypothesis_id": "auto-01",
                            "headline": "Treatment raises y.",
                            "quantified_effect": "ATE ≈ 2.0",
                            "domain_implication": "The intervention works on this synthetic data.",
                            "suggested_next_step": "Confirm in a real RCT.",
                            "confidence": "high",
                            "caveats": [],
                            "estimator_used": "python.dml.linear",
                        }
                    ],
                    "overall_caveats": ["Synthetic data."],
                    "validation_warnings": [],
                }
            elif name == "CrossExperimentAnalysis":
                payload = {
                    "contradictions": [],
                    "reinforcements": [],
                    "chain_narratives": [],
                    "overall_theme": "",
                }
            else:
                # Fallback: empty model_dump for any schema we didn't anticipate.
                # This often fires for downstream LLM augmentations we don't
                # need to mock precisely.
                payload = {}

            try:
                parsed = schema.model_validate(payload)
            except Exception:
                parsed = None
            return LLMResponse(
                parsed=parsed,
                raw=json.dumps(payload),
                model=self.model,
                key="fake",
                retries=0,
                errors=[],
            )

    # 3. Init protocol
    protocol = StudyProtocol(name="fake_loop")

    config = LoopConfig(
        n_experiments=1,
        foundation_allowed=False,
        candidate_queue_size=3,
        propose_k=1,
        critic_enabled=True,
    )

    fake = FakeClient()
    events = list(
        run_master_loop(
            protocol=protocol,
            project_dir=tmp_path,
            dataset_path=csv_path,
            discovery_client=fake,  # type: ignore[arg-type]
            expert_client=fake,  # type: ignore[arg-type]
            config=config,
        )
    )

    # 4. Assertions
    done = next((e for e in events if e.kind == "done"), None)
    assert done is not None, "loop never emitted 'done'"
    assert done.payload["completed"] == 1
    walks = protocol.roadmap_walks
    assert len(walks) == 1
    walk = next(iter(walks.values()))
    assert walk.q7_estimates
    est = walk.q7_estimates[-1]
    # ATE estimate should be close to 2.0
    assert abs(est.point_estimate - 2.0) < 0.6, f"got {est.point_estimate}"
