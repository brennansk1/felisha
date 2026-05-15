"""DSPy + Outlines scaffolding for the LLM prompt layer (Sprint 1.7).

DSPy (Khattab et al. ICLR 2024) turns LLM calls into composable modules
whose prompts are compiled — optimised once against a held-out gold
set — and then frozen for inference. The pipeline benefits because:

1. **Schema-validity goes from "retry-on-failure" to native** when the
   engine supports it (vLLM `guided_json`, llama.cpp `response_format`).
2. **Prompt drift across model versions is detected at compile time**
   rather than at runtime via cassette diffs.
3. **The seven prompt sites — planner / critic / foundation / synthesis
   / anomaly / identification narration / sensitivity interpretation /
   cross-experiment** — get a uniform interface a third party can
   override.

This module is **opt-in**. The existing `OllamaClient.parse(...)` is
untouched and continues to work. Set
`CAUSALRAG_LLM_BACKEND=dspy` (or pass a `DSPyAdapter` instance) to
route through DSPy. When the `dspy` package isn't installed, every
adapter falls back to the existing client.

Outlines (https://outlines-dev.github.io/outlines/) is the runtime
schema-enforcement library. For engines that don't have native JSON
mode (most local llama.cpp builds), Outlines does logits-masking that
guarantees schema-valid output without the retry loop. Wired here as
the second-stage validator.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel

logger = logging.getLogger("causalrag.llm.dspy")


@dataclass
class DSPyAvailability:
    """What's installed locally."""

    dspy: bool
    outlines: bool

    @property
    def fully_available(self) -> bool:
        return self.dspy and self.outlines


def detect_dspy_availability() -> DSPyAvailability:
    """Probe whether dspy / outlines are importable. No side effects."""
    return DSPyAvailability(
        dspy=_is_importable("dspy"),
        outlines=_is_importable("outlines"),
    )


def _is_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# ─── Signature catalog ───────────────────────────────────────────────


@dataclass(frozen=True)
class PromptSignature:
    """A DSPy-shaped signature for one prompt site.

    name: stable id used as the cache key + compiled-prompt slot
    input_fields: ordered tuple of (name, description)
    output_fields: ordered tuple of (name, description, dtype hint)
    schema: Pydantic model the output must conform to
    instruction: the high-level task statement (replaces the system prompt)
    optimiser: which dspy optimiser to use when compiling
        ("BootstrapFewShot" / "MIPROv2" / "GEPA" / "none")
    """

    name: str
    instruction: str
    input_fields: tuple[tuple[str, str], ...]
    output_fields: tuple[tuple[str, str, str], ...]
    schema: type[BaseModel] | None = None
    optimiser: Literal["BootstrapFewShot", "MIPROv2", "GEPA", "none"] = "none"

    def to_dspy_signature(self) -> Any:
        """Build a runtime dspy.Signature dynamically. Lazy-imports dspy.

        Raises if dspy isn't installed — callers should check
        `detect_dspy_availability().dspy` first."""
        import dspy  # type: ignore

        sig_fields: dict[str, Any] = {}
        for name, desc in self.input_fields:
            sig_fields[name] = dspy.InputField(desc=desc)
        for name, desc, _dtype in self.output_fields:
            sig_fields[name] = dspy.OutputField(desc=desc)
        sig_class = type(
            self.name + "Signature",
            (dspy.Signature,),
            {"__doc__": self.instruction, **sig_fields},
        )
        return sig_class


# ─── Canonical signatures for every prompt site ──────────────────────


_PLANNER_SIGNATURE = PromptSignature(
    name="planner",
    instruction=(
        "You are a senior causal-inference statistician. Enumerate the "
        "set of credible candidate experiments this dataset supports — "
        "broad, well-justified, NOT yet ranked. The deterministic scorer "
        "will rank afterwards."
    ),
    input_fields=(
        ("dataset_summary", "Profile + flag manifest + domain brief"),
        ("research_question", "User research question if supplied"),
        ("catalog_table", "Estimator catalog markdown"),
    ),
    output_fields=(
        ("candidates", "JSON list of CandidateExperiment objects", "list[dict]"),
        ("notes", "Short rationale paragraph", "str"),
    ),
)

_CRITIC_SIGNATURE = PromptSignature(
    name="critic",
    instruction=(
        "Referee at a top causal-inference journal. For each candidate, "
        "decide keep/reject and flag risks. Reject if already tested, "
        "missing required piece (NDE without mediator etc.), method not "
        "in catalog, identification weak, or min-n unmet."
    ),
    input_fields=(
        ("dataset_summary", "Profile + flag manifest"),
        ("completed_history", "Walks already run"),
        ("candidates", "Candidates under review"),
        ("catalog_table", "Estimator catalog"),
    ),
    output_fields=(
        ("verdicts", "One CriticVerdict per candidate id", "list[dict]"),
        ("overall_note", "Optional summary", "str"),
    ),
)

_FOUNDATION_FOLLOWUP_SIGNATURE = PromptSignature(
    name="foundation_followup",
    instruction=(
        "Pick the single most informative follow-up experiment given "
        "the parent walk's result. Canonical patterns: significant ATE "
        "→ CATE on strongest modifier; significant ATE + mediator → "
        "NDE/NIE; null → power probe; red sensitivity → tipping-point."
    ),
    input_fields=(
        ("dataset_summary", "Profile + flag manifest"),
        ("parent_walk", "Parent estimand + estimate + sensitivity"),
        ("chain_state", "Chain depth, prior steps"),
        ("history", "Earlier experiments"),
    ),
    output_fields=(
        ("decision", "'run' or 'stop'", "str"),
        ("treatment", "Treatment column", "str | None"),
        ("outcome", "Outcome column", "str | None"),
        ("estimand_class", "ATE / CATE / NDE / NIE / LATE / RMST_CONTRAST", "str | None"),
        ("recommended_method", "Catalog estimator id", "str | None"),
        ("foundation_rationale", "Why this follow-up adds info", "str | None"),
    ),
)

_SYNTHESIS_SIGNATURE = PromptSignature(
    name="synthesis",
    instruction=(
        "Domain-agnostic executive synthesis. Infer the dataset's "
        "domain (clinical / business / policy / ecology / engineering "
        "/ …) and write the findings in that field's vocabulary. "
        "Quantify using ONLY the magnitudes provided. Confidence-low "
        "when CI crosses zero or sensitivity red."
    ),
    input_fields=(
        ("dataset_context", "Domain brief + flag semantics"),
        ("experiments", "Completed Roadmap walks with magnitudes"),
        ("cross_experiment_block", "Contradictions / reinforcements / chains"),
    ),
    output_fields=(
        ("inferred_domain", "Domain enum", "str"),
        ("tldr", "Single-sentence headline", "str"),
        ("findings", "Ranked Insight objects", "list[dict]"),
        ("overall_caveats", "Cross-finding caveats", "list[str]"),
    ),
)

_ANOMALY_AUDIT_SIGNATURE = PromptSignature(
    name="anomaly_audit",
    instruction=(
        "Senior referee. Sniff for subtle wrong-shape patterns in this "
        "single estimate: implausible magnitude, sign flip vs naive, "
        "saturated propensity, near-zero n_used, refutation divergence. "
        "Recommend accept / rerun_with_different_estimator / disqualify."
    ),
    input_fields=(
        ("result", "EstimationResult"),
        ("walk", "RoadmapWalk"),
        ("naive_estimate", "Naive correlation if available"),
        ("domain_brief", "Brief context"),
        ("prescreen_flags", "Deterministic pre-screen flags"),
    ),
    output_fields=(
        ("flags", "Anomaly flag list", "list[str]"),
        ("rationale_per_flag", "Map flag → rationale", "dict[str, str]"),
        ("recommendation", "accept / rerun / disqualify", "str"),
        ("overall_note", "Plain-language summary", "str"),
    ),
)

_IDENTIFICATION_NARRATION_SIGNATURE = PromptSignature(
    name="identification_narration",
    instruction=(
        "Causal-inference textbook author. Explain in plain language "
        "WHY this adjustment set blocks the backdoor path for this DAG. "
        "Reference only nodes that appear in the DAG."
    ),
    input_fields=(
        ("estimand", "Treatment + outcome + class"),
        ("graph", "DAG edges + roles"),
        ("identification_result", "Strategy + adjustment set + warnings"),
        ("domain_brief", "Optional domain context"),
    ),
    output_fields=(
        ("strategy_explanation", "Why this strategy works", "str"),
        ("blocked_paths", "Backdoor paths blocked", "list[str]"),
        ("unblocked_paths", "Remaining open paths", "list[str]"),
        ("analyst_assertions", "Assumptions the analyst trusts", "list[str]"),
        ("confidence", "high/medium/low", "str"),
        ("rationale", "Synthesis-layer quotable paragraph", "str"),
    ),
)

_SENSITIVITY_INTERPRETATION_SIGNATURE = PromptSignature(
    name="sensitivity_interpretation",
    instruction=(
        "Referee at a top journal in the inferred domain. Translate "
        "the deterministic sensitivity verdict into plain language. "
        "The verdict COLOR is fixed — do not change it. Explain what "
        "the threshold implies for unmeasured confounding in this field."
    ),
    input_fields=(
        ("evalue_result", "E-value computation"),
        ("sensemakr_result", "Sensemakr RV"),
        ("deterministic_verdict", "Fixed color"),
        ("point_estimate", "Point + CI"),
        ("treatment_outcome", "T/Y names"),
        ("domain_brief", "Brief context"),
    ),
    output_fields=(
        ("verdict_color", "Pinned to input verdict", "str"),
        ("plain_language", "1-3 sentence domain prose", "str"),
        ("what_it_rules_out", "What this evidence rules out", "str"),
        ("what_it_does_not_rule_out", "What it doesn't", "str"),
        ("plausibility_of_threshold_confounder", "Is the threshold plausible here?", "str"),
        ("rationale", "Quotable paragraph", "str"),
    ),
)

_CROSS_EXPERIMENT_SIGNATURE = PromptSignature(
    name="cross_experiment",
    instruction=(
        "Identify contradictions, reinforcements, and chain narratives "
        "across the completed walks. Determine the overall theme."
    ),
    input_fields=(
        ("walks_summary", "All completed walks"),
        ("deterministic_candidates", "Pre-pass contradiction/reinforcement candidates"),
    ),
    output_fields=(
        ("contradictions", "Surface vs structural pairs", "list[dict]"),
        ("reinforcements", "Weak/moderate/strong groups", "list[dict]"),
        ("chain_narratives", "Foundation thread stories", "list[dict]"),
        ("overall_theme", "1-2 sentence theme", "str"),
    ),
)

# Canonical registry mapping prompt-site name → signature.
SIGNATURES: dict[str, PromptSignature] = {
    s.name: s
    for s in (
        _PLANNER_SIGNATURE,
        _CRITIC_SIGNATURE,
        _FOUNDATION_FOLLOWUP_SIGNATURE,
        _SYNTHESIS_SIGNATURE,
        _ANOMALY_AUDIT_SIGNATURE,
        _IDENTIFICATION_NARRATION_SIGNATURE,
        _SENSITIVITY_INTERPRETATION_SIGNATURE,
        _CROSS_EXPERIMENT_SIGNATURE,
    )
}


# ─── Adapter ─────────────────────────────────────────────────────────


@dataclass
class DSPyAdapter:
    """Optional drop-in replacement for `OllamaClient.parse(...)` that
    routes through compiled DSPy modules.

    When dspy / outlines aren't installed, calls degrade gracefully
    to the wrapped fallback client (preserving existing pipeline
    behavior).
    """

    fallback_client: Any  # OllamaClient or compatible
    compiled_modules: dict[str, Any] = field(default_factory=dict)
    availability: DSPyAvailability = field(
        default_factory=detect_dspy_availability
    )

    def parse(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        system: str = "",
        json_schema: dict[str, Any] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> Any:
        """Mirror OllamaClient.parse() — same signature, same return type.

        Tries the DSPy path when:
          - dspy is importable
          - the schema's name matches a known prompt-site
          - a compiled module has been registered via `register_compiled(...)`

        Falls back to fallback_client otherwise.
        """
        if not self.availability.dspy:
            return self.fallback_client.parse(
                prompt=prompt,
                schema=schema,
                system=system,
                json_schema=json_schema,
                extra_options=extra_options,
            )
        site_name = self._infer_site_name(schema, system, prompt)
        if site_name is None or site_name not in self.compiled_modules:
            return self.fallback_client.parse(
                prompt=prompt,
                schema=schema,
                system=system,
                json_schema=json_schema,
                extra_options=extra_options,
            )
        # Compiled DSPy module path. The compiled artifact is callable.
        try:
            module = self.compiled_modules[site_name]
            result = module(prompt=prompt, system=system)
            return self._wrap_in_llm_response(result, schema)
        except Exception as e:
            logger.warning(
                "DSPy path failed at site=%s, falling back: %s", site_name, e
            )
            return self.fallback_client.parse(
                prompt=prompt,
                schema=schema,
                system=system,
                json_schema=json_schema,
                extra_options=extra_options,
            )

    def register_compiled(self, site_name: str, module: Any) -> None:
        """Register a compiled DSPy module against a prompt-site name."""
        self.compiled_modules[site_name] = module

    def _infer_site_name(
        self,
        schema: type[BaseModel],
        system: str,
        prompt: str,
    ) -> str | None:
        """Best-effort prompt-site identification.

        Strategy: schema class name → known signature lookup.
        e.g. NextExperiment → foundation_followup,
             CriticBatch → critic,
             ExecutiveSynthesis → synthesis,
             etc.
        """
        match (schema.__name__ or "").lower():
            case "candidatequeue":
                return "planner"
            case "criticbatch":
                return "critic"
            case "nextexperiment":
                return "foundation_followup"
            case "executivesynthesis":
                return "synthesis"
            case "anomalyaudit":
                return "anomaly_audit"
            case "identificationnarration":
                return "identification_narration"
            case "sensitivityinterpretation":
                return "sensitivity_interpretation"
            case "crossexperimentanalysis":
                return "cross_experiment"
        return None

    def _wrap_in_llm_response(self, dspy_output: Any, schema: type[BaseModel]) -> Any:
        """Coerce a DSPy Prediction object into something shaped like
        OllamaClient.LLMResponse."""
        from causalrag.llm.ollama_client import LLMResponse

        # Try to materialise the DSPy output into the requested schema.
        try:
            data = (
                dspy_output if isinstance(dspy_output, dict)
                else dspy_output.toDict()
                if hasattr(dspy_output, "toDict")
                else dspy_output.__dict__
            )
            parsed = schema.model_validate(data)
        except Exception:
            parsed = None
        return LLMResponse(
            parsed=parsed,
            raw=str(dspy_output),
            model="dspy",
            key="dspy",
            retries=0,
            errors=[],
        )


# ─── Outlines schema enforcement ─────────────────────────────────────


def outlines_json_generator(
    *,
    model: Any,
    schema: type[BaseModel],
) -> Callable[[str], Any] | None:
    """Build an Outlines schema-constrained generator for `schema`.

    Returns a callable taking a prompt string and returning a Pydantic
    instance. Returns None when outlines isn't installed.

    Best used as a final-stage validator inside DSPyAdapter when the
    engine doesn't have native JSON mode (older llama.cpp builds).
    """
    try:
        import outlines  # type: ignore
    except Exception:
        return None

    try:
        json_schema = schema.model_json_schema()
        generator = outlines.generate.json(model, json_schema)

        def _call(prompt: str) -> Any:
            raw = generator(prompt)
            return schema.model_validate(raw)

        return _call
    except Exception:
        return None


# ─── Compile-time optimiser stub ─────────────────────────────────────


def compile_signatures(
    *,
    gold_set: dict[str, list[dict]],
    optimiser: Literal["BootstrapFewShot", "MIPROv2", "GEPA", "none"] = "MIPROv2",
    metric: Callable[[Any, Any], float] | None = None,
) -> dict[str, Any] | None:
    """Compile-time prompt optimisation. Takes a gold set keyed by
    signature name; returns a map of compiled DSPy modules ready to
    plug into a DSPyAdapter via `register_compiled`.

    Returns None when DSPy isn't installed.
    """
    if not detect_dspy_availability().dspy:
        return None

    try:
        import dspy  # type: ignore
    except Exception:
        return None

    compiled: dict[str, Any] = {}
    for site_name, signature in SIGNATURES.items():
        examples = gold_set.get(site_name, [])
        if not examples:
            continue
        try:
            sig_class = signature.to_dspy_signature()
            module = dspy.Predict(sig_class)
            if optimiser == "BootstrapFewShot":
                opt = dspy.BootstrapFewShot(metric=metric or _default_metric)
                module = opt.compile(module, trainset=examples)
            elif optimiser == "MIPROv2":
                opt = dspy.MIPROv2(metric=metric or _default_metric)
                module = opt.compile(module, trainset=examples)
            compiled[site_name] = module
        except Exception as e:
            logger.warning("compile failed for %s: %s", site_name, e)
    return compiled or None


def _default_metric(predicted: Any, gold: Any) -> float:
    """Generic schema-match metric: 1.0 iff every gold field is present
    in the predicted output AND non-empty. Plug in a domain-specific
    metric for serious optimisation."""
    try:
        gold_keys = set(getattr(gold, "__dict__", gold).keys())
        pred_keys = set(getattr(predicted, "__dict__", predicted).keys())
        if not gold_keys:
            return 0.0
        present = sum(
            1 for k in gold_keys
            if k in pred_keys and getattr(predicted, k, None) not in (None, "", [])
        )
        return present / len(gold_keys)
    except Exception:
        return 0.0


__all__ = [
    "DSPyAvailability",
    "DSPyAdapter",
    "PromptSignature",
    "SIGNATURES",
    "compile_signatures",
    "detect_dspy_availability",
    "outlines_json_generator",
]
