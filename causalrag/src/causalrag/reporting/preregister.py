"""Preregistration exporters (Sprint 1.5).

Three artefacts can be exported from a :class:`StudyProtocol`:

1. **OSF preregistration JSON** — uploadable to https://osf.io/registrations
   to lock in a preregistration before estimation runs.
2. **AsPredicted.org 9-question Markdown** — the lightweight community
   alternative to OSF.
3. **Target-trial-emulation (TTE) protocol** — Markdown following the
   Hubbard et al., *NEJM* 2024 seven-element template (eligibility,
   treatment strategies, assignment, follow-up, outcome, contrast,
   analysis plan).

The current :class:`StudyProtocol` does not carry every TTE field
directly. Where a field has to be inferred from ``protocol.discovery``
or filled with a sensible default, we surface this with an
``[INFERRED]`` marker so the analyst can complete the document by hand
before depositing it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import Hypothesis, StudyProtocol
from causalrag.core.roles import VariableRole, VariableSpec


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _columns(protocol: StudyProtocol) -> tuple[VariableSpec, ...]:
    """Best-effort: discovery.columns first, then dataset.columns."""
    if protocol.discovery is not None and protocol.discovery.columns:
        return protocol.discovery.columns
    if protocol.dataset is not None and protocol.dataset.columns:
        return protocol.dataset.columns
    return ()


def _columns_with_role(
    protocol: StudyProtocol, role: VariableRole
) -> list[str]:
    return [c.name for c in _columns(protocol) if c.role == role]


def _hypothesis_dict(h: Hypothesis) -> dict[str, Any]:
    estimand_class = None
    if h.estimand is not None:
        estimand_class = h.estimand.klass.value
    return {
        "id": h.id,
        "treatment": h.treatment,
        "outcome": h.outcome,
        "modifiers": list(h.modifiers),
        "counterfactual": h.counterfactual,
        "rationale": h.rationale,
        "estimand_class": estimand_class,
    }


def _unique(seq: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in seq:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _treatments(protocol: StudyProtocol) -> list[str]:
    role_cols = _columns_with_role(protocol, VariableRole.TREATMENT)
    queue_cols = [h.treatment for h in protocol.hypothesis_queue]
    return _unique([*role_cols, *queue_cols])


def _outcomes(protocol: StudyProtocol) -> list[str]:
    role_cols = _columns_with_role(protocol, VariableRole.OUTCOME)
    queue_cols = [h.outcome for h in protocol.hypothesis_queue]
    return _unique([*role_cols, *queue_cols])


def _confounders(protocol: StudyProtocol) -> list[str]:
    return _columns_with_role(protocol, VariableRole.CONFOUNDER)


def _estimator_ids(protocol: StudyProtocol) -> list[str]:
    ids: list[str] = []
    for h in protocol.hypothesis_queue:
        if h.estimand is not None:
            ids.append(h.estimand.klass.value)
    return _unique(ids)


def _missing_data_note(protocol: StudyProtocol) -> str:
    all_flags = set(protocol.flags)
    if protocol.discovery is not None:
        all_flags |= set(protocol.discovery.flags)
    if DataFlag.HEAVY_MISSINGNESS in all_flags:
        return (
            "HEAVY_MISSINGNESS flag set: multiple imputation or "
            "missingness-aware estimator will be used in place of "
            "complete-case analysis."
        )
    return "Complete-case unless flagged HEAVY_MISSINGNESS"


def _study_design(protocol: StudyProtocol) -> str:
    pieces: list[str] = []
    all_flags = set(protocol.flags)
    if protocol.discovery is not None:
        all_flags |= set(protocol.discovery.flags)
    if DataFlag.LONGITUDINAL in all_flags or DataFlag.PANEL_STRUCTURE in all_flags:
        pieces.append("longitudinal / panel")
    elif DataFlag.CROSS_SECTIONAL_SLICE in all_flags:
        pieces.append("cross-sectional")
    if DataFlag.RIGHT_CENSORED_OUTCOME in all_flags:
        pieces.append("time-to-event with right censoring")
    if DataFlag.DIFF_IN_DIFF_CANDIDATE in all_flags:
        pieces.append("difference-in-differences candidate")
    if DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT in all_flags:
        pieces.append("instrumental-variable candidate")
    if not pieces:
        pieces.append("observational, single-cohort")
    return "Observational study; " + ", ".join(pieces)


def _exploratory_hypotheses(protocol: StudyProtocol) -> list[str]:
    """Hypotheses that did not pre-specify an estimand are treated as exploratory."""
    return [h.id for h in protocol.hypothesis_queue if h.estimand is None]


# ---------------------------------------------------------------------------
# OSF preregistration (openEnded-2.0)
# ---------------------------------------------------------------------------


def export_osf_preregistration(
    protocol: StudyProtocol, output_path: Path
) -> Path:
    """Write an OSF-preregistration-compatible JSON file.

    Schema follows OSF's openEnded-2.0 registration schema; the user
    uploads the file at https://osf.io/registrations to obtain a
    timestamped, version-of-record preregistration.
    """
    output_path = Path(output_path)
    _ensure_parent(output_path)

    hypotheses_payload = [_hypothesis_dict(h) for h in protocol.hypothesis_queue]

    treatments = _treatments(protocol)
    outcomes = _outcomes(protocol)
    confounders = _confounders(protocol)
    estimators = _estimator_ids(protocol)
    exploratory = _exploratory_hypotheses(protocol)

    dataset_source = (
        protocol.dataset.source if protocol.dataset is not None else "Not specified"
    )
    sample_size: str | int
    if protocol.dataset is not None and protocol.dataset.n_rows is not None:
        sample_size = protocol.dataset.n_rows
    else:
        sample_size = "Not specified"

    payload: dict[str, Any] = {
        "registration_schema": "openEnded-2.0",
        "title": protocol.name,
        "study_information": {
            "hypotheses": hypotheses_payload,
        },
        "design_plan": {
            "study_type": "Observational - Causal Inference",
            "blinding": "No blinding",
            "study_design": _study_design(protocol),
            "randomization": "Not randomized",
        },
        "sampling_plan": {
            "existing_data": "Registration prior to analysis of the data",
            "data_collection_procedures": dataset_source,
            "sample_size": sample_size,
            "stopping_rule": "All available rows",
        },
        "variables": {
            "manipulated_variables": treatments,
            "measured_variables": outcomes,
            "indices": (
                confounders
                if confounders
                else "Not applicable — no derived indices used."
            ),
        },
        "analysis_plan": {
            "statistical_models": (
                estimators
                if estimators
                else "Estimators to be assigned by the routing layer (PDD §15)."
            ),
            "transformations": "Not applicable",
            "inference_criteria": (
                f"alpha=0.05; multiple testing: {protocol.multiple_testing}"
            ),
            "data_exclusion": _missing_data_note(protocol),
            "missing_data": "Complete-case unless flagged HEAVY_MISSINGNESS",
            "exploratory_analysis": (
                exploratory
                if exploratory
                else "All pre-specified hypotheses are confirmatory."
            ),
        },
    }

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=str),
        encoding="utf-8",
    )
    return output_path


# ---------------------------------------------------------------------------
# AsPredicted.org 9-question markdown
# ---------------------------------------------------------------------------


_ASPREDICTED_QUESTIONS = [
    "Have any data been collected for this study already?",
    "What's the main question being asked or hypothesis being tested in this study?",
    "Describe the key dependent variable(s) specifying how they will be measured.",
    "How many and which conditions will participants be assigned to?",
    "Specify exactly which analyses you will conduct to examine the main question/hypothesis.",
    "Describe exactly how outliers will be defined and handled, and your precise rule(s) "
    "for excluding observations.",
    "How many observations will be collected or what will determine sample size? "
    "No need to justify decision, but be precise about exactly how the number will be determined.",
    "Anything else you would like to pre-register? "
    "(e.g., secondary analyses, variables collected for exploratory purposes, "
    "unusual analyses planned?)",
    "Name and affiliation of all authors / co-authors / sponsors.",
]


def export_aspredicted_markdown(
    protocol: StudyProtocol, output_path: Path
) -> Path:
    """Write the AsPredicted.org 9-question form as Markdown."""
    output_path = Path(output_path)
    _ensure_parent(output_path)

    treatments = _treatments(protocol)
    outcomes = _outcomes(protocol)
    estimators = _estimator_ids(protocol)
    exploratory = _exploratory_hypotheses(protocol)

    if protocol.hypothesis_queue:
        hypotheses_md = "\n".join(
            f"- **{h.id}**: effect of `{h.treatment}` on `{h.outcome}`"
            + (f" — {h.rationale}" if h.rationale else "")
            for h in protocol.hypothesis_queue
        )
    else:
        hypotheses_md = "_No hypotheses pre-specified yet._"

    sample_size_str = (
        str(protocol.dataset.n_rows)
        if protocol.dataset is not None and protocol.dataset.n_rows is not None
        else "All available rows in the source dataset"
    )

    answers = [
        "Yes — secondary analysis of an existing dataset. Registration occurs "
        "prior to running any estimation under the current protocol version.",
        hypotheses_md,
        (
            "Outcome variable(s): "
            + (", ".join(f"`{o}`" for o in outcomes) if outcomes else "_to be specified_")
            + "."
        ),
        (
            "Treatment / exposure variable(s): "
            + (", ".join(f"`{t}`" for t in treatments) if treatments else "_to be specified_")
            + ". This is an observational study; assignment is not randomized."
        ),
        (
            "Estimators: "
            + (", ".join(f"`{e}`" for e in estimators) if estimators else "to be selected by the routing layer (PDD §15)")
            + f". Multiple-testing correction: `{protocol.multiple_testing}`; alpha = 0.05."
        ),
        _missing_data_note(protocol),
        f"Sample size: {sample_size_str}.",
        (
            "Exploratory analyses: "
            + (", ".join(f"`{eid}`" for eid in exploratory) if exploratory else "none — all hypotheses are confirmatory.")
        ),
        "_To be completed by the analyst before submission._",
    ]

    lines = [f"# AsPredicted Preregistration — {protocol.name}", ""]
    for i, (q, a) in enumerate(zip(_ASPREDICTED_QUESTIONS, answers), start=1):
        lines.append(f"## {i}. {q}")
        lines.append("")
        lines.append(a)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Target-trial emulation protocol (Hubbard et al., NEJM 2024)
# ---------------------------------------------------------------------------


TTE_ELEMENT_HEADINGS = (
    "Eligibility criteria",
    "Treatment strategies",
    "Assignment procedures",
    "Follow-up period",
    "Outcome",
    "Causal contrast",
    "Analysis plan",
)


def _inferred(text: str) -> str:
    return f"[INFERRED] {text}"


def _specified(text: str) -> str:
    return f"[SPECIFIED] {text}"


def export_target_trial_protocol(
    protocol: StudyProtocol, output_path: Path
) -> Path:
    """Write a target-trial-emulation Markdown protocol (NEJM 2024)."""
    output_path = Path(output_path)
    _ensure_parent(output_path)

    treatments = _treatments(protocol)
    outcomes = _outcomes(protocol)
    confounders = _confounders(protocol)
    estimators = _estimator_ids(protocol)

    all_flags = set(protocol.flags)
    if protocol.discovery is not None:
        all_flags |= set(protocol.discovery.flags)

    dataset_source = (
        protocol.dataset.source if protocol.dataset is not None else "unspecified source"
    )
    sample_size_str = (
        str(protocol.dataset.n_rows)
        if protocol.dataset is not None and protocol.dataset.n_rows is not None
        else "unknown"
    )

    # 1. Eligibility — always inferred from the dataset alone.
    eligibility = _inferred(
        f"All rows present in `{dataset_source}` (n={sample_size_str}). "
        "No further eligibility filter has been specified in the protocol."
    )

    # 2. Treatment strategies.
    if treatments:
        treat_md = ", ".join(f"`{t}`" for t in treatments)
        treatment_strategies = _specified(
            f"Treatment / exposure variable(s): {treat_md}. "
            "Strategies contrast each variable's natural levels."
        )
    else:
        treatment_strategies = _inferred(
            "No treatment column identified yet; strategies will be set "
            "when the hypothesis queue is populated."
        )

    # 3. Assignment procedures.
    assignment = _inferred(
        "Observational: treatment assignment is not under analyst control. "
        "Conditional exchangeability is assumed given the adjustment set; "
        + (
            f"candidate confounders: {', '.join(f'`{c}`' for c in confounders)}."
            if confounders
            else "no confounders are yet labelled."
        )
    )

    # 4. Follow-up period.
    if DataFlag.RIGHT_CENSORED_OUTCOME in all_flags:
        follow_up = _specified(
            "Time-to-event outcome present (RIGHT_CENSORED_OUTCOME flag). "
            "Follow-up extends to event or administrative censoring."
        )
    elif DataFlag.LONGITUDINAL in all_flags or DataFlag.PANEL_STRUCTURE in all_flags:
        follow_up = _specified(
            "Longitudinal / panel structure flagged. "
            "Follow-up covers all observed time points per unit."
        )
    else:
        follow_up = _inferred(
            "Cross-sectional snapshot; no explicit follow-up window defined."
        )

    # 5. Outcome.
    if outcomes:
        outcome_md = ", ".join(f"`{o}`" for o in outcomes)
        outcome_section = _specified(f"Outcome variable(s): {outcome_md}.")
    else:
        outcome_section = _inferred(
            "Outcome column(s) not yet identified in the discovery report."
        )

    # 6. Causal contrast.
    if estimators:
        contrast = _specified(
            "Estimand class(es): " + ", ".join(f"`{e}`" for e in estimators) + "."
        )
    else:
        contrast = _inferred(
            "Estimand class not yet assigned; routing layer will choose "
            "(ATE/ATT/CATE/...) based on flags."
        )

    # 7. Analysis plan.
    analysis_plan_lines = [
        _specified(
            f"Multiple testing: `{protocol.multiple_testing}`; alpha = 0.05."
        ),
        _specified(_missing_data_note(protocol)),
    ]
    if estimators:
        analysis_plan_lines.append(
            _specified(
                "Pre-registered estimators: "
                + ", ".join(f"`{e}`" for e in estimators)
                + "."
            )
        )
    else:
        analysis_plan_lines.append(
            _inferred(
                "Estimator selection will be performed by the routing layer "
                "from the dispatched DataFlag set."
            )
        )
    analysis_plan = "\n\n".join(analysis_plan_lines)

    sections = [
        eligibility,
        treatment_strategies,
        assignment,
        follow_up,
        outcome_section,
        contrast,
        analysis_plan,
    ]

    lines: list[str] = [
        f"# Target Trial Emulation Protocol — {protocol.name}",
        "",
        "_Following the seven-element template of Hubbard et al., "
        "*NEJM* 2024._",
        "",
        "Each section is tagged `[SPECIFIED]` (filled from the StudyProtocol) "
        "or `[INFERRED]` (filled from a default — the analyst should review "
        "before deposit).",
        "",
    ]
    for heading, body in zip(TTE_ELEMENT_HEADINGS, sections):
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


__all__ = [
    "TTE_ELEMENT_HEADINGS",
    "export_aspredicted_markdown",
    "export_osf_preregistration",
    "export_target_trial_protocol",
]
