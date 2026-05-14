"""Auto-selection across the Python estimator catalog.

Picks one estimator id for a given (estimand, flags, n, user-preference)
tuple. The selection rules below encode the literature consensus where one
exists and the project's defensible-by-default policy where it doesn't.

Auto rules (highest priority first):

1. ``HIGH_DIMENSIONAL`` flag → ``python.dml.sparse_linear`` (Lasso final stage).
2. Non-linear heterogeneity expected (``≥3`` modifiers, ``n ≥ 500``)
   → ``python.dml.causal_forest`` for richer CATE.
3. ``SMALL_SAMPLE`` flag (n < 200) → ``python.dml.linear`` — meta-learners
   and forests overfit; linear DML stays interpretable.
4. ``BINARY_TREATMENT`` + ``BINARY_OUTCOME`` and Bayesian intervals requested
   → ``python.bart.dml`` if available, else ``python.dr.dr_learner``.
5. ``BINARY_TREATMENT`` (rare, prevalence < 15%) → ``python.meta.x_learner``
   (Künzel et al. show X-learner wins under treatment imbalance).
6. Default → ``python.dml.linear`` (PDD §29.1 v0.1 default).

Users can override via ``prefer=<estimator_id>`` (returned verbatim if the
estimator supports the requested estimand and is not excluded by flags) or
``prefer=<family>`` where family ∈ ``{dml, forest, sparse, meta, bart}``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from causalrag.core.flags import DataFlag
from causalrag.core.registry import EstimatorEntry, get_registry


@dataclass(frozen=True)
class SelectionContext:
    estimand: str  # "ATE", "CATE", "RMST_CONTRAST", ...
    flags: frozenset[DataFlag]
    n: int | None = None
    n_modifiers: int = 0
    treatment_prevalence: float | None = None
    want_bayesian: bool = False


_FAMILY_MAP: dict[str, tuple[str, ...]] = {
    "dml": ("python.dml.linear",),
    "sparse": ("python.dml.sparse_linear",),
    "forest": ("python.dml.causal_forest",),
    "bart": ("python.bart.dml",),
    "meta": (
        "python.dr.dr_learner",
        "python.meta.x_learner",
        "python.meta.t_learner",
        "python.meta.s_learner",
    ),
}


def select_estimator(
    *,
    estimand: str,
    flags: Iterable[DataFlag] = (),
    n: int | None = None,
    n_modifiers: int = 0,
    treatment_prevalence: float | None = None,
    want_bayesian: bool = False,
    prefer: str | None = None,
) -> EstimatorEntry:
    """Pick the best registered estimator for the given situation.

    Raises ``LookupError`` if no estimator can satisfy the constraints — the
    caller is expected to surface that to the analyst rather than silently
    pick a worse one.
    """
    ctx = SelectionContext(
        estimand=estimand,
        flags=frozenset(flags),
        n=n,
        n_modifiers=n_modifiers,
        treatment_prevalence=treatment_prevalence,
        want_bayesian=want_bayesian,
    )
    reg = get_registry()
    candidates = reg.candidates_for(estimand=estimand, required=ctx.flags, n=n)
    if not candidates:
        raise LookupError(
            f"No registered estimator supports estimand={estimand!r} with flags={sorted(f.value for f in ctx.flags)}"
        )
    by_id = {c.id: c for c in candidates}

    # User-pinned preference: id first, then family.
    if prefer:
        if prefer in by_id:
            return by_id[prefer]
        for candidate_id in _FAMILY_MAP.get(prefer, ()):
            if candidate_id in by_id:
                return by_id[candidate_id]

    # Rule cascade — return the first registered match.
    for rule_id in _rule_cascade(ctx):
        if rule_id in by_id:
            return by_id[rule_id]

    return candidates[0]


def _rule_cascade(ctx: SelectionContext) -> list[str]:
    cascade: list[str] = []

    # ─── 1. Specialized-outcome routes (R bridge wins) ────────────────────
    # Right-censored survival → CSF for CATE, survRM2 for ATE/RMST contrast.
    if DataFlag.RIGHT_CENSORED_OUTCOME in ctx.flags:
        if ctx.n_modifiers >= 1 and (ctx.n or 0) >= 200:
            cascade.append("rbridge.grf.causal_survival_forest")
        cascade.append("rbridge.survrm2")

    # ─── 2. Specialized-treatment routes ─────────────────────────────────
    # Mixture exposures → multi-treatment lmtp.
    if DataFlag.MIXTURE_EXPOSURE in ctx.flags:
        cascade.append("rbridge.lmtp.mixture")

    # Categorical / multi-arm treatment.
    if DataFlag.CATEGORICAL_TREATMENT in ctx.flags and (ctx.n or 0) >= 500:
        cascade.append("rbridge.grf.multi_arm_causal_forest")

    # Continuous treatment with dose-response question → lmtp.shift /
    # marginaleffects slopes. SDR preferred at smaller n; TMLE at larger n.
    if DataFlag.CONTINUOUS_TREATMENT in ctx.flags:
        if (ctx.n or 0) < 500:
            cascade.append("rbridge.lmtp.sdr")
        else:
            cascade.append("rbridge.lmtp.shift")
        cascade.append("rbridge.marginaleffects.slopes")

    # IV pathway.
    if DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT in ctx.flags and (ctx.n or 0) >= 500:
        cascade.append("rbridge.grf.instrumental_forest")

    # Mediation: NDE/NIE-class estimand requires the mediation package.
    if (
        DataFlag.MEDIATOR_PROPOSED in ctx.flags
        and ctx.estimand in ("NDE", "NIE")
    ):
        cascade.append("rbridge.mediation")

    # Positivity violation under binary treatment → matching trims; weighting
    # is the second-best fallback before raw DML which loses identification.
    if (
        DataFlag.POSITIVITY_VIOLATION in ctx.flags
        and DataFlag.BINARY_TREATMENT in ctx.flags
    ):
        cascade.append("rbridge.matchit")
        cascade.append("rbridge.weightit")

    # Bayesian-credible-interval request → bartCause (R) is the canonical
    # implementation; falls back to Python BART when R isn't available.
    if ctx.want_bayesian:
        cascade.extend(["rbridge.bartcause", "python.bart.dml", "python.dr.dr_learner"])

    # DiD / staggered adoption: TODO future rbridge.did route; for now fall
    # through to the default ladder. Surface a comment for downstream readers.
    if (
        DataFlag.DIFF_IN_DIFF_CANDIDATE in ctx.flags
        or DataFlag.STAGGERED_ADOPTION in ctx.flags
    ):
        # TODO(estimators): wire ``rbridge.did`` (Callaway-Sant'Anna staggered
        # DiD) once the R bridge ships it. Until then the default DML ladder
        # below is the best available estimator.
        pass

    # Zero-inflated count outcome: no dedicated estimator yet; DML picks up
    # the slack at the bottom of the cascade.
    if DataFlag.ZERO_INFLATED_OUTCOME in ctx.flags:
        # TODO(estimators): wire a hurdle/ZIP-aware estimator
        # (``rbridge.lmtp.zip`` or ``python.glm.zip``). For now DML with the
        # default flexible nuisance is the least-bad fallback.
        pass

    # ─── 3. Tiny samples: OLS is the honest answer ────────────────────────
    # The DML / forest / metalearner family overfits at n < 100; OLS with
    # HC3 robust SE is the small-sample textbook default.
    if (ctx.n or 0) < 100 or DataFlag.SMALL_SAMPLE in ctx.flags:
        cascade.append("python.linear.ols")

    # ─── 4. High-dim adjustment ──────────────────────────────────────────
    if DataFlag.HIGH_DIMENSIONAL in ctx.flags:
        cascade.append("python.dml.sparse_linear")

    # ─── 5. CATE-richness path ───────────────────────────────────────────
    # Prefer grf::causal_forest at n ≥ 500 (reference impl); fall back to
    # EconML's CausalForestDML otherwise. When the analyst has explicitly
    # asked for effect modification, lower the modifier-count threshold to 1
    # — they want CATE even with a single moderator.
    cate_threshold = 1 if DataFlag.EFFECT_MODIFICATION_OF_INTEREST in ctx.flags else 3
    if ctx.n_modifiers >= cate_threshold and (ctx.n or 0) >= 500:
        cascade.append("rbridge.grf.causal_forest")
        cascade.append("python.dml.causal_forest")

    if DataFlag.SMALL_SAMPLE in ctx.flags:
        cascade.append("python.dml.linear")

    if ctx.want_bayesian:
        cascade.extend(["python.bart.dml", "python.dr.dr_learner"])

    # Rare outcome (binary T): stabilized-weight DR-learner is the canonical
    # choice. Guard against OLS / raw DML by promoting DR ahead of them.
    if DataFlag.RARE_OUTCOME in ctx.flags and DataFlag.BINARY_TREATMENT in ctx.flags:
        cascade.insert(0, "python.dr.dr_learner")

    # Imbalanced treatment flag → X-learner (Künzel et al.). Keep the legacy
    # prevalence-based check as a fallback for callers that haven't migrated.
    if DataFlag.IMBALANCED_TREATMENT in ctx.flags:
        cascade.append("python.meta.x_learner")
    if (
        DataFlag.BINARY_TREATMENT in ctx.flags
        and ctx.treatment_prevalence is not None
        and (ctx.treatment_prevalence < 0.15 or ctx.treatment_prevalence > 0.85)
    ):
        cascade.append("python.meta.x_learner")

    # Bounded outcome (proportion / rate): exclude raw OLS — silent misuse on
    # [0, 1] data — and prefer DML, which can be paired with a logit-link
    # final stage. We do not remove OLS from the registry candidate set here
    # (the registry filter would still keep it); instead we strip it from the
    # cascade so it cannot win on rank.
    if DataFlag.BOUNDED_OUTCOME in ctx.flags:
        cascade = [c for c in cascade if c != "python.linear.ols"]
        cascade.append("python.dml.linear")

    # Continuous outcome → DML linear is the right default. We surface this
    # branch explicitly so the flow-audit sees CONTINUOUS_OUTCOME consumed.
    # (Functionally redundant with the default ladder; semantically required.)
    if DataFlag.CONTINUOUS_OUTCOME in ctx.flags:
        cascade.append("python.dml.linear")

    # Heavy censoring under right-censored outcome → survRM2 / CSF lead but
    # add WeightIt as an IPCW-style fallback when sensitivity to the
    # censoring model matters. When HEAVY_CENSORING is set in isolation
    # (without the right-censoring flag), it's a discovery hint that the
    # analyst should consider IPCW. We route to WeightIt as a defensible
    # weighting fallback.
    if DataFlag.HEAVY_CENSORING in ctx.flags:
        if DataFlag.RIGHT_CENSORED_OUTCOME not in ctx.flags:
            cascade.append("rbridge.weightit")
        # Otherwise the RIGHT_CENSORED_OUTCOME branch above already
        # surfaced CSF + survRM2.

    # Heavy missingness → push complete-case OLS out of the cascade and
    # prefer estimators that natively handle missing-at-random via the
    # cross-fitted nuisance (DML, doubly-robust learners). The MICE / IPCW
    # routes are recommended downstream by data/missingness.py but those
    # aren't catalog estimators — they're preprocessing recommendations.
    if DataFlag.HEAVY_MISSINGNESS in ctx.flags:
        cascade = [c for c in cascade if c != "python.linear.ols"]
        cascade.extend(["python.dr.dr_learner", "python.dml.linear"])

    # Negative-control availability → unlock the proximal-CI estimator
    # which requires (NCE, NCO) pairs. Otherwise NCO falsification only.
    if DataFlag.NEGATIVE_CONTROL_AVAILABLE in ctx.flags:
        cascade.append("python.proximal.regression")

    # Default ladder
    cascade.extend(
        [
            "python.dml.linear",
            "python.dml.causal_forest",
            "python.dr.dr_learner",
            "python.meta.x_learner",
        ]
    )

    # Final filter: BOUNDED_OUTCOME must never pick OLS, even if the default
    # ladder reintroduced it.
    if DataFlag.BOUNDED_OUTCOME in ctx.flags:
        cascade = [c for c in cascade if c != "python.linear.ols"]

    # Deduplicate preserving order
    seen: set[str] = set()
    return [c for c in cascade if not (c in seen or seen.add(c))]


__all__ = ["select_estimator", "SelectionContext"]
