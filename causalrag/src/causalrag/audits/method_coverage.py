"""Method-coverage matrix audit (Sprint 9.5.2) — v1.0 ship gate.

Where :mod:`causalrag.audits.end_to_end_flow` asks *"does every node in the
pipeline graph have a producer and a consumer?"*, this audit asks the
complementary question: *"for every (estimand × realistic flag combo)
the pipeline could plausibly encounter, does the routing cascade hand us
at least one estimator, and which sensitivity panels are available to
back it up?"*.

The output is a sparse coverage matrix. Cells with zero reachable
estimators become v1.1 ticket suggestions — they are the gaps where an
analyst would land on ``LookupError`` (or the catch-all default) without
a literature-backed route.

Implementation
--------------

For each cell:

1. We build a :class:`SelectionContext` from the (estimand, flag combo)
   with neutral defaults for the non-flag sample-shape inputs (``n=500``,
   one modifier, no Bayesian preference). 500 is chosen because it lets
   the n≥500-gated rules (instrumental forest, multi-arm forest, larger
   CATE branch) fire — i.e., we ask "is there *any* sample shape under
   which the cascade would route here?".
2. We call :func:`_rule_cascade` to get the ordered list of estimator
   ids the cascade would try.
3. We intersect that ordered list with the catalog rows that (a) support
   the current ``estimand``, (b) have ``required_flags ⊆ combo``, and
   (c) have ``excluded_flags`` disjoint from ``combo``. Anything left is
   "reachable for this cell" — the cascade names it and the catalog
   admits it.
4. We resolve the applicable sensitivity panels from the flag combo via
   :data:`_PANEL_TRIGGERS`. Panels with no trigger are treated as
   universally available (E-value, sensemakr, anomaly_audit, …).

The matrix is *static*: no estimators are run, no data is loaded. Like
its sibling audit, it is safe to run in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

from causalrag.core.flags import DataFlag
from causalrag.estimators.catalog import CATALOG, MethodSpec
from causalrag.estimators.python.select import SelectionContext, _rule_cascade


# ─────────────────────────────────────────────────────────────────────────
# Flag universe + panel triggers
# ─────────────────────────────────────────────────────────────────────────


# The full cartesian product over every DataFlag explodes (2**N for N≈35
# flags). We restrict to the subset the cascade actually branches on —
# anything else would only inflate cell count without changing coverage
# verdicts. If the cascade learns new branches, extend this tuple.
_RELEVANT_FLAGS: tuple[DataFlag, ...] = (
    DataFlag.BINARY_TREATMENT,
    DataFlag.CATEGORICAL_TREATMENT,
    DataFlag.CONTINUOUS_TREATMENT,
    DataFlag.MIXTURE_EXPOSURE,
    DataFlag.BINARY_OUTCOME,
    DataFlag.CONTINUOUS_OUTCOME,
    DataFlag.RIGHT_CENSORED_OUTCOME,
    DataFlag.RARE_OUTCOME,
    DataFlag.BOUNDED_OUTCOME,
    DataFlag.ZERO_INFLATED_OUTCOME,
    DataFlag.IMBALANCED_TREATMENT,
    DataFlag.STAGGERED_ADOPTION,
    DataFlag.DIFF_IN_DIFF_CANDIDATE,
    DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT,
    DataFlag.MEDIATOR_PROPOSED,
    DataFlag.NEGATIVE_CONTROL_AVAILABLE,
    DataFlag.EFFECT_MODIFICATION_OF_INTEREST,
    DataFlag.SMALL_SAMPLE,
    DataFlag.HIGH_DIMENSIONAL,
    DataFlag.HEAVY_MISSINGNESS,
    DataFlag.HEAVY_CENSORING,
    DataFlag.POSITIVITY_VIOLATION,
)


# Mutually exclusive flag families: at most one treatment-type flag in a
# combo, at most one outcome-type flag. Reproduced here so the matrix
# generation does not depend on ``core.flags.validate_flag_set`` raising.
_TREATMENT_FLAGS: frozenset[DataFlag] = frozenset(
    {
        DataFlag.BINARY_TREATMENT,
        DataFlag.CATEGORICAL_TREATMENT,
        DataFlag.CONTINUOUS_TREATMENT,
        DataFlag.MIXTURE_EXPOSURE,
        DataFlag.TIME_VARYING_TREATMENT,
    }
)

_OUTCOME_FLAGS: frozenset[DataFlag] = frozenset(
    {
        DataFlag.BINARY_OUTCOME,
        DataFlag.CONTINUOUS_OUTCOME,
        DataFlag.COUNT_OUTCOME,
        DataFlag.RIGHT_CENSORED_OUTCOME,
        DataFlag.REPEATED_OUTCOME,
    }
)


# Panel → flag(s) that gate it. A panel with no trigger is universally
# available (E-value, sensemakr, refutation summary, anomaly audit run on
# every estimate). Panels with a trigger only surface when *any* of the
# trigger flags is present — i.e., the flag explains the threat model the
# panel addresses.
_PANEL_TRIGGERS: dict[str, frozenset[DataFlag]] = {
    "e_value": frozenset(),
    "sensemakr": frozenset(),
    "refutation_summary": frozenset(),
    "anomaly_audit": frozenset(),
    "ovb_chernozhukov": frozenset(),
    "tipping_point": frozenset({DataFlag.BINARY_TREATMENT, DataFlag.BINARY_OUTCOME}),
    "rosenbaum": frozenset({DataFlag.BINARY_TREATMENT, DataFlag.POSITIVITY_VIOLATION}),
    "manski": frozenset({DataFlag.HEAVY_MISSINGNESS, DataFlag.POSITIVITY_VIOLATION}),
    "negative_control": frozenset({DataFlag.NEGATIVE_CONTROL_AVAILABLE}),
}


# Neutral defaults used to build the SelectionContext for each cell. We
# pick n=500 + n_modifiers=1 so the n≥500-gated branches (instrumental
# forest, multi-arm forest, the CATE branch when EFFECT_MODIFICATION is
# also set) can fire. The question we are answering is "could the cascade
# *ever* surface an estimator for this cell?" — picking a permissive
# sample shape is the right semantic.
_CONTEXT_DEFAULT_N: int = 500
_CONTEXT_DEFAULT_N_MODIFIERS: int = 1


# ─────────────────────────────────────────────────────────────────────────
# Public dataclasses
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MethodCoverageCell:
    """One row of the coverage matrix: one estimand × one flag combo."""

    estimand: str
    flag_combo: frozenset[str]
    estimators_reachable: tuple[str, ...]
    sensitivity_panels_available: tuple[str, ...]
    is_covered: bool = field(default=False)


@dataclass
class MethodCoverageMatrix:
    """Sparse coverage matrix produced by
    :func:`build_method_coverage_matrix`. ``empty_cells`` is the audit's
    actionable surface — every entry is a v1.1 candidate."""

    cells: tuple[MethodCoverageCell, ...]
    n_cells: int
    n_covered: int
    coverage_pct: float
    empty_cells: tuple[MethodCoverageCell, ...]
    suggested_tickets: tuple[str, ...]


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _is_consistent_combo(combo: tuple[DataFlag, ...]) -> bool:
    """Reject combos that violate the basic kind-exclusivity rules: at
    most one treatment-type flag and at most one outcome-type flag.
    Without this guard the matrix would enumerate physically impossible
    cells like (BINARY_TREATMENT, CONTINUOUS_TREATMENT) and report them
    all as uncovered — pure noise."""
    s = set(combo)
    if len(s & _TREATMENT_FLAGS) > 1:
        return False
    if len(s & _OUTCOME_FLAGS) > 1:
        return False
    return True


def _enumerate_flag_combos(
    flags: tuple[DataFlag, ...],
    max_size: int,
) -> Iterable[tuple[DataFlag, ...]]:
    """Yield every consistent combination of ``flags`` of size 0..max_size."""
    for k in range(max_size + 1):
        for combo in combinations(flags, k):
            if _is_consistent_combo(combo):
                yield combo


def _catalog_admits(
    spec: MethodSpec,
    estimand: str,
    combo: frozenset[DataFlag],
) -> bool:
    """Whether the catalog spec is admissible for (estimand, combo).

    Mirrors :meth:`Registry.candidates_for` — required_flags must be a
    subset, excluded_flags must be disjoint, and the spec must declare
    ``estimand`` as supported. Sample-size is *not* checked here because
    the matrix asks "could this ever route?", not "does it route at this
    specific n?".
    """
    if estimand not in spec.estimands:
        return False
    if not frozenset(spec.required_flags).issubset(combo):
        return False
    if frozenset(spec.excluded_flags) & combo:
        return False
    return True


def _reachable_estimators(
    estimand: str,
    combo: frozenset[DataFlag],
    catalog: tuple[MethodSpec, ...],
) -> tuple[str, ...]:
    """Estimator ids the cascade names AND the catalog admits.

    The cascade is the routing brain; the catalog is the contract. Both
    have to agree before we declare the cell "covered" — an id named by
    the cascade but rejected by the catalog (estimand mismatch, excluded
    flag) is not actually reachable in production.
    """
    ctx = SelectionContext(
        estimand=estimand,
        flags=combo,
        n=_CONTEXT_DEFAULT_N,
        n_modifiers=_CONTEXT_DEFAULT_N_MODIFIERS,
        treatment_prevalence=None,
        want_bayesian=False,
    )
    try:
        ordered_ids = _rule_cascade(ctx)
    except Exception:  # noqa: BLE001 — audit must never crash on cascade drift
        ordered_ids = []

    admitted_ids = {
        spec.estimator_id
        for spec in catalog
        if _catalog_admits(spec, estimand, combo)
    }
    return tuple(eid for eid in ordered_ids if eid in admitted_ids)


def _applicable_panels(combo: frozenset[DataFlag]) -> tuple[str, ...]:
    """Sensitivity panels available for this cell.

    Untriggered panels (E-value, sensemakr, …) are always available.
    Triggered panels surface only when at least one of their trigger
    flags is present.
    """
    out: list[str] = []
    for panel, triggers in _PANEL_TRIGGERS.items():
        if not triggers or (triggers & combo):
            out.append(panel)
    return tuple(out)


def _format_flag_combo(combo: frozenset[str]) -> str:
    """Stable, sorted, brace-wrapped rendering for ticket strings."""
    if not combo:
        return "{}"
    return "{" + ", ".join(sorted(combo)) + "}"


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


def build_method_coverage_matrix(
    *,
    estimands: tuple[str, ...] = (
        "ATE",
        "ATT",
        "ATC",
        "CATE",
        "LATE",
        "NDE",
        "NIE",
        "RMST_CONTRAST",
        "MODIFIED_TREATMENT_POLICY",
    ),
    flag_subsets_max_size: int = 2,
    relevant_flags: tuple[DataFlag, ...] = _RELEVANT_FLAGS,
    catalog: tuple[MethodSpec, ...] = CATALOG,
) -> MethodCoverageMatrix:
    """Build the (estimand × flag-combo) coverage matrix.

    Parameters
    ----------
    estimands:
        The estimand classes to enumerate. Defaults match the PDD §29
        catalog of effect targets the project commits to supporting.
    flag_subsets_max_size:
        Cap on flag-combo size. 2 keeps the matrix tractable (cartesian
        product of subsets explodes); raise locally if you want a deeper
        audit, but expect O(C(N, k)) growth.
    relevant_flags:
        The flag universe the cascade actually inspects. The default is
        curated to the routing brain's branching set; passing more flags
        only inflates the cell count without changing verdicts.
    catalog:
        The :data:`CATALOG` tuple to validate cascade picks against.
        Exposed for tests.
    """
    cells: list[MethodCoverageCell] = []

    for combo in _enumerate_flag_combos(relevant_flags, flag_subsets_max_size):
        combo_set: frozenset[DataFlag] = frozenset(combo)
        combo_names: frozenset[str] = frozenset(f.name for f in combo_set)
        panels = _applicable_panels(combo_set)
        for estimand in estimands:
            reachable = _reachable_estimators(estimand, combo_set, catalog)
            cells.append(
                MethodCoverageCell(
                    estimand=estimand,
                    flag_combo=combo_names,
                    estimators_reachable=reachable,
                    sensitivity_panels_available=panels,
                    is_covered=len(reachable) > 0,
                )
            )

    empty_cells = tuple(c for c in cells if not c.is_covered)
    n_cells = len(cells)
    n_covered = n_cells - len(empty_cells)
    coverage_pct = (100.0 * n_covered / n_cells) if n_cells else 0.0

    tickets: list[str] = []
    for cell in empty_cells:
        tickets.append(
            f"v1.1: build estimator for {cell.estimand} under "
            f"{_format_flag_combo(cell.flag_combo)}"
        )

    return MethodCoverageMatrix(
        cells=tuple(cells),
        n_cells=n_cells,
        n_covered=n_covered,
        coverage_pct=coverage_pct,
        empty_cells=empty_cells,
        suggested_tickets=tuple(tickets),
    )


# ─────────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────────


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_matrix_html(m: MethodCoverageMatrix) -> str:
    """Render the matrix as a self-contained HTML fragment.

    The output is a plain ``<table>`` with one row per cell, suitable for
    pasting into a CI artifact or a Quarto report. Covered cells get a
    green background; empty cells get a red background and surface in the
    ticket list below the table.
    """
    rows: list[str] = []
    rows.append(
        '<div style="font-family:sans-serif;">'
        '<h2 style="margin-bottom:4px;">Method coverage matrix</h2>'
        f'<p style="color:#555;">{m.n_covered}/{m.n_cells} cells covered '
        f'({m.coverage_pct:.1f}%) · {len(m.empty_cells)} empty cells '
        f"flagged for v1.1.</p>"
    )

    rows.append(
        '<table style="border-collapse:collapse;width:100%;font-size:0.9em;">'
        "<thead><tr>"
        '<th style="text-align:left;border:1px solid #ddd;padding:4px 8px;">Estimand</th>'
        '<th style="text-align:left;border:1px solid #ddd;padding:4px 8px;">Flag combo</th>'
        '<th style="text-align:left;border:1px solid #ddd;padding:4px 8px;">Estimators reachable</th>'
        '<th style="text-align:left;border:1px solid #ddd;padding:4px 8px;">Sensitivity panels</th>'
        '<th style="text-align:left;border:1px solid #ddd;padding:4px 8px;">Covered</th>'
        "</tr></thead><tbody>"
    )

    for cell in m.cells:
        bg = "#e8f5e9" if cell.is_covered else "#ffebee"
        combo_render = _esc(_format_flag_combo(cell.flag_combo))
        est_render = _esc(", ".join(cell.estimators_reachable)) or "<em>none</em>"
        panel_render = _esc(", ".join(cell.sensitivity_panels_available))
        flag = "yes" if cell.is_covered else "no"
        rows.append(
            f'<tr style="background:{bg};">'
            f'<td style="border:1px solid #ddd;padding:4px 8px;"><code>{_esc(cell.estimand)}</code></td>'
            f'<td style="border:1px solid #ddd;padding:4px 8px;"><code>{combo_render}</code></td>'
            f'<td style="border:1px solid #ddd;padding:4px 8px;">{est_render}</td>'
            f'<td style="border:1px solid #ddd;padding:4px 8px;">{panel_render}</td>'
            f'<td style="border:1px solid #ddd;padding:4px 8px;">{flag}</td>'
            "</tr>"
        )

    rows.append("</tbody></table>")

    if m.suggested_tickets:
        rows.append('<h3 style="margin-top:14px;">Suggested v1.1 tickets</h3><ol>')
        for t in m.suggested_tickets:
            rows.append(f"<li>{_esc(t)}</li>")
        rows.append("</ol>")

    rows.append("</div>")
    return "".join(rows)


__all__ = [
    "MethodCoverageCell",
    "MethodCoverageMatrix",
    "build_method_coverage_matrix",
    "render_matrix_html",
]
