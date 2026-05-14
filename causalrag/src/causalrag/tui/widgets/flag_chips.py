"""FlagChipBar — horizontal row of flag chips with hover tooltips.

Sprint 4.6 hover-help widget. Each chip is a small token bearing a
:class:`~causalrag.core.flags.DataFlag` value; hovering reveals a tooltip
combining the canonical summary, analysis implication, and estimator
routing pulled from :mod:`causalrag.core.flag_descriptions`.

Group → colour mapping (per PDD §4.6):

* treatment-shape  → blue
* outcome-shape    → green
* structural       → orange
* design hint      → purple

Chips for unknown / ungrouped flags fall back to a neutral grey.

The widget is intentionally lightweight: it renders to a single styled
string with embedded ``title=`` HTML attributes, so the same render path
works in Textual's console output and in any future HTML export.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

from causalrag.core.flag_descriptions import describe_safe
from causalrag.core.flags import DataFlag


# ───────────────────── group / colour map ─────────────────────

_TREATMENT_SHAPE: frozenset[DataFlag] = frozenset(
    {
        DataFlag.BINARY_TREATMENT,
        DataFlag.CATEGORICAL_TREATMENT,
        DataFlag.CONTINUOUS_TREATMENT,
        DataFlag.MIXTURE_EXPOSURE,
        DataFlag.TIME_VARYING_TREATMENT,
        DataFlag.IMBALANCED_TREATMENT,
    }
)

_OUTCOME_SHAPE: frozenset[DataFlag] = frozenset(
    {
        DataFlag.BINARY_OUTCOME,
        DataFlag.CONTINUOUS_OUTCOME,
        DataFlag.COUNT_OUTCOME,
        DataFlag.RIGHT_CENSORED_OUTCOME,
        DataFlag.RARE_OUTCOME,
        DataFlag.BOUNDED_OUTCOME,
        DataFlag.ZERO_INFLATED_OUTCOME,
        DataFlag.COMPETING_RISKS,
        DataFlag.REPEATED_OUTCOME,
    }
)

_STRUCTURAL: frozenset[DataFlag] = frozenset(
    {
        DataFlag.SMALL_SAMPLE,
        DataFlag.HIGH_DIMENSIONAL,
        DataFlag.POSITIVITY_VIOLATION,
        DataFlag.HEAVY_MISSINGNESS,
        DataFlag.HEAVY_CENSORING,
        DataFlag.SUSPECTED_INFORMATIVE_CENSORING,
        DataFlag.PANEL_STRUCTURE,
        DataFlag.LONGITUDINAL,
        DataFlag.CLUSTERED,
        DataFlag.NETWORK_INTERFERENCE,
        DataFlag.SINGLE_TREATED_UNIT,
        DataFlag.CROSS_SECTIONAL_SLICE,
    }
)

_DESIGN: frozenset[DataFlag] = frozenset(
    {
        DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT,
        DataFlag.MEDIATOR_PROPOSED,
        DataFlag.EFFECT_MODIFICATION_OF_INTEREST,
        DataFlag.NEGATIVE_CONTROL_AVAILABLE,
        DataFlag.DIFF_IN_DIFF_CANDIDATE,
        DataFlag.STAGGERED_ADOPTION,
        DataFlag.IDENTIFICATION_FAILED,
    }
)


# Background colours: pleasant in a dark TUI, also valid CSS for HTML export.
_GROUP_BG: dict[str, str] = {
    "treatment": "#3a6ea5",   # blue
    "outcome":   "#3a8a55",   # green
    "structural": "#c2772a",  # orange
    "design":    "#7a4ea8",   # purple
    "unknown":   "#4d5773",   # neutral grey
}

_FG = "#f2f4f8"


def _group_of(flag: DataFlag) -> str:
    if flag in _TREATMENT_SHAPE:
        return "treatment"
    if flag in _OUTCOME_SHAPE:
        return "outcome"
    if flag in _STRUCTURAL:
        return "structural"
    if flag in _DESIGN:
        return "design"
    return "unknown"


def _coerce(flag: object) -> DataFlag | None:
    """Best-effort coercion of incoming members to ``DataFlag``."""
    if isinstance(flag, DataFlag):
        return flag
    if isinstance(flag, str):
        try:
            return DataFlag(flag)
        except ValueError:
            return None
    return None


def _tooltip_text(flag: DataFlag) -> str:
    """Build the multi-line tooltip body for one flag chip."""
    d = describe_safe(flag)
    routes = ", ".join(d.routes_to) if d.routes_to else "—"
    # Newlines render as line breaks in Textual's tooltip; HTML title=
    # attributes flatten them to spaces, which is still legible.
    return (
        f"{flag.value}\n"
        f"{d.summary}\n"
        f"→ {d.implication}\n"
        f"routes_to: {routes}"
    )


def _render_chip(flag: DataFlag) -> str:
    """Render a single chip as a Rich-markup span with embedded tooltip."""
    bg = _GROUP_BG[_group_of(flag)]
    tip = _tooltip_text(flag).replace('"', "'")
    # `title=` is the universal hover-help attribute; rich's markup parser
    # will preserve it verbatim inside the span when markup=False, and
    # any HTML exporter (rich.console.export_html) will surface it as a
    # real <abbr>-style tooltip on the rendered chip.
    return (
        f'<chip flag="{flag.value}" title="{tip}" '
        f'style="background:{bg};color:{_FG}">'
        f' {flag.value} '
        f'</chip>'
    )


# ───────────────────── widget ─────────────────────


class FlagChipBar(Static):
    """Horizontal bar of flag chips with hover descriptions.

    Update via :meth:`set_flags`. The rendered string contains one
    ``<chip …>`` token per active flag with a ``title=`` tooltip showing
    the flag's summary, implication, and ``routes_to`` list from
    :mod:`causalrag.core.flag_descriptions`. Tooltips Just Work in both
    Textual's console hover and in any HTML export of the same render.
    """

    DEFAULT_CSS = """
    FlagChipBar {
        height: auto;
        min-height: 1;
        padding: 0 1;
    }
    """

    flags: reactive[frozenset] = reactive(frozenset())

    def __init__(self, **kwargs: object) -> None:  # type: ignore[override]
        super().__init__("", markup=False, **kwargs)  # type: ignore[arg-type]

    # --- Public API ------------------------------------------------------

    def set_flags(self, flags: set | frozenset) -> None:
        """Replace the active flag set and re-render.

        Accepts ``set`` or ``frozenset`` of ``DataFlag`` (or raw enum-value
        strings — useful when the payload comes off the wire). Unknown
        strings are silently dropped to stay crash-safe in the loop.
        """
        coerced: set[DataFlag] = set()
        for f in flags or ():
            c = _coerce(f)
            if c is not None:
                coerced.add(c)
        self.flags = frozenset(coerced)

    # --- Reactives -------------------------------------------------------

    def watch_flags(self, _old: frozenset, _new: frozenset) -> None:
        # Re-render whenever the reactive set changes.
        self.update(self.render())

    # --- Render ----------------------------------------------------------

    def render(self) -> str:  # type: ignore[override]
        if not self.flags:
            # Graceful empty bar — single space keeps Textual layout stable
            # without leaking the placeholder into HTML exports.
            return ""
        # Sort by (group order, flag name) for stable, group-clustered output.
        group_order = {"treatment": 0, "outcome": 1, "structural": 2, "design": 3, "unknown": 4}
        ordered = sorted(
            self.flags,
            key=lambda f: (group_order[_group_of(f)], f.value),
        )
        return " ".join(_render_chip(f) for f in ordered)


__all__ = ["FlagChipBar"]
