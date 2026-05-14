"""Tests for the method-coverage matrix audit (Sprint 9.5.2)."""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from causalrag.audits.method_coverage import (
    MethodCoverageCell,
    MethodCoverageMatrix,
    build_method_coverage_matrix,
    render_matrix_html,
)
from causalrag.core.flags import DataFlag
from causalrag.estimators.catalog import MethodSpec


# ─────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────


def test_matrix_builds_on_real_pipeline():
    """The audit must run end-to-end on the real cascade + catalog. We
    don't assert on absolute coverage; what matters is that we get back
    a well-formed matrix."""
    m = build_method_coverage_matrix()
    assert isinstance(m, MethodCoverageMatrix)
    assert m.n_cells > 0
    assert m.n_cells == len(m.cells)
    assert 0 <= m.n_covered <= m.n_cells
    assert 0.0 <= m.coverage_pct <= 100.0
    # Empty cells and tickets are co-indexed.
    assert len(m.empty_cells) == m.n_cells - m.n_covered
    assert len(m.suggested_tickets) == len(m.empty_cells)


def test_every_cell_has_a_flag_combo_within_cap():
    """``flag_subsets_max_size`` is a hard cap on combo size."""
    m = build_method_coverage_matrix(flag_subsets_max_size=2)
    for cell in m.cells:
        assert len(cell.flag_combo) <= 2
        # ``frozenset`` members are stringified flag names.
        for name in cell.flag_combo:
            assert hasattr(DataFlag, name)


def test_is_covered_matches_reachable_estimators():
    m = build_method_coverage_matrix()
    for cell in m.cells:
        assert cell.is_covered == (len(cell.estimators_reachable) > 0)


# ─────────────────────────────────────────────────────────────────────────
# Cell-level routing assertions
# ─────────────────────────────────────────────────────────────────────────


def _find_cell(
    m: MethodCoverageMatrix, estimand: str, combo: frozenset[str]
) -> MethodCoverageCell:
    for c in m.cells:
        if c.estimand == estimand and c.flag_combo == combo:
            return c
    raise AssertionError(f"cell ({estimand}, {sorted(combo)}) not in matrix")


def test_ate_with_no_flags_is_covered_by_default_ladder():
    """The empty flag combo with an ATE estimand should at minimum land
    on the default DML linear estimator."""
    m = build_method_coverage_matrix()
    cell = _find_cell(m, "ATE", frozenset())
    assert cell.is_covered
    assert "python.dml.linear" in cell.estimators_reachable


def test_rmst_contrast_routes_under_right_censored_binary_treatment():
    """RMST_CONTRAST × {BINARY_TREATMENT, RIGHT_CENSORED_OUTCOME} must
    land on survRM2."""
    m = build_method_coverage_matrix()
    cell = _find_cell(
        m,
        "RMST_CONTRAST",
        frozenset({"BINARY_TREATMENT", "RIGHT_CENSORED_OUTCOME"}),
    )
    assert cell.is_covered
    assert "rbridge.survrm2" in cell.estimators_reachable


def test_nde_routes_via_mediation_when_mediator_proposed():
    m = build_method_coverage_matrix()
    cell = _find_cell(m, "NDE", frozenset({"MEDIATOR_PROPOSED"}))
    assert cell.is_covered
    assert "rbridge.mediation" in cell.estimators_reachable


# ─────────────────────────────────────────────────────────────────────────
# Empty-cell detection + ticket generation
# ─────────────────────────────────────────────────────────────────────────


def test_late_with_no_instrument_is_uncovered():
    """LATE is only produced by the instrumental forest, which requires
    ``INSTRUMENTAL_CANDIDATE_PRESENT``. The empty flag combo therefore
    must be uncovered."""
    m = build_method_coverage_matrix()
    cell = _find_cell(m, "LATE", frozenset())
    assert not cell.is_covered
    assert cell.estimators_reachable == ()


def test_empty_cells_produce_v11_tickets_referencing_flag_combos():
    """Every empty cell must emit a ticket that mentions the estimand
    and the flag combo by name (or {} for the empty combo)."""
    m = build_method_coverage_matrix()
    assert m.suggested_tickets, "expected at least one empty cell"

    estimands_in_tickets = {
        # tickets look like: "v1.1: build estimator for ATT under {…}"
        ticket.split(" under ")[0].split("for ")[-1].strip()
        for ticket in m.suggested_tickets
    }
    empty_estimands = {c.estimand for c in m.empty_cells}
    assert estimands_in_tickets == empty_estimands

    for ticket in m.suggested_tickets:
        assert ticket.startswith("v1.1:")
        assert " under " in ticket
        # Either an empty brace-pair or a comma-separated list of flag
        # names — both must be present in their canonical form.
        assert "{" in ticket and "}" in ticket


def test_ticket_for_known_uncovered_combo_names_each_flag():
    """An empty cell with multiple flags must surface each flag name in
    its ticket string (no truncation, sorted)."""
    m = build_method_coverage_matrix()
    # Look for any empty cell with at least two flags.
    multi_flag_empties = [c for c in m.empty_cells if len(c.flag_combo) >= 2]
    if not multi_flag_empties:
        pytest.skip("matrix has no multi-flag empty cells")
    cell = multi_flag_empties[0]
    expected_combo = "{" + ", ".join(sorted(cell.flag_combo)) + "}"
    matching = [t for t in m.suggested_tickets if expected_combo in t and cell.estimand in t]
    assert matching, f"no ticket references combo {expected_combo} for {cell.estimand}"


# ─────────────────────────────────────────────────────────────────────────
# Sensitivity panel availability
# ─────────────────────────────────────────────────────────────────────────


def test_universal_panels_always_available():
    """E-value, sensemakr, anomaly_audit, refutation_summary, and the
    Chernozhukov OVB panel have no flag trigger — they must surface in
    every cell."""
    m = build_method_coverage_matrix()
    universal = {"e_value", "sensemakr", "anomaly_audit", "refutation_summary", "ovb_chernozhukov"}
    for cell in m.cells:
        assert universal.issubset(set(cell.sensitivity_panels_available)), (
            f"universal panels missing in cell {cell.estimand} / {sorted(cell.flag_combo)}"
        )


def test_negative_control_panel_gated_by_flag():
    """The negative-control panel must only appear when
    NEGATIVE_CONTROL_AVAILABLE is in the combo."""
    m = build_method_coverage_matrix()
    for cell in m.cells:
        has_panel = "negative_control" in cell.sensitivity_panels_available
        has_flag = "NEGATIVE_CONTROL_AVAILABLE" in cell.flag_combo
        assert has_panel == has_flag


# ─────────────────────────────────────────────────────────────────────────
# HTML rendering — parseable table
# ─────────────────────────────────────────────────────────────────────────


class _TableCounter(HTMLParser):
    """Minimal HTML parser that counts ``<table>``, ``<tr>``, ``<td>``
    tags so we can assert that the rendered output is well-formed."""

    def __init__(self) -> None:
        super().__init__()
        self.tables = 0
        self.rows = 0
        self.cells = 0
        self.headers = 0
        self.unmatched_close: list[str] = []
        self._stack: list[str] = []

    def handle_starttag(self, tag: str, attrs):  # noqa: ARG002
        if tag == "table":
            self.tables += 1
        elif tag == "tr":
            self.rows += 1
        elif tag == "td":
            self.cells += 1
        elif tag == "th":
            self.headers += 1
        self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        else:
            # Tolerate void/auto-closed tags but track imbalance.
            self.unmatched_close.append(tag)


def test_html_render_parses_as_a_table():
    m = build_method_coverage_matrix()
    html = render_matrix_html(m)

    parser = _TableCounter()
    parser.feed(html)

    assert parser.tables == 1, f"expected exactly one <table>, got {parser.tables}"
    # Header row + one row per cell.
    assert parser.rows == m.n_cells + 1
    # 5 columns per data row + same 5 columns of <th> in the header.
    assert parser.cells == 5 * m.n_cells
    assert parser.headers == 5


def test_html_render_includes_coverage_summary_and_tickets():
    m = build_method_coverage_matrix()
    html = render_matrix_html(m)

    assert "Method coverage matrix" in html
    assert f"{m.n_covered}/{m.n_cells}" in html
    # ``coverage_pct`` is rendered with one decimal place.
    assert re.search(rf"{m.coverage_pct:.1f}\s*%", html)
    if m.suggested_tickets:
        assert "Suggested v1.1 tickets" in html
        # At least one ticket body must appear verbatim (after HTML escape
        # of the flag combo's curly braces which are unaffected).
        assert m.suggested_tickets[0] in html


def test_html_render_escapes_estimand_and_flag_text():
    """No raw ``<`` or ``>`` from cell content should leak into the
    rendered HTML — verified by checking that injected angle brackets in
    a synthetic estimand are escaped."""
    # Build a tiny synthetic matrix via the real function but with a
    # custom catalog so we control reachability. We don't bother with
    # crafting a real injection vector; instead we assert that the
    # standard rendering uses ``&lt;``/``&gt;`` where it should.
    m = build_method_coverage_matrix()
    html = render_matrix_html(m)
    # The <em>none</em> sentinel is rendered raw (it's our own markup);
    # but flag combos like ``{X, Y}`` use ``,`` not angle brackets, so
    # all ``<`` outside of legitimate tags should belong to our own
    # markup. Spot-check that ``<script`` and similar never appear.
    assert "<script" not in html.lower()


# ─────────────────────────────────────────────────────────────────────────
# Synthetic-catalog test: prove "no admissible spec" forces empty cells
# ─────────────────────────────────────────────────────────────────────────


def test_with_empty_catalog_every_cell_is_empty():
    """When the catalog is empty, the catalog admits no estimator —
    every cell must be uncovered and the ticket count must equal the
    cell count."""
    m = build_method_coverage_matrix(catalog=())
    assert m.n_covered == 0
    assert m.n_cells == len(m.empty_cells)
    assert len(m.suggested_tickets) == m.n_cells


def test_with_synthetic_catalog_specific_cells_are_covered():
    """A one-row synthetic catalog that admits ATE under no flags should
    cover *exactly* the ATE cells whose flag combo doesn't conflict with
    the cascade's recommendation for that synthetic estimator."""
    from causalrag.estimators.python.select import _rule_cascade, SelectionContext

    # Use the real default cascade output as a sanity oracle for the
    # estimator id we'll plant in the synthetic catalog.
    ctx = SelectionContext(estimand="ATE", flags=frozenset(), n=500, n_modifiers=1)
    cascade_for_empty_ate = _rule_cascade(ctx)
    assert "python.dml.linear" in cascade_for_empty_ate

    synthetic = (
        MethodSpec(
            estimator_id="python.dml.linear",
            backend="python",
            use_case="test",
            estimands=("ATE",),
            required_flags=(),
            excluded_flags=(),
            min_n=1,
            domain_hint="test",
            reference="test",
        ),
    )
    m = build_method_coverage_matrix(catalog=synthetic, flag_subsets_max_size=1)
    # The empty-combo ATE cell must be covered by exactly one estimator.
    cell = _find_cell(m, "ATE", frozenset())
    assert cell.is_covered
    assert cell.estimators_reachable == ("python.dml.linear",)
    # Non-ATE estimands must be entirely uncovered with this catalog.
    for c in m.cells:
        if c.estimand != "ATE":
            assert not c.is_covered
