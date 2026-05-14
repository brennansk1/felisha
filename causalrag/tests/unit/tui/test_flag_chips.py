"""Unit tests for FlagChipBar (Sprint 4.6 hover-help on flag chips)."""

from __future__ import annotations

from causalrag.core.flag_descriptions import describe_safe
from causalrag.core.flags import DataFlag
from causalrag.tui.widgets.flag_chips import FlagChipBar


# ─────────────────────────── happy path ───────────────────────────


def test_flag_chip_bar_renders_three_flag_names_and_summaries() -> None:
    """Three chips render with all flag names and their summary text."""
    bar = FlagChipBar()
    flags = {
        DataFlag.BINARY_TREATMENT,
        DataFlag.CONTINUOUS_OUTCOME,
        DataFlag.SMALL_SAMPLE,
    }
    bar.set_flags(flags)
    rendered = bar.render()

    # Every flag's enum-value appears as the chip label.
    for f in flags:
        assert f.value in rendered, f"missing chip label for {f.value}"

    # Every flag's canonical summary appears inside the tooltip body.
    for f in flags:
        summary = describe_safe(f).summary
        assert summary in rendered, f"missing summary for {f.value}"


def test_flag_chip_bar_tooltip_includes_implication_and_routes() -> None:
    """Tooltips also carry the implication and routes_to estimator list."""
    bar = FlagChipBar()
    flag = DataFlag.BINARY_TREATMENT
    bar.set_flags({flag})
    rendered = bar.render()

    desc = describe_safe(flag)
    assert desc.implication in rendered
    # At least one routed estimator id should surface in the tooltip.
    assert any(r in rendered for r in desc.routes_to)
    # The hover-help attribute is the load-bearing surface.
    assert "title=" in rendered


# ─────────────────────────── colours ───────────────────────────


def test_flag_chip_bar_colors_chips_by_group() -> None:
    """Each flag group paints its chip with the spec-mandated background."""
    bar = FlagChipBar()
    bar.set_flags(
        {
            DataFlag.BINARY_TREATMENT,            # treatment → blue
            DataFlag.CONTINUOUS_OUTCOME,          # outcome   → green
            DataFlag.SMALL_SAMPLE,                # structural → orange
            DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT,  # design → purple
        }
    )
    rendered = bar.render()
    # Group-colour signatures from flag_chips._GROUP_BG.
    assert "#3a6ea5" in rendered  # blue   (treatment)
    assert "#3a8a55" in rendered  # green  (outcome)
    assert "#c2772a" in rendered  # orange (structural)
    assert "#7a4ea8" in rendered  # purple (design)


# ─────────────────────────── empty / edge ───────────────────────────


def test_flag_chip_bar_empty_renders_gracefully() -> None:
    """No flags → empty render, no exception, no leaked placeholder chips."""
    bar = FlagChipBar()
    rendered = bar.render()
    assert isinstance(rendered, str)
    assert "<chip" not in rendered
    # And explicit empty-set update doesn't change that.
    bar.set_flags(set())
    assert "<chip" not in bar.render()


def test_flag_chip_bar_accepts_raw_string_flag_values() -> None:
    """Strings from the wire are coerced; unknown strings are dropped."""
    bar = FlagChipBar()
    bar.set_flags({"binary_treatment", "not_a_real_flag"})
    rendered = bar.render()
    assert "binary_treatment" in rendered
    assert "not_a_real_flag" not in rendered


def test_flag_chip_bar_set_flags_is_idempotent_replace() -> None:
    """Each call to set_flags fully replaces the prior set (not a merge)."""
    bar = FlagChipBar()
    bar.set_flags({DataFlag.BINARY_TREATMENT})
    bar.set_flags({DataFlag.CONTINUOUS_OUTCOME})
    rendered = bar.render()
    assert "continuous_outcome" in rendered
    assert "binary_treatment" not in rendered
