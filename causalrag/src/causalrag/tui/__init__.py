"""CausalRoadmap TUI — a Claude-Code-style terminal interface to the pipeline.

Built with Textual. Palette + chrome ported from the design bundle
(claude.ai/design handoff) — dark navy, Merriweather-styled prose where the
terminal allows, JetBrains-Mono-styled data tables, `/slash` command menu,
four-layer hallucination-guard status strip.

Entry point: ``causalrag tui`` (or ``python -m causalrag.tui``).
"""

from causalrag.tui.app import CausalRoadmapTUI, run

__all__ = ["CausalRoadmapTUI", "run"]
