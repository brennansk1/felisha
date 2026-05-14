"""Specification curve / multiverse analysis (PDD §11.1).

Subpackage entry point bundling two complementary multiverse tools:

- :mod:`causalrag.multiverse.specr` (Sprint 6.2) — the
  Simonsohn-Simmons-Nelson 2020 specification curve over the
  (adjustment-set x estimator x trim x time-window) product, with the
  Del Giudice-Gangestad 2021 caveat on principled equivalence.
- :mod:`causalrag.multiverse.dag_bma` (Sprint 6.3) — multiverse-of-DAGs
  Bayesian model averaging across candidate causal graphs.
"""

from causalrag.multiverse.dag_bma import (
    DAGBMAFinding,
    DAGBMAReport,
    dag_bma,
)
from causalrag.multiverse.specr import (
    SpecCurve,
    SpecResult,
    render_html,
    specification_curve,
)

__all__ = [
    "DAGBMAFinding",
    "DAGBMAReport",
    "SpecCurve",
    "SpecResult",
    "dag_bma",
    "render_html",
    "specification_curve",
]
