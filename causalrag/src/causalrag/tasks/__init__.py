"""Downstream task adapters that consume a fitted CausalRoadmap pipeline.

Each task module exposes a single high-level entry point so the master loop
(or external callers) can request a specific analytical mode without having
to assemble the lower-level estimators by hand.

Currently exported:
    * :func:`causalrag.tasks.rca.attribute_metric_change` — two-period
      root-cause attribution for a metric (Sprint 5.3).
    * :func:`causalrag.tasks.impact.analyze_impact` — causal-forecasting
      / intervention-impact analysis combining Brodersen ``CausalImpact``,
      Ben-Michael augmented SCM, and Athey matrix completion
      (Sprint 5.4).
"""

from __future__ import annotations

__all__: list[str] = []

try:  # pragma: no cover - optional sibling task (Sprint 5.4)
    from causalrag.tasks.impact import (  # noqa: F401
        ImpactFinding,
        ImpactReport,
        analyze_impact,
    )
except ImportError:
    pass
else:  # pragma: no cover - executed only when impact module exists
    __all__ += ["ImpactFinding", "ImpactReport", "analyze_impact"]

# RCA agent (Sprint 5.3) re-exports are appended here once that module
# lands so callers can do ``from causalrag.tasks import RootCauseReport``.
try:  # pragma: no cover - optional sibling task
    from causalrag.tasks.rca import (  # noqa: F401
        RootCauseFinding,
        RootCauseReport,
        attribute_metric_change,
    )
except ImportError:
    pass
else:  # pragma: no cover - executed only when rca module exists
    __all__ += ["RootCauseFinding", "RootCauseReport", "attribute_metric_change"]

# Uplift / policy-targeting agent (Sprint 5.5).
try:  # pragma: no cover - optional sibling task
    from causalrag.tasks.uplift import (  # noqa: F401
        TargetingReport,
        UpliftCurve,
        build_targeting_report,
        policy_tree,
    )
except ImportError:
    pass
else:  # pragma: no cover - executed only when uplift module exists
    __all__ += [
        "TargetingReport",
        "UpliftCurve",
        "build_targeting_report",
        "policy_tree",
    ]
