"""R-bridged estimators (PDD §17 + §29).

The R bridge gives us first-class access to causal-inference methods that
have no production Python equivalent or whose R implementations are
canonical:

- **grf** — Athey-Wager causal forest + causal survival forest
- **lmtp** — longitudinal modified treatment policies + stochastic
  interventions + dosage (mixture exposure)
- **MatchIt** + **marginaleffects** — propensity score matching with
  full post-estimation g-computation
- **sensemakr** — partial-R² sensitivity (full benchmarking)
- **survRM2** — RMST contrast for survival
- **mediation** + **CMAverse** — NDE/NIE causal mediation
- **WeightIt** + **cobalt** — propensity weighting + balance diagnostics
- **EValue** + **tipr** — sensitivity with tipping-point analysis
- **tmle3** / **sl3** / **origami** — TMLE for ATE/NIE/longitudinal
- **bartCause** / **BCF** — Bayesian Causal Forest
- **bnlearn** — Bayesian network discovery

Every wrapper module self-registers with the estimator registry under the
appropriate ``DataFlag`` combinations so the auto-selector can route
flag situations to the right R-bridged method. Modules import the R
package lazily — importing this package does not start an R session.
"""

from causalrag.estimators.rbridge._r import (
    RBridgeError,
    RPackageMissing,
    RSessionInfo,
    converter,
    r_call,
    r_session,
    r_session_metadata,
    require,
)

__all__ = [
    "RBridgeError",
    "RPackageMissing",
    "RSessionInfo",
    "converter",
    "r_call",
    "r_session",
    "r_session_metadata",
    "require",
]
