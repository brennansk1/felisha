"""Loop scoring — information-theoretic chain-continuation rules.

Sprint 3.1 (Expected Information Gain) + Sprint 3.6 (Bayesian
saturation stopping) of PDD §33. A pure, side-effect-free sibling
module to :mod:`causalrag.master_loop`; it is *never* imported by the
master loop directly — instead the loop's wiring code can swap the
legacy ``info_gain_streak_below_eps`` heuristic for
:func:`should_continue_chain_eig` once the upstream call sites are
refactored.

The math follows Lindley (1956) / Chaloner & Verdinelli (1995) under a
Gaussian-approximate posterior on the treatment effect τ.
"""

from causalrag.loop_scoring.eig import (
    expected_information_gain,
    saturation_probability,
    should_continue_chain_eig,
)

__all__ = [
    "expected_information_gain",
    "saturation_probability",
    "should_continue_chain_eig",
]
