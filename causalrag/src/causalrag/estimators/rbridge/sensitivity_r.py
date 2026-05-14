"""R-based sensitivity wrappers — sensemakr (full), EValue, tipr.

These augment the Python-side sensitivity stack with the canonical R
implementations:

- ``sensemakr::sensemakr`` — partial-R² sensitivity with full benchmark
  output (more complete than PySensemakr; matches Cinelli-Hazlett 2020
  exactly).
- ``EValue::evalues.OR/HR/RR/MD`` — E-value for various effect scales
  (matches the headline result from the Python evalue.py but with the
  proper scale handling).
- ``tipr::tipr_adjust`` — tipping-point analysis: how much
  unmeasured-confounder bias would flip the conclusion?

Each function below returns a plain Python dict (not an EstimationResult)
because these are sensitivity / interpretation tools, not estimators.
The CLI ``sensitivity`` command can call them to enrich the verdict.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from causalrag.estimators.rbridge._r import (
    converter,
    r_session,
    r_session_metadata,
    require,
)


def sensemakr_full(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
    benchmark_covariates: tuple[str, ...] = (),
    q: float = 1.0,
    alpha: float = 0.05,
    kd: tuple[float, ...] = (1.0, 2.0, 3.0),
    ky: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Full sensemakr partial-R² sensitivity.

    Returns the headline robustness value, the extreme robustness value
    (RV(q)), the benchmark-adjusted-bias contours at the chosen kd / ky,
    and the standardized treatment t-stat. The dict structure matches
    the official sensemakr ``summary()`` output.
    """
    require("sensemakr")
    ro = r_session()
    cols = [outcome, treatment, *covariates]
    work = df[cols].dropna()
    with converter():
        ro.globalenv["df_"] = ro.conversion.py2rpy(work)
    formula = f"{outcome} ~ {treatment} + " + " + ".join(covariates)
    ro.r(f"olsfit_ <- lm({formula}, data = df_)")
    ky_arg = "ky" if ky is not None else "kd"
    if ky is not None:
        ro.r(f"ky_v <- c({', '.join(str(v) for v in ky)})")
    ro.r(f"kd_v <- c({', '.join(str(v) for v in kd)})")
    bench_arg = ""
    if benchmark_covariates:
        bench_str = ", ".join(f'"{b}"' for b in benchmark_covariates)
        bench_arg = f', benchmark_covariates = c({bench_str})'
    if ky is not None:
        ro.r(
            f'res_ <- sensemakr::sensemakr(model = olsfit_, treatment = "{treatment}", '
            f"q = {q}, alpha = {alpha}, kd = kd_v, ky = ky_v{bench_arg})"
        )
    else:
        ro.r(
            f'res_ <- sensemakr::sensemakr(model = olsfit_, treatment = "{treatment}", '
            f"q = {q}, alpha = {alpha}, kd = kd_v{bench_arg})"
        )
    out = {
        "estimate": float(list(ro.r("res_$sensitivity_stats$estimate"))[0]),
        "se": float(list(ro.r("res_$sensitivity_stats$se"))[0]),
        "t_value": float(list(ro.r("res_$sensitivity_stats$t_statistic"))[0]),
        "robustness_value_q": float(list(ro.r("res_$sensitivity_stats$rv_q"))[0]),
        "robustness_value_qa": float(list(ro.r("res_$sensitivity_stats$rv_qa"))[0]),
        "partial_r2_tz_x": float(list(ro.r("res_$sensitivity_stats$r2yz.dx"))[0])
        if "r2yz.dx" in list(ro.r("names(res_$sensitivity_stats)"))
        else None,
        "backend": "sensemakr (R)",
        "r_session_metadata": r_session_metadata(),
    }
    # Benchmark bounds, if benchmarks supplied
    if benchmark_covariates:
        try:
            bench_df = ro.r("res_$bounds")
            out["bounds"] = {
                "r2dz_x": list(ro.r("res_$bounds$r2dz.x")),
                "r2yz_dx": list(ro.r("res_$bounds$r2yz.dx")),
                "adjusted_estimate": list(ro.r("res_$bounds$adjusted_estimate")),
                "adjusted_se": list(ro.r("res_$bounds$adjusted_se")),
            }
        except Exception:
            pass
    return out


def evalue_r(
    point_estimate: float,
    *,
    scale: str = "RR",
    lo: float | None = None,
    hi: float | None = None,
    true_value: float = 1.0,
    outcome_prevalence: float | None = None,
) -> dict[str, Any]:
    """E-value for OR / RR / HR / MD via the R EValue package.

    ``scale``: one of ``RR``, ``OR``, ``HR``, ``OLS`` (mean difference).
    Returns the headline E-value and CI-bound E-value.
    """
    require("EValue")
    ro = r_session()
    fn = {
        "RR": "evalues.RR",
        "OR": "evalues.OR",
        "HR": "evalues.HR",
        "OLS": "evalues.OLS",
        "MD": "evalues.MD",
    }.get(scale.upper())
    if fn is None:
        raise ValueError(f"scale must be one of RR / OR / HR / OLS / MD; got {scale!r}")
    args = [f"est = {point_estimate}", f"true = {true_value}"]
    if lo is not None:
        args.append(f"lo = {lo}")
    if hi is not None:
        args.append(f"hi = {hi}")
    if fn == "evalues.OR" and outcome_prevalence is not None:
        args.append(f"rare = {'FALSE' if outcome_prevalence > 0.15 else 'TRUE'}")
    expr = f"EValue::{fn}({', '.join(args)})"
    res = ro.r(expr)
    # The R result is a 2x3 matrix; first row is the estimate, second is the CI bound
    # Column 2 is the E-value
    mat = ro.r(f"as.matrix({expr})")
    rows = ro.r("nrow(M_ <- as.matrix(E_obj_ <- {})); rownames(M_)".format(expr))
    e_point = float(list(ro.r("as.matrix({})['E-values','point']".format(expr)))[0])
    try:
        e_lo = float(list(ro.r("as.matrix({})['E-values','lower']".format(expr)))[0])
    except Exception:
        e_lo = None
    try:
        e_hi = float(list(ro.r("as.matrix({})['E-values','upper']".format(expr)))[0])
    except Exception:
        e_hi = None
    return {
        "e_value_point": e_point,
        "e_value_lower": e_lo,
        "e_value_upper": e_hi,
        "scale": scale,
        "backend": "EValue (R)",
        "r_session_metadata": r_session_metadata(),
    }


def tipping_point(
    *,
    estimate: float,
    se: float,
    n_treated: int,
    n_untreated: int,
    confounder_prevalence: float = 0.5,
) -> dict[str, Any]:
    """tipr::tipr — tipping-point sensitivity.

    How strong an unmeasured confounder would need to be to render the
    estimate non-significant? Reports the smallest such confounder
    strength in standardized-bias units.
    """
    require("tipr")
    ro = r_session()
    ro.r(
        f"tip_ <- tipr::tip_with_continuous("
        f"effect_observed = {estimate}, se = {se}, "
        f"n_obs = {n_treated + n_untreated}, p1 = {confounder_prevalence})"
    )
    return {
        "tipping_smd": float(list(ro.r("tip_$smd"))[0]) if list(ro.r("'smd' %in% names(tip_)")) else None,
        "tipping_confounder_strength": float(list(ro.r("tip_$gamma"))[0])
        if list(ro.r("'gamma' %in% names(tip_)"))
        else None,
        "backend": "tipr (R)",
        "r_session_metadata": r_session_metadata(),
    }
