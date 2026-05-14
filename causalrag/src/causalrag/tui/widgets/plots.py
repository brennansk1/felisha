"""In-terminal plot panel (Sprint 4.2).

Renders five diagnostic plots used throughout the CausalRoadmap loop:

  * Power curves (Phase 2 feasibility)
  * Love plots — SMD before/after balancing (Phase 4 estimate)
  * Propensity-overlap densities (Phase 4 estimate)
  * CATE-vs-feature partial-dependence (Phase 4 estimate, heterogeneity)
  * Sensemakr contour plots (Phase 5 sensitivity)

Rendering is delegated to ``textual-plotext`` / ``plotext`` when those
optional deps are importable.  When they are not, every method becomes a
graceful no-op that stores the last payload on ``self.last_payload`` and
emits a small ASCII summary so the panel still has something to show.

The widget intentionally does NOT raise on missing deps — the TUI must
remain usable in minimal environments (CI, slim Docker, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from textual.widgets import Static

# ---------------------------------------------------------------------------
# Optional-dependency probe.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised indirectly
    import plotext as _plt  # type: ignore[import-not-found]

    _HAS_PLOTEXT = True
except Exception:  # noqa: BLE001 - any import-time failure is treated as absent
    _plt = None  # type: ignore[assignment]
    _HAS_PLOTEXT = False

try:  # pragma: no cover - exercised indirectly
    from textual_plotext import PlotextPlot  # type: ignore[import-not-found]

    _HAS_TEXTUAL_PLOTEXT = True
except Exception:  # noqa: BLE001
    PlotextPlot = None  # type: ignore[assignment]
    _HAS_TEXTUAL_PLOTEXT = False


def _plotting_available() -> bool:
    """True when *some* plot backend is importable."""
    return _HAS_PLOTEXT or _HAS_TEXTUAL_PLOTEXT


class PlotPanel(Static):
    """Container for an in-terminal plot.

    Renders via plotext / textual-plotext when those are importable;
    otherwise emits a short ASCII summary so the panel is never empty.
    Each ``render_*`` method stores the payload on
    ``self.last_payload`` and the rendered text on ``self.last_text``
    for inspection in tests.
    """

    DEFAULT_CSS = ""

    def __init__(self, title: str = "", **kwargs: Any) -> None:
        super().__init__("", markup=False, **kwargs)
        self.title: str = title
        self.last_kind: str | None = None
        self.last_payload: dict[str, Any] | None = None
        self.last_text: str = ""
        if title:
            self._emit(self._header())

    # ------------------------------------------------------------------ utils

    def _header(self) -> str:
        if not self.title:
            return ""
        return f"── {self.title} ──"

    def _emit(self, body: str) -> None:
        """Push ``body`` into the underlying ``Static`` and remember it."""
        text = body if not self.title else f"{self._header()}\n{body}"
        self.last_text = text
        self.update(text)

    def _plotext_canvas(self, width: int = 60, height: int = 18) -> str:
        """Render the *current* plotext figure to a string and clear it."""
        if not _HAS_PLOTEXT:  # pragma: no cover - defensive
            return ""
        try:
            _plt.plotsize(width, height)
            out = _plt.build()
            _plt.clear_figure()
            return out
        except Exception as exc:  # noqa: BLE001
            return f"<plotext error: {exc!s}>"

    # ----------------------------------------------------------- power curve

    def render_power_curve(self, ns: list[int], powers: list[float]) -> None:
        """Power vs sample-size curve (Phase 2 feasibility)."""
        self.last_kind = "power_curve"
        self.last_payload = {"ns": list(ns), "powers": list(powers)}

        if not ns or not powers:
            self._emit("power curve: <no data>")
            return

        if _HAS_PLOTEXT:
            _plt.clear_figure()
            _plt.plot(list(ns), list(powers), marker="dot")
            _plt.title("power curve")
            _plt.xlabel("n")
            _plt.ylabel("power")
            self._emit(self._plotext_canvas())
            return

        # Fallback ASCII summary.
        peak = max(range(len(powers)), key=lambda i: powers[i])
        lines = [
            f"power curve  (n={len(ns)} points)",
            f"  min power: {min(powers):.3f}  at n={ns[powers.index(min(powers))]}",
            f"  max power: {max(powers):.3f}  at n={ns[peak]}",
        ]
        self._emit("\n".join(lines))

    # ------------------------------------------------------------- love plot

    def render_love_plot(
        self,
        smds_before: dict[str, float],
        smds_after: dict[str, float],
    ) -> None:
        """Absolute SMD before vs after balancing — a 'love plot'."""
        self.last_kind = "love_plot"
        self.last_payload = {
            "smds_before": dict(smds_before),
            "smds_after": dict(smds_after),
        }

        keys = sorted(set(smds_before) | set(smds_after))
        if not keys:
            self._emit("love plot: <no covariates>")
            return

        before = [abs(smds_before.get(k, 0.0)) for k in keys]
        after = [abs(smds_after.get(k, 0.0)) for k in keys]

        if _HAS_PLOTEXT:
            _plt.clear_figure()
            xs = list(range(len(keys)))
            _plt.scatter(xs, before, marker="x", label="before")
            _plt.scatter(xs, after, marker="o", label="after")
            _plt.title("love plot — |SMD|")
            _plt.xlabel("covariate idx")
            _plt.ylabel("|SMD|")
            self._emit(self._plotext_canvas())
            return

        # ASCII fallback — one bar per covariate.
        max_len = max((len(k) for k in keys), default=1)
        lines = ["love plot — |SMD| before → after"]
        for k, b, a in zip(keys, before, after):
            arrow = "↓" if a < b else ("↑" if a > b else "·")
            lines.append(f"  {k.ljust(max_len)}  {b:5.3f}  {arrow}  {a:5.3f}")
        self._emit("\n".join(lines))

    # ----------------------------------------------------- propensity overlap

    def render_propensity_overlap(
        self,
        scores_treated: np.ndarray,
        scores_control: np.ndarray,
    ) -> None:
        """Density overlap of estimated propensities by treatment arm."""
        self.last_kind = "propensity_overlap"
        treated = np.asarray(scores_treated, dtype=float).ravel()
        control = np.asarray(scores_control, dtype=float).ravel()
        self.last_payload = {
            "scores_treated": treated,
            "scores_control": control,
        }

        if treated.size == 0 and control.size == 0:
            self._emit("propensity overlap: <no scores>")
            return

        if _HAS_PLOTEXT:
            _plt.clear_figure()
            bins = 20
            if treated.size:
                _plt.hist(treated.tolist(), bins=bins, label="treated")
            if control.size:
                _plt.hist(control.tolist(), bins=bins, label="control")
            _plt.title("propensity overlap")
            _plt.xlabel("ê(x)")
            _plt.ylabel("count")
            self._emit(self._plotext_canvas())
            return

        def _stats(a: np.ndarray) -> str:
            if a.size == 0:
                return "n=0"
            return (
                f"n={a.size}  mean={a.mean():.3f}  "
                f"min={a.min():.3f}  max={a.max():.3f}"
            )

        lines = [
            "propensity overlap",
            f"  treated: {_stats(treated)}",
            f"  control: {_stats(control)}",
        ]
        self._emit("\n".join(lines))

    # ------------------------------------------------------------ CATE PDP

    def render_cate_pdp(
        self,
        x: np.ndarray,
        cate: np.ndarray,
        feature_name: str = "x",
    ) -> None:
        """CATE vs a single feature — partial-dependence style."""
        self.last_kind = "cate_pdp"
        x_arr = np.asarray(x, dtype=float).ravel()
        c_arr = np.asarray(cate, dtype=float).ravel()
        self.last_payload = {
            "x": x_arr,
            "cate": c_arr,
            "feature_name": feature_name,
        }

        if x_arr.size == 0 or c_arr.size == 0:
            self._emit(f"CATE pdp ({feature_name}): <no data>")
            return

        if _HAS_PLOTEXT:
            _plt.clear_figure()
            _plt.plot(x_arr.tolist(), c_arr.tolist(), marker="dot")
            _plt.title(f"CATE vs {feature_name}")
            _plt.xlabel(feature_name)
            _plt.ylabel("CATE")
            self._emit(self._plotext_canvas())
            return

        lines = [
            f"CATE pdp  (feature={feature_name}, n={x_arr.size})",
            f"  CATE  min={c_arr.min():+.3f}  "
            f"mean={c_arr.mean():+.3f}  "
            f"max={c_arr.max():+.3f}",
            f"  {feature_name} range  [{x_arr.min():.3f}, {x_arr.max():.3f}]",
        ]
        self._emit("\n".join(lines))

    # ------------------------------------------------------- sensemakr contour

    def render_sensemakr_contour(self, partial_r2_grid: dict) -> None:
        """Sensemakr contour plot of bias-adjusted estimate.

        ``partial_r2_grid`` is expected to contain ``r2dz_x`` (treatment-side
        partial-R²), ``r2yz_dx`` (outcome-side partial-R²), and ``z``
        (the adjusted-estimate grid, shape ``(len(r2yz_dx), len(r2dz_x))``).
        Extra keys are tolerated.
        """
        self.last_kind = "sensemakr_contour"
        self.last_payload = dict(partial_r2_grid) if partial_r2_grid else {}

        r2dz = np.asarray(
            self.last_payload.get("r2dz_x", []), dtype=float
        ).ravel()
        r2yz = np.asarray(
            self.last_payload.get("r2yz_dx", []), dtype=float
        ).ravel()
        z = np.asarray(self.last_payload.get("z", []), dtype=float)

        if r2dz.size == 0 or r2yz.size == 0 or z.size == 0:
            self._emit("sensemakr contour: <no grid>")
            return

        if _HAS_PLOTEXT:
            _plt.clear_figure()
            try:
                # plotext gained matrix_plot in recent versions; fall back to
                # a heatmap-ish scatter if missing.
                if hasattr(_plt, "matrix_plot"):
                    _plt.matrix_plot(z.tolist())
                else:  # pragma: no cover
                    ys, xs = np.indices(z.shape)
                    _plt.scatter(xs.ravel().tolist(), ys.ravel().tolist())
                _plt.title("sensemakr contour — adj. estimate")
                _plt.xlabel("R²(D~Z|X)")
                _plt.ylabel("R²(Y~Z|D,X)")
                self._emit(self._plotext_canvas())
                return
            except Exception as exc:  # noqa: BLE001
                self._emit(f"sensemakr contour: plot error {exc!s}")
                return

        lines = [
            "sensemakr contour",
            f"  R²(D~Z|X) grid: n={r2dz.size}  "
            f"range=[{r2dz.min():.3f}, {r2dz.max():.3f}]",
            f"  R²(Y~Z|D,X) grid: n={r2yz.size}  "
            f"range=[{r2yz.min():.3f}, {r2yz.max():.3f}]",
            f"  adj. estimate  min={float(z.min()):+.3f}  "
            f"max={float(z.max()):+.3f}",
        ]
        self._emit("\n".join(lines))


__all__ = ["PlotPanel"]
