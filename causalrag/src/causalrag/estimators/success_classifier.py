"""Per-estimator success classifier (Sprint 9.3).

Each registered estimator gets its own thin :class:`LogisticRegression`
that predicts P(this estimator fits cleanly) given the context
``(flags, n, p, missingness_rate)``. The auto-cascade can call
:meth:`EstimatorSuccessClassifier.predict` *before* attempting a fit and
short-circuit candidates whose predicted success probability is too low —
saving wall clock on estimators we already know don't work in this regime.

Training data
-------------
The corpus is built from postmortem records persisted across runs (see
``loop_observability.postmortem``). Each ``history_rows`` dict is expected
to expose:

* ``estimator_id``  — string id matching the catalog (None on failure).
* ``estimator_attempts`` — list of estimator ids that were tried; the
  rows whose ``estimator_id is None`` and where ``estimator_attempts``
  contains *us* count as a failure for that estimator.
* ``flags`` — iterable of :class:`DataFlag` (or their string values).
* ``n``, ``p``, ``missingness_rate`` — the context the encoder needs.

Anything missing is treated permissively: rows that don't mention this
estimator (neither in ``estimator_id`` nor ``estimator_attempts``) are
ignored.

Encoder
-------
The feature vector mirrors the Sprint 9.2 priors encoder:

* one binary indicator per :class:`DataFlag` value (sorted by name),
* ``log10(n + 1)``,
* ``log10(p + 1)``,
* ``missingness_rate`` (already a 0..1 scalar).

This keeps coefficients directly readable as "how does this flag /
sample-size axis move the log-odds of success" which we expose via
:attr:`SuccessPrediction.top_features_contributing`.

Persistence
-----------
:meth:`save` / :meth:`load` use :mod:`pickle` (sklearn estimators
pickle cleanly and we deliberately avoid adding a joblib dep). The
on-disk payload also records the encoder schema (sorted flag names) so
loading a model trained against an older flag vocabulary stays safe.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # sklearn is already a transitive dep via econml; soft-import for clarity.
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "scikit-learn is required for the success classifier "
        "(causalrag.estimators.success_classifier)"
    ) from exc

from causalrag.core.flags import DataFlag


# ─────────── feature schema ──────────────────────────────────────────────


def _flag_vocab() -> list[str]:
    """Sorted list of DataFlag *values* — stable feature ordering."""
    return sorted(f.value for f in DataFlag)


def _flag_value(f: Any) -> str:
    """Coerce a flag-ish thing to its string value."""
    if isinstance(f, DataFlag):
        return f.value
    return str(f)


def _feature_names(vocab: list[str]) -> list[str]:
    return [f"flag::{v}" for v in vocab] + ["log10_n", "log10_p", "missingness_rate"]


# ─────────── dataclasses ─────────────────────────────────────────────────


@dataclass
class SuccessPrediction:
    """Output of :meth:`EstimatorSuccessClassifier.predict`."""

    estimator_id: str
    probability_of_success: float
    top_features_contributing: list[tuple[str, float]]
    rationale: str


# ─────────── per-estimator classifier ────────────────────────────────────


@dataclass
class EstimatorSuccessClassifier:
    """Logistic-regression success predictor for a single estimator id.

    The classifier is *unfitted* until :meth:`train` is called. When
    fewer than two distinct labels are observed, training degrades to a
    trivial constant predictor (no LogisticRegression fitted) — sklearn
    raises in that regime and the policy answer is unambiguous anyway.
    """

    estimator_id: str
    _model: LogisticRegression | None = field(default=None, init=False, repr=False)
    _vocab: list[str] = field(default_factory=_flag_vocab, init=False, repr=False)
    _constant: float | None = field(default=None, init=False, repr=False)
    _n_train: int = field(default=0, init=False, repr=False)

    # -- encoding ---------------------------------------------------------

    def encode(
        self,
        *,
        flags: Iterable[Any],
        n: int,
        p: int,
        missingness_rate: float,
    ) -> np.ndarray:
        """Map context → feature vector (1-D, dtype float64)."""
        present = {_flag_value(f) for f in flags}
        flag_bits = np.array(
            [1.0 if v in present else 0.0 for v in self._vocab],
            dtype=np.float64,
        )
        tail = np.array(
            [
                float(np.log10(max(int(n), 0) + 1)),
                float(np.log10(max(int(p), 0) + 1)),
                float(np.clip(missingness_rate, 0.0, 1.0)),
            ],
            dtype=np.float64,
        )
        return np.concatenate([flag_bits, tail])

    # -- training corpus extraction --------------------------------------

    def _row_label(self, row: dict[str, Any]) -> int | None:
        """Return 1 (success), 0 (failure), or None (row not about us)."""
        if row.get("estimator_id") == self.estimator_id:
            # Estimator successfully produced a point estimate iff it set
            # estimator_id; we still allow callers to mark a row failed
            # explicitly via ``success=False`` (used by the test harness).
            success = row.get("success", True)
            return 1 if success else 0
        attempts = row.get("estimator_attempts") or []
        if self.estimator_id in attempts and row.get("estimator_id") != self.estimator_id:
            return 0
        return None

    def _extract(
        self, history_rows: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        X: list[np.ndarray] = []
        y: list[int] = []
        for row in history_rows:
            label = self._row_label(row)
            if label is None:
                continue
            X.append(
                self.encode(
                    flags=row.get("flags") or [],
                    n=int(row.get("n", 0)),
                    p=int(row.get("p", 0)),
                    missingness_rate=float(row.get("missingness_rate", 0.0)),
                )
            )
            y.append(label)
        if not X:
            return np.empty((0, len(self._vocab) + 3)), np.empty((0,), dtype=int)
        return np.vstack(X), np.array(y, dtype=int)

    # -- training --------------------------------------------------------

    def train(self, history_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Fit the per-estimator logistic regression.

        Returns a small summary dict (n_train, class balance, fitted flag)
        suitable for logging without leaking model internals.
        """
        X, y = self._extract(history_rows)
        self._n_train = int(X.shape[0])
        if self._n_train == 0:
            self._model = None
            self._constant = 0.5
            return {
                "estimator_id": self.estimator_id,
                "n_train": 0,
                "fitted": False,
                "reason": "no_relevant_rows",
            }

        unique = np.unique(y)
        if unique.size < 2:
            # Degenerate corpus — store the constant and skip sklearn.
            self._model = None
            self._constant = float(unique[0])
            return {
                "estimator_id": self.estimator_id,
                "n_train": self._n_train,
                "fitted": False,
                "reason": "single_class",
                "constant": self._constant,
            }

        # L2-regularized, no scaling needed — features are bounded.
        model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        model.fit(X, y)
        self._model = model
        self._constant = None
        return {
            "estimator_id": self.estimator_id,
            "n_train": self._n_train,
            "fitted": True,
            "pos_rate": float(np.mean(y)),
        }

    # -- inference -------------------------------------------------------

    def _predict_proba(self, x: np.ndarray) -> float:
        if self._model is not None:
            return float(self._model.predict_proba(x.reshape(1, -1))[0, 1])
        if self._constant is not None:
            return float(self._constant)
        return 0.5

    def _top_features(self, x: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        if self._model is None:
            return []
        coefs = self._model.coef_[0]
        contributions = coefs * x  # per-feature log-odds contribution at this point
        names = _feature_names(self._vocab)
        order = np.argsort(-np.abs(contributions))[:k]
        return [(names[i], float(contributions[i])) for i in order]

    def predict(
        self,
        *,
        flags: Iterable[Any],
        n: int,
        p: int,
        missingness_rate: float,
    ) -> SuccessPrediction:
        x = self.encode(flags=flags, n=n, p=p, missingness_rate=missingness_rate)
        prob = self._predict_proba(x)
        top = self._top_features(x)
        if self._model is None and self._constant is not None:
            rationale = (
                f"degenerate corpus (constant predictor = {self._constant:.2f}); "
                f"n_train={self._n_train}"
            )
        elif self._model is None:
            rationale = "untrained — defaulting to uninformative prior of 0.5"
        else:
            driver = top[0][0] if top else "(none)"
            direction = "raises" if (top and top[0][1] > 0) else "lowers"
            rationale = (
                f"P(success)={prob:.2f}; dominant driver '{driver}' {direction} "
                f"log-odds; n_train={self._n_train}"
            )
        return SuccessPrediction(
            estimator_id=self.estimator_id,
            probability_of_success=prob,
            top_features_contributing=top,
            rationale=rationale,
        )

    # -- persistence -----------------------------------------------------

    def _state(self) -> dict[str, Any]:
        return {
            "estimator_id": self.estimator_id,
            "vocab": self._vocab,
            "model": self._model,
            "constant": self._constant,
            "n_train": self._n_train,
            "schema_version": 1,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self._state(), fh)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as fh:
            state = pickle.load(fh)
        if state.get("schema_version") != 1:
            raise ValueError(
                f"unsupported success-classifier schema_version: "
                f"{state.get('schema_version')!r}"
            )
        self.estimator_id = state["estimator_id"]
        self._vocab = list(state["vocab"])
        self._model = state["model"]
        self._constant = state["constant"]
        self._n_train = int(state["n_train"])


# ─────────── fleet manager ───────────────────────────────────────────────


@dataclass
class FleetSuccessClassifier:
    """One :class:`EstimatorSuccessClassifier` per registered estimator id."""

    classifiers: dict[str, EstimatorSuccessClassifier] = field(default_factory=dict)

    def add(self, estimator_id: str) -> None:
        if estimator_id not in self.classifiers:
            self.classifiers[estimator_id] = EstimatorSuccessClassifier(
                estimator_id=estimator_id
            )

    def train_all(self, history_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            eid: clf.train(history_rows) for eid, clf in self.classifiers.items()
        }

    def predict_all(
        self,
        *,
        flags: Iterable[Any],
        n: int,
        p: int,
        missingness_rate: float,
    ) -> list[SuccessPrediction]:
        return [
            clf.predict(flags=flags, n=n, p=p, missingness_rate=missingness_rate)
            for clf in self.classifiers.values()
        ]


__all__ = [
    "SuccessPrediction",
    "EstimatorSuccessClassifier",
    "FleetSuccessClassifier",
]
