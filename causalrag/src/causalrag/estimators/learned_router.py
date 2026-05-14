"""ML-learned dispatch router (Sprint 9.2).

Trains a multi-class classifier offline on dispatch telemetry rows of the form
``(flags vector, n, p, n_modifiers, treatment_prevalence) → chosen_estimator``
and serves predictions to augment — never replace — the hand-coded rule
cascade in :mod:`causalrag.estimators.python.select`.

Design notes
------------
* The router is intentionally *augmenting*: it produces a recommended
  estimator plus a top-3 candidate list, and the caller (rule cascade) is
  free to ignore it. Telemetry-driven learning closes the loop on
  patterns the human-authored rules miss.
* We use scikit-learn's :class:`~sklearn.ensemble.GradientBoostingClassifier`
  to keep the dependency footprint minimal — XGBoost would be a hard new
  dep. Trees + softmax-style decision_function provide reasonable
  per-feature attribution via :func:`numpy.gradient`-free fallback
  (we read ``feature_importances_`` × ``feature_value`` as a SHAP
  surrogate; full SHAP would add the ``shap`` dep we want to avoid).
* The model is serialized via :mod:`pickle`. The format is internal —
  callers should not depend on it across releases.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from causalrag.core.flags import DataFlag

if TYPE_CHECKING:  # pragma: no cover - typing only
    from causalrag.estimators.python.select import SelectionContext


# Canonical feature ordering. We sort DataFlag by name once so the encoder
# is stable across Python invocations (StrEnum iteration order is
# definition-order, but we want a deterministic alphabetical layout).
_FLAG_FEATURES: tuple[DataFlag, ...] = tuple(sorted(DataFlag, key=lambda f: f.name))
_CONTINUOUS_FEATURES: tuple[str, ...] = ("n", "p", "n_modifiers", "treatment_prevalence")
FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"flag::{f.value}" for f in _FLAG_FEATURES] + list(_CONTINUOUS_FEATURES)
)


@dataclass
class RouterPrediction:
    """Output of :meth:`LearnedDispatchRouter.predict`."""

    recommended_estimator: str
    probability: float
    top_3_candidates: list[tuple[str, float]]
    feature_contributions: dict[str, float]
    rationale: str = ""
    # Optional metadata for debugging; not part of the public contract.
    extras: dict[str, Any] = field(default_factory=dict)


def _coerce_flag(value: Any) -> DataFlag | None:
    if isinstance(value, DataFlag):
        return value
    if isinstance(value, str):
        try:
            return DataFlag(value)
        except ValueError:
            try:
                return DataFlag[value]
            except KeyError:
                return None
    return None


def _encode_row(
    *,
    flags: Any,
    n: float | int | None,
    p: float | int | None,
    n_modifiers: float | int | None,
    treatment_prevalence: float | None,
) -> np.ndarray:
    """Encode a single (flags, scalars) tuple into the canonical feature vector.

    Missing scalar values are replaced with ``0.0``; this matches the
    convention used in the rule cascade (None → "unknown, treat as
    neutral").
    """
    flag_set: set[DataFlag] = set()
    if flags is not None:
        for raw in flags:
            flag = _coerce_flag(raw)
            if flag is not None:
                flag_set.add(flag)

    vec = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for idx, flag in enumerate(_FLAG_FEATURES):
        if flag in flag_set:
            vec[idx] = 1.0

    offset = len(_FLAG_FEATURES)
    vec[offset + 0] = float(n) if n is not None else 0.0
    vec[offset + 1] = float(p) if p is not None else 0.0
    vec[offset + 2] = float(n_modifiers) if n_modifiers is not None else 0.0
    vec[offset + 3] = (
        float(treatment_prevalence) if treatment_prevalence is not None else 0.0
    )
    return vec


class LearnedDispatchRouter:
    """ML router for estimator selection.

    Trains offline from telemetry; serves predictions at inference time.
    The hand-coded rule cascade remains the source of truth — this router
    is meant to be queried alongside it (e.g., to log disagreements or to
    break ties).
    """

    def __init__(self, *, model_path: Path | None = None) -> None:
        self._model: Any = None
        self._classes: list[str] = []
        self._n_train: int = 0
        self._cv_accuracy: float | None = None
        if model_path is not None:
            self.load(Path(model_path))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, ctx: SelectionContext) -> np.ndarray:
        """Encode a :class:`SelectionContext` into the canonical feature vector.

        ``p`` (covariate count) is not on :class:`SelectionContext`, so
        we use ``n_modifiers`` as a stand-in proxy — telemetry-supplied
        rows will carry the real ``p`` directly.
        """
        return _encode_row(
            flags=ctx.flags,
            n=ctx.n,
            p=getattr(ctx, "p", None),
            n_modifiers=ctx.n_modifiers,
            treatment_prevalence=ctx.treatment_prevalence,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, telemetry_rows: list[dict]) -> dict:
        """Train on rows of ``{flags, n, p, n_modifiers, prevalence, chosen_estimator}``.

        Returns a dict with ``n_train``, ``cv_accuracy`` and ``classes``.
        """
        if not telemetry_rows:
            raise ValueError("telemetry_rows is empty; cannot train router")

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score
        except ImportError as exc:  # pragma: no cover - exercised only without sklearn
            raise ImportError(
                "scikit-learn is required for LearnedDispatchRouter; install the "
                "'estimators' extra (`pip install causalrag[estimators]`)."
            ) from exc

        X = np.vstack(
            [
                _encode_row(
                    flags=row.get("flags"),
                    n=row.get("n"),
                    p=row.get("p"),
                    n_modifiers=row.get("n_modifiers"),
                    treatment_prevalence=row.get("prevalence")
                    if "prevalence" in row
                    else row.get("treatment_prevalence"),
                )
                for row in telemetry_rows
            ]
        )
        y = np.asarray(
            [str(row["chosen_estimator"]) for row in telemetry_rows], dtype=object
        )

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError(
                f"telemetry must contain >=2 distinct estimator classes; got {classes.tolist()}"
            )

        model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            random_state=0,
        )
        model.fit(X, y)
        self._model = model
        self._classes = [str(c) for c in model.classes_]
        self._n_train = int(X.shape[0])

        # Cross-validated accuracy — small CV (cv=3) to keep training cheap.
        cv = int(min(3, counts.min()))
        if cv >= 2:
            scores = cross_val_score(
                GradientBoostingClassifier(
                    n_estimators=80,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=0,
                ),
                X,
                y,
                cv=cv,
                scoring="accuracy",
            )
            self._cv_accuracy = float(scores.mean())
        else:
            self._cv_accuracy = None

        return {
            "n_train": self._n_train,
            "cv_accuracy": self._cv_accuracy,
            "classes": list(self._classes),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, ctx: SelectionContext) -> RouterPrediction:
        if self._model is None:
            raise RuntimeError("router not trained or loaded; call train() or load() first")

        x = self.encode(ctx).reshape(1, -1)
        probs = self._model.predict_proba(x)[0]
        order = np.argsort(probs)[::-1]
        top_3 = [(self._classes[i], float(probs[i])) for i in order[:3]]
        recommended = top_3[0][0]
        prob = top_3[0][1]

        contributions = self._feature_contributions(x[0])
        rationale = self._rationale(recommended, top_3, contributions)
        return RouterPrediction(
            recommended_estimator=recommended,
            probability=prob,
            top_3_candidates=top_3,
            feature_contributions=contributions,
            rationale=rationale,
        )

    def _feature_contributions(self, x: np.ndarray) -> dict[str, float]:
        """SHAP-style per-feature contributions.

        We approximate via ``feature_importances_ * feature_value``. This is
        a coarse surrogate — the real SHAP library would give exact additive
        attributions — but it suffices for human-readable rationales and
        avoids the extra dependency. Importances are normalized to sum
        to 1 over present features so the report is easy to interpret.
        """
        importances = getattr(self._model, "feature_importances_", None)
        if importances is None:
            return {}
        raw = importances * x
        denom = float(np.sum(np.abs(raw)))
        if denom == 0.0:
            return {name: 0.0 for name in FEATURE_NAMES}
        return {
            name: float(raw[i] / denom) for i, name in enumerate(FEATURE_NAMES)
        }

    @staticmethod
    def _rationale(
        recommended: str,
        top_3: list[tuple[str, float]],
        contributions: dict[str, float],
    ) -> str:
        if not contributions:
            return f"Recommended {recommended} (prob={top_3[0][1]:.2f})."
        ranked = sorted(
            contributions.items(), key=lambda kv: abs(kv[1]), reverse=True
        )
        drivers = [name for name, value in ranked if abs(value) > 0][:3]
        driver_str = ", ".join(drivers) if drivers else "no dominant feature"
        runners = ", ".join(f"{cid}={p:.2f}" for cid, p in top_3[1:])
        return (
            f"Recommended {recommended} (prob={top_3[0][1]:.2f}); top drivers: "
            f"{driver_str}. Runners-up: {runners or 'n/a'}."
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("nothing to save; router not trained")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "classes": self._classes,
            "n_train": self._n_train,
            "cv_accuracy": self._cv_accuracy,
            "feature_names": FEATURE_NAMES,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)

    def load(self, path: Path) -> None:
        path = Path(path)
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        self._model = payload["model"]
        self._classes = list(payload["classes"])
        self._n_train = int(payload.get("n_train", 0))
        self._cv_accuracy = payload.get("cv_accuracy")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def n_train(self) -> int:
        return self._n_train

    @property
    def cv_accuracy(self) -> float | None:
        return self._cv_accuracy

    @property
    def is_trained(self) -> bool:
        return self._model is not None


def explain(prediction: RouterPrediction) -> str:
    """Render a plain-language explanation of a :class:`RouterPrediction`.

    Surfaces the recommended estimator, its competitors, and the top
    positive / negative feature contributions in a paragraph designed to
    drop straight into an analyst-facing report.
    """
    parts: list[str] = []
    parts.append(
        f"The learned router recommends '{prediction.recommended_estimator}' "
        f"with probability {prediction.probability:.2f}."
    )
    if prediction.top_3_candidates:
        runners = [
            f"{cid} ({p:.2f})"
            for cid, p in prediction.top_3_candidates
            if cid != prediction.recommended_estimator
        ]
        if runners:
            parts.append("Close alternatives: " + ", ".join(runners) + ".")
    if prediction.feature_contributions:
        ranked = sorted(
            prediction.feature_contributions.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        positives = [(n, v) for n, v in ranked if v > 0][:3]
        negatives = [(n, v) for n, v in ranked if v < 0][:3]
        if positives:
            parts.append(
                "Features pushing toward this choice: "
                + ", ".join(f"{n} (+{v:.2f})" for n, v in positives)
                + "."
            )
        if negatives:
            parts.append(
                "Features pushing against alternatives: "
                + ", ".join(f"{n} ({v:.2f})" for n, v in negatives)
                + "."
            )
    if prediction.rationale and prediction.rationale not in " ".join(parts):
        parts.append(prediction.rationale)
    return " ".join(parts)


__all__ = [
    "FEATURE_NAMES",
    "LearnedDispatchRouter",
    "RouterPrediction",
    "explain",
]
