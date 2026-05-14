"""Local RAG over prior CausalRAG runs — Sprint 3.5 / case-based reasoning.

Maintains a local sentence-embedding (or TF-IDF fallback) index over
prior ``(flags, hypothesis, estimator, verdict)`` tuples so the planner
can be conditioned on what worked / didn't work on similar past
problems.

Design notes
------------
* **Local only.** No remote embedding service, no external vector DB.
* **Two backends.**
    - ``sentence-transformers`` (preferred when installed) → MiniLM
      cosine similarity.
    - ``tfidf`` fallback via ``sklearn`` — always available because
      sklearn is already a hard dep.
* **Persistence.** ``save`` / ``load`` round-trip the case bank through
  JSONL at ``~/.causalrag/history/cases.jsonl`` (or any user-supplied
  path). Embeddings are recomputed on load; we persist the *cases*, not
  vectors, since the embedding model can change between sessions.
* **Read-only by the master loop.** Nothing here mutates protocol or
  walk objects; the planner pulls few-shot examples out of
  ``few_shot_examples`` and the caller decides how to wire them into the
  prompt.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

__all__ = ["HistoryCase", "HistoryRAG", "default_cache_path"]


Backend = Literal["sentence-transformers", "tfidf"]


def default_cache_path() -> Path:
    """Canonical on-disk location for the persistent case bank."""

    return Path.home() / ".causalrag" / "history" / "cases.jsonl"


# ─────────── HistoryCase ──────────────────────────────────────────────────


@dataclass
class HistoryCase:
    """One canonicalised past experience.

    ``text`` is the natural-language projection that gets embedded; the
    structured fields are kept for filtering / rendering. ``metadata``
    holds anything extra the caller wants to round-trip (point
    estimate, CI, dataset name, timestamps, ...).
    """

    case_id: str
    flags: list[str]  # sorted DataFlag values
    treatment: str
    outcome: str
    estimand_class: str
    estimator_id: str
    sensitivity_verdict: str
    failure_reason: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ IO

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HistoryCase:
        # Defensive: keep unknown keys out of __init__, drop into metadata.
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        extras = {k: v for k, v in d.items() if k not in known}
        meta = dict(kwargs.get("metadata") or {})
        meta.update(extras)
        kwargs["metadata"] = meta
        # Normalise types.
        kwargs.setdefault("flags", [])
        kwargs["flags"] = sorted(str(f) for f in kwargs["flags"])
        kwargs.setdefault("failure_reason", None)
        return cls(**kwargs)


# ─────────── canonical text rendering ─────────────────────────────────────


def _flag_phrase(flags: Iterable[str]) -> str:
    fs = sorted({str(f) for f in flags})
    if not fs:
        return "no special data flags"
    return ", ".join(fs)


def _canonicalise(
    *,
    flags: Iterable[str],
    treatment: str,
    outcome: str,
    estimand_class: str,
    estimator_id: str,
    sensitivity_verdict: str,
    failure_reason: str | None,
    extra: str | None = None,
) -> str:
    """Render a HistoryCase to the natural-language string we embed."""

    parts = [
        f"Treatment: {treatment}.",
        f"Outcome: {outcome}.",
        f"Estimand class: {estimand_class}.",
        f"Estimator: {estimator_id}.",
        f"Sensitivity verdict: {sensitivity_verdict}.",
        f"Data flags: {_flag_phrase(flags)}.",
    ]
    if failure_reason:
        parts.append(f"Failure reason: {failure_reason}.")
    if extra:
        parts.append(extra)
    return " ".join(parts)


# ─────────── HistoryRAG ───────────────────────────────────────────────────


class HistoryRAG:
    """Local sentence-embedding RAG over prior runs.

    Builds an index from a list of :class:`HistoryCase` entries. Queries
    return the top-k most similar cases by cosine similarity. Designed
    to be persisted locally so multiple sessions accumulate knowledge.
    """

    def __init__(
        self,
        cases: list[HistoryCase] | None = None,
        *,
        backend: Backend = "tfidf",
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
    ) -> None:
        self._cases: list[HistoryCase] = []
        self._embeddings: np.ndarray | None = None  # (N, D), L2-normalised
        self._model_name = model_name
        self._cache_dir = Path(cache_dir) if cache_dir else None

        # Resolve backend with graceful fallback.
        if backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer  # noqa: F401

                self._backend: Backend = "sentence-transformers"
                self._st_model = self._load_st_model(model_name)
                self._tfidf = None
            except Exception:
                self._backend = "tfidf"
                self._st_model = None
                self._tfidf = None
        else:
            self._backend = "tfidf"
            self._st_model = None
            self._tfidf = None

        if cases:
            for c in cases:
                self._cases.append(c)
            self._rebuild_index()

    # --------------------------------------------------------------- props

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def cases(self) -> list[HistoryCase]:
        return list(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    # ----------------------------------------------------- backend helpers

    def _load_st_model(self, name: str) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an (N, D) float32 matrix of L2-normalised embeddings."""

        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self._backend == "sentence-transformers" and self._st_model is not None:
            vecs = np.asarray(
                self._st_model.encode(texts, normalize_embeddings=True),
                dtype=np.float32,
            )
            return vecs

        # TF-IDF: fit on the corpus the first time, then transform.
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
            )
            mat = self._tfidf.fit_transform(texts).toarray().astype(np.float32)
        else:
            mat = self._tfidf.transform(texts).toarray().astype(np.float32)

        # L2-normalise rows so dot product == cosine similarity.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def _rebuild_index(self) -> None:
        """Recompute embeddings for the full case bank."""

        if not self._cases:
            self._embeddings = None
            # Reset TF-IDF fit so the next add seeds a fresh vocabulary.
            self._tfidf = None
            return

        texts = [c.text for c in self._cases]
        # For TF-IDF we want the vocab fit over the full corpus.
        self._tfidf = None
        self._embeddings = self._embed_texts(texts)

    def _embed_query(self, query: str) -> np.ndarray:
        if self._backend == "sentence-transformers" and self._st_model is not None:
            v = np.asarray(
                self._st_model.encode([query], normalize_embeddings=True),
                dtype=np.float32,
            )
            return v
        if self._tfidf is None:
            # No corpus yet — return a zero vector of dim 1; callers will
            # short-circuit on empty index anyway.
            return np.zeros((1, 1), dtype=np.float32)
        mat = self._tfidf.transform([query]).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    # ------------------------------------------------------------- mutate

    def add_case(self, case: HistoryCase) -> None:
        """Append a case and refresh the index.

        We rebuild rather than incrementally append because TF-IDF needs
        the global vocabulary to stay coherent and sentence-transformer
        embeddings are cheap to recompute at this scale.
        """

        self._cases.append(case)
        self._rebuild_index()

    def add_from_walk(
        self,
        walk: Any,
        flags: set[Any] | Iterable[Any],
        dataset_label: str,
    ) -> HistoryCase:
        """Build a :class:`HistoryCase` from a completed RoadmapWalk.

        Pulls (treatment, outcome) from the walk's statistical estimand
        when available, falls back to the q4 observed-data spec, and
        defaults to ``"?"`` so a partial walk still produces a usable
        case rather than crashing.
        """

        # Flatten flag values (enum.value attribute or str).
        flag_values: list[str] = []
        for f in flags:
            v = getattr(f, "value", f)
            flag_values.append(str(v))
        flag_values = sorted(set(flag_values))

        treatment = "?"
        outcome = "?"
        estimand_class = "?"
        estimator_id = "—"
        point_estimate: float | None = None

        q6 = getattr(walk, "q6_statistical_estimand", None)
        if q6 is not None:
            treatment = getattr(q6, "treatment", treatment) or treatment
            outcome = getattr(q6, "outcome", outcome) or outcome

        q4 = getattr(walk, "q4_observed_data_spec", None) or {}
        if isinstance(q4, dict):
            treatment = str(q4.get("treatment", treatment) or treatment)
            outcome = str(q4.get("outcome", outcome) or outcome)

        q3 = getattr(walk, "q3_estimand", None)
        if q3 is not None:
            cls = getattr(q3, "estimand_class", None) or getattr(q3, "klass", None)
            if cls is not None:
                estimand_class = str(getattr(cls, "value", cls))

        ests = getattr(walk, "q7_estimates", ()) or ()
        if ests:
            last = ests[-1]
            estimator_id = getattr(last, "estimator_id", estimator_id) or estimator_id
            point_estimate = getattr(last, "point_estimate", None)
            if not estimand_class or estimand_class == "?":
                estimand_class = (
                    getattr(last, "estimand_class", estimand_class) or estimand_class
                )

        verdict = getattr(walk, "sensitivity_verdict", None) or "unknown"
        failure_reason = getattr(walk, "failure_reason", None)

        text = _canonicalise(
            flags=flag_values,
            treatment=treatment,
            outcome=outcome,
            estimand_class=estimand_class,
            estimator_id=estimator_id,
            sensitivity_verdict=verdict,
            failure_reason=failure_reason,
            extra=f"Dataset: {dataset_label}." if dataset_label else None,
        )

        case = HistoryCase(
            case_id=getattr(walk, "hypothesis_id", None) or f"case-{uuid.uuid4().hex[:8]}",
            flags=flag_values,
            treatment=str(treatment),
            outcome=str(outcome),
            estimand_class=str(estimand_class),
            estimator_id=str(estimator_id),
            sensitivity_verdict=str(verdict),
            failure_reason=str(failure_reason) if failure_reason else None,
            text=text,
            metadata={
                "dataset_label": dataset_label,
                "point_estimate": point_estimate,
                "chain_id": getattr(walk, "chain_id", None),
                "parent_id": getattr(walk, "parent_id", None),
            },
        )
        self.add_case(case)
        return case

    # -------------------------------------------------------------- query

    def search(self, query_text: str, *, top_k: int = 5) -> list[HistoryCase]:
        """Return up to ``top_k`` cases ranked by cosine similarity."""

        if not self._cases or self._embeddings is None or top_k <= 0:
            return []
        q = self._embed_query(query_text)
        if q.shape[1] != self._embeddings.shape[1]:
            # Shape mismatch — happens if the index was built with a
            # different vocabulary. Rebuild and retry once.
            self._rebuild_index()
            q = self._embed_query(query_text)
            if q.shape[1] != self._embeddings.shape[1]:
                return []
        sims = (self._embeddings @ q.T).ravel()
        k = min(top_k, len(self._cases))
        # argsort ascending → take last k → reverse.
        idx = np.argsort(sims)[-k:][::-1]
        return [self._cases[int(i)] for i in idx]

    def few_shot_examples(
        self,
        *,
        current_flags: set[Any] | Iterable[Any],
        current_treatment_hint: str | None = None,
        current_outcome_hint: str | None = None,
        top_k: int = 3,
    ) -> str:
        """Render top-k retrieved cases as a markdown few-shot block."""

        flag_values = sorted(
            {str(getattr(f, "value", f)) for f in current_flags}
        )
        query = _canonicalise(
            flags=flag_values,
            treatment=current_treatment_hint or "?",
            outcome=current_outcome_hint or "?",
            estimand_class="?",
            estimator_id="?",
            sensitivity_verdict="?",
            failure_reason=None,
        )
        hits = self.search(query, top_k=top_k)
        if not hits:
            return ""

        lines: list[str] = [
            "## Prior-run case bank (retrieved by similarity)",
            "",
            (
                "These are summaries of past CausalRAG runs whose data "
                "flags / treatment / outcome resemble the current "
                "problem. Use them as soft priors — *don't* copy "
                "verdicts blindly."
            ),
            "",
        ]
        for i, c in enumerate(hits, start=1):
            failure = f" — failed: {c.failure_reason}" if c.failure_reason else ""
            lines.extend(
                [
                    f"### Case {i}: `{c.case_id}`",
                    f"- **Treatment → Outcome:** `{c.treatment}` → `{c.outcome}`",
                    f"- **Estimand class:** `{c.estimand_class}`",
                    f"- **Estimator used:** `{c.estimator_id}`",
                    f"- **Sensitivity verdict:** `{c.sensitivity_verdict}`{failure}",
                    f"- **Flags:** {', '.join(c.flags) if c.flags else '(none)'}",
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    # ---------------------------------------------------------- persistence

    def save(self, path: Path) -> None:
        """Persist the case bank as JSONL.

        Embeddings are *not* persisted — they're recomputed on load so
        the file stays portable across embedding-model upgrades.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for c in self._cases:
                fh.write(json.dumps(c.to_dict(), sort_keys=True))
                fh.write("\n")

    def load(self, path: Path) -> None:
        """Replace the in-memory case bank with the contents of ``path``."""

        path = Path(path)
        cases: list[HistoryCase] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    cases.append(HistoryCase.from_dict(json.loads(line)))
        self._cases = cases
        self._rebuild_index()
