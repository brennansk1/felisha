"""Estimator registry + flag-aware dispatch (PDD §15.2).

The registry is a process-wide singleton. Estimator modules register themselves
at import time via :func:`register`. The query interface
:meth:`Registry.candidates_for` returns the methods admissible under a given
``(estimand, required, excluded)`` triple.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from causalrag.core.flags import DataFlag


@dataclass(frozen=True)
class EstimatorEntry:
    id: str
    factory: type
    backend: str  # "python" | "r"
    supported_estimands: frozenset[str]
    required_flags: frozenset[DataFlag]
    excluded_flags: frozenset[DataFlag]
    min_sample_size: int
    produces_cate: bool
    produces_full_counterfactual: bool
    propensity_required: bool


@dataclass
class Registry:
    _by_id: dict[str, EstimatorEntry] = field(default_factory=dict)

    def register(self, entry: EstimatorEntry) -> None:
        if entry.id in self._by_id:
            raise ValueError(f"Estimator id already registered: {entry.id}")
        self._by_id[entry.id] = entry

    def unregister(self, estimator_id: str) -> None:
        self._by_id.pop(estimator_id, None)

    def get(self, estimator_id: str) -> EstimatorEntry:
        return self._by_id[estimator_id]

    def all(self) -> tuple[EstimatorEntry, ...]:
        return tuple(self._by_id.values())

    def candidates_for(
        self,
        estimand: str,
        required: Iterable[DataFlag] = (),
        excluded: Iterable[DataFlag] = (),
        backends: Iterable[str] = ("python", "r"),
        n: int | None = None,
    ) -> tuple[EstimatorEntry, ...]:
        """Return estimators that:
        - declare ``estimand`` as supported,
        - require a subset of the provided ``required`` flags (so adding more
          flags can only *expand* a method's qualification — required is the
          floor),
        - declare no excluded flag that is present in ``required``,
        - support one of the requested ``backends``,
        - and meet the optional ``min_sample_size`` if ``n`` is given.

        The query semantics follow PDD §15.2: required flags describe the
        *situation*; an estimator qualifies if every one of its declared
        requirements is satisfied by the situation, and none of its declared
        exclusions is present.
        """
        required_set: frozenset[DataFlag] = frozenset(required)
        excluded_set: frozenset[DataFlag] = frozenset(excluded)
        backends_set = frozenset(backends)

        out: list[EstimatorEntry] = []
        for entry in self._by_id.values():
            if entry.backend not in backends_set:
                continue
            if estimand not in entry.supported_estimands:
                continue
            if not entry.required_flags.issubset(required_set):
                continue
            if entry.excluded_flags & required_set:
                continue
            if excluded_set & entry.required_flags:
                continue
            if n is not None and n < entry.min_sample_size:
                continue
            out.append(entry)
        return tuple(out)


_registry = Registry()


def register(entry: EstimatorEntry) -> None:
    _registry.register(entry)


def get_registry() -> Registry:
    return _registry


def reset_registry() -> None:
    """Reset the process-wide registry. Test-only."""
    global _registry
    _registry = Registry()
