"""InterferenceGraph — unit adjacency for network-interference settings.

The pipeline's default analysis assumes SUTVA (no interference between
units). When SUTVA fails — e.g., vaccinated friends protect unvaccinated
ones, a price cut at one store cannibalises sales at neighbouring stores,
a behavioural-economics nudge spills from treated household members to
untreated ones — the analyst supplies an ``InterferenceGraph`` describing
*which* units can interfere with which.

Two interference assumptions are encoded:

- ``'partial'`` (Hudgens-Halloran 2008): interference is contained inside
  pre-specified clusters (households, classrooms, geographic blocks).
  Units in different clusters are SUTVA w.r.t. each other.
- ``'general'`` (Sävje-Aronow-Hudgens 2021): no structural assumption.
  An exposure mapping summarises spillover; estimands target the
  *expected* average treatment effect over the realised assignment
  distribution rather than a structural mediator.

The dataclass is intentionally minimal — it just stores adjacency and
clusters. Estimators in :mod:`causalrag.estimators.python.interference`
consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class InterferenceGraph:
    """Unit adjacency graph for network-interference settings.

    Parameters
    ----------
    n_units:
        Number of rows in the dataset the graph applies to. Unit indices
        are 0..n_units-1 and must align with the row order of the data
        the estimator sees.
    adjacency:
        Mapping ``unit -> frozenset of neighbour unit indices``. The
        relation is treated as the spillover-can-flow relation, so it
        should normally be symmetric; the constructor does not enforce
        symmetry (so the user can model directed spillovers if desired).
    clusters:
        Optional ``unit -> cluster_id`` mapping. Required when
        ``interference_kind == 'partial'``. Units in different clusters
        are assumed not to interfere.
    interference_kind:
        ``'partial'`` for Hudgens-Halloran-style cluster interference,
        ``'general'`` for Sävje-Aronow-Hudgens.
    """

    n_units: int
    adjacency: dict[int, frozenset[int]] = field(default_factory=dict)
    clusters: dict[int, int] | None = None
    interference_kind: Literal["partial", "general"] = "general"

    def __post_init__(self) -> None:
        if self.n_units < 0:
            raise ValueError(f"n_units must be ≥ 0; got {self.n_units}")
        if self.interference_kind not in ("partial", "general"):
            raise ValueError(
                f"interference_kind must be 'partial' or 'general'; got "
                f"{self.interference_kind!r}"
            )
        # Default each unit to an empty neighbour set so downstream code
        # can call neighbours(i) for any valid i.
        for i in range(self.n_units):
            self.adjacency.setdefault(i, frozenset())
        for u, ns in self.adjacency.items():
            if u < 0 or u >= self.n_units:
                raise ValueError(f"adjacency key {u} outside [0, {self.n_units})")
            for v in ns:
                if v < 0 or v >= self.n_units:
                    raise ValueError(f"neighbour {v} of unit {u} outside [0, {self.n_units})")
        if self.interference_kind == "partial" and self.clusters is None:
            raise ValueError(
                "interference_kind='partial' requires a clusters mapping "
                "(unit -> cluster_id)."
            )
        if self.clusters is not None:
            for u in self.clusters:
                if u < 0 or u >= self.n_units:
                    raise ValueError(f"cluster key {u} outside [0, {self.n_units})")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def neighbours(self, unit: int) -> frozenset[int]:
        """Return the unit indices that can receive spillover from ``unit``."""
        if unit < 0 or unit >= self.n_units:
            raise IndexError(f"unit {unit} outside [0, {self.n_units})")
        return self.adjacency.get(unit, frozenset())

    def degree(self, unit: int) -> int:
        return len(self.neighbours(unit))

    def cluster_of(self, unit: int) -> int | None:
        if self.clusters is None:
            return None
        return self.clusters.get(unit)

    def exposure_at_unit(self, unit: int, treatments: np.ndarray) -> float:
        """Fraction of ``unit``'s neighbours that are treated.

        Returns 0.0 for isolated units (no neighbours) — consumers
        should distinguish "no neighbours" from "neighbours all
        untreated" via :meth:`degree` if necessary.
        """
        ns = self.neighbours(unit)
        if not ns:
            return 0.0
        t = np.asarray(treatments)
        if t.shape[0] != self.n_units:
            raise ValueError(
                f"treatments length {t.shape[0]} != n_units {self.n_units}"
            )
        idx = np.fromiter(ns, dtype=int)
        return float(np.mean(t[idx]))

    def exposure_vector(self, treatments: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`exposure_at_unit` over all units."""
        t = np.asarray(treatments)
        if t.shape[0] != self.n_units:
            raise ValueError(
                f"treatments length {t.shape[0]} != n_units {self.n_units}"
            )
        out = np.zeros(self.n_units, dtype=float)
        for i in range(self.n_units):
            ns = self.adjacency.get(i, frozenset())
            if ns:
                idx = np.fromiter(ns, dtype=int)
                out[i] = float(np.mean(t[idx]))
        return out

    def cluster_members(self) -> dict[int, list[int]]:
        """Inverse of ``self.clusters``: cluster_id -> sorted list of units."""
        if self.clusters is None:
            return {}
        out: dict[int, list[int]] = {}
        for u, c in self.clusters.items():
            out.setdefault(c, []).append(u)
        for c in out:
            out[c].sort()
        return out

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_edge_list(
        cls,
        n_units: int,
        edges: list[tuple[int, int]],
        *,
        clusters: dict[int, int] | None = None,
        interference_kind: Literal["partial", "general"] = "general",
        symmetric: bool = True,
    ) -> InterferenceGraph:
        """Build from a list of (u, v) edges.

        Self-loops are silently dropped. When ``symmetric`` is True
        (default) each (u, v) edge also records (v, u).
        """
        adj: dict[int, set[int]] = {i: set() for i in range(n_units)}
        for u, v in edges:
            if u == v:
                continue
            if u < 0 or u >= n_units or v < 0 or v >= n_units:
                raise ValueError(
                    f"edge ({u}, {v}) references unit outside [0, {n_units})"
                )
            adj[u].add(v)
            if symmetric:
                adj[v].add(u)
        frozen = {u: frozenset(ns) for u, ns in adj.items()}
        return cls(
            n_units=n_units,
            adjacency=frozen,
            clusters=clusters,
            interference_kind=interference_kind,
        )

    @classmethod
    def from_distance_matrix(
        cls,
        dist: np.ndarray,
        threshold: float,
        *,
        clusters: dict[int, int] | None = None,
        interference_kind: Literal["partial", "general"] = "general",
    ) -> InterferenceGraph:
        """Build a graph by thresholding a pairwise distance matrix.

        Two units i, j are adjacent iff ``dist[i, j] <= threshold`` and
        ``i != j``. ``dist`` is expected square; symmetry is not
        enforced (the user can pass an asymmetric "i affects j" cost).
        """
        dist = np.asarray(dist, dtype=float)
        if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
            raise ValueError(f"dist must be a square matrix; got shape {dist.shape}")
        n = dist.shape[0]
        adj: dict[int, frozenset[int]] = {}
        for i in range(n):
            row = dist[i]
            mask = (row <= threshold) & (np.arange(n) != i)
            adj[i] = frozenset(int(j) for j in np.where(mask)[0])
        return cls(
            n_units=n,
            adjacency=adj,
            clusters=clusters,
            interference_kind=interference_kind,
        )

    @classmethod
    def from_clusters(
        cls,
        clusters: dict[int, int],
        n_units: int | None = None,
    ) -> InterferenceGraph:
        """Build a partial-interference graph: every unit is adjacent
        to every other unit in its cluster.
        """
        if n_units is None:
            n_units = (max(clusters) + 1) if clusters else 0
        members: dict[int, list[int]] = {}
        for u, c in clusters.items():
            members.setdefault(c, []).append(u)
        adj: dict[int, frozenset[int]] = {}
        for u, c in clusters.items():
            adj[u] = frozenset(v for v in members[c] if v != u)
        for i in range(n_units):
            adj.setdefault(i, frozenset())
        return cls(
            n_units=n_units,
            adjacency=adj,
            clusters=dict(clusters),
            interference_kind="partial",
        )
