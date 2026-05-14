"""Tests for InterferenceGraph (PDD §33 / Sprint 6.5.4)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrag.core.interference import InterferenceGraph


def test_from_edge_list_roundtrip_symmetric() -> None:
    g = InterferenceGraph.from_edge_list(
        n_units=4, edges=[(0, 1), (1, 2), (2, 3)]
    )
    assert g.n_units == 4
    assert g.neighbours(0) == frozenset({1})
    assert g.neighbours(1) == frozenset({0, 2})
    assert g.neighbours(2) == frozenset({1, 3})
    assert g.neighbours(3) == frozenset({2})
    # Isolated units default to empty.
    g2 = InterferenceGraph.from_edge_list(n_units=5, edges=[])
    for i in range(5):
        assert g2.neighbours(i) == frozenset()


def test_from_edge_list_directed() -> None:
    g = InterferenceGraph.from_edge_list(
        n_units=3, edges=[(0, 1), (0, 2)], symmetric=False
    )
    assert g.neighbours(0) == frozenset({1, 2})
    assert g.neighbours(1) == frozenset()
    assert g.neighbours(2) == frozenset()


def test_from_edge_list_drops_self_loops() -> None:
    g = InterferenceGraph.from_edge_list(n_units=3, edges=[(0, 0), (0, 1)])
    assert g.neighbours(0) == frozenset({1})


def test_from_edge_list_bad_index_raises() -> None:
    with pytest.raises(ValueError):
        InterferenceGraph.from_edge_list(n_units=3, edges=[(0, 5)])


def test_from_distance_matrix_threshold() -> None:
    # 4 units on a line at x = 0, 1, 2, 5. Threshold = 1.5 → near-neighbours.
    pts = np.array([0.0, 1.0, 2.0, 5.0])
    dist = np.abs(pts[:, None] - pts[None, :])
    g = InterferenceGraph.from_distance_matrix(dist, threshold=1.5)
    assert g.neighbours(0) == frozenset({1})
    assert g.neighbours(1) == frozenset({0, 2})
    assert g.neighbours(2) == frozenset({1})
    assert g.neighbours(3) == frozenset()


def test_from_distance_matrix_non_square_raises() -> None:
    with pytest.raises(ValueError):
        InterferenceGraph.from_distance_matrix(np.zeros((3, 4)), threshold=1.0)


def test_exposure_at_unit_chain() -> None:
    """5-unit chain 0—1—2—3—4 with treatment vector [1, 0, 1, 0, 1].

    Expected neighbour-treatment fractions:
      unit 0 — neighbour {1} → 0/1 = 0.0
      unit 1 — neighbours {0, 2} → 2/2 = 1.0
      unit 2 — neighbours {1, 3} → 0/2 = 0.0
      unit 3 — neighbours {2, 4} → 2/2 = 1.0
      unit 4 — neighbour {3} → 0/1 = 0.0
    """
    g = InterferenceGraph.from_edge_list(
        n_units=5, edges=[(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    t = np.array([1, 0, 1, 0, 1])
    assert g.exposure_at_unit(0, t) == pytest.approx(0.0)
    assert g.exposure_at_unit(1, t) == pytest.approx(1.0)
    assert g.exposure_at_unit(2, t) == pytest.approx(0.0)
    assert g.exposure_at_unit(3, t) == pytest.approx(1.0)
    assert g.exposure_at_unit(4, t) == pytest.approx(0.0)
    # Vectorised form matches.
    np.testing.assert_allclose(
        g.exposure_vector(t), np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    )


def test_exposure_isolated_returns_zero() -> None:
    g = InterferenceGraph.from_edge_list(n_units=3, edges=[])
    t = np.array([1, 1, 1])
    for i in range(3):
        assert g.exposure_at_unit(i, t) == 0.0


def test_exposure_bad_treatment_length() -> None:
    g = InterferenceGraph.from_edge_list(n_units=3, edges=[(0, 1)])
    with pytest.raises(ValueError):
        g.exposure_at_unit(0, np.array([1, 0]))


def test_partial_kind_requires_clusters() -> None:
    with pytest.raises(ValueError, match="clusters"):
        InterferenceGraph(n_units=4, interference_kind="partial")


def test_cluster_of_and_members() -> None:
    g = InterferenceGraph.from_clusters({0: 7, 1: 7, 2: 9, 3: 9})
    assert g.cluster_of(0) == 7
    assert g.cluster_of(2) == 9
    members = g.cluster_members()
    assert members[7] == [0, 1]
    assert members[9] == [2, 3]
    # Default adjacency = same-cluster mates.
    assert g.neighbours(0) == frozenset({1})
    assert g.neighbours(2) == frozenset({3})


def test_bad_interference_kind_raises() -> None:
    with pytest.raises(ValueError):
        InterferenceGraph(n_units=2, interference_kind="weird")  # type: ignore[arg-type]


def test_neighbours_out_of_bounds() -> None:
    g = InterferenceGraph.from_edge_list(n_units=3, edges=[])
    with pytest.raises(IndexError):
        g.neighbours(5)


def test_negative_n_units_raises() -> None:
    with pytest.raises(ValueError):
        InterferenceGraph(n_units=-1)
