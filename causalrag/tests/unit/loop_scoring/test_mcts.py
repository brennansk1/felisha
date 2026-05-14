"""Unit tests for Sprint 9.1 — MCTS with progressive widening."""

from __future__ import annotations

import math
from typing import Any

import pytest

from causalrag.loop_scoring.mcts import (
    HypothesisMCTS,
    MCTSConfig,
    MCTSNode,
)


# ─────────── Deterministic callbacks ─────────────────────────────────────


def _make_static_proposer(actions: list[str]):
    """Always proposes the same labelled actions, in the same order."""

    def proposer(state: dict[str, Any], n: int) -> list[str]:
        del state
        return list(actions)[:n]

    return proposer


def _make_value_table(table: dict[str, float]):
    """Return a simulator that looks up the realised value by action."""

    def simulate(state: dict[str, Any], action: str) -> float:
        del state
        return table.get(action, 0.0)

    return simulate


# ─────────── MCTSNode mechanics ──────────────────────────────────────────


class TestMCTSNode:
    def test_mean_value_zero_when_unvisited(self) -> None:
        n = MCTSNode(state={"x": 1})
        assert n.mean_value == 0.0
        assert n.is_root

    def test_mean_value_average(self) -> None:
        n = MCTSNode(state={}, visits=4, total_value=2.0)
        assert n.mean_value == pytest.approx(0.5)


# ─────────── Config validation ───────────────────────────────────────────


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_iterations": -1},
            {"max_depth": 0},
            {"exploration_c": -0.1},
            {"progressive_widening_k": 0.0},
            {"progressive_widening_alpha": 0.0},
            {"progressive_widening_alpha": 1.5},
            {"discount": 0.0},
            {"discount": 1.1},
        ],
    )
    def test_invalid_config_rejected(self, kwargs: dict[str, Any]) -> None:
        cfg = MCTSConfig(**kwargs)
        with pytest.raises(ValueError):
            HypothesisMCTS(
                root_state={},
                propose_actions=_make_static_proposer(["a"]),
                simulate_value=_make_value_table({"a": 1.0}),
                config=cfg,
            )

    def test_non_callable_proposer_rejected(self) -> None:
        with pytest.raises(TypeError):
            HypothesisMCTS(
                root_state={},
                propose_actions="not-callable",  # type: ignore[arg-type]
                simulate_value=_make_value_table({}),
            )

    def test_non_callable_simulator_rejected(self) -> None:
        with pytest.raises(TypeError):
            HypothesisMCTS(
                root_state={},
                propose_actions=_make_static_proposer(["a"]),
                simulate_value="not-callable",  # type: ignore[arg-type]
            )


# ─────────── Search behaviour ────────────────────────────────────────────


class TestSearchBestAction:
    def test_best_action_maximises_long_run_value(self) -> None:
        """With a clear value ordering, the search should pick the
        highest-value root action under enough iterations."""
        proposer = _make_static_proposer(["good", "ok", "bad"])
        simulator = _make_value_table({"good": 1.0, "ok": 0.5, "bad": 0.0})
        mcts = HypothesisMCTS(
            root_state={"chain": "c0", "depth": 0},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(
                max_iterations=200,
                max_depth=2,
                exploration_c=0.5,  # exploit-leaning so the test is sharp
            ),
        )
        best, root = mcts.search()
        assert best == "good"
        # Root visit count equals max_iterations exactly.
        assert root.visits == 200
        # Children appear in insertion order.
        labels = [c.last_action for c in root.children]
        assert labels == ["good", "ok", "bad"]
        # The winner is most visited.
        good_child = next(c for c in root.children if c.last_action == "good")
        for other in root.children:
            if other is good_child:
                continue
            assert good_child.visits >= other.visits

    def test_no_actions_raises(self) -> None:
        """If the proposer offers nothing, ``search`` raises."""
        proposer = _make_static_proposer([])
        simulator = _make_value_table({})
        mcts = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(max_iterations=5),
        )
        with pytest.raises(RuntimeError):
            mcts.search()

    def test_zero_iterations_still_raises_with_no_children(self) -> None:
        proposer = _make_static_proposer(["a"])
        simulator = _make_value_table({"a": 1.0})
        mcts = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(max_iterations=0),
        )
        # No iterations run → no children expanded at root → RuntimeError.
        with pytest.raises(RuntimeError):
            mcts.search()

    def test_non_finite_simulator_value_treated_as_zero(self) -> None:
        """A NaN / inf simulator return must not poison ancestor means."""
        proposer = _make_static_proposer(["a", "b"])
        simulator = _make_value_table({"a": float("nan"), "b": 0.5})
        mcts = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(max_iterations=50, max_depth=1),
        )
        best, root = mcts.search()
        # b has the only finite positive value.
        assert best == "b"
        for child in root.children:
            assert math.isfinite(child.total_value)
            assert math.isfinite(child.mean_value)


# ─────────── Progressive widening cap ────────────────────────────────────


class TestProgressiveWidening:
    def test_child_count_within_pw_cap(self) -> None:
        """At every node, ``len(children) ≤ ceil(k · n_visits^α)``
        (or 1, the floor)."""
        # Many candidate labels so the proposer never runs dry.
        labels = [f"h{i:02d}" for i in range(20)]
        proposer = _make_static_proposer(labels)
        # Slight value gradient — keeps UCB1 from collapsing to one arm.
        simulator = _make_value_table({lab: 1.0 / (i + 1) for i, lab in enumerate(labels)})

        k = 1.0
        alpha = 0.5
        mcts = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(
                max_iterations=100,
                max_depth=3,
                progressive_widening_k=k,
                progressive_widening_alpha=alpha,
            ),
        )
        mcts.search()

        def cap(n: int) -> int:
            return max(1, math.ceil(k * (n**alpha))) if n > 0 else 1

        def walk(node: MCTSNode) -> None:
            assert len(node.children) <= cap(node.visits), (
                f"node visits={node.visits} has {len(node.children)} children, "
                f"cap was {cap(node.visits)}"
            )
            for child in node.children:
                walk(child)

        walk(mcts.root)

    def test_smaller_alpha_gives_narrower_tree(self) -> None:
        """Halving α should shrink the root's branching factor under
        the same iteration budget."""
        labels = [f"h{i:02d}" for i in range(20)]
        proposer = _make_static_proposer(labels)
        simulator = _make_value_table({lab: 0.0 for lab in labels})

        wide = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(
                max_iterations=80,
                max_depth=2,
                progressive_widening_k=1.0,
                progressive_widening_alpha=0.9,
            ),
        )
        narrow = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(
                max_iterations=80,
                max_depth=2,
                progressive_widening_k=1.0,
                progressive_widening_alpha=0.3,
            ),
        )
        wide.search()
        narrow.search()

        assert len(wide.root.children) > len(narrow.root.children)

    def test_pw_floor_allows_one_child_at_zero_visits(self) -> None:
        """The brand-new root must be allowed at least one child even
        though ``ceil(k · 0^α) == 0``."""
        proposer = _make_static_proposer(["only"])
        simulator = _make_value_table({"only": 0.0})
        mcts = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(max_iterations=1, max_depth=1),
        )
        best, root = mcts.search()
        assert best == "only"
        assert len(root.children) == 1


# ─────────── UCB1 behaviour ──────────────────────────────────────────────


class TestUCB1:
    def test_unvisited_children_explored_first(self) -> None:
        """After two iterations with a two-action proposer, both root
        children should have been visited at least once."""
        proposer = _make_static_proposer(["a", "b"])
        simulator = _make_value_table({"a": 1.0, "b": 1.0})
        mcts = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(
                max_iterations=10,
                max_depth=1,
                progressive_widening_k=2.0,  # cap allows both at root
                progressive_widening_alpha=0.5,
            ),
        )
        mcts.search()
        labels = {c.last_action: c.visits for c in mcts.root.children}
        assert set(labels) == {"a", "b"}
        assert all(v >= 1 for v in labels.values())


# ─────────── Discounting / depth ─────────────────────────────────────────


class TestDiscountAndDepth:
    def test_max_depth_respected(self) -> None:
        """No node should sit deeper than ``max_depth`` from the root."""
        labels = [f"h{i}" for i in range(10)]
        proposer = _make_static_proposer(labels)
        simulator = _make_value_table({lab: 0.1 for lab in labels})
        mcts = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(max_iterations=60, max_depth=3),
        )
        mcts.search()

        def depth_of(node: MCTSNode) -> int:
            d = 0
            cur = node
            while cur.parent is not None:
                d += 1
                cur = cur.parent
            return d

        def walk(node: MCTSNode) -> None:
            assert depth_of(node) <= 3
            for child in node.children:
                walk(child)

        walk(mcts.root)

    def test_discount_shrinks_deep_contribution(self) -> None:
        """With γ < 1, a value realised k steps below the root
        contributes γ^k to the root's total. Verify the back-up math."""
        proposer = _make_static_proposer(["x"])
        simulator = _make_value_table({"x": 2.0})
        cfg = MCTSConfig(
            max_iterations=1,
            max_depth=1,
            discount=0.5,
        )
        mcts = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
            config=cfg,
        )
        _, root = mcts.search()
        # Single iteration: child leaf gets value 2.0; root receives
        # γ · 2.0 = 1.0.
        child = root.children[0]
        assert child.total_value == pytest.approx(2.0)
        assert root.total_value == pytest.approx(1.0)
        assert root.visits == 1
        assert child.visits == 1


# ─────────── Renderable tree ─────────────────────────────────────────────


class TestRenderTree:
    def test_render_tree_returns_string_with_root_and_children(self) -> None:
        proposer = _make_static_proposer(["alpha", "beta"])
        simulator = _make_value_table({"alpha": 1.0, "beta": 0.2})
        mcts = HypothesisMCTS(
            root_state={"chain_id": "demo"},
            propose_actions=proposer,
            simulate_value=simulator,
            config=MCTSConfig(
                max_iterations=20,
                max_depth=2,
                progressive_widening_k=2.0,
            ),
        )
        mcts.search()
        rendered = mcts.render_tree()
        assert isinstance(rendered, str)
        assert "<root>" in rendered
        assert "alpha" in rendered
        assert "visits=" in rendered
        assert "mean=" in rendered
        # Indented child line — at least one row starts with two spaces
        # of indent.
        assert any(line.startswith("  -") for line in rendered.splitlines())

    def test_render_tree_no_iterations(self) -> None:
        """Rendering an unsearched tree should still work — just the
        root, no children."""
        proposer = _make_static_proposer(["a"])
        simulator = _make_value_table({"a": 0.0})
        mcts = HypothesisMCTS(
            root_state={},
            propose_actions=proposer,
            simulate_value=simulator,
        )
        # Don't call .search() — render the bare tree.
        rendered = mcts.render_tree()
        assert "<root>" in rendered
        assert "visits=0" in rendered
