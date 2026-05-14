"""MCTS with progressive widening over the hypothesis tree.

Implements Sprint 9.1 of PDD §33: a Monte-Carlo Tree Search rollout
over the master-loop hypothesis space. Where the EIG / bandit modules
score a *single* next decision, MCTS plans several steps ahead — it
samples plausible follow-up hypotheses, simulates their information
yield, and backs the realised value up to the root so the loop can
pick the next experiment that maximises long-run discovery.

Tree topology
-------------
* **Node** = chain state snapshot (whatever ``dict`` the master loop
  hands in — typically the chain id, current point/SE, depth,
  outstanding follow-up questions).
* **Action** = a string label for a follow-up hypothesis to propose.
  The label is opaque to the search; only the value-simulator
  interprets it.
* **Value** = realised information gain (Bayesian surprise on
  ``|point/SE|``, expressed in nats), provided by the
  ``simulate_value`` callback.

Progressive widening (Couëtoux et al. 2011; AutoDiscovery 2024) bounds
the branching factor under continuous / very-large action spaces. At a
node with ``n`` visits, the cap on children is

.. math::

    K(n) = \\lceil k \\cdot n^{\\alpha} \\rceil

with defaults ``k = 1.0`` and ``\\alpha = 0.5``. A child is only
expanded when ``len(children) < K(n)``; otherwise the loop must descend
through the existing children and play UCB1.

The module is pure: it never imports the master loop, the LLM client,
or any I/O surface. The user wires in their own ``propose_actions``
and ``simulate_value`` callbacks — typically an LLM-driven proposer
and a Bayesian-surprise simulator — and the search returns the best
next action plus the full search tree so the TUI / decision ledger
can render *why* the action was picked.

References
----------
* Kocsis & Szepesvári (2006). "Bandit based Monte-Carlo Planning."
  UCB1-applied-to-trees (UCT).
* Couëtoux, Hoock, Sokolovska, Teytaud, Bonnard (2011).
  "Continuous Upper Confidence Trees." Introduces progressive
  widening for continuous action spaces.
* Lu, Williams, Mansinghka, et al. (2024). "AutoDiscovery." Applies
  PW-MCTS to LLM-driven scientific hypothesis spaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "MCTSConfig",
    "MCTSNode",
    "HypothesisMCTS",
]


# ─────────── Data classes ────────────────────────────────────────────────


@dataclass
class MCTSNode:
    """A single node in the hypothesis search tree.

    Attributes
    ----------
    state:
        Snapshot of the chain state at this node. Opaque to MCTS;
        only the user-supplied callbacks read its fields.
    visits:
        UCT visit count. Incremented once per backpropagation pass.
    total_value:
        Sum of discounted realised information gains backed up
        through this node. Mean value is ``total_value / visits``.
    children:
        Ordered list of expanded children. The order matches the
        order in which the proposer offered the corresponding actions
        — important for reproducibility under deterministic
        proposers.
    parent:
        Parent node, or ``None`` for the root.
    last_action:
        The action label that led to this node from its parent.
        ``None`` at the root.
    """

    state: dict[str, Any]
    visits: int = 0
    total_value: float = 0.0
    children: list["MCTSNode"] = field(default_factory=list)
    parent: "MCTSNode | None" = None
    last_action: str | None = None

    @property
    def mean_value(self) -> float:
        """Empirical mean of backed-up values; 0.0 if never visited."""
        if self.visits <= 0:
            return 0.0
        return self.total_value / self.visits

    @property
    def is_root(self) -> bool:
        return self.parent is None


@dataclass
class MCTSConfig:
    """Tunable knobs for :class:`HypothesisMCTS`.

    Attributes
    ----------
    max_iterations:
        Total number of select/expand/simulate/backprop passes the
        search executes before returning.
    max_depth:
        Hard cap on rollout depth measured from the root. Prevents
        unbounded recursion when the proposer keeps offering follow-up
        hypotheses indefinitely.
    exploration_c:
        UCB1 exploration constant ``c``. ``sqrt(2) ≈ 1.41`` is the
        Auer et al. (2002) textbook choice.
    progressive_widening_k:
        Linear coefficient ``k`` in the PW child cap
        ``K(n) = ceil(k · n^α)``.
    progressive_widening_alpha:
        Exponent ``α``. ``0.5`` is the Couëtoux 2011 default.
    discount:
        Per-step discount on backed-up value. ``1.0`` is undiscounted
        UCT; ``0.95`` slightly prefers near-term information gain,
        which matches the master loop's preference for compact
        chains.
    """

    max_iterations: int = 50
    max_depth: int = 5
    exploration_c: float = 1.41
    progressive_widening_k: float = 1.0
    progressive_widening_alpha: float = 0.5
    discount: float = 0.95


# ─────────── Search driver ───────────────────────────────────────────────


# Callback types — kept as plain ``Callable`` aliases so users can wire
# in lambdas, partials, or fully-typed classes without inheriting from
# anything.
ProposeActions = Callable[[dict[str, Any], int], list[str]]
SimulateValue = Callable[[dict[str, Any], str], float]


class HypothesisMCTS:
    """MCTS rollout over the hypothesis tree with progressive widening.

    The user supplies:

    * ``propose_actions(state, n) -> list[str]`` — typically an
      LLM-driven proposer that returns *up to* ``n`` candidate
      follow-up hypothesis labels for a given chain state.
    * ``simulate_value(state, action) -> float`` — the realised
      information gain (e.g. Bayesian surprise on ``|point/SE|``)
      that the loop *would* observe if it took ``action`` from
      ``state``. The simulator is also expected to return the
      successor state implicitly through its action label; MCTS
      itself does not need to know what the new state looks like,
      because progressive widening only needs the parent's visit
      count.

    The search runs :attr:`MCTSConfig.max_iterations` UCT passes and
    returns the best next action (highest mean value among the root's
    children) plus the full tree for downstream rendering.

    Notes
    -----
    * The tree is built lazily. ``children`` is grown by progressive
      widening, so the first iterations explore a narrow trunk and
      branching opens up only as the root accumulates visits.
    * The simulator is called once per iteration on the freshly
      expanded leaf — this is the standard single-rollout UCT, not the
      full rollout-to-terminal variant.
    * The class holds no mutable global state and is safe to
      instantiate multiple times concurrently.
    """

    def __init__(
        self,
        root_state: dict[str, Any],
        propose_actions: ProposeActions,
        simulate_value: SimulateValue,
        config: MCTSConfig | None = None,
    ) -> None:
        if not callable(propose_actions):
            raise TypeError("propose_actions must be callable")
        if not callable(simulate_value):
            raise TypeError("simulate_value must be callable")

        self.config = config if config is not None else MCTSConfig()
        if self.config.max_iterations < 0:
            raise ValueError(
                f"max_iterations must be ≥ 0; got {self.config.max_iterations}"
            )
        if self.config.max_depth < 1:
            raise ValueError(f"max_depth must be ≥ 1; got {self.config.max_depth}")
        if self.config.exploration_c < 0.0:
            raise ValueError(
                f"exploration_c must be ≥ 0; got {self.config.exploration_c}"
            )
        if self.config.progressive_widening_k <= 0.0:
            raise ValueError(
                "progressive_widening_k must be > 0; got "
                f"{self.config.progressive_widening_k}"
            )
        if not (0.0 < self.config.progressive_widening_alpha <= 1.0):
            raise ValueError(
                "progressive_widening_alpha must be in (0, 1]; got "
                f"{self.config.progressive_widening_alpha}"
            )
        if not (0.0 < self.config.discount <= 1.0):
            raise ValueError(
                f"discount must be in (0, 1]; got {self.config.discount}"
            )

        self.root = MCTSNode(state=dict(root_state))
        self._propose_actions = propose_actions
        self._simulate_value = simulate_value

    # ─────────── Public API ──────────────────────────────────────────────

    def search(self) -> tuple[str, MCTSNode]:
        """Run the MCTS loop and return ``(best_action, root_node)``.

        ``best_action`` is the ``last_action`` label of the root's
        most-visited child, with ties broken by mean value, and
        then by insertion order (i.e. the order the proposer offered
        them). This is the conventional MCTS *robust child* rule
        (Browne et al. 2012) — visit count is a more stable signal
        than mean value alone.

        Raises
        ------
        RuntimeError
            If the proposer offers no children at the root (so there
            is no action to return).
        """
        for _ in range(self.config.max_iterations):
            leaf = self._select(self.root, depth=0)
            value = self._simulate(leaf)
            self._backpropagate(leaf, value)

        if not self.root.children:
            # The proposer never offered an action at the root — the
            # search has nothing to recommend. We surface this as a
            # RuntimeError so callers don't silently no-op.
            raise RuntimeError(
                "HypothesisMCTS.search: proposer returned no actions at root"
            )

        best = max(
            self.root.children,
            key=lambda child: (child.visits, child.mean_value),
        )
        # ``best.last_action`` was set when the child was expanded; it
        # cannot be None for a non-root node.
        assert best.last_action is not None
        return best.last_action, self.root

    def render_tree(self) -> str:
        """Pretty-print the search tree.

        Each line shows the action label that led to the node, its
        visit count, mean value, and total backed-up value. The root
        is annotated with ``<root>``. Children are indented two
        spaces per depth level so the structure is preview-friendly
        in a TUI panel.
        """
        lines: list[str] = []
        self._render(self.root, depth=0, lines=lines)
        return "\n".join(lines)

    # ─────────── UCT core ────────────────────────────────────────────────

    def _select(self, node: MCTSNode, depth: int) -> MCTSNode:
        """Descend from ``node`` to a leaf, expanding via PW as we go.

        The selection rule is:

        1. If we have hit ``max_depth`` from the root, stop and
           return ``node`` — it acts as a leaf for this iteration.
        2. If progressive widening allows another child at this
           node, expand one and return the new child (this is the
           leaf for this iteration).
        3. Otherwise pick the UCB1-best existing child and recurse.

        Returns the leaf at which the rollout / simulation will run.
        """
        if depth >= self.config.max_depth:
            return node

        if self._can_widen(node):
            child = self._expand(node)
            if child is not None:
                return child
            # If expansion failed (proposer exhausted), fall through
            # to UCB1 descent below. If there are no children at all,
            # we have no choice but to stop here.
            if not node.children:
                return node

        if not node.children:
            return node

        best_child = self._uct_select(node)
        return self._select(best_child, depth + 1)

    def _can_widen(self, node: MCTSNode) -> bool:
        """Progressive-widening predicate.

        Returns True iff the node may still acquire a new child under
        ``K(n) = ceil(k · n^α)``.

        At ``n_visits == 0`` (e.g. the brand-new root) we still allow
        one child so the search has *something* to descend into —
        ``ceil(k · 0^α)`` would otherwise be ``0`` and the search
        would be stuck. This matches the AutoDiscovery 2024 convention.
        """
        cap = self._pw_cap(node.visits)
        return len(node.children) < cap

    def _pw_cap(self, visits: int) -> int:
        """``K(n) = max(1, ceil(k · n^α))``.

        The ``max(1, …)`` floor guarantees every node has at least one
        chance to expand a child before UCB1 takes over — without it
        the root would be stuck with zero children when ``visits == 0``.
        """
        k = self.config.progressive_widening_k
        alpha = self.config.progressive_widening_alpha
        # n^α is well-defined for n=0 only when α>0, where 0^α=0; we
        # rely on the max(1, …) floor for that edge case.
        raw = k * (visits**alpha) if visits > 0 else 0.0
        return max(1, math.ceil(raw))

    def _expand(self, node: MCTSNode) -> MCTSNode | None:
        """Ask the proposer for one fresh action and add it as a child.

        We request ``cap`` actions to give the proposer enough room to
        return a label we haven't seen yet, then filter out duplicates
        of existing children. Returns the new child node, or ``None``
        if the proposer is exhausted (no new labels available).
        """
        cap = self._pw_cap(node.visits)
        existing = {c.last_action for c in node.children}
        proposed = self._propose_actions(node.state, cap) or []
        for action in proposed:
            if action in existing:
                continue
            child_state = dict(node.state)  # cheap snapshot for the child
            child = MCTSNode(
                state=child_state,
                parent=node,
                last_action=action,
            )
            node.children.append(child)
            return child
        return None

    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 / UCT among ``node.children``.

        Unvisited children get an infinite score so they are picked
        first, preserving the UCB1 invariant that every arm is played
        at least once. Ties are broken by insertion order — the first
        such child wins — to keep behaviour deterministic for tests.
        """
        log_parent = (
            math.log(node.visits) if node.visits > 0 else 0.0
        )
        c = self.config.exploration_c

        best: MCTSNode | None = None
        best_score = -math.inf
        for child in node.children:
            if child.visits <= 0:
                score = math.inf
            else:
                exploit = child.mean_value
                explore = c * math.sqrt(log_parent / child.visits)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best = child
        assert best is not None  # node.children was non-empty by caller
        return best

    def _simulate(self, leaf: MCTSNode) -> float:
        """Single-step rollout: ask the simulator for the leaf's value.

        Root nodes have no inbound action, so there is nothing to
        simulate — they receive value ``0.0`` and act as a pass-through
        in backpropagation. This case only occurs on the very first
        iteration if PW expansion failed at the root, which already
        raises in :meth:`search`.
        """
        if leaf.last_action is None:
            return 0.0
        # The simulator reads the *parent's* state plus the action.
        # The leaf carries a snapshot of the parent's state by design
        # (see :meth:`_expand`), so passing ``leaf.state`` is correct
        # and lets users mutate the snapshot if they like.
        parent_state = leaf.parent.state if leaf.parent is not None else leaf.state
        value = float(self._simulate_value(parent_state, leaf.last_action))
        if not math.isfinite(value):
            # Reject non-finite signals — they would poison every
            # ancestor's mean. Treat as zero information.
            return 0.0
        return value

    def _backpropagate(self, leaf: MCTSNode, value: float) -> None:
        """Propagate the rollout value back up to the root.

        Each ancestor's ``visits`` increments by 1 and its
        ``total_value`` accumulates the discounted value, with the
        discount applied per step of distance from the leaf. This
        means a value of ``v`` realised three steps below the root
        contributes ``γ³ · v`` to the root's running total, which
        biases the root toward action sequences that pay off soon.
        """
        gamma = self.config.discount
        node: MCTSNode | None = leaf
        contribution = value
        while node is not None:
            node.visits += 1
            node.total_value += contribution
            contribution *= gamma
            node = node.parent

    # ─────────── Rendering ───────────────────────────────────────────────

    def _render(self, node: MCTSNode, depth: int, lines: list[str]) -> None:
        indent = "  " * depth
        label = node.last_action if node.last_action is not None else "<root>"
        lines.append(
            f"{indent}- {label}  "
            f"[visits={node.visits}, mean={node.mean_value:.4f}, "
            f"total={node.total_value:.4f}]"
        )
        # Render children in insertion order — matches selection order
        # under deterministic proposers, which the TUI relies on.
        for child in node.children:
            self._render(child, depth + 1, lines)
