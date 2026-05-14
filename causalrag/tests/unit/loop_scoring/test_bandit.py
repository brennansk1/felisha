"""Unit tests for Sprint 3.2 — Thompson sampling + UCB1 chain allocation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

from causalrag.loop_scoring.bandit import (
    BanditArm,
    thompson_sample_chain,
    ucb1_chain_choice,
)


@dataclass
class _FakeChain:
    """Duck-typed stand-in for :class:`master_loop.ChainState`."""

    chain_id: str
    last_point: float | None = None
    last_se: float | None = None
    depth: int = 0


# ─────────── Thompson sampling ───────────────────────────────────────────


class TestThompsonSample:
    def test_strong_signal_chain_picked_majority(self) -> None:
        """Over 1000 Thompson draws, a chain with |point/SE|=4
        should beat a chain with |point/SE|=0.5 well over 70 % of
        the time."""
        strong = _FakeChain(chain_id="strong", last_point=4.0, last_se=1.0, depth=3)
        weak = _FakeChain(chain_id="weak", last_point=0.5, last_se=1.0, depth=3)

        wins: Counter[str] = Counter()
        rng = np.random.default_rng(20260513)
        for _ in range(1000):
            chosen, _arms = thompson_sample_chain(chains=[strong, weak], rng=rng)
            wins[chosen] += 1

        assert wins["strong"] / 1000 > 0.70, wins

    def test_three_null_chains_roughly_uniform(self) -> None:
        """Three chains with effectively identical null signals
        should be pulled at roughly the same rate (within 10 %)."""
        chains = [
            _FakeChain(chain_id=f"null_{i}", last_point=0.05, last_se=1.0, depth=2)
            for i in range(3)
        ]
        wins: Counter[str] = Counter()
        rng = np.random.default_rng(424242)
        n_draws = 3000
        for _ in range(n_draws):
            chosen, _ = thompson_sample_chain(chains=chains, rng=rng)
            wins[chosen] += 1

        expected = n_draws / 3
        for cid in ("null_0", "null_1", "null_2"):
            rel_err = abs(wins[cid] - expected) / expected
            assert rel_err < 0.10, (cid, wins, rel_err)

    def test_empty_chains_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            thompson_sample_chain(chains=[])

    def test_invalid_prior_variance_raises(self) -> None:
        chain = _FakeChain(chain_id="x", last_point=1.0, last_se=1.0, depth=1)
        with pytest.raises(ValueError, match="prior_variance"):
            thompson_sample_chain(chains=[chain], prior_variance=0.0)

    def test_chain_with_no_last_se_uses_prior(self) -> None:
        """A chain with no observation yet falls back to the prior —
        so its posterior_mean equals prior_mean and posterior_variance
        equals prior_variance."""
        chain = _FakeChain(chain_id="fresh", last_point=None, last_se=None, depth=0)
        rng = np.random.default_rng(0)
        chosen, arms = thompson_sample_chain(
            chains=[chain],
            rng=rng,
            prior_mean=1.0,
            prior_variance=4.0,
        )
        assert chosen == "fresh"
        assert len(arms) == 1
        assert arms[0].prior_used == "uniform"
        assert arms[0].posterior_mean == pytest.approx(1.0)
        assert arms[0].posterior_variance == pytest.approx(4.0)
        assert arms[0].n_pulls == 0

    def test_nan_inputs_treated_as_no_observation(self) -> None:
        """NaN point or SE → treated as missing → prior is kept."""
        nan_point = _FakeChain(
            chain_id="nan_point", last_point=float("nan"), last_se=1.0, depth=2
        )
        nan_se = _FakeChain(
            chain_id="nan_se", last_point=1.0, last_se=float("nan"), depth=2
        )
        rng = np.random.default_rng(7)
        _chosen, arms = thompson_sample_chain(chains=[nan_point, nan_se], rng=rng)
        for arm in arms:
            assert arm.prior_used == "uniform"
            assert arm.posterior_mean == pytest.approx(1.0)
            assert arm.posterior_variance == pytest.approx(4.0)

    def test_zero_se_is_skipped_gracefully(self) -> None:
        chain = _FakeChain(chain_id="degenerate", last_point=1.0, last_se=0.0, depth=2)
        rng = np.random.default_rng(0)
        chosen, arms = thompson_sample_chain(chains=[chain], rng=rng)
        assert chosen == "degenerate"
        assert arms[0].prior_used == "uniform"

    def test_posterior_tightens_with_more_pulls(self) -> None:
        """More observations → smaller posterior variance."""
        shallow = _FakeChain(chain_id="shallow", last_point=2.0, last_se=1.0, depth=1)
        deep = _FakeChain(chain_id="deep", last_point=2.0, last_se=1.0, depth=10)
        rng = np.random.default_rng(0)
        _, arms = thompson_sample_chain(chains=[shallow, deep], rng=rng)
        shallow_arm = next(a for a in arms if a.chain_id == "shallow")
        deep_arm = next(a for a in arms if a.chain_id == "deep")
        assert deep_arm.posterior_variance < shallow_arm.posterior_variance

    def test_rng_is_reproducible(self) -> None:
        chains = [
            _FakeChain(chain_id="a", last_point=1.0, last_se=1.0, depth=1),
            _FakeChain(chain_id="b", last_point=1.5, last_se=1.0, depth=1),
            _FakeChain(chain_id="c", last_point=0.5, last_se=1.0, depth=1),
        ]
        rng1 = np.random.default_rng(2026)
        rng2 = np.random.default_rng(2026)
        seq1 = [thompson_sample_chain(chains=chains, rng=rng1)[0] for _ in range(20)]
        seq2 = [thompson_sample_chain(chains=chains, rng=rng2)[0] for _ in range(20)]
        assert seq1 == seq2

    def test_returns_one_arm_per_chain(self) -> None:
        chains = [
            _FakeChain(chain_id=f"c{i}", last_point=float(i), last_se=1.0, depth=i)
            for i in range(5)
        ]
        rng = np.random.default_rng(1)
        chosen, arms = thompson_sample_chain(chains=chains, rng=rng)
        assert len(arms) == 5
        assert {a.chain_id for a in arms} == {f"c{i}" for i in range(5)}
        assert chosen in {a.chain_id for a in arms}


# ─────────── UCB1 ────────────────────────────────────────────────────────


class TestUcb1:
    def test_deterministic_given_same_input(self) -> None:
        chains = [
            _FakeChain(chain_id="a", last_point=2.0, last_se=1.0, depth=3),
            _FakeChain(chain_id="b", last_point=0.5, last_se=1.0, depth=3),
            _FakeChain(chain_id="c", last_point=1.0, last_se=1.0, depth=5),
        ]
        chosen1, arms1 = ucb1_chain_choice(chains)
        chosen2, arms2 = ucb1_chain_choice(chains)
        assert chosen1 == chosen2
        assert [a.chain_id for a in arms1] == [a.chain_id for a in arms2]
        assert [a.posterior_mean for a in arms1] == [a.posterior_mean for a in arms2]

    def test_unpulled_chain_picked_first(self) -> None:
        """UCB1 must play every arm at least once."""
        chains = [
            _FakeChain(chain_id="seasoned", last_point=10.0, last_se=1.0, depth=20),
            _FakeChain(chain_id="virgin", last_point=None, last_se=None, depth=0),
        ]
        chosen, arms = ucb1_chain_choice(chains)
        assert chosen == "virgin"
        virgin_arm = next(a for a in arms if a.chain_id == "virgin")
        assert virgin_arm.n_pulls == 0
        assert virgin_arm.prior_used == "ucb_seeded"

    def test_strong_chain_wins_when_all_pulled(self) -> None:
        chains = [
            _FakeChain(chain_id="strong", last_point=5.0, last_se=1.0, depth=2),
            _FakeChain(chain_id="weak", last_point=0.1, last_se=1.0, depth=2),
        ]
        chosen, _ = ucb1_chain_choice(chains)
        assert chosen == "strong"

    def test_empty_chains_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ucb1_chain_choice([])

    def test_bonus_decreases_with_n_pulls(self) -> None:
        """The exploration bonus is larger for a less-pulled arm at
        the same payoff."""
        chains = [
            _FakeChain(chain_id="shallow", last_point=1.0, last_se=1.0, depth=1),
            _FakeChain(chain_id="deep", last_point=1.0, last_se=1.0, depth=100),
        ]
        _, arms = ucb1_chain_choice(chains, c=1.41)
        shallow_arm = next(a for a in arms if a.chain_id == "shallow")
        deep_arm = next(a for a in arms if a.chain_id == "deep")
        # posterior_variance is the squared bonus → strictly larger
        # for the shallow arm.
        assert shallow_arm.posterior_variance > deep_arm.posterior_variance

    def test_nan_payoff_treated_as_unpulled(self) -> None:
        chains = [
            _FakeChain(chain_id="strong", last_point=5.0, last_se=1.0, depth=4),
            _FakeChain(chain_id="nan", last_point=float("nan"), last_se=1.0, depth=4),
        ]
        chosen, _ = ucb1_chain_choice(chains)
        assert chosen == "nan"


# ─────────── BanditArm shape ─────────────────────────────────────────────


class TestBanditArm:
    def test_arm_fields_populated(self) -> None:
        chain = _FakeChain(chain_id="x", last_point=2.0, last_se=1.0, depth=4)
        rng = np.random.default_rng(0)
        _, arms = thompson_sample_chain(chains=[chain], rng=rng)
        arm = arms[0]
        assert isinstance(arm, BanditArm)
        assert arm.chain_id == "x"
        assert arm.n_pulls == 4
        assert math.isfinite(arm.posterior_mean)
        assert math.isfinite(arm.posterior_variance)
        assert arm.posterior_variance > 0.0
        assert arm.prior_used in {"uniform", "user_specified", "ucb_seeded"}
