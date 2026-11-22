import numpy as np

from cs2p.elo import Elo, EloParams, win_prob
from cs2p.pickem import optimise, score
from cs2p.swiss import ADV, ADV_30, ELIM, ELIM_03, bo1_from_bo3, simulate


def test_win_prob_symmetric():
    assert win_prob(1500, 1500) == 0.5
    assert abs(win_prob(1600, 1500) + win_prob(1500, 1600) - 1) < 1e-12


def test_elo_update_zero_sum_for_established_teams():
    elo = Elo(EloParams(n_new=0, decay_days=1e9))
    import pandas as pd

    d = pd.Timestamp("2026-01-01")
    elo.update("A", "B", d)
    assert abs(elo.rating["a"] + elo.rating["b"] - 2 * elo.p.init) < 1e-9
    assert elo.rating["a"] > elo.rating["b"]


def test_bo1_inverse():
    p = np.array([0.3, 0.5, 0.8])
    q = bo1_from_bo3(p)
    assert np.allclose(q**2 * (3 - 2 * q), p, atol=1e-6)
    assert q[2] < p[2] and q[0] > p[0]  # Bo1 is closer to a coin flip


def test_swiss_invariants():
    rng = np.random.default_rng(0)
    r = rng.normal(1500, 100, 16)
    p = win_prob(r[:, None], r[None, :])
    out = simulate(p, n_sims=300)
    for row in out:
        assert (row == ADV_30).sum() == 2
        assert (row == ELIM_03).sum() == 2
        assert ((row == ADV_30) | (row == ADV)).sum() == 8
        assert ((row == ELIM_03) | (row == ELIM)).sum() == 8


def test_optimiser_returns_valid_disjoint_picks():
    r = np.linspace(1800, 1300, 16)
    out = simulate(win_prob(r[:, None], r[None, :]), n_sims=2000)
    picks, (p5, exp) = optimise(out)
    flat = [t for v in picks.values() for t in v]
    assert len(flat) == 10 and len(set(flat)) == 10
    assert len(picks[ADV_30]) == 2 and len(picks[ELIM_03]) == 2 and len(picks[ADV]) == 6
    assert 0 <= p5 <= 1 and np.isclose(exp, score(out, picks).mean())
    # Strongest teams should dominate the 3-0 slot, weakest the 0-3 slot.
    assert min(picks[ADV_30]) < 4 and max(picks[ELIM_03]) > 11
