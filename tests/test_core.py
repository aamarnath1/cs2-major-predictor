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


def test_fixed_rounds_replay_real_results_exactly():
    from cs2p.swiss import run_stage

    p = np.full((16, 16), 0.5)
    rng = np.random.default_rng(3)
    # Seed 0..7 win round 1; then fix round 2 high group fully.
    r1 = [(i, i + 8, i) for i in range(8)]
    r2 = [(0, 7, 0), (1, 6, 1), (2, 5, 2), (3, 4, 3)]
    for _ in range(50):
        codes, order = run_stage(p, rng, fixed_rounds=[r1, r2])
        assert all(codes[t] != ELIM_03 for t in range(8))  # every round-1 winner has a win
        assert all(codes[t] != ADV_30 for t in range(8, 16))  # every round-1 loser has a loss
        assert len(order) == 16


def test_partial_round_fills_remaining_pairs():
    from cs2p.swiss import run_stage

    p = np.full((16, 16), 0.5)
    rng = np.random.default_rng(4)
    codes, _ = run_stage(p, rng, fixed_rounds=[[(0, 8, 8)]])  # only one R1 result known
    assert codes[0] != ADV_30 and codes[8] != ELIM_03


def test_bo5_more_decisive_than_bo3():
    from cs2p.swiss import bo5_from_bo3

    p = np.array([0.3, 0.5, 0.7])
    b5 = bo5_from_bo3(p)
    assert np.isclose(b5[1], 0.5) and b5[2] > p[2] and b5[0] < p[0]


def test_bracket_enumeration_and_optimiser():
    from cs2p.playoffs import all_brackets, optimise_bracket, simulate_bracket

    br = all_brackets()
    assert len(br) == 128 and len({tuple(b) for b in br}) == 128
    r = np.linspace(1800, 1400, 8)
    w = simulate_bracket(win_prob(r[:, None], r[None, :]), n_sims=3000)
    best, dist = optimise_bracket(w)
    assert best[-1] == 0 and np.isclose(dist.sum(), 1)  # strongest seed picked to win


def test_pick_status():
    from cs2p.run import pick_status

    assert pick_status("3-0", 3, 0) == "hit" and pick_status("3-0", 1, 1) == "miss"
    assert pick_status("0-3", 0, 3) == "hit" and pick_status("0-3", 1, 0) == "miss"
    assert pick_status("3-1/3-2", 3, 0) == "miss" and pick_status("3-1/3-2", 3, 2) == "hit"
    assert pick_status("3-1/3-2", 2, 2) == "open" and pick_status("3-1/3-2", 1, 3) == "miss"
