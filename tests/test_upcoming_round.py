import numpy as np

from cs2p.live import fixed_rounds as validate
from cs2p.swiss import run_stage, upcoming_round


def test_upcoming_round_replays_the_simulator_pairings():
    rng = np.random.default_rng(3)
    for _ in range(20):
        strength = rng.permutation(16)
        p = (strength[:, None] > strength[None, :]).astype(float)  # deterministic winners
        rounds: list = []
        while (nxt := upcoming_round(16, rounds)) is not None:
            rnd, pairs = nxt
            assert rnd == len(rounds) + 1
            rounds.append([(a, b, a if p[a, b] else b) for a, b, _ in pairs])
        seeds = [f"T{i}" for i in range(16)]
        matches = [{"round": r, "a": seeds[a], "b": seeds[b], "winner": seeds[w]}
                   for r, fx in enumerate(rounds, 1) for a, b, w in fx]
        validate(seeds, matches)  # legal Swiss: records match, nobody plays twice per round
        free, _ = run_stage(p, np.random.default_rng(0), p_bo1=p)
        replay, _ = run_stage(p, np.random.default_rng(0), p_bo1=p, fixed_rounds=rounds)
        assert (free == replay).all()


def test_partial_round_keeps_played_matches():
    first = upcoming_round(16, [])[1]
    played = [(a, b, a) for a, b, _ in first[:3]]
    rnd, pairs = upcoming_round(16, [played])
    assert rnd == 1 and pairs[:3] == played and len(pairs) == 8
