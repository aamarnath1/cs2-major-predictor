"""Stage hand-off: the next stage's field and seeding follow from the rulebook once a stage ends."""

import numpy as np

from cs2p import live, teams
from cs2p.swiss import run_stage, upcoming_round

STAGE1 = [d for d, _ in teams.STAGE1]
VRS = dict(live.ALL_TEAMS)


def play_stage(seeds: list[str], rounds: int = 5, max_in_last: int = 99, seed: int = 3) -> list[dict]:
    """Made-up results (random winners) following Swiss pairing rules."""
    rng = np.random.default_rng(seed)
    matches: list[dict] = []
    while (nxt := upcoming_round(16, live.fixed_rounds(seeds, matches))) is not None:
        rnd, pairs = nxt
        if rnd > rounds:
            break
        for a, b, w in pairs:
            if w is None and not (rnd == rounds and sum(m["round"] == rnd for m in matches) >= max_in_last):
                matches.append({"round": rnd, "date": "2026-11-25", "a": seeds[a], "b": seeds[b],
                                "winner": seeds[a] if rng.random() < .5 else seeds[b]})
        if rnd == rounds and max_in_last < 99:
            break
    return matches


def test_next_stage_seeds_match_simulator_standings():
    matches = play_stage(STAGE1)
    fr = live.fixed_rounds(STAGE1, matches)
    codes, order = run_stage(np.full((16, 16), .5), np.random.default_rng(0), fixed_rounds=fr)
    qualifiers = [STAGE1[t] for t in order if codes[t] in (0, 1)]
    seeds = live.next_stage_seeds(2, {"seeds": STAGE1, "matches": matches})
    assert seeds == [d for d, _ in teams.STAGE2_INVITES] + qualifiers
    assert len(set(seeds)) == 16


def fake_liquipedia(monkeypatch, pages: dict) -> None:
    monkeypatch.setattr(live, "fetch_pages", lambda titles, max_age_h=0: {live.PLAYOFF_PAGE: {"content": ""}})
    monkeypatch.setattr(live, "_team_maps", lambda: ({}, {}))
    monkeypatch.setattr(live, "parse_stage", lambda event, title, tmap, lp: pages[int(title[-1])])
    monkeypatch.setattr(live, "parse_playoffs", lambda text, tmap, lp: {"slots": {}, "status": "upcoming"})


def lp(seeds: list[str], matches: list[dict]) -> dict:
    return {"teams": [{"name": VRS[t]} for t in seeds], "start_date": None,
            "matches": [{"round": m["round"], "date": m["date"], "team1": VRS[m["a"]], "team2": VRS[m["b"]],
                         "winner": VRS[m["winner"]]} for m in matches]}


def test_stage2_field_comes_from_rulebook_when_liquipedia_lags(monkeypatch):
    m1 = play_stage(STAGE1)
    invites2, invites3 = [d for d, _ in teams.STAGE2_INVITES], [d for d, _ in teams.STAGE3_INVITES]
    fake_liquipedia(monkeypatch, {1: lp(STAGE1, m1), 2: lp(invites2, []), 3: lp(invites3, [])})
    state = live.fetch_state()
    assert state[1]["status"] == "complete"
    assert state[2]["seeds"] == live.next_stage_seeds(2, state[1])
    assert state[2]["status"] == "upcoming"
    assert state[3]["seeds"] is None


def test_misordered_liquipedia_field_falls_back_to_rulebook(monkeypatch):
    m1 = play_stage(STAGE1)
    s2 = live.next_stage_seeds(2, {"seeds": STAGE1, "matches": m1})
    m2 = play_stage(s2, rounds=1, max_in_last=4)
    wrong = lp(s2, m2) | {"teams": [{"name": VRS[t]} for t in sorted(s2)]}
    fake_liquipedia(monkeypatch, {1: lp(STAGE1, m1), 2: wrong, 3: lp([d for d, _ in teams.STAGE3_INVITES], [])})
    state = live.fetch_state()
    assert state[2]["seeds"] == s2
    assert state[2]["status"] == "live"
