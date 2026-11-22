"""Live tournament state for the PGL Singapore Major, pulled from Liquipedia.

For each Swiss stage: seeds (once Liquipedia publishes them) and every finished match,
converted into `fixed_rounds` so simulations continue from what actually happened.
Stage 2/3 seeding follows the official rule: invites 1-8 by VRS, previous-stage
qualifiers 9-16 by Swiss record then Buchholz.
"""

import functools

import numpy as np
import pandas as pd

from cs2p import teams
from cs2p.elo import PROC
from cs2p.ingest_liquipedia import fetch_pages, parse_stage
from cs2p.playoffs import QF_SEEDS
from cs2p.swiss import ADV, ADV_30, bo5_from_bo3, run_stage

ALL_TEAMS = teams.STAGE1 + teams.STAGE2_INVITES + teams.STAGE3_INVITES
STAGE_PAGES = {1: "PGL/2026/Singapore/Stage 1", 2: "PGL/2026/Singapore/Stage 2", 3: "PGL/2026/Singapore/Stage 3"}
INVITES = {1: [], 2: teams.STAGE2_INVITES, 3: teams.STAGE3_INVITES}
DISPLAY = {v.casefold(): d for d, v in ALL_TEAMS}


@functools.cache
def _team_maps() -> tuple[dict, dict]:
    tm = pd.read_csv(PROC / "liquipedia_team_map.csv")
    tmap = {c: (v if isinstance(v, str) else None) for c, v in zip(tm.code, tm.vrs_name)}
    return tmap, dict(zip(tm.code, tm.liquipedia_name))


@functools.cache
def _latest_vrs() -> dict[str, float]:
    pts = pd.read_csv(PROC / "vrs_points.csv", parse_dates=["snapshot"])
    return pts[pts.snapshot == pts.snapshot.max()].groupby("team").points.max().to_dict()


def vrs_order(display_names: list[str]) -> list[str]:
    last, vrs = _latest_vrs(), dict(ALL_TEAMS)
    return sorted(display_names, key=lambda t: -last.get(vrs[t], 0))


def fetch_state(max_age_h: float = 0.25) -> dict[int, dict]:
    """Return {stage: {seeds, matches, status}} with team names as display names."""
    fetch_pages(list(STAGE_PAGES.values()), max_age_h=max_age_h)  # refresh cache
    tmap, lp = _team_maps()
    state = {}
    for n, title in STAGE_PAGES.items():
        st = parse_stage(f"Singapore Stage {n}", title, tmap, lp)
        seeds = [DISPLAY.get(t["name"].casefold()) for t in st["teams"]]
        known = len(seeds) == 16 and all(seeds)
        matches = [
            {"round": m["round"], "date": m["date"], "a": DISPLAY.get(m["team1"].casefold()),
             "b": DISPLAY.get(m["team2"].casefold()), "winner": DISPLAY.get(m["winner"].casefold())}
            for m in st["matches"]
        ]
        matches = [m for m in matches if m["a"] and m["b"] and m["winner"] and m["round"]]
        done = known and _finished(seeds, matches)
        status = "complete" if done else "live" if matches else "upcoming"
        state[n] = {"seeds": seeds if known else None, "matches": matches, "status": status,
                    "start_date": st["start_date"]}
    if state[1]["seeds"] is None:  # Stage 1 is published; fall back to our static list
        state[1]["seeds"] = [d for d, _ in teams.STAGE1]
    return state


def _finished(seeds: list[str], matches: list[dict]) -> bool:
    w = {t: 0 for t in seeds}
    l = {t: 0 for t in seeds}
    for m in matches:
        w[m["winner"]] += 1
        l[m["b"] if m["winner"] == m["a"] else m["a"]] += 1
    return all(w[t] == 3 or l[t] == 3 for t in seeds)


def fixed_rounds(seeds: list[str], matches: list[dict]) -> list[list[tuple[int, int, int]]]:
    idx = {t: i for i, t in enumerate(seeds)}
    rounds: list[list] = [[] for _ in range(5)]
    for m in matches:
        if m["a"] in idx and m["b"] in idx and 1 <= m["round"] <= 5:
            rounds[m["round"] - 1].append((idx[m["a"]], idx[m["b"]], idx[m["winner"]]))
    while rounds and not rounds[-1]:
        rounds.pop()
    return rounds


def major_matches(state: dict) -> pd.DataFrame:
    """Finished Major matches as rating-update rows (VRS names)."""
    vrs = dict(ALL_TEAMS)
    rows = [{"date": m["date"], "winner": vrs[m["winner"]], "loser": vrs[m["b"] if m["winner"] == m["a"] else m["a"]]}
            for st in state.values() for m in st["matches"] if m["date"]]
    return pd.DataFrame(rows, columns=["date", "winner", "loser"]).assign(date=lambda d: pd.to_datetime(d.date))


def simulate_major(P: pd.DataFrame, state: dict, n: int, seed: int = 1) -> dict:
    """Simulate the rest of the Major from the current state.

    Returns per-stage outcome codes for all 32 teams (-1 = not in that stage), playoff
    seeds per sim, and slot winners per sim.
    """
    names = list(P.index)
    col = {t: i for i, t in enumerate(names)}
    Pv = P.values
    rng = np.random.default_rng(seed)
    codes = {s: np.full((n, len(names)), -1, dtype=np.int8) for s in (1, 2, 3)}
    playoff_seeds = np.empty((n, 8), dtype=np.int16)
    winners = np.empty((n, 7), dtype=np.int16)
    fixed = {s: fixed_rounds(st["seeds"], st["matches"]) if st["seeds"] else [] for s, st in state.items()}
    p5 = bo5_from_bo3(Pv)
    for i in range(n):
        qualifiers: list[str] = []
        for s in (1, 2, 3):
            if state[s]["seeds"]:
                field = state[s]["seeds"]
            else:
                field = vrs_order([d for d, _ in INVITES[s]]) + qualifiers
            ix = [col[t] for t in field]
            c, order = run_stage(Pv[np.ix_(ix, ix)], rng, all_bo3=(s == 3), fixed_rounds=fixed[s])
            codes[s][i, ix] = c
            qualifiers = [field[t] for t in order if c[t] in (ADV_30, ADV)]
        ps = [col[t] for t in qualifiers]  # Stage 3 standings order = playoff seeds
        playoff_seeds[i] = ps
        qf = [ps[a] if rng.random() < Pv[ps[a], ps[b]] else ps[b] for a, b in QF_SEEDS]
        sf = [qf[0] if rng.random() < Pv[qf[0], qf[1]] else qf[1], qf[2] if rng.random() < Pv[qf[2], qf[3]] else qf[3]]
        fin = sf[0] if rng.random() < p5[sf[0], sf[1]] else sf[1]
        winners[i] = qf + sf + [fin]
    return {"names": names, "codes": codes, "playoff_seeds": playoff_seeds, "winners": winners}
