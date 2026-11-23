"""Live tournament state for the PGL Singapore Major, pulled from Liquipedia.

For each Swiss stage: seeds (once Liquipedia publishes them) and every finished match,
converted into `fixed_rounds` so simulations continue from what actually happened.
Stage 2/3 seeding follows the official rule: invites 1-8 by VRS, previous-stage
qualifiers 9-16 by Swiss record then Buchholz.
"""

import functools
from collections import Counter

import numpy as np
import pandas as pd

from cs2p import teams
from cs2p.elo import PROC
from cs2p.ingest_liquipedia import (COMMENT, bracket_slot, fetch_pages, find_templates, opponent, parse_matches,
                                    parse_stage, split_params, to_vrs)
from cs2p.playoffs import QF_SEEDS
from cs2p.swiss import ADV, ADV_30, bo5_from_bo3, run_stage

ALL_TEAMS = teams.STAGE1 + teams.STAGE2_INVITES + teams.STAGE3_INVITES
STAGE_PAGES = {1: "PGL/2026/Singapore/Stage 1", 2: "PGL/2026/Singapore/Stage 2", 3: "PGL/2026/Singapore/Stage 3"}
PLAYOFF_PAGE = "PGL/2026/Singapore/Playoffs"
# Liquipedia Bracket/8 slots in our order: QF1..QF4, SF1, SF2, Final (matches QF_SEEDS / playoffs.py).
PLAYOFF_SLOTS = ["R1M1", "R1M2", "R1M3", "R1M4", "R2M1", "R2M2", "R3M1"]
INVITES = {1: [], 2: teams.STAGE2_INVITES, 3: teams.STAGE3_INVITES}
DISPLAY = {v.casefold(): d for d, v in ALL_TEAMS}


@functools.cache
def _team_maps() -> tuple[dict, dict]:
    tm = pd.read_csv(PROC / "liquipedia_team_map.csv")
    tmap = {c: (v if isinstance(v, str) else None) for c, v in zip(tm.code, tm.vrs_name)}
    return tmap, dict(zip(tm.code, tm.liquipedia_name))


def fetch_state(max_age_h: float = 0.25) -> dict[int, dict]:
    """Return {stage: {seeds, matches, status}} with team names as display names."""
    pages = fetch_pages(list(STAGE_PAGES.values()) + [PLAYOFF_PAGE], max_age_h=max_age_h)  # refresh cache
    tmap, lp = _team_maps()
    state = {}
    for n, title in STAGE_PAGES.items():
        st = parse_stage(f"Singapore Stage {n}", title, tmap, lp)
        seeds = [DISPLAY.get(t["name"].casefold()) for t in st["teams"]]
        confirmed = seeds if len(seeds) == 16 and all(seeds) else None
        if len(seeds) == 16 and confirmed is None:  # field published but a team didn't map: say so loudly
            missing = [t["name"] for t, d in zip(st["teams"], seeds) if d is None]
            print(f"WARNING: Stage {n} field is published but these teams are unmapped: {missing}. "
                  "Add them to cs2p/teams.py or the Liquipedia team map, or the stage stays in preview.")
        if n == 1 and confirmed is None:  # Stage 1 has a published static field.
            confirmed = [d for d, _ in teams.STAGE1]
        matches = [
            {"round": m["round"], "date": m["date"], "a": DISPLAY.get(m["team1"].casefold()),
             "b": DISPLAY.get(m["team2"].casefold()), "winner": DISPLAY.get(m["winner"].casefold())}
            for m in st["matches"]
        ]
        try:
            if confirmed is not None:
                fixed_rounds(confirmed, matches)
                if n == 1:
                    if set(confirmed) != {d for d, _ in teams.STAGE1}:
                        raise ValueError("The field disagrees with the published Stage 1 roster.")
                else:
                    previous = state[n - 1]
                    if previous["status"] != "complete":
                        raise ValueError("The previous Swiss stage is incomplete.")
                    wins = Counter(m["winner"] for m in previous["matches"])
                    qualifiers = {team for team, count in wins.items() if count == 3}
                    invites = {team for team, _ in INVITES[n]}
                    if set(confirmed) != invites | qualifiers:
                        raise ValueError("The field disagrees with invites and previous qualifiers.")
            elif matches:
                raise ValueError("Results are present without a confirmed field.")
        except ValueError as exc:
            raise ValueError(f"Invalid Singapore Stage {n} results: {exc}") from exc
        done = confirmed is not None and _finished(confirmed, matches)
        status = "complete" if done else "live" if matches else "upcoming"
        state[n] = {"seeds": confirmed, "matches": matches, "status": status,
                    "start_date": st["start_date"]}
    state["playoffs"] = parse_playoffs(pages[PLAYOFF_PAGE]["content"] or "", tmap, lp)
    return state


def parse_playoffs(text: str, tmap: dict, lp: dict) -> dict:
    """Bracket slots -> {a, b, winner, date} (display names; winner None until played)."""
    text = COMMENT.sub("", text)
    finished = {m["slot"]: m for m in parse_matches(text)[0] if m["slot"]}
    slots = {}
    for start, _, inner in find_templates(text, "Match"):
        slot = bracket_slot(text, start)
        if slot not in PLAYOFF_SLOTS:
            continue
        _, p = split_params(inner)
        codes = [opponent(p.get(k, ""))[0] for k in ("opponent1", "opponent2")]
        names = [DISPLAY.get(to_vrs(c, None, tmap, lp)[0].casefold()) if c else None for c in codes]
        f = finished.get(slot)
        winner = names[f["winner"] - 1] if f and all(names) else None
        slots[slot] = {"a": names[0], "b": names[1], "winner": winner, "date": f["date"] if f else None}
    played = sum(1 for v in slots.values() if v["winner"])
    status = "complete" if slots.get("R3M1", {}).get("winner") else "live" if played else "upcoming"
    return {"slots": slots, "status": status}


def _finished(seeds: list[str], matches: list[dict]) -> bool:
    w = {t: 0 for t in seeds}
    l = {t: 0 for t in seeds}
    for m in matches:
        w[m["winner"]] += 1
        l[m["b"] if m["winner"] == m["a"] else m["a"]] += 1
    return all(w[t] == 3 or l[t] == 3 for t in seeds)


def fixed_rounds(seeds: list[str], matches: list[dict]) -> list[list[tuple[int, int, int]]]:
    """Validate a Swiss result prefix and translate teams to simulation indices."""
    idx = {t: i for i, t in enumerate(seeds)}
    if len(seeds) != 16 or len(idx) != 16:
        raise ValueError("The Swiss field must contain 16 distinct teams.")

    rounds: list[list[tuple[int, int, int]]] = [[] for _ in range(5)]
    for m in matches:
        rnd, a, b, winner = m["round"], m["a"], m["b"], m["winner"]
        if type(rnd) is not int or not 1 <= rnd <= 5:
            raise ValueError("A match is missing a valid Swiss round.")
        if a not in idx or b not in idx or a == b or winner not in (a, b):
            raise ValueError(f"Round {rnd} has invalid teams or winner: {a}, {b}, {winner}.")
        if rnd == 1 and abs(idx[a] - idx[b]) != 8:
            raise ValueError(f"Round 1 pairing disagrees with the seeds: {a}, {b}.")
        rounds[rnd - 1].append((idx[a], idx[b], idx[winner]))
    while rounds and not rounds[-1]:
        rounds.pop()

    wins, losses = [0] * 16, [0] * 16
    for rnd, fixtures in enumerate(rounds, start=1):
        alive = {i for i in range(16) if wins[i] < 3 and losses[i] < 3}
        seen: set[int] = set()
        for a, b, winner in fixtures:
            if a in seen or b in seen:
                raise ValueError(f"A team appears more than once in round {rnd}.")
            if a not in alive or b not in alive or (wins[a], losses[a]) != (wins[b], losses[b]):
                raise ValueError(f"Round {rnd} pairs teams with incompatible Swiss records.")
            seen.update((a, b))
            wins[winner] += 1
            losses[b if winner == a else a] += 1
        if rnd < len(rounds) and seen != alive:
            raise ValueError(f"Round {rnd} is incomplete but results appear in a later round.")
    return rounds


def major_matches(state: dict) -> pd.DataFrame:
    """Finished Major matches as rating-update rows (VRS names)."""
    vrs = dict(ALL_TEAMS)
    rows = [{"date": m["date"], "winner": vrs[m["winner"]], "loser": vrs[m["b"] if m["winner"] == m["a"] else m["a"]]}
            for s, st in state.items() if s != "playoffs" for m in st["matches"] if m["date"]]
    for m in state.get("playoffs", {}).get("slots", {}).values():
        if m["winner"] and m["date"]:
            rows.append({"date": m["date"], "winner": vrs[m["winner"]], "loser": vrs[m["b"] if m["winner"] == m["a"] else m["a"]]})
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
    fixed = {s: fixed_rounds(state[s]["seeds"], state[s]["matches"]) if state[s]["seeds"] else [] for s in (1, 2, 3)}
    po = state.get("playoffs", {"slots": {}})["slots"]
    real_qf = [(po[k]["a"], po[k]["b"]) for k in PLAYOFF_SLOTS[:4] if k in po and po[k]["a"] and po[k]["b"]]
    known = {k: col[v["winner"]] for k, v in po.items() if v["winner"]}
    p5 = bo5_from_bo3(Pv)
    for i in range(n):
        qualifiers: list[str] = []
        for s in (1, 2, 3):
            if state[s]["seeds"]:
                field = state[s]["seeds"]
            else:
                field = [d for d, _ in INVITES[s]] + qualifiers
            ix = [col[t] for t in field]
            sub = Pv[np.ix_(ix, ix)]  # raw odds for Bo1 as well as Bo3 (see run.stage_block)
            c, order = run_stage(sub, rng, all_bo3=(s == 3), fixed_rounds=fixed[s], p_bo1=sub)
            codes[s][i, ix] = c
            qualifiers = [field[t] for t in order if c[t] in (ADV_30, ADV)]
        ps = [col[t] for t in qualifiers]  # Stage 3 standings order = playoff seeds
        if len(real_qf) == 4:  # bracket published: lay real pairings onto the QF_SEEDS slots
            ps = [0] * 8
            for (sa, sb), (a, b) in zip(QF_SEEDS, real_qf):
                ps[sa], ps[sb] = col[a], col[b]
        playoff_seeds[i] = ps

        def play(slot: str, a: int, b: int, p: np.ndarray) -> int:
            if slot in known and known[slot] in (a, b):
                return known[slot]
            return a if rng.random() < p[a, b] else b

        qf = [play(PLAYOFF_SLOTS[k], ps[a], ps[b], Pv) for k, (a, b) in enumerate(QF_SEEDS)]
        sf = [play("R2M1", qf[0], qf[1], Pv), play("R2M2", qf[2], qf[3], Pv)]
        fin = play("R3M1", sf[0], sf[1], p5)
        winners[i] = qf + sf + [fin]
    return {"names": names, "codes": codes, "playoff_seeds": playoff_seeds, "winners": winners}
