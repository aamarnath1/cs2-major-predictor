"""End-to-end: live state -> ratings -> stage/playoff sims -> Pick'em picks -> web/data.json.

Matchup probability = logit average of four raters (Elo, Glicko-2, TrueSkill, Bradley-Terry).
The model zoo (cs2p/zoo.py) found no statistically significant difference between raters,
so the ensemble is used for robustness.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cs2p import teams
from cs2p.bt import fit_bt
from cs2p.elo import PROC, load_matches, win_prob
from cs2p.features import build_features, ts_win_prob, TS
from cs2p.live import fetch_state, fixed_rounds, major_matches, simulate_major
from cs2p.pickem import NEED, optimise, score
from cs2p.playoffs import all_brackets
from cs2p.swiss import ADV, ADV_30, ELIM_03, bo1_from_bo3, simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WEB = ROOT / "web"
AS_OF = max(pd.Timestamp("2026-11-24"), pd.Timestamp.now().normalize() + pd.Timedelta(days=1))
LABELS = {ADV_30: "3-0", ADV: "3-1/3-2", ELIM_03: "0-3"}
ALL_TEAMS = teams.STAGE1 + teams.STAGE2_INVITES + teams.STAGE3_INVITES
RATERS = ("Elo", "Glicko-2", "TrueSkill", "Bradley-Terry")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def matchup_matrix(
    m: pd.DataFrame,
    roster: list[tuple[str, str]] = ALL_TEAMS,
    as_of: pd.Timestamp = AS_OF,
    stand_ins: dict[str, int] = teams.STAND_INS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """P(row team beats column team) in a Bo3 using only matches before `as_of`; plus each rater's matrix.

    `roster` is a list of (display name, VRS name).
    """
    m = m[m.date < as_of]
    _, state = build_features(m, start=as_of)
    elo, gl, ts = state["elo"], state["glicko"], state["ts"]
    btp = json.loads((PROC / "bt_params.json").read_text())
    bt = fit_bt(m, as_of, btp["half_life_days"], btp["prior"])
    names = [d for d, _ in roster]
    keys = [v.casefold() for _, v in roster]
    e = np.array([elo.current(v, as_of, stand_ins.get(d, 0)) for d, v in roster])
    b = np.array([bt.get(k, 0.0) for k in keys])
    n = len(names)
    mats = {r: np.full((n, n), 0.5) for r in RATERS}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            mats["Elo"][i, j] = win_prob(e[i], e[j], elo.p.scale)
            mats["Glicko-2"][i, j] = gl.expect(keys[i], keys[j], as_of)[0]
            mats["TrueSkill"][i, j] = ts_win_prob(ts.get(keys[i], TS.create_rating()), ts.get(keys[j], TS.create_rating()))
            mats["Bradley-Terry"][i, j] = 1 / (1 + np.exp(b[j] - b[i]))
    ens = 1 / (1 + np.exp(-np.mean([logit(mats[r]) for r in RATERS], axis=0)))
    np.fill_diagonal(ens, 0.5)
    wrap = lambda a: pd.DataFrame(a, index=names, columns=names)  # noqa: E731
    return wrap(ens), {r: wrap(a) for r, a in mats.items()}


def power_rating(P: pd.DataFrame) -> pd.Series:
    """Average win probability vs the other 31 Major teams (a field-relative strength)."""
    return ((P.sum(1) - 0.5) / (len(P) - 1)).sort_values(ascending=False)


def actual_records(seeds: list[str], matches: list[dict]) -> dict[str, list[int]]:
    rec = {t: [0, 0] for t in seeds}
    for m in matches:
        loser = m["b"] if m["winner"] == m["a"] else m["a"]
        rec[m["winner"]][0] += 1
        rec[loser][1] += 1
    return rec


def pick_status(slot: str, w: int, l: int) -> str:
    """'hit', 'miss' or 'open' for one Swiss pick given the team's current record."""
    if slot == "3-0":
        return "miss" if l > 0 else "hit" if w == 3 else "open"
    if slot == "0-3":
        return "miss" if w > 0 else "hit" if l == 3 else "open"
    if w == 3:
        return "hit" if l > 0 else "miss"
    return "miss" if l == 3 else "open"


def stage_block(s: int, st: dict, P: pd.DataFrame, per_rater: dict, major: dict, n_stage: int) -> dict:
    """Odds, picks and (once started) live pick scoring for one Swiss stage."""
    seeds_known = st["seeds"] is not None
    fr: list = []
    visits, round1 = None, []
    if seeds_known:
        cols = st["seeds"]
        fr = fixed_rounds(cols, st["matches"])
        visits = np.zeros((16, 4, 4))
        Ps = P.loc[cols, cols].values
        outcomes = simulate(Ps, n_sims=n_stage, seed=s, all_bo3=(s == 3), fixed_rounds=fr, visits=visits)
        in_stage = np.ones(len(cols))
        p1 = Ps if s == 3 else bo1_from_bo3(Ps)  # round 1 is Bo1 in Stages 1-2, Bo3 in Stage 3
        round1 = [{"a": cols[i], "b": cols[i + 8], "p_a": float(p1[i, i + 8])} for i in range(8)]
    else:  # field not set yet: preview from full-Major sims (teams may or may not reach this stage)
        cm = major["codes"][s]
        keep = (cm >= 0).mean(0) > 0
        cols = [t for t, k in zip(major["names"], keep) if k]
        outcomes = cm[:, keep]
        in_stage = (outcomes >= 0).mean(0)

    picks_path = OUT / f"picks_stage{s}.json"
    if st["status"] == "upcoming" or not picks_path.exists():
        picks, _ = optimise(outcomes)
        pick_names = {LABELS[c]: [cols[t] for t in ts] for c, ts in picks.items()}
        if seeds_known and st["status"] == "upcoming":
            picks_path.write_text(json.dumps({"picks": pick_names, "frozen_at": str(pd.Timestamp.now())}, indent=2))
    else:  # stage has started: keep the picks we committed to
        pick_names = json.loads(picks_path.read_text())["picks"]
    idx = {t: i for i, t in enumerate(cols)}
    code_of = {v: k for k, v in LABELS.items()}
    picks = {code_of[k]: [idx[t] for t in v if t in idx] for k, v in pick_names.items()}
    sc = score(outcomes, picks)
    dist = np.bincount(sc, minlength=11)[:11] / len(sc)

    rec = actual_records(st["seeds"], st["matches"]) if seeds_known else {}
    rows = []
    for i, t in enumerate(cols):
        o = outcomes[:, i]
        row = {"team": t, "seed": (i + 1) if seeds_known else None, "p_in_stage": float(in_stage[i]),
               "p_3_0": float((o == ADV_30).mean()), "p_advance_31_32": float((o == ADV).mean()),
               "p_0_3": float((o == ELIM_03).mean())}
        row["p_qualify"] = row["p_3_0"] + row["p_advance_31_32"]
        if t in rec:
            row["record"] = rec[t]
        rows.append(row)
    if seeds_known:  # per-rater qualify odds show where the models disagree
        for r, Pr in per_rater.items():
            o = simulate(Pr.loc[cols, cols].values, n_sims=3000, seed=7, all_bo3=(s == 3), fixed_rounds=fr)
            for i, row in enumerate(rows):
                row[f"q_{r}"] = float(((o[:, i] == ADV_30) | (o[:, i] == ADV)).mean())

    status = {}
    if rec:
        status = {slot: {t: pick_status(slot, *rec[t]) for t in ts} for slot, ts in pick_names.items()}
    return {
        "stage": s, "status": st["status"], "seeds_known": seeds_known, "start_date": st["start_date"],
        "bo": "All matches Bo3" if s == 3 else "Bo1, with Bo3 for advancement and elimination matches",
        "teams": sorted(rows, key=lambda r: -r["p_qualify"]),
        "picks": pick_names, "p_at_least_5": float((sc >= NEED).mean()), "expected_correct": float(sc.mean()),
        "score_dist": dist.tolist(), "pick_status": status, "matches": st["matches"],
        "round1": round1, "visits": visits.round(4).tolist() if visits is not None else None,
        "outcome_cols": cols, "outcomes": outcomes[:3000].tolist(),
    }


def playoff_block(major: dict, po: dict) -> dict:
    """Bracket picks (frozen once the playoffs start) and live ✓/✗ against real results."""
    names, seeds, winners = major["names"], major["playoff_seeds"], major["winners"]
    fixed = bool((seeds == seeds[0]).all())
    out = {"seeds_known": fixed, "status": po["status"], "slots": po["slots"]}
    if not fixed:
        return out
    sd = seeds[0]
    path = OUT / "picks_playoffs.json"
    if po["status"] == "upcoming" or not path.exists():
        best, best_exp = None, -1.0
        for b in all_brackets():
            c = (winners == sd[np.array(b)]).mean(0).sum()
            if c > best_exp:
                best, best_exp = b, c
        picks = [names[sd[j]] for j in best]
        if po["status"] == "upcoming":
            path.write_text(json.dumps({"picks": picks, "frozen_at": str(pd.Timestamp.now())}, indent=2))
    else:
        picks = json.loads(path.read_text())["picks"]
    col = {t: i for i, t in enumerate(names)}
    c = (winners == np.array([col[t] for t in picks])).sum(1)
    order = ["R1M1", "R1M2", "R1M3", "R1M4", "R2M1", "R2M2", "R3M1"]
    lost = {v["b"] if v["winner"] == v["a"] else v["a"] for v in po["slots"].values() if v["winner"]}
    status = []
    for slot, t in zip(order, picks):
        w = po["slots"].get(slot, {}).get("winner")
        status.append("hit" if w == t else "miss" if (w or t in lost) else "open")
    out.update(seeds=[names[i] for i in sd], picks=picks, pick_status=status,
               expected_correct=float(c.mean()), score_dist=(np.bincount(c, minlength=8) / len(c)).tolist())
    return out


def reach_table(major: dict) -> pd.DataFrame:
    names, n = major["names"], len(major["winners"])

    def share(arr: np.ndarray) -> pd.Series:
        return pd.Series(np.bincount(arr.ravel(), minlength=len(names)) / n, index=names)

    df = pd.DataFrame({
        "stage2": pd.Series((major["codes"][2] >= 0).mean(0), index=names),
        "stage3": pd.Series((major["codes"][3] >= 0).mean(0), index=names),
        "playoffs": share(major["playoff_seeds"]),
        "semis": share(major["winners"][:, :4]),
        "final": share(major["winners"][:, 4:6]),
        "champion": share(major["winners"][:, 6]),
    })
    return df.sort_values("champion", ascending=False)


def main(n_stage: int = 20_000, n_major: int = 5_000) -> None:
    OUT.mkdir(exist_ok=True)
    state = fetch_state()
    m = load_matches()
    mm = major_matches(state)
    if len(mm):
        # Append only Major results not already present (never dedupe history: same-day rematches are real).
        have = set(zip(m.date, m.winner, m.loser))
        mm = mm[[k not in have for k in zip(mm.date, mm.winner, mm.loser)]]
        m = pd.concat([m, mm], ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
    P, per_rater = matchup_matrix(m)
    major = simulate_major(P, state, n_major)

    stages = [stage_block(s, state[s], P, per_rater, major, n_stage) for s in (1, 2, 3)]
    playoffs = playoff_block(major, state["playoffs"])
    reach = reach_table(major)
    reach.to_csv(OUT / "major_odds.csv", index_label="team")
    P.to_csv(OUT / "matchups.csv")
    export(m, P, stages, playoffs, reach)

    for b in stages:
        print(f"Stage {b['stage']} [{b['status']}, seeds {'known' if b['seeds_known'] else 'projected'}] "
              f"P(>=5)={b['p_at_least_5']:.3f} exp={b['expected_correct']:.2f} picks={b['picks']}")
    print(reach.head(8).round(3).to_string())


def export(m: pd.DataFrame, P: pd.DataFrame, stages: list, playoffs: dict, reach: pd.DataFrame) -> None:
    """Write everything the static site needs into web/data.json."""
    WEB.mkdir(exist_ok=True)

    def load(p: Path) -> dict | None:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    sentiment = load(OUT / "sentiment.json")
    if sentiment:
        for t in sentiment.get("teams", {}).values():  # publish aggregates only, never third-party post text
            t.pop("examples", None)
    entry = {d: s for s, lst in (("Stage 1", teams.STAGE1), ("Stage 2", teams.STAGE2_INVITES),
                                 ("Stage 3", teams.STAGE3_INVITES)) for d, _ in lst}
    data = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data_through": str(m.date.max().date()),
        "n_matches": len(m),
        "stages": stages,
        "playoffs": playoffs,
        "major": [{"team": t, "entry": entry[t], **{k: round(float(v), 4) for k, v in r.items()}}
                  for t, r in reach.iterrows()],
        "power": [{"team": t, "power": round(float(v), 4), "entry": entry[t]} for t, v in power_rating(P).items()],
        "matchups": {"teams": list(P.index), "p": P.round(3).values.tolist()},
        "zoo": load(PROC / "zoo.json"),
        "replay": load(PROC / "replay.json"),
        "maps": load(PROC / "map_backtest.json"),
        "sentiment": sentiment,
        "stand_ins": teams.STAND_INS,
    }
    (WEB / "data.json").write_text(json.dumps(data))


if __name__ == "__main__":
    main()
