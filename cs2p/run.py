"""End-to-end: ratings -> Stage 1 sim -> Pick'em picks -> full-Major odds -> web/data.json.

Matchup probability = logit average of four raters (Elo, Glicko-2, TrueSkill, Bradley-Terry).
The model zoo (cs2p/zoo.py) found no statistically significant difference between raters,
so the ensemble is used for robustness.
"""

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cs2p import teams
from cs2p.bt import fit_bt
from cs2p.elo import PROC, load_matches, win_prob
from cs2p.features import build_features, ts_win_prob, TS
from cs2p.pickem import optimise, score
from cs2p.swiss import ADV, ADV_30, ELIM_03, simulate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WEB = ROOT / "web"
AS_OF = pd.Timestamp("2026-11-24")
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


@functools.cache
def _latest_vrs() -> dict[str, float]:
    pts = pd.read_csv(PROC / "vrs_points.csv", parse_dates=["snapshot"])
    return pts[pts.snapshot == pts.snapshot.max()].groupby("team").points.max().to_dict()


def vrs_order(names: list[str]) -> list[str]:
    last = _latest_vrs()
    vrs_name = dict(ALL_TEAMS)
    return sorted(names, key=lambda t: -last.get(vrs_name[t], 0))


def power_rating(P: pd.DataFrame) -> pd.Series:
    """Average win probability vs the other 31 Major teams (a field-relative strength)."""
    return ((P.sum(1) - 0.5) / (len(P) - 1)).sort_values(ascending=False)


def full_major(P: pd.DataFrame, n: int, seed: int = 1) -> pd.DataFrame:
    """Chain Stage 1 -> 2 -> 3 -> playoffs. Stage 2/3 seeded by VRS order (approximation)."""
    rng = np.random.default_rng(seed)
    reach = {k: pd.Series(0.0, index=P.index) for k in ("stage2", "stage3", "playoffs", "semis", "final", "champion")}
    for i in range(n):
        field = [d for d, _ in teams.STAGE1]
        for stage, invites in (("stage2", teams.STAGE2_INVITES), ("stage3", teams.STAGE3_INVITES), ("playoffs", None)):
            res = simulate(P.loc[field, field].values, n_sims=1, seed=seed * 1_000_003 + i * 7 + len(field))[0]
            adv = [field[t] for t in range(16) if res[t] == ADV_30] + [field[t] for t in range(16) if res[t] == ADV]
            reach[stage][adv] += 1
            if invites is None:
                break
            field = vrs_order(adv + [d for d, _ in invites])
        order = [adv[j] for j in (0, 7, 3, 4, 1, 6, 2, 5)]
        for stage in ("semis", "final", "champion"):
            order = [a if rng.random() < P.loc[a, b] else b for a, b in zip(order[::2], order[1::2])]
            reach[stage][order] += 1
    out = pd.DataFrame(reach) / n
    out.loc[[d for d, _ in teams.STAGE2_INVITES + teams.STAGE3_INVITES], "stage2"] = 1.0
    out.loc[[d for d, _ in teams.STAGE3_INVITES], "stage3"] = 1.0
    return out.sort_values("champion", ascending=False)


def main(n_stage: int = 20_000, n_major: int = 4_000) -> None:
    OUT.mkdir(exist_ok=True)
    m = load_matches()
    P, per_rater = matchup_matrix(m)

    names = [d for d, _ in teams.STAGE1]
    outcomes = simulate(P.loc[names, names].values, n_sims=n_stage)
    picks, (p5, exp) = optimise(outcomes)
    np.save(OUT / "stage1_outcomes.npy", outcomes)

    power = power_rating(P)
    probs = pd.DataFrame({
        "seed": range(1, 17),
        "team": names,
        "power": power[names].values,
        "p_3_0": (outcomes == ADV_30).mean(0),
        "p_advance_31_32": (outcomes == ADV).mean(0),
        "p_0_3": (outcomes == ELIM_03).mean(0),
    })
    probs["p_qualify"] = probs.p_3_0 + probs.p_advance_31_32
    # Per-rater qualify odds: shows where the models disagree.
    for r, Pr in per_rater.items():
        o = simulate(Pr.loc[names, names].values, n_sims=4000, seed=7)
        probs[f"q_{r}"] = ((o == ADV_30) | (o == ADV)).mean(0)
    probs.to_csv(OUT / "stage1_probs.csv", index=False)

    pick_out = {LABELS[c]: [names[t] for t in ts] for c, ts in picks.items()}
    dist = np.bincount(score(outcomes, picks), minlength=11) / n_stage
    summary = {"picks": pick_out, "p_at_least_5": p5, "expected_correct": exp, "n_sims": n_stage,
               "score_dist": dist.tolist()}
    (OUT / "stage1_picks.json").write_text(json.dumps(summary, indent=2))

    major = full_major(P, n_major)
    major.to_csv(OUT / "major_odds.csv", index_label="team")
    P.to_csv(OUT / "matchups.csv")

    export(m, P, probs, summary, major, power, outcomes)
    print(probs.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "score_dist"}, indent=2))
    print(major.head(10).round(3).to_string())


def export(m, P, probs, summary, major, power, outcomes) -> None:
    """Write everything the static site needs into web/data.json."""
    WEB.mkdir(exist_ok=True)
    zoo = json.loads((PROC / "zoo.json").read_text()) if (PROC / "zoo.json").exists() else None
    sentiment = None
    if (OUT / "sentiment.json").exists():
        sentiment = json.loads((OUT / "sentiment.json").read_text(encoding="utf-8"))
        for t in sentiment.get("teams", {}).values():  # publish aggregates only, never third-party post text
            t.pop("examples", None)
    stage_of = {d: s for s, lst in (("Stage 1", teams.STAGE1), ("Stage 2", teams.STAGE2_INVITES),
                                    ("Stage 3", teams.STAGE3_INVITES)) for d, _ in lst}
    data = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data_through": str(m.date.max().date()),
        "n_matches": len(m),
        "stage1": {
            "teams": probs.round(4).to_dict(orient="records"),
            "picks": summary,
            # Compact outcomes sample for in-browser scoring of custom picks.
            "outcomes": outcomes[:5000].tolist(),
        },
        "major": [{"team": t, "entry": stage_of[t], **{k: round(float(v), 4) for k, v in r.items()}}
                  for t, r in major.iterrows()],
        "power": [{"team": t, "power": round(float(v), 4), "entry": stage_of[t]} for t, v in power.items()],
        "matchups": {"teams": list(P.index), "p": P.round(3).values.tolist()},
        "zoo": zoo,
        "sentiment": sentiment,
        "replay": json.loads((PROC / "replay.json").read_text()) if (PROC / "replay.json").exists() else None,
        "stand_ins": teams.STAND_INS,
    }
    (WEB / "data.json").write_text(json.dumps(data))


if __name__ == "__main__":
    main()
