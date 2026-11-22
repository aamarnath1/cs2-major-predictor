"""Pick'em replay: what would each strategy have scored at past Major Swiss stages?

For every stage in data/processed/past_major_stages.json, ratings are rebuilt from matches
strictly before the stage start, picks are optimised, then scored against the real records.
"""

import json

import numpy as np
import pandas as pd

from cs2p.elo import PROC, load_matches
from cs2p.pickem import NEED, optimise
from cs2p.run import RATERS, matchup_matrix
from cs2p.swiss import ADV, ADV_30, ELIM, ELIM_03, simulate


def actual_codes(teams: list[dict]) -> np.ndarray:
    def code(t: dict) -> int:
        if t["wins"] == 3:
            return ADV_30 if t["losses"] == 0 else ADV
        return ELIM_03 if t["wins"] == 0 else ELIM

    return np.array([code(t) for t in teams])


def seed_picks() -> dict[int, list[int]]:
    """Naive baseline: top 2 seeds 3-0, seeds 3-8 advance, bottom 2 seeds 0-3."""
    return {ADV_30: [0, 1], ADV: [2, 3, 4, 5, 6, 7], ELIM_03: [14, 15]}


def correct(picks: dict[int, list[int]], actual: np.ndarray) -> int:
    return int(sum((actual[ts] == code).sum() for code, ts in picks.items()))


def main(n_sims: int = 10_000) -> None:
    stages = json.loads((PROC / "past_major_stages.json").read_text())
    m = load_matches()
    rows = []
    for st in stages:
        teams = sorted(st["teams"], key=lambda t: t["seed"])
        if len(teams) != 16:
            print(f"skip {st['event']} {st['stage']}: {len(teams)} teams")
            continue
        start = pd.Timestamp(st["start_date"])
        roster = [(t["name"], t["name"]) for t in teams]
        P, per = matchup_matrix(m, roster, start, {})
        actual = actual_codes(teams)
        strategies = {"Ensemble (optimised)": P, **{f"{r} (optimised)": per[r] for r in RATERS}}
        res = {"event": st["event"], "stage": st["stage"], "start": str(start.date())}
        for name, Pm in strategies.items():
            out = simulate(Pm.values, n_sims=n_sims, seed=11)
            picks, (p5, _) = optimise(out)
            res[name] = correct(picks, actual)
            if name.startswith("Ensemble"):
                res["model_p5"] = p5
                res["picks"] = {str(c): [teams[i]["name"] for i in ts] for c, ts in picks.items()}
        res["Seed baseline"] = correct(seed_picks(), actual)
        rows.append(res)
        print(res["event"], res["stage"], {k: v for k, v in res.items() if isinstance(v, int)})

    df = pd.DataFrame(rows)
    strat_cols = [c for c in df.columns if c.endswith("(optimised)") or c == "Seed baseline"]
    summary = {c: {"mean_correct": float(df[c].mean()), "coins": int((df[c] >= NEED).sum()), "stages": len(df)}
               for c in strat_cols}
    print(pd.DataFrame(summary).T.round(2).to_string())
    (PROC / "replay.json").write_text(json.dumps({"stages": rows, "summary": summary}, indent=2, default=str))


if __name__ == "__main__":
    main()
