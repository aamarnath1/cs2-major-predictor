"""Pick'em replay: what would each strategy have scored at past Major Swiss stages?

For every stage in data/processed/past_major_stages.json, ratings are rebuilt from matches
strictly before the stage start, picks are optimised, then scored against the real records.
Caveat: Elo/BT hyperparameters were tuned on Jul 2025-Feb 2026 data, which overlaps or
postdates some replayed stages. The seed baseline uses no tuning.
"""

import json
from collections import Counter
from typing import Literal

import numpy as np
import pandas as pd

from cs2p.elo import PROC, load_matches
from cs2p.pickem import NEED, optimise
from cs2p.run import RATERS, matchup_matrix
from cs2p.swiss import ADV, ADV_30, ELIM, ELIM_03, simulate


MIN_HISTORY_MONTHS = 6  # ratings need history; the data starts Feb 2024


def swiss_format(stage: dict) -> Literal["mixed", "all_bo3"]:
    """Identify the format from round and series lengths in a complete cached stage."""
    formats = Counter((m.get("round"), m.get("best_of")) for m in stage.get("matches", []))
    if formats == {(1, 3): 8, (2, 3): 8, (3, 3): 8, (4, 3): 6, (5, 3): 3}:
        return "all_bo3"
    if formats == {(1, 1): 8, (2, 1): 8, (3, 1): 4, (3, 3): 4, (4, 3): 6, (5, 3): 3}:
        return "mixed"
    raise ValueError(f"Unknown or incomplete Swiss format for {stage['event']} {stage['stage']}")


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
        if start < m.date.min() + pd.DateOffset(months=MIN_HISTORY_MONTHS):
            print(f"skip {st['event']} {st['stage']}: under {MIN_HISTORY_MONTHS} months of prior data")
            continue
        match_format = swiss_format(st)
        roster = [(t["name"], t["name"]) for t in teams]
        P, per = matchup_matrix(m, roster, start, {})
        actual = actual_codes(teams)
        strategies = {"Ensemble (optimised)": P, **{f"{r} (optimised)": per[r] for r in RATERS}}
        res = {"event": st["event"], "stage": st["stage"], "start": str(start.date()),
               "format": match_format}
        for name, Pm in strategies.items():
            out = simulate(Pm.values, n_sims=n_sims, seed=11, all_bo3=match_format == "all_bo3")
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
    # Paired comparison: ensemble minus seed baseline, per stage, with a bootstrap 95% CI.
    d = (df["Ensemble (optimised)"] - df["Seed baseline"]).to_numpy(dtype=float)
    boot = np.random.default_rng(0).choice(d, (10_000, len(d))).mean(1)
    paired = {"mean_diff": float(d.mean()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
              "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}
    print("ensemble vs seeds:", paired)
    (PROC / "replay.json").write_text(json.dumps({"stages": rows, "summary": summary, "vs_seed": paired},
                                                 indent=2, default=str))


if __name__ == "__main__":
    main()
