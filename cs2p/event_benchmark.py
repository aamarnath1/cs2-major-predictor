"""Compare rating models and Pick'em cards on chronological Major folds."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from cs2p.elo import PROC, EloParams, load_matches
from cs2p.pickem import NEED, objective, optimise
from cs2p.replay import MIN_HISTORY_MONTHS, actual_codes, correct, seed_picks, swiss_format
from cs2p.run import logit, matchup_matrix
from cs2p.swiss import simulate
from cs2p.zoo import sig

MODELS = ("Elo", "Elo+BT", "Four raters", "VRS")
N_SIMS = 10_000
PUBLICATION = PROC.parents[1] / "research" / "vrs_snapshot_publication.json"


def fit_slope(winner_logits: np.ndarray) -> float:
    """Fit a symmetric calibration slope from an already completed set of matches."""
    if not len(winner_logits) or not np.isfinite(winner_logits).all():
        raise ValueError("Calibration needs finite earlier match forecasts.")
    fit = minimize_scalar(
        lambda scale: np.logaddexp(0, -scale * winner_logits).mean(),
        bounds=(0, 3), method="bounded",
    )
    if not fit.success:
        raise RuntimeError(f"Calibration failed: {fit.message}")
    return float(fit.x)


def published_vrs(points_path: Path, manifest_path: Path) -> pd.DataFrame:
    """Bind effective ranking dates to the last-changing public Valve commits."""
    manifest = json.loads(manifest_path.read_bytes())
    rows = [
        {
            "snapshot": pd.to_datetime(row["file"], format="standings_global_%Y_%m_%d.md"),
            "published": pd.Timestamp(row["committed_at"]).tz_convert(None).normalize(),
        }
        for row in manifest["files"]
    ]
    points = pd.read_csv(points_path, parse_dates=["snapshot"]).merge(pd.DataFrame(rows))
    points["team"] = points.team.str.casefold()
    points["available"] = points[["snapshot", "published"]].max(axis=1)
    return points


def stage_inputs(history: pd.DataFrame, stage: dict, vrs_history: pd.DataFrame) -> tuple:
    """Build stage-start logits and index the actual fixtures by seed."""
    start = pd.Timestamp(stage["start_date"])
    roster = sorted(stage["teams"], key=lambda t: t["seed"])
    names = [t["name"] for t in roster]
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError(f"Invalid roster for {stage['event']} {stage['stage']}.")
    index = {name: i for i, name in enumerate(names)}
    series, per = matchup_matrix(
        history, [(t, t) for t in names], start, {},
        elo_params=EloParams(roster_shrink=0.0),
        bt_params={"half_life_days": 180.0, "prior": 1.0},
        batch_by_day=True,
    )
    elo, bt = logit(per["Elo"].values), logit(per["Bradley-Terry"].values)
    available = vrs_history[vrs_history.available < start]
    latest = available[available.snapshot == available.snapshot.max()].groupby("team").points.max()
    points = np.array([latest.get(t.casefold(), np.nan) for t in names])
    vrs = (points[:, None] - points[None, :]) * np.log(10) / 400
    logits = {"Elo": elo, "Elo+BT": (elo + bt) / 2, "Four raters": logit(series.values), "VRS": vrs}
    formats = np.array([m["best_of"] for m in stage["matches"]])
    winner, loser = [], []
    for match in stage["matches"]:
        a, b = match["team1"], match["team2"]
        w = match["winner"]
        if w not in (a, b):
            raise ValueError(f"Invalid winner in {stage['event']} {stage['stage']}.")
        winner.append(index[w])
        loser.append(index[b if w == a else a])
    match_logits = {name: value[winner, loser] for name, value in logits.items()}
    return roster, logits, formats, match_logits


def match_scores(rows: pd.DataFrame, names: tuple[str, ...]) -> dict:
    """Score identical complete-case fixtures using probability of the actual winner."""
    scores = {}
    for name in names:
        p = rows[name].to_numpy(dtype=float)
        scores[name] = {
            "logloss": float(-np.log(np.clip(p, 1e-6, 1)).mean()),
            "brier": float(((1 - p) ** 2).mean()),
        }
    return {"matches": len(rows), **scores}


def run_benchmark(history: pd.DataFrame, stages: list[dict], vrs_history: pd.DataFrame) -> dict:
    """Fit slopes on earlier events, then score fixtures and cards on the next Major."""
    earliest = history.date.min() + pd.DateOffset(months=MIN_HISTORY_MONTHS)
    order = sorted(stages, key=lambda row: row["start_date"])
    groups = {}
    for stage in order:
        if pd.Timestamp(stage["start_date"]) >= earliest:
            groups.setdefault(stage["event"], []).append(stage)
    tuning_rows, match_rows, cards, folds = [], [], [], []
    prior_end = None
    for event, block in groups.items():
        start = pd.Timestamp(block[0]["start_date"])
        end = max(pd.Timestamp(m["date"]) for s in block for m in s["matches"])
        if prior_end is not None and prior_end >= start:
            raise ValueError("Major blocks overlap. Check their fixture dates.")
        prior_end = end
        calibration = pd.concat(tuning_rows).dropna() if tuning_rows else None
        slopes = {} if calibration is None else {
            name: {
                fmt: fit_slope(calibration.loc[calibration.best_of == fmt, name].to_numpy())
                for fmt in (1, 3)
            } for name in MODELS
        }
        folds.append({"event": event, "start": str(start.date()), "end": str(end.date()),
                      "training_matches": 0 if calibration is None else len(calibration),
                      "slopes": slopes})
        earlier_stage_rows = []
        for stage in block:
            all_bo3 = swiss_format(stage) == "all_bo3"
            roster, matrices, formats, actual_logits = stage_inputs(history, stage, vrs_history)
            rows = pd.DataFrame({"best_of": formats, **actual_logits})
            earlier_stage_rows.append(rows)
            if calibration is None:
                continue
            forecast = rows[["best_of"]].assign(event=event)
            outcome = actual_codes(roster)
            hits = {"Seeds": correct(seed_picks(), outcome)}
            chance = {}
            for name, matrix in matrices.items():
                p1, p3 = (sig(slopes[name][fmt] * np.nan_to_num(matrix)) for fmt in (1, 3))
                picked, _ = optimise(simulate(p3, N_SIMS, seed=11, all_bo3=all_bo3, p_bo1=p1))
                paths = simulate(p3, N_SIMS, seed=29, all_bo3=all_bo3, p_bo1=p1)
                chance[name] = objective(paths, picked)[0]
                hits[name] = correct(picked, outcome)
                scale = np.where(formats == 1, slopes[name][1], slopes[name][3])
                forecast[name] = sig(scale * actual_logits[name])
            match_rows.append(forecast.assign(**{"50/50": 0.5}))
            cards.append({"event": event, "stage": stage["stage"], "correct": hits,
                          "simulated_p5": chance})
        tuning_rows.extend(earlier_stage_rows)

    predictions = pd.concat(match_rows, ignore_index=True)
    common = predictions.dropna()
    summary = {
        "input_fixtures": len(predictions),
        "vrs_covered": len(common),
        "all_raters": match_scores(predictions, MODELS[:-1] + ("50/50",)),
        "common_fixtures": match_scores(common, MODELS + ("50/50",)),
        "by_format": {
            str(fmt): match_scores(part, MODELS + ("50/50",))
            for fmt, part in common.groupby("best_of")
        },
        "by_event": {
            event: match_scores(part, MODELS + ("50/50",))
            for event, part in common.groupby("event", sort=False)
        },
        "cards": {
            name: {
                "mean_correct": float(np.mean([row["correct"][name] for row in cards])),
                "five_plus": sum(row["correct"][name] >= NEED for row in cards),
                "stages": len(cards),
            } for name in MODELS + ("Seeds",)
        },
    }
    return {"summary": summary, "folds": folds, "stages": cards, "simulations_per_card": N_SIMS}


def main() -> None:
    """Run the retrospective benchmark without updating the live report."""
    history = load_matches()
    stages_path = PROC / "past_major_stages.json"
    stages = json.loads(stages_path.read_bytes())
    results = run_benchmark(history, stages, published_vrs(PROC / "vrs_points.csv", PUBLICATION))
    results["input_sha256"] = {
        filename: sha256((PROC / filename).read_bytes()).hexdigest()
        for filename in ("matches.csv", "matches_liquipedia.csv",
                         "past_major_stages.json", "vrs_points.csv")
    }
    results["input_sha256"][PUBLICATION.name] = sha256(PUBLICATION.read_bytes()).hexdigest()
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
