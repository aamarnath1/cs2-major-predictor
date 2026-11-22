"""Compare Bo1 conversions on historical fixtures and score the frozen Singapore forecast."""

import argparse
import json
from datetime import datetime
from hashlib import sha256
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from cs2p import teams
from cs2p.elo import PROC, load_matches
from cs2p.replay import MIN_HISTORY_MONTHS, swiss_format
from cs2p.run import OUT, PICK_CUTOFFS, logit, matchup_matrix
from cs2p.swiss import bo1_from_bo3
from cs2p.zoo import metrics, sig

DEVELOPMENT_FROM = pd.Timestamp("2026-06-01")
HOLDOUT_FROM = pd.Timestamp("2026-06-22")
THROUGH = pd.Timestamp("2026-11-22")
ROUND1_VARIANTS = ("converted", "raw", "calibrated")
LOGLOSS_FLOOR = 1e-6


def fixture_predictions(
    history: pd.DataFrame, fixtures: pd.DataFrame, start: pd.Timestamp
) -> pd.DataFrame:
    """Predict all fixtures from the same pre-event or pre-stage team strengths."""
    names = sorted(set(fixtures.winner) | set(fixtures.loser))
    matrix, _ = matchup_matrix(history, [(t, t) for t in names], start, {})
    earlier = history[history.date < start]
    seen = set(earlier.winner.str.casefold()) | set(earlier.loser.str.casefold())
    both_seen = fixtures.winner.str.casefold().isin(seen) & fixtures.loser.str.casefold().isin(seen)
    return pd.DataFrame({
        "p": [matrix.at[w, l] for w, l in zip(fixtures.winner, fixtures.loser)],
        "both_seen": both_seen,
    }, index=fixtures.index)


def major_predictions(history: pd.DataFrame, stages: list[dict]) -> pd.DataFrame:
    """Collect Bo1 predictions for historical Major stages with sufficient history."""
    forecasts = []
    earliest = history.date.min() + pd.DateOffset(months=MIN_HISTORY_MONTHS)
    for stage in stages:
        start = pd.Timestamp(stage["start_date"])
        if not earliest <= start < HOLDOUT_FROM or swiss_format(stage) == "all_bo3":
            continue
        fixtures = pd.DataFrame(m for m in stage["matches"] if m["best_of"] == 1)
        fixtures["loser"] = np.where(
            fixtures.winner == fixtures.team1, fixtures.team2, fixtures.team1
        )
        forecasts.append(
            fixture_predictions(history, fixtures, start).assign(event=stage["event"], start=start)
        )
    return pd.concat(forecasts, ignore_index=True)


def later_predictions(history: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Score later S/A/B-tier events from ratings frozen before their first cached match."""
    forecasts = []
    for event, event_matches in matches.groupby("event"):
        start = event_matches.date.min()
        if start < HOLDOUT_FROM:
            continue
        fixtures = event_matches[
            event_matches.best_of.eq(1) & event_matches.tier.isin(["S-Tier", "A-Tier", "B-Tier"])
        ].drop_duplicates(subset=["date", "winner", "loser", "winner_score", "loser_score", "maps"])
        if len(fixtures):
            forecasts.append(
                fixture_predictions(history, fixtures, start).assign(
                    event=event, start=start, tier=fixtures.tier
                )
            )
    return pd.concat(forecasts, ignore_index=True)


def variants(p: np.ndarray, slope: float) -> dict[str, np.ndarray]:
    """Use one base forecast for the live conversion and both challengers."""
    return {"converted": bo1_from_bo3(p), "raw": p, "calibrated": sig(slope * logit(p))}


def summarize(rows: pd.DataFrame, slope: float) -> dict:
    """Aggregate paired scores and resample entire events for descriptive intervals."""
    labels = rows.event.to_numpy()
    p = variants(rows.p.to_numpy(), slope)
    y = np.ones(len(rows))
    # Random orientation is unnecessary for scoring winner probabilities. For readability,
    # report calibration bins by favourite strength and observed favourite win fraction.
    base = -np.log(np.clip(p["converted"], 1e-6, 1))
    scores = {}
    for name, values in p.items():
        scores[name] = {k: v for k, v in metrics(values, y).items() if k in ("logloss", "brier")}
        favourite = np.maximum(values, 1 - values)
        bins = np.minimum((favourite * 10).astype(int), 9)
        scores[name]["calibration"] = [
            {"n": int(mask.sum()), "mean_forecast": float(favourite[mask].mean()),
             "observed_fraction": float(((values[mask] > .5) + .5 * (values[mask] == .5)).mean())}
            for b in range(5, 10) if (mask := bins == b).any()
        ]
        if name != "converted":
            delta = -np.log(np.clip(values, 1e-6, 1)) - base
            groups = [delta[labels == event] for event in np.unique(labels)]
            scores[name]["delta"] = float(delta.mean())
            scores[name]["delta_by_event"] = {
                event: float(group.mean()) for event, group in zip(np.unique(labels), groups)
            }
            if len(groups) > 1:
                draw = np.random.default_rng(304).integers(len(groups), size=(4000, len(groups)))
                totals = np.array([g.sum() for g in groups])
                sizes = np.array([len(g) for g in groups])
                scores[name]["event_bootstrap_95"] = np.quantile(
                    totals[draw].sum(1) / sizes[draw].sum(1), [.025, .975]
                ).tolist()
    return {"matches": len(rows), "both_seen": int(rows.both_seen.sum()),
            "events": rows.event.value_counts().to_dict(), "scores": scores}


def singapore_openers(card_path: Path, slope: float) -> dict:
    """Apply each challenger to the exact same archived Singapore forecast."""
    raw = card_path.read_bytes()
    card = json.loads(raw)
    if datetime.fromisoformat(card["frozen_at"]) >= PICK_CUTOFFS[1]:
        raise ValueError("Singapore challenger requires a pre-cutoff Swiss snapshot.")
    field = {team: i for i, team in enumerate(card["field"])}
    fixtures = card["round1"]
    if len(field) != 16 or len(fixtures) != 8:
        raise ValueError("Singapore challenger requires the complete opening field.")
    matrix = np.array(card["p_bo3"])
    p_bo3 = np.array([matrix[field[m["a"]], field[m["b"]]] for m in fixtures])
    probabilities = variants(p_bo3, slope)
    if not np.allclose(probabilities["converted"], [m["p_a"] for m in fixtures]):
        raise ValueError("Archived opener probabilities differ from the current conversion.")
    return {
        "base_card_sha256": sha256(raw).hexdigest(),
        "base_card_frozen_at": card["frozen_at"],
        "matches": [
            {"a": m["a"], "b": m["b"],
             "p_a": {name: float(p[i]) for name, p in probabilities.items()}}
            for i, m in enumerate(fixtures)
        ],
    }


def fixture_key(match: object) -> frozenset[str]:
    """Match fixtures independent of which team is listed first."""
    if not isinstance(match, dict):
        raise ValueError("A fixture must be an object.")
    a, b = match.get("a"), match.get("b")
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b or a == b:
        raise ValueError("A fixture must name two different teams.")
    return frozenset((a, b))


def score_singapore_round1(forecast_path: Path, results_path: Path) -> dict:
    """Score the frozen forecast only when all eight expected winners are present."""
    forecast_bytes = forecast_path.read_bytes()
    forecast = json.loads(forecast_bytes)
    frozen_at = datetime.fromisoformat(forecast["frozen_at"])
    cutoff = PICK_CUTOFFS[1]
    if frozen_at.tzinfo is None or frozen_at >= cutoff:
        raise ValueError("The forecast must be frozen before the Singapore Stage 1 cutoff.")
    if datetime.fromisoformat(forecast["cutoff"]) != cutoff:
        raise ValueError("The forecast cutoff does not match Singapore Stage 1.")

    field = [name for name, _ in teams.STAGE1]
    expected = {frozenset((a, b)) for a, b in zip(field[:8], field[8:])}
    fixtures = {}
    for match in forecast["matches"]:
        key = fixture_key(match)
        if key not in expected or key in fixtures:
            raise ValueError("The frozen forecast contains a duplicate or unexpected fixture.")
        probabilities = match["p_a"]
        if (
            not isinstance(probabilities, dict)
            or set(probabilities) != set(ROUND1_VARIANTS)
            or any(
                type(p) not in (int, float) or not 0 <= p <= 1 for p in probabilities.values()
            )
        ):
            raise ValueError("Each fixture needs three finite probabilities between zero and one.")
        fixtures[key] = match
    if fixtures.keys() != expected:
        raise ValueError("The frozen forecast must cover all eight Singapore Round 1 fixtures.")

    results_bytes = results_path.read_bytes()
    results = json.loads(results_bytes)
    stages = [
        stage for stage in results["stages"] if isinstance(stage, dict) and stage.get("stage") == 1
    ]
    if len(stages) != 1:
        raise ValueError("The report must contain exactly one Stage 1 results block.")
    winners = {}
    for match in stages[0]["matches"]:
        if not isinstance(match, dict):
            raise ValueError("A Stage 1 result must be an object.")
        round_number = match.get("round")
        if type(round_number) is not int or not 1 <= round_number <= 5:
            raise ValueError("A Stage 1 result has an invalid round number.")
        if round_number != 1:
            continue
        key = fixture_key(match)
        if key not in fixtures or key in winners:
            raise ValueError("Round 1 results contain a duplicate or unexpected fixture.")
        winner = match.get("winner")
        if not isinstance(winner, str) or winner not in key:
            raise ValueError("A Round 1 winner is not one of the fixture's teams.")
        winners[key] = winner

    scored = len(winners) == len(fixtures)
    report = {
        "status": "scored" if scored else "pending",
        "matches_expected": len(fixtures),
        "matches_completed": len(winners),
        "forecast_sha256": sha256(forecast_bytes).hexdigest(),
        "forecast_lf_sha256": sha256(forecast_bytes.replace(b"\r\n", b"\n")).hexdigest(),
        "results_sha256": sha256(results_bytes).hexdigest(),
    }
    if scored:
        report["logloss_floor"] = LOGLOSS_FLOOR
        report["scores"] = {}
        for model in ROUND1_VARIANTS:
            p_win = [
                m["p_a"][model] if winners[key] == m["a"] else 1 - m["p_a"][model]
                for key, m in fixtures.items()
            ]
            report["scores"][model] = {
                "logloss": -sum(log(max(p, LOGLOSS_FLOOR)) for p in p_win) / len(p_win),
                "brier": sum((1 - p) ** 2 for p in p_win) / len(p_win),
            }
    return report


def main(card_path: Path) -> None:
    """Write reproducible evaluation results to stdout."""
    history = load_matches()
    stages_path = PROC / "past_major_stages.json"
    match_path = PROC / "matches_liquipedia.csv"
    stages = json.loads(stages_path.read_bytes())
    recent = pd.read_csv(match_path, parse_dates=["date"], low_memory=False)
    majors = major_predictions(history, stages)
    train = majors[majors.start < DEVELOPMENT_FROM]
    x = logit(train.p.to_numpy())
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("No complete finite Bo1 calibration fixtures.")
    # Each row is oriented to the winner, so this fits a symmetric no-intercept model.
    fit = minimize_scalar(lambda s: np.logaddexp(0, -s * x).mean(), bounds=(0, 3), method="bounded")
    if not fit.success:
        raise RuntimeError(f"Bo1 calibration fit failed: {fit.message}")
    slope = float(fit.x)
    holdout = later_predictions(history, recent[recent.date <= THROUGH])
    result = {
        "train_matches": len(train), "train_events": train.event.value_counts().to_dict(),
        "slope": slope, "holdout_from": str(HOLDOUT_FROM.date()), "through": str(THROUGH.date()),
        "input_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in (stages_path, match_path, PROC / "matches.csv", PROC / "vrs_points.csv",
                         PROC / "elo_params.json", PROC / "bt_params.json")
        },
        "cologne_development": summarize(majors[majors.start >= DEVELOPMENT_FROM], slope),
        "later_s_a": summarize(holdout[holdout.tier.isin(["S-Tier", "A-Tier"])], slope),
        "later_b": summarize(holdout[holdout.tier == "B-Tier"], slope),
        "later_b_seen": summarize(holdout[(holdout.tier == "B-Tier") & holdout.both_seen], slope),
        "singapore_stage1_round1": singapore_openers(card_path, slope),
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=OUT / "picks_stage1.json")
    parser.add_argument("--score-round1", type=Path, metavar="REPORT")
    parser.add_argument("--forecast", type=Path, default=OUT / "bo1_singapore_stage1.json")
    args = parser.parse_args()
    if args.score_round1:
        try:
            report = score_singapore_round1(args.forecast, args.score_round1)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, indent=2))
    else:
        main(args.card)
