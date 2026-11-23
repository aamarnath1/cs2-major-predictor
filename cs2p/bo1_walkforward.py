"""Research-only Bo1 conversion check with frozen pre-event ratings."""

import json
from dataclasses import asdict
from hashlib import sha256

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from cs2p.bo1_experiment import THROUGH, variants
from cs2p.bt import fit_bt
from cs2p.elo import PROC, Elo, EloParams, load_matches, win_prob
from cs2p.features import TS, Glicko2, ts_win_prob
from cs2p.run import logit
from cs2p.zoo import sig

EVAL_FROM = pd.Timestamp("2025-01-01")
MIN_TRAIN_MATCHES = 100
ELO_PARAMS = EloParams(roster_shrink=0.0)
BT_PARAMS = {"half_life_days": 180.0, "prior": 1.0}


def bo1_events(matches: pd.DataFrame, earliest: pd.Timestamp) -> pd.DataFrame:
    """Build event blocks conservatively by grouping published colon-suffix subevents."""
    m = matches.loc[matches.date <= THROUGH].copy()
    m["block"] = m.event.str.split(": ", n=1).str[0]
    dates = m.groupby("block").date.agg(block_start="min", end="max")
    m["start"] = m.groupby("event").date.transform("min")
    fixtures = m.loc[m.best_of.eq(1) & m.tier.isin(["S-Tier", "A-Tier", "B-Tier"])]
    fixtures = fixtures.drop_duplicates(
        subset=["block", "date", "winner", "loser", "winner_score", "loser_score", "maps"]
    ).join(dates, on="block")
    fixtures = fixtures.loc[fixtures.block_start >= earliest].copy()
    fixtures["cohort"] = np.where(fixtures.tier.eq("B-Tier"), "B", "S/A")
    size = fixtures.groupby("block").date.transform("size")
    return fixtures.loc[size >= 2].sort_values(["start", "block", "date", "winner", "loser"])


def predict_events(history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Advance online raters between cutoffs, and refit BT at each cutoff."""
    canonical = history.sort_values(
        ["date", "winner", "loser", "winner_roster", "loser_roster"], kind="stable"
    )
    rows = iter(canonical.itertuples(index=False))
    current = next(rows, None)
    elo, glicko, ts = Elo(ELO_PARAMS), Glicko2(), {}
    forecasts = []
    for cutoff, games in fixtures.groupby("start", sort=True):
        while current is not None and current.date < cutoff:
            w, l = current.winner.casefold(), current.loser.casefold()
            elo.update(w, l, current.date, current.winner_roster, current.loser_roster)
            glicko.update(w, l, current.date)
            ta, tb = ts.get(w, TS.create_rating()), ts.get(l, TS.create_rating())
            (ts[w],), (ts[l],) = TS.rate([(ta,), (tb,)], ranks=[0, 1])
            current = next(rows, None)
        bt = fit_bt(canonical, cutoff, **BT_PARAMS)
        for game in games.itertuples(index=False):
            w, l = game.winner.casefold(), game.loser.casefold()
            p = np.array([
                win_prob(elo.current(w, cutoff), elo.current(l, cutoff), elo.p.scale),
                glicko.expect(w, l, cutoff)[0],
                ts_win_prob(ts.get(w, TS.create_rating()), ts.get(l, TS.create_rating())),
                sig(bt.get(w, 0.0) - bt.get(l, 0.0)),
            ])
            forecasts.append({
                "date": game.date, "block": game.block, "start": cutoff,
                "block_start": game.block_start, "end": game.end,
                "tier": game.tier, "cohort": game.cohort, "p": float(sig(logit(p).mean())),
                "both_seen": w in elo.games and l in elo.games,
            })
    return pd.DataFrame(forecasts)


def score_walkforward(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Fit using blocks whose latest archived match predates the evaluation block."""
    scored, excluded = [], []
    for (event, cohort), rows in forecasts[forecasts.block_start >= EVAL_FROM].groupby(
        ["block", "cohort"], sort=False
    ):
        start = rows.block_start.iloc[0]
        train = forecasts.loc[(forecasts.end < start) & (forecasts.cohort == cohort)]
        if len(train) < MIN_TRAIN_MATCHES:
            excluded.append({"block": event, "cohort": cohort, "training": len(train)})
            continue
        x = logit(train.p.to_numpy())
        fit = minimize_scalar(
            lambda slope: np.logaddexp(0, -slope * x).mean(), bounds=(0, 3), method="bounded"
        )
        if not fit.success:
            raise RuntimeError(f"Bo1 slope fit failed before {event}: {fit.message}")
        probabilities = variants(rows.p.to_numpy(), float(fit.x))
        scored.append(rows.assign(**probabilities, slope=float(fit.x), training=len(train)))
    if not scored:
        raise ValueError("No Bo1 evaluation block has enough earlier calibration fixtures.")
    return pd.concat(scored, ignore_index=True), excluded


def summarize(rows: pd.DataFrame) -> dict:
    """Report matched log-loss and Brier, with event-block resampling."""
    group = rows.block.to_numpy()
    events = np.unique(group)
    result = {"matches": len(rows), "events": len(events), "both_seen": int(rows.both_seen.sum())}
    if rows.empty:
        return result
    horizon = (rows.date - rows.start).dt.days
    sizes = np.array([(group == event).sum() for event in events])
    base = -np.log(rows.converted.clip(1e-6, 1))
    draw = np.random.default_rng(411).integers(len(events), size=(4000, len(events)))
    metrics = {}
    for name in ("converted", "raw", "calibrated"):
        q = rows[name]
        loss = -np.log(q.clip(1e-6, 1))
        delta = loss - base
        event_totals = np.array([delta[group == event].sum() for event in events])
        metrics[name] = {
            "logloss": float(loss.mean()), "brier": float(((1 - q) ** 2).mean()),
            "delta_vs_converted": float(delta.mean()),
            "events_better_than_converted": int((event_totals < 0).sum()),
        }
        if len(events) > 1 and name != "converted":
            draws = event_totals[draw].sum(1) / sizes[draw].sum(1)
            metrics[name]["event_bootstrap_95"] = np.quantile(draws, [.025, .975]).tolist()
    result.update(
        scores=metrics, slope_range=[float(rows.slope.min()), float(rows.slope.max())],
        median_days_after_freeze=float(horizon.median()), max_days_after_freeze=int(horizon.max()),
    )
    return result


def cohorts(scored: pd.DataFrame) -> dict:
    """Break out tier and prior team coverage without changing the matched fixtures."""
    high = scored.cohort.eq("S/A")
    known = scored.both_seen
    checks = {
        "s_a": high, "s_only": scored.tier.eq("S-Tier"), "a_only": scored.tier.eq("A-Tier"),
        "s_a_known": high & known, "b": ~high, "b_known": ~high & known,
        "s_a_2026": high & scored.block_start.ge("2026-01-01"),
    }
    return {name: summarize(scored[mask]) for name, mask in checks.items()}


def main() -> None:
    """Run on the cached history without writing a production artifact."""
    history = load_matches()
    path = PROC / "matches_liquipedia.csv"
    fixtures = bo1_events(
        pd.read_csv(path, parse_dates=["date"], low_memory=False),
        history.date.min() + pd.DateOffset(months=6),
    )
    scored, excluded = score_walkforward(predict_events(history, fixtures))
    coarse_fixtures = fixtures.assign(start=fixtures.block_start)
    coarse, _ = score_walkforward(predict_events(history, coarse_fixtures))
    print(json.dumps({
        "eval_from": str(EVAL_FROM.date()), "through": str(THROUGH.date()),
        "elo_params": asdict(ELO_PARAMS), "bt_params": BT_PARAMS,
        "minimum_prior_cohort_matches": MIN_TRAIN_MATCHES, "min_fixtures_per_block": 2,
        "input_sha256": {
            p.name: sha256(p.read_bytes()).hexdigest() for p in (path, PROC / "matches.csv")
        },
        "excluded_insufficient_training": excluded,
        "results": cohorts(scored),
        "first_parent_block_day_sensitivity": cohorts(coarse),
        "evaluation_blocks": [
            {"block": event, "cohort": cohort, "block_start": str(rows.block_start.iloc[0].date()),
             "matches": len(rows), "slope": rows.slope.iloc[0],
             "training": int(rows.training.iloc[0])}
            for (event, cohort), rows in scored.groupby(["block", "cohort"], sort=False)
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
