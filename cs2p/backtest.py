"""Tune Elo params on a tuning window that ends before the model-zoo test window (no leakage)."""

import itertools
import json

import numpy as np
import pandas as pd

from cs2p.elo import PROC, EloParams, load_matches, run

TUNE_FROM = pd.Timestamp("2025-07-01")
TUNE_TO = pd.Timestamp("2026-03-01")  # zoo.py evaluates from here on



def log_loss(p: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p, 1e-6, 1))))


def brier(p: np.ndarray) -> float:
    return float(np.mean((1 - p) ** 2))


def vrs_baseline(m: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Winner points and loser points from the latest VRS snapshot strictly before each match."""
    pts = pd.read_csv(PROC / "vrs_points.csv", parse_dates=["snapshot"])
    snaps = np.sort(pts.snapshot.unique())
    lookup = {(s, t): p for s, t, p in pts.itertuples(index=False)}
    idx = np.searchsorted(snaps, m.date.values, side="left") - 1
    wp, lp = np.full(len(m), np.nan), np.full(len(m), np.nan)
    for i, (j, w, l) in enumerate(zip(idx, m.winner, m.loser)):
        if j >= 0:
            s = pd.Timestamp(snaps[j])
            wp[i], lp[i] = lookup.get((s, w), np.nan), lookup.get((s, l), np.nan)
    return wp, lp


def main() -> None:
    m = load_matches()
    wp, lp = vrs_baseline(m)
    # Top-tier eval set: both teams in VRS with >= 1400 points at the prior snapshot.
    mask = (m.date >= TUNE_FROM).values & (m.date < TUNE_TO).values & (wp >= 1400) & (lp >= 1400)
    print(f"tuning matches: {mask.sum()} (top-tier, {TUNE_FROM.date()} to {TUNE_TO.date()})")

    # VRS baseline: fit logistic scale on points diff (generous to the baseline).
    diff = (wp - lp)[mask]
    best_vrs = min(
        ((s, log_loss(1 / (1 + 10 ** (-diff / s)))) for s in range(100, 2001, 50)), key=lambda x: x[1]
    )
    pv = 1 / (1 + 10 ** (-diff / best_vrs[0]))
    print(f"VRS baseline   scale={best_vrs[0]}  logloss={best_vrs[1]:.4f}  brier={brier(pv):.4f}  acc={np.mean(pv > .5):.3f}")

    grid = {
        "k": [12, 16, 20, 24],
        "roster_shrink": [0.0, 0.05, 0.1],
        "decay_days": [365, 730, 10_000],
        "init": [1300],
    }
    results = []
    for vals in itertools.product(*grid.values()):
        params = EloParams(**dict(zip(grid, vals)))
        _, p = run(m, params)
        results.append((log_loss(p[mask]), brier(p[mask]), float(np.mean(p[mask] > .5)), dict(zip(grid, vals))))
    results.sort(key=lambda r: r[0])
    for ll, br, acc, kw in results[:5]:
        print(f"Elo {kw}  logloss={ll:.4f}  brier={br:.4f}  acc={acc:.3f}")

    best = results[0]
    (PROC / "elo_params.json").write_text(json.dumps(best[3]))

    summary = {
        "eval_n": int(mask.sum()),
        "vrs": {"logloss": best_vrs[1], "brier": brier(pv), "acc": float(np.mean(pv > .5))},
        "elo": {"logloss": best[0], "brier": best[1], "acc": best[2], "params": best[3]},
    }
    (PROC / "backtest.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
