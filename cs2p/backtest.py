"""Tune Elo params and compare against Valve's VRS points on recent top-tier matches."""

import itertools
import json

import numpy as np
import pandas as pd

from cs2p.elo import PROC, EloParams, load_matches, run

EVAL_FROM = pd.Timestamp("2026-03-01")
FIT_FROM = pd.Timestamp("2025-03-01")


def fit_logistic(x: np.ndarray, iters: int = 50) -> np.ndarray:
    """No-intercept logistic regression where every row is the winner's perspective (y=1)."""
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        s = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (1 - s)
        hess = -(x * (s * (1 - s))[:, None]).T @ x
        w -= np.linalg.solve(hess, grad)
    return w


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
    mask = (m.date >= EVAL_FROM).values & (wp >= 1400) & (lp >= 1400)
    print(f"eval matches: {mask.sum()} (top-tier, since {EVAL_FROM.date()})")

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

    # Blend: logistic regression on [Elo logit diff, VRS points diff], fit on an earlier window.
    _, p = run(m, EloParams(**best[3]))
    x = np.column_stack([np.log(p / (1 - p)), wp - lp])
    top = (wp >= 1400) & (lp >= 1400)
    fit_mask = top & (m.date >= FIT_FROM).values & (m.date < EVAL_FROM).values
    coef = fit_logistic(x[fit_mask])
    pb = 1 / (1 + np.exp(-x[mask] @ coef))
    blend = {"logloss": log_loss(pb), "brier": brier(pb), "acc": float(np.mean(pb > .5))}
    print(f"Blend coef={coef.round(5)} fit_n={fit_mask.sum()}  logloss={blend['logloss']:.4f}  "
          f"brier={blend['brier']:.4f}  acc={blend['acc']:.3f}")
    # Refit on all top-tier data from FIT_FROM for live use.
    coef_live = fit_logistic(x[top & (m.date >= FIT_FROM).values])
    (PROC / "blend.json").write_text(json.dumps({"elo": coef_live[0], "vrs": coef_live[1]}))
    summary = {
        "eval_n": int(mask.sum()),
        "vrs": {"logloss": best_vrs[1], "brier": brier(pv), "acc": float(np.mean(pv > .5))},
        "elo": {"logloss": best[0], "brier": best[1], "acc": best[2], "params": best[3]},
        "blend": blend,
    }
    (PROC / "backtest.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
