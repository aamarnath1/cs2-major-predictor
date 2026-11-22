"""Tune Bradley-Terry params on the tuning window (ends before the model-zoo test window)."""

import glob
import itertools
import json
import re

import numpy as np
import pandas as pd

from cs2p.backtest import TUNE_FROM, TUNE_TO, log_loss, vrs_baseline
from cs2p.bt import fit_bt
from cs2p.elo import PROC, EloParams, load_matches, run


def regions() -> dict[str, str]:
    reg = {}
    for f in sorted(glob.glob(str(PROC.parent / "raw/vrs/live/*/standings_*_*.md"))):
        r = re.search(r"standings_(europe|americas|asia)_", f)
        if r:
            for line in open(f, encoding="utf-8"):
                p = [x.strip() for x in line.split("|")]
                if len(p) > 4 and p[1].isdigit():
                    reg[p[3].casefold()] = r.group(1)
    return reg


def bt_probs(m: pd.DataFrame, mask: np.ndarray, hl: float, prior: float) -> np.ndarray:
    """Refit on the first of each month, predict that month's masked matches."""
    out = np.full(len(m), np.nan)
    months = pd.date_range(TUNE_FROM, TUNE_TO, freq="MS")
    for start, end in zip(months[:-1], months[1:]):
        idx = np.where(mask & (m.date >= start).values & (m.date < end).values)[0]
        if len(idx) == 0:
            continue
        s = fit_bt(m, start, hl, prior)
        w = m.winner.str.casefold().values[idx]
        l = m.loser.str.casefold().values[idx]
        out[idx] = 1 / (1 + np.exp(np.array([s.get(x, 0) for x in l]) - np.array([s.get(x, 0) for x in w])))
    return out


def main() -> None:
    m = load_matches()
    wp, lp = vrs_baseline(m)
    top = (wp >= 1400) & (lp >= 1400)
    mask = top & (m.date >= TUNE_FROM).values & (m.date < TUNE_TO).values
    reg = regions()
    rw, rl = m.winner.str.casefold().map(reg).values, m.loser.str.casefold().map(reg).values
    cross = mask & pd.notna(rw) & pd.notna(rl) & (rw != rl)

    _, pe = run(m, EloParams(**json.loads((PROC / "elo_params.json").read_text())))
    pv = 1 / (1 + 10 ** (-(wp - lp) / 1000))
    print(f"n={mask.sum()} cross-region n={cross.sum()}")
    print(f"Elo  all={log_loss(pe[mask]):.4f} cross={log_loss(pe[cross]):.4f} acc={np.mean(pe[mask] > .5):.3f}")
    print(f"VRS  all={log_loss(pv[mask]):.4f} cross={log_loss(pv[cross]):.4f} acc={np.mean(pv[mask] > .5):.3f}")

    res = []
    for hl, prior in itertools.product([90, 180, 365], [0.5, 2, 5]):
        pb = bt_probs(m, mask, hl, prior)
        res.append((log_loss(pb[mask]), log_loss(pb[cross]), float(np.mean(pb[mask] > .5)), hl, prior))
        print(f"BT hl={hl} prior={prior}  all={res[-1][0]:.4f} cross={res[-1][1]:.4f} acc={res[-1][2]:.3f}")
    best = min(res, key=lambda r: r[1])  # select on cross-region: regional inflation is the risk
    (PROC / "bt_params.json").write_text(json.dumps({"half_life_days": best[3], "prior": best[4]}))
    bt = json.loads((PROC / "backtest.json").read_text())
    bt["bt"] = {"logloss": best[0], "cross_logloss": best[1], "acc": best[2], "half_life_days": best[3], "prior": best[4]}
    bt["elo"]["cross_logloss"] = log_loss(pe[cross])
    bt["vrs"]["cross_logloss"] = log_loss(pv[cross])
    bt["cross_n"] = int(cross.sum())
    # Ensemble used in production: 50/50 logit average of Elo and best BT.
    pb = bt_probs(m, mask, best[3], best[4])
    lg = lambda p: np.log(p / (1 - p))  # noqa: E731
    pens = 1 / (1 + np.exp(-(0.5 * lg(pe) + 0.5 * lg(np.where(np.isnan(pb), 0.5, pb)))))
    bt["ensemble"] = {
        "logloss": log_loss(pens[mask]), "cross_logloss": log_loss(pens[cross]), "acc": float(np.mean(pens[mask] > .5))
    }
    print(f"Ensemble all={bt['ensemble']['logloss']:.4f} cross={bt['ensemble']['cross_logloss']:.4f}")
    (PROC / "backtest.json").write_text(json.dumps(bt, indent=2))


if __name__ == "__main__":
    main()
