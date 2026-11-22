"""Time-weighted Bradley-Terry ratings, fit globally with the MM algorithm (Hunter 2004).

Unlike sequential Elo, a global fit cannot leak rating points into a closed regional pool:
a pool's level is set only by its results against other pools.
"""

import numpy as np
import pandas as pd


def fit_bt(
    m: pd.DataFrame,
    as_of: pd.Timestamp,
    half_life_days: float = 180.0,
    prior: float = 1.0,
    iters: int = 300,
) -> dict[str, float]:
    """Return {team (casefolded): log-strength} using matches strictly before `as_of`.

    Each team also plays `prior` weighted wins and losses against a virtual team of
    log-strength 0, which regularises teams with few games.
    """
    d = m[m.date < as_of]
    w_name, l_name = d.winner.str.casefold().values, d.loser.str.casefold().values
    names, inv = np.unique(np.concatenate([w_name, l_name]), return_inverse=True)
    wi, li = inv[: len(d)], inv[len(d) :]
    wt = 0.5 ** ((as_of - d.date).dt.days.values / half_life_days)
    n = len(names)
    wins = np.bincount(wi, weights=wt, minlength=n) + prior
    r = np.ones(n)
    for _ in range(iters):
        inv_sum = wt / (r[wi] + r[li])
        denom = np.bincount(wi, inv_sum, n) + np.bincount(li, inv_sum, n) + 2 * prior / (r + 1.0)
        r_new = wins / denom
        if np.max(np.abs(np.log(r_new) - np.log(r))) < 1e-7:
            r = r_new
            break
        r = r_new
    return dict(zip(names, np.log(r)))


def prob(s: dict[str, float], a: str, b: str) -> float:
    da, db = s.get(a.casefold(), 0.0), s.get(b.casefold(), 0.0)
    return 1.0 / (1.0 + np.exp(db - da))
