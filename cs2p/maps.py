"""Map-pool model: per-team map strengths + veto simulation.

Model (per map played):  P(a wins map m) = sigmoid(theta_a - theta_b + phi[a,m] - phi[b,m])
fitted by L2-regularised, time-weighted logistic regression on Liquipedia map results.
`phi` offsets get a stronger penalty than `theta` (via column scaling), so teams with few
maps on a given map shrink toward "no map preference".

Veto (Valve Major format), team A = higher-rated side:
  Bo1: A bans 2, B bans 3, A bans 1, last map is played.
  Bo3: A ban, B ban, A pick, B pick, A ban, B ban, last map is the decider.
Each team bans its worst remaining map and picks its best (greedy, by phi difference).

In the pipeline, the ensemble's neutral-map strength sets the base probability and map
offsets shift it per map: q_m = sigmoid(logit(q_neutral) + w * (phi[a,m] - phi[b,m])).
"""

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from cs2p.elo import PROC

ALIASES = {"Dust 2": "Dust II"}
TUNE_FROM, TUNE_TO = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-03-01")


def load_map_rows() -> pd.DataFrame:
    """One row per map: date, a (map winner), b (map loser), map, series best_of, tier."""
    m = pd.read_csv(PROC / "matches_liquipedia.csv", parse_dates=["date"]).dropna(subset=["maps"])
    rows = []
    for r in m.itertuples(index=False):
        for part in r.maps.split(";"):
            name, _, score = part.rpartition(":")
            try:
                s1, s2 = (int(x) for x in score.split("-"))
            except ValueError:
                continue
            if s1 == s2 or not name:
                continue
            name = ALIASES.get(name, name)
            w, l = (r.winner, r.loser) if s1 > s2 else (r.loser, r.winner)  # scores are from the series winner's view
            rows.append((r.date, w.casefold(), l.casefold(), name, r.best_of, r.tier, r.winner.casefold(), r.loser.casefold()))
    return pd.DataFrame(rows, columns=["date", "a", "b", "map", "best_of", "tier", "series_w", "series_l"])


def active_pool(rows: pd.DataFrame, as_of: pd.Timestamp, days: int = 120, n: int = 7) -> list[str]:
    """The n most-played maps in the window before as_of (the Active Duty pool)."""
    recent = rows[(rows.date < as_of) & (rows.date >= as_of - pd.Timedelta(days=days))]
    return recent["map"].value_counts().head(n).index.tolist()


class MapModel:
    def __init__(self, half_life: float = 180.0, C: float = 1.0, phi_scale: float = 0.3):
        self.half_life, self.C, self.phi_scale = half_life, C, phi_scale

    def fit(self, rows: pd.DataFrame, as_of: pd.Timestamp) -> "MapModel":
        d = rows[rows.date < as_of]
        teams = pd.Index(pd.unique(np.concatenate([d.a.values, d.b.values])))
        maps = pd.Index(sorted(d["map"].unique()))
        nt, nm = len(teams), len(maps)
        ia, ib, im = teams.get_indexer(d.a), teams.get_indexer(d.b), maps.get_indexer(d["map"])
        n = len(d)
        # Antisymmetric design; duplicate each row flipped so the no-intercept fit is balanced.
        r_idx = np.concatenate([np.arange(n)] * 4)
        c_idx = np.concatenate([ia, ib, nt + ia * nm + im, nt + ib * nm + im])
        vals = np.concatenate([np.ones(n), -np.ones(n), np.full(n, self.phi_scale), np.full(n, -self.phi_scale)])
        X = sparse.csr_matrix((vals, (r_idx, c_idx)), shape=(n, nt + nt * nm))
        X = sparse.vstack([X, -X])
        y = np.concatenate([np.ones(n), np.zeros(n)])
        w = 0.5 ** ((as_of - d.date).dt.days.values / self.half_life)
        w = np.concatenate([w, w])
        clf = LogisticRegression(C=self.C, fit_intercept=False, max_iter=500, solver="lbfgs")
        clf.fit(X, y, sample_weight=w)
        coef = clf.coef_[0]
        self.theta = dict(zip(teams, coef[:nt]))
        phi = coef[nt:].reshape(nt, nm) * self.phi_scale
        self.phi = {(t, mp): phi[i, j] for i, t in enumerate(teams) for j, mp in enumerate(maps) if phi[i, j] != 0}
        self.n_maps = d.groupby(["a"]).size().add(d.groupby(["b"]).size(), fill_value=0).to_dict()
        return self

    def offset(self, a: str, b: str, mp: str) -> float:
        return self.phi.get((a, mp), 0.0) - self.phi.get((b, mp), 0.0)

    def map_prob(self, a: str, b: str, mp: str, use_phi: bool = True) -> float:
        z = self.theta.get(a, 0.0) - self.theta.get(b, 0.0) + (self.offset(a, b, mp) if use_phi else 0.0)
        return 1 / (1 + np.exp(-z))


def veto(adv: dict[str, float], best_of: int) -> list[str]:
    """Maps played (in order) for team A vs B, given A's per-map advantage `adv`.

    Bo3 returns [A's pick, B's pick, decider]; Bo1 returns [decider].
    """
    left = sorted(adv, key=adv.get)  # ascending: worst for A first
    if best_of == 1:
        order = ["A", "A", "B", "B", "B", "A"]
        for who in order:
            left.remove(left[0] if who == "A" else left[-1])
        return left
    played = []
    for step in ["A-ban", "B-ban", "A-pick", "B-pick", "A-ban", "B-ban"]:
        who, act = step.split("-")
        worst_for_me = left[0] if who == "A" else left[-1]
        best_for_me = left[-1] if who == "A" else left[0]
        mp = worst_for_me if act == "ban" else best_for_me
        left.remove(mp)
        if act == "pick":
            played.append(mp)
    return played + left


def series_prob(q: list[float], best_of: int) -> float:
    """P(A wins series) given A's per-map win probs in play order."""
    if best_of == 1:
        return q[0]
    a, b, c = q[:3]
    return a * b + (a * (1 - b) + (1 - a) * b) * c


def matchup_prob(q_neutral: float, adv: dict[str, float], best_of: int, w: float = 1.0) -> float:
    """Series win prob for A from a neutral-map prob plus per-map advantages and the veto."""
    played = veto(adv, best_of)
    base = np.log(q_neutral / (1 - q_neutral))
    qs = [1 / (1 + np.exp(-(base + w * adv[mp]))) for mp in played]
    return series_prob(qs, best_of)


def backtest() -> dict:
    """Walk-forward monthly: tune on TUNE window, report on Mar 2026+ (never used for tuning)."""
    from cs2p.zoo import metrics

    rows = load_map_rows()
    lq = pd.read_csv(PROC / "matches_liquipedia.csv", parse_dates=["date"]).dropna(subset=["maps"])
    lq = lq[lq.best_of.isin([1, 3]) & lq.tier.isin(["S-Tier", "A-Tier"])]
    lq = lq.assign(w=lq.winner.str.casefold(), l=lq.loser.str.casefold())

    def run(window: tuple, hl: float, C: float, s: float, wt: float) -> dict:
        months = pd.date_range(window[0], window[1], freq="MS")
        out = {"map_base": [], "map_phi": [], "y_map": [], "series_base": [], "series_veto": [], "bo": []}
        for start, end in zip(months[:-1], months[1:]):
            mm = MapModel(hl, C, s).fit(rows, start)
            pool = active_pool(rows, start)
            mr = rows[(rows.date >= start) & (rows.date < end) & rows.tier.isin(["S-Tier", "A-Tier"])]
            for r in mr.itertuples(index=False):  # map-level, sides randomised by hash for balance
                a, b, y = (r.a, r.b, 1) if hash((r.a, r.b, r.date)) % 2 else (r.b, r.a, 0)
                out["map_base"].append(mm.map_prob(a, b, r.map, use_phi=False))
                out["map_phi"].append(mm.map_prob(a, b, r.map))
                out["y_map"].append(y)
            sr = lq[(lq.date >= start) & (lq.date < end)]
            for r in sr.itertuples(index=False):
                a, b = sorted([r.w, r.l], key=lambda t: -mm.theta.get(t, 0.0))  # A = higher rated
                q = mm.map_prob(a, b, "__neutral__", use_phi=False)
                adv = {mp: mm.offset(a, b, mp) for mp in pool}
                out["series_base"].append(series_prob([q] * 3, r.best_of))
                out["series_veto"].append(matchup_prob(q, adv, r.best_of, wt))
                out["bo"].append(r.best_of)
                out.setdefault("y_series", []).append(1 if a == r.w else 0)
        return {k: np.array(v) for k, v in out.items()}

    grid = [(hl, C, s, wt) for hl in (60, 120, 180) for C in (0.3, 1.0) for s in (0.1, 0.15, 0.3) for wt in (0.5, 1.0)]
    best, best_ll = None, 9.0
    for g in grid:
        o = run((TUNE_FROM, TUNE_TO), *g)
        ll = metrics(o["series_veto"], o["y_series"])["logloss"]
        if ll < best_ll:
            best, best_ll = g, ll
    o = run((TUNE_TO, rows.date.max() + pd.offsets.MonthBegin(1)), *best)
    res = {"params": dict(zip(["half_life", "C", "phi_scale", "w"], best))}
    res["map_level"] = {"n": int(len(o["y_map"])), "no_map_offsets": metrics(o["map_base"], o["y_map"]),
                        "with_map_offsets": metrics(o["map_phi"], o["y_map"])}
    for bo in (1, 3):
        k = o["bo"] == bo
        res[f"bo{bo}"] = {"n": int(k.sum()), "no_veto": metrics(o["series_base"][k], o["y_series"][k]),
                          "with_veto": metrics(o["series_veto"][k], o["y_series"][k])}
    # Paired bootstrap on per-match log-loss difference (veto minus base), all series.
    def ll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    d = ll(o["series_veto"], o["y_series"]) - ll(o["series_base"], o["y_series"])
    bs = d[np.random.default_rng(0).integers(0, len(d), (2000, len(d)))].mean(1)
    res["series_delta"] = {"mean": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    dm = ll(o["map_phi"], o["y_map"]) - ll(o["map_base"], o["y_map"])
    bm = dm[np.random.default_rng(1).integers(0, len(dm), (2000, len(dm)))].mean(1)
    res["map_delta"] = {"mean": float(dm.mean()), "ci": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}
    (PROC / "map_backtest.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(backtest(), indent=2))
