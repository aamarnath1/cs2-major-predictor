"""One chronological pass over all matches producing pre-match features for every model.

Each row describes the match from the winner's perspective ("a" = winner, "b" = loser);
the zoo harness randomly flips rows so label balance is 50/50 for learned models.
Online raters: Elo, Glicko-2, TrueSkill. Context: form, activity, head-to-head, VRS points.
"""

import json
import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import trueskill

from cs2p.elo import PROC, Elo, EloParams, win_prob

# ---------------------------------------------------------------- Glicko-2
G2_SCALE = 173.7178
G2_TAU = 0.5


class Glicko2:
    """Glicko-2 with each match treated as its own rating period; RD grows with idle days."""

    def __init__(self, rd_growth_per_day: float = 0.6):
        self.mu: dict = {}
        self.phi: dict = {}
        self.sigma: dict = {}
        self.last: dict = {}
        self.c = rd_growth_per_day / G2_SCALE

    def _get(self, t: str, date: pd.Timestamp) -> tuple[float, float, float]:
        mu = self.mu.get(t, 0.0)
        phi = self.phi.get(t, 350 / G2_SCALE)
        sig = self.sigma.get(t, 0.06)
        if t in self.last:
            days = (date - self.last[t]).days
            phi = min(math.sqrt(phi**2 + (self.c**2) * days), 350 / G2_SCALE)
        return mu, phi, sig

    @staticmethod
    def _g(phi: float) -> float:
        return 1 / math.sqrt(1 + 3 * phi**2 / math.pi**2)

    def expect(self, a: str, b: str, date: pd.Timestamp) -> tuple[float, float, float]:
        """Return (P(a beats b), rating a, rating b) on Glicko scale."""
        ma, pa, _ = self._get(a, date)
        mb, pb, _ = self._get(b, date)
        p = 1 / (1 + math.exp(-self._g(math.sqrt(pa**2 + pb**2)) * (ma - mb)))
        return p, ma, mb

    def _update_one(self, t: str, opp: str, score: float, date: pd.Timestamp) -> tuple:
        mu, phi, sig = self._get(t, date)
        mo, po, _ = self._get(opp, date)
        g = self._g(po)
        e = 1 / (1 + math.exp(-g * (mu - mo)))
        v = 1 / (g**2 * e * (1 - e))
        delta = v * g * (score - e)
        a = math.log(sig**2)

        def f(x: float) -> float:
            ex = math.exp(x)
            return ex * (delta**2 - phi**2 - v - ex) / (2 * (phi**2 + v + ex) ** 2) - (x - a) / G2_TAU**2

        A = a
        if delta**2 > phi**2 + v:
            B = math.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * G2_TAU) < 0:
                k += 1
            B = a - k * G2_TAU
        fa, fb = f(A), f(B)
        for _ in range(50):
            if abs(B - A) < 1e-6:
                break
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            if fc * fb <= 0:
                A, fa = B, fb
            else:
                fa /= 2
            B, fb = C, fc
        sig_new = math.exp(A / 2)
        phi_star = math.sqrt(phi**2 + sig_new**2)
        phi_new = 1 / math.sqrt(1 / phi_star**2 + 1 / v)
        mu_new = mu + phi_new**2 * g * (score - e)
        return mu_new, phi_new, sig_new

    def update(self, w: str, l: str, date: pd.Timestamp) -> None:
        nw = self._update_one(w, l, 1.0, date)
        nl = self._update_one(l, w, 0.0, date)
        for t, (mu, phi, sig) in ((w, nw), (l, nl)):
            self.mu[t], self.phi[t], self.sigma[t] = mu, phi, sig
            self.last[t] = date


# ---------------------------------------------------------------- TrueSkill
TS = trueskill.TrueSkill(mu=25.0, sigma=25 / 3, beta=25 / 6, tau=25 / 300, draw_probability=0.0)


def ts_win_prob(ra: trueskill.Rating, rb: trueskill.Rating) -> float:
    denom = math.sqrt(2 * TS.beta**2 + ra.sigma**2 + rb.sigma**2)
    return TS.cdf((ra.mu - rb.mu) / denom)


# ---------------------------------------------------------------- main pass
def vrs_lookup(dates: pd.Series) -> callable:
    pts = pd.read_csv(PROC / "vrs_points.csv", parse_dates=["snapshot"])
    pts["team"] = pts.team.str.casefold()
    pts = pts.groupby(["snapshot", "team"]).points.max()
    snaps = np.sort(pts.index.get_level_values(0).unique().values)
    table = pts.to_dict()

    def get(team: str, date: pd.Timestamp) -> float:
        j = np.searchsorted(snaps, np.datetime64(date), side="left") - 1
        return table.get((pd.Timestamp(snaps[j]), team), np.nan) if j >= 0 else np.nan

    return get


def build_features(
    m: pd.DataFrame, start: pd.Timestamp | None = None, *, batch_by_day: bool = False,
    elo_params: EloParams | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return (one feature row per match with winner = side a, final rater states).

    Only matches on/after `start` are emitted as rows; all matches update the raters.
    With `batch_by_day`, predictions use prior-day ratings and context. Same-day
    results update the state after every prediction for that day, in canonical order.
    """
    if elo_params is None:
        elo_params = EloParams(**json.loads((PROC / "elo_params.json").read_text()))
    elo = Elo(elo_params)
    gl = Glicko2()
    ts: dict = {}
    hist: dict = defaultdict(lambda: deque(maxlen=30))  # (date, won, opp_elo)
    h2h: dict = defaultdict(list)  # frozenset -> [(date, winner)]
    games: dict = defaultdict(int)
    last: dict = {}
    vrs = vrs_lookup(m.date)
    if batch_by_day:
        canonical = m.assign(date=m.date.dt.normalize()).sort_values(
            ["date", "winner", "loser", "winner_roster", "loser_roster"], kind="stable"
        )
        batches = (
            group.itertuples(index=False) for _, group in canonical.groupby("date", sort=False)
        )
    else:
        batches = ((r,) for r in m.itertuples(index=False))
    rows = []
    for batch in batches:
        updates = []
        for r in batch:
            w, l, d = r.winner.casefold(), r.loser.casefold(), r.date
            row = None
            if start is None or d >= start:
                row = {"date": d, "a": w, "b": l}
                row["elo_a"] = elo.current(w, d)
                row["elo_b"] = elo.current(l, d)
                pg, ga, gb = gl.expect(w, l, d)
                row.update(glicko_p=pg, glicko_a=ga, glicko_b=gb,
                           glicko_rd_a=gl._get(w, d)[1], glicko_rd_b=gl._get(l, d)[1])
                ta, tb = ts.get(w, TS.create_rating()), ts.get(l, TS.create_rating())
                row.update(ts_p=ts_win_prob(ta, tb), ts_mu_a=ta.mu, ts_mu_b=tb.mu,
                           ts_sig_a=ta.sigma, ts_sig_b=tb.sigma)
                for side, t in (("a", w), ("b", l)):
                    hh = hist[t]
                    last10 = list(hh)[-10:]
                    row[f"form10_{side}"] = np.mean([x[1] for x in last10]) if last10 else 0.5
                    recent = [x for x in hh if (d - x[0]).days <= 60]
                    row[f"form60_{side}"] = np.mean([x[1] for x in recent]) if recent else 0.5
                    row[f"sos10_{side}"] = np.mean([x[2] for x in last10]) if last10 else 1300.0
                    row[f"games_{side}"] = games[t]
                    row[f"idle_{side}"] = min((d - last[t]).days, 365) if t in last else 365
                    row[f"vrs_{side}"] = vrs(t, d)
                past = [x for x in h2h[frozenset((w, l))] if (d - x[0]).days <= 365]
                row["h2h_n"] = len(past)
                row["h2h_a"] = sum(1 for x in past if x[1] == w)
                if batch_by_day:
                    row["elo_p"] = win_prob(row["elo_a"], row["elo_b"], elo.p.scale)
                rows.append(row)
            updates.append((r, w, l, d, row, elo.current(w, d), elo.current(l, d)))
        for r, w, l, d, row, elo_a_pre, elo_b_pre in updates:
            row_p = elo.update(r.winner, r.loser, d, r.winner_roster, r.loser_roster)
            if row is not None and not batch_by_day:
                row["elo_p"] = row_p
            gl.update(w, l, d)
            ta, tb = ts.get(w, TS.create_rating()), ts.get(l, TS.create_rating())
            (ta,), (tb,) = TS.rate([(ta,), (tb,)], ranks=[0, 1])
            ts[w], ts[l] = ta, tb
            hist[w].append((d, 1.0, elo_b_pre))
            hist[l].append((d, 0.0, elo_a_pre))
            h2h[frozenset((w, l))].append((d, w))
            games[w] += 1
            games[l] += 1
            last[w] = last[l] = d
    return pd.DataFrame(rows), {"elo": elo, "glicko": gl, "ts": ts}
