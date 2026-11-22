"""Series-level Elo with roster-change shrinkage and inactivity decay."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
MEAN = 1500.0


@dataclass
class EloParams:
    k: float = 32.0
    k_new: float = 64.0  # K for a team's first `n_new` matches
    n_new: int = 15
    init: float = 1400.0  # unseen teams start below average (most are tier 3+)
    roster_shrink: float = 0.15  # per changed player, pull toward MEAN
    decay_days: float = 365.0  # inactivity pull toward MEAN, e-folding time
    scale: float = 400.0


def win_prob(ra: float, rb: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / scale))


def _roster_set(s: object) -> frozenset | None:
    return frozenset(p.strip().lower() for p in s.split(",")) if isinstance(s, str) else None


@dataclass
class Elo:
    p: EloParams = field(default_factory=EloParams)
    rating: dict = field(default_factory=dict)
    games: dict = field(default_factory=dict)
    roster: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)

    def _prep(self, team: str, roster: frozenset | None, date: pd.Timestamp) -> float:
        r = self.rating.get(team, self.p.init)
        if team in self.last:
            days = (date - self.last[team]).days
            r = MEAN + (r - MEAN) * np.exp(-days / self.p.decay_days)
        old = self.roster.get(team)
        if roster is not None and old is not None:
            changed = len(roster - old)
            if changed:
                r = MEAN + (r - MEAN) * max(0.0, 1 - self.p.roster_shrink * changed)
        if roster is not None:
            self.roster[team] = roster
        self.rating[team] = r
        self.last[team] = date
        return r

    def update(self, w: str, l: str, date: pd.Timestamp, wr: object = None, lr: object = None) -> float:
        """Apply one result; return the pre-match win prob the model gave the winner."""
        w, l = w.casefold(), l.casefold()
        ra, rb = self._prep(w, _roster_set(wr), date), self._prep(l, _roster_set(lr), date)
        pw = win_prob(ra, rb, self.p.scale)
        for t, sign in ((w, 1), (l, -1)):
            n = self.games.get(t, 0)
            k = self.p.k_new if n < self.p.n_new else self.p.k
            self.rating[t] += sign * k * (1 - pw)
            self.games[t] = n + 1
        return pw

    def current(self, team: str, as_of: pd.Timestamp, stand_ins: int = 0) -> float:
        """Rating projected to `as_of`, with optional shrink for stand-in players."""
        team = team.casefold()
        r = self.rating.get(team, self.p.init)
        if team in self.last:
            r = MEAN + (r - MEAN) * np.exp(-(as_of - self.last[team]).days / self.p.decay_days)
        return MEAN + (r - MEAN) * max(0.0, 1 - self.p.roster_shrink * stand_ins)


def run(matches: pd.DataFrame, params: EloParams) -> tuple[Elo, np.ndarray]:
    """Fit Elo over matches in date order; return model and per-match winner probabilities."""
    elo = Elo(params)
    probs = np.empty(len(matches))
    for i, row in enumerate(matches.itertuples(index=False)):
        probs[i] = elo.update(row.winner, row.loser, row.date, row.winner_roster, row.loser_roster)
    return elo, probs


def load_matches() -> pd.DataFrame:
    """VRS match history, extended with Liquipedia results outside the VRS date range.

    Liquipedia rows are only used before the first / after the last VRS date, so the two
    sources never double-count the same match.
    """
    m = pd.read_csv(PROC / "matches.csv", parse_dates=["date"])
    lq_path = PROC / "matches_liquipedia.csv"
    if lq_path.exists():
        lq = pd.read_csv(lq_path, parse_dates=["date"])
        lq = lq[(lq.date > m.date.max()) | (lq.date < m.date.min())]
        m = pd.concat([m, lq[["date", "winner", "loser"]]], ignore_index=True)
        m = m.sort_values("date", kind="stable").reset_index(drop=True)
    return m
