"""Choose Pick'em picks that maximise P(>= 5 correct) over simulated Swiss outcomes.

Assumed Valve Pick'em rules per stage: 2 teams to go 3-0, 2 to go 0-3, and 6 teams to
advance with a 3-1 or 3-2 record. A pick is only correct for its own slot.
"""

import numpy as np

from cs2p.swiss import ADV, ADV_30, ELIM_03

SLOTS = [(ADV_30, 2), (ELIM_03, 2), (ADV, 6)]
NEED = 5


def score(outcomes: np.ndarray, picks: dict[int, list[int]]) -> np.ndarray:
    """Number of correct picks in each simulation."""
    total = np.zeros(len(outcomes), dtype=np.int16)
    for code, teams in picks.items():
        total += (outcomes[:, teams] == code).sum(axis=1)
    return total


def objective(outcomes: np.ndarray, picks: dict[int, list[int]]) -> tuple[float, float]:
    s = score(outcomes, picks)
    return float(np.mean(s >= NEED)), float(np.mean(s))


def optimise(outcomes: np.ndarray, max_iter: int = 200) -> tuple[dict[int, list[int]], tuple[float, float]]:
    """Greedy start by marginal probability, then hill-climb with replace/swap moves."""
    n = outcomes.shape[1]
    marg = {code: (outcomes == code).mean(axis=0) for code, _ in SLOTS}
    picks: dict[int, list[int]] = {}
    used: set = set()
    for code, k in SLOTS:  # fill the scarcest-probability slots first
        order = [t for t in np.argsort(-marg[code]) if t not in used][:k]
        picks[code] = [int(t) for t in order]
        used |= set(order)

    best = objective(outcomes, picks)
    for _ in range(max_iter):
        improved = False
        for code, _ in SLOTS:
            for i in range(len(picks[code])):
                # Replace with an unused team.
                for t in range(n):
                    if t in used:
                        continue
                    trial = {c: list(v) for c, v in picks.items()}
                    trial[code][i] = t
                    val = objective(outcomes, trial)
                    if val > best:
                        used = (used - {picks[code][i]}) | {t}
                        picks, best, improved = trial, val, True
                # Swap with a team in another slot.
                for code2, _ in SLOTS:
                    if code2 == code:
                        continue
                    for j in range(len(picks[code2])):
                        trial = {c: list(v) for c, v in picks.items()}
                        trial[code][i], trial[code2][j] = picks[code2][j], picks[code][i]
                        val = objective(outcomes, trial)
                        if val > best:
                            picks, best, improved = trial, val, True
        if not improved:
            break
    return picks, best
