"""Monte Carlo simulator for a 16-team Valve Major Swiss stage.

Rules (PGL Singapore 2026, per Liquipedia):
- Round 1: seed i vs seed i+8.
- Later rounds: teams grouped by record; within a group sorted by Buchholz
  (sum of opponents' W-L) then initial seed; highest plays lowest, avoiding rematches.
- Matches where either team can be eliminated or advance are Bo3, the rest Bo1.
- 3 wins advance, 3 losses eliminated.
"""

import numpy as np

# Outcome codes per team per simulation.
ADV_30, ADV, ELIM_03, ELIM = 0, 1, 2, 3


def bo1_from_bo3(p: np.ndarray) -> np.ndarray:
    """Map prob q such that a Bo3 at q gives p: solve q^2 (3 - 2q) = p by bisection."""
    lo, hi = np.zeros_like(p), np.ones_like(p)
    for _ in range(40):
        mid = (lo + hi) / 2
        f = mid**2 * (3 - 2 * mid)
        lo = np.where(f < p, mid, lo)
        hi = np.where(f < p, hi, mid)
    return (lo + hi) / 2


def _pair(group: list[int], played: set) -> list[tuple[int, int]] | None:
    """Pair sorted group top-vs-bottom, backtracking to avoid rematches."""
    if not group:
        return []
    a = group[0]
    for j in range(len(group) - 1, 0, -1):
        b = group[j]
        if (a, b) not in played:
            rest = _pair(group[1:j] + group[j + 1 :], played)
            if rest is not None:
                return [(a, b)] + rest
    return None


def simulate(p_bo3: np.ndarray, n_sims: int = 20_000, seed: int = 0) -> np.ndarray:
    """Simulate a Swiss stage.

    Args:
        p_bo3: (16, 16) matrix, p_bo3[i, j] = P(team i beats team j in a Bo3).
            Teams are indexed by seed (0 = top seed).
        n_sims: number of simulated stages.

    Returns:
        (n_sims, 16) int array of outcome codes (ADV_30, ADV, ELIM_03, ELIM).
    """
    n = p_bo3.shape[0]
    p_bo1 = bo1_from_bo3(p_bo3)
    rng = np.random.default_rng(seed)
    out = np.empty((n_sims, n), dtype=np.int8)
    for s in range(n_sims):
        wins, losses = [0] * n, [0] * n
        opps: list[list[int]] = [[] for _ in range(n)]
        played: set = set()
        pairs = [(i, i + n // 2) for i in range(n // 2)]
        while True:
            for a, b in pairs:
                decider = wins[a] == 2 or losses[a] == 2
                pa = (p_bo3 if decider else p_bo1)[a, b]
                w, l = (a, b) if rng.random() < pa else (b, a)
                wins[w] += 1
                losses[l] += 1
                opps[a].append(b)
                opps[b].append(a)
                played |= {(a, b), (b, a)}
            alive = [t for t in range(n) if wins[t] < 3 and losses[t] < 3]
            if not alive:
                break
            buch = [sum(wins[o] - losses[o] for o in opps[t]) for t in range(n)]
            pairs = []
            for rec in sorted({(wins[t], losses[t]) for t in alive}, reverse=True):
                group = sorted((t for t in alive if (wins[t], losses[t]) == rec), key=lambda t: (-buch[t], t))
                pairs += _pair(group, played) or [(group[i], group[-1 - i]) for i in range(len(group) // 2)]
        for t in range(n):
            if wins[t] == 3:
                out[s, t] = ADV_30 if losses[t] == 0 else ADV
            else:
                out[s, t] = ELIM_03 if wins[t] == 0 else ELIM
    return out
