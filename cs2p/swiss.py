"""Monte Carlo simulator for a 16-team Valve Major Swiss stage.

Rules (PGL Singapore 2026, per Liquipedia):
- Round 1: seed i vs seed i+8.
- Later rounds: teams grouped by record; within a group sorted by Buchholz
  (sum of opponents' W-L) then initial seed; highest plays lowest, avoiding rematches.
- Stages 1-2: elimination/advancement matches Bo3, the rest Bo1. Stage 3: all Bo3.
- 3 wins advance, 3 losses eliminated.

Real results can be injected with `fixed_rounds` so live odds condition on what happened.
"""

import numpy as np

# Outcome codes per team per simulation.
ADV_30, ADV, ELIM_03, ELIM = 0, 1, 2, 3

# fixed_rounds: one list per played/announced round of (team_a, team_b, winner or None).
FixedRounds = list[list[tuple[int, int, int | None]]]


def bo1_from_bo3(p: np.ndarray) -> np.ndarray:
    """Map prob q such that a Bo3 at q gives p: solve q^2 (3 - 2q) = p by bisection."""
    p = np.asarray(p, dtype=float)
    lo, hi = np.zeros_like(p), np.ones_like(p)
    for _ in range(40):
        mid = (lo + hi) / 2
        f = mid**2 * (3 - 2 * mid)
        lo = np.where(f < p, mid, lo)
        hi = np.where(f < p, hi, mid)
    return (lo + hi) / 2


def bo5_from_bo3(p: np.ndarray) -> np.ndarray:
    """Bo5 win prob for the same per-map strength q: P(win >= 3 of 5)."""
    q = bo1_from_bo3(p)
    return q**3 * (10 - 15 * q + 6 * q**2)


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


def _with_fixed(default: list[tuple[int, int, None]], fixed: list[tuple[int, int, int | None]]) -> list:
    """Round-1 pairs: real fixtures first, then default seed pairs for teams not yet covered."""
    busy = {t for a, b, _ in fixed for t in (a, b)}
    return list(fixed) + [p for p in default if p[0] not in busy and p[1] not in busy]


def run_stage(
    p_bo3: np.ndarray,
    rng: np.random.Generator,
    all_bo3: bool = False,
    fixed_rounds: FixedRounds | None = None,
    p_bo1: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Play one Swiss stage.

    Returns:
        (outcome code per team, final standings order: record, then Buchholz, then seed).
    """
    n = p_bo3.shape[0]
    if p_bo1 is None:
        p_bo1 = bo1_from_bo3(p_bo3)
    fixed_rounds = fixed_rounds or []
    wins, losses = [0] * n, [0] * n
    opps: list[list[int]] = [[] for _ in range(n)]
    played: set = set()
    rnd = 0
    pairs = _with_fixed([(i, i + n // 2, None) for i in range(n // 2)], fixed_rounds[0] if fixed_rounds else [])
    while True:
        for a, b, known in pairs:
            if known is None:
                decider = all_bo3 or wins[a] == 2 or losses[a] == 2
                pa = (p_bo3 if decider else p_bo1)[a, b]
                known = a if rng.random() < pa else b
            w, l = (a, b) if known == a else (b, a)
            wins[w] += 1
            losses[l] += 1
            opps[a].append(b)
            opps[b].append(a)
            played |= {(a, b), (b, a)}
        rnd += 1
        alive = [t for t in range(n) if wins[t] < 3 and losses[t] < 3]
        if not alive:
            break
        fixed = fixed_rounds[rnd] if rnd < len(fixed_rounds) else []
        busy = {t for a, b, _ in fixed for t in (a, b)}
        buch = [sum(wins[o] - losses[o] for o in opps[t]) for t in range(n)]
        pairs = []
        for rec in sorted({(wins[t], losses[t]) for t in alive}, reverse=True):
            group = sorted((t for t in alive if (wins[t], losses[t]) == rec and t not in busy),
                           key=lambda t: (-buch[t], t))
            got = _pair(group, played) or [(group[i], group[-1 - i]) for i in range(len(group) // 2)]
            pairs += [(a, b, None) for a, b in got]
        pairs = list(fixed) + pairs
    buch = [sum(wins[o] - losses[o] for o in opps[t]) for t in range(n)]
    codes = np.empty(n, dtype=np.int8)
    for t in range(n):
        if wins[t] == 3:
            codes[t] = ADV_30 if losses[t] == 0 else ADV
        else:
            codes[t] = ELIM_03 if wins[t] == 0 else ELIM
    order = sorted(range(n), key=lambda t: (-(wins[t] - losses[t]), -buch[t], t))
    return codes, order


def simulate(
    p_bo3: np.ndarray,
    n_sims: int = 20_000,
    seed: int = 0,
    all_bo3: bool = False,
    fixed_rounds: FixedRounds | None = None,
) -> np.ndarray:
    """Simulate a Swiss stage many times.

    Args:
        p_bo3: (16, 16) matrix, p_bo3[i, j] = P(team i beats team j in a Bo3).
            Teams are indexed by seed (0 = top seed).
        n_sims: number of simulated stages.
        all_bo3: every match is Bo3 (Stage 3).
        fixed_rounds: real pairings/results to replay before simulating the rest.

    Returns:
        (n_sims, 16) int array of outcome codes (ADV_30, ADV, ELIM_03, ELIM).
    """
    rng = np.random.default_rng(seed)
    p_bo1 = bo1_from_bo3(p_bo3)
    out = np.empty((n_sims, p_bo3.shape[0]), dtype=np.int8)
    for s in range(n_sims):
        out[s], _ = run_stage(p_bo3, rng, all_bo3, fixed_rounds, p_bo1)
    return out
