"""8-team single-elimination playoffs: simulation and bracket Pick'em optimiser.

Seeding (Liquipedia Bracket/8): QF1 1v8, QF2 4v5, QF3 2v7, QF4 3v6; SF1 = QF1/QF2 winners,
SF2 = QF3/QF4 winners. QF/SF are Bo3, Grand Final Bo5. Valve's playoff card has 7 picks
(4 QF, 2 SF, final); picks are chained, so an SF pick must be one of your QF winners.
"""

import itertools

import numpy as np

from cs2p.swiss import bo5_from_bo3

QF_SEEDS = [(0, 7), (3, 4), (1, 6), (2, 5)]


def simulate_bracket(p_bo3: np.ndarray, n_sims: int = 20_000, seed: int = 0) -> np.ndarray:
    """p_bo3: (8, 8) matrix indexed by playoff seed. Returns (n_sims, 7) winner indices
    for slots [QF1, QF2, QF3, QF4, SF1, SF2, Final]."""
    rng = np.random.default_rng(seed)
    p_bo5 = bo5_from_bo3(p_bo3)
    out = np.empty((n_sims, 7), dtype=np.int8)
    for s in range(n_sims):
        qf = [a if rng.random() < p_bo3[a, b] else b for a, b in QF_SEEDS]
        sf = [qf[0] if rng.random() < p_bo3[qf[0], qf[1]] else qf[1],
              qf[2] if rng.random() < p_bo3[qf[2], qf[3]] else qf[3]]
        fin = sf[0] if rng.random() < p_bo5[sf[0], sf[1]] else sf[1]
        out[s] = qf + sf + [fin]
    return out


def all_brackets() -> list[list[int]]:
    """All 128 internally consistent 7-pick brackets."""
    out = []
    for qf in itertools.product(*QF_SEEDS):
        for sf in itertools.product((qf[0], qf[1]), (qf[2], qf[3])):
            for fin in sf:
                out.append(list(qf) + list(sf) + [fin])
    return out


def optimise_bracket(winners: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Maximise all three Pick'em challenges, breaking ties with expected hits."""
    best, best_score, best_dist = [], (-1.0, -1.0), np.array([])
    for bracket in all_brackets():
        correct = winners == np.array(bracket)
        hits = correct.sum(axis=1)
        score = (float(playoff_challenge(correct).mean()), float(hits.mean()))
        if score > best_score:
            best, best_score = bracket, score
            best_dist = np.bincount(hits, minlength=8) / len(hits)
    return best, best_dist


def playoff_challenge(correct: np.ndarray) -> np.ndarray:
    """Return whether each simulation clears the quarterfinal, semifinal and final challenges."""
    return (correct[:, :4].sum(1) >= 2) & correct[:, 4:6].any(1) & correct[:, 6]
