import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cs2p import replay
from cs2p.swiss import ADV, ADV_30, ELIM, ELIM_03, _late_pair, run_stage


def test_late_pair_uses_valve_priority() -> None:
    # These results reach 1-2 with group [11, 15, 1, 5, 7, 14] in Buchholz order.
    # The earlier 1-7 and 7-15 pairings rule out Valve's first two table rows.
    fixed = [
        [(0, 8, 8), (1, 9, 9), (2, 10, 10), (3, 11, 11), (4, 12, 12),
         (5, 13, 13), (6, 14, 6), (7, 15, 15)],
        [(6, 15, 6), (8, 13, 13), (9, 12, 9), (10, 11, 10), (0, 14, 0),
         (1, 7, 7), (2, 5, 5), (3, 4, 3)],
        [(13, 10, 10), (6, 9, 9), (8, 7, 8), (11, 0, 0), (15, 3, 3),
         (5, 12, 12), (1, 4, 1), (2, 14, 14)],
    ]
    probabilities = np.ones((16, 16))
    probabilities[11, 7] = probabilities[15, 14] = 0
    visits = np.zeros((16, 4, 4))

    run_stage(probabilities, np.random.default_rng(1), fixed_rounds=fixed, visits=visits)

    advanced_to_round_5 = {t for t in (11, 15, 1, 5, 7, 14) if visits[t, 2, 2]}
    assert advanced_to_round_5 == {7, 14, 1}

    visits.fill(0)
    fixed.append([(7, 11, 11)])
    run_stage(probabilities, np.random.default_rng(1), fixed_rounds=fixed, visits=visits)

    advanced_to_round_5 = {t for t in (11, 15, 1, 5, 7, 14) if visits[t, 2, 2]}
    assert advanced_to_round_5 == {11, 14, 1}


def test_late_pair_respects_a_confirmed_fixture() -> None:
    group = [0, 1, 2, 3, 4, 5]

    assert _late_pair(group, set(), [(3, 0, 3)]) == [(1, 5), (2, 4)]
    assert _late_pair(group, set(), [(5, 0, 5), (1, 4, 1), (3, 2, None)]) == []


def test_replay_uses_each_cached_stage_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [(3, 0)] * 2 + [(3, 1)] * 3 + [(3, 2)] * 3
    records += [(2, 3)] * 3 + [(1, 3)] * 3 + [(0, 3)] * 2
    teams = [
        {"name": f"Team {i}", "seed": i + 1, "wins": wins, "losses": losses}
        for i, (wins, losses) in enumerate(records)
    ]
    counts = (8, 8, 8, 6, 3)
    stages = [
        {
            "event": "Major",
            "stage": f"Stage {i}",
            "start_date": "2026-06-11",
            "teams": teams,
            "matches": [
                {"round": rnd, "best_of": 3 if all_bo3 or rnd >= 4 or (rnd == 3 and j >= 4) else 1}
                for rnd, count in enumerate(counts, 1)
                for j in range(count)
            ],
        }
        for i, all_bo3 in enumerate((False, True), 1)
    ]
    (tmp_path / "past_major_stages.json").write_text(json.dumps(stages), encoding="utf-8")
    history = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]})
    prob = pd.DataFrame(np.full((16, 16), 0.5))
    calls: list[bool] = []

    def simulate(probabilities: np.ndarray, n_sims: int, seed: int, all_bo3: bool) -> np.ndarray:
        calls.append(all_bo3)
        return np.tile([ADV_30] * 2 + [ADV] * 6 + [ELIM] * 6 + [ELIM_03] * 2, (n_sims, 1))

    monkeypatch.setattr(replay, "PROC", tmp_path)
    monkeypatch.setattr(replay, "load_matches", lambda: history)
    monkeypatch.setattr(replay, "matchup_matrix", lambda *args: (prob, dict.fromkeys(replay.RATERS, prob)))
    monkeypatch.setattr(replay, "simulate", simulate)

    replay.main(n_sims=3)

    assert calls == [False] * 5 + [True] * 5
    rows = json.loads((tmp_path / "replay.json").read_text(encoding="utf-8"))["stages"]
    assert [row["format"] for row in rows] == ["mixed", "all_bo3"]

    stages[1]["matches"].pop()
    (tmp_path / "past_major_stages.json").write_text(json.dumps(stages), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown or incomplete Swiss format for Major Stage 2"):
        replay.main(n_sims=3)
