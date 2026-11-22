from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cs2p import features


def test_day_batched_features_ignore_same_day_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(features, "PROC", tmp_path)
    (tmp_path / "elo_params.json").write_text('{"roster_shrink": 0}', encoding="utf-8")
    (tmp_path / "vrs_points.csv").write_text(
        "snapshot,team,points\n2026-03-01,A,1400\n2026-03-02,A,1600\n",
        encoding="utf-8",
    )
    matches = pd.DataFrame(
        {
            "winner": ["A", "A", "A", "A"],
            "loser": ["B", "B", "C", "B"],
            "date": pd.to_datetime(
                ["2026-03-02 08:00", "2026-03-02 10:00", "2026-03-02 14:00", "2026-03-03 00:00"]
            ),
            "winner_roster": [None] * 4,
            "loser_roster": [None] * 4,
        }
    )

    batched, _ = features.build_features(matches, batch_by_day=True)
    reversed_day, _ = features.build_features(matches.iloc[[2, 1, 0, 3]], batch_by_day=True)
    pd.testing.assert_frame_equal(batched, reversed_day)
    first_day = batched.iloc[:3]
    assert (first_day[["games_a", "h2h_n"]] == 0).all().all()
    assert np.allclose(first_day[["elo_p", "glicko_p", "ts_p", "form10_a"]], 0.5)
    assert first_day.vrs_a.tolist() == [1400, 1400, 1400]
    next_day = batched.iloc[3]
    assert next_day.games_a == 3
    assert next_day.h2h_n == 2
    assert next_day.vrs_a == 1600
    assert (next_day[["elo_p", "glicko_p", "ts_p"]] > 0.5).all()

    live, _ = features.build_features(matches)
    assert live.iloc[1].games_a == 1
    assert live.iloc[1].elo_p > 0.5
