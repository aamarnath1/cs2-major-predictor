import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cs2p import run
from cs2p.playoffs import all_brackets, optimise_bracket, playoff_challenge


def test_stale_feed_preserves_pre_cutoff_swiss_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cutoff = run.PICK_CUTOFFS[1]

    class Clock(datetime):
        current = cutoff - timedelta(seconds=1)

        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return cls.current

    monkeypatch.setattr(run, "datetime", Clock)
    monkeypatch.setattr(run, "OUT", tmp_path)
    teams = [f"T{i}" for i in range(16)]
    matrix = pd.DataFrame(0.5, index=teams, columns=teams)
    state = {"seeds": teams, "matches": [], "status": "upcoming", "start_date": None}
    before = run.stage_block(1, state, matrix, {}, {}, 100)
    path = tmp_path / "picks_stage1.json"
    snapshot = path.read_bytes()
    saved = json.loads(snapshot)
    assert datetime.fromisoformat(saved["frozen_at"]).tzinfo == UTC
    assert len(saved["round1"]) == 8 and len(saved["seed_picks"]["3-0"]) == 2

    Clock.current = cutoff
    after = run.stage_block(1, state, matrix, {}, {}, 100)
    assert after["pick_locked"] and after["picks"] == before["picks"]
    assert path.read_bytes() == snapshot

    path.unlink()
    missing = run.stage_block(1, state, matrix, {}, {}, 100)
    assert missing["picks"] is None and missing["p_at_least_5"] is None
    assert missing["pick_status"] == {} and not path.exists()

    state["status"] = "live"
    state["matches"] = [{"a": teams[0], "b": teams[8], "round": 1, "winner": teams[8]}]
    live = run.stage_block(1, state, matrix, {}, {}, 100)
    assert live["picks"] is None and live["pick_status"] == {} and not path.exists()


def test_cutoff_is_rechecked_when_publishing_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cutoff = run.PICK_CUTOFFS[1]
    times = iter([cutoff - timedelta(microseconds=1), cutoff])

    class Clock(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return next(times)

    monkeypatch.setattr(run, "datetime", Clock)
    path = tmp_path / "picks.json"
    path.write_text("existing")
    assert not run.save_card(path, cutoff, {})
    assert path.read_text() == "existing" and list(tmp_path.iterdir()) == [path]


def test_invalid_playoff_card_is_withheld_and_seed_order_is_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cutoff = run.PICK_CUTOFFS["playoffs"]

    class Clock(datetime):
        current = cutoff - timedelta(minutes=1)

        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return cls.current

    monkeypatch.setattr(run, "datetime", Clock)
    monkeypatch.setattr(run, "OUT", tmp_path)
    seeds = np.array([[7, 2, 6, 0, 5, 3, 4, 1]])
    winners = seeds[:, all_brackets()[0]]
    names = [f"T{i}" for i in range(8)]
    major = {"names": names, "playoff_seeds": seeds, "winners": winners}
    state = {"status": "upcoming", "slots": {}}
    first = run.playoff_block(major, state)
    assert first["picks"] == [names[i] for i in winners[0]]
    assert first["p_playoff_challenges"] == 1.0

    path = tmp_path / "picks_playoffs.json"
    payload = json.loads(path.read_bytes())
    payload["picks"] = [names[0]] * 7
    path.write_text(json.dumps(payload))
    Clock.current = cutoff
    locked = run.playoff_block(major, state)
    assert locked["pick_locked"] and locked["picks"] is None


def test_playoff_objective_can_choose_different_card_than_expected_hits() -> None:
    brackets = np.array(all_brackets())
    rng = np.random.default_rng(3)
    winners = brackets[rng.choice(len(brackets), 71)]
    expected_hits = [float((winners == card).sum(1).mean()) for card in brackets]
    best_hits = max(expected_hits)
    hits_cards = [card for card, hits in zip(brackets, expected_hits) if hits == best_hits]
    challenge_card, _ = optimise_bracket(winners)
    assert playoff_challenge(winners == challenge_card).mean() > max(
        playoff_challenge(winners == card).mean() for card in hits_cards
    )
