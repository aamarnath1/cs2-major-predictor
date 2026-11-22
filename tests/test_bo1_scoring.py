import json
from hashlib import sha256
from math import log
from pathlib import Path

import pytest

from cs2p.bo1_experiment import score_singapore_round1
from cs2p.run import OUT


def test_scoring_joins_winners_to_frozen_orientation(tmp_path: Path) -> None:
    forecast = OUT / "bo1_singapore_stage1.json"
    frozen = json.loads(forecast.read_bytes())
    results = tmp_path / "data.json"
    matches = [
        {"round": 1, "a": match["b"], "b": match["a"], "winner": match["a" if i % 2 else "b"]}
        for i, match in enumerate(frozen["matches"])
    ]
    matches.append({**matches[0], "round": 2})
    results.write_text(json.dumps({"stages": [{"stage": 1, "matches": matches}]}))

    scored = score_singapore_round1(forecast, results)
    assert scored["status"] == "scored"
    assert scored["matches_completed"] == scored["matches_expected"] == 8
    assert scored["forecast_sha256"] == sha256(forecast.read_bytes()).hexdigest()
    assert scored["forecast_lf_sha256"] == sha256(
        forecast.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    for name, score in scored["scores"].items():
        actual_winner_prob = [
            row["p_a"][name] if i % 2 else 1 - row["p_a"][name]
            for i, row in enumerate(frozen["matches"])
        ]
        assert score["logloss"] == pytest.approx(-sum(map(log, actual_winner_prob)) / 8)
        assert score["brier"] == pytest.approx(sum((1 - p) ** 2 for p in actual_winner_prob) / 8)


def test_scoring_waits_for_all_round1_winners(tmp_path: Path) -> None:
    forecast = OUT / "bo1_singapore_stage1.json"
    first = json.loads(forecast.read_bytes())["matches"][0]
    results = tmp_path / "data.json"
    one_result = {"round": 1, "a": first["a"], "b": first["b"], "winner": first["a"]}
    results.write_text(json.dumps({"stages": [{"stage": 1, "matches": [one_result]}]}))

    pending = score_singapore_round1(forecast, results)
    assert pending["status"] == "pending"
    assert pending["matches_expected"] == 8
    assert pending["matches_completed"] == 1
    assert "scores" not in pending


@pytest.mark.parametrize("fault", ["duplicate", "conflict", "wrong_winner", "wrong_fixture"])
def test_scoring_rejects_ambiguous_results(tmp_path: Path, fault: str) -> None:
    forecast = OUT / "bo1_singapore_stage1.json"
    first = json.loads(forecast.read_bytes())["matches"][0]
    a, b = first["a"], first["b"]
    match = {"round": 1, "a": a, "b": b, "winner": a}
    broken = {
        "duplicate": [match, {**match, "a": b, "b": a}],
        "conflict": [match, {**match, "winner": b}],
        "wrong_winner": [{**match, "winner": "wrong team"}],
        "wrong_fixture": [{**match, "b": "wrong opponent"}],
    }
    results = tmp_path / "data.json"
    results.write_text(json.dumps({"stages": [{"stage": 1, "matches": broken[fault]}]}))

    with pytest.raises(ValueError, match="duplicate or unexpected fixture|not one of the fixture"):
        score_singapore_round1(forecast, results)


def test_scoring_rejects_post_cutoff_forecast_and_invalid_probability(tmp_path: Path) -> None:
    frozen = json.loads((OUT / "bo1_singapore_stage1.json").read_bytes())
    forecast = tmp_path / "forecast.json"
    results = tmp_path / "unused.json"
    forecast.write_text(json.dumps({**frozen, "frozen_at": frozen["cutoff"]}))

    with pytest.raises(ValueError, match="before the Singapore Stage 1 cutoff"):
        score_singapore_round1(forecast, results)

    frozen["matches"][0]["p_a"]["raw"] = float("nan")
    forecast.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="finite probabilities"):
        score_singapore_round1(forecast, results)
