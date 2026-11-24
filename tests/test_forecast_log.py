import json

from cs2p import run


def block(matches, nxt):
    return {"stage": 2, "seeds_known": True, "outcome_cols": ["A", "B", "C", "D"], "matches": matches,
            "next_round": nxt}


def test_log_keeps_first_forecast_and_attaches_results(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OUT", tmp_path)
    nxt = {"round": 2, "matches": [{"a": "A", "b": "C", "p_a": 0.6, "winner": None}]}
    out = run.update_forecast_log([block([], nxt)])
    assert out == [{"stage": 2, "round": 2, "a": "A", "b": "C", "p_a": 0.6, "made_at": out[0]["made_at"], "winner": None}]
    # later run: ratings moved, but the logged pre-match odds must not change; result arrives
    nxt2 = {"round": 2, "matches": [{"a": "A", "b": "C", "p_a": 0.9, "winner": "C"}]}
    done = [{"round": 2, "a": "A", "b": "C", "winner": "C"}]
    out = run.update_forecast_log([block(done, nxt2)])
    assert len(out) == 1 and out[0]["p_a"] == 0.6 and out[0]["winner"] == "C"
    assert json.loads((tmp_path / "forecast_log.json").read_text())[0]["p_a"] == 0.6
